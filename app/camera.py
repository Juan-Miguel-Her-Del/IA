"""Captura local en un hilo: nunca ejecuta llamadas de Streamlit."""
import atexit
import os
import secrets
import threading
import time
from dataclasses import dataclass

import cv2
from app.localization import Localizer, Localization, iou
from app.vision import bgr_to_rgb
from app.stability import PredictionFilter


def open_capture(index):
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS,30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
    raise RuntimeError("No se pudo abrir la cámara. Revisa el índice, los permisos y si otra aplicación la está usando.")


@dataclass(frozen=True)
class Snapshot:
    status: str
    frame: object
    reading: object
    reading_at: float
    error: str
    analysis_frame: object = None
    localization: object = None
    background_ready: bool = False
    temporal_reading: object = None
    objects: tuple = ()


class CameraStream:
    def __init__(self, service, index=0, fraction=.72, contours=False, interval=.8,
                 capture_factory=None, idle_timeout=15, mode="full", manual_bounds=(.2, .2, .8, .8), detector=None):
        self.service, self.index = service, index
        self.fraction, self.contours, self.interval = fraction, contours, interval
        self.capture_factory, self.idle_timeout = capture_factory or open_capture, idle_timeout
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame_number = 0
        self.preview_token = secrets.token_urlsafe(24)
        self._inference_thread = None
        self._status, self._error = "opening", ""
        self._frame, self._reading, self._reading_at = None, None, 0
        self._analysis_frame, self._localization = None, None
        self._temporal_reading = None
        self._background_request, self._clear_background_request = threading.Event(), threading.Event()
        self.prediction_filter = PredictionFilter()
        self.detector = detector
        self.neural = mode == "neural"
        if self.neural and detector is None:
            raise ValueError("No está disponible el detector neuronal entrenado.")
        self._objects = ()
        from app.detector import ObjectTracker
        self.object_tracker = ObjectTracker()
        self.localizer = Localizer("full" if self.neural else ("auto" if contours else mode), manual_bounds)
        self._heartbeat = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"vision-camera-{index}")

    @property
    def alive(self):
        return self._thread.is_alive()

    @property
    def active(self):
        return self.alive and not self._stop.is_set()

    def start(self):
        self._thread.start()
        return self

    def snapshot(self):
        with self._lock:
            self._heartbeat = time.monotonic()
            frame = self._frame.copy() if self._frame is not None else None
            analysis = self._analysis_frame.copy() if self._analysis_frame is not None else None
            return Snapshot(self._status, frame, self._reading, self._reading_at, self._error, analysis, self._localization,
                            self.localizer.background is not None, self._temporal_reading, self._objects)

    def wait_frame(self,previous,timeout=.5):
        """One latest frame, never a buffered queue, for continuous browser preview."""
        with self._new_frame:
            self._new_frame.wait_for(lambda:self._stop.is_set() or self._frame_number!=previous,timeout)
            if self._stop.is_set() or self._frame is None or self._frame_number==previous:return None,previous
            number=self._frame_number
        return self.snapshot(),number

    def capture_background(self):
        self._background_request.set()

    def clear_background(self):
        self._clear_background_request.set()

    def stop(self):
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        with self._lock:
            if self._status not in ("error", "stopped"):
                self._status = "stopping"
            self._frame, self._reading = None, None
            self._analysis_frame, self._localization = None, None
            self._temporal_reading = None
            self._objects = ()
        if self.alive:
            self._thread.join(timeout=.2)

    def _run_neural_inference(self):
        """Latest frame only: capture/preview never wait for the networks."""
        number, next_reading = 0, 0.
        try:
            while not self._stop.is_set():
                with self._new_frame:
                    self._new_frame.wait_for(lambda: self._stop.is_set() or self._frame_number != number, timeout=.5)
                    if self._stop.is_set():return
                    if self._frame is None or self._frame_number == number:continue
                    rgb, number = self._frame.copy(), self._frame_number
                if time.monotonic() < next_reading:
                    self._stop.wait(min(.02, next_reading-time.monotonic()))
                    continue
                started = time.monotonic()
                objects = self.object_tracker.update(rgb,self.detector.detect(rgb),self.service)
                first = objects[0] if objects else None
                with self._lock:
                    if self._stop.is_set():return
                    self._analysis_frame = rgb
                    self._objects = tuple(objects)
                    self._reading = first.reading if first else None
                    self._temporal_reading = first.reading if first and first.confirmed else None
                    self._reading_at = time.monotonic()
                    self._localization = Localization(first.reading.region if first else None,
                        f"{len(objects)} residuos candidatos." if objects else "Sin residuos detectados.",len(objects))
                next_reading = started + self.interval
        except Exception as exc:
            with self._lock:self._status,self._error = "error",str(exc)
            self._stop.set()

    def _run(self):
        capture = None
        try:
            capture = self.capture_factory(self.index)
            if not capture.isOpened():
                raise RuntimeError("La cámara no está disponible. Revisa permisos, conexión e índice.")
            if self.neural:
                self._inference_thread = threading.Thread(target=self._run_neural_inference,daemon=True,
                                                          name=f"vision-inference-{self.index}")
                self._inference_thread.start()
            failures, next_reading = 0, 0
            while not self._stop.is_set():
                with self._lock:
                    idle = time.monotonic() - self._heartbeat > self.idle_timeout
                if idle:
                    break
                capture_started=time.monotonic()
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError("Se perdió la imagen de la cámara. Detén el video y revisa la conexión.")
                    self._stop.wait(.1)
                    continue
                failures = 0
                rgb = bgr_to_rgb(bgr)
                if self.neural:
                    with self._new_frame:
                        if self._stop.is_set():break
                        self._frame,self._status = rgb,"running"
                        self._frame_number += 1
                        # The detector and every preview reader need the SAME latest
                        # frame. notify() would divide the frame rate among them.
                        self._new_frame.notify_all()
                    self._stop.wait(max(0.,1/30-(time.monotonic()-capture_started)))
                    continue
                if self._clear_background_request.is_set():
                    self.localizer.clear_background()
                    self.prediction_filter.reset()
                    self._clear_background_request.clear()
                if self._background_request.is_set():
                    self.localizer.set_background(rgb)
                    self.prediction_filter.reset()
                    self._background_request.clear()
                if self.localizer.mode == "auto" and self.localizer.background is None:
                    localized = Localization(None, "Sin región fiable: captura primero el fondo sin residuo, con la cámara fija.")
                else:
                    localized = self.localizer.locate(rgb)
                reading = None
                lost = localized.region is None
                with self._lock:
                    last_reading = self._reading
                moved = last_reading is not None and not lost and iou(last_reading.region, localized.region) < .15
                if not lost and (time.monotonic() >= next_reading or last_reading is None or moved):
                    if moved:
                        self.prediction_filter.reset()
                    reading = self.service.classify(rgb, region=localized.region)
                    temporal = self.prediction_filter.update(rgb, reading)
                    next_reading = time.monotonic() + self.interval
                with self._lock:
                    if not self._stop.is_set():
                        self._frame, self._status = rgb, "running"
                        self._localization = localized
                        if lost:
                            self._reading, self._analysis_frame = None, None
                            self._temporal_reading = None
                            self.prediction_filter.reset()
                        if reading is not None:
                            self._reading, self._reading_at = reading, time.monotonic()
                            self._analysis_frame = rgb.copy()
                            self._temporal_reading = temporal
                self._stop.wait(1 / 12)
        except Exception as exc:
            with self._lock:
                self._status, self._error = "error", str(exc)
        finally:
            self._stop.set()
            with self._new_frame:self._new_frame.notify_all()
            if self._inference_thread is not None:self._inference_thread.join(timeout=1.)
            if capture is not None:
                capture.release()
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"
                self._frame, self._reading = None, None
                self._analysis_frame, self._localization = None, None
                self._temporal_reading = None
                self._objects = ()


class CameraPool:
    """Una cámara física por sesión; cierre al salir, detener o perder el cliente."""
    def __init__(self):
        self._streams, self._lock = {}, threading.Lock()
        atexit.register(self.stop_all)

    def get(self, owner):
        with self._lock:
            return self._streams.get(owner)

    def start(self, owner, service, **options):
        with self._lock:
            old = self._streams.get(owner)
            if old is not None:
                old.stop()
                if old.alive:
                    raise RuntimeError("La cámara se está cerrando. Espera unos segundos antes de activarla.")
            index = options.get("index", 0)
            for other_owner, stream in self._streams.items():
                if other_owner != owner and stream.index == index and stream.alive:
                    raise RuntimeError("Esta cámara ya está activa en otra pestaña de Vision-IA.")
            # Retener solo sesiones activas evita acumular imágenes/objetos al cerrar pestañas.
            self._streams = {key: value for key, value in self._streams.items() if value.alive}
            stream = CameraStream(service, **options).start()
            self._streams[owner] = stream
            return stream

    def stop(self, owner):
        stream = self.get(owner)
        if stream is not None:
            stream.stop()

    def stop_all(self):
        with self._lock:
            for stream in self._streams.values():
                stream.stop()
