"""Pruebas ligeras: solo inferencia y cámaras simuladas; nunca abre fotografías test."""
import csv
import hashlib
import io
import json
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.camera import CameraPool, CameraStream
from app.recycling import CONDITIONS, LABELS, guidance
from app.resources import get_model, reserved_image_hashes
from app.settings import checkpoint_signature, load_settings, validate_options
from app.vision import InferenceService, Reading, annotate_region, central_region, decode_image, find_region
from utils.config import PROJECT_ROOT
from utils.dataset import sha256


def wait_until(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return False


class FakeCapture:
    def __init__(self, opened=True, readable=True):
        self.opened, self.readable, self.released = opened, readable, False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.readable, np.zeros((100, 120, 3), dtype=np.uint8) if self.readable else None

    def release(self):
        self.released = True


class FakeService:
    def __init__(self):
        self.calls = 0

    def classify(self, rgb, fraction=.72, contours=False, *, region=None):
        self.calls += 1
        return Reading("plastico", .4, dict(zip(LABELS, [.4, .2, .1, .1, .1, .1])), region or central_region(rgb, fraction), 1.)


class ImageAndRulesTests(unittest.TestCase):
    def test_decode_bad_oversize_small_and_reserved_files(self):
        with self.assertRaises(ValueError):
            decode_image(b"not an image")
        with self.assertRaises(ValueError):
            decode_image(b"123", max_bytes=2)
        payload = io.BytesIO()
        Image.new("RGB", (64, 64)).save(payload, "PNG")
        data = payload.getvalue()
        self.assertEqual(decode_image(data).shape, (64, 64, 3))
        with self.assertRaisesRegex(ValueError, "prueba reservado"):
            decode_image(data, reserved_hashes={hashlib.sha256(data).hexdigest()})
        with self.assertRaisesRegex(ValueError, "demasiado grande"):
            decode_image(data, max_pixels=100)

    def test_contour_proposal_and_explicit_failure(self):
        rgb = np.full((300, 400, 3), 245, np.uint8)
        self.assertIsNone(find_region(rgb, use_contours=True))
        cv2.rectangle(rgb, (165, 115), (235, 185), (30, 30, 30), -1)
        roi = find_region(rgb, use_contours=True)
        self.assertTrue(roi.method.startswith("auto-"))
        x1, y1, x2, y2 = roi.box
        self.assertTrue(x1 < 165 and x2 > 235 and y1 < 115 and y2 > 185)
        original = rgb.copy()
        result = annotate_region(rgb, roi)
        self.assertTrue(np.array_equal(rgb, original))
        self.assertFalse(np.array_equal(rgb, result))

    def test_all_materials_conditions_and_operational_groups(self):
        for material in LABELS:
            for condition in CONDITIONS:
                result = guidance(material, condition)
                self.assertTrue(result["tips"] and result["advice"])
            self.assertEqual(guidance(material)["group"].startswith("Orgánico"), material == "organicos")
        self.assertIn("Negro", guidance("papel", "Con restos o humedad")["destination"])
        self.assertIn("Verde", guidance("organicos")["destination"])
        self.assertIn("Blanco", guidance("vidrio")["destination"])
        self.assertEqual(guidance("metal", "Residuo especial o peligroso")["destination"], "Entrega diferenciada")
        with self.assertRaises(ValueError):
            guidance("trash")

    def test_settings_paths_and_option_validation(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_signature("models/does-not-exist.pth")
        for fraction, interval in ((.1, .8), (.72, float("nan")), (.72, .05)):
            with self.assertRaises(ValueError):
                validate_options(fraction, interval)


class CameraTests(unittest.TestCase):
    def test_background_periodic_readings_and_explicit_release(self):
        capture, service = FakeCapture(), FakeService()
        stream = CameraStream(service, interval=.3, capture_factory=lambda _: capture)
        started = time.monotonic()
        stream.start()
        self.assertLess(time.monotonic() - started, .3)
        self.assertTrue(wait_until(lambda: service.calls >= 2))
        snapshot = stream.snapshot()
        self.assertEqual(snapshot.status, "running")
        self.assertEqual(snapshot.reading.material, "plastico")
        self.assertLessEqual(service.calls, 4)
        stream.stop()
        self.assertTrue(wait_until(lambda: capture.released))
        self.assertIsNone(stream.snapshot().frame)
        self.assertIsNone(stream.snapshot().reading)

    def test_open_error_and_read_error_release(self):
        for capture in (FakeCapture(opened=False), FakeCapture(readable=False)):
            stream = CameraStream(FakeService(), capture_factory=lambda _: capture).start()
            self.assertTrue(wait_until(lambda: not stream.alive))
            self.assertEqual(stream.snapshot().status, "error")
            self.assertTrue(stream.snapshot().error)
            self.assertTrue(capture.released)

    def test_idle_client_releases_camera(self):
        capture = FakeCapture()
        stream = CameraStream(FakeService(), capture_factory=lambda _: capture, idle_timeout=.12).start()
        self.assertTrue(wait_until(lambda: not stream.alive))
        self.assertTrue(capture.released)
        self.assertEqual(stream.snapshot().status, "stopped")

    def test_camera_exclusive_across_tabs_and_stop_all(self):
        pool, capture = CameraPool(), FakeCapture()
        pool.start("first", FakeService(), capture_factory=lambda _: capture)
        try:
            with self.assertRaisesRegex(RuntimeError, "otra pestaña"):
                pool.start("second", FakeService(), capture_factory=lambda _: FakeCapture())
        finally:
            pool.stop_all()
        self.assertTrue(wait_until(lambda: capture.released))


class RealModelAndUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings()
        cls.signature = checkpoint_signature(cls.settings["checkpoint"])
        cls.path = cls.signature[0]
        cls.before = sha256(cls.path)
        cls.service = get_model(*cls.signature, 2)
        root = PROJECT_ROOT / "data/experiments/B/processed"
        with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] == "val"]
        cls.photos = [root / row["path"] for row in rows[:2]]
        cls.payload = cls.photos[0].read_bytes()

    def test_real_cpu_metadata_transform_and_cached_model(self):
        self.assertIs(get_model(*self.signature, 2), self.service)
        self.assertEqual(tuple(LABELS), self.service.classes)
        self.assertEqual(next(self.service.model.parameters()).device.type, "cpu")
        self.assertFalse(self.service.model.training)
        expected_parameters={"VisionCNN-v1":89686,"VisionResidual-v2":2800678 if self.service.metadata["config"].get("pooling")=="avgmax" else 1240222}
        self.assertEqual(self.service.parameter_count,expected_parameters[self.service.metadata["architecture"]])
        self.assertTrue(torch.equal(self.service.transform.mean[:, 0, 0], torch.tensor(self.service.metadata["config"]["normalization"]["mean"])))
        for path in self.photos:
            rgb = decode_image(path.read_bytes(), reserved_hashes=reserved_image_hashes())
            reading = self.service.classify(rgb)
            self.assertIn(reading.material, self.service.classes)
            self.assertAlmostEqual(sum(reading.probabilities.values()), 1., places=5)
            self.assertGreaterEqual(reading.confidence, 0.)
            self.assertLessEqual(reading.confidence, 1.)
        self.assertEqual(sha256(self.path), self.before)

    def test_default_ui_is_camera_without_technical_controls(self):
        app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.radio[0].value, "C\u00e1mara en vivo")
        self.assertEqual(len(app.slider),0)
        self.assertEqual(len(app.text_input),0)
        self.assertEqual(len(app.selectbox),0)
        self.assertEqual(len(app.button),0)
        self.assertEqual(len(app.get("component_instance")),1)
        app.radio[0].set_value('Tomar foto').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('camera_input')),1)
        app.radio[0].set_value('Subir imagen').run()
        self.assertEqual(len(app.get('file_uploader')),1)

    def test_ui_missing_checkpoint_displays_error(self):
        from app.settings import load_settings
        settings=load_settings();settings['checkpoint']='missing-checkpoint.pth'
        with patch('app.settings.load_settings',return_value=settings):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No se pudo cargar' in entry.value for entry in app.error))

    def test_browser_video_frame_and_stop_without_server_camera(self):
        import base64
        from app.detector import Detection
        from app.localization import Region
        class Detector:
            def detect(self,rgb,confidence=None):return [Detection((10,10,50,60),.9,Region((5,5,65,65),'neuronal'))]
        frame={'event':'frame','stream_id':'browser-test','sequence':1,'image':base64.b64encode(self.payload).decode()}
        # Browser protocol limits live frames to 640 pixels; use a small real JPEG.
        buf=io.BytesIO();Image.new('RGB',(200,200)).save(buf,'JPEG')
        frame['image']=base64.b64encode(buf.getvalue()).decode()
        with patch('app.camera.open_capture',side_effect=AssertionError('Server camera must not open')),patch('app.browser_camera.camera_component',return_value=frame) as component,patch('app.resources.get_detector',return_value=Detector()):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
            self.assertFalse(app.exception)
            recognition=app.session_state['browser_recognition'][1]
            self.assertTrue(recognition.running)
            self.assertEqual(len(recognition.objects),1)
            app.run()
            self.assertEqual(component.call_args.kwargs['response']['sequence'],1)
            component.return_value={'event':'stopped','stream_id':'browser-test','sequence':2}
            app.run()
            self.assertFalse(recognition.running);self.assertEqual(recognition.objects,())
            app.radio[0].set_value('Subir imagen').run()
            self.assertFalse(app.exception)
            self.assertNotIn('browser_recognition',app.session_state)

    def test_invalid_upload_is_handled(self):
        with patch('streamlit.file_uploader',return_value=io.BytesIO(b'invalid')):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
            app.radio[0].set_value('Subir imagen').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No se pudo reconocer' in entry.value for entry in app.error))

    def test_uploaded_photo_shows_material_destination_and_caches_inference(self):
        from app.detector import Detection,crop_box
        from app.localization import Region
        class Detector:
            def detect(self,rgb):
                h,w=rgb.shape[:2]
                return [Detection((0.,0.,float(w),float(h)),.9,Region((0,0,w,h),'neuronal'))]
        mock_service=FakeService()
        with patch('streamlit.file_uploader',return_value=io.BytesIO(self.payload)), patch('app.resources.get_detector',return_value=Detector()),patch('app.resources.get_model',return_value=mock_service):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
            app.radio[0].set_value('Subir imagen').run()
            self.assertFalse(app.exception)
            self.assertEqual(mock_service.calls,1)
            self.assertTrue(any('Blanco' in m.value for m in app.markdown))
            self.assertTrue(any('Posible' in m.value for m in app.markdown))
            app.run()
            self.assertEqual(mock_service.calls,1)
            self.assertFalse(any('Checkpoint' in e.label for e in app.expander))

    def test_previous_models_and_reports_are_unchanged(self):
        expected = json.loads((PROJECT_ROOT / "data/automation/phase3_preserved.json").read_text(encoding="utf-8"))
        for relative, digest in expected.items():
            self.assertEqual(sha256(PROJECT_ROOT / relative), digest, relative)


if __name__ == "__main__":
    unittest.main()
