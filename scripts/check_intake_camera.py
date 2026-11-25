"""Exercise the actual browser recognition pipeline with a physical local camera."""
import argparse
import base64
import json
from pathlib import Path
import time

import cv2
import numpy as np

from app.browser_camera import BrowserRecognition
from app.camera import open_capture
from app.detector import NeuralDetector
from app.localization import manual_region
from app.vision import InferenceService
from utils.config import CLASSES
from utils.dataset import sha256
from training.intake_training import atomic_save


def diagnose_camera(settings):
    classifier = InferenceService(settings['checkpoint'], settings['cpu_threads'])
    detector = NeuralDetector(settings['detector_checkpoint'], settings['cpu_threads'],
        settings['detection_threshold'], settings['detection_nms_iou'], settings['detection_views'])
    cap = open_capture(settings.get('camera_index', 0))
    try:
        for _ in range(8): ok, bgr = cap.read()
        if not ok: raise RuntimeError('No camera frame.')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        proposals = []
        for detection in detector.detect(rgb, confidence=.20)[:15]:
            reading = classifier.classify(rgb, region=detection.region)
            x1, y1, x2, y2 = detection.box
            proposals.append({'box': list(detection.box), 'detection_confidence': detection.confidence,
                'area_fraction': (x2-x1)*(y2-y1)/(width*height),
                'material': reading.material, 'material_confidence': reading.confidence})
        regions = []
        for bounds in ((0., 0., 1., 1.), (.16, .10, .84, .90), (.3, .2, .7, .8)):
            reading = classifier.classify(rgb, region=manual_region(rgb, bounds))
            regions.append({'bounds': list(bounds), 'material': reading.material, 'confidence': reading.confidence})
        return {'proposals': proposals, 'diagnostic_regions': regions, 'frames_saved': False}
    finally: cap.release()


def check_camera(settings, expected, frames=16):
    classifier = InferenceService(settings['checkpoint'], settings['cpu_threads'],
        settings.get('material_candidate_checkpoint'),
        candidate_weight=settings.get('material_candidate_weight'))
    detector = NeuralDetector(settings['detector_checkpoint'], settings['cpu_threads'],
        settings['detection_threshold'], settings['detection_nms_iou'], settings['detection_views'])
    recognition = BrowserRecognition(settings['low_confidence_threshold'])
    cap = open_capture(settings.get('camera_index', 0))
    timings, rows = [], []
    try:
        for _ in range(8): cap.read()
        for seq in range(1, frames+1):
            ok, bgr = cap.read()
            if not ok: raise RuntimeError('Camera stopped providing frames.')
            ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok: raise RuntimeError('JPEG encode failed.')
            start = time.perf_counter()
            recognition.consume({'event': 'frame', 'stream_id': 'physical-check', 'sequence': seq,
                                 'image': base64.b64encode(encoded).decode()}, detector, classifier)
            timings.append((time.perf_counter()-start)*1000)
            if 'error' in recognition.response: raise RuntimeError(recognition.response['error'])
            item = recognition.objects[0] if recognition.objects else None
            rows.append({'frame': seq, 'material': item.reading.material if item else None,
                'confidence': item.reading.confidence if item else None,
                'confirmed': bool(item and item.confirmed),
                'box': list(item.detection.box) if item else None})
            time.sleep(.15)
        confirmed = [r for r in rows if r['confirmed']]
        result = {'camera_opened': True, 'expected_material_human': expected, 'frames': rows,
            'confirmed_frames': len(confirmed), 'confirmed_expected': sum(r['material'] == expected for r in confirmed),
            'latency_p50_ms': float(np.median(timings)), 'latency_p95_ms': float(np.percentile(timings, 95)),
            'frames_saved': False, 'accuracy_claim': 'one user-labelled scene; no generalization estimate'}
        paths = [settings['checkpoint'], settings['detector_checkpoint']]
        if settings.get('material_candidate_checkpoint'): paths.append(settings['material_candidate_checkpoint'])
        result['checkpoint_hashes'] = {path: sha256(path) for path in paths}
        result['settings'] = settings
        result['camera_code_sha256'] = sha256(Path('app/browser_camera.py'))
        recognition.consume({'event': 'stopped', 'stream_id': 'physical-check', 'sequence': frames+1}, detector, classifier)
        result['stop_cleared_objects'] = not recognition.objects and not recognition.running
        return result
    finally:
        cap.release()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', default='config/app.json')
    p.add_argument('--expected', choices=CLASSES, required=True)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--diagnose', action='store_true', help='Inspect proposal geometry and crop predictions without saving frames.')
    p.add_argument('--preview', action='store_true', help='Return a small in-memory diagnostic frame; no image file.')
    p.add_argument('--output', help='Optional technical metrics checkpoint outside the dataset; no frames.')
    a = p.parse_args()
    if a.preview:
        cap = open_capture(0)
        try:
            for _ in range(8): ok, frame = cap.read()
            if not ok: raise RuntimeError('No camera frame.')
            ok, encoded = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 45])
            if not ok: raise RuntimeError('Preview encode failed.')
            print('CAMERA_PREVIEW:'+base64.b64encode(encoded).decode(), flush=True)
        finally: cap.release()
        raise SystemExit(0)
    settings = json.loads(Path(a.settings).read_text(encoding='utf-8'))
    result = diagnose_camera(settings) if a.diagnose else check_camera(settings, a.expected, a.frames)
    if a.output:
        output = Path(a.output).resolve()
        policy = json.loads(Path('config/material_intake_policy.json').read_text(encoding='utf-8'))
        if output.is_relative_to(Path(policy['dataset_root']).resolve()) or output.exists():
            raise ValueError('Use a new checkpoint outside the dataset.')
        atomic_save(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
