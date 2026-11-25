"""Regresión de procesamiento/localización y registros, sin entrenar ni abrir test."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.camera import CameraStream
from app.field_data import analyze_samples, save_sample
from app.localization import Localizer, LocalizationError, Region, manual_region
from app.resources import get_model
from app.settings import checkpoint_signature, load_settings
from app.vision import bgr_to_rgb, crop_region, decode_image
from scripts.diagnose_app import diagnose
from tests.test_app import FakeCapture, FakeService, wait_until
from utils.config import PROJECT_ROOT


class LocalizationTests(unittest.TestCase):
    def object_frame(self, x=30):
        frame = np.full((240, 480, 3), 245, dtype=np.uint8)
        cv2.rectangle(frame, (x, 60), (x+75, 165), (35, 75, 95), -1)
        return frame

    def test_rgb_conversion_and_full_resolution_decode(self):
        bgr = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
        self.assertEqual(bgr_to_rgb(bgr).tolist(), [[[255, 0, 0], [0, 0, 255]]])
        memory = io.BytesIO()
        Image.new("RGB", (1400, 100)).save(memory, "PNG")
        self.assertEqual(decode_image(memory.getvalue()).shape, (100, 1400, 3))

    def test_auto_finds_object_outside_center_with_margin(self):
        rgb = self.object_frame()
        result = Localizer().locate(rgb)
        self.assertIsNotNone(result.region)
        x1, y1, x2, y2 = result.region.box
        self.assertLess(x1, 30)
        self.assertLess(y1, 60)
        self.assertGreater(x2, 105)
        self.assertGreater(y2, 165)
        self.assertLess(x2, 170)  # Objeto fuera del marco central anterior.

    def test_jitter_smoothing_never_cuts_current_candidate_and_loss_resets(self):
        localizer = Localizer()
        first = localizer.locate(self.object_frame(30)).region
        second = localizer.locate(self.object_frame(42)).region
        self.assertGreater(second.box[0], first.box[0])
        self.assertLess(second.box[0], 42)
        self.assertGreater(second.box[2], 117)
        lost = localizer.locate(np.full((240, 480, 3), 245, np.uint8))
        self.assertIsNone(lost.region)
        self.assertIsNone(localizer.previous)

    def test_ambiguity_and_cut_objects_do_not_fall_back_to_center(self):
        rgb = self.object_frame()
        cv2.rectangle(rgb, (300, 60), (375, 165), (35, 75, 95), -1)
        self.assertIsNone(Localizer().locate(rgb).region)
        self.assertIsNone(Localizer().locate(self.object_frame(0)).region)

    def test_manual_moves_and_checks_bounds(self):
        rgb = self.object_frame()
        roi = manual_region(rgb, (0., .1, .3, .9))
        self.assertEqual(roi.box, (0, 24, 144, 216))
        self.assertEqual(crop_region(rgb, roi).shape, (192, 144, 3))
        for bounds in ((.8, .2, .2, .7), (0, 0, .001, .001)):
            with self.assertRaises(ValueError):
                manual_region(rgb, bounds)
        with self.assertRaises(ValueError):
            crop_region(rgb, Region((-1, 0, 25, 30), "manual"))

    def test_auto_failure_never_invokes_classifier(self):
        service, capture = FakeService(), FakeCapture()
        stream = CameraStream(service, mode="auto", capture_factory=lambda _: capture).start()
        try:
            self.assertTrue(wait_until(lambda: stream.snapshot().localization is not None))
            snapshot = stream.snapshot()
            self.assertIsNone(snapshot.localization.region)
            self.assertIsNone(snapshot.reading)
            self.assertEqual(service.calls, 0)
        finally:
            stream.stop()


class ProcessingAndRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = get_model(*checkpoint_signature(load_settings()["checkpoint"]), 2)

    def test_stratified_validation_regression_exactly_matches_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = diagnose(per_class=1, output=Path(tmp)/"regression")
        self.assertEqual(report["images"], 6)
        self.assertEqual(report["tensor_max_abs_error"], 0.)
        self.assertEqual(report["probability_max_abs_error"], 0.)
        self.assertEqual(report["historical_prediction_mismatches"], 0)
        self.assertTrue(report["passed"])

    def test_record_exact_crop_tensor_duplicate_and_mismatch_protection(self):
        rgb = LocalizationTests().object_frame()
        region = manual_region(rgb, (.02, .1, .4, .9))
        reading = self.service.classify(rgb, region=region)
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_sample(rgb, reading, "plastico", self.service, "Foto con cámara", root=tmp, known_pixels=set())
            with Image.open(folder/"crop.png") as image:
                self.assertTrue(np.array_equal(np.asarray(image), crop_region(rgb, region)))
            saved = json.loads((folder/"record.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["tensor_sha256"], reading.tensor_sha256)
            with self.assertRaisesRegex(ValueError, "ya fue registrada"):
                save_sample(rgb, reading, "plastico", self.service, "Foto con cámara", root=tmp, known_pixels=set())
            different = rgb.copy()
            different[80, 40] = [255, 0, 0]
            with self.assertRaisesRegex(ValueError, "mismo recorte"):
                save_sample(different, reading, "plastico", self.service, "Foto con cámara", root=tmp, known_pixels=set())

    def test_field_report_counts_failed_localization_separately(self):
        rgb = LocalizationTests().object_frame()
        with tempfile.TemporaryDirectory() as tmp:
            save_sample(rgb, None, "metal", self.service, "Video en vivo", root=Path(tmp)/"samples-root", known_pixels=set(), message="Sin candidato")
            report, _ = analyze_samples(Path(tmp)/"samples-root", Path(tmp)/"report")
            self.assertEqual(report["records_available"], 1)
            group = next(iter(report["groups"].values()))
            self.assertEqual(group["localization_failures"], 1)
            self.assertIsNone(group["accuracy_classified_only"])
            self.assertEqual(group["confusion_matrix"]["metal"]["sin-localizacion"], 1)
            with self.assertRaises(FileExistsError):
                analyze_samples(Path(tmp)/"samples-root", Path(tmp)/"report")

    def test_video_diagnostic_frame_matches_reading_even_between_updates(self):
        class ChangingCapture(FakeCapture):
            number = 0
            def read(self):
                self.number += 1
                return True, np.full((100, 120, 3), self.number % 250, np.uint8)
        capture = ChangingCapture()
        stream = CameraStream(self.service, mode="full", interval=1., capture_factory=lambda _: capture).start()
        try:
            self.assertTrue(wait_until(lambda: stream.snapshot().reading is not None))
            self.assertTrue(wait_until(lambda: capture.number >= 3))
            snapshot = stream.snapshot()
            self.assertFalse(np.array_equal(snapshot.frame, snapshot.analysis_frame))
            crop = crop_region(snapshot.analysis_frame, snapshot.reading.region)
            self.assertEqual(hashlib.sha256(crop.tobytes()).hexdigest(), snapshot.reading.crop_sha256)
        finally:
            stream.stop()

    def test_simple_ui_retains_manual_photo_fallback_without_diagnostic_forms(self):
        memory=io.BytesIO()
        Image.fromarray(LocalizationTests().object_frame()).save(memory,'PNG')
        with patch('streamlit.file_uploader',return_value=memory):
            app=AppTest.from_file(str(PROJECT_ROOT/'app/main.py'),default_timeout=30).run()
            app.radio[0].set_value('Subir imagen').run()
            app.slider(key='manual_x').set_range(0,30).run()
            next(b for b in app.button if b.label=='Reconocer recorte').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Blanco' in m.value or 'Verde' in m.value for m in app.markdown))
            self.assertFalse(any('Diagn' in e.label or 'Registrar' in e.label for e in app.expander))
            self.assertEqual(len(app.checkbox),0)


if __name__ == "__main__":
    unittest.main()
