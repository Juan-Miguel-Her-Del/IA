"""Leakage guards, real crop geometry and detector annotation eligibility."""
import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from app.detector import crop_box
from training.intake_data import collect_material_rows
from training.intake_detection import group_detection_rows, IntakeDetectionDataset
from training.intake_release import RuntimeCropDataset, classification_metrics, final_test, activate_release, rollback_release
from training.hybrid import build_hybrid
from training.detector_own import VisionLocaliza
from utils.dataset import sha256
from training.compact import build_transform
from tests.test_hybrid import fixture, small_config
from utils.config import CLASSES


class IntakeReleaseTests(unittest.TestCase):
    def test_image_only_plastic_never_becomes_a_detector_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root, extras=True)
            rows, _ = collect_material_rows(root, 'train', small_config(root))
            grouped = group_detection_rows(rows)
            self.assertEqual(len(grouped), 6)
            self.assertEqual(sum(len(r['boxes']) for r in grouped), 6)
            self.assertFalse(any('extra_plastico' in r['path'] for r in grouped))
            for row in grouped:
                self.assertEqual(row['boxes'], [[2, 3, 27, 27]])

    def test_test_reader_requires_explicit_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            with self.assertRaises(ValueError):
                collect_material_rows(root, 'test', small_config(root))
            for material in CLASSES:
                path = root/material/'test'/'_annotations.coco.json'
                doc = json.loads(path.read_text())
                doc['annotations'] = [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [2, 3, 25, 24]}]
                path.write_text(json.dumps(doc))
            # Even the final metadata loader does not decode test pixels.
            with patch('PIL.Image.open', side_effect=AssertionError('Test pixels opened early')):
                rows, _ = collect_material_rows(root, 'test', small_config(root), final_test=True)
            self.assertEqual(len(rows), 6)

    def test_runtime_crop_matches_detector_geometry_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            config = small_config(root)
            rows, _ = collect_material_rows(root, 'val', config)
            row = rows[0]
            pixels = np.arange(40*32*3, dtype=np.uint8).reshape(32, 40, 3)
            Image.fromarray(pixels).save(row['path'])
            stat = Path(row['path']).stat()
            row.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            observed, label = RuntimeCropDataset([row], config)[0]
            with Image.open(row['path']) as source:
                image = source.convert('RGB').crop(crop_box((2, 3, 27, 27), 40, 32))
            expected = build_transform(config, False)(image)
            self.assertTrue(torch.equal(observed, expected))
            self.assertEqual(label, row['label'])

    def test_detection_letterbox_scales_actual_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root)
            config = small_config(root); config['image_size'] = 64
            rows, _ = collect_material_rows(root, 'val', config)
            grouped = group_detection_rows(rows)
            tensor, boxes, ident = IntakeDetectionDataset(grouped, config)[0]
            self.assertEqual(tensor.shape, (3, 64, 64))
            # 40x32 -> 64x51, with six pixels padding at the top.
            np.testing.assert_allclose(boxes.numpy(), [[3.2, 3*51/32+6, 43.2, 27*51/32+6]], atol=1e-5)
            self.assertEqual(ident, 0)

    def test_classification_matrix_keeps_absent_classes_in_macro_f1(self):
        result = classification_metrics([0, 0, 3, 3], [0, 3, 0, 3])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['glass_as_plastic'], 1)
        self.assertEqual(result['plastic_as_glass'], 1)
        self.assertAlmostEqual(result['macro_f1'], 1/6)

    def test_final_test_cannot_run_without_freeze_or_be_used_again(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaises(FileNotFoundError): final_test(run)
            (run/'test_started.pth').write_bytes(b'existing final attempt')
            with self.assertRaises(ValueError): final_test(run)

    def test_activation_preserves_old_weights_and_rollback_restores_config_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'runs/release'; run.mkdir(parents=True)
            (root/'models/runtime').mkdir(parents=True); (root/'config').mkdir()
            old_settings = b'{"checkpoint": "previous.pth"}\n'
            old_manifest = b'{}\n'
            (root/'config/app.json').write_bytes(old_settings)
            (root/'models/runtime/manifest.json').write_bytes(old_manifest)
            historical = root/'models/runtime/previous.pth'; historical.write_bytes(b'KEEP ORIGINAL WEIGHTS')
            cfg = small_config()
            classifier = run/'classifier.pth'
            torch.save({'architecture': 'VisionHybrid-v1', 'classes': list(CLASSES), 'image_size': 32,
                'config': cfg, 'model_state_dict': build_hybrid(cfg).state_dict(), 'manifest_sha256': 'fixture',
                'initialization': 'random_from_scratch', 'foreign_pretrained_weights': False, 'epoch': 1}, classifier)
            detector = run/'detector.pth'
            torch.save({'architecture': 'VisionLocaliza-D2', 'state_dict': VisionLocaliza().state_dict(),
                'image_size': 128, 'epoch': 1, 'provenance': {'pretrained': False, 'initialization': 'random',
                'official_detector_components': False}}, detector)
            hashes = {str(classifier): sha256(classifier), str(detector): sha256(detector)}
            settings = {'checkpoint': str(classifier), 'material_candidate_checkpoint': None,
                'material_candidate_weight': None, 'cpu_threads': 2, 'detector_checkpoint': str(detector),
                'detection_threshold': .27, 'detection_nms_iou': .45, 'detection_views': []}
            torch.save({'settings': settings, 'checkpoint_hashes': hashes}, run/'frozen_release.pth')
            torch.save({'frozen_release_sha256': sha256(run/'frozen_release.pth')}, run/'final_test.pth')
            torch.save({'camera_opened': True, 'stop_cleared_objects': True, 'confirmed_expected': 1,
                        'checkpoint_hashes': hashes, 'settings': settings}, run/'camera_validation.pth')
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()): activate_release(run)
                active = json.loads((root/'config/app.json').read_text(encoding='utf-8'))
                self.assertEqual(sha256(root/active['checkpoint']), hashes[str(classifier)])
                self.assertEqual(historical.read_bytes(), b'KEEP ORIGINAL WEIGHTS')
                with contextlib.redirect_stdout(io.StringIO()): rollback_release(run)
                self.assertEqual((root/'config/app.json').read_bytes(), old_settings)
                self.assertEqual((root/'models/runtime/manifest.json').read_bytes(), old_manifest)
                self.assertTrue((root/active['checkpoint']).exists())
            finally: os.chdir(previous_cwd)


if __name__ == '__main__': unittest.main()
