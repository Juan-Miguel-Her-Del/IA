"""Ejercita el evaluador con escenas temporales; el test reservado no tiene archivo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from app.detector import Detection
from app.localization import Region
from scripts.evaluate_runtime import evaluate


class RuntimeEvaluationTests(unittest.TestCase):
    def evaluate_fixture(self, folder, object_count, smoke=False):
        root = Path(folder)
        original, scenes = root / "original", root / "scenes"
        original.mkdir()
        (scenes / "images/val").mkdir(parents=True)
        picture = original / "val.jpg"
        Image.new("RGB", (48, 48), (180, 80, 40)).save(picture)
        content = picture.read_bytes()
        (scenes / "images/val/00001.jpg").write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        with (original / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("split", "path", "sha256", "class"))
            writer.writeheader()
            writer.writerow(dict(split="val", path="val.jpg", sha256=digest, **{"class": "plastico"}))
            writer.writerow(dict(split="test", path="MUST_NOT_BE_OPENED.jpg", sha256="test", **{"class": "metal"}))
        boxes = [{"xyxy_normalized": [0, 0, .5, 1], "category": "Plastic bottle", "annotation_id": 10}]
        if object_count == 2:
            boxes.append({"xyxy_normalized": [.5, 0, 1, 1], "category": "Plastic bottle", "annotation_id": 11})
        rows = [{"id": 1, "split": "val", "group": "val-session", "sha256": digest,
                 "pixels_sha256": "val-pixels", "boxes": boxes},
                {"id": 2, "split": "test", "group": "test-session", "sha256": "test",
                 "pixels_sha256": "test-pixels", "boxes": []}]
        (scenes / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
        labels = root / "labels.json"
        labels.write_text(json.dumps([{"split": "val", "image_id": 1, "id": box["annotation_id"],
                                       "class": "plastico"} for box in boxes]), encoding="utf-8")
        fake_checkpoint = root / "checkpoint.pth"
        fake_checkpoint.write_bytes(b"unit-test-only")
        settings = {"checkpoint": str(fake_checkpoint), "detector_checkpoint": str(fake_checkpoint),
                    "cpu_threads": 2, "detection_threshold": .27}
        config = root / "app.json"
        config.write_text(json.dumps(settings), encoding="utf-8")
        detection = Detection((0, 0, 24, 48), .9, Region((0, 0, 48, 48), "fixture"))
        detector = SimpleNamespace(detect=lambda rgb, confidence=None: [detection])
        classifier = SimpleNamespace(classify=lambda *a, **k: SimpleNamespace(material="plastico"),
                                     metadata={"provenance": {"smoke_test": smoke}})
        args = argparse.Namespace(config=config, original=original, scenes=scenes, labels=labels,
                                  output=root / "report.json")
        with patch("scripts.evaluate_runtime.NeuralDetector", return_value=detector), \
             patch("scripts.evaluate_runtime.InferenceService", return_value=classifier), \
             patch("scripts.evaluate_runtime.read_manifest", return_value=([
                 {"split": "val", "path": "val.jpg", "sha256": digest, "class": "plastico"},
                 {"split": "test", "path": "MUST_NOT_BE_OPENED.jpg", "sha256": "test", "class": "metal"}
             ], {})):
            return evaluate(args)

    def test_single_object_success_and_reserved_test_not_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.evaluate_fixture(folder, 1)
        self.assertFalse(report["test_used"])
        self.assertEqual(report["metrics"]["single_object_scenes"], {"images": 1, "correct": 1, "accuracy": 1.})
        self.assertEqual(report["metrics"]["scene"]["images"], 1)

    def test_multiple_objects_are_only_part_of_primary_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.evaluate_fixture(folder, 2)
        self.assertEqual(report["metrics"]["primary_joint_diagnostic"]["recall"], .5)
        self.assertEqual(report["metrics"]["single_object_scenes"], {"images": 0, "correct": 0, "accuracy": None})
        self.assertEqual(report["metrics"]["scene"]["images"], 2)

    def test_smoke_checkpoint_is_not_used_for_candidate_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'prueba corta'):
                self.evaluate_fixture(folder, 1, smoke=True)


if __name__ == "__main__":
    unittest.main()
