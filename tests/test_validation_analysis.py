"""Analisis val e inferencia CPU; nunca entrena ni evalua fotografias test."""
import copy
import csv
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import CLASSES, PROJECT_ROOT
from utils.dataset import sha256

try:
    import numpy as np
    import torch
    from PIL import Image
    from training.data import ImageTransform, make_loader, read_manifest
    from training.prediction import predict_image, protect_test_image
    from training.runtime import load_checkpoint, seed_everything
    from training.validation import evaluate_validation, plot_misclassified, validation_metrics
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Instalar requirements-training.txt.")
class ValidationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = PROJECT_ROOT / "data/experiments/A/processed"
        if not (cls.root / "manifest.csv").exists():
            raise unittest.SkipTest("Se necesita el experimento A preparado.")
        cls.rows, _ = read_manifest(cls.root)
        cls.photo = cls.root / next(r["path"] for r in cls.rows if r["split"] == "val")
        cls.checkpoint = PROJECT_ROOT / "models/best_model.pth"

    def require_checkpoint(self):
        if not self.checkpoint.is_file():
            self.skipTest("Copiar el checkpoint entrenado a models/best_model.pth.")

    def test_warn_only_is_enabled_and_seeds_remain_reproducible(self):
        seed_everything(42)
        first = (random.random(), float(np.random.rand()), torch.rand(3))
        seed_everything(42)
        second = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_checkpoint_rejects_reordered_classes_and_missing_normalization(self):
        self.require_checkpoint()
        _, original = load_checkpoint(self.checkpoint)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata_test.pth"
            changed = copy.deepcopy(original)
            changed["classes"][0], changed["classes"][1] = changed["classes"][1], changed["classes"][0]
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "Orden de clases"):
                load_checkpoint(path)
            changed = copy.deepcopy(original)
            changed["config"].pop("normalization")
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "configuracion de entrada"):
                load_checkpoint(path)
            changed = copy.deepcopy(original)
            changed["config"]["normalization"]["std"][0] = 0
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "Normalizacion"):
                load_checkpoint(path)

    def test_prediction_uses_checkpoint_normalization_without_augmentation(self):
        self.require_checkpoint()
        _, original = load_checkpoint(self.checkpoint)
        changed = copy.deepcopy(original)
        changed["config"]["normalization"] = {"mean": [.2, .3, .4], "std": [.7, .8, .9]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporary_normalization_test.pth"
            torch.save(changed, path)
            with patch("training.prediction.ImageTransform", wraps=ImageTransform) as transform:
                result = predict_image(path, self.photo)
            self.assertEqual(transform.call_args.kwargs["normalization"], changed["config"]["normalization"])
            self.assertFalse(transform.call_args.kwargs["training"])
            self.assertEqual(result["classes"], list(CLASSES))
            self.assertTrue(0 <= result["confidence"] <= 1)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1., places=6)

    def test_metric_definitions_include_per_class_accuracy_without_confusion(self):
        # Fixture aritmetico, no resultados de un modelo entrenado.
        result = validation_metrics([0, 0, 1, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4, 5], list(CLASSES))
        plastic = result["per_class"]["plastico"]
        self.assertEqual(plastic["tp"], 1)
        self.assertEqual(plastic["fn"], 1)
        self.assertAlmostEqual(plastic["recall"], .5)
        self.assertAlmostEqual(plastic["accuracy_on_class"], .5)
        self.assertAlmostEqual(plastic["accuracy_one_vs_rest"], 6 / 7)
        self.assertAlmostEqual(result["accuracy"], 6 / 7)

    def test_validation_pipeline_opens_only_val_and_never_constructs_test_loader(self):
        self.require_checkpoint()
        original_open = Image.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                candidate = Path(path).resolve()
                if candidate.is_relative_to(self.root):
                    self.assertTrue(candidate.is_relative_to(self.root / "val"), str(candidate))
                    opened.append(candidate)
            return original_open(path, *args, **kwargs)

        def only_val(root, split, *args, **kwargs):
            self.assertEqual(split, "val")
            return make_loader(root, split, *args, **kwargs)

        before = sha256(self.checkpoint)
        with tempfile.TemporaryDirectory() as tmp, patch("training.validation.make_loader", side_effect=only_val) as factory, patch("PIL.Image.open", side_effect=guarded_open):
            output = Path(tmp) / "analysis"
            result = evaluate_validation(self.checkpoint, self.root, output, maximum_errors=4)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(result["images"], 421)
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(len(set(opened)), 421)
            with (output / "predictions.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 421)
            for filename in ("confusion_matrix.png", "per_class_metrics.csv", "misclassified.csv", "misclassified_examples.jpg", "REPORT.md"):
                self.assertTrue((output / filename).is_file())
        self.assertEqual(before, sha256(self.checkpoint))

    def test_test_images_are_rejected_without_opening_them(self):
        test_relative = next(r["path"] for r in self.rows if r["split"] == "test")
        with patch("PIL.Image.open", side_effect=AssertionError("No abrir test")):
            with self.assertRaisesRegex(ValueError, "PRUEBA"):
                protect_test_image(self.root / test_relative)
        # Simular una copia de test usando metadatos de fixture y una foto val, sin leer fotos test.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "copy.jpg"
            shutil.copyfile(self.photo, photo)
            manifest = root / "data/experiments/A/processed/manifest.csv"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("split,sha256\ntest," + sha256(photo) + "\n", encoding="utf-8")
            with patch("training.prediction.PROJECT_ROOT", root), patch("training.prediction.data_root", return_value=root / "data"):
                with self.assertRaisesRegex(ValueError, "PRUEBA"):
                    protect_test_image(photo)

    def test_visualization_handles_no_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "no_errors.jpg"
            self.assertEqual(plot_misclassified(self.root, [], output), [])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
