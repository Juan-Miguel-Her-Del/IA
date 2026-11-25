"""Pruebas pequenas: ningun entrenamiento por epocas ni evaluacion real de test."""
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import PROJECT_ROOT
from utils.dataset import sha256, write_json

try:
    import torch
    from torch import nn
    from training.data import ImageTransform, read_manifest, resolve_path, training_loaders
    from training.engine import run_epoch
    from training.metrics import final_metrics, plot_confusion, plot_history
    from training.model import VisionCNN, parameter_count
    from training.runtime import EarlyStopping, load_checkpoint, load_config, save_checkpoint, seed_everything
    from utils.colab_package import validate_notebook, verify_package
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Instalar requirements-training.txt para las pruebas de CNN.")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_config(overrides={"batch_size": 2, "num_workers": 0})
        cls.root = resolve_path(cls.config["dataset_dir"])
        if not (cls.root / "manifest.csv").exists():
            raise unittest.SkipTest("Se necesita el experimento A preparado.")
        cls.rows, cls.summary = read_manifest(cls.root)

    def batch(self):
        return next(iter(training_loaders(self.root, self.config, self.rows)["train"]))

    def test_architecture_random_initialization_real_batch_and_backward(self):
        seed_everything(42)
        with patch("torch.load", side_effect=AssertionError("No cargar pesos al construir")), patch(
                "urllib.request.urlopen", side_effect=AssertionError("No acceder a red")):
            model = VisionCNN()
        self.assertEqual(parameter_count(model), 89686)
        self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in model.modules()), 3)
        first_weights = model.features[0].weight.detach().clone()
        seed_everything(43)
        self.assertFalse(torch.equal(first_weights, VisionCNN().features[0].weight))
        images, labels = self.batch()
        before = [p.detach().clone() for p in model.parameters()]
        logits = model(images)
        self.assertEqual(tuple(images.shape), (2, 3, 128, 128))
        self.assertEqual(tuple(logits.shape), (2, 6))
        nn.CrossEntropyLoss()(logits, labels).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        # No se instancia optimizador ni se da un paso de actualizacion.
        self.assertTrue(all(torch.equal(old, current) for old, current in zip(before, model.parameters())))

    def test_loaders_use_only_train_val_and_validation_is_deterministic(self):
        loaders = training_loaders(self.root, self.config, self.rows)
        self.assertEqual(set(loaders), {"train", "val"})
        self.assertEqual(len(loaders["train"].dataset), 1953)
        self.assertEqual(len(loaders["val"].dataset), 421)
        dataset = loaders["val"].dataset
        source = self.root / dataset.rows[0]["path"]
        before = sha256(source)
        first, target = dataset[0]
        second, _ = dataset[0]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(-1 <= float(first.min()) <= float(first.max()) <= 1)
        self.assertEqual(before, sha256(source))
        self.assertFalse(dataset.transform.training)
        self.assertTrue(loaders["train"].dataset.transform.training)

    def test_own_temporary_checkpoint_loads_on_cpu_with_same_outputs(self):
        model = VisionCNN().eval()
        images, _ = self.batch()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporary_random_test.pth"
            save_checkpoint(path, model, self.config, 0, 0., self.summary["manifest_sha256"])
            loaded, checkpoint = load_checkpoint(path, "cpu")
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(images), loaded(images)))
            self.assertEqual(next(loaded.parameters()).device.type, "cpu")
            self.assertEqual(checkpoint["classes"], list(loaders_class_order()))
        # El checkpoint fue temporal: no se entrega como modelo entrenado.

    def test_early_stopping_metrics_and_plot_outputs_in_temporary_directory(self):
        stopper = EarlyStopping(patience=2, min_delta=.01)
        self.assertFalse(stopper.step(.3))
        self.assertFalse(stopper.step(.305))
        self.assertTrue(stopper.step(.306))
        report = final_metrics(list(range(6)), list(range(6)))  # fixture de metricas, no inferencia del modelo
        self.assertEqual(report["accuracy"], 1.)
        self.assertEqual(report["macro_f1"], 1.)
        self.assertEqual(len(report["confusion_matrix"]), 6)
        fixture_history = [{"epoch": 1, **{f"{s}_{m}": .5 for s in ("train", "val") for m in ("loss", "accuracy", "macro_f1")}}]
        with tempfile.TemporaryDirectory() as tmp:
            plot_history(fixture_history, Path(tmp) / "curves.png")
            plot_confusion(report["confusion_matrix"], Path(tmp) / "confusion.png")
            self.assertGreater((Path(tmp) / "curves.png").stat().st_size, 1000)
            self.assertGreater((Path(tmp) / "confusion.png").stat().st_size, 1000)

    def test_validation_epoch_helper_on_one_small_train_batch_without_updates(self):
        model = VisionCNN().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = run_epoch(model, [self.batch()], nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertEqual(set(result), {"loss", "accuracy", "macro_f1"})
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_manifest_rejects_visual_group_leakage(self):
        altered = copy.deepcopy(self.rows)
        training_row = next(row for row in altered if row["split"] == "train")
        validation_row = next(row for row in altered if row["split"] == "val")
        validation_row["group_id"] = training_row["group_id"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(altered[0]))
                writer.writeheader()
                writer.writerows(altered)
            summary = {**self.summary, "manifest_sha256": sha256(root / "manifest.csv")}
            write_json(root / "split_summary.json", summary)
            with self.assertRaisesRegex(ValueError, "grupo visual"):
                read_manifest(root)

    def test_notebook_is_unexecuted_parseable_and_has_explicit_drive_consent(self):
        notebook = json.loads((PROJECT_ROOT / "notebooks/train_colab.ipynb").read_text(encoding="utf-8"))
        self.assertGreater(validate_notebook(notebook), 10)
        changed = copy.deepcopy(notebook)
        first_code = next(c for c in changed["cells"] if c["cell_type"] == "code")
        first_code["execution_count"] = 1
        with self.assertRaises(ValueError):
            validate_notebook(changed)

    def test_exported_package_is_complete_without_raw_or_weights(self):
        path = PROJECT_ROOT / "exports/Vision-IA_Colab_A.zip"
        if not path.exists():
            self.skipTest("Ejecutar scripts.export_colab para verificar tambien el paquete.")
        result = verify_package(path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["image_count"], 2791)
        self.assertEqual(result["split_counts"], {"train": 1953, "val": 421, "test": 417})


def loaders_class_order():
    from utils.config import CLASSES
    return CLASSES


if __name__ == "__main__":
    unittest.main()
