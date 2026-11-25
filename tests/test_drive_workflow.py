"""Pruebas locales del protocolo Drive; ningun entrenamiento ni autenticacion."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import CLASSES, PROJECT_ROOT
from utils.dataset import sha256, write_json
from utils.drive_sync import (check_preserved_A, configure_folder, import_finished_run,
                              load_sync_settings, publish_package, sync_results)
from utils.run_files import finish_run, safe_child, validate_finished_run


def fixture_run(shared, status="completed"):
    """Artefactos de fixture solo en temporales: no son resultados del experimento B."""
    folder = Path(shared) / "results/A/A_fixture_protocol"
    folder.mkdir(parents=True)
    write_json(folder / "config.json", {"experiment": "A"})
    write_json(folder / "class_to_idx.json", {label: i for i, label in enumerate(CLASSES)})
    write_json(folder / "run_metadata.json", {"architecture": "VisionCNN-v1", "parameters": 89686,
               "initialization": "random_from_scratch", "test_used_during_training": False})
    write_json(folder / "history.json", [{"epoch": 1, "val_accuracy": .5, "val_macro_f1": .4}])
    write_json(folder / "training_summary.json", {"epochs_completed": 1, "best_epoch": 1,
               "best_val_macro_f1": .4, "stopped_early": False, "test_used_during_training": False})
    (folder / "history.csv").write_text("epoch,val_accuracy,val_macro_f1\n1,.5,.4\n", encoding="utf-8")
    (folder / "best_model.pth").write_bytes(b"fixture-checksum-only-not-a-trained-model")
    (folder / "training_curves.png").write_bytes(b"fixture-image-checksum-only")
    finish_run(folder, "A", folder.name, "fixture-package", status)
    return folder


class DriveProtocolTests(unittest.TestCase):
    def test_folder_must_exist_and_local_configuration_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                configure_folder(root / "absent", local_path=root / "local.json")
            settings = configure_folder(root, "Vision-IA", "B", root / "local.json")
            self.assertEqual(settings["local_shared_dir"], str(root.resolve()))
            loaded = load_sync_settings(settings_path=PROJECT_ROOT / "config/sync.json", local_path=root / "local.json")
            self.assertEqual(loaded["selected_experiment"], "B")
            with self.assertRaises(ValueError):
                configure_folder(root, "../outside", local_path=root / "bad.json")
            base = json.loads((PROJECT_ROOT / "config/sync.json").read_text())
            base["selected_experiment"] = "A"
            write_json(root / "settings.json", base)
            self.assertEqual(load_sync_settings(settings_path=root / "settings.json", local_path=root / "local.json")["selected_experiment"], "A")

    def test_partial_and_modified_results_are_not_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            output = fixture_run(shared)
            (output / "history.csv").unlink()
            result = sync_results({"local_shared_dir": str(shared)}, root / "local")
            self.assertEqual(result["runs"], [])
            self.assertEqual(len(result["pending"]), 1)
            self.assertFalse((root / "local/A/A_fixture_protocol").exists())

    def test_complete_results_import_once_without_overwriting_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            source = fixture_run(shared)
            destination = root / "local"
            protected = root / "best_model_A.pth"
            protected.write_bytes(b"protected-A")
            first = sync_results({"local_shared_dir": str(shared)}, destination)
            self.assertEqual(len(first["runs"]), 1)
            self.assertEqual(first["runs"][0]["best_val_accuracy"], .5)
            self.assertEqual(protected.read_bytes(), b"protected-A")
            second = import_finished_run(source, destination)
            self.assertEqual(second, destination / "A/A_fixture_protocol")
            self.assertEqual(len(list((destination / "A").iterdir())), 1)
            self.assertTrue((destination / "summary.csv").is_file())
            (source / "best_model.pth").write_bytes(b"altered")
            with self.assertRaises(ValueError):
                import_finished_run(source, destination)

    def test_missing_marker_and_failed_run_never_become_successful_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixture_run(root / "shared", status="failed")
            result = sync_results({"local_shared_dir": str(root / "shared")}, root / "local")
            self.assertEqual(result["runs"][0]["status"], "failed")
            self.assertEqual(result["runs"][0]["checkpoint"], "")
            (source / "FINISHED.json").unlink()
            with self.assertRaises(OSError):
                validate_finished_run(source)
            incomplete = root / "incomplete"
            with self.assertRaises(ValueError):
                finish_run(incomplete, "B", "B_missing", "fixture", "completed")
            self.assertFalse((incomplete / "FINISHED.json").exists())

    def test_safe_paths_and_source_experiment_association(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../outside", "/absolute", "C:\\outside"):
                with self.assertRaises(ValueError):
                    safe_child(root, name)
            source = fixture_run(root / "shared")
            marker = json.loads((source / "FINISHED.json").read_text())
            marker["experiment"] = "B"
            write_json(source / "FINISHED.json", marker)
            with self.assertRaises(ValueError):
                import_finished_run(source, root / "local")

    def test_publication_uses_versions_and_preserves_A_control_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            shared.mkdir()
            package = root / "Vision-IA_Colab_B.zip"
            package.write_bytes(b"fixture-package-not-real-data")
            write_json(shared / "control.json", {"schema_version": 1, "packages": {"A": {"package_id": "preserved"}}})
            settings = {"local_shared_dir": str(shared), "drive_relative_root": "Vision-IA"}
            # El contenido ZIP real se verifica en la prueba de paquete; aqui se prueba el protocolo de archivos.
            with patch("utils.colab_package.verify_package", return_value={"experiment": "B", "workflow": "drive", "image_count": 1}):
                first = publish_package(package, "B", settings)
                second = publish_package(package, "B", settings)
            self.assertEqual(first["package_id"], second["package_id"])
            self.assertEqual(sha256(Path(first["local_target"])), first["package_sha256"])
            control = json.loads((shared / "control.json").read_text())
            self.assertEqual(control["selected_experiment"], "B")
            self.assertEqual(control["packages"]["A"]["package_id"], "preserved")
            self.assertTrue((shared / "notebooks/train_drive.ipynb").exists())

    def test_launcher_logs_and_finalizes_success_or_failure_without_training(self):
        from scripts.run_colab import execute_run
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results/A/A_fixture_protocol"
            plan = {"experiment": "A", "run_id": output.name, "output": str(output), "package_id": "fixture",
                    "device": "cpu", "config_path": "config/train_A.json",
                    "config": {"epochs": 1, "batch_size": 2, "learning_rate": .001, "num_workers": 0}}

            def fake_process(*args, **kwargs):
                fixture_run(root)
                process = Mock()
                process.stdout = io.StringIO("fixture: subprocess simulated, no training\n")
                process.wait.return_value = 0
                process.poll.return_value = 0
                return process

            with patch("scripts.run_colab.subprocess.Popen", side_effect=fake_process) as process:
                self.assertEqual(execute_run(plan), output)
                self.assertNotIn("--checkpoint", process.call_args.args[0])
            self.assertEqual(validate_finished_run(output)[0]["status"], "completed")
            self.assertIn("simulated", (output / "training.log").read_text())
            failure = {**plan, "run_id": "A_failed_fixture", "output": str(root / "results/A/A_failed_fixture")}
            with patch("scripts.run_colab.subprocess.Popen", side_effect=OSError("fixture failure")):
                with self.assertRaises(OSError):
                    execute_run(failure)
            self.assertEqual(validate_finished_run(Path(failure["output"]))[0]["status"], "failed")
            self.assertFalse((Path(failure["output"]) / "best_model.pth").exists())


class PreparedBTests(unittest.TestCase):
    def test_notebook_and_prepared_B_package_are_consistent(self):
        from utils.colab_package import validate_drive_notebook, verify_package
        notebook = json.loads((PROJECT_ROOT / "notebooks/train_drive.ipynb").read_text(encoding="utf-8"))
        self.assertGreater(validate_drive_notebook(notebook), 5)
        package = PROJECT_ROOT / "exports/sync/B/Vision-IA_Colab_B.zip"
        if not package.exists():
            self.skipTest("Preparar primero el paquete B con scripts.prepare_training.")
        verified = verify_package(package)
        self.assertEqual(verified["experiment"], "B")
        self.assertEqual(verified["image_count"], 5875)
        self.assertEqual(verified["split_counts"], {"train": 4112, "val": 884, "test": 879})

    def test_B_plan_is_fresh_same_architecture_and_does_not_execute_training(self):
        import torch
        from training.model import VisionCNN, parameter_count
        from training.data import make_loader, resolve_path
        from scripts.run_colab import plan_run
        torch.set_num_threads(2)
        if not (PROJECT_ROOT / "data/experiments/B/processed/manifest.csv").exists():
            self.skipTest("Preparar primero el experimento B.")
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.run_colab.subprocess.Popen", side_effect=AssertionError("No entrenar")):
            first = plan_run("B", tmp, "fixture-package", {"batch_size": 2, "num_workers": 0})
            second = plan_run("B", tmp, "fixture-package")
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertFalse(Path(first["output"]).exists())
            self.assertEqual(first["initialization"], "random_from_scratch")
            self.assertEqual(first["train"], 4112)
            self.assertEqual(first["val"], 884)
            with patch("torch.load", side_effect=AssertionError("No cargar el modelo A")):
                model = VisionCNN().eval()
            self.assertEqual(parameter_count(model), 89686)
            config = first["config"]
            # Solo un forward de dos fotos B, sin backward, optimizador o metricas de rendimiento.
            images, _ = next(iter(make_loader(resolve_path(config["dataset_dir"]), "train", config)))
            with torch.inference_mode():
                self.assertEqual(tuple(model(images).shape), (2, 6))
        self.assertGreater(check_preserved_A(), 0)


if __name__ == "__main__":
    unittest.main()
