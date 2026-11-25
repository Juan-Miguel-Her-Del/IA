"""Pruebas de corrupcion, fuga de datos, importaciones y reproducibilidad."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from scripts.prepare_trashnet import import_archive, import_directory
from scripts.split_dataset import allocate, split_records
from utils.config import CLASSES, PROJECT_ROOT, TRASHNET_CLASSES, data_root
from utils.dataset import inventory, require_no_conflicts


def make_image(path, number):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (number % 256, number // 256, 17)).save(path)
    return path


class DatasetTests(unittest.TestCase):
    def test_audit_detects_corruption_duplicates_and_label_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_image(root / "raw/trashnet/plastico/a.png", 1)
            (first.parent / "duplicate.png").write_bytes(first.read_bytes())
            (first.parent / "broken.jpg").write_bytes(b"invalid image")
            report = inventory(root)
            self.assertEqual(report["files_found"], 3)
            self.assertEqual(len(report["corrupt"]), 1)
            self.assertEqual(len(report["duplicates"]), 1)
            self.assertEqual(report["unique_counts"]["plastico"], 1)
            conflict = root / "raw/organicos/source/conflict.png"
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(first.read_bytes())
            report = inventory(root)
            self.assertEqual(len(report["label_conflicts"]), 1)
            with self.assertRaises(ValueError):
                require_no_conflicts(report)

    def test_allocation_and_deterministic_split(self):
        for total in range(5, 3000):
            counts = allocate(total)
            self.assertEqual(sum(counts), total)
            self.assertTrue(all(count > 0 for count in counts))
            for count, ratio in zip(counts, (0.70, 0.15, 0.15)):
                self.assertLess(abs(count - total * ratio), 1.000000001)
        rows = [{"class": "plastico", "sha256": str(i), "path": f"{i}.png", "width": 1, "height": 1} for i in range(20)]
        first = split_records(rows, 42)
        self.assertEqual(first, split_records(rows, 42))
        self.assertNotEqual(first, split_records(rows, 43))
        self.assertEqual(len({r["sha256"] for r in first}), 20)
        self.assertEqual([sum(r["split"] == s for r in first) for s in ("train", "val", "test")], [14, 3, 3])
        with self.assertRaises(ValueError):
            split_records(rows[:4], 42)

    def test_zip_import_excludes_trash_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for i, label in enumerate([*TRASHNET_CLASSES, "trash"]):
                    path = make_image(base / f"{label}.png", i)
                    bundle.write(path, f"dataset-resized/{label}/{label}.png")
            root = base / "data"
            counts, excluded = import_archive(archive, root)
            self.assertEqual(sum(counts.values()), 5)
            self.assertEqual(excluded, 1)
            import_archive(archive, root)
            self.assertEqual(len(list((root / "raw").rglob("*.png"))), 5)
            self.assertFalse((root / "raw/organicos").exists())
            with zipfile.ZipFile(base / "unsafe.zip", "w") as bundle:
                bundle.writestr("../outside.png", b"invalid")
            with self.assertRaises(ValueError):
                import_archive(base / "unsafe.zip", root)

    def test_pipeline_and_manifest_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, root, organics = base / "trashnet", base / "data", base / "organics"
            for class_index, label in enumerate(TRASHNET_CLASSES):
                for number in range(20):
                    make_image(source / label / f"{number}.png", class_index * 20 + number)
            make_image(source / "trash/excluded.png", 230)
            counts, excluded = import_directory(source, root)
            self.assertEqual(sum(counts.values()), 100)
            self.assertEqual(excluded, 1)
            for number in range(20):
                make_image(organics / f"{number}.png", 100 + number)

            def run(module, *extra, expected=0):
                result = subprocess.run(
                    [sys.executable, "-m", f"scripts.{module}", "--data-dir", str(root), *extra],
                    cwd=PROJECT_ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result

            run("import_organics", "--input-dir", str(organics), "--source-name", "synthetic-test",
                "--source-url", "synthetic:test-only", "--license", "generated-test-fixtures")
            self.assertTrue((root / "reports/imports.json").exists())
            duplicate = next((root / "raw/trashnet/plastico").glob("*.png"))
            (duplicate.parent / "duplicate.png").write_bytes(duplicate.read_bytes())
            (duplicate.parent / "broken.jpg").write_bytes(b"broken")
            run("audit_dataset", expected=1)
            (root / "processed").mkdir()
            (root / "processed/.gitkeep").touch()
            run("split_dataset", "--require-organics")
            original_manifest = (root / "processed/manifest.csv").read_bytes()
            run("dataset_stats")
            rows = list(csv.DictReader(original_manifest.decode().splitlines()))
            self.assertEqual(len(rows), 120)
            self.assertEqual(len({r["sha256"] for r in rows}), 120)
            summary = json.loads((root / "processed/split_summary.json").read_text())
            self.assertEqual(summary["total"], 120)
            import hashlib
            self.assertEqual(summary["manifest_sha256"], hashlib.sha256(original_manifest).hexdigest())
            self.assertEqual(summary["exact_duplicates_excluded"], 1)
            self.assertEqual(summary["corrupt_excluded"], 1)
            for label in CLASSES:
                self.assertEqual([summary["counts"][s][label] for s in ("train", "val", "test")], [14, 3, 3])
            run("split_dataset", expected=1)
            self.assertEqual(original_manifest, (root / "processed/manifest.csv").read_bytes())
            run("split_dataset", "--rebuild")
            self.assertEqual(original_manifest, (root / "processed/manifest.csv").read_bytes())
            make_image(root / "raw/organicos/synthetic-test/new.png", 240)
            run("dataset_stats", expected=1)
            run("split_dataset", "--rebuild")
            run("dataset_stats")
            cross_class = root / "raw/organicos/synthetic-test/cross-class.png"
            cross_class.write_bytes(duplicate.read_bytes())
            run("split_dataset", "--rebuild", expected=1)
            run("split_dataset", "--rebuild", "--exclude-label-conflicts")
            run("dataset_stats")
            policy = json.loads((root / "processed/split_summary.json").read_text())
            self.assertEqual(policy["label_conflict_files_excluded"], 3)
            self.assertEqual(len(policy["label_conflict_hashes_excluded"]), 1)
            copied = next((root / "processed/train/plastico").glob("*.png"))
            copied.write_bytes(b"modified")
            run("dataset_stats", expected=1)

    def test_routes_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            original_env = os.environ.pop("VISION_IA_DATA_DIR", None)
            try:
                os.chdir(tmp)
                self.assertEqual(data_root(), PROJECT_ROOT / "data")
                os.environ["VISION_IA_DATA_DIR"] = tmp
                self.assertEqual(data_root(), Path(tmp).resolve())
                self.assertEqual(data_root(PROJECT_ROOT / "data"), PROJECT_ROOT / "data")
            finally:
                os.chdir(previous)
                if original_env is None:
                    os.environ.pop("VISION_IA_DATA_DIR", None)
                else:
                    os.environ["VISION_IA_DATA_DIR"] = original_env


if __name__ == "__main__":
    unittest.main()
