"""Regresiones de fase 1.5 usando copias temporales de fotografias reales."""
import copy
import io
import json
import re
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from utils.config import CLASSES, PROJECT_ROOT
from utils.dataset import sha256, write_json
from utils.experiment_pipeline import build_experiment, split_groups, verify_experiment
from utils.experiments import load_experiment, source_records, verify_phase1
from utils.realwaste import RangeUnavailable, RemoteZip, import_zip
from utils.visual_audit import analyze

DATA = PROJECT_ROOT / "data"


class PartialResponse(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status, self.headers = status, headers


class ExperimentTests(unittest.TestCase):
    def test_experiment_selection_keeps_A_and_B_separate(self):
        name, _, selected = load_experiment()
        self.assertEqual(name, "A")
        self.assertEqual(selected["realwaste_categories"], ["Food Organics"])
        _, _, selected_b = load_experiment("B")
        self.assertEqual(len(selected_b["realwaste_categories"]), 6)
        self.assertNotIn("Vegetation", selected_b["realwaste_categories"])
        self.assertNotIn("Miscellaneous Trash", selected_b["realwaste_categories"])

    def test_remote_zip_only_reads_ranges_and_refuses_full_response(self):
        archive = io.BytesIO()
        payload = bytes(range(256)) * 4000
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("large.bin", payload)
            bundle.writestr("small.txt", b"selected category metadata")
        content = archive.getvalue()

        def respond(url, headers):
            value = headers["Range"]
            if value.startswith("bytes=-"):
                start, end = max(0, len(content) - int(value[7:])), len(content) - 1
            else:
                start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", value).groups())
            return PartialResponse(content[start:end + 1], 206, {"Content-Range": f"bytes {start}-{end}/{len(content)}"})

        with patch("utils.realwaste.request", side_effect=respond):
            with RemoteZip("https://example.invalid/test.zip") as remote:
                with zipfile.ZipFile(remote) as bundle:
                    self.assertEqual(bundle.read("small.txt"), b"selected category metadata")
                self.assertLess(remote.transferred, len(content) // 4)
        with patch("utils.realwaste.request", return_value=PartialResponse(b"not read", 200, {})):
            with self.assertRaises(RangeUnavailable):
                RemoteZip("https://example.invalid/full.zip")

    def test_group_split_is_reproducible_and_never_separates_variants(self):
        rows = [{"class": label, "group_id": f"{label}-{group}", "path": f"{label}/{group}/{i}.jpg", "sha256": f"{label}-{group}-{i}"}
                for label in CLASSES for group in range(10) for i in range(3 if group == 0 else 1)]
        first = split_groups(copy.deepcopy(rows), 42, {"train": .7, "val": .15, "test": .15})
        self.assertEqual(first, split_groups(copy.deepcopy(rows), 42, {"train": .7, "val": .15, "test": .15}))
        for group in {r["group_id"] for r in first}:
            self.assertEqual(len({r["split"] for r in first if r["group_id"] == group}), 1)
        for label in CLASSES:
            self.assertEqual({r["split"] for r in first if r["class"] == label}, {"train", "val", "test"})

    def test_visual_audit_finds_real_photo_variants_and_corruption(self):
        source = next((DATA / "raw/trashnet/plastico").glob("*.jpg"), None)
        if source is None:
            self.skipTest("Importe TrashNet para las pruebas con fotografias reales.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(source, root / "original.jpg")
            shutil.copyfile(source, root / "copy.jpg")
            with Image.open(source) as image:
                image.convert("RGB").save(root / "recoded.png")
                image.resize((256, 192), Image.Resampling.LANCZOS).transpose(Image.Transpose.ROTATE_90).save(root / "rotated.png")
            (root / "broken.jpg").write_bytes(b"invalid image")
            rows = [{"path": name, "class": "plastico", "dataset": "TrashNet"}
                    for name in ("original.jpg", "copy.jpg", "recoded.png", "rotated.png", "broken.jpg")]
            report = analyze(root, rows, {"phash_distance": 6, "dhash_distance": 10, "orientations": 8})
            self.assertEqual(len(report["corrupt"]), 1)
            self.assertEqual(len(report["exact_duplicate_groups"]), 1)
            self.assertEqual(len({r["group_id"] for r in report["usable"]}), 1)
            self.assertEqual(len(report["usable"]), 2)  # original/recodificacion exacta se deduplican; variante se agrupa
            rows[1]["class"] = "metal"
            report = analyze(root, rows, {"phash_distance": 6, "dhash_distance": 10, "orientations": 8})
            self.assertEqual(len(report["usable"]), 0)
            self.assertTrue(any(g["conflict"] for g in report["groups"]))

    def test_uci_zip_import_is_selective_with_per_image_provenance(self):
        source = next((DATA / "raw/realwaste/organicos").glob("*.jpg"), None)
        if source is None:
            self.skipTest("Importe RealWaste A para esta prueba.")
        _, config, _ = load_experiment("A")
        config["realwaste"]["categories"]["Food Organics"]["expected_images"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("realwaste-main/RealWaste/Food Organics/photo.jpg", source.read_bytes())
                bundle.writestr("realwaste-main/RealWaste/Vegetation/ignored.jpg", source.read_bytes())
            with zipfile.ZipFile(archive) as bundle:
                rows = import_zip(bundle, Path(tmp), ["Food Organics"], config, "test_local_uci_structure")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sha256"], sha256(source))
            self.assertEqual(rows[0]["original_category"], "Food Organics")
            self.assertEqual(len(list(Path(tmp).rglob("*.jpg"))), 1)

    def test_end_to_end_source_isolation_preservation_and_tamper_detection(self):
        registry_path = DATA / "raw/realwaste/provenance.json"
        if not registry_path.exists():
            self.skipTest("Importe RealWaste A para esta prueba.")
        name, config, selected = load_experiment("A")
        config["realwaste"]["categories"]["Food Organics"]["expected_images"] = 5
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in CLASSES[:-1]:
                for path in sorted((DATA / "raw/trashnet" / label).glob("*.jpg"))[:5]:
                    destination = root / path.relative_to(DATA)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            organic = [r for r in registry if r["original_category"] == "Food Organics"][:5]
            for row in organic:
                destination = root / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DATA / row["path"], destination)
            # B no se filtra a A aunque su registro este presente.
            unrelated = {**organic[0], "original_category": "Plastic", "class": "plastico", "path": "unused-B.jpg"}
            write_json(root / "raw/realwaste/provenance.json", organic + [unrelated])
            self.assertEqual(len(source_records(root, config, selected)), 30)
            summary = build_experiment(root, name, config, selected)
            self.assertEqual(summary["total"], 30)
            self.assertFalse(verify_phase1(root))
            self.assertTrue(verify_experiment(root, name, config, selected)["valid"])
            manifest = root / "experiments/A/processed/manifest.csv"
            first_hash = sha256(manifest)
            with self.assertRaises(ValueError):
                build_experiment(root, name, config, selected)
            build_experiment(root, name, config, selected, rebuild=True)
            self.assertEqual(first_hash, sha256(manifest))
            copied = next((root / "experiments/A/processed/train/plastico").glob("*.jpg"))
            copied.write_bytes(b"tampered")
            self.assertFalse(verify_experiment(root, name, config, selected)["valid"])
            original = next((root / "raw/trashnet/plastico").glob("*.jpg"))
            original.write_bytes(b"tampered phase1")
            self.assertTrue(verify_phase1(root))


if __name__ == "__main__":
    unittest.main()
