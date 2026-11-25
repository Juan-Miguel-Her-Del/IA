"""Configuracion y procedencia de experimentos independientes de la fase 1."""
import csv
import io
import json
import shutil
from pathlib import Path

from utils.config import CLASSES, PROJECT_ROOT, TRASHNET_CLASSES
from utils.dataset import image_paths, sha256, write_json

CONFIG_PATH = PROJECT_ROOT / "config" / "experiments.json"


def load_experiment(name=None, config_path=None):
    path = Path(config_path) if config_path else CONFIG_PATH
    config = json.loads(path.read_text(encoding="utf-8"))
    name = name or config["active_experiment"]
    if name not in ("A", "B") or name not in config["experiments"]:
        raise ValueError("Experimento desconocido; elija A o B.")
    selected = config["experiments"][name]
    categories = selected["realwaste_categories"]
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("La seleccion RealWaste esta vacia o contiene categorias repetidas.")
    if any(c not in config["realwaste"]["categories"] for c in categories):
        raise ValueError("Categoria RealWaste fuera de las seis permitidas.")
    mappings = config["realwaste"]["categories"]
    if set(row["class"] for row in mappings.values()) != set(CLASSES) or any(
            not isinstance(row["expected_images"], int) or row["expected_images"] <= 0 for row in mappings.values()):
        raise ValueError("El mapeo RealWaste debe cubrir las seis clases con cantidades positivas.")
    ratios = list(config["ratios"].values())
    if list(config["ratios"]) != ["train", "val", "test"] or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Las proporciones train/val/test deben sumar 1.")
    visual = config["visual_audit"]
    if not 0 <= visual["phash_distance"] <= 16 or not 0 <= visual["dhash_distance"] <= 20 or visual["orientations"] != 8:
        raise ValueError("Configuracion visual fuera de los limites admitidos.")
    return name, config, selected


def write_csv(path, rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output.getvalue(), encoding="utf-8", newline="\n")


def preserve_phase1(root):
    """Registra y copia informes; no mueve ni modifica originales o particiones."""
    destination = root / "experiments" / "phase1_snapshot"
    index = destination / "files.json"
    if not index.exists():
        paths = image_paths(root / "raw" / "trashnet") + image_paths(root / "processed")
        for relative in ("reports/audit.json", "reports/statistics.json", "reports/imports.json",
                         "processed/manifest.csv", "processed/split_summary.json"):
            path = root / relative
            if path.exists():
                paths.append(path)
                backup = destination / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, backup)
        write_json(index, {p.relative_to(root).as_posix(): sha256(p) for p in paths})
    return index


def verify_phase1(root):
    index = root / "experiments" / "phase1_snapshot" / "files.json"
    if not index.exists():
        return ["Falta el registro de preservacion de fase 1."]
    expected = json.loads(index.read_text(encoding="utf-8"))
    return [f"Fase 1 modificada: {relative}" for relative, digest in expected.items()
            if not (root / relative).is_file() or sha256(root / relative) != digest]


def source_records(root, config, selected):
    """Nunca descubre clases por glob global: solo las declaradas en el experimento."""
    rows = []
    inverse = {label: original for original, label in TRASHNET_CLASSES.items()}
    for label in CLASSES[:-1]:
        for path in image_paths(root / "raw" / "trashnet" / label):
            # El importador anterior antepuso SHA256_ al nombre original.
            original = path.name[65:] if len(path.name) > 65 and path.name[64] == "_" else path.name
            rows.append({
                "dataset": "TrashNet", "original_category": inverse[label], "source_id": Path(original).stem,
                "source_path": f"dataset-resized/{inverse[label]}/{original}",
                "source_url": "https://github.com/garythung/trashnet",
                "license": "MIT", "path": path.relative_to(root).as_posix(), "class": label,
            })
    registry_path = root / "raw" / "realwaste" / "provenance.json"
    if not registry_path.exists():
        raise ValueError("Falta RealWaste. Ejecute scripts.prepare_realwaste para este experimento.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    allowed = set(selected["realwaste_categories"])
    for category in sorted(allowed):
        entries = [r for r in registry if r["original_category"] == category]
        expected = config["realwaste"]["categories"][category]["expected_images"]
        if len(entries) != expected:
            raise ValueError(f"{category}: se esperaban {expected} registros de origen; hay {len(entries)}.")
        for entry in entries:
            if entry["class"] != config["realwaste"]["categories"][category]["class"]:
                raise ValueError(f"Mapeo de clase inesperado en {entry['path']}")
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to((root / "raw" / "realwaste").resolve()):
                raise ValueError("La procedencia contiene una ruta fuera de raw/realwaste.")
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Archivo de origen ausente o alterado: {entry['path']}")
            rows.append(dict(entry))
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Rutas repetidas en el inventario de procedencia.")
    return sorted(rows, key=lambda row: row["path"])
