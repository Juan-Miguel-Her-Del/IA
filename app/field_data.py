"""Registro voluntario de fotos reales y análisis independiente de los datasets oficiales."""
import csv
import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.recycling import LABELS
from app.vision import crop_region
from training.compact import build_transform
from utils.config import PROJECT_ROOT, data_root
from utils.dataset import sha256, write_json

FIELD_ROOT = PROJECT_ROOT / "data/field_samples"


def pixel_hash(rgb):
    dimensions = (rgb.shape[1], rgb.shape[0])
    return hashlib.sha256(str(dimensions).encode() + b"\0" + rgb.tobytes()).hexdigest()


def official_pixels():
    """Solo metadatos; no abre ninguna fotografía oficial."""
    values = set()
    for root in {PROJECT_ROOT / "data", data_root()}:
        for experiment in ("A", "B"):
            manifest = root / "experiments" / experiment / "processed/manifest.csv"
            if manifest.exists():
                with manifest.open(encoding="utf-8", newline="") as handle:
                    values.update(row["pixel_sha256"] for row in csv.DictReader(handle))
    return values


def save_sample(rgb, reading, true_class, service, source, *, message="", root=FIELD_ROOT, known_pixels=None):
    if true_class not in service.classes or source not in ("Foto con cámara", "Video en vivo", "Subir imagen"):
        raise ValueError("Indica la categoría verdadera y una fuente válida.")
    digest = pixel_hash(rgb)
    if digest in (official_pixels() if known_pixels is None else known_pixels):
        raise ValueError("Esta fotografía coincide con datos oficiales; no puede registrarse como una foto real nueva.")
    root = Path(root).resolve()
    for data in {PROJECT_ROOT / "data", data_root()}:
        if any(root.is_relative_to((data / name).resolve()) for name in ("experiments", "processed", "raw")):
            raise ValueError("Las fotografías reales deben estar separadas de los datasets oficiales.")
    for path in (root / "samples").glob("*/record.json"):
        if json.loads(path.read_text(encoding="utf-8"))["original_pixel_sha256"] == digest:
            raise ValueError("Esta fotografía ya fue registrada. Captura una fotografía diferente.")
    crop = crop_region(rgb, reading.region) if reading else None
    if reading:
        transform = build_transform(service.metadata["config"], training=False)
        tensor = transform(Image.fromarray(crop)).unsqueeze(0)
        tensor_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        if hashlib.sha256(crop.tobytes()).hexdigest() != reading.crop_sha256 or tensor_hash != reading.tensor_sha256:
            raise ValueError("La foto y la lectura no corresponden al mismo recorte. Congela una lectura nueva.")
    identifier = uuid.uuid4().hex
    folder = root / "samples" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(folder / "original.png")
    if crop is not None:
        Image.fromarray(crop).save(folder / "crop.png")
        size = service.metadata["image_size"]
        transform = build_transform(service.metadata["config"], training=False)
        preview = transform.prepare_image(Image.fromarray(crop)) if hasattr(transform, "prepare_image") else Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR)
        preview.save(folder / f"input_{size}.png")
        if service.metadata["config"].get("inference_hflip",False):
            from PIL import ImageOps
            ImageOps.mirror(preview).save(folder / f"input_{size}_flipped.png")
    record = {"schema": 1, "id": identifier, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "true_class": true_class, "source": source, "original_pixel_sha256": digest,
              "checkpoint_sha256": service.checkpoint_sha256, "classes": list(service.classes),
              "normalization": service.metadata["config"]["normalization"], "image_size": service.metadata["image_size"],
              "inference_hflip": service.metadata["config"].get("inference_hflip",False),
              "inference_scale": service.metadata["config"].get("inference_scale",1.),
              "region": list(reading.region.box) if reading else None,
              "method": reading.region.method if reading else "sin-localizacion", "message": message,
              "predicted": reading.material if reading else None,
              "confidence": reading.confidence if reading else None,
              "probabilities": reading.probabilities if reading else {},
              "crop_sha256": reading.crop_sha256 if reading else None,
              "tensor_sha256": reading.tensor_sha256 if reading else None,
              "files": {path.name: sha256(path) for path in folder.glob("*.png")}}
    write_json(folder / "record.json", record)
    return folder


def analyze_samples(root=FIELD_ROOT, output=None, maximum=10):
    """Usa predicciones registradas, sin inferir ni entrenar. Separa método y checkpoint."""
    if maximum < 1:
        raise ValueError("El máximo por clase debe ser positivo.")
    root = Path(root).resolve()
    groups, seen = {}, set()
    paths = sorted((root / "samples").glob("*/record.json"))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, digest in record["files"].items():
            target = (path.parent / name).resolve()
            if not target.is_relative_to(path.parent.resolve()) or sha256(target) != digest:
                raise ValueError(f"Archivo modificado o inválido: {path}")
        actual, predicted = record["true_class"], record["predicted"]
        if actual not in LABELS or (predicted is not None and predicted not in LABELS):
            raise ValueError(f"Etiquetas inválidas: {path}")
        duplicate = record["original_pixel_sha256"]
        if duplicate in seen:
            raise ValueError("Hay fotografías exactas repetidas en el registro.")
        seen.add(duplicate)
        key = record["checkpoint_sha256"] + "/" + record["method"]
        groups.setdefault(key, []).append(record)
    report = {"scope": "Fotografías reales independientes; no es validación ni test oficial.",
              "records_available": len(paths), "maximum_per_class_per_group": maximum, "groups": {}}
    lines = ["# Diagnóstico con fotografías reales", "", report["scope"],
             "Se analizan las predicciones guardadas, sin recalcularlas ni entrenar.", ""]
    if not paths:
        lines.append("**No hay fotografías reales registradas. No se pueden calcular aciertos ni mejoras.**")
    for key, records in groups.items():
        records.sort(key=lambda r: (r["recorded_at"], r["id"]))
        counts, selected = Counter(), []
        for record in records:
            if counts[record["true_class"]] < maximum:
                selected.append(record)
                counts[record["true_class"]] += 1
        columns = [*LABELS, "sin-localizacion"]
        matrix = {label: {column: 0 for column in columns} for label in LABELS}
        for record in selected:
            matrix[record["true_class"]][record["predicted"] or "sin-localizacion"] += 1
        correct = sum(record["true_class"] == record["predicted"] for record in selected)
        classified = sum(record["predicted"] is not None for record in selected)
        values = {"images": len(selected), "counts": {label: counts[label] for label in LABELS},
                  "correct": correct, "incorrect_classifications": classified-correct,
                  "localization_failures": len(selected)-classified,
                  "accuracy_end_to_end": correct/len(selected),
                  "accuracy_classified_only": correct/classified if classified else None,
                  "confusion_matrix": matrix, "record_ids": [record["id"] for record in selected]}
        report["groups"][key] = values
        lines += [f"## {key}", "", f"Fotos: {len(selected)}; aciertos: {correct}; errores de clase: {classified-correct}; sin localización: {len(selected)-classified}.",
                  f"Aciertos / todas las fotos: {values['accuracy_end_to_end']:.1%}.", "",
                  "| Real / predicción | " + " | ".join(columns) + " |", "|---|" + "---:|"*len(columns)]
        for label in LABELS:
            lines.append("| " + label + " | " + " | ".join(str(matrix[label][column]) for column in columns) + " |")
        lines += ["", "Clases con menos de 5 fotos: " + ", ".join(label for label in LABELS if counts[label] < 5),
                  "Muestra exploratoria pequeña, posiblemente correlacionada. No mezclar checkpoints/métodos ni extrapolar a otros entornos.", ""]
    if output is None:
        output = PROJECT_ROOT / "runs/field_analysis" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ_") + uuid.uuid4().hex[:8])
    output = Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Guarda el informe fuera de la carpeta de muestras.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, output
