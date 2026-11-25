"""Evalúa la configuración real en val B y escenas; nunca abre test ni cambia pesos."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from app.detector import NeuralDetector, primary_detection, crop_box
from app.localization import Region
from app.vision import InferenceService
from scripts.diagnose_scene_detector import load_scene, validation_rows
from training.data import read_manifest
from training.detection_metrics import match, spatial_metrics
from utils.config import CLASSES, PROJECT_ROOT
from utils.quality_route import PROTOCOL, classification_metrics, file_hash, read_json, write_new_json


def verified_rgb(path, expected):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Imagen modificada: {path}")
    with Image.open(BytesIO(content)) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def joint_counts(predictions, truth, labels, material_predictions):
    known = sum(t["annotation_key"] in labels for t in truth)
    pairs = [(i, j) for i, j, _ in match(predictions, truth) if truth[j]["annotation_key"] in labels]
    correct = sum(material_predictions[i] == labels[truth[j]["annotation_key"]] for i, j in pairs)
    return {"objects": known, "matched": len(pairs), "correct": correct}


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f"El informe ya existe: {args.output}. Usa un nombre nuevo.")
    settings = read_json(args.config)
    rows = validation_rows(args.scenes)
    labels_manifest = read_json(args.labels)
    val_labels = [r for r in labels_manifest if r["split"] == "val"]
    labels = {(r["image_id"], r["id"]): r["class"] for r in val_labels}
    if len(labels) != len(val_labels) or not labels:
        raise ValueError("Etiquetas de escenas vacías o repetidas")
    all_original_rows, _ = read_manifest(args.original)
    original_rows = [r for r in all_original_rows if r["split"] == "val"]
    if not original_rows or not rows:
        raise ValueError("Falta validación B o de escenas")
    # Fallar antes de cargar modelos si las fotos aún no están restauradas.
    for row in original_rows:
        if not (args.original / row["path"]).is_file():
            raise FileNotFoundError("Restaurar B: python -m scripts.restore_public_data B")
    for row in rows:
        if not (args.scenes / "images" / "val" / f"{row['id']:05d}.jpg").is_file():
            raise FileNotFoundError("Restaurar escenas: python -m scripts.restore_public_data refine_D2")

    def checkpoint(key):
        return (PROJECT_ROOT / settings[key]).resolve()

    fingerprint_paths = {"config": args.config, "original_manifest": args.original / "manifest.csv",
                         "scene_manifest": args.scenes / "manifest.json", "material_manifest": args.labels}
    fingerprint_paths.update({key: checkpoint(key) for key in
                             ("checkpoint", "detector_checkpoint", "material_candidate_checkpoint") if settings.get(key)})
    fingerprints = {key: file_hash(path) for key, path in fingerprint_paths.items()}
    code_paths = ["scripts/evaluate_runtime.py", "scripts/diagnose_scene_detector.py", "utils/quality_route.py",
                  "app/vision.py", "app/detector.py", "app/localization.py", "training/runtime.py",
                  "training/data.py", "training/compact.py", "training/model.py", "training/prediction.py",
                  "training/detection_data.py", "training/detection_metrics.py", "training/detector_own.py"]
    code_hashes = {p: file_hash(PROJECT_ROOT / p) for p in code_paths}
    threads = settings["cpu_threads"]
    candidate = checkpoint("material_candidate_checkpoint") if settings.get("material_candidate_checkpoint") else None
    classifier = InferenceService(checkpoint("checkpoint"), threads, candidate,
                                  settings.get("material_candidate_threshold", .4),
                                  settings.get("material_candidate_weight"))
    for service in (classifier, getattr(classifier, "candidate", None)):
        if service and getattr(service, "metadata", {}).get("provenance", {}).get("smoke_test"):
            raise ValueError("Un checkpoint de prueba corta no es candidato de selección. Ejecutar el entrenamiento completo.")
    detector = NeuralDetector(checkpoint("detector_checkpoint"), threads, settings["detection_threshold"],
                              settings.get("detection_nms_iou", .45), settings.get("detection_views", ()))
    original_truth, original_predictions = [], []
    for i, row in enumerate(original_rows):
        rgb = verified_rgb(args.original / row["path"], row["sha256"])
        h, w = rgb.shape[:2]
        reading = classifier.classify(rgb, region=Region((0, 0, w, h), "evaluation-full"))
        original_truth.append(row["class"])
        original_predictions.append(reading.material)
        if i % 100 == 0:
            print(f"B val: {i + 1}/{len(original_rows)}", flush=True)

    oracle_truth, oracle_predictions, scene_records, timings = [], [], [], []
    runtime_decisions = []
    joint, primary_joint = defaultdict(int), defaultdict(int)
    single = {"images": 0, "correct": 0}
    errors = []
    seen_labels = set()
    for index, row in enumerate(rows):
        rgb = load_scene(args.scenes, row)
        h, w = rgb.shape[:2]
        truth = [{"box": [v * s for v, s in zip(b["xyxy_normalized"], (w, h, w, h))],
                  "category": b["category"], "annotation_key": (row["id"], b["annotation_id"])} for b in row["boxes"]]
        if index == 0:
            for _ in range(5):
                warm = primary_detection(detector.detect(rgb))
                if warm:
                    classifier.classify(rgb, region=warm.region)
        start = time.perf_counter()
        detections = detector.detect(rgb)
        primary = primary_detection(detections)
        primary_reading = classifier.classify(rgb, region=primary.region) if primary else None
        timings.append(1000 * (time.perf_counter() - start))
        # Ranking AP con umbral bajo; su coste NO se presenta como latencia de la app.
        ranking = detector.detect(rgb, confidence=.001)
        scene_records.append({"image_id": row["id"], "truth": truth, "size": (w, h),
                              "source": row.get("source_dataset", "TACO"),
                              "predictions": [{"box": d.box, "score": d.confidence} for d in ranking]})
        predictions = [{"box": d.box, "score": d.confidence} for d in detections]
        materials = [primary_reading.material if d is primary else classifier.classify(rgb, region=d.region).material
                     for d in detections]
        runtime_decisions.append({"image_id": row["id"], "detections": [
            {"box": [round(float(v), 6) for v in d.box], "score": round(float(d.confidence), 6), "material": material}
            for d, material in zip(detections, materials)],
            "primary_material": primary_reading.material if primary_reading else None})
        primary_predictions = [{"box": primary.box, "score": primary.confidence}] if primary else []
        totals = joint_counts(predictions, truth, labels, materials)
        primary_totals = joint_counts(primary_predictions, truth, labels, [primary_reading.material] if primary else [])
        for key, value in totals.items():
            joint[key] += value
        for key, value in primary_totals.items():
            primary_joint[key] += value
        # Denominador adecuado para una UI de un residuo: escena con un único objeto anotado y conocido.
        if len(truth) == 1 and truth[0]["annotation_key"] in labels:
            single["images"] += 1
            single["correct"] += primary_totals["correct"]
        if totals["correct"] < totals["objects"]:
            errors.append({"image_id": row["id"], "source": row.get("source_dataset", "TACO"), **totals})
        for item in truth:
            key = item["annotation_key"]
            if key not in labels:
                continue
            region = Region(crop_box(item["box"], w, h), "evaluation-human-box")
            oracle_truth.append(labels[key])
            oracle_predictions.append(classifier.classify(rgb, region=region).material)
            seen_labels.add(key)
        if index % 40 == 0:
            print(f"Escenas val: {index + 1}/{len(rows)}", flush=True)
    if seen_labels != set(labels):
        raise ValueError("Las etiquetas de material no corresponden a las escenas evaluadas")
    for totals in (joint, primary_joint):
        totals["recall"] = totals["correct"] / totals["objects"] if totals["objects"] else None
    single["accuracy"] = single["correct"] / single["images"] if single["images"] else None
    source_records = defaultdict(list)
    for record in scene_records:
        source_records[record["source"]].append(record)
    if any(file_hash(path) != fingerprints[key] for key, path in fingerprint_paths.items()):
        raise ValueError("Datos, configuración o pesos cambiaron durante la evaluación. Repetir con archivos estables.")
    if any(file_hash(PROJECT_ROOT / path) != digest for path, digest in code_hashes.items()):
        raise ValueError("El código cambió durante la evaluación; repetir.")
    checkpoint_hashes = {k: fingerprints[k] for k in
                         ("checkpoint", "detector_checkpoint", "material_candidate_checkpoint") if settings.get(k)}
    decision_bytes = json.dumps({"original": original_predictions, "human_crop": oracle_predictions,
                                 "runtime": runtime_decisions}, sort_keys=True).encode("utf-8")
    report = {
        "schema_version": 1, "protocol": PROTOCOL, "evidence": "fresh_runtime_inference", "test_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": settings,
        "config_sha256": file_hash(args.config), "checkpoint_hashes": checkpoint_hashes,
        "evaluation_code": code_hashes,
        "reproducibility_digest": hashlib.sha256(decision_bytes).hexdigest(),
        "cohort": {"original_manifest_sha256": file_hash(args.original / "manifest.csv"),
                   "scene_manifest_sha256": file_hash(args.scenes / "manifest.json"),
                   "material_manifest_sha256": file_hash(args.labels), "original_images": len(original_rows),
                   "scene_images": len(rows), "material_objects": len(labels), "split": "val",
                   "crop": "raw_RGB_square_12_percent_margin", "classes": list(CLASSES)},
        "environment": {"processor": platform.processor(), "machine": platform.machine(),
                        "python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "pillow": Image.__version__,
                        "opencv": __import__("cv2").__version__, "cpu_threads": threads, "device": "cpu"},
        "metrics": {"original": classification_metrics(original_truth, original_predictions, CLASSES),
                    "scene": classification_metrics(oracle_truth, oracle_predictions, CLASSES),
                    "detector": spatial_metrics(scene_records, settings["detection_threshold"]),
                    "detector_by_source": {s: spatial_metrics(r, settings["detection_threshold"]) for s, r in source_records.items()},
                    "joint": dict(joint), "primary_joint_diagnostic": dict(primary_joint),
                    "single_object_scenes": single,
                    "latency": {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                                "images": len(timings), "warmup": 5,
                                "scope": "CPU detector + primary classifier; no decoding/render/network"}},
        "errors": errors,
        "limitations": ["Validación ya usada para seleccionar; no es test independiente",
                        "Material en recortes RGB actuales; no mezclar con métricas antiguas sobre JPEG reencodificados",
                        "primary_joint_diagnostic cuenta todos los objetos; single_object_scenes solo cuenta escenas de un objeto",
                        "Sin medición de vídeo, escenas vacías ni objetos desconocidos"]}
    write_new_json(args.output, report)
    print(f"Informe nuevo: {args.output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/app.json")
    parser.add_argument("--original", type=Path, default=PROJECT_ROOT / "data/experiments/B/processed")
    parser.add_argument("--scenes", type=Path, default=PROJECT_ROOT / "data/detection/refine_D2")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data/reference/scene_materials/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"No se pudo evaluar: {exc}\n")


if __name__ == "__main__":
    main()
