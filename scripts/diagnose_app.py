"""Regresión app/evaluación sobre una muestra estratificada de VALIDACIÓN B; nunca test."""
import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from app.localization import Localizer, manual_region
from app.field_data import FIELD_ROOT
from app.settings import checkpoint_signature, load_settings
from app.vision import InferenceService, bgr_to_rgb, central_region, decode_image
from training.data import WasteDataset, read_manifest
from training.prediction import probabilities_from_logits
from utils.config import PROJECT_ROOT
from utils.dataset import sha256, write_json


def diagnose(per_class=10, checkpoint=None, output=None, manual_regions=None):
    if not 1 <= per_class <= 30:
        raise ValueError("Selecciona entre 1 y 30 fotografías de validación por clase.")
    root = PROJECT_ROOT / "data/experiments/B/processed"
    rows, summary = read_manifest(root)
    # Esta regresión histórica compara B; no debe cambiar al actualizar el modelo activo.
    reference = "runs/synced/B/B_20261121_032819_705556Z_19e1a326/best_model.pth"
    service = InferenceService(checkpoint_signature(checkpoint or reference)[0], 2)
    if service.metadata["config"].get("experiment") != "B" or service.metadata["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Se requiere el checkpoint B correspondiente al manifiesto B.")
    selected, counts = [], Counter()
    for row in rows:
        if row["split"] == "val" and counts[row["class"]] < per_class:
            selected.append(row)
            counts[row["class"]] += 1
    if any(counts[label] != per_class for label in service.classes):
        raise ValueError("No hay suficientes fotografías de validación por clase.")
    dataset = WasteDataset(root, "val", service.metadata["config"], selected)
    with (PROJECT_ROOT / "runs/B_validation/predictions.csv").open(encoding="utf-8", newline="") as handle:
        baseline = {row["path"]: row for row in csv.DictReader(handle)}
    records, tensor_error, probability_error = [], 0., 0.
    gallery = Image.new("RGB", (900, 6*220), "white")
    draw = ImageDraw.Draw(gallery)
    shown = Counter()
    for index, row in enumerate(selected):
        path = root / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("Fotografía val modificada: " + row["path"])
        rgb = decode_image(path.read_bytes())
        # Emular el contrato de captura: OpenCV entrega BGR y se convierte una sola vez.
        camera_rgb = bgr_to_rgb(rgb[:, :, ::-1].copy())
        reference_tensor, _ = dataset[index]
        full = Localizer("full").locate(camera_rgb).region
        app_tensor = service.preprocess(camera_rgb, full)[0]
        tensor_delta = float((app_tensor-reference_tensor).abs().max())
        with torch.inference_mode():
            expected = probabilities_from_logits(service.model(reference_tensor.unsqueeze(0)))[0].numpy()
        reading = service.classify(camera_rgb, mode="full")
        observed = np.array([reading.probabilities[label] for label in service.classes])
        error = float(np.abs(observed-expected).max())
        historical = baseline[row["path"]]
        historical_error = max(abs(observed[i]-float(historical["prob_"+label])) for i, label in enumerate(service.classes))
        central = central_region(rgb)
        old = service.classify(rgb, region=central)
        automatic = Localizer("auto").locate(rgb)
        records.append({"path": row["path"], "actual": row["class"], "baseline_predicted": historical["predicted"],
                        "app_full_predicted": reading.material, "legacy_central_predicted": old.material,
                        "crop_changes_prediction": old.material != reading.material,
                        "tensor_max_abs_error": tensor_delta, "probability_max_abs_error": error,
                        "historical_probability_max_abs_error": historical_error,
                        "auto_found": automatic.region is not None, "auto_message": automatic.message})
        tensor_error, probability_error = max(tensor_error, tensor_delta), max(probability_error, error)
        if shown[row["class"]] == 0:
            line = list(service.classes).index(row["class"])*220
            original = Image.fromarray(rgb)
            preview = original.copy()
            ImageDraw.Draw(preview).rectangle(central.box, outline="red", width=3)
            cells = [preview, original.crop(central.box), original.resize((128, 128), Image.Resampling.BILINEAR)]
            captions = [f"{row['class']}: original / marco anterior", f"Recorte anterior: {old.material}", f"Entrada completa: {reading.material}"]
            for column, (cell, caption) in enumerate(zip(cells, captions)):
                thumb = ImageOps.contain(cell, (288, 180))
                gallery.paste(thumb, (column*300, line+25))
                draw.text((column*300+4, line+5), caption, fill="black")
            shown[row["class"]] += 1
    mismatches = sum(record["app_full_predicted"] != record["baseline_predicted"] for record in records)
    historical_error = max(record["historical_probability_max_abs_error"] for record in records)
    report = {"scope": "Regresión sobre val B; no mide rendimiento de cámara.", "images": len(records),
              "per_class": dict(counts), "tensor_max_abs_error": tensor_error, "probability_max_abs_error": probability_error,
              "historical_probability_max_abs_error": historical_error, "historical_prediction_mismatches": mismatches,
              "legacy_crop_prediction_changes": sum(record["crop_changes_prediction"] for record in records),
              "legacy_crop_correct": sum(record["legacy_central_predicted"] == record["actual"] for record in records),
              "full_image_correct": sum(record["app_full_predicted"] == record["actual"] for record in records),
              "auto_candidates": sum(record["auto_found"] for record in records), "checkpoint_sha256": service.checkpoint_sha256,
              "classes": list(service.classes), "normalization": service.metadata["config"]["normalization"],
              "architecture_parameters": sum(p.numel() for p in service.model.parameters()), "test_evaluated": False,
              "passed": tensor_error == 0 and probability_error <= 1e-6 and historical_error <= 2e-6 and mismatches == 0}
    if output is None:
        output = PROJECT_ROOT / "runs/diagnostics" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ_")+uuid.uuid4().hex[:8])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "regression.json", report)
    with (output / "regression.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    gallery.save(output / "crop_comparison.jpg", quality=92)
    # Fotografías reales ya etiquetadas por una persona; nunca se agregan al dataset B.
    field_records = []
    regions_path = Path(manual_regions or PROJECT_ROOT / "config/field_regions_initial.csv")
    manual_boxes = {}
    if regions_path.exists():
        with regions_path.open(encoding="utf-8-sig", newline="") as handle:
            manual_boxes = {row["filename"]: tuple(float(row[key]) for key in ("left", "top", "right", "bottom")) for row in csv.DictReader(handle)}
    field_paths = sorted((FIELD_ROOT / "samples").glob("*/record.json"))
    field_gallery = Image.new("RGB", (1200, max(1, len(field_paths))*220), "white")
    field_draw = ImageDraw.Draw(field_gallery)
    for field_index, record_path in enumerate(field_paths):
        saved = json.loads(record_path.read_text(encoding="utf-8"))
        photo = record_path.parent / "original.png"
        if sha256(photo) != saved["files"]["original.png"]:
            raise ValueError("Foto real modificada: " + str(photo))
        rgb = decode_image(photo.read_bytes())
        full_reading = service.classify(rgb, mode="full")
        old_reading = service.classify(rgb, region=central_region(rgb))
        location = Localizer("auto").locate(rgb)
        auto_reading = service.classify(rgb, region=location.region) if location.region else None
        name = saved["message"].partition("Archivo: ")[-1] or saved["id"]
        selected_region = manual_region(rgb, manual_boxes[name]) if name in manual_boxes else None
        manual_reading = service.classify(rgb, region=selected_region) if selected_region else None
        field_records.append({"id": saved["id"], "source": name,
                              "actual": saved["true_class"], "full_predicted": full_reading.material,
                              "full_confidence": full_reading.confidence, "legacy_predicted": old_reading.material,
                              "legacy_confidence": old_reading.confidence,
                              "auto_predicted": auto_reading.material if auto_reading else None,
                              "auto_region": list(location.region.box) if location.region else None,
                              "manual_region": list(selected_region.box) if selected_region else None,
                              "manual_predicted": manual_reading.material if manual_reading else None,
                              "manual_confidence": manual_reading.confidence if manual_reading else None,
                              "auto_message": location.message})
        original = Image.fromarray(rgb)
        framed = original.copy()
        central = central_region(rgb)
        ImageDraw.Draw(framed).rectangle(central.box, outline="red", width=3)
        cells = [framed, original.crop(central.box), original.resize((128, 128), Image.Resampling.BILINEAR)]
        titles = [f"Real: {saved['true_class']} / marco anterior", f"Recorte anterior: {old_reading.material}", f"Completa: {full_reading.material}"]
        if manual_reading:
            cells.append(original.crop(selected_region.box))
            titles.append(f"Manual inspeccionada: {manual_reading.material}")
        for column, (cell, title) in enumerate(zip(cells, titles)):
            field_gallery.paste(ImageOps.contain(cell, (288, 180)), (column*300, field_index*220+25))
            field_draw.text((column*300+4, field_index*220+5), title, fill="black")
    if field_records:
        field_gallery.save(output / "field_crop_comparison.jpg", quality=92)
        write_json(output / "field_comparison.json", {"checkpoint_sha256": service.checkpoint_sha256, "cases": field_records})
        lines = ["# Fotos reales: comparación del encuadre", "", "Etiquetas confirmadas por el usuario; escenas correlacionadas, muestra insuficiente.", "",
                 "| Foto | Real | Completa | Confianza completa | Recorte anterior | Manual | Automática |", "|---|---|---|---:|---|---|---|"]
        for case in field_records:
            lines.append(f"| {case['source']} | {case['actual']} | {case['full_predicted']} | {case['full_confidence']:.1%} | {case['legacy_predicted']} | {case['manual_predicted'] or 'sin ROI anotada'} | {case['auto_predicted'] or 'sin región fiable'} |")
        lines += ["", "Son comparaciones de las mismas fotos, no nuevas muestras ni métricas del experimento B. "
                  "La localización es geométrica y una propuesta puede ser un fragmento equivocado: revisar el recorte manualmente. "
                  "Las cajas manuales de config/field_regions_initial.csv fueron elegidas una vez por inspección visual para diagnóstico, "
                  "sin buscar la mayor probabilidad. No son anotaciones de cajas confirmadas por el usuario. Algunas capturas ya cortan el objeto original."]
        (output / "FIELD_COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Regresión del flujo de inferencia\n\n" + json.dumps(report, ensure_ascii=False, indent=2) +
        "\n\nLa imagen completa reproduce validación. Un recorte distinto no tiene por qué producir la misma predicción. "
        "Los candidatos automáticos son propuestas geométricas sin anotaciones reales de cajas; encontrarlos no prueba localización correcta. "
        "Esta muestra no sustituye las métricas B ni mide mejoras con cámara. Test permanece reservado.\n", encoding="utf-8")
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--manual-regions", help="CSV opcional de cajas para comparar las fotos reales; por defecto config/field_regions_initial.csv.")
    args = parser.parse_args()
    report, output = diagnose(args.per_class, args.checkpoint, args.output_dir, args.manual_regions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Informe: " + str(output))
    if not report["passed"]:
        raise SystemExit("ERROR: la regresión no coincide con validación.")


if __name__ == "__main__":
    main()
