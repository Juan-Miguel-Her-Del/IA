"""Analisis exclusivo de validacion: A (421) o B (884), nunca train/test."""
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn

from training.data import make_loader, read_manifest, verify_files
from training.metrics import final_metrics, plot_confusion
from training.prediction import probabilities_from_logits
from training.runtime import choose_device, load_checkpoint, seed_everything
from utils.dataset import sha256, write_json
from utils.dataset_samples import font
from utils.experiments import write_csv


def validate_selection(checkpoint, summary, rows, experiment=None):
    expected_counts = {"A": 421, "B": 884}
    selected = experiment or summary.get("experiment")
    if selected not in expected_counts:
        raise ValueError("La evaluacion de validacion admite solo A o B.")
    if summary.get("experiment") != selected or checkpoint["config"].get("experiment") != selected:
        raise ValueError("Checkpoint, particiones y experimento seleccionado no coinciden.")
    expected = expected_counts[selected]
    if sum(row["split"] == "val" for row in rows) != expected:
        raise ValueError(f"Se exigen exactamente {expected} imagenes de VALIDACION del experimento {selected}.")
    if checkpoint["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("El checkpoint no corresponde a estas particiones: SHA-256 de manifiesto distinto.")
    return selected, expected


def validation_metrics(truth, predicted, classes):
    result = final_metrics(truth, predicted)
    if result["class_order"] != classes:
        raise ValueError("Las metricas y el checkpoint utilizan ordenes de clases distintos.")
    matrix = result["confusion_matrix"]
    total = len(truth)
    per_class = {}
    for index, label in enumerate(classes):
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        tn = total - tp - fn - fp
        values = result["per_class"][label]
        per_class[label] = {"precision": values["precision"], "recall": values["recall"],
                            "f1": values["f1-score"], "support": int(values["support"]),
                            "accuracy_one_vs_rest": (tp + tn) / total,
                            "accuracy_on_class": tp / (tp + fn) if tp + fn else 0.,
                            "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    result["per_class"] = per_class
    result["accuracy_definitions"] = {
        "accuracy": "Predicciones correctas / total de validacion.",
        "accuracy_one_vs_rest": "(TP + TN) / total de validacion, considerando cada clase frente al resto.",
        "accuracy_on_class": "TP / soporte de esa clase; coincide matematicamente con recall.",
    }
    return result


def plot_misclassified(dataset_dir, records, path, maximum=24):
    errors = sorted((r for r in records if not r["correct"]), key=lambda r: (-r["confidence"], r["path"]))[:maximum]
    if not errors:
        image = Image.new("RGB", (800, 100), "white")
        ImageDraw.Draw(image).text((20, 35), "No hay clasificaciones incorrectas en esta evaluacion de validacion.", fill="black", font=font(16))
        image.save(path)
        return []
    columns, width, height = 4, 280, 280
    sheet = Image.new("RGB", (columns * width, math.ceil(len(errors) / columns) * height), "#eeeeee")
    for index, record in enumerate(errors):
        left, top = (index % columns) * width, (index // columns) * height
        with Image.open(Path(dataset_dir) / record["path"]) as original:
            photo = ImageOps.contain(ImageOps.exif_transpose(original).convert("RGB"), (width - 12, 196))
            sheet.paste(photo, (left + (width - photo.width) // 2, top + 3))
        draw = ImageDraw.Draw(sheet)
        lines = [f"Real: {record['actual']}", f"Prediccion: {record['predicted']} | {record['confidence']:.1%}", record["source_id"][:34]]
        for number, text in enumerate(lines):
            draw.text((left + 8, top + 207 + 22 * number), text, fill="black", font=font(15))
    sheet.save(path, quality=92)
    return [record["path"] for record in errors]


def evaluate_validation(checkpoint_path, dataset_dir, output, device="cpu", batch_size=32, num_workers=0, maximum_errors=24, experiment=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    dataset_dir, output = Path(dataset_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output):
        raise ValueError("La salida debe estar fuera del dataset.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("La salida ya contiene resultados. Elija otra carpeta o use el destino automatico por fecha.")
    if batch_size < 1 or num_workers < 0 or maximum_errors < 1:
        raise ValueError("batch_size/max_errors deben ser positivos; num_workers no negativo.")
    device = choose_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    seed_everything(checkpoint["config"]["seed"])
    rows, summary = read_manifest(dataset_dir)
    validation_rows = [row for row in rows if row["split"] == "val"]
    selected, expected = validate_selection(checkpoint, summary, rows, experiment)
    # read_manifest lee metadatos de separacion; solo se verifican/abren fotografias val.
    verify_files(dataset_dir, rows, ("val",))
    config = {**checkpoint["config"], "batch_size": batch_size, "num_workers": num_workers}
    loader = make_loader(dataset_dir, "val", config, validation_rows, device.type == "cuda")
    classes = checkpoint["classes"]
    if loader.dataset.class_to_idx != {label: index for index, label in enumerate(classes)}:
        raise ValueError("DataLoader y checkpoint no coinciden en el orden de las clases.")
    truth, predicted, probabilities, accumulated_loss = [], [], [], 0.
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probability_batch = probabilities_from_logits(logits)
            accumulated_loss += float(criterion(logits, labels)) * labels.size(0)
            truth.extend(labels.cpu().tolist())
            predicted.extend(probability_batch.argmax(dim=1).cpu().tolist())
            probabilities.extend(probability_batch.cpu().tolist())
    if len(truth) != expected:
        raise ValueError(f"No se evaluaron exactamente las {expected} imagenes previstas.")
    records = []
    for row, actual, prediction, probability in zip(loader.dataset.rows, truth, predicted, probabilities):
        records.append({
            "path": row["path"], "dataset": row["dataset"], "source_id": row["source_id"],
            "actual": classes[actual], "predicted": classes[prediction], "confidence": probability[prediction],
            "correct": actual == prediction,
            **{f"prob_{label}": probability[index] for index, label in enumerate(classes)},
        })
    metrics = validation_metrics(truth, predicted, classes)
    report = {**metrics, "loss": accumulated_loss / len(truth), "experiment": selected, "split": "val", "images": len(truth),
              "correct": sum(r["correct"] for r in records), "incorrect": sum(not r["correct"] for r in records),
              "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
              "checkpoint_sha256": sha256(checkpoint_path), "manifest_sha256": summary["manifest_sha256"],
              "normalization": checkpoint["config"]["normalization"], "architecture": checkpoint["architecture"],
              "device": str(device), "test_evaluated": False,
              "confidence_definition": "Probabilidad softmax maxima, no calibrada.",
              "deterministic_algorithms": {"enabled": True, "warn_only": True},
              "torch_version": str(torch.__version__), "training_torch_version": checkpoint.get("torch_version")}
    output.mkdir(parents=True, exist_ok=True)
    fields = ("path", "dataset", "source_id", "actual", "predicted", "confidence", "correct", *(f"prob_{label}" for label in classes))
    write_csv(output / "predictions.csv", records, fields)
    write_csv(output / "misclassified.csv", [r for r in records if not r["correct"]], fields)
    write_csv(output / "per_class_metrics.csv", [{"class": label, **report["per_class"][label]} for label in classes],
              ("class", "precision", "recall", "f1", "support", "accuracy_one_vs_rest", "accuracy_on_class", "tp", "tn", "fp", "fn"))
    write_csv(output / "confusion_matrix.csv", [
        {"actual_class": label, **dict(zip(classes, report["confusion_matrix"][index]))} for index, label in enumerate(classes)
    ], ("actual_class", *classes))
    plot_confusion(report["confusion_matrix"], output / "confusion_matrix.png", classes, f"Matriz de confusion - VALIDACION {selected} ({expected} imagenes)")
    shown = plot_misclassified(dataset_dir, records, output / "misclassified_examples.jpg", maximum_errors)
    report["visualized_errors"] = shown
    write_json(output / "metrics.json", report)
    lines = [f"# Analisis exclusivo de validacion {selected}", "",
             f"Imagenes: {expected}. Correctas: {report['correct']}. Incorrectas: {report['incorrect']}.",
             f"Accuracy global: {report['accuracy']:.6f}. Macro F1: {report['macro_f1']:.6f}.",
             f"Checkpoint: epoca {report['checkpoint_epoch']}. Test evaluado: NO.", "",
             "| Categoria | Precision | Recall | F1 | Accuracy OvR | Soporte |", "|---|---:|---:|---:|---:|---:|"]
    for label in classes:
        row = report["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy_one_vs_rest']:.4f} | {row['support']} |")
    lines += ["", f"Accuracy OvR = (TP+TN)/{expected}. Accuracy sobre las imagenes de cada clase coincide con recall.",
              "La confianza es softmax y no esta calibrada. Se muestran los errores de mayor confianza primero.",
              "Validacion ya intervino en la seleccion del checkpoint: este analisis no es una estimacion independiente final."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"VALIDACION {selected}: {expected} imagenes; accuracy={report['accuracy']:.6f}; macro F1={report['macro_f1']:.6f}.")
    print(f"Correctas: {report['correct']}; incorrectas: {report['incorrect']}. PRUEBA no evaluada.")
    print("Resultados: " + str(output))
    return report
