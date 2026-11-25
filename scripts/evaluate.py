"""Evaluacion final independiente: ejecutar solo despues de fijar el mejor modelo."""
import argparse
from pathlib import Path

import torch
from torch import nn

from training.data import make_loader, read_manifest, resolve_path, verify_files
from training.metrics import final_metrics, plot_confusion
from training.runtime import choose_device, load_checkpoint
from utils.config import CLASSES
from utils.dataset import sha256, write_json
from utils.experiments import write_csv


def evaluate(checkpoint_path, dataset_dir, output, device="cpu", batch_size=32, num_workers=0):
    checkpoint_path, dataset_dir, output = Path(checkpoint_path), Path(dataset_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output):
        raise ValueError("La salida de evaluacion debe ser independiente del dataset.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("La salida ya contiene resultados; elija otra carpeta.")
    device = choose_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    rows, summary = read_manifest(dataset_dir)
    if checkpoint["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("El checkpoint y las particiones tienen manifiestos diferentes.")
    verify_files(dataset_dir, rows, ("test",))
    config = {**checkpoint["config"], "batch_size": batch_size, "num_workers": num_workers}
    loader = make_loader(dataset_dir, "test", config, rows, device.type == "cuda")
    truth, predictions, losses = [], [], 0.
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses += float(criterion(logits, labels)) * labels.size(0)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    report = {**final_metrics(truth, predictions), "loss": losses / len(truth), "images": len(truth),
              "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
              "manifest_sha256": summary["manifest_sha256"], "split": "test"}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", report)
    write_csv(output / "per_class_metrics.csv", [
        {"class": label, **report["per_class"][label]} for label in CLASSES
    ], ("class", "precision", "recall", "f1-score", "support"))
    write_csv(output / "confusion_matrix.csv", [
        {"actual_class": label, **dict(zip(CLASSES, report["confusion_matrix"][index]))}
        for index, label in enumerate(CLASSES)
    ], ("actual_class", *CLASSES))
    write_csv(output / "predictions.csv", [
        {"path": row["path"], "dataset": row["dataset"], "source_id": row["source_id"],
         "actual": CLASSES[actual], "predicted": CLASSES[predicted], "sha256": row["sha256"]}
        for row, actual, predicted in zip(loader.dataset.rows, truth, predictions)
    ], ("path", "dataset", "source_id", "actual", "predicted", "sha256"))
    plot_confusion(report["confusion_matrix"], output / "confusion_matrix.png")
    print(f"Evaluacion final: {len(truth)} imagenes; accuracy={report['accuracy']:.4f}; macro F1={report['macro_f1']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="data/experiments/A/processed")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size debe ser positivo y num-workers no negativo.")
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output_dir) if args.output_dir else checkpoint.parent / "evaluation"
    evaluate(checkpoint, resolve_path(args.dataset_dir), output, args.device, args.batch_size, args.num_workers)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
