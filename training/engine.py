"""Entrenamiento explicito: solo train y val, sin ejecucion al importar."""
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from training.data import read_manifest, resolve_path, training_loaders, verify_files
from training.metrics import plot_history, scores
from training.model import ARCHITECTURE, VisionCNN, parameter_count
from training.runtime import EarlyStopping, choose_device, save_checkpoint, seed_everything
from utils.config import CLASSES
from utils.dataset import write_json


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    truth, predictions, accumulated_loss, total = [], [], 0., 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("La loss no es finita; revisar datos y configuracion.")
            if training:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            accumulated_loss += float(loss.detach()) * labels.size(0)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if total == 0:
        raise ValueError("DataLoader vacio.")
    return {"loss": accumulated_loss / total, **scores(truth, predictions)}


def write_history(output, history):
    write_json(output / "history.json", history)
    temporary = output / "history.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "history.csv")
    plot_history(history, output / "training_curves.png")


def train(config):
    dataset_dir, output = resolve_path(config["dataset_dir"]), resolve_path(config["output_dir"])
    if output.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output):
        raise ValueError("La salida debe ser independiente del directorio de datos.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("La salida ya contiene resultados. Elija otra carpeta para conservar la ejecucion anterior.")
    rows, summary = read_manifest(dataset_dir)
    if summary.get("experiment") != config["experiment"]:
        raise ValueError("El experimento del manifiesto y la configuracion no coinciden.")
    verify_files(dataset_dir, rows, ("train", "val"))
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    loaders = training_loaders(dataset_dir, config, rows, device.type == "cuda")
    model = VisionCNN(dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    stopper = EarlyStopping(**config["early_stopping"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "class_to_idx.json", {label: index for index, label in enumerate(CLASSES)})
    write_json(output / "run_metadata.json", {
        "architecture": ARCHITECTURE, "parameters": parameter_count(model), "initialization": "random_from_scratch",
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "manifest_sha256": summary["manifest_sha256"], "started_utc": datetime.now(timezone.utc).isoformat(),
        "training_images": len(loaders["train"].dataset), "validation_images": len(loaders["val"].dataset),
        "test_used_during_training": False,
    })
    history, best_f1, best_epoch, stopped_early = [], -float("inf"), 0, False
    print(f"Dispositivo: {device}; parametros: {parameter_count(model)}; solo train/val.", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            save_checkpoint(output / "best_model.pth", model, config, epoch, best_f1, summary["manifest_sha256"])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}, "best_checkpoint": is_best,
               "seconds": time.perf_counter() - started}
        history.append(row)
        write_history(output, history)
        print(json.dumps(row), flush=True)
        if stopper.step(val_metrics["macro_f1"]):
            stopped_early = True
            print(f"Early stopping por macro F1 de validacion en epoca {epoch}.", flush=True)
            break
    write_json(output / "training_summary.json", {
        "epochs_completed": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "stopped_early": stopped_early, "test_used_during_training": False,
        "checkpoint": "best_model.pth", "selection_metric": "validation_macro_f1",
    })
    return output
