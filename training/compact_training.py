"""Entrenamiento CUDA desde cero sobre train/val B intactos; checkpoints propios reanudables."""
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from training.compact import ARCHITECTURE, CompactCNN
from training.data import WasteDataset, read_manifest, seed_worker, verify_files, resolve_path
from training.metrics import final_metrics, plot_history, scores
from training.runtime import seed_everything
from utils.config import CLASSES
from utils.dataset import write_json, sha256


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    truth, predicted, probabilities = [], [], []
    total_loss = 0.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach())*len(labels)
            truth.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(1).cpu().tolist())
            if not training:
                probabilities.extend(logits.detach().float().softmax(1).cpu().tolist())
    return {"loss": total_loss/len(truth), **scores(truth, predicted)}, truth, predicted, probabilities


def train_compact(config, output, resume=False):
    if config["initialization"] != "random_from_scratch" or config["architecture"] != ARCHITECTURE:
        raise ValueError("Solo arquitectura propia con inicialización aleatoria.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    seed_everything(config["seed"])
    dataset = Path(config["dataset_dir"]).resolve()
    output = Path(output).resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("La salida debe estar fuera del dataset.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Salida ocupada: elegir otra carpeta o reanudar explícitamente.")
    rows, summary = read_manifest(dataset)
    if summary["experiment"] != config["data_experiment"]:
        raise ValueError("Experimento de datos incorrecto.")
    verify_files(dataset, rows, ("train", "val"))
    generator = torch.Generator().manual_seed(config["seed"])
    supplement,extra_manifest_sha=None,None
    if config.get("extra_train_manifest"):
        from training.supplement import TrainingSupplement
        supplement=TrainingSupplement(resolve_path(config["extra_train_manifest"]),config)
        extra_manifest_sha=sha256(supplement.manifest)
    loaders = {}
    for split in ("train", "val"):
        part=WasteDataset(dataset,split,config,rows)
        if split=="train" and supplement is not None:part=ConcatDataset([part,supplement])
        loaders[split] = DataLoader(part, batch_size=config["batch_size"],
            shuffle=split == "train", num_workers=config["num_workers"], pin_memory=device.type == "cuda",
            persistent_workers=config["num_workers"] > 0, worker_init_fn=seed_worker, generator=generator)
    model = CompactCNN(config["image_size"], config["dropout"], config.get("channels",(24,48,96,160)),config.get("pooling","avg")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    warmup = config["warmup_epochs"]
    def schedule(index):
        if index < warmup:
            return (index+1)/warmup
        return .01 + .99*(1+math.cos(math.pi*(index-warmup)/max(1, config["epochs"]-warmup)))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = Counter(row["class"] for row in rows if row["split"] == "train")
    if supplement is not None:counts.update(supplement.counts)
    weights = torch.tensor([(sum(counts.values())/(6*counts[label]))**.5 for label in CLASSES], device=device)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights if config.get("balanced_loss") else None, label_smoothing=config["label_smoothing"])
    validation_criterion = nn.CrossEntropyLoss()
    history, start, best_f1, best_epoch, bad_epochs = [], 1, -1., 0, 0
    if resume:
        saved = torch.load(output/"last_training.pth", map_location="cpu", weights_only=True)
        if saved["manifest_sha256"] != summary["manifest_sha256"] or saved["config"] != config or saved.get("extra_manifest_sha256")!=extra_manifest_sha:
            raise ValueError("La reanudación requiere la misma configuración y particiones.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"])
        numpy_state = saved["numpy_rng"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        history, best_f1, best_epoch, bad_epochs = saved["history"], saved["best_f1"], saved["best_epoch"], saved["bad_epochs"]
        start = saved["epoch"]+1
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    write_json(output/"run_metadata.json", {"architecture": ARCHITECTURE, "parameters": sum(p.numel() for p in model.parameters()),
        "initialization": "random_from_scratch", "dataset_manifest_sha256": summary["manifest_sha256"],
        "train_images": len(loaders["train"].dataset), "val_images": len(loaders["val"].dataset),
        "extra_train_manifest_sha256":extra_manifest_sha,"training_class_counts":dict(counts),
        "test_used": False, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    print(json.dumps({"device":str(device),"parameters":sum(p.numel() for p in model.parameters()),"start_epoch":start}), flush=True)
    for number in range(start, config["epochs"]+1):
        started = time.perf_counter()
        train_values, _, _, _ = epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_values, _, _, _ = epoch(model, loaders["val"], validation_criterion, device)
        row = {"epoch": number, **{"train_"+key:value for key,value in train_values.items()},
               **{"val_"+key:value for key,value in val_values.items()}, "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter()-started}
        history.append(row)
        if val_values["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_values["macro_f1"], number, 0
            checkpoint = {"format_version": 2, "architecture": ARCHITECTURE, "classes": list(CLASSES),
                "image_size": config["image_size"], "config": config, "model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "epoch": number, "val_macro_f1": best_f1, "val_accuracy": val_values["accuracy"], "manifest_sha256": summary["manifest_sha256"],
                "extra_train_manifest_sha256":extra_manifest_sha}
            atomic_save(output/"best_model.pth", checkpoint)
        else:
            bad_epochs += 1
        scheduler.step()
        state = np.random.get_state()
        atomic_save(output/"last_training.pth", {"config":config,"model_state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
            "epoch":number,"history":history,"best_f1":best_f1,"best_epoch":best_epoch,"bad_epochs":bad_epochs,
            "manifest_sha256":summary["manifest_sha256"],"extra_manifest_sha256":extra_manifest_sha,"torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
            "cuda_rng":torch.cuda.get_rng_state_all() if device.type == "cuda" else [],"loader_rng":generator.get_state(),
            "numpy_rng":(state[0],state[1].tolist(),*state[2:])})
        write_json(output/"history.json", history)
        with (output/"history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        write_json(output/"status.json", {"state":"running","epoch":number,"best_epoch":best_epoch,"best_macro_f1":best_f1,**val_values})
        print(json.dumps(row), flush=True)
        if bad_epochs >= config["patience"] and number >= 30:
            break
    best = torch.load(output/"best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    values, truth, predicted, probabilities = epoch(model, loaders["val"], validation_criterion, device)
    metrics = {**final_metrics(truth,predicted),"loss":values["loss"],"best_epoch":best_epoch,"epochs_completed":len(history),
               "images":len(truth),"split":"val","test_used":False,"initialization":"random_from_scratch"}
    metrics["per_class"]={name:metrics["per_class"][name] for name in CLASSES}
    write_json(output/"metrics.json", metrics)
    write_json(output/"validation_predictions.json", [{"path":r["path"],"actual":CLASSES[t],"predicted":CLASSES[p],"probabilities":prob}
        for r,t,p,prob in zip(loaders["val"].dataset.rows,truth,predicted,probabilities)])
    plot_history(history, output/"training_curves.png")
    write_json(output/"status.json", {"state":"complete",**metrics})
    print("COMPLETE " + json.dumps(metrics), flush=True)
    return output
