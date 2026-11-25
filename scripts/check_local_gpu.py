"""Una actualización CUDA con datos sintéticos por arquitectura; no mide precisión."""
import argparse
import json

import torch

from training.compact import CompactCNN
from training.detector_own import VisionLocaliza, detection_loss
from training.model import VisionCNN
from utils.quality_route import write_new_json


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en este Python. Revisa la instalación de PyTorch.")
    torch.manual_seed(42)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    checks = []
    for name, model, size in (("VisionCNN-v1", VisionCNN(), 128),
                              ("VisionResidual-v2", CompactCNN(160, .25, (20, 36, 64, 112), "avgmax"), 160),
                              ("VisionLocaliza-D2", VisionLocaliza(), 512)):
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
        before = next(model.parameters()).detach().clone()
        images = torch.rand(2, 3, size, size, device=device)
        outputs = model(images)
        if name == "VisionLocaliza-D2":
            boxes = [torch.tensor([[80., 60., 220., 260.]], device=device) for _ in range(2)]
            loss, _ = detection_loss(outputs, boxes)
        else:
            loss = torch.nn.functional.cross_entropy(outputs, torch.tensor([0, 1], device=device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Pérdida no finita: {name}")
        loss.backward()
        if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"Gradientes no finitos: {name}")
        optimizer.step()
        torch.cuda.synchronize()
        if torch.equal(before, next(model.parameters()).detach()):
            raise RuntimeError(f"Los pesos no se actualizaron: {name}")
        checks.append({"architecture": name, "loss_finite": True, "weights_updated": True})
        del model, optimizer, outputs, images, loss, before
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "checks": checks, "training_data_used": False,
            "note": "Solo comprueba cálculo y actualización con datos sintéticos; no evalúa modelos entrenados."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output:
        from pathlib import Path
        if Path(args.output).exists():
            parser.error("El informe ya existe; usa un nombre nuevo.")
    report = check_gpu()
    if args.output:
        write_new_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
