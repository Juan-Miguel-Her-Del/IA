"""Importa una ejecución C terminada desde Drive, comprobando hashes y sin sobrescribir."""
import argparse
import json
from pathlib import Path

from training.runtime import load_checkpoint
from utils.config import PROJECT_ROOT
from utils.dataset import sha256
from utils.drive_sync import copy_verified, load_sync_settings
from utils.run_files import safe_child


def import_run(name):
    if not name.startswith("C") or Path(name).name != name or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Nombre de ejecución C inválido.")
    drive=Path(load_sync_settings()["local_shared_dir"])
    source=drive/"results/C"/name/"selected"
    ready=json.loads((source/"READY.json").read_text(encoding="utf-8"))
    if not {"best_model.pth","metrics.json","model_config.json","selection.json","history.json","run_metadata.json"}.issubset(ready["files"]):
        raise ValueError("Publicación incompleta.")
    for relative,digest in ready["files"].items():
        if sha256(safe_child(source,relative))!=digest:
            raise ValueError("Drive aún no terminó de sincronizar: "+relative)
    _,metadata=load_checkpoint(source/"best_model.pth")
    metrics=json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    if metadata["architecture"]!="VisionResidual-v2" or metrics["split"]!="val" or metrics["images"]!=884 or metrics["test_used"]:
        raise ValueError("No es una evaluación C sobre las 884 imágenes val B.")
    if metrics["checkpoint_sha256"]!=ready["files"]["best_model.pth"]:
        raise ValueError("El checkpoint no coincide con las métricas.")
    destination=PROJECT_ROOT/"runs/synced/C"/name
    for relative in ready["files"]:
        copy_verified(safe_child(source,relative),safe_child(destination,relative))
    copy_verified(source/"READY.json",destination/"READY.json")
    print(destination)
    print({k:metrics[k] for k in ("accuracy","macro_f1","images")})
    return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    args=parser.parse_args()
    import_run(args.run_name)
