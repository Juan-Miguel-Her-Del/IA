"""Protocolo de archivos completos: sin servicios, autenticacion o evaluacion test."""
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from utils.dataset import sha256, write_json
from utils.config import CLASSES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_child(root, relative):
    root = Path(root).resolve()
    parts = PurePosixPath(relative)
    path = (root / relative).resolve()
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or not path.is_relative_to(root) or path == root:
        raise ValueError(f"Ruta relativa no permitida: {relative}")
    return path


def finish_run(output, experiment, run_id, package_id, status, error=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_state.json", {"experiment": experiment, "run_id": run_id, "package_id": package_id,
               "status": status, "finished_utc": utc_now(), "error": error, "test_evaluated": False})
    if status == "completed":
        for name in ("best_model.pth", "history.json", "history.csv", "training_summary.json", "config.json",
                     "class_to_idx.json", "training_curves.png", "run_metadata.json"):
            if not (output / name).is_file():
                raise ValueError(f"No se marca la ejecucion completa: falta {name}.")
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("FINISHED.json", "results_manifest.json") and not path.name.endswith(".tmp"):
            entries.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(output / "results_manifest.json", {"experiment": experiment, "run_id": run_id, "files": entries})
    # Se escribe al final; el receptor verifica hashes aunque Drive sincronice fuera de orden.
    write_json(output / "FINISHED.json", {"experiment": experiment, "run_id": run_id, "status": status,
               "manifest_sha256": sha256(output / "results_manifest.json"), "finished_utc": utc_now()})


def validate_finished_run(output):
    output = Path(output)
    marker = json.loads((output / "FINISHED.json").read_text(encoding="utf-8"))
    if marker.get("experiment") not in ("A", "B") or marker.get("status") not in ("completed", "failed"):
        raise ValueError("Marcador de resultados invalido.")
    manifest = output / "results_manifest.json"
    if sha256(manifest) != marker["manifest_sha256"]:
        raise ValueError("El manifiesto todavia no esta sincronizado o fue alterado.")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["experiment"] != marker["experiment"] or data["run_id"] != marker["run_id"]:
        raise ValueError("Resultados de ejecuciones distintas.")
    names = set()
    for item in data["files"]:
        path = safe_child(output, item["path"])
        if item["path"] in names or not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Archivo incompleto, duplicado o alterado: {item['path']}")
        names.add(item["path"])
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    if state["experiment"] != marker["experiment"] or state["run_id"] != marker["run_id"] or state["status"] != marker["status"]:
        raise ValueError("Estado de ejecucion inconsistente.")
    if marker["status"] == "completed":
        required = {"best_model.pth", "history.json", "history.csv", "training_summary.json", "config.json",
                    "class_to_idx.json", "training_curves.png", "run_metadata.json", "run_state.json"}
        if not required.issubset(names):
            raise ValueError("Faltan resultados obligatorios.")
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        if config["experiment"] != marker["experiment"] or metadata.get("test_used_during_training") is not False:
            raise ValueError("Experimento incorrecto o test no reservado.")
        if metadata.get("initialization") != "random_from_scratch" or metadata.get("architecture") != "VisionCNN-v1" or metadata.get("parameters") != 89686:
            raise ValueError("Los resultados no corresponden a la CNN propia inicializada desde cero.")
        mapping = json.loads((output / "class_to_idx.json").read_text(encoding="utf-8"))
        if mapping != {label: index for index, label in enumerate(CLASSES)}:
            raise ValueError("Orden de categorias distinto del proyecto.")
    return marker, data
