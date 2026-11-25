"""Preparacion/publicacion y lectura de resultados mediante una carpeta Drive local."""
import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from utils.config import PROJECT_ROOT
from utils.dataset import sha256, write_json
from utils.experiments import write_csv
from utils.run_files import safe_child, utc_now, validate_finished_run

DEFAULT_SETTINGS = PROJECT_ROOT / "config/sync.json"
LOCAL_SETTINGS = PROJECT_ROOT / "config/sync.local.json"
RUN_ID = re.compile(r"[AB]_[A-Za-z0-9_-]{1,100}\Z")


def load_sync_settings(require_folder=True, settings_path=None, local_path=None):
    settings = json.loads(Path(settings_path or DEFAULT_SETTINGS).read_text(encoding="utf-8"))
    local = Path(local_path or LOCAL_SETTINGS)
    if local.exists():
        private = json.loads(local.read_text(encoding="utf-8"))
        # La seleccion y parametros viven en sync.json; el archivo privado guarda solo rutas.
        settings.update({key: private[key] for key in ("local_shared_dir", "drive_relative_root") if key in private})
    if settings["selected_experiment"] not in ("A", "B"):
        raise ValueError("selected_experiment debe ser A o B.")
    relative = settings["drive_relative_root"]
    if not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts or "\\" in relative or ":" in relative:
        raise ValueError("drive_relative_root debe ser una ruta relativa dentro de Mi unidad, con /.")
    allowed = {"epochs", "batch_size", "learning_rate", "num_workers"}
    overrides = settings.get("training_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Los overrides solo permiten epocas, batch, learning rate y workers.")
    for key in ("epochs", "batch_size", "num_workers"):
        if key in overrides and (type(overrides[key]) is not int or overrides[key] < (0 if key == "num_workers" else 1)):
            raise ValueError(f"Override entero invalido: {key}")
    if "learning_rate" in overrides and (not isinstance(overrides["learning_rate"], (int, float)) or not math.isfinite(overrides["learning_rate"]) or overrides["learning_rate"] <= 0):
        raise ValueError("learning_rate debe ser positivo y finito.")
    folder = settings.get("local_shared_dir")
    if require_folder and (not folder or not Path(folder).expanduser().is_dir()):
        raise ValueError("Falta una carpeta sincronizada real. Configure Google Drive para escritorio y ejecute scripts.configure_sync --folder RUTA --drive-relative-root Vision-IA. Para solo preparar localmente, use --local-only.")
    return settings


def configure_folder(folder, relative="Vision-IA", experiment="B", local_path=None):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("La carpeta no existe. Cree/seleccione primero una carpeta real de Mi unidad en Drive para escritorio.")
    target = Path(local_path or LOCAL_SETTINGS)
    configured = {"local_shared_dir": str(folder), "drive_relative_root": relative}
    # Validacion antes de persistir: no inventar una ruta si no esta montada.
    if experiment not in ("A", "B") or not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts or "\\" in relative or ":" in relative:
        raise ValueError("Experimento o ruta relativa de Drive invalida.")
    write_json(target, configured)
    return {**configured, "selected_experiment": experiment}


def check_preserved_A():
    index = PROJECT_ROOT / "data/automation/preserved_A.json"
    if not index.exists():
        files = [p for p in (PROJECT_ROOT / "data/experiments/A").rglob("*") if p.is_file()]
        for folder in (PROJECT_ROOT / "runs").glob("A*"):
            if folder.is_dir():
                files += [p for p in folder.rglob("*") if p.is_file()]
        files += [PROJECT_ROOT / "models/best_model.pth", PROJECT_ROOT / "training/model.py", PROJECT_ROOT / "config/train_A.json"]
        write_json(index, {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files if p.is_file()})
    expected = json.loads(index.read_text(encoding="utf-8"))
    changed = [relative for relative, digest in expected.items()
               if not (PROJECT_ROOT / relative).is_file() or sha256(PROJECT_ROOT / relative) != digest]
    if changed:
        raise ValueError("Se alteraron archivos de A: " + ", ".join(changed[:5]))
    return len(expected)


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    digest = sha256(source)
    if target.exists():
        if sha256(target) == digest:
            return
        raise ValueError(f"No se sobrescribe una version existente distinta: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise ValueError("La copia no coincide con el archivo preparado.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_package(package, experiment, settings):
    from utils.colab_package import verify_package
    package = Path(package)
    verified = verify_package(package)
    if verified["experiment"] != experiment or verified["workflow"] != "drive":
        raise ValueError("El paquete no coincide con el flujo/experimento solicitado.")
    folder = Path(settings["local_shared_dir"]).resolve()
    if not folder.is_dir():
        raise ValueError("La carpeta de Drive ya no esta disponible.")
    digest = sha256(package)
    package_id = digest[:20]
    relative = f"packages/{experiment}/{package_id}/{package.name}"
    target = safe_child(folder, relative)
    copy_verified(package, target)
    info = {"experiment": experiment, "package_id": package_id, "package_relative_path": relative,
            "package_sha256": digest, "bytes": package.stat().st_size,
            "image_count": verified["image_count"], "prepared_utc": utc_now()}
    write_json(target.parent / "package_ready.json", info)
    notebook = PROJECT_ROOT / "notebooks/train_drive.ipynb"
    notebook_target = folder / "notebooks/train_drive.ipynb"
    notebook_target.parent.mkdir(parents=True, exist_ok=True)
    if not notebook_target.exists() or sha256(notebook_target) != sha256(notebook):
        temporary = notebook_target.with_suffix(".ipynb.part")
        shutil.copyfile(notebook, temporary)
        temporary.replace(notebook_target)
    control_path = folder / "control.json"
    control = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {"schema_version": 1, "packages": {}}
    control["packages"][experiment] = info
    control.update(selected_experiment=experiment, drive_relative_root=settings["drive_relative_root"],
                   training_overrides=settings.get("training_overrides", {}), updated_utc=utc_now())
    write_json(control_path, control)  # Ultimo; Colab tambien valida tamano y hash del ZIP.
    (folder / "results").mkdir(exist_ok=True)
    return {**info, "local_target": str(target), "control": str(control_path),
            "notebook": str(notebook_target), "cloud_upload_confirmed": False}


def import_finished_run(source, destination_root):
    source, destination_root = Path(source), Path(destination_root).resolve()
    marker, manifest = validate_finished_run(source)
    experiment, run_id = marker["experiment"], marker["run_id"]
    if not RUN_ID.fullmatch(run_id) or not run_id.startswith(experiment + "_"):
        raise ValueError("Identificador de ejecucion no valido.")
    if source.name != run_id or source.parent.name != experiment:
        raise ValueError("La carpeta compartida y los metadatos de experimento/ejecucion no coinciden.")
    target = destination_root / experiment / run_id
    if target.parent.is_symlink() or not target.resolve().is_relative_to(destination_root):
        raise ValueError("El destino resuelto sale de la carpeta local de resultados.")
    if target.exists():
        old, _ = validate_finished_run(target)
        if old["manifest_sha256"] != marker["manifest_sha256"]:
            raise ValueError("Ya existe otra version local de esta ejecucion; no se sobrescribe.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sync-", dir=target.parent) as tmp:
        staging = Path(tmp) / run_id
        staging.mkdir()
        names = [entry["path"] for entry in manifest["files"]] + ["results_manifest.json", "FINISHED.json"]
        for relative in names:
            origin, destination = safe_child(source, relative), safe_child(staging, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        validate_finished_run(staging)
        staging.rename(target)
    return target


def summarize_local_runs(destination_root):
    destination_root = Path(destination_root)
    rows = []
    for experiment in ("A", "B"):
        for path in sorted((destination_root / experiment).glob("*/FINISHED.json")):
            folder = path.parent
            try:
                marker, _ = validate_finished_run(folder)
                row = {"experiment": experiment, "run_id": marker["run_id"], "status": marker["status"],
                       "epochs": "", "best_epoch": "", "best_val_macro_f1": "", "best_val_accuracy": "",
                       "stopped_early": "", "checkpoint": str(folder / "best_model.pth") if marker["status"] == "completed" else ""}
                if marker["status"] == "completed":
                    summary = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
                    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
                    if len(history) != summary["epochs_completed"] or summary.get("test_used_during_training") is not False:
                        raise ValueError("Historial/resumen incompleto o test no reservado.")
                    best = next(h for h in history if h["epoch"] == summary["best_epoch"])
                    f1, accuracy = float(summary["best_val_macro_f1"]), float(best["val_accuracy"])
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (f1, accuracy)):
                        raise ValueError("Metricas de validacion invalidas.")
                    row.update(epochs=summary["epochs_completed"], best_epoch=summary["best_epoch"],
                               best_val_macro_f1=f1, best_val_accuracy=accuracy, stopped_early=summary["stopped_early"])
                rows.append(row)
            except (OSError, ValueError, KeyError, StopIteration):
                continue
    destination_root.mkdir(parents=True, exist_ok=True)
    fields = ("experiment", "run_id", "status", "epochs", "best_epoch", "best_val_macro_f1", "best_val_accuracy", "stopped_early", "checkpoint")
    write_csv(destination_root / "summary.csv", rows, fields)
    lines = ["# Entrenamientos sincronizados", "", "Solo metricas de validacion. Test no evaluado.", "",
             "A y B tienen pruebas/validaciones distintas: no interpretar esta tabla como una comparacion independiente concluyente.", "",
             "| Experimento | Ejecucion | Estado | Epocas | Mejor epoca | Macro F1 val | Accuracy val |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in ("experiment", "run_id", "status", "epochs", "best_epoch", "best_val_macro_f1", "best_val_accuracy")) + " |")
    if not rows:
        lines += ["", "Aun no hay entrenamientos completos sincronizados. No se inventan resultados de B."]
    (destination_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def sync_results(settings, destination_root=None):
    shared = Path(settings["local_shared_dir"]).resolve()
    if not shared.is_dir():
        raise ValueError("La carpeta Drive no esta disponible; abra Drive para escritorio.")
    destination = Path(destination_root or PROJECT_ROOT / "runs/synced").resolve()
    imported, pending = [], []
    for experiment in ("A", "B"):
        for source in sorted((shared / "results" / experiment).glob("*")):
            if not source.is_dir() or not RUN_ID.fullmatch(source.name):
                continue
            if not (source / "FINISHED.json").exists():
                pending.append({"run_id": source.name, "reason": "Todavia en ejecucion o sincronizacion."})
                continue
            try:
                target = import_finished_run(source, destination)
                imported.append(str(target))
            except (OSError, ValueError, KeyError) as exc:
                pending.append({"run_id": source.name, "reason": str(exc)})
    rows = summarize_local_runs(destination)
    result = {"imported": imported, "pending": pending, "runs": rows, "summary": str(destination / "SUMMARY.md"),
              "test_evaluated": False}
    write_json(destination / "sync_status.json", result)
    return result
