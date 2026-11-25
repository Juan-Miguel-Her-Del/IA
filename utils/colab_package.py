"""Empaquetado explicito y verificacion sin entrenar, acceder a Colab o duplicar datos."""
import ast
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from training.data import read_manifest, verify_files
from utils.config import CLASSES, PROJECT_ROOT, SPLITS
from utils.dataset import sha256, write_json

PACKAGE_NAME = "Vision-IA_Colab_A.zip"
ROOT_NAME = "Vision-IA"
CODE_FILES = (
    "training/__init__.py", "training/model.py", "training/data.py", "training/runtime.py",
    "training/metrics.py", "training/engine.py", "scripts/__init__.py", "scripts/train.py",
    "scripts/evaluate.py", "scripts/inspect_training_data.py", "utils/__init__.py", "utils/config.py",
    "utils/dataset.py", "utils/experiments.py", "requirements-data.txt", "requirements-training.txt",
    "config/train_A.json", "notebooks/train_colab.ipynb", "README.md", "docs/TRAINING.md",
    "docs/DATASETS.md", "docs/TRASHNET_LICENSE.txt",
    "training/prediction.py", "training/validation.py", "utils/dataset_samples.py",
    "scripts/evaluate_validation.py", "scripts/predict_image.py", "docs/VALIDATION.md",
)
DRIVE_FILES = ("notebooks/train_drive.ipynb", "scripts/run_colab.py", "utils/run_files.py", "docs/SYNC_TRAINING.md")


def package_files(experiment="A", workflow="legacy"):
    if experiment not in ("A", "B") or workflow not in ("legacy", "drive"):
        raise ValueError("Experimento o flujo de paquete invalido.")
    files = [name.replace("config/train_A.json", f"config/train_{experiment}.json") for name in CODE_FILES]
    if workflow == "drive":
        files.remove("notebooks/train_colab.ipynb")
        files += list(DRIVE_FILES)
    return tuple(files)


def validate_drive_notebook(notebook):
    if notebook.get("nbformat") != 4:
        raise ValueError("Formato de notebook invalido.")
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if not cell.get("id"):
            raise ValueError("Falta identificador de celda.")
        source = "".join(cell["source"])
        sources.append(source)
        if cell["cell_type"] == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise ValueError("El notebook Drive debe entregarse sin ejecutar.")
            ast.parse(source, filename=f"drive_cell_{index}")
    combined = "\n".join(sources)
    for required in ("AUTORIZAR_MONTAJE = False", "control.json", "selected_experiment", "package_sha256",
                     "PACKAGE_MANIFEST.json", "scripts.inspect_training_data", "scripts.run_colab", "--results-root", "'cuda'"):
        if required not in combined:
            raise ValueError(f"Falta instruccion Drive: {required}")
    return len(notebook["cells"])


def validate_notebook(notebook):
    if notebook.get("nbformat") != 4:
        raise ValueError("Formato de notebook invalido.")
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        sources.append(source)
        if not cell.get("id"):
            raise ValueError("Falta identificador de celda.")
        if cell["cell_type"] == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise ValueError("El notebook exportado debe estar sin ejecutar.")
            ast.parse(source, filename=f"notebook_cell_{index}")
    combined = "\n".join(sources)
    for required in ("AUTORIZAR_MONTAJE = False", PACKAGE_NAME, "PACKAGE_MANIFEST.json", "PROJECT_DIR",
                     "scripts.inspect_training_data", "scripts.train", "RESULTS_DIR", "--device", "'cuda'", "files.download"):
        if required not in combined:
            raise ValueError(f"El notebook no contiene la instruccion necesaria: {required}")
    return len(notebook["cells"])


def verify_package(path):
    with zipfile.ZipFile(path) as bundle:
        index = json.loads(bundle.read(ROOT_NAME + "/PACKAGE_MANIFEST.json"))
        experiment, workflow = index.get("experiment", "A"), index.get("workflow", "legacy")
        code_files = package_files(experiment, workflow)
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Rutas repetidas dentro del ZIP.")
        for name in names:
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != ROOT_NAME or ".." in parts or "\\" in name or PurePosixPath(name).is_absolute():
                raise ValueError("Ruta insegura en el ZIP.")
            other_experiment = "B" if experiment == "A" else "A"
            if any(p in (".venv", "__pycache__", "raw", other_experiment, "runs", "checkpoints") for p in parts) or Path(name).suffix.lower() in (".pth", ".pt", ".ckpt", ".safetensors"):
                raise ValueError("Archivo excluido presente en el ZIP.")
        expected_names = {ROOT_NAME + "/" + item["path"] for item in index["files"]}
        if expected_names | {ROOT_NAME + "/PACKAGE_MANIFEST.json"} != set(names):
            raise ValueError("El inventario no coincide con el contenido del ZIP.")
        for item in index["files"]:
            digest, size = hashlib.sha256(), 0
            with bundle.open(ROOT_NAME + "/" + item["path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != item["sha256"] or size != item["bytes"]:
                raise ValueError(f"Contenido incompleto o alterado: {item['path']}")
        prefix = ROOT_NAME + f"/data/experiments/{experiment}/processed/"
        rows = list(csv.DictReader(io.StringIO(bundle.read(prefix + "manifest.csv").decode("utf-8"))))
        expected_images = {prefix + row["path"] for row in rows}
        actual_images = {name for name in names if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")}
        if expected_images != actual_images or len(rows) != index["image_count"]:
            raise ValueError("Las imagenes del ZIP no coinciden exactamente con el manifiesto.")
        if len({row["sha256"] for row in rows}) != len(rows):
            raise ValueError("Hay imagenes redundantes en el paquete.")
        items = {item["path"]: item for item in index["files"]}
        for row in rows:
            if items[f"data/experiments/{experiment}/processed/" + row["path"]]["sha256"] != row["sha256"]:
                raise ValueError("Hash de imagen distinto del manifiesto de datos.")
        for relative in code_files:
            if ROOT_NAME + "/" + relative not in names:
                raise ValueError(f"Falta codigo requerido: {relative}")
        config = json.loads(bundle.read(ROOT_NAME + f"/config/train_{experiment}.json"))
        if config["dataset_dir"] != f"data/experiments/{experiment}/processed" or config["experiment"] != experiment:
            raise ValueError("La configuracion no localiza los datos portatiles.")
        preparation_config = json.loads(bundle.read(ROOT_NAME + f"/config/experiment_{experiment}.json"))
        if set(preparation_config["experiments"]) != {experiment}:
            raise ValueError("El paquete mezcla configuraciones de experimentos.")
        expected_categories = ["Food Organics"] if experiment == "A" else ["Food Organics", "Cardboard", "Glass", "Metal", "Paper", "Plastic"]
        if preparation_config["experiments"][experiment]["realwaste_categories"] != expected_categories:
            raise ValueError("Las categorias del paquete no corresponden al experimento.")
        notebook_name = "train_drive.ipynb" if workflow == "drive" else "train_colab.ipynb"
        notebook = json.loads(bundle.read(ROOT_NAME + "/notebooks/" + notebook_name))
        cells = validate_drive_notebook(notebook) if workflow == "drive" else validate_notebook(notebook)
        for relative in code_files:
            if relative.endswith(".py"):
                ast.parse(bundle.read(ROOT_NAME + "/" + relative).decode("utf-8"), filename=relative)
        runtime = ast.parse(bundle.read(ROOT_NAME + "/training/runtime.py").decode("utf-8"))
        calls = [node for node in ast.walk(runtime) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "use_deterministic_algorithms"]
        if len(calls) != 1 or not any(kw.arg == "warn_only" and isinstance(kw.value, ast.Constant)
                                     and kw.value.value is True for kw in calls[0].keywords):
            raise ValueError("El ZIP conserva un runtime CUDA sin la correccion warn_only=True.")
        return {"valid": True, "experiment": experiment, "workflow": workflow, "files": len(names), "image_count": len(rows), "notebook_cells": cells,
                "split_counts": {split: sum(r["split"] == split for r in rows) for split in SPLITS}}


def export_package(output=None, experiment="A", drive_workflow=False):
    workflow = "drive" if drive_workflow else "legacy"
    if experiment == "B" and not drive_workflow:
        raise ValueError("B utiliza el notebook reutilizable; indique --drive-workflow.")
    code_files = package_files(experiment, workflow)
    name = f"Vision-IA_Colab_{experiment}.zip"
    output = Path(output).resolve() if output else PROJECT_ROOT / "exports" / name
    processed = PROJECT_ROOT / "data" / "experiments" / experiment / "processed"
    rows, _ = read_manifest(processed)
    verify_files(processed, rows, SPLITS)
    notebook_name = "train_drive.ipynb" if drive_workflow else "train_colab.ipynb"
    notebook = json.loads((PROJECT_ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8"))
    (validate_drive_notebook if drive_workflow else validate_notebook)(notebook)
    sources = {relative: PROJECT_ROOT / relative for relative in code_files}
    for row in rows:
        relative = f"data/experiments/{experiment}/processed/" + row["path"]
        sources[relative] = processed / row["path"]
    for relative in ("processed/manifest.csv", "processed/split_summary.json", "reports/provenance.csv",
                     "reports/audit.json", "reports/counts.csv", "reports/visual_candidates.csv",
                     "reports/REPORT.md", "reports/realwaste_import.json", "reports/validation.json"):
        path = PROJECT_ROOT / "data" / "experiments" / experiment / relative
        sources[f"data/experiments/{experiment}/" + relative] = path
    # Solo la configuracion seleccionada: no incluir datos ni pesos del otro experimento.
    original = json.loads((PROJECT_ROOT / "config/experiments.json").read_text(encoding="utf-8"))
    a_config = {**original, "active_experiment": experiment, "experiments": {experiment: original["experiments"][experiment]}}
    generated = {f"config/experiment_{experiment}.json": (json.dumps(a_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    # El ZIP es de entrenamiento: su README no promete scripts o muestras de fases anteriores no incluidos.
    sources.pop("README.md")
    generated["README.md"] = ("# Vision-IA: paquete de entrenamiento " + experiment + "\n\n"
        "Abre notebooks/train_colab.ipynb en Colab y sigue sus celdas manualmente.\n"
        "Sube este ZIP a Mi unidad/Vision-IA/Vision-IA_Colab_A.zip y autoriza Drive cuando quieras continuar.\n\n"
        "Incluye una CNN propia aleatoria, codigo train/evaluate, configuracion, 2791 imagenes A y metadatos.\n"
        "No contiene modelos entrenados, pesos externos, raw/, entornos, caches ni datos B.\n"
        "raw_path en los manifiestos es trazabilidad historica; la lectura usa processed/ y no requiere raw/.\n\n"
        "Documentacion: docs/TRAINING.md, docs/DATASETS.md y docs/TRASHNET_LICENSE.txt.\n"
        "Instalacion: python -m pip install -r requirements-training.txt\n"
        "Comprobacion: python -m scripts.inspect_training_data --verify-hashes\n"
        "Entrenamiento manual: python -m scripts.train --config config/train_A.json\n"
        "Analisis solo validacion: python -m scripts.evaluate_validation --checkpoint RUTA/best_model.pth --device cpu\n"
        "Fotografia nueva: python -m scripts.predict_image --checkpoint RUTA/best_model.pth --image RUTA/foto.jpg\n"
        "Test permanece reservado en fase 2.5. Consultar docs/VALIDATION.md. Runtime CUDA corregido con warn_only=True.\n"
    ).encode("utf-8")
    if drive_workflow:
        generated["README.md"] = (f"# Vision-IA: experimento {experiment}\n\n"
            "Usar notebooks/train_drive.ipynb y docs/SYNC_TRAINING.md. El notebook lee control.json de la carpeta Drive configurada.\n"
            "No requiere subir ni descargar ZIPs manualmente cuando Drive para escritorio esta configurado.\n"
            f"Incluye {len(rows)} imagenes del experimento {experiment}, codigo y metadatos. No contiene pesos ni el otro experimento.\n"
            "La CNN se inicializa desde cero; test permanece reservado. No iniciar entrenamiento automaticamente.\n"
            f"Instalacion: python -m pip install -r requirements-training.txt\nConfiguracion: config/train_{experiment}.json\n"
        ).encode("utf-8")
    entries = []
    for relative, source in sorted(sources.items()):
        if not source.is_file():
            raise ValueError(f"Falta un archivo necesario: {source}")
        entries.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256(source)})
    entries += [{"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                for relative, payload in generated.items()]
    index = {"schema_version": 1, "experiment": experiment, "workflow": workflow, "image_count": len(rows), "classes": list(CLASSES),
             "files": sorted(entries, key=lambda item: item["path"]),
             "note": "Solo copias particionadas; raw_path conserva la trazabilidad pero raw/ no se incluye. No contiene pesos."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for relative, source in sorted(sources.items()):
                compression = zipfile.ZIP_STORED if source.suffix.lower() in (".jpg", ".jpeg", ".png") else zipfile.ZIP_DEFLATED
                info = zipfile.ZipInfo(ROOT_NAME + "/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = compression
                with source.open("rb") as incoming, bundle.open(info, "w") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            for relative, payload in generated.items():
                info = zipfile.ZipInfo(ROOT_NAME + "/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, payload)
            info = zipfile.ZipInfo(ROOT_NAME + "/PACKAGE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, json.dumps(index, ensure_ascii=False, indent=2))
        result = verify_package(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(zip_bytes=output.stat().st_size, zip_sha256=sha256(output))
    write_json(output.with_suffix(".verification.json"), result)
    output.with_suffix(".zip.sha256").write_text(result["zip_sha256"] + "  " + output.name + "\n", encoding="ascii")
    return output, result
