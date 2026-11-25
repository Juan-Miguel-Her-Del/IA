"""Publica código y suplemento C en Drive; reutiliza B sin copiar ni extraer test."""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from training.data import resolve_path
from training.supplement import TrainingSupplement
from utils.config import PROJECT_ROOT
from utils.dataset import sha256, write_json
from utils.drive_sync import copy_verified, load_sync_settings


def export(config_path):
    path=resolve_path(config_path)
    config=json.loads(path.read_text(encoding="utf-8"))
    if config["architecture"]!="VisionResidual-v2" or config["initialization"]!="random_from_scratch":
        raise ValueError("Se exige la CNN propia con pesos aleatorios.")
    name=config["experiment"]
    if not name.startswith("C") or not name.isalnum():raise ValueError("Experimento C inválido.")
    shared=Path(load_sync_settings()["local_shared_dir"])
    control=json.loads((shared/"control.json").read_text(encoding="utf-8"))
    baseline=control["packages"]["B"]
    if not (shared/baseline["package_relative_path"]).is_file():raise ValueError("Falta el paquete B original en Drive.")
    files=[PROJECT_ROOT/folder/file for folder,names in {
        "training":["__init__.py","compact.py","compact_training.py","supplement.py","data.py","model.py","runtime.py","prediction.py","metrics.py"],
        "utils":["__init__.py","config.py","dataset.py"],
        "scripts":["__init__.py","train_compact.py","evaluate_compact.py","finalize_compact.py"]}.items() for file in names]
    template=PROJECT_ROOT/"notebooks/train_compact_drive.ipynb"
    files += [path,template]
    if config.get("extra_train_manifest"):
        supplement=TrainingSupplement(resolve_path(config["extra_train_manifest"]),config)
        files += [supplement.manifest,supplement.root/"provenance.json"]
        files += [supplement.root/row["path"] for row in supplement.rows]
    # Crear primero en disco local: Drive puede interrumpir un ZIP escrito directamente.
    with tempfile.TemporaryDirectory() as temporary:
        package=Path(temporary)/"training.zip"
        with zipfile.ZipFile(package,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for file in sorted(files):archive.write(file,file.resolve().relative_to(PROJECT_ROOT).as_posix())
        with zipfile.ZipFile(package) as archive:
            if archive.testzip():raise ValueError("Paquete corrupto.")
        digest=sha256(package)
        relative=f"automation/{name}/packages/{digest[:20]}.zip"
        copy_verified(package,shared/relative)
        notebook_relative=f"automation/{name}/notebooks/{digest[:20]}.ipynb"
        notebook=json.loads(template.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            cell["source"]=[line.replace("EXPERIMENT='C2'",f"EXPERIMENT='{name}'").replace("/content/vision_c2","/content/vision_compact") for line in cell["source"]]
        local_notebook=Path(temporary)/"training.ipynb"
        write_json(local_notebook,notebook)
        copy_verified(local_notebook,shared/notebook_relative)
    manifest={"experiment":name,"config":path.relative_to(PROJECT_ROOT).as_posix(),"package":relative,"notebook":notebook_relative,
              "sha256":digest,"base_package":baseline,"initialization":"random_from_scratch"}
    write_json(shared/f"automation/{name}/package_ready.json",manifest)
    print(shared/notebook_relative)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="config/train_C.json")
    export(parser.parse_args().config)
