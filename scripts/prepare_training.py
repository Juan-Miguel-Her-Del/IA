"""Un comando prepara A/B y publica el paquete en la carpeta Drive configurada."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.config import PROJECT_ROOT
from utils.drive_sync import check_preserved_A, load_sync_settings, publish_package
from utils.experiment_pipeline import build_experiment, verify_experiment
from utils.experiments import load_experiment, source_records
from utils.colab_package import export_package


def prepare(experiment=None, local_only=False, rebuild_data=False):
    settings = load_sync_settings(require_folder=not local_only)
    name = experiment or settings["selected_experiment"]
    if name not in ("A", "B"):
        raise ValueError("Elija A o B.")
    if name == "A" and rebuild_data:
        raise ValueError("Las particiones A se conservan; no se reconstruyen en este flujo.")
    check_preserved_A()
    _, config, selected = load_experiment(name)
    data = PROJECT_ROOT / "data"
    try:
        source_records(data, config, selected)
    except ValueError:
        subprocess.run([sys.executable, "-m", "scripts.prepare_realwaste", "--experiment", name,
                        "--transport", "authors", "--data-dir", str(data)], cwd=PROJECT_ROOT, check=True)
    if not (data / "experiments" / name / "processed").exists() or rebuild_data:
        build_experiment(data, name, config, selected, rebuild=rebuild_data)
    else:
        validation = verify_experiment(data, name, config, selected)
        if not validation["valid"]:
            raise ValueError("Datos modificados: " + "; ".join(validation["errors"][:3]) + ". Para B use --rebuild-data.")
    destination = PROJECT_ROOT / "exports" / "sync" / name / f"Vision-IA_Colab_{name}.zip"
    package, result = export_package(destination, name, drive_workflow=True)
    print(f"Paquete {name} verificado: {result['image_count']} imagenes, sin pesos.", flush=True)
    if local_only:
        publication = {"published": False, "local_package": str(package), "reason": "Preparacion local solicitada."}
    else:
        publication = {"published": True, **publish_package(package, name, settings)}
    check_preserved_A()
    print(json.dumps(publication, ensure_ascii=False, indent=2))
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"))
    parser.add_argument("--local-only", action="store_true", help="Solo exports/, sin asumir una carpeta sincronizada.")
    parser.add_argument("--rebuild-data", action="store_true", help="Reconstruye B explicitamente, nunca A.")
    parser.add_argument("--watch", action="store_true", help="Tras publicar, observar resultados hasta Ctrl+C.")
    args = parser.parse_args()
    if args.local_only and args.watch:
        parser.error("--watch necesita una carpeta sincronizada; no combinar con --local-only.")
    prepare(args.experiment, args.local_only, args.rebuild_data)
    if args.watch:
        subprocess.run([sys.executable, "-m", "scripts.sync_results", "--watch"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Observacion detenida; paquete y resultados conservados.")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
