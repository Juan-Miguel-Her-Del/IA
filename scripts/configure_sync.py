"""Configura una carpeta Drive ya existente; no instala ni autentica servicios."""
import argparse
import json

from utils.drive_sync import DEFAULT_SETTINGS, configure_folder
from utils.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Carpeta local real dentro de Mi unidad.")
    parser.add_argument("--drive-relative-root", default="Vision-IA", help="La misma carpeta, relativa a Mi unidad en Colab.")
    parser.add_argument("--experiment", choices=("A", "B"), default="B")
    args = parser.parse_args()
    print(json.dumps(configure_folder(args.folder, args.drive_relative_root, args.experiment), indent=2))
    settings = json.loads(DEFAULT_SETTINGS.read_text(encoding="utf-8"))
    settings["selected_experiment"] = args.experiment
    write_json(DEFAULT_SETTINGS, settings)
    print("Ruta configurada localmente. Drive para escritorio gestiona la sincronizacion; no se verifico la nube ni se inicio sesion automaticamente.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
