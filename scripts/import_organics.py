"""Incorpora imagenes organicas elegidas y revisadas por el usuario."""
import argparse
import re
from pathlib import Path

from utils.config import add_data_argument, data_root
from utils.dataset import image_paths, record_import, store_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Solo imagenes de residuos organicos.")
    parser.add_argument("--source-name", required=True, help="Identificador: letras minusculas, numeros, _ o -.")
    parser.add_argument("--source-url", required=True, help="URL real del dataset o referencia de captura propia.")
    parser.add_argument("--license", required=True, help="Licencia o permiso real de uso.")
    add_data_argument(parser)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.source_name):
        parser.error("--source-name debe tener entre 1 y 64 caracteres: a-z, 0-9, _ y -.")
    if not args.source_url.strip() or not args.license.strip():
        parser.error("La fuente y la licencia no pueden estar vacias.")
    source = args.input_dir.expanduser().resolve()
    root = data_root(args.data_dir)
    destination = root / "raw" / "organicos" / args.source_name
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("La carpeta de origen debe ser independiente de la carpeta de destino.")
    paths = image_paths(source)
    if not paths:
        parser.error("No se encontraron imagenes en --input-dir.")
    for path in paths:
        store_file(path, destination)
    record_import(root, {
        "dataset": args.source_name, "class": "organicos", "source_url": args.source_url,
        "license": args.license, "local_source": str(source), "images_read": len(paths),
        "review_required": "Confirmar manualmente que cada imagen representa residuos organicos.",
    })
    print(f"Organicos: {len(paths)} imagenes leidas. Ejecute audit_dataset y split_dataset de nuevo.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
