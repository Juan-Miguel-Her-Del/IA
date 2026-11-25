"""Importa fotos reales con etiquetas asignadas por una persona; no altera datos oficiales."""
import argparse
import csv
import json
from pathlib import Path

from app.field_data import FIELD_ROOT, pixel_hash, save_sample
from app.localization import Localizer
from app.resources import reserved_image_hashes
from app.settings import checkpoint_signature, load_settings
from app.vision import InferenceService, decode_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--labels", required=True, help="CSV: filename,true_class; para manual también left,top,right,bottom normalizados.")
    parser.add_argument("--mode", choices=("full", "auto", "manual"), default="full")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    service = InferenceService(checkpoint_signature(args.checkpoint or load_settings()["checkpoint"])[0], 2)
    existing = {}
    for path in (FIELD_ROOT / "samples").glob("*/record.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        existing[saved["original_pixel_sha256"]] = saved["true_class"]
    with Path(args.labels).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        photo = (directory / row["filename"]).resolve()
        if not photo.is_relative_to(directory):
            raise ValueError("La foto debe estar dentro de la carpeta indicada.")
        rgb = decode_image(photo.read_bytes(), reserved_hashes=reserved_image_hashes())
        if pixel_hash(rgb) in existing:
            if existing[pixel_hash(rgb)] != row["true_class"]:
                raise ValueError("La etiqueta indicada difiere del registro existente: " + photo.name)
            print("Ya registrada, se conserva: " + photo.name)
            continue
        bounds = tuple(float(row[key]) for key in ("left", "top", "right", "bottom")) if args.mode == "manual" else (.2, .2, .8, .8)
        result = Localizer(args.mode, bounds).locate(rgb)
        reading = service.classify(rgb, region=result.region) if result.region else None
        folder = save_sample(rgb, reading, row["true_class"], service, "Subir imagen", message=result.message + " Archivo: " + photo.name)
        existing[pixel_hash(rgb)] = row["true_class"]
        print(f"{photo.name}: real={row['true_class']}; predicción={reading.material if reading else 'sin localización'}; {folder.name}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
