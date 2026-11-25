"""Comprueba manifiesto y train/val sin entrenar ni abrir imagenes de prueba."""
import argparse
from collections import Counter

from training.data import read_manifest, resolve_path, verify_files
from training.runtime import load_config
from utils.config import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_A.json")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, {"dataset_dir": args.dataset_dir})
    root = resolve_path(config["dataset_dir"])
    rows, summary = read_manifest(root)
    if args.verify_hashes:
        verify_files(root, rows, ("train", "val"))
    print(f"Experimento {summary['experiment']}; seis categorias; test reservado para evaluacion independiente.")
    print(f"{'Categoria':<12} {'Train':>8} {'Val':>8}")
    for label in CLASSES:
        print(f"{label:<12} " + " ".join(f"{sum(r['class'] == label and r['split'] == split for r in rows):>8}" for split in ("train", "val")))
    print("Total: " + str(sum(r["split"] == "train" for r in rows)) + " train, " + str(sum(r["split"] == "val" for r in rows)) + " val.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
