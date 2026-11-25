"""Prepara o verifica A/B: procedencia, auditoria, grupos, particiones e inspeccion."""
import argparse
from pathlib import Path

from utils.config import CLASSES, SPLITS, add_data_argument, data_root
from utils.experiment_pipeline import build_experiment, verify_experiment
from utils.experiments import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"))
    parser.add_argument("--config", type=Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--rebuild", action="store_true")
    choice.add_argument("--verify", action="store_true", help="Verifica datos actuales sin reconstruir.")
    add_data_argument(parser)
    args = parser.parse_args()
    name, config, selected = load_experiment(args.experiment, args.config)
    root = data_root(args.data_dir)
    if args.verify:
        result = verify_experiment(root, name, config, selected)
        print(f"Experimento {name}: {result['images_checked']} imagenes; fase 1 conservada: {result['phase1_preserved']}.")
        if not result["valid"]:
            raise ValueError("; ".join(result["errors"][:10]))
        print("Verificacion correcta: hashes, procedencia, grupos y particiones.")
    else:
        summary = build_experiment(root, name, config, selected, args.rebuild)
        print(f"Experimento {name}: {summary['total']} imagenes utilizables de {summary['imported']} importadas.")
        print(f"{'Clase':<12} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
        for label in CLASSES:
            values = [summary["counts"][split][label] for split in SPLITS]
            print(f"{label:<12} " + " ".join(f"{value:>7}" for value in [*values, sum(values)]))
        print(f"Conflictos: {summary['conflict_groups']} grupos, {summary['conflict_files']} archivos excluidos.")
        print(f"Parejas visuales candidatas para revision: {summary['visual_candidate_pairs']}.")
        print("Reportes y muestras: " + str(root / "experiments" / name))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
