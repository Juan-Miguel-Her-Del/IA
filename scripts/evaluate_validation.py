"""Evalua exclusivamente val de A (421) o B (884); nunca train/test."""
import argparse
from datetime import datetime, timezone

from training.data import resolve_path
from training.validation import evaluate_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--checkpoint", help="Para B indicar el checkpoint de su ejecucion sincronizada.")
    parser.add_argument("--dataset-dir", help="Por defecto data/experiments/<experimento>/processed.")
    parser.add_argument("--output-dir", help="Carpeta vacia; por defecto una carpeta nueva bajo runs/validation/.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=24)
    args = parser.parse_args()
    if args.experiment == "B" and not args.checkpoint:
        parser.error("B requiere --checkpoint con la ruta de su ejecucion sincronizada; no se utiliza el modelo A por defecto.")
    checkpoint = args.checkpoint or "models/best_model.pth"
    dataset = args.dataset_dir or f"data/experiments/{args.experiment}/processed"
    output = args.output_dir or f"runs/validation/{args.experiment}_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    evaluate_validation(resolve_path(checkpoint), resolve_path(dataset), resolve_path(output),
                        args.device, args.batch_size, args.num_workers, args.max_errors, experiment=args.experiment)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
