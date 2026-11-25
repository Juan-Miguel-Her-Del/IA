"""Entrena una CNN residual propia desde cero; reanudación solo del mismo experimento."""
import argparse
import json
from pathlib import Path
from training.compact_training import train_compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_C.json")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.dataset_dir:
        config["dataset_dir"] = args.dataset_dir
    train_compact(config, args.output_dir, args.resume)
