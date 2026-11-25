"""Entrena la CNN propia solo cuando se ejecuta este comando explicitamente."""
import argparse

from training.engine import train
from training.runtime import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_A.json")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    output = train(load_config(config_path, args))
    print("Resultados guardados en: " + str(output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
