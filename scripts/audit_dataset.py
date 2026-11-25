"""Detecta archivos corruptos, SHA-256 repetidos y conflictos entre etiquetas."""
import argparse

from utils.config import add_data_argument, data_root
from utils.dataset import reject_unknown_trashnet_classes, save_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_argument(parser)
    args = parser.parse_args()
    root = data_root(args.data_dir)
    reject_unknown_trashnet_classes(root)
    report = save_audit(root)
    print("Reporte: " + str(root / "reports" / "audit.json"))
    # Codigo distinto de cero para que una automatizacion pueda detectar problemas.
    return 1 if report["corrupt"] or report["label_conflicts"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
