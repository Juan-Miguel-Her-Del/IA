"""Crea o verifica el ZIP A, sin entrenamiento y sin conexiones a Colab."""
import argparse
import json

from utils.colab_package import export_package, verify_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="ZIP de salida; por defecto exports/Vision-IA_Colab_A.zip.")
    parser.add_argument("--experiment", choices=("A", "B"), default="A")
    parser.add_argument("--drive-workflow", action="store_true", help="Notebook reutilizable y protocolo de sincronizacion.")
    parser.add_argument("--verify", metavar="ZIP", help="Solo comprobar un ZIP existente.")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_package(args.verify), indent=2))
    else:
        output, result = export_package(args.output, args.experiment, args.drive_workflow)
        print(output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
