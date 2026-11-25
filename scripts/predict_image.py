"""Clasifica una fotografia nueva en Windows, sin interfaz ni camara."""
import argparse

from training.data import resolve_path
from training.prediction import predict_image
from utils.config import PROJECT_ROOT, data_root
from utils.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Ruta de una fotografia nueva; no admite imagenes test.")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", help="JSON opcional con las seis probabilidades y trazabilidad.")
    args = parser.parse_args()
    result = predict_image(resolve_path(args.checkpoint), resolve_path(args.image), args.device)
    print("Categoria predicha: " + result["predicted_class"])
    print(f"Confianza softmax: {result['confidence']:.2%} (no calibrada)")
    if args.output:
        output = resolve_path(args.output)
        if output == resolve_path(args.image) or output == resolve_path(args.checkpoint):
            raise ValueError("El JSON de salida no puede sobrescribir la foto o el checkpoint.")
        if output.suffix.lower() != ".json" or any(output.is_relative_to(root.resolve()) for root in (PROJECT_ROOT / "data", data_root())):
            raise ValueError("La salida debe ser un archivo .json fuera de los datos y particiones protegidos.")
        write_json(output, result)
        print("Detalle: " + str(output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
