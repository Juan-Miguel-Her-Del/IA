"""Entrenamiento explícito de CNN/híbrido propios con el dataset COCO nuevo."""
import argparse
import json
from pathlib import Path

from training.intake_training import prepare_data, train_intake, validate_config
from training.material_models import build_material_model
from utils.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'config/train_intake_hybrid.json')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true', help='Solo metadatos: sin abrir fotos ni entrenar ni escribir archivos.')
    parser.add_argument('--smoke', action='store_true', help='Dos lotes; verifica actualizaciones y descarta los pesos.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.inspect and (args.smoke or args.resume):
        parser.error('--inspect no se combina con entrenamiento.')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.dataset:
        config['dataset_dir'] = str(args.dataset.resolve())
    validate_config(config)
    if args.inspect:
        _, _, _, data = prepare_data(config)
        model = build_material_model(config)
        print(json.dumps({'architecture': config['architecture'], 'parameters': sum(p.numel() for p in model.parameters()),
                          'data': data, 'images_opened': 0, 'training_started': False, 'files_written': 0}, indent=2))
        return
    if not args.smoke and args.output is None:
        parser.error('El entrenamiento completo requiere --output fuera del dataset.')
    train_intake(config, args.output, args.device, args.smoke, args.resume)


if __name__ == '__main__':
    main()
