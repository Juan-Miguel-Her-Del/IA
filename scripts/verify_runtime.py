"""Verify and exercise the active Vision-IA models on CPU from any project copy."""
import json
import numpy as np

from app.detector import NeuralDetector
from app.settings import checkpoint_signature, load_settings
from app.vision import InferenceService
from utils.config import PROJECT_ROOT
from utils.dataset import sha256


def verify():
    folder = PROJECT_ROOT / 'models/runtime'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    for name, metadata in manifest.items():
        if sha256(folder / name) != metadata['sha256']:
            raise ValueError(f'Checkpoint alterado: {name}')
    settings = load_settings()
    detector = NeuralDetector(checkpoint_signature(settings['detector_checkpoint'])[0],
                              cpu_threads=settings['cpu_threads'], confidence=settings['detection_threshold'],
                              nms_iou=settings['detection_nms_iou'], views=settings['detection_views'])
    candidate = settings.get('material_candidate_checkpoint')
    classifier = InferenceService(checkpoint_signature(settings['checkpoint'])[0], settings['cpu_threads'],
                                  checkpoint_signature(candidate)[0] if candidate else None,
                                  candidate_weight=settings.get('material_candidate_weight'))
    rgb = np.full((240, 320, 3), 100, dtype=np.uint8)
    rgb[70:195, 120:230] = (200, 130, 60)
    detector.detect(rgb)
    material = classifier.classify(rgb, mode='manual', manual_bounds=(.2,.2,.8,.8))
    if len(material.probabilities) != 6 or not np.isclose(sum(material.probabilities.values()), 1, atol=1e-6):
        raise ValueError('La salida de la CNN no representa seis clases normalizadas.')
    print('Checkpoints verificados. Ejecución CPU correcta para el detector y los clasificadores activos.')


if __name__ == '__main__':
    verify()
