"""Factoría de clasificadores propios; construir no carga ningún checkpoint."""
from training.compact import CompactCNN
from training.hybrid import ARCHITECTURE as HYBRID_ARCHITECTURE, build_hybrid


def build_material_model(config):
    if config.get('initialization') != 'random_from_scratch':
        raise ValueError('Los candidatos de material deben empezar aleatoriamente.')
    if config.get('architecture') == HYBRID_ARCHITECTURE:
        return build_hybrid(config)
    if config.get('architecture') == 'VisionResidual-v2':
        return CompactCNN(config['image_size'], config['dropout'], config['channels'], config['pooling'])
    raise ValueError('El comparador admite la CNN residual propia y el híbrido propio.')
