"""Media fija de dos checkpoints de la misma red propia y su misma inicialización."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from scripts.train_detector_own import atomic_save
from training.detector_own import ARCHITECTURE, VisionLocaliza


def blend(start,refinement,output,stability=False):
    start,refinement,output=map(Path,(start,refinement,output))
    if output.exists():raise ValueError('Conservar el resultado anterior')
    parent=torch.load(start,map_location='cpu',weights_only=True)
    child=torch.load(refinement,map_location='cpu',weights_only=True)
    for c in (parent,child):
        p=c['provenance']
        if not (c['architecture']==ARCHITECTURE and p['initialization']=='random'
                and p['pretrained'] is False and p['official_detector_components'] is False):
            raise ValueError('Solo nuestra arquitectura y pesos propios desde cero')
    start_sha=hashlib.sha256(start.read_bytes()).hexdigest()
    if parent['provenance']['initial_state_sha256']!=child['provenance']['initial_state_sha256']:
        raise ValueError('Se exige la misma inicialización propia')
    if stability:
        if (not child['provenance'].get('continuation_of_own_random_initialization') or
                child['provenance'].get('own_weight_interpolation',{}).get('fraction_of_refinement')!=.5):
            raise ValueError('Primero combinar nuestra continuación con su checkpoint de partida')
    elif child['provenance']['own_parent_checkpoint_sha256']!=start_sha:
        raise ValueError('La candidata debe continuar este mismo checkpoint propio')
    # Verificar estructura completa; nunca combinar arquitecturas distintas.
    verifier=VisionLocaliza()
    verifier.load_state_dict(parent['state_dict'],strict=True)
    verifier.load_state_dict(child['state_dict'],strict=True)
    result=deepcopy(child)
    for name,tensor in result['state_dict'].items():
        if tensor.is_floating_point():tensor.mul_(.5).add_(parent['state_dict'][name],alpha=.5)
        if not torch.isfinite(tensor).all():raise ValueError('Pesos no finitos')
    if stability:
        result['provenance']['own_stability_interpolation']={
            'fraction_of_refinement':.5,'own_previous_runtime_sha256':start_sha,
            'own_candidate_sha256':hashlib.sha256(refinement.read_bytes()).hexdigest()}
    else:
        result['provenance']['own_weight_interpolation']={
            'fraction_of_refinement':.5,'own_start_sha256':start_sha,
            'own_refinement_sha256':hashlib.sha256(refinement.read_bytes()).hexdigest(),
            'refinement_epoch':child['epoch']}
    output.parent.mkdir(parents=True,exist_ok=True);atomic_save(result,output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',required=True);p.add_argument('--refinement',required=True)
    p.add_argument('--output',required=True);p.add_argument('--stability',action='store_true');a=p.parse_args()
    blend(a.start,a.refinement,a.output,a.stability)
