"""Importa exclusivamente el detector propio terminado, conservando todos los experimentos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def import_model(source,root=Path('.')):
    import torch
    source,root=Path(source),Path(root).resolve()
    selection=json.loads((source/'selected.json').read_text(encoding='utf-8'))
    if selection.get('stopped') is not False:
        raise ValueError('La ejecución está detenida; conservar checkpoints sin activarla como selección final.')
    run=source/selection['run']
    checkpoint=run/selection.get('checkpoint','best.pth')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['provenance']['initialization']!='random' or payload['provenance']['pretrained']:
        raise ValueError('No se admite un detector preentrenado.')
    if payload.get('architecture')!='VisionLocaliza-D2' or payload['provenance'].get('official_detector_components') is not False:
        raise ValueError('D1/YOLO no cumple arquitectura propia; importar solo VisionLocaliza-D2.')
    destination=root/'models/detector_D2'/run.name
    destination.mkdir(parents=True,exist_ok=True)
    for name in (checkpoint.name,'provenance.json','history.json','selection.json'):
        src=run/name
        if not src.exists():continue
        dst=destination/name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).digest()!=hashlib.sha256(src.read_bytes()).digest():
            raise ValueError(f'No sobrescribir un artefacto diferente: {dst}')
        shutil.copy2(src,dst)
    (destination/'checkpoint_provenance.json').write_text(json.dumps(payload['provenance'],indent=2),encoding='utf-8')
    config_path=root/'config/app.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    cfg['detector_checkpoint']=(destination/checkpoint.name).relative_to(root).as_posix()
    if (destination/'selection.json').exists():
        frozen=json.loads((destination/'selection.json').read_text(encoding='utf-8'))
        if frozen['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('La configuración congelada corresponde a otro checkpoint.')
        cfg['detection_threshold']=frozen['confidence']
        cfg['detection_nms_iou']=frozen.get('nms_iou',.45)
    config_path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    print(destination/checkpoint.name)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='runs/detector_D2')
    a=p.parse_args();import_model(a.source)
