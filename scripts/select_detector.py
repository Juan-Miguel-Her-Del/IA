"""Congela checkpoint y umbral usando únicamente predicciones de validación."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_detector import match


def select(checkpoint,validation,app_config='config/app.json'):
    checkpoint=Path(checkpoint)
    validations=[Path(v) for v in validation] if isinstance(validation,(tuple,list)) else [Path(validation)]
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sweep=[]
    actual=None
    for validation in validations:
        metadata=json.loads((validation/'metrics.json').read_text(encoding='utf-8'))
        if metadata['split']!='val' or metadata['checkpoint_sha256']!=sha:
            raise ValueError('Seleccionar solo con validación del mismo checkpoint.')
        rows=json.loads((validation/'predictions.json').read_text(encoding='utf-8'))
        truth=[(r['image_id'],r['truth']) for r in rows]
        if actual is not None and truth!=actual:
            raise ValueError('Las alternativas deben usar exactamente la misma validación.')
        actual=truth
        total=sum(len(r['truth']) for r in rows)
        for step in range(2,19):
            confidence=round(step*.05,2)
            tp,predicted=0,0
            for row in rows:
                ps=[p for p in row['predictions'] if p['score']>=confidence]
                tp+=len(match(ps,row['truth']));predicted+=len(ps)
            precision,recall=tp/max(predicted,1),tp/max(total,1)
            f1=2*precision*recall/max(precision+recall,1e-10)
            sweep.append({'confidence':confidence,'nms_iou':metadata.get('nms_iou',.45),
                          'precision':precision,'recall':recall,'F1':f1})
    chosen=max(sweep,key=lambda r:(r['F1'],r['precision'],r['confidence'],r['nms_iou']))
    result={'checkpoint_sha256':sha,'confidence':chosen['confidence'],'nms_iou':chosen['nms_iou'],
            'selection':'maximum detection F1 on validation at IoU50; confidence and NMS',
            'validation_images':len(rows),'operating_point':chosen,'validation_sweep':sweep,
            'test_used':False,'test_evaluated':False}
    destination=checkpoint.parent/'selection.json'
    if destination.exists():
        existing=json.loads(destination.read_text(encoding='utf-8'))
        if existing!=result:raise ValueError('Ya hay una selección distinta congelada. Conserva el experimento original.')
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    config=Path(app_config)
    cfg=json.loads(config.read_text(encoding='utf-8'));cfg['detection_threshold']=chosen['confidence']
    cfg['detection_nms_iou']=chosen['nms_iou']
    config.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['confidence','nms_iou','operating_point','test_used']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--validation',required=True,nargs='+')
    a=p.parse_args();select(a.checkpoint,a.validation)
