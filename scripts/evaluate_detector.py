"""Evaluación espacial independiente de la CNN; test solo tras congelar la selección."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from app.detector import NeuralDetector, box_iou, annotate_objects, analyze_objects
from app.vision import InferenceService
from training.detection_metrics import match, spatial_metrics, coco_reference

# Solo categorías con material explícito; los envases ambiguos/multicapa se excluyen.
MATERIALS = {
    'Aluminium foil':'metal','Aluminium blister pack':'metal','Metal bottle cap':'metal',
    'Food Can':'metal','Aerosol':'metal','Drink can':'metal','Metal lid':'metal','Pop tab':'metal','Scrap metal':'metal',
    'Other plastic bottle':'plastico','Clear plastic bottle':'plastico','Plastic bottle cap':'plastico',
    'Disposable plastic cup':'plastico','Other plastic cup':'plastico','Plastic lid':'plastico',
    'Other plastic':'plastico','Plastic film':'plastico','Six pack rings':'plastico','Garbage bag':'plastico',
    'Other plastic wrapper':'plastico','Single-use carrier bag':'plastico','Polypropylene bag':'plastico',
    'Plastic glooves':'plastico','Plastic utensils':'plastico','Plastic straw':'plastico',
    'Glass bottle':'vidrio','Broken glass':'vidrio','Glass cup':'vidrio','Glass jar':'vidrio',
    'Toilet tube':'carton','Other carton':'carton','Egg carton':'carton','Corrugated carton':'carton',
    'Meal carton':'carton','Pizza box':'carton','Magazine paper':'papel','Tissues':'papel',
    'Wrapping paper':'papel','Normal paper':'papel','Paper bag':'papel','Paper straw':'papel','Food waste':'organicos'
}


def evaluate(checkpoint,root,output,split='val',classifier=None,confidence=.35,final=False,nms_iou=.45,candidate=None,candidate_threshold=.4):
    root,output=Path(root),Path(output)
    if split=='test' and not final:
        raise ValueError('Test reservado: requiere --final y selección congelada.')
    if output.exists():raise ValueError('La salida ya existe; conserva los resultados originales.')
    sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    selection=Path(checkpoint).parent/'selection.json'
    if split=='test':
        if not selection.is_file():raise ValueError('Falta selection.json congelado tras validación.')
        frozen=json.loads(selection.read_text())
        if frozen.get('test_evaluated'):
            raise ValueError('La evaluación final de esta selección ya está guardada.')
        if frozen['checkpoint_sha256']!=sha or frozen['confidence']!=confidence or frozen.get('nms_iou',.45)!=nms_iou:
            raise ValueError('El modelo o umbral cambiaron después de congelar selección.')
    detector=NeuralDetector(checkpoint,confidence=confidence,nms_iou=nms_iou)
    service=InferenceService(classifier,candidate_checkpoint=candidate,candidate_threshold=candidate_threshold) if classifier else None
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    rows=[r for r in manifest if r['split']==split]
    if len(rows)<2:raise ValueError('La partición debe contener al menos dos imágenes.')
    output.mkdir(parents=True,exist_ok=False)
    records,timings=[],[]
    crop_correct,matched_classifiable,mappable_total=0,0,0
    oracle_correct,oracle_total,full_correct,full_total=0,0,0,0
    oracle_single_correct=0
    paired_total,paired_crop_correct,paired_full_correct=0,0,0
    # IoU metrics never use CNN confidence or classification as spatial evidence.
    for row in rows:
        path=root/'images'/split/f"{row['id']:05d}.jpg"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Imagen modificada.')
        rgb=np.asarray(Image.open(path).convert('RGB'))
        h,w=rgb.shape[:2]
        start=time.perf_counter()
        detections=detector.detect(rgb,confidence=.001)
        elapsed=(time.perf_counter()-start)*1000
        timings.append(elapsed)
        truth=[{'box':[b['xyxy_normalized'][i]*[w,h,w,h][i] for i in range(4)],'category':b['category']} for b in row['boxes']]
        predictions=[{'box':list(d.box),'score':d.confidence} for d in detections]
        record={'image_id':row['id'],'truth':truth,'predictions':predictions,'size':[w,h]}
        if service:
            operational=[(d,p) for d,p in zip(detections,predictions) if p['score']>=confidence]
            labels={i:service.classify(rgb,region=d.region).material for i,(d,p) in enumerate(operational)}
            pairs=match([p for d,p in operational],truth)
            joint=[]
            mappable_total+=sum(t['category'] in MATERIALS for t in truth)
            for i,j,iou in pairs:
                expected=MATERIALS.get(truth[j]['category'])
                if expected:
                    matched_classifiable+=1
                    correct=labels[i]==expected
                    crop_correct+=correct
                    joint.append({'truth_index':j,'actual':expected,'predicted':labels[i],'IoU':iou})
            from app.detector import crop_box
            from app.localization import Region
            oracle_labels={}
            for truth_index,t in enumerate(truth):
                expected=MATERIALS.get(t['category'])
                if expected:
                    oracle_total+=1
                    oracle_labels[truth_index]=service.classify(rgb,region=Region(crop_box(t['box'],w,h),'anotacion-humana')).material
                    oracle_correct+=oracle_labels[truth_index]==expected
            if len(truth)==1 and truth[0]['category'] in MATERIALS:
                full_total+=1
                expected=MATERIALS[truth[0]['category']]
                full_label=service.classify(rgb,mode='full').material
                full_correct+=full_label==expected
                oracle_single_correct+=oracle_labels[0]==expected
                for prediction_index,truth_index,iou in pairs:
                    if truth_index==0:
                        paired_total+=1
                        paired_crop_correct+=labels[prediction_index]==expected
                        paired_full_correct+=full_label==expected
            record['joint']=joint
        records.append(record)
    metrics=spatial_metrics(records,confidence)
    try:
        reference=coco_reference(records)
        metrics['COCO_reference']=reference
        metrics['COCO_max_AP_difference']=max(abs(metrics[k]-reference[k]) for k in ('mAP50','mAP50_95'))
        if metrics['COCO_max_AP_difference']>1e-6:
            raise ValueError('La implementación AP difiere de COCO. Revisar antes de publicar métricas.')
    except ImportError:
        metrics['COCO_reference']=None
    metrics.update({'split':split,'checkpoint_sha256':sha,'detector_CPU_ms_median':float(np.median(timings[1:])),
                    'detector_CPU_ms_p95':float(np.percentile(timings[1:],95)),
                    'detector_CPU_FPS':1000/float(np.mean(timings[1:])),'warmup_images':1,
                    'test_used_for_selection':False,'timing_detection_confidence':.001,'CPU_threads':2,'nms_iou':nms_iou})
    if service:
        metrics['CNN']={'checkpoint_sha256':service.checkpoint_sha256,'candidate_checkpoint_sha256':service.candidate_checkpoint_sha256,
             'candidate_threshold':candidate_threshold if candidate else None,'matched_known_material_objects':matched_classifiable,
             'correct_material_on_matched_objects':crop_correct,'classification_accuracy_matched':crop_correct/max(matched_classifiable,1),
             'end_to_end_recall_IoU50_correct_material':crop_correct/max(mappable_total,1),'known_material_objects':mappable_total,
             'oracle_crop_accuracy':oracle_correct/max(oracle_total,1),
             'full_image_accuracy_single_object':full_correct/max(full_total,1),'full_image_comparison_images':full_total,
             'oracle_crop_accuracy_same_single_objects':oracle_single_correct/max(full_total,1),
             'paired_localized_single_objects':{'images':paired_total,
                 'detector_crop_accuracy':paired_crop_correct/max(paired_total,1),
                 'full_image_accuracy_same_objects':paired_full_correct/max(paired_total,1)},
             'note':'Material mapping uses only explicit TACO materials. Mixed/unknown categories excluded. No material AP claimed.'}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(records),encoding='utf-8')
    if split=='test':
        frozen['test_evaluated']=True
        frozen['final_metrics']=str((output/'metrics.json').resolve())
        selection.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ('split','images','objects','mAP50','mAP50_95','precision','recall','detector_CPU_ms_median')}),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--root',default='data/detection/taco')
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--classifier');p.add_argument('--confidence',type=float,default=.35);p.add_argument('--final',action='store_true')
    p.add_argument('--nms-iou',type=float,default=.45)
    p.add_argument('--candidate');p.add_argument('--candidate-threshold',type=float,default=.4)
    a=p.parse_args();evaluate(a.checkpoint,a.root,a.output,a.split,a.classifier,a.confidence,a.final,a.nms_iou,a.candidate,a.candidate_threshold)
