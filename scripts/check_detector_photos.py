"""Comprueba fotos humanas existentes y fondos de validación, sin webcam ni selección con test."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from app.detector import NeuralDetector, analyze_objects, annotate_objects, box_iou
from app.vision import InferenceService, decode_image


def check(checkpoint,classifier,output,root='data/detection/taco',confidence=.35,nms_iou=.45,candidate=None,candidate_threshold=.4):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    detector=NeuralDetector(checkpoint,confidence=confidence,nms_iou=nms_iou)
    cnn=InferenceService(classifier,candidate_checkpoint=candidate,candidate_threshold=candidate_threshold)
    reports=[]
    for path in sorted(Path('data/field_samples/samples').glob('*/record.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        frame=decode_image((path.parent/'original.png').read_bytes())
        start=time.perf_counter();objects=analyze_objects(frame,detector,cnn)
        elapsed=(time.perf_counter()-start)*1000
        Image.fromarray(annotate_objects(frame,objects)).save(output/f'{path.parent.name}.jpg')
        full=cnn.classify(frame,mode='full')
        reports.append({'sample':path.parent.name,'actual':record['true_class'],'detections':len(objects),
                        'full_prediction':full.material,'pipeline_CPU_ms':elapsed,
                        'predictions':[{'box':o.detection.box,'detector_confidence':o.detection.confidence,
                                        'material':o.reading.material,'material_confidence':o.reading.confidence,
                                        'crop_box':o.reading.region.box} for o in objects]})
    # Recortes naturales que no intersectan ninguna caja anotada; inspección visual
    # complementaria obligatoria porque TACO puede contener residuos sin etiquetar.
    manifest=json.loads((Path(root)/'manifest.json').read_text(encoding='utf-8'))
    backgrounds=[]
    for row in manifest:
        if row['split']!='val':continue
        frame=np.asarray(Image.open(Path(root)/'images/val'/f"{row['id']:05d}.jpg").convert('RGB'))
        h,w=frame.shape[:2]
        side=round(min(w,h)*.55)
        for x,y in [(0,0),(w-side,0),(0,h-side),(w-side,h-side)]:
            box=(x/w,y/h,(x+side)/w,(y+side)/h)
            if any(box_iou(box,t['xyxy_normalized'])>0 for t in row['boxes']):continue
            crop=frame[y:y+side,x:x+side].copy()
            objects=detector.detect(crop)
            name=f"background_{len(backgrounds):02d}.jpg"
            Image.fromarray(crop).save(output/name)
            backgrounds.append({'image_id':row['id'],'crop':box,'file':name,'detections':len(objects),
                                'max_confidence':max((o.confidence for o in objects),default=0.)})
            break
        if len(backgrounds)>=20:break
    sheet=Image.new('RGB',(5*192,4*216),'white');draw=ImageDraw.Draw(sheet)
    for i,b in enumerate(backgrounds):
        picture=Image.open(output/b['file']).resize((192,192))
        x,y=(i%5)*192,(i//5)*216
        sheet.paste(picture,(x,y));draw.text((x+4,y+194),f"{i:02d}: {b['detections']} det",fill='black')
    sheet.save(output/'background_contact_sheet.jpg')
    # Un video a partir de fotos desplazadas sirve solo como prueba técnica de
    # seguimiento, no se presenta como video real ni sustituye una webcam.
    result={'confidence':confidence,'nms_iou':nms_iou,'photos':reports,'background_candidates':backgrounds,
            'classifier_sha256':cnn.checkpoint_sha256,'candidate_sha256':cnn.candidate_checkpoint_sha256,
            'candidate_threshold':candidate_threshold if candidate else None,
            'background_note':'Natural validation crops disjoint from annotated boxes; review contact sheet for unannotated objects.',
            'physical_webcam_tested':False,'real_video_tested':False,
            'field_boxes_note':'Existing manual regions are diagnostic crops, not authoritative tight detection annotations; no field IoU/AP claimed.'}
    (output/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'photos':len(reports),'detections_per_photo':[r['detections'] for r in reports],
                      'background_candidates':len(backgrounds),'physical_webcam_tested':False}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--classifier',required=True)
    p.add_argument('--output',required=True);p.add_argument('--root',default='data/detection/taco')
    p.add_argument('--confidence',type=float,default=.35)
    p.add_argument('--nms-iou',type=float,default=.45)
    p.add_argument('--candidate');p.add_argument('--candidate-threshold',type=float,default=.4)
    a=p.parse_args();check(a.checkpoint,a.classifier,a.output,a.root,a.confidence,a.nms_iou,a.candidate,a.candidate_threshold)
