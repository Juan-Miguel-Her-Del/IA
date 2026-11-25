"""Recortes de material explícito desde cajas humanas TACO; nunca usa test."""
from collections import Counter
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from app.detector import crop_box
from scripts.evaluate_detector import MATERIALS


def prepare(shared_root=None, source='data/detection/refine_D2', output='data/supplemental/material_crops'):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Conservar la preparación anterior')
    output.mkdir(parents=True)
    rows=json.loads((source/'manifest.json').read_text())
    protected=json.loads(Path('exports/protected_classification_hashes.json').read_text())
    hashes=set(protected['sha256']);pixels=set(protected['pixels_sha256'])
    records=[];excluded=Counter()
    for row in rows:
        if row['split'] not in ('train','val') or row.get('source_dataset')=='MJU-Waste':continue
        if row['sha256'] in hashes or row['pixels_sha256'] in pixels:
            excluded['source_matches_classifier']+=1;continue
        path=source/'images'/row['split']/f"{row['id']:05d}.jpg"
        data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
        rgb=np.array(Image.open(io.BytesIO(data)).convert('RGB'));h,w=rgb.shape[:2]
        for box in row['boxes']:
            label=MATERIALS.get(box['category'])
            if not label:continue
            actual=[v*s for v,s in zip(box['xyxy_normalized'],[w,h,w,h])]
            if min(actual[2]-actual[0],actual[3]-actual[1])<16:excluded['too_small']+=1;continue
            region=crop_box(actual,w,h);x1,y1,x2,y2=region
            conflicting=False
            for other in row['boxes']:
                material=MATERIALS.get(other['category'])
                if not material or material==label:continue
                b=[v*s for v,s in zip(other['xyxy_normalized'],[w,h,w,h])]
                overlap=max(0,min(x2,b[2])-max(x1,b[0]))*max(0,min(y2,b[3])-max(y1,b[1]))
                if overlap/((x2-x1)*(y2-y1))>.15:conflicting=True;break
            if conflicting:excluded['mixed_material_crop']+=1;continue
            im=Image.fromarray(rgb[y1:y2,x1:x2]);im.thumbnail((256,256))
            buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=95);content=buffer.getvalue()
            sha=hashlib.sha256(content).hexdigest();pixelsha=hashlib.sha256(np.array(Image.open(io.BytesIO(content)).convert('RGB')).tobytes()).hexdigest()
            if sha in hashes or pixelsha in pixels:excluded['exact_duplicate']+=1;continue
            hashes.add(sha);pixels.add(pixelsha)
            name=f"{row['split']}/{label}/{row['id']}_{box['annotation_id']}.jpg"
            dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(content)
            records.append({'path':name,'split':row['split'],'class':label,'category':box['category'],
                'image_id':row['id'],'annotation_id':box['annotation_id'],'source_sha256':row['sha256'],
                'source_human_box':actual,'crop_box':region,'sha256':sha,'pixels_sha256':pixelsha,
                'license':row['license']})
    audit={'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        'counts':{s:dict(Counter(r['class'] for r in records if r['split']==s)) for s in ['train','val']},
        'excluded':dict(excluded),'uses_only_explicit_material_categories':True,
        'test_images_opened':0,'foreign_annotations':False,'original_classifier_data_unchanged':True,
        'exact_matches_to_original_classifier_excluded':True,
        'perceptual_audit':'Inherited TACO source splits; exact checks against original classifier hashes. No claim of exhaustive cross-dataset perceptual separation.'}
    assert not ({r['image_id'] for r in records if r['split']=='train'} & {r['image_id'] for r in records if r['split']=='val'})
    (output/'manifest.json').write_text(json.dumps(records,indent=2));(output/'audit.json').write_text(json.dumps(audit,indent=2))
    if shared_root is not None:
        archive=Path(shared_root)/'material_refinement/material_crops.zip';archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for r in records:z.write(output/r['path'],r['path'])
            for n in ['manifest.json','audit.json']:z.write(output/n,n)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shared-root')
    parser.add_argument('--root',default='data/detection/refine_D2')
    parser.add_argument('--output',default='data/supplemental/material_crops')
    args=parser.parse_args();prepare(args.shared_root,args.root,args.output)
