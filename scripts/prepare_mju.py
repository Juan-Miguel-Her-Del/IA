"""Añade RGB oficiales de MJU-train; cajas desde polígonos contrastados con sus máscaras.

El bbox del JSON oficial tiene un error de escala. No se utiliza ni se
anotan imágenes con modelos: se convierte la geometría humana existente.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import zipfile

import cv2
import imagehash
import numpy as np
from PIL import Image
import requests

SOURCE='https://github.com/realwecan/mju-waste'
ANNOTATIONS='https://raw.githubusercontent.com/realwecan/mju-waste/master/train.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def polygon_boxes(annotations,width,height):
    raster=np.zeros((height,width),np.uint8)
    boxes=[]
    for annotation in annotations:
        segmentation=annotation.get('segmentation',[])
        if not isinstance(segmentation,list) or not segmentation:raise ValueError('Polígono humano ausente.')
        polygons=[segmentation] if isinstance(segmentation[0],(int,float)) else segmentation
        points=[]
        for polygon in polygons:
            xy=np.asarray(polygon,dtype=np.float32).reshape(-1,2)
            if len(xy)<3 or not np.isfinite(xy).all():raise ValueError('Polígono inválido.')
            points.append(xy)
            cv2.fillPoly(raster,[np.round(xy).astype(np.int32)],1)
        xy=np.concatenate(points)
        x1,y1=np.maximum(xy.min(0),0);x2,y2=np.minimum(xy.max(0),[width,height])
        if min(x2-x1,y2-y1)<2:raise ValueError('Instancia demasiado pequeña.')
        boxes.append({'annotation_id':annotation['id'],'category':'MJU waste',
                      'provided_bbox_not_used':annotation['bbox'],'human_polygon':segmentation,
                      'xyxy_normalized':[float(x1/width),float(y1/height),float(x2/width),float(y2/height)]})
    return boxes,raster


def prepare(archive,taco_root,output,limit=600,classification_hashes=None):
    taco_root,output=Path(taco_root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    source_data=requests.get(ANNOTATIONS,timeout=40);source_data.raise_for_status()
    source=json.loads(source_data.content)
    license_data=requests.get('https://raw.githubusercontent.com/realwecan/mju-waste/master/LICENSE',timeout=30)
    license_data.raise_for_status()
    (output/'MJU_LICENSE').write_bytes(license_data.content)
    (output/'MJU_train_original.json').write_bytes(source_data.content)
    original_manifest=(taco_root/'manifest.json').read_bytes()
    original_audit=(taco_root/'audit.json').read_bytes()
    original=json.loads(original_manifest)
    parent_sha=sha(original_audit)
    if (output/'parent_audit.json').exists() and (output/'parent_audit.json').read_bytes()!=original_audit:
        raise ValueError('No sobrescribir una preparación diferente.')
    (output/'parent_audit.json').write_bytes(original_audit)
    (output/'parent_manifest.json').write_bytes(original_manifest)
    forbidden_sha={r['sha256'] for r in original}
    forbidden_pixels={r['pixels_sha256'] for r in original}
    if classification_hashes:
        protected=json.loads(Path(classification_hashes).read_text())
        forbidden_sha.update(protected['sha256']);forbidden_pixels.update(protected['pixels_sha256'])
    perceptual=[int(r['phash'],16) for r in original]
    by_image=defaultdict(list)
    for annotation in source['annotations']:by_image[annotation['image_id']].append(annotation)
    candidates=sorted(source['images'],key=lambda r:r['file_name'])
    random.Random(42).shuffle(candidates)
    accepted=[];excluded=Counter();seconds=set();own_pixels=set();mask_ious=[]
    with zipfile.ZipFile(archive) as z:
        for info in candidates:
            annotations=by_image[info['id']]
            if not annotations:excluded['missing_instance_annotations']+=1;continue
            second=info['file_name'][:19]
            if second in seconds:excluded['same_capture_second']+=1;continue
            payload=z.read('JPEGImages/'+info['file_name'])
            rgb=np.asarray(Image.open(io.BytesIO(payload)).convert('RGB'))
            height,width=rgb.shape[:2]
            if (width,height)!=(info['width'],info['height']):raise ValueError('Dimensiones originales inconsistentes.')
            mask_payload=z.read('SegmentationClass/'+info['file_name'])
            mask=np.asarray(Image.open(io.BytesIO(mask_payload)))
            if mask.shape!=(height,width) or not set(np.unique(mask)).issubset({0,1}):
                excluded['unexpected_human_mask']+=1;continue
            try:boxes,raster=polygon_boxes(annotations,width,height)
            except ValueError:excluded['invalid_polygon']+=1;continue
            foreground=mask==0  # Convención comprobada en las máscaras originales de MJU.
            union=np.logical_or(foreground,raster>0).sum()
            mask_iou=float(np.logical_and(foreground,raster>0).sum()/max(union,1))
            if mask_iou<.95:excluded['polygon_mask_disagreement']+=1;continue
            pixel_hash=sha(rgb.tobytes());raw_sha=sha(payload)
            if raw_sha in forbidden_sha or pixel_hash in forbidden_pixels:
                excluded['existing_dataset_exact_match']+=1;continue
            if pixel_hash in own_pixels:excluded['duplicate_pixels']+=1;continue
            perceptual_hash=int(str(imagehash.phash(Image.fromarray(rgb))),16)
            if any((perceptual_hash^other).bit_count()<=6 for other in perceptual):
                excluded['TACO_perceptual_candidate']+=1;continue
            ident=300000+info['id']
            path=output/'images/train'/f'{ident:05d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=95)
            canonical=buffer.getvalue()
            if path.exists() and path.read_bytes()!=canonical:raise ValueError('No sobrescribir imagen distinta.')
            path.write_bytes(canonical)
            decoded=Image.open(io.BytesIO(canonical)).convert('RGB')
            accepted.append({'id':ident,'source_image_id':info['id'],'source_dataset':'MJU-Waste',
                'source_original_split':'train','file_name':'MJU/'+info['file_name'],'width':width,'height':height,
                'split':'train','group':'MJU_train_only','source_url':SOURCE,'license':'MIT, Tao Wang (2020), official repository',
                'source_png_sha256':raw_sha,'source_human_mask_sha256':sha(mask_payload),
                'sha256':sha(canonical),'pixels_sha256':sha(np.asarray(decoded).tobytes()),
                'phash':f'{perceptual_hash:016x}','human_polygon_mask_IoU':mask_iou,'boxes':boxes})
            seconds.add(second);own_pixels.add(pixel_hash);mask_ious.append(mask_iou)
            if len(accepted)>=limit:break
    if len(accepted)<100:raise ValueError(f'Pocas imágenes verificadas: {len(accepted)}, {excluded}')
    # Enlazar/copiar las imágenes originales sin modificar ningún píxel ni partición.
    for row in original:
        if row['split']=='test':continue  # Test permanece reservado, sin extraerlo.
        name=f"{row['id']:05d}.jpg"
        src=taco_root/'images'/row['split']/name
        dst=output/'images'/row['split']/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst.read_bytes())!=row['sha256']:raise ValueError('Imagen original alterada.')
        else:
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
    combined=original+accepted
    holdout=[r for r in original if r['split']!='train']
    assert [r for r in combined if r['split']!='train']==holdout
    report={'parent_audit_sha256':parent_sha,'parent_manifest_sha256':sha(original_manifest),
            'preserved_original_rows':True,'added_only_train':True,'source':SOURCE,
            'annotation_source':ANNOTATIONS,'annotation_sha256':sha(source_data.content),
            'license_source':SOURCE+'/blob/master/LICENSE','MJU_images':len(accepted),
            'TACO_images':len(original),'splits':dict(Counter(r['split'] for r in combined)),
            'cross_split_candidates':0,'perceptual_threshold':6,'excluded':dict(excluded),
            'min_human_polygon_mask_IoU':min(mask_ious),'foundation_annotations_used':False,
            'bbox_method':'min/max of original instance polygons, each verified against the supplied human mask; official bbox ignored due scale error',
            'MJU_original_val_and_test_images_opened':False}
    (output/'manifest.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',default='data/detection/mju/original.zip')
    p.add_argument('--taco-root',default='data/detection/taco');p.add_argument('--output',default='data/detection/mixed_D2')
    p.add_argument('--limit',type=int,default=600);p.add_argument('--classification-hashes',default='exports/protected_classification_hashes.json')
    a=p.parse_args();prepare(a.archive,a.taco_root,a.output,a.limit,a.classification_hashes)
