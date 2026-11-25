"""Preserve D2 train/TACO holdouts; add audited official MJU validation only."""
from collections import Counter,defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import imagehash
import numpy as np
from PIL import Image

from scripts.prepare_mju import polygon_boxes


def prepare():
    parent=Path('data/detection/mixed_D2');output=Path('data/detection/refine_D2')
    output.mkdir(parents=True,exist_ok=True)
    original_bytes=(parent/'manifest.json').read_bytes();original=json.loads(original_bytes)
    annotations=Path('data/detection/mju/val_original.json')
    if not annotations.is_file():annotations=Path('data/reference/MJU_val.json')
    annotations_bytes=annotations.read_bytes();source=json.loads(annotations_bytes)
    by_image=defaultdict(list)
    for annotation in source['annotations']:by_image[annotation['image_id']].append(annotation)
    hashes={r['sha256'] for r in original}|{r.get('source_png_sha256') for r in original}
    pixels={r['pixels_sha256'] for r in original}
    perceptual=[int(r['phash'],16) for r in original]
    seconds={r['file_name'].split('/')[-1][:19] for r in original if r.get('source_dataset')=='MJU-Waste'}
    identifiers={r['id'] for r in original};excluded=Counter();added=[]
    with zipfile.ZipFile('data/detection/mju/original.zip') as archive:
        for info in sorted(source['images'],key=lambda row:row['id']):
            if info['file_name'][:19] in seconds:excluded['same_capture_second_as_train']+=1;continue
            if not by_image[info['id']]:excluded['no_instance_annotation']+=1;continue
            raw=archive.read('JPEGImages/'+info['file_name'])
            rgb=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));height,width=rgb.shape[:2]
            if (width,height)!=(info['width'],info['height']):raise ValueError('Unexpected source dimensions')
            mask_bytes=archive.read('SegmentationClass/'+info['file_name'])
            mask=np.asarray(Image.open(io.BytesIO(mask_bytes)))
            if mask.shape!=(height,width) or not set(np.unique(mask)).issubset({0,1}):raise ValueError('Unexpected human mask')
            boxes,raster=polygon_boxes(by_image[info['id']],width,height)
            foreground=mask==0;union=np.logical_or(foreground,raster>0).sum()
            overlap=float(np.logical_and(foreground,raster>0).sum()/max(union,1))
            if overlap<.95:excluded['polygon_mask_disagreement']+=1;continue
            phash=int(str(imagehash.phash(Image.fromarray(rgb))),16)
            if any((phash^other).bit_count()<=6 for other in perceptual):excluded['perceptual_candidate']+=1;continue
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=95);jpeg=buffer.getvalue()
            pixel_sha=hashlib.sha256(np.asarray(Image.open(io.BytesIO(jpeg))).tobytes()).hexdigest()
            raw_sha=hashlib.sha256(raw).hexdigest();sha=hashlib.sha256(jpeg).hexdigest()
            if raw_sha in hashes or sha in hashes or pixel_sha in pixels:excluded['exact_duplicate']+=1;continue
            ident=300000+info['id']
            if ident in identifiers:raise ValueError('Source split identifiers overlap')
            path=output/'images/val'/f'{ident:05d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=jpeg:raise ValueError('Preserve previous preparation')
            path.write_bytes(jpeg)
            added.append({'id':ident,'source_image_id':info['id'],'source_dataset':'MJU-Waste','source_original_split':'val',
                'file_name':'MJU/'+info['file_name'],'width':width,'height':height,'split':'val','group':'MJU_val_'+info['file_name'][:19],
                'sha256':sha,'pixels_sha256':pixel_sha,'phash':f'{phash:016x}','source_png_sha256':raw_sha,
                'source_human_mask_sha256':hashlib.sha256(mask_bytes).hexdigest(),'human_polygon_mask_IoU':overlap,
                'license':'MIT, Tao Wang (2020), official repository','boxes':boxes})
            perceptual.append(phash);hashes.add(raw_sha);hashes.add(sha);pixels.add(pixel_sha);identifiers.add(ident)
    if len(added)<25:raise ValueError(f'Insufficient independent validation examples: {len(added)}, {excluded}')
    for row in original:
        if row['split']=='test':continue
        rel=Path('images')/row['split']/f"{row['id']:05d}.jpg";target=output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:os.link(parent/rel,target)
            except OSError:shutil.copy2(parent/rel,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    combined=original+added
    assert [r for r in combined if r['split']=='test']==[r for r in original if r['split']=='test']
    report={'parent_manifest_sha256':hashlib.sha256(original_bytes).hexdigest(),
        'original_rows_preserved':True,'train_images':sum(r['split']=='train' for r in combined),
        'TACO_val':197,'MJU_val':len(added),'MJU_val_instances':sum(len(r['boxes']) for r in added),
        'MJU_annotation_source':'https://raw.githubusercontent.com/realwecan/mju-waste/master/val.json',
        'MJU_annotations_sha256':hashlib.sha256(annotations_bytes).hexdigest(),'excluded':dict(excluded),
        'minimum_polygon_mask_IoU':min(r['human_polygon_mask_IoU'] for r in added),
        'cross_split_perceptual_candidates':0,'original_test_unchanged':True,'test_images_opened':0,
        'foreign_weights_or_foundation_annotations':False}
    (output/'manifest.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'parent_manifest.json').write_bytes(original_bytes)
    (output/'MJU_val_original.json').write_bytes(annotations_bytes)
    for name in ['parent_audit.json','MJU_LICENSE']:shutil.copy2(parent/name,output/name)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':prepare()
