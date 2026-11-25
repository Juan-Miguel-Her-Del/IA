"""TACO oficial: cajas humanas, miniaturas verificadas, partición por lote y hashes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.request
import requests

import imagehash
from PIL import Image, ImageOps
import yaml

ANNOTATIONS = 'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root, workers=12):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    annotation_path = root / 'annotations.json'
    if not annotation_path.exists():
        urllib.request.urlretrieve(ANNOTATIONS, annotation_path)
    source = json.loads(annotation_path.read_text(encoding='utf-8'))
    originals = root / 'source'
    originals.mkdir(exist_ok=True)

    def download(info):
        path = originals / f"{info['id']:05d}.jpg"
        metadata_path=path.with_suffix('.json')
        url = info.get('flickr_640_url') or info['flickr_url']
        for attempt in range(4):
            try:
                if path.exists():
                    data=path.read_bytes()
                else:
                    response=requests.get(url,timeout=30)
                    response.raise_for_status()
                    data=response.content
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    # Algunos originales OpenLitterMap necesitan EXIF para coincidir
                    # con las dimensiones/anotaciones COCO. Verificar antes de transformar.
                    ratio = (rgb.width / rgb.height) / (info['width'] / info['height'])
                    if abs(ratio - 1) > .015:
                        rgb=ImageOps.exif_transpose(im).convert('RGB')
                        ratio=(rgb.width/rgb.height)/(info['width']/info['height'])
                        if abs(ratio-1)>.015:
                            raise ValueError(f"Aspect ratio inconsistent after EXIF: {info['file_name']}")
                    if not path.exists():
                        metadata_path.write_text(json.dumps({'downloaded_sha256':digest(data),'downloaded_url':url}),encoding='utf-8')
                        # Canonicalizar originales grandes, sin alterar coordenadas normalizadas.
                        if max(rgb.size)>640 or rgb.size!=im.size:
                            rgb.thumbnail((640,640),Image.Resampling.LANCZOS)
                            buffer=io.BytesIO();rgb.save(buffer,format='JPEG',quality=92)
                            data=buffer.getvalue()
                            rgb=Image.open(io.BytesIO(data)).convert('RGB')
                    pixel_hash = digest(rgb.tobytes())
                    perceptual = str(imagehash.phash(rgb))
                    dimensions = rgb.size
                if not path.exists():
                    path.write_bytes(data)
                download_metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {'downloaded_url':url,'downloaded_sha256':digest(data)}
                return dict(info, **download_metadata, sha256=digest(data), pixels_sha256=pixel_hash,
                            phash=perceptual, downloaded_size=dimensions)
            except Exception as exc:
                if attempt == 3:
                    return {'id': info['id'], 'error': str(exc)}
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(download, source['images']):
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"Descargadas/verificadas {len(rows)}/{len(source['images'])}", flush=True)
    failures = [r for r in rows if 'error' in r]
    (root/'download_errors.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} imágenes no verificadas; reejecuta para reintentar, sin partición parcial.')

    # Mantener lotes completos evita repartir series de una misma adquisición entre conjuntos.
    batches = sorted({r['file_name'].split('/')[0] for r in rows})
    parent = {b: b for b in batches}
    def find(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b
    def union(a,b):
        parent[find(a)] = find(b)
    pairs = []
    for i, first in enumerate(rows):
        for second in rows[:i]:
            distance = (int(first['phash'],16) ^ int(second['phash'],16)).bit_count()
            same = first['sha256'] == second['sha256'] or first['pixels_sha256'] == second['pixels_sha256']
            same_url = first['flickr_url'] == second['flickr_url']
            if same or same_url or distance <= 6:
                union(first['file_name'].split('/')[0], second['file_name'].split('/')[0])
                pairs.append({'first':first['id'],'second':second['id'],'phash_distance':distance,'exact':same})
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['file_name'].split('/')[0])].append(row)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    # Determinístico e independiente de las etiquetas y de cualquier resultado del detector.
    assignment, counts = {}, Counter()
    for key in keys:
        split = 'test' if counts['test'] < len(rows)*.13 else ('val' if counts['val'] < len(rows)*.13 else 'train')
        for row in groups[key]:
            assignment[row['id']] = split
        counts[split] += len(groups[key])
    if counts['train'] < len(rows)*.5 or min(counts['val'],counts['test']) < 100:
        raise ValueError(f'Insufficient independent groups: {counts}')
    annotations = defaultdict(list)
    for ann in source['annotations']:
        annotations[ann['image_id']].append(ann)
    category_counts = {s:Counter() for s in ('train','val','test')}
    cats = {c['id']:c['name'] for c in source['categories']}
    records = []
    clipped_annotations=[]
    for row in rows:
        split = assignment[row['id']]
        folder = root/'images'/split
        labels = root/'labels'/split
        folder.mkdir(parents=True,exist_ok=True)
        labels.mkdir(parents=True,exist_ok=True)
        name = f"{row['id']:05d}"
        (folder/f'{name}.jpg').write_bytes((originals/f'{name}.jpg').read_bytes())
        lines = []
        boxes = []
        for ann in annotations[row['id']]:
            x,y,w,h = ann['bbox']
            iw,ih = row['width'],row['height']
            if not (w>0 and h>0 and x>=-2 and y>=-2 and x+w<=iw+2 and y+h<=ih+2):
                raise ValueError(f"Invalid source box {ann['id']}")
            x2,y2 = min(iw,x+w),min(ih,y+h)
            if x<0 or y<0 or x+w>iw or y+h>ih:
                clipped_annotations.append({'annotation_id':ann['id'],'original_bbox':ann['bbox']})
            x,y=max(0,x),max(0,y)
            lines.append(f'0 {(x+x2)/2/iw:.8f} {(y+y2)/2/ih:.8f} {(x2-x)/iw:.8f} {(y2-y)/ih:.8f}')
            boxes.append({'annotation_id':ann['id'],'category':cats[ann['category_id']],
                          'original_bbox':ann['bbox'],'xyxy_normalized':[x/iw,y/ih,x2/iw,y2/ih]})
            category_counts[split][cats[ann['category_id']]] += 1
        (labels/f'{name}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        records.append(dict(row,split=split,boxes=boxes,group=find(row['file_name'].split('/')[0])))
    for pair in pairs:
        assert assignment[pair['first']] == assignment[pair['second']]
    report = {'source':ANNOTATIONS,'annotation_sha256':digest(annotation_path.read_bytes()),
              'images':len(rows),'objects':len(source['annotations']),'splits':dict(counts),
              'category_counts':{s:dict(c) for s,c in category_counts.items()},'seed':42,
              'split_method':'source batches, union of byte/pixel/URL matches and pHash distance <=6',
              'duplicate_candidates':pairs,'cross_split_candidates':0,
              'boundary_clipping_max_2_pixels':clipped_annotations,
              'license_counts':dict(Counter(str(r.get('license')) for r in rows)),
              'annotation_license':'CC BY 4.0; Pedro F. Proença and Pedro Simões, TACO (2020)',
              'image_license_policy':'Preserve original license and URL per image. Missing license: default CC BY 4.0 according to TACO terms; CC unspecified and ODbL retained verbatim. Do not redistribute images without source terms.',
              'license_source':'https://huggingface.co/datasets/roboticslaburjc/ExtendedTACO-reviewed',
              'toolkit_license':'MIT, https://github.com/pedropro/TACO/blob/master/LICENSE'}
    (root/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (root/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for final in (False,True):
        data = {'path':str(root),'train':'images/train','val':'images/val','names':{0:'residuo'}}
        if final:
            data['test'] = 'images/test'
        (root/('final.yaml' if final else 'development.yaml')).write_text(yaml.safe_dump(data),encoding='utf-8')
    print(json.dumps(report['splits']),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/detection/taco')
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    prepare(a.root,a.workers)
