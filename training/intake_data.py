"""Lectura COCO para clasificación: recortes humanos/autorizados, sin tocar test ni escribir datos."""
from collections import Counter, OrderedDict, defaultdict
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image
from torch.utils.data import Dataset

from training.compact import build_transform
from utils.config import CLASSES, PROJECT_ROOT


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _original_name(image):
    name = image.get('extra', {}).get('name', image['file_name'])
    return re.sub(r'_(?:jpg|jpeg|png)\.rf\.[^.]+\.(?:jpg|jpeg|png)$', '', Path(name).name,
                  flags=re.IGNORECASE)


def _source_group(image, document, material):
    source = image.get('source_dataset') or document.get('info', {}).get('url') or material
    return str(source)+'/'+_original_name(image)


def reserved_test_groups(root):
    """Solo identidades del COCO test; nunca abre, decodifica ni evalúa sus fotografías."""
    root = Path(root).resolve()
    groups = set()
    for material in CLASSES:
        metadata = root/material/'test'/'_annotations.coco.json'
        document = _load(metadata)
        groups.update(_source_group(image, document, material) for image in document['images'])
    return groups


def reserve_partitions(root, train_rows, val_rows):
    """Una familia test no entra a desarrollo; una familia val no entra a train.

    La selección está en memoria. No se reasigna ni elimina ningún archivo.
    """
    test_groups = reserved_test_groups(root)
    original_train, original_val = train_rows, val_rows
    val_rows = [r for r in val_rows if r['group'] not in test_groups]
    val_groups = {r['group'] for r in val_rows}
    train_rows = [r for r in train_rows if r['group'] not in test_groups and r['group'] not in val_groups]
    stats = {'train_samples_reserved_for_test': sum(r['group'] in test_groups for r in original_train),
             'val_samples_reserved_for_test': sum(r['group'] in test_groups for r in original_val),
             'train_samples_reserved_for_val': sum(r['group'] in val_groups for r in original_train),
             'test_metadata_groups': len(test_groups)}
    return train_rows, val_rows, stats, test_groups


def _historical_reference(config):
    if not config.get('protect_historical_test', True):
        return {}, set()
    path = Path(config.get('historical_test_reference', 'data/reference/B/manifest.csv'))
    if not path.is_absolute():
        path = PROJECT_ROOT/path
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return ({r['source_id']: r for r in rows}, {r['group_id'] for r in rows if r['split'] == 'test'})


def collect_material_rows(root, split, config, policy=None, *, final_test=False):
    if split not in ('train', 'val') and not (split == 'test' and final_test is True):
        raise ValueError('El entrenamiento solo lee train y val; test permanece reservado.')
    root = Path(root).resolve()
    if policy is None:
        policy_path = PROJECT_ROOT/'config/material_intake_policy.json'
        policy = _load(policy_path) if policy_path.exists() else {}
    exceptions = {r['source'] for r in policy.get('annotation_source_exceptions', [])
                  if r.get('allowed_for_current_training') is True}
    historical, reserved_groups = _historical_reference(config)
    rows, fingerprints = [], []
    stats = Counter()
    for material in CLASSES:
        folder = root/material/split
        if folder.resolve() != folder.absolute():
            raise ValueError('No se admiten enlaces que cambien la partición física.')
        annotation_file = folder/'_annotations.coco.json'
        raw = annotation_file.read_bytes()
        document = json.loads(raw.decode('utf-8-sig'))
        fingerprints.append((str(annotation_file), hashlib.sha256(raw).hexdigest()))
        images = {i['id']: i for i in document['images']}
        if len(images) != len(document['images']):
            raise ValueError('IDs de imagen repetidos en COCO.')
        categories = {c['id']: c['name'].strip().lower() for c in document['categories']}
        annotations = defaultdict(list)
        annotation_ids = set()
        for annotation in document['annotations']:
            if (annotation['image_id'] not in images or annotation['category_id'] not in categories or
                    annotation['id'] in annotation_ids):
                raise ValueError('Referencias o IDs de anotaciones COCO inválidos.')
            annotation_ids.add(annotation['id'])
            annotations[annotation['image_id']].append(annotation)
        for image in document['images']:
            stats['source_images'] += 1
            name = _original_name(image)
            original_id = Path(name).stem
            historical_row = historical.get(original_id)
            if split != 'test' and historical_row and historical_row['group_id'] in reserved_groups:
                stats['historical_test_images_skipped'] += 1
                continue
            origin = image.get('annotation_origin', '').lower().replace(' ', '')
            if 'sam3' in origin and image.get('source_dataset') not in exceptions:
                stats['unapproved_annotation_images_skipped'] += 1
                continue
            path = (folder/image['file_name']).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                raise ValueError(f'Imagen fuera de su partición o ausente: {image["file_name"]}')
            width, height = image['width'], image['height']
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (width, height)):
                raise ValueError('Dimensiones COCO inválidas.')
            group = _source_group(image, document, material)
            stat = path.stat()
            common = {'path': str(path), 'group': group, 'split': split, 'width': width, 'height': height,
                      'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            current = annotations[image['id']]
            if current:
                for annotation in current:
                    label = categories[annotation['category_id']]
                    box = annotation.get('bbox', [])
                    if label not in CLASSES or len(box) != 4 or any(
                            not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                        raise ValueError('Etiqueta o caja fuera del contrato de seis materiales.')
                    x, y, w, h = box
                    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width+.01 or y+h > height+.01:
                        raise ValueError('Caja fuera de sus dimensiones originales.')
                    rows.append(dict(common, label=CLASSES.index(label), crop=box, annotation_id=annotation['id']))
                    stats['box_crops'] += 1
            elif image.get('annotation_status') == 'image_label_only' and image.get('material_label') in CLASSES:
                rows.append(dict(common, label=CLASSES.index(image['material_label']), crop=None, annotation_id=None))
                stats['image_label_samples'] += 1
            else:
                stats['unverified_unboxed_images_skipped'] += 1
    if not rows:
        raise ValueError(f'No hay muestras aprobadas para {split}.')
    counts = Counter(CLASSES[r['label']] for r in rows)
    fingerprint = hashlib.sha256(json.dumps({'annotations': fingerprints, 'rows': rows},
                                            sort_keys=True).encode()).hexdigest()
    return rows, {'fingerprint': fingerprint, 'samples': len(rows), 'class_counts': dict(counts), **dict(stats)}


def assert_disjoint(train_rows, val_rows):
    for key in ('path', 'group'):
        if {r[key] for r in train_rows} & {r[key] for r in val_rows}:
            raise ValueError(f'Una misma foto aparece en train y val ({key}).')


class IntakeMaterialDataset(Dataset):
    def __init__(self, rows, config, training=False):
        self.rows = rows
        self.transform = build_transform(config, training)
        self.margin = float(config.get('crop_margin', .12))
        self.cache_limit = int(config.get('cache_max_bytes', 128*1024*1024))
        if not 0 <= self.margin <= .5 or self.cache_limit < 0:
            raise ValueError('Margen o límite de caché inválido.')
        self.cache, self.cache_bytes = OrderedDict(), 0

    def __len__(self):
        return len(self.rows)

    def _bytes(self, row):
        path = Path(row['path'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['size'], row['mtime_ns']):
            raise ValueError('Una imagen cambió desde la preparación; no continuar el entrenamiento.')
        if row['path'] in self.cache:
            value = self.cache.pop(row['path'])
            self.cache[row['path']] = value
            return value
        value = path.read_bytes()
        if len(value) <= self.cache_limit:
            while self.cache and self.cache_bytes+len(value) > self.cache_limit:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= len(removed)
            self.cache[row['path']] = value
            self.cache_bytes += len(value)
        return value

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(BytesIO(self._bytes(row))) as source:
            source.load()
            if source.size != (row['width'], row['height']) or source.getexif().get(274, 1) != 1:
                raise ValueError('Dimensiones u orientación distintas de las anotaciones.')
            image = source.convert('RGB')
        if row['crop'] is not None:
            x, y, w, h = row['crop']
            side = max(w, h)*(1+2*self.margin)
            cx, cy = x+w/2, y+h/2
            box = (max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
                   min(image.width, math.ceil(cx+side/2)), min(image.height, math.ceil(cy+side/2)))
            image = image.crop(box)
        return self.transform(image), row['label']
