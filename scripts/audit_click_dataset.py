"""Auditoría local de imágenes/COCO/YOLO de CLICK; sin modelos ni entrenamiento."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import imagehash
from PIL import Image
import yaml

from utils.config import PROJECT_ROOT
from utils.quality_route import file_hash, read_json, write_new_json


def finite(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def polygon_area(points):
    return abs(sum(x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
                   for i, (x, y) in enumerate(points))) / 2


def inspect_photo(task):
    path, split, entry = task
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            orientation = image.getexif().get(274, 1)
            rgb = image.convert('RGB')
            pixels = rgb.tobytes()
            raw_hash = hashlib.sha256(pixels).hexdigest()
            with_size = hashlib.sha256(str(rgb.size).encode() + b'\0' + pixels).hexdigest()
            del pixels
            gray = rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
            operations = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                          Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
                          Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]
            hashes = [str(imagehash.phash(gray if operation is None else gray.transpose(operation))) for operation in operations]
            return {'file_name': path.name, 'split': split, 'sha256': file_hash(path),
                    'pixel_sha256': raw_hash, 'pixel_sha256_with_size': with_size,
                    'width': rgb.width, 'height': rgb.height, 'exif_orientation': orientation,
                    'dimensions_match_coco': (rgb.width, rgb.height) == (entry['width'], entry['height']),
                    'phash': hashes[0], 'phash_variants': hashes}, None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, {'path': str(path), 'error': str(exc)}


def audit(root, output, workers=2):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    errors, warnings, metadata, feature_tasks = [], [], {}, []
    yaml_results, per_class, variant_summaries = {}, {}, {}
    annotation_fingerprints = {}
    max_yolo_error = 0.
    for variant in ('multi class', 'single class'):
        base = root / variant
        names = [s.strip() for s in (base / 'classes.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        settings = yaml.safe_load((base / 'data.yaml').read_text(encoding='utf-8-sig'))
        configured = settings['names']
        configured = [configured[i] for i in sorted(configured)] if isinstance(configured, dict) else configured
        if names != configured or len(names) != settings['nc']:
            errors.append({'variant': variant, 'error': 'classes.txt y YAML no coinciden'})
        yaml_results[variant] = {'classes': names, 'paths': {s: str((base / settings[s]).resolve()) for s in ('train', 'val', 'test')}}
        variant_summaries[variant] = {}
        for split in ('train', 'val', 'test'):
            source_path = base / 'annotations' / f'instances_{split}.json'
            document = read_json(source_path)
            annotation_fingerprints[f'{variant}/{split}'] = file_hash(source_path)
            images, annotations = document['images'], document['annotations']
            ids = [r['id'] for r in images]
            annotation_ids = [r['id'] for r in annotations]
            if len(set(ids)) != len(ids) or len(set(annotation_ids)) != len(annotation_ids):
                errors.append({'variant': variant, 'split': split, 'error': 'IDs COCO duplicados'})
            by_id = {r['id']: r for r in images}
            categories = {r['id']: r['name'] for r in document['categories']}
            if set(categories.values()) != set(names):
                errors.append({'variant': variant, 'split': split, 'error': 'Categorías COCO distintas de classes.txt'})
            by_image, counts, geometry = defaultdict(list), Counter(), Counter()
            boxes_seen, geometry_classes = set(), defaultdict(set)
            for annotation in annotations:
                entry = by_id.get(annotation['image_id'])
                if entry is None or annotation['category_id'] not in categories:
                    errors.append({'variant': variant, 'split': split, 'annotation': annotation['id'], 'error': 'Referencia COCO inexistente'})
                    continue
                label = categories[annotation['category_id']]
                counts[label] += 1
                by_image[entry['id']].append(annotation)
                box = annotation.get('bbox', [])
                reason = None
                if len(box) != 4 or not finite(box):
                    reason = 'Caja inválida/no finita'
                else:
                    x, y, w, h = box
                    if min(w, h) <= 0 or x < -.05 or y < -.05 or x + w > entry['width'] + .05 or y + h > entry['height'] + .05:
                        reason = 'Caja degenerada/fuera de imagen'
                vertices, areas = [], []
                polygons = annotation.get('segmentation')
                if not isinstance(polygons, list) or not polygons:
                    reason = 'No hay polígonos COCO'
                else:
                    for polygon in polygons:
                        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 or not finite(polygon):
                            reason = 'Polígono inválido'
                            continue
                        points = list(zip(polygon[::2], polygon[1::2]))
                        if any(x < -.05 or y < -.05 or x > entry['width'] + .05 or y > entry['height'] + .05 for x, y in points):
                            reason = 'Polígono fuera de imagen'
                        vertices.extend(points)
                        areas.append(polygon_area(points))
                if vertices and len(box) == 4 and finite(box):
                    bounds = [min(v[0] for v in vertices), min(v[1] for v in vertices),
                              max(v[0] for v in vertices) - min(v[0] for v in vertices),
                              max(v[1] for v in vertices) - min(v[1] for v in vertices)]
                    if max(abs(a - b) for a, b in zip(box, bounds)) > .05:
                        reason = 'Caja distinta de límites del polígono'
                    area = annotation.get('area')
                    if not finite([area]) or area <= 0 or abs(sum(areas) - area) > max(1., sum(areas) * .001):
                        reason = 'Área distinta del polígono/degenerada'
                if reason:
                    errors.append({'variant': variant, 'split': split, 'annotation': annotation['id'], 'error': reason})
                key = (entry['file_name'], tuple(round(v, 3) for v in box))
                if (key, label) in boxes_seen:
                    errors.append({'variant': variant, 'split': split, 'annotation': annotation['id'], 'error': 'Caja duplicada'})
                boxes_seen.add((key, label));geometry_classes[key].add(label)
                geometry[(entry['file_name'], json.dumps({'bbox': box, 'segmentation': polygons, 'area': annotation.get('area')}, sort_keys=True))] += 1
            conflicts = sum(len(labels) > 1 for labels in geometry_classes.values())
            if conflicts:
                errors.append({'variant': variant, 'split': split, 'error': 'Cajas iguales con clases distintas', 'count': conflicts})
            declared = [r['file_name'] for r in images]
            actual = {p.name for p in (root / 'images' / split).iterdir() if p.is_file()}
            if len(set(declared)) != len(declared) or set(declared) != actual:
                errors.append({'variant': variant, 'split': split, 'error': 'Fotos COCO ausentes/extra/duplicadas',
                               'missing': sorted(set(declared) - actual), 'extra': sorted(actual - set(declared))})
            labels_dir = base / 'labels' / split
            expected_label_names = {Path(n).stem + '.txt' for n in declared}
            if expected_label_names != {p.name for p in labels_dir.glob('*.txt')}:
                errors.append({'variant': variant, 'split': split, 'error': 'Etiquetas YOLO ausentes/extra'})
            for entry in images:
                name = entry['file_name']
                if Path(name).name != name or min(entry['width'], entry['height']) <= 0:
                    errors.append({'variant': variant, 'split': split, 'file': name, 'error': 'Ruta/dimensiones inválidas'})
                    continue
                path = labels_dir / (Path(name).stem + '.txt')
                if not path.is_file():
                    continue
                labels = []
                for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        fields = line.split()
                        if len(fields) != 5:
                            raise ValueError('Se esperan cinco valores YOLO')
                        category, box = int(fields[0]), [float(v) for v in fields[1:]]
                        if not 0 <= category < len(names) or not finite(box) or not all(0 <= v <= 1 for v in box) or min(box[2:]) <= 0:
                            raise ValueError('Clase/coordenadas YOLO inválidas')
                        cx, cy, w, h = box
                        if min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
                            raise ValueError('Caja YOLO fuera de imagen')
                        labels.append((names[category], box))
                    except (ValueError, IndexError) as exc:
                        errors.append({'variant': variant, 'split': split, 'file': name, 'line': number, 'error': str(exc)})
                expected = []
                for ann in by_image[entry['id']]:
                    x, y, w, h = ann['bbox']
                    expected.append((categories[ann['category_id']], [(x+w/2)/entry['width'], (y+h/2)/entry['height'], w/entry['width'], h/entry['height']]))
                if len(labels) != len(expected):
                    errors.append({'variant': variant, 'split': split, 'file': name, 'error': 'Cantidad YOLO/COCO distinta'})
                for label, box in labels:
                    possible = [(max(abs(a-b) for a, b in zip(box, target)), i) for i, (category, target) in enumerate(expected) if category == label]
                    distance, best = min(possible, default=(float('inf'), None))
                    if distance > 1e-4:
                        errors.append({'variant': variant, 'split': split, 'file': name, 'error': 'YOLO/COCO no coinciden'})
                    else:
                        max_yolo_error = max(max_yolo_error, distance);expected.pop(best)
            metadata[(variant, split)] = {'images': images, 'geometry': geometry}
            if variant == 'multi class':
                per_class[split] = {name: counts[name] for name in names}
                if split in ('train', 'val'):
                    feature_tasks.extend((root/'images'/split/r['file_name'], split, r) for r in images)
            variant_summaries[variant][split] = {'images': len(images), 'objects': len(annotations),
                                               'empty_images': sum(not by_image[r['id']] for r in images),
                                               'licenses_metadata': document.get('licenses', [])}
    for split in ('train', 'val', 'test'):
        a, b = metadata[('multi class', split)], metadata[('single class', split)]
        if a['images'] != b['images'] or a['geometry'] != b['geometry']:
            errors.append({'split': split, 'error': 'Las dos versiones no describen las mismas imágenes/geometría'})
    split_names = {s: {r['file_name'] for r in metadata[('multi class', s)]['images']} for s in ('train', 'val', 'test')}
    for a, b in combinations(split_names, 2):
        shared = sorted(split_names[a] & split_names[b])
        if shared:
            errors.append({'error': 'Nombres repetidos entre particiones', 'splits': [a, b], 'files': shared})
    features, corrupt = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (row, error) in enumerate(pool.map(inspect_photo, feature_tasks), 1):
            if row:features.append(row)
            if error:corrupt.append(error)
            if index % 100 == 0:print(f'Fotos train/val verificadas: {index}/{len(feature_tasks)}', flush=True)
    dimensions = [r['file_name'] for r in features if not r['dimensions_match_coco']]
    orientations = [r['file_name'] for r in features if r['exif_orientation'] != 1]
    exact = {}
    for kind in ('sha256', 'pixel_sha256'):
        groups = defaultdict(list)
        for row in features:groups[row[kind]].append({'split': row['split'], 'file_name': row['file_name']})
        exact[kind] = [group for group in groups.values() if len(group) > 1]
    near = []
    train, val = ([r for r in features if r['split'] == s] for s in ('train', 'val'))
    for a in train:
        hashes = [int(v, 16) for v in a['phash_variants']]
        for b in val:
            distance = min((p ^ int(b['phash'], 16)).bit_count() for p in hashes)
            if distance <= 6:
                near.append({'train': a['file_name'], 'val': b['file_name'], 'phash_distance': distance})
    photo_report = {'decoded_train_val': len(features), 'corrupt': corrupt, 'dimension_mismatches': dimensions,
                    'nonidentity_exif_orientation': orientations, 'exact_duplicates': exact,
                    'cross_split_near_duplicate_candidates': near, 'test_photos_decoded_or_hashed': 0}
    if corrupt or dimensions or orientations:
        errors.append({'error': 'Problemas de integridad/dimensiones/orientación en fotos', 'details': photo_report})
    warnings.extend(['No se abrieron ni se calcularon hashes de fotos test; se revisaron sus metadatos y etiquetas.',
                     'La similitud perceptual no acredita por sí sola que dos fotos sean el mismo objeto/sesión.',
                     'No hay licencia, README ni exportación de Label Studio en la carpeta local.',
                     'La fuente pública de CLICK declara propuestas SAM 3 con revisión humana; no cumple la prohibición actual de anotaciones de modelos fundacionales.'])
    all_files = [p for p in root.rglob('*') if p.is_file()]
    result = {'dataset_root': str(root), 'files': len(all_files), 'bytes': sum(p.stat().st_size for p in all_files),
              'variants': variant_summaries, 'yaml': yaml_results, 'class_counts': per_class,
              'annotations_sha256': annotation_fingerprints, 'max_yolo_coco_error_normalized': max_yolo_error,
              'structural_errors': errors, 'photos': photo_report, 'warnings': warnings,
              'training_started': False, 'eligible_under_current_project_rules': False,
              'annotation_origin_source': 'https://data.mendeley.com/datasets/htcvvz6fm2/2',
              'annotation_origin_correspondence': 'Matches published structure, exact counts, class names and example labels; no claim of a release-wide cryptographic identity.'}
    write_new_json(output/'audit.json', result)
    write_new_json(output/'train_val_fingerprints.json', features)
    with (output/'class_counts.csv').open('x', encoding='utf-8', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['class','train','val','test'])
        for label in yaml_results['multi class']['classes']:writer.writerow([label]+[per_class[s][label] for s in ('train','val','test')])
    print(json.dumps({'files': result['files'], 'GiB': round(result['bytes']/2**30, 3),
                      'structural_errors': len(errors), 'photos_checked': len(features), 'corrupt_photos': len(corrupt),
                      'exact_duplicate_groups': {k: len(v) for k,v in exact.items()}, 'near_train_val_pairs': len(near),
                      'eligible_under_current_project_rules': False, 'report': str(output/'audit.json')}, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1,5), default=2)
    args=parser.parse_args()
    audit(args.root,args.output,args.workers)
