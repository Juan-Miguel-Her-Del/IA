"""Organiza copias de fotos por material, conserva cajas y protege test de todas las fuentes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import imagehash
from PIL import Image

from scripts.group_click_materials import image_group
from utils.config import CLASSES, PROJECT_ROOT


LABELS = {'plastic': 'plastico', 'paper': 'papel', 'cardboard': 'carton',
          'glass': 'vidrio', 'metal': 'metal', 'trash': 'revision'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items() if key in fields})


def contained_path(folder, name):
    folder = Path(folder).resolve()
    path = (folder / name).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f'Ruta fuera de la fuente: {name}')
    return path


def coco_errors(document):
    errors = []
    images = {r['id']: r for r in document['images']}
    categories = {r['id']: r['name'] for r in document['categories']}
    if len(images) != len(document['images']):
        errors.append('image_id duplicado')
    if len(categories) != len(document['categories']):
        errors.append('category_id duplicado')
    ids = [r['id'] for r in document['annotations']]
    if len(set(ids)) != len(ids):
        errors.append('annotation_id duplicado')
    for entry in document['images']:
        if any(not isinstance(entry.get(k), (int, float)) or not math.isfinite(entry[k]) or entry[k] <= 0
               for k in ('width', 'height')):
            errors.append(f'dimensiones inválidas: {entry["id"]}')
    for ann in document['annotations']:
        if ann['image_id'] not in images or ann['category_id'] not in categories:
            errors.append(f'referencia inválida: {ann["id"]}')
            continue
        box = ann.get('bbox', [])
        if len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            errors.append(f'caja inválida: {ann["id"]}')
            continue
        x, y, w, h = box
        im = images[ann['image_id']]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > im['width'] + .01 or y + h > im['height'] + .01:
            errors.append(f'caja fuera de límites: {ann["id"]}')
        area = ann.get('area', w * h)
        if not isinstance(area, (int, float)) or not math.isfinite(area) or area < 0:
            errors.append(f'área inválida: {ann["id"]}')
    return errors


def historical_index(reference, raw_root):
    with Path(reference).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    refs = {r['source_id']: r for r in rows if r['dataset'] == 'TrashNet'}
    by_hash = {r['sha256']: r for r in rows if r['dataset'] == 'TrashNet'}
    # El nombre del archivo raw ya conserva su SHA; no se abren fotos test para resolver alias.
    for path in Path(raw_root).rglob('*.jpg'):
        digest, separator, source_id = path.stem.partition('_')
        if separator and source_id not in refs and digest in by_hash:
            refs[source_id] = by_hash[digest]
    return rows, refs


def protected_groups(reference_rows, refs, newly_reserved_ids):
    groups = {r['group_id'] for r in reference_rows if r['split'] == 'test'}
    groups.update(refs[k]['group_id'] for k in newly_reserved_ids if k in refs)
    hashes = {r['sha256'] for r in reference_rows if r['group_id'] in groups}
    pixels = {r['pixel_sha256'] for r in reference_rows if r['group_id'] in groups}
    return groups, hashes, pixels


def inspect_image(path):
    path = Path(path)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            orientation = im.getexif().get(274, 1)
            rgb = im.convert('RGB')
            gray = rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
            operations = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                          Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
                          Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
                          Image.Transpose.TRANSVERSE]
            return str(path), {'sha256': sha256(path), 'width': rgb.width, 'height': rgb.height,
                               'orientation': orientation,
                               'pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                               'phash_variants': [str(imagehash.phash(gray if op is None else gray.transpose(op)))
                                                  for op in operations]}
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return str(path), {'error': str(exc)}


def read_exports(root):
    exports, audits = [], []
    for folder in sorted(Path(root).glob('*.coco')):
        for path in sorted(folder.glob('*/_annotations.coco.json')):
            document = read_json(path)
            errors = coco_errors(document)
            by_image = defaultdict(list)
            categories = {r['id']: r['name'] for r in document['categories']}
            for ann in document['annotations']:
                by_image[ann['image_id']].append(ann)
            full = duplicates = 0
            for im in document['images']:
                current = by_image[im['id']]
                boxes = [tuple(ann['bbox']) for ann in current]
                duplicates += len(boxes) - len(set(boxes))
                full += sum(list(box) == [0, 0, im['width'], im['height']] for box in boxes)
                source_path = contained_path(path.parent, im['file_name'])
                if not source_path.is_file():
                    errors.append(f'imagen ausente: {im["file_name"]}')
                original = im.get('extra', {}).get('name', im['file_name'])
                exports.append({'dataset': folder.name, 'annotation_file': str(path.resolve()),
                                'annotation_sha256': sha256(path), 'path': str(source_path),
                                'original_id': Path(original).stem, 'original_name': original,
                                'export_split': 'val' if path.parent.name == 'valid' else path.parent.name,
                                'image': im, 'annotations': current, 'categories': categories,
                                'metadata_errors': errors})
            audits.append({'dataset': folder.name, 'split': path.parent.name,
                           'annotation_file': str(path.resolve()), 'sha256': sha256(path),
                           'images': len(document['images']), 'boxes': len(document['annotations']),
                           'images_without_boxes': sum(not by_image[im['id']] for im in document['images']),
                           'whole_image_boxes': full, 'duplicate_boxes': duplicates,
                           'info': document.get('info'), 'licenses': document.get('licenses'),
                           'categories': document['categories'], 'errors': errors})
    return exports, audits


def organize(args):
    policy_path = PROJECT_ROOT/'config/material_intake_policy.json'
    policy = read_json(policy_path) if policy_path.exists() else {}
    if policy.get('reports_enabled') is False:
        raise ValueError('El usuario pidió un dataset limpio, sin reportes ni auxiliares. No regenerar esta organización histórica.')
    if policy.get('partition_policy') == 'preserve_source_splits':
        raise ValueError('El usuario exige conservar train/val/test originales. No usar el organizador histórico B; '
                         'la estructura corregida se documenta con scripts.restore_intake_source_splits.')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'La salida ya existe; usa otra versión: {output}')
    exports, export_audit = read_exports(args.source)
    reference_rows, refs = historical_index(PROJECT_ROOT/'data/reference/B/manifest.csv',
                                            PROJECT_ROOT/'data/raw/trashnet')
    reserved_ids = {r['original_id'] for r in exports if r['export_split'] == 'test'}
    protected, reserved_hashes, reserved_pixels = protected_groups(reference_rows, refs, reserved_ids)
    raw_records = {}
    for folder in sorted(args.trashnet.iterdir()):
        if not folder.is_dir() or folder.name not in LABELS:
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            ref = refs.get(path.stem)
            key = ref['source_id'] if ref else path.stem
            row = raw_records.setdefault(key, {'identity': 'TrashNet/'+key, 'source_id': key,
                                              'group': LABELS[folder.name], 'raw_paths': [],
                                              'ref': ref, 'exports': []})
            if row['group'] != LABELS[folder.name]:
                raise ValueError(f'Etiquetas contradictorias para {key}')
            row['raw_paths'].append(str(path.resolve()))
    for export in exports:
        ref = refs.get(export['original_id'])
        key = ref['source_id'] if ref else export['original_id']
        if key not in raw_records:
            raise ValueError(f'Falta el original para verificar {export["original_id"]}')
        raw_records[key]['exports'].append(export)
    for row in raw_records.values():
        ref = row['ref']
        row['split'] = ref['split'] if ref else 'sin_dividir'
        row['reserved'] = bool((ref and ref['group_id'] in protected) or
                               any(e['export_split'] == 'test' for e in row['exports']))
        row['reserved_reason'] = ('test histórico o grupo de foto reservado en exportaciones nuevas'
                                  if row['reserved'] else '')
    # Esta lista se construye antes de abrir imágenes: todas las variantes de un test quedan fuera.
    tasks = set()
    for row in raw_records.values():
        if row['reserved']:
            continue
        tasks.update(row['raw_paths'])
        tasks.update(e['path'] for e in row['exports'])
    features = {}
    print(f'Verificando {len(tasks)} archivos nuevos no reservados...', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, (path, feature) in enumerate(pool.map(inspect_image, sorted(tasks)), 1):
            features[path] = feature
            if number % 300 == 0:
                print(f'Integridad: {number}/{len(tasks)}', flush=True)

    mapping = read_json(PROJECT_ROOT/'config/click_material_mapping.json')['classes']
    catalog_root = PROJECT_ROOT/'data/review/click_material_groups_v1'
    old_summary = read_json(catalog_root/'summary.json')
    for split in ('train', 'val', 'test'):
        path = args.source/'multi class/annotations'/f'instances_{split}.json'
        if sha256(path) != old_summary['annotations_sha256'][f'multi class/{split}']:
            raise ValueError('CLICK cambió desde la auditoría: requiere nueva auditoría antes de copiar.')
    fingerprints = {(r['split'], r['file_name']): r for r in read_json(
        PROJECT_ROOT/'runs/dataset_audits/click_20261124/train_val_fingerprints.json')}
    click = read_json(catalog_root/'catalog_train_val.json')
    rows, warnings = [], []
    for source in raw_records.values():
        ref = source['ref']
        row = {'identity': source['identity'], 'source': 'TrashNet', 'group': source['group'],
               'split': 'test_reservado' if source['reserved'] else source['split'],
               'original_paths': source['raw_paths'], 'export_variants': len(source['exports']),
               'reserved': source['reserved'], 'reserved_reason': source['reserved_reason'],
               'existing_in_project': bool(ref), 'historical_split': ref['split'] if ref else '',
               'approved_for_training': False, 'annotation_origin': 'etiqueta humana de TrashNet',
               'copy_path': '', 'boxes': [], 'issues': []}
        if not ref and row['group'] in CLASSES:
            row['proposed_material'] = row['group']
            row['group'] = 'revision'
            row['issues'].append('sin correspondencia en manifiesto B; revisar posibles conflictos de etiqueta')
        if source['reserved']:
            rows.append(row)
            continue
        raw_path = source['raw_paths'][0]
        raw = features[raw_path]
        if 'error' in raw:
            row['issues'].append('original corrupto: '+raw['error'])
            row['group'] = 'revision'
            rows.append(row)
            continue
        if ref and raw['sha256'] != ref['sha256']:
            row['issues'].append('bytes originales distintos del manifiesto histórico')
        row.update(source_path=raw_path, sha256=raw['sha256'], width=raw['width'], height=raw['height'],
                   pixel_sha256=raw['pixel_sha256'])
        if raw['sha256'] in reserved_hashes or raw['pixel_sha256'] in reserved_pixels:
            row.update(reserved=True, split='test_reservado', reserved_reason='coincidencia exacta con test')
            rows.append(row)
            continue
        for export in source['exports']:
            feat = features[export['path']]
            if 'error' in feat:
                row['issues'].append('exportación corrupta: '+export['path'])
                continue
            im = export['image']
            if (feat['width'], feat['height']) != (im['width'], im['height']) or feat['orientation'] != 1:
                row['issues'].append('dimensiones u orientación incorrectas: '+export['path'])
                continue
            distance = min((int(raw['phash_variants'][0], 16) ^ int(v, 16)).bit_count()
                           for v in feat['phash_variants'])
            if distance > 6:
                row['issues'].append(f'coincidencia visual pendiente ({distance}): '+export['path'])
                continue
            if export['metadata_errors']:
                row['issues'].append('errores COCO: '+export['annotation_file'])
                continue
            if not export['annotations']:
                row['issues'].append('exportación sin cajas pese a etiqueta de material: '+export['path'])
                continue
            row.update(source_path=export['path'], sha256=feat['sha256'], width=feat['width'],
                       height=feat['height'], pixel_sha256=feat['pixel_sha256'],
                       annotation_origin='cajas humanas confirmadas por el usuario' if args.manual_boxes_confirmed
                                         else 'origen de cajas pendiente',
                       annotation_file=export['annotation_file'], export_split=export['export_split'])
            row['boxes'] = [{'annotation_id': ann['id'], 'source_class': export['categories'][ann['category_id']],
                             'material': row['group'] if row['group'] in CLASSES else None, 'bbox': ann['bbox'],
                             'segmentation': ann.get('segmentation', [])} for ann in export['annotations']]
            if source['split'] != 'sin_dividir' and export['export_split'] != source['split']:
                row['issues'].append('se conserva la partición histórica, distinta de la exportación nueva')
            break
        if source['group'] == 'revision':
            row['issues'].append('trash no equivale a orgánicos; requiere revisión humana')
        rows.append(row)
    for photo in click:
        feat = fingerprints[photo['split'], photo['file_name']]
        group, materials, unresolved = image_group([o['source_class'] for o in photo['objects']], mapping)
        reserved = feat['sha256'] in reserved_hashes or feat['pixel_sha256'] in reserved_pixels
        rows.append({'identity': f'CLICK/{photo["split"]}/{photo["image_id"]}', 'source': 'CLICK',
                     'group': group, 'split': 'test_reservado' if reserved else photo['split'],
                     'source_path': photo['source_path'], 'original_paths': [photo['source_path']],
                     'sha256': feat['sha256'], 'pixel_sha256': feat['pixel_sha256'],
                     'width': photo['width'], 'height': photo['height'], 'reserved': reserved,
                     'reserved_reason': 'coincidencia exacta con test histórico' if reserved else '',
                     'existing_in_project': True, 'approved_for_training': False,
                     'annotation_origin': 'SAM 3 seguido de revisión humana, según autor',
                     'boxes': photo['objects'], 'copy_path': '',
                     'issues': ['origen de anotaciones excluido por política actual'] +
                               (['material pendiente: '+', '.join(unresolved)] if unresolved else [])})
    click_test = read_json(args.source/'multi class/annotations/instances_test.json')
    by_image = defaultdict(list)
    for ann in click_test['annotations']:
        by_image[ann['image_id']].append(ann)
    categories = {r['id']: r['name'] for r in click_test['categories']}
    for im in click_test['images']:
        group, _, _ = image_group([categories[a['category_id']] for a in by_image[im['id']]], mapping)
        rows.append({'identity': f'CLICK/test/{im["id"]}', 'source': 'CLICK', 'group': group,
                     'split': 'test_reservado', 'reserved': True, 'reserved_reason': 'test CLICK',
                     'original_paths': [str(args.source/'images/test'/im['file_name'])],
                     'copy_path': '', 'approved_for_training': False, 'issues': []})
    policy_path = PROJECT_ROOT/'config/material_intake_policy.json'
    policy = read_json(policy_path) if policy_path.exists() else {}
    required = set(policy.get('require_human_boxes_for_materials', []))
    for row in rows:
        if row['group'] in required and not row['reserved'] and not (
                row.get('annotation_origin') == 'cajas humanas confirmadas por el usuario' and row.get('boxes')):
            row['excluded'] = True
            row['exclusion_reason'] = 'Excluido por el usuario: pendiente de fotos con etiquetas y cajas humanas.'
            row['issues'].append(row['exclusion_reason'])
    planned = [r for r in rows if not r['reserved'] and not r.get('excluded') and r.get('source_path')]
    needed = sum(Path(r['source_path']).stat().st_size for r in planned)
    if shutil.disk_usage(output.parent).free < needed + 1024**3:
        raise OSError('Espacio insuficiente para copias independientes más 1 GiB de margen.')
    output.mkdir(parents=True, exist_ok=False)
    for group in (*CLASSES, 'revision', 'mixtas', 'sin_objetos'):
        (output/group).mkdir()
    for number, row in enumerate(planned, 1):
        destination = output/row['group']/row['split']/row['source']/Path(row['source_path']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(row['source_path'], destination)
        if sha256(destination) != row['sha256']:
            raise ValueError(f'La imagen cambió desde la verificación: {row["source_path"]}')
        row['copy_path'] = str(destination.relative_to(output))
        write_json(destination.with_suffix('.anotaciones.json'), {
            'identity': row['identity'], 'image': destination.name, 'material': row['group'],
            'proposed_material': row.get('proposed_material'),
            'split': row['split'], 'width': row['width'], 'height': row['height'],
            'sha256': row['sha256'], 'source_path': row['source_path'],
            'annotation_origin': row['annotation_origin'], 'approved_for_training': False,
            'canonical_classes': list(CLASSES), 'objects': [dict(obj, class_id=CLASSES.index(obj['material'])
                                                            if obj.get('material') in CLASSES else None)
                                                         for obj in row['boxes']], 'issues': row['issues']})
        if number % 250 == 0:
            print(f'Copias verificadas: {number}/{len(planned)}', flush=True)
    for ann in export_audit:
        destination = output/'_auditoria/anotaciones_originales'/ann['dataset']/ann['split']/'_annotations.coco.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ann['annotation_file'], destination)
        if sha256(destination) != ann['sha256']:
            raise ValueError('COCO cambió durante la organización')
    shutil.copyfile(PROJECT_ROOT/'docs/TRASHNET_LICENSE.txt', output/'_auditoria/TRASHNET_LICENSE.txt')
    fields = ['identity', 'source', 'group', 'proposed_material', 'split', 'copy_path', 'source_path', 'original_paths',
              'excluded', 'exclusion_reason', 'excluded_copy_path',
              'existing_in_project', 'historical_split', 'export_split', 'export_variants',
              'annotation_origin', 'approved_for_training', 'reserved', 'reserved_reason', 'sha256', 'issues']
    write_csv(output/'inventario.csv', rows, fields)
    write_csv(output/'test_reservado/solo_metadatos.csv', [r for r in rows if r['reserved']], fields)
    issues = [r for r in rows if r['issues']]
    write_csv(output/'_auditoria/pendientes.csv', issues, fields)
    write_json(output/'_auditoria/coco_verificacion.json', export_audit)
    write_json(output/'_auditoria/fingerprints_nuevos_no_test.json', features)
    write_json(output/'_auditoria/procedencia.json', {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'new_coco_boxes_human_confirmed': args.manual_boxes_confirmed,
        'evidence': 'Respuesta explícita del usuario: Todas dibujadas manualmente por personas'
                    if args.manual_boxes_confirmed else 'sin confirmación',
        'sources': sorted({r['dataset'] for r in export_audit}),
        'image_source': 'TrashNet: coincidencia por nombre, huella histórica y pHash para exportaciones no reservadas',
        'new_exports_license_metadata': 'Private; se conserva el dato del exportador. Sin publicación externa.',
        'click_annotation_origin_unchanged': True})
    category_counts = {}
    for material in CLASSES:
        copied = [r for r in rows if r['group'] == material and r['copy_path']]
        category_counts[material] = {
            'train': sum(r['split'] == 'train' for r in copied),
            'val': sum(r['split'] == 'val' for r in copied),
            'without_split': sum(r['split'] == 'sin_dividir' for r in copied),
            'total_copied_photos': len(copied),
            'click_photos': sum(r['source'] == 'CLICK' for r in copied),
            'trashnet_photos_already_present': sum(r['source'] == 'TrashNet' for r in copied),
            'photos_with_human_boxes': sum(r['source'] == 'TrashNet' and bool(r['boxes']) for r in copied),
            'test_metadata_only': sum(r['group'] == material and r['reserved'] for r in rows),
            'new_independent_photo_claimed': 0}
    summary = {'source_roots': [str(args.source), str(args.trashnet)], 'output': str(output),
               'classes': category_counts, 'copied_photo_groups': dict(Counter(r['group'] for r in planned)),
               'total_copied_photos': len(planned), 'copied_bytes': needed,
               'new_coco_export_photos': len(exports), 'new_coco_boxes': sum(a['boxes'] for a in export_audit),
               'new_coco_metadata_errors': sum(len(a['errors']) for a in export_audit),
               'new_coco_photos_without_boxes': sum(a['images_without_boxes'] for a in export_audit),
               'new_coco_whole_image_boxes': sum(a['whole_image_boxes'] for a in export_audit),
               'safe_photos_without_boxes': sum(any('sin cajas' in issue for issue in r['issues'])
                                                for r in planned),
               'unmapped_photos_quarantined': [r['identity'] for r in rows if r.get('proposed_material')],
               'new_coco_test_overlaps_historical_train_val': sum(
                   r['reserved'] and r.get('historical_split') in ('train', 'val')
                   for r in rows if r['source'] == 'TrashNet'),
               'image_integrity_errors': {p: f['error'] for p, f in features.items() if 'error' in f},
               'test_image_files_opened_or_copied': 0, 'test_protection_reference': 'B y test nuevos de CLICK/COCO',
               'training_started': False,
               'material_intake_policy': policy,
               'excluded_photos_total': sum(bool(r.get('excluded')) for r in rows),
               'excluded_by_material': dict(Counter(r['group'] for r in rows if r.get('excluded'))),
               'historical_base_train_val': dict(Counter(r['class'] for r in reference_rows if r['split'] != 'test')),
               'limits': ['La organización no aprueba entrenamiento ni modifica los manifiestos originales.',
                          'El protocolo histórico es B; A tiene particiones distintas y no se mezcla como evaluación independiente.',
                          'No se cuentan las exportaciones COCO como fotos nuevas de TrashNet.',
                          'Las fotos mixtas y con material incierto se conservan separadas.',
                          'No se certificó independencia de objetos físicos o sesiones nuevas.',
                          'Las cajas de imagen completa requieren revisión humana de calidad.']}
    write_json(output/'resumen.json', summary)
    write_csv(output/'conteos_por_categoria.csv', [dict(material=k, **v) for k, v in category_counts.items()],
              ['material', 'train', 'val', 'without_split', 'total_copied_photos', 'click_photos',
               'trashnet_photos_already_present', 'photos_with_human_boxes', 'test_metadata_only'])
    render_report(output, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return summary


def render_report(output, summary):
    lines = ['# Datos separados por seis materiales', '',
             'Copias independientes de imágenes; los originales y los modelos permanecen intactos. '
             'Esta es una carpeta de revisión, no una entrada de entrenamiento ya aprobada.', '',
             '## Fotos completas copiadas (test fuera)', '',
             '| Material | Train | Val | Sin dividir | Total | Con cajas humanas nuevas |',
             '|---|---:|---:|---:|---:|---:|']
    for material, counts in summary['classes'].items():
        lines.append(f'| {material} | {counts["train"]} | {counts["val"]} | {counts["without_split"]} | '
                     f'{counts["total_copied_photos"]} | {counts["photos_with_human_boxes"]} |')
    excluded = summary.get('excluded_by_material', {})
    if excluded:
        lines += ['', '## Materiales excluidos por solicitud del usuario', '',
                  '; '.join(f'{material}: {count} fotos excluidas' for material, count in excluded.items())+'.',
                  'Papel queda pendiente de las nuevas fotos con etiquetas y cajas humanas. '
                  'Las copias anteriores apartadas y las fuentes originales se conservan; no cuentan como fotos activas.', '']
    lines += ['', 'Las carpetas de material contienen fotos completas con una etiqueta de material. '
              'Los objetos conocidos en fotos mixtas de CLICK se conservan con sus cajas en `mixtas` o `revision`; '
              'no se inventan recortes ni se cuentan como fotos puras.', '',
              '## Qué falta', '',
              '- No ingresaron fotos nuevas de orgánicos en estas fuentes. La base histórica B ya conserva '
              f'{summary["historical_base_train_val"].get("organicos", 0)} fotos de orgánicos train/val; no se copiaron aquí.',
              '- Cartón, vidrio y metal nuevos son exportaciones de fotos TrashNet ya existentes. Aportan cajas humanas '
              'confirmadas por el usuario; no aumentan la diversidad de fotos independientes.',
              f'- Hay {summary["new_coco_photos_without_boxes"]} fotos en las exportaciones sin cajas; necesitan anotación '
              'o confirmación explícita como negativos. No se tomaron como negativos automáticamente.',
              f'- Se detectaron {summary["new_coco_whole_image_boxes"]} cajas que cubren toda la imagen en los metadatos '
              '(incluido test). Una persona debe revisar en train/val si cada caja sigue los límites del objeto.',
              '- CLICK sigue pendiente por su origen SAM 3; asignar materiales no cambia la procedencia de sus cajas.',
              '- Resolver materiales en `revision`, incluidas tapas, envases y trash. `trash` no equivale a orgánicos.',
              f'- {len(summary.get("unmapped_photos_quarantined", []))} fotos sin correspondencia en el manifiesto B '
              'se conservan en revisión con su material propuesto; no se asumen etiquetas aprobadas ni particiones nuevas.',
              '- Recoger fotos de otros objetos, fondos y sesiones; confirmar etiquetas y procedencia. '
              'La cuota final se calculará sobre el conjunto aprobado del proyecto, no sumando duplicados.', '',
              '## Anotaciones y particiones', '',
              'Cada foto tiene un archivo `.anotaciones.json`: dimensiones, SHA-256, origen, material y cajas cuando '
              'existen. Los IDs canónicos son 0 plastico, 1 papel, 2 carton, 3 vidrio, 4 metal, 5 organicos. '
              'Las etiquetas de imagen completa sin cajas sirven para clasificación, no constituyen cajas del detector.', '',
              'Se conservaron los COCO originales en `_auditoria/anotaciones_originales`. Las cajas se confirmaron '
              'humanas por respuesta del usuario; las exportaciones declaran licencia Private, que se conserva como '
              'metadato. No hubo subida ni publicación.', '',
              'La partición histórica B tiene prioridad para fotos ya conocidas. Cualquier foto o grupo que sea test '
              'de B o test de una exportación nueva queda reservado en todas sus variantes. '
              f'{summary["new_coco_test_overlaps_historical_train_val"]} fotos reservadas nuevas ya figuraban en train/val '
              'histórico: ese test nuevo no demuestra generalización de modelos anteriores. No se abrieron ni copiaron '
              'imágenes test; `test_reservado/solo_metadatos.csv` permite contar sin previsualizar.', '',
              'A y B son experimentos históricos con particiones distintas. Este inventario sigue B; las métricas '
              'de A no se deben usar como evaluación independiente de un modelo entrenado con B.', '',
              'No se entrenó, no se balancearon muestras artificialmente ni se cambiaron los pesos o manifiestos activos.', '',
              '## Archivos', '',
              '- `inventario.csv`: correspondencia foto original, copia, material, partición y pendientes.',
              '- `conteos_por_categoria.csv` y `resumen.json`: conteos medidos.',
              '- `_auditoria/pendientes.csv`: incidencias de material, cajas, procedencia o partición.',
              '- `_auditoria/coco_verificacion.json`: integridad estructural de las nueve anotaciones COCO.',
              '- `_auditoria/fotos_sin_cajas_train_val.csv`: fotos no reservadas con cajas pendientes.',
              '- `_auditoria/cajas_imagen_completa_train_val.csv`: cajas no reservadas que requieren revisar sus límites.',
              '- `_auditoria/procedencia.json`: confirmación humana y procedencia.', '']
    (output/'LEEME.md').write_text('\n'.join(lines), encoding='utf-8')
    for material in CLASSES:
        (output/material/'LEEME.txt').write_text(
            f'Material: {material}\nFotos copiadas: {summary["classes"][material]["total_copied_photos"]}\n'
            'Consulta LEEME.md e inventario.csv en la carpeta principal para anotaciones, pendientes y restricciones.\n',
            encoding='utf-8')
    table = ''.join('<tr><td><a href="'+m+'/">'+m+'</a></td><td>'+str(c['train'])+'</td><td>'+
                    str(c['val'])+'</td><td>'+str(c['total_copied_photos'])+'</td><td>'+
                    str(c['photos_with_human_boxes'])+'</td></tr>' for m, c in summary['classes'].items())
    page = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Datos separados · seis materiales</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#183632}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087a61}.note{background:#fff3d7;padding:16px}</style>
<h1>Datos separados por material</h1><p>Fotos copiadas con sus anotaciones; imágenes test fuera de las carpetas.</p>
<table><tr><th>Carpeta</th><th>Train</th><th>Val</th><th>Total</th><th>Con cajas humanas nuevas</th></tr>'''+table+'''
</table><p><a href="revision/">Revisión del material</a> · <a href="mixtas/">Fotos mixtas</a> · <a href="sin_objetos/">Sin objetos</a></p>
<p class="note">Faltan nuevas fotos de orgánicos y mayor variedad de objetos y fondos. Las exportaciones de cartón, vidrio y metal reutilizan TrashNet: aportan anotaciones, no fotos independientes nuevas. CLICK conserva su restricción por origen SAM 3.</p>
<p><a href="LEEME.md">Detalle de pendientes</a> · <a href="inventario.csv">Inventario</a> · <a href="conteos_por_categoria.csv">Conteos</a> · <a href="test_reservado/solo_metadatos.csv">Test: solo metadatos</a></p>
<p>No se inició entrenamiento. Consulta los pendientes antes de usar estas carpetas.</p></html>'''
    if excluded:
        note = '<p class="note">Papel: '+str(excluded.get('papel', 0))+\
               ' fotos anteriores excluidas por solicitud del usuario. Pendiente de las nuevas imágenes con etiquetas y cajas humanas.</p>'
        page = page.replace('</table>', '</table>'+note)
    (output/'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--trashnet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manual-boxes-confirmed', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    organize(parser.parse_args())
