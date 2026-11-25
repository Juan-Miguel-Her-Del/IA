"""Restaura las particiones entregadas por el usuario sin decodificar imágenes ni entrenar."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from utils.config import CLASSES, PROJECT_ROOT


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('original_paths', 'issues'):
            if key in row:
                row[key] = json.loads(row[key]) if row[key] else []
        for key in ('reserved', 'excluded', 'approved_for_training', 'existing_in_project'):
            if row.get(key) in ('True', 'False'):
                row[key] = row[key] == 'True'
    return rows


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Ruta fuera del destino autorizado: {path}')
    return path


def copy_photo(source, destination):
    # Copia para almacenamiento: no Image.open, hashes de fotos ni inferencia.
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if source.stat().st_size != destination.stat().st_size:
        raise OSError(f'Tamaño de copia incorrecto: {source}')


def move_inside(root, source, destination):
    source, destination = inside(root, source), inside(root, destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def reconcile_rows(rows, replacement, extra, materials, backup_relative):
    replaced = {r['identity'] for r in replacement}
    result = {}
    for original in rows:
        row = dict(original)
        if row['group'] in materials and row.get('copy_path') and row['identity'] not in replaced:
            row['archived_copy_path'] = str(Path(backup_relative)/'materiales'/row['copy_path'])
            row['archive_reason'] = 'Otra fuente conservada en la estructura anterior; no mezclar con el COCO manual.'
            row['copy_path'] = ''
        result[row['identity']] = row
    result.update({r['identity']: r for r in replacement+extra})
    return list(result.values())


def stage_coco(folder, material, staging, final_root, old_rows, reference):
    records, counts = [], {}
    for original_split, split in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        annotation_file = folder/original_split/'_annotations.coco.json'
        document = load(annotation_file)
        normalized = copy.deepcopy(document)
        categories = {r['id']: r['name'] for r in document['categories']}
        used = {a['category_id'] for a in document['annotations']}
        if any(categories[k] != material for k in used):
            raise ValueError(f'Categorías inesperadas en {annotation_file}')
        normalized['categories'] = [{'id': i, 'name': name, 'supercategory': 'material'}
                                    for i, name in enumerate(CLASSES)]
        for annotation in normalized['annotations']:
            annotation['category_id'] = CLASSES.index(material)
        by_image = defaultdict(list)
        for annotation in document['annotations']:
            by_image[annotation['image_id']].append(annotation)
        destination = staging/material/split
        destination.mkdir(parents=True)
        for image in document['images']:
            name = image['file_name']
            if Path(name).name != name:
                raise ValueError('Nombre de imagen fuera del conjunto')
            original_name = image.get('extra', {}).get('name', name)
            identity = 'TrashNet/'+Path(original_name).stem
            ref = reference.get(Path(original_name).stem)
            previous = old_rows.get(identity, {})
            current = by_image[image['id']]
            issues = []
            if not current:
                issues.append('Sin cajas: confirmar anotación o negativo mediante revisión humana.')
            if any(a['bbox'] == [0, 0, image['width'], image['height']] for a in current):
                issues.append('Caja de imagen completa: revisar límites del objeto cuando la partición lo permita.')
            if ref and ref['split'] == 'test' and split != 'test':
                issues.append('Test histórico B aparece en train/val de la fuente; bloqueado para entrenamiento.')
            if ref and ref['split'] in ('train', 'val') and split == 'test':
                issues.append('Test de la fuente ya visto en train/val B; no evalúa independientemente los modelos B.')
            if not ref:
                issues.append('Sin correspondencia en manifiesto B; pendiente de revisión de etiqueta.')
            boxes = [{'annotation_id': a['id'], 'source_class': categories[a['category_id']],
                      'material': material, 'class_id': CLASSES.index(material), 'bbox': a['bbox'],
                      'segmentation': a.get('segmentation', [])} for a in current]
            relative = Path(material)/split/name
            source = folder/original_split/name
            copy_photo(source, destination/name)
            row = dict(previous, identity=identity, source='COCO humano', source_dataset=folder.name,
                       group=material, split=split, original_source_split=original_split,
                       copy_path=str(relative), source_path=str(source.resolve()), image_id=image['id'],
                       reserved=split == 'test', reserved_reason='Test original de la fuente' if split == 'test' else '',
                       historical_split=ref['split'] if ref else '', approved_for_training=False,
                       blocked_by_historical_B_test=bool(ref and ref['split'] == 'test' and split != 'test'),
                       excluded=False, exclusion_reason='', excluded_copy_path='',
                       annotation_origin='cajas humanas confirmadas por el usuario',
                       sha256='', boxes_count=len(current), issues=issues)
            save((destination/name).with_suffix('.anotaciones.json'), {
                'identity': identity, 'image': name, 'material': material, 'split': split,
                'source_dataset': folder.name, 'original_source_split': original_split,
                'width': image['width'], 'height': image['height'], 'source_path': str(source.resolve()),
                'annotation_origin': row['annotation_origin'], 'canonical_classes': list(CLASSES),
                'objects': boxes, 'approved_for_training': False, 'reserved_for_test': split == 'test',
                'blocked_by_historical_B_test': row['blocked_by_historical_B_test'], 'issues': issues,
                'copy_verified_by_file_size': True, 'pixels_decoded': False})
            records.append(row)
        shutil.copyfile(annotation_file, destination/'_annotations.original.coco.json')
        save(destination/'_annotations.coco.json', normalized)
        counts[split] = len(document['images'])
        assert len(list(destination.glob('*.jpg'))) == counts[split]
        assert [(a['image_id'], a['bbox']) for a in normalized['annotations']] == [
            (a['image_id'], a['bbox']) for a in document['annotations']]
    return records, counts


def restore(source, output):
    policy_path = PROJECT_ROOT/'config/material_intake_policy.json'
    if policy_path.exists() and load(policy_path).get('reports_enabled') is False:
        raise ValueError('El usuario pidió conservar solo imágenes y anotaciones; no reconstruir reportes ni auditorías en el dataset.')
    source, output = source.resolve(), output.resolve()
    summary = load(output/'resumen.json')
    if summary.get('partition_policy') == 'preserve_source_splits':
        raise ValueError('La corrección ya se aplicó; revisar nuevos datos antes de repetirla.')
    job = output/'_auditoria/restauracion_particiones_20261125'
    if job.exists():
        if not (job/'estructura_restaurada.json').exists():
            raise ValueError('Existe una restauración incompleta; comprobar su estado antes de mover más archivos.')
        plan = load(job/'plan.json')
        return finish_restoration(output, job, job/'antes', csv_rows(job/'antes/inventario.csv'),
                                  plan['replacement'], plan['extra'], plan['sources'], plan['partitions'])
    job.mkdir(exist_ok=False)
    before, staging = job/'antes', job/'preparado'
    before.mkdir()
    for name in ('inventario.csv', 'resumen.json', 'conteos_por_categoria.csv', 'LEEME.md', 'index.html'):
        shutil.copyfile(output/name, before/name)
    shutil.copyfile(output/'_auditoria/verificacion_final.json', before/'verificacion_final.json')
    for name in ('pendientes.csv', 'fotos_sin_cajas_train_val.csv', 'cajas_imagen_completa_train_val.csv'):
        path = output/'_auditoria'/name
        if path.exists():
            shutil.copyfile(path, before/name)
    rows = csv_rows(output/'inventario.csv')
    old = {r['identity']: r for r in rows}
    with (PROJECT_ROOT/'data/reference/B/manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        reference = {r['source_id']: r for r in csv.DictReader(stream) if r['dataset'] == 'TrashNet'}
    provenance = load(output/'_auditoria/procedencia.json')
    if provenance['new_coco_boxes_human_confirmed'] is not True:
        raise ValueError('Falta la confirmación humana de las cajas de estas fuentes.')
    audit = load(output/'_auditoria/coco_verificacion.json')
    for record in audit:
        actual = hashlib.sha256(Path(record['annotation_file']).read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise ValueError('Cambió un COCO; auditar la nueva versión antes de restaurar.')
    sources = {'carton': 'Cartones.v1-cartones-70-20-10.coco',
               'vidrio': 'Envases de vidrio.v1-exportacion-coco.coco',
               'metal': 'Metal - Deteccion.v2-metal-deteccion---tag-metal.coco'}
    replacement, partitions = [], {}
    for material, name in sources.items():
        current, counts = stage_coco(source/name, material, staging, output, old, reference)
        replacement.extend(current)
        partitions[material] = counts
    assert len(replacement) == 1314
    assert len({r['identity'] for r in replacement}) == 1314
    replaced_ids = {r['identity'] for r in replacement}
    # Completar test de las otras fuentes que ya formaban la colección, salvo papel excluido.
    extra = []
    click = load(source/'multi class/annotations/instances_test.json')
    click_images = {str(i['id']): i for i in click['images']}
    click_classes = {i['id']: i['name'] for i in click['categories']}
    click_boxes = defaultdict(list)
    for a in click['annotations']:
        click_boxes[a['image_id']].append(a)
    mapping = load(PROJECT_ROOT/'config/click_material_mapping.json')['classes']
    for original in rows:
        if original['identity'] in replaced_ids or not original.get('reserved') or original['group'] == 'papel':
            continue
        if original.get('copy_path'):
            raise ValueError('Test ya tenía copias; revisar antes de sobrescribir.')
        row = dict(original)
        material = row['group']
        if material not in ('plastico', 'mixtas', 'revision', 'sin_objetos'):
            continue
        path = Path(row['original_paths'][0])
        if row['source'] == 'CLICK':
            im = click_images[row['identity'].rsplit('/', 1)[1]]
            width, height = im['width'], im['height']
            boxes = [{'annotation_id': a['id'], 'source_class': click_classes[a['category_id']],
                      'material': mapping[click_classes[a['category_id']]]['material'], 'bbox': a['bbox']}
                     for a in click_boxes[im['id']]]
            origin = 'SAM 3 seguido de revisión humana, según autor'
        else:
            ref = reference[row['identity'].split('/', 1)[1]]
            width, height, boxes, origin = int(ref['width']), int(ref['height']), [], 'etiqueta humana de TrashNet'
        relative = Path(material)/'test'/row['source']/path.name
        copy_photo(path, staging/relative)
        row.update(split='test', copy_path=str(relative), source_path=str(path), reserved=True,
                   reserved_reason='Test original; reservado para evaluación final', annotation_origin=origin,
                   approved_for_training=False, sha256='', boxes_count=len(boxes))
        save((staging/relative).with_suffix('.anotaciones.json'), {
            'identity': row['identity'], 'image': path.name, 'material': material, 'split': 'test',
            'width': width, 'height': height, 'source_path': str(path), 'objects': boxes,
            'annotation_origin': origin, 'approved_for_training': False, 'reserved_for_test': True,
            'copy_verified_by_file_size': True, 'pixels_decoded': False})
        extra.append(row)
    save(job/'plan.json', {'replacement': replacement, 'extra': extra, 'sources': sources, 'partitions': partitions})
    # Revisar antes de mover: solo se toca lo producido por el organizador, nunca papel.
    for material in sources:
        expected = {str((output/r['copy_path']).resolve()) for r in rows if r['group'] == material and r.get('copy_path')}
        actual = {str(p.resolve()) for p in (output/material).rglob('*.jpg')}
        if expected != actual:
            raise ValueError(f'Hay cambios no inventariados en {material}; se conservan intactos.')
    for material in sources:
        move_inside(output, output/material, before/'materiales'/material)
        move_inside(output, staging/material, output/material)
    for row in rows:
        if row['identity'] in replaced_ids and row.get('copy_path') and row['group'] not in sources:
            path = output/row['copy_path']
            move_inside(output, path, before/'variantes_reemplazadas'/row['copy_path'])
            move_inside(output, path.with_suffix('.anotaciones.json'),
                        (before/'variantes_reemplazadas'/row['copy_path']).with_suffix('.anotaciones.json'))
    for material in ('plastico', 'mixtas', 'revision', 'sin_objetos'):
        test = staging/material/'test'
        if test.exists():
            move_inside(output, test, output/material/'test')
    save(job/'estructura_restaurada.json', {'complete': True, 'paper_folder_modified': False})
    return finish_restoration(output, job, before, rows, replacement, extra, sources, partitions)


def finish_restoration(output, job, before, rows, replacement, extra, sources, partitions):
    summary = load(before/'resumen.json')
    rows = reconcile_rows(rows, replacement, extra, sources, before.relative_to(output))
    # Papel puede estar recibiendo archivos del usuario: no se enumera, mueve ni escribe en esa carpeta.
    for material in ('plastico', 'carton', 'vidrio', 'metal', 'organicos'):
        for split in ('train', 'val', 'test'):
            (output/material/split).mkdir(parents=True, exist_ok=True)
    active = [r for r in rows if r.get('copy_path') and not r.get('excluded')]
    for material in CLASSES:
        current = [r for r in active if r['group'] == material]
        counts = {split: sum(r['split'] == split for r in current) for split in ('train', 'val', 'test')}
        summary['classes'][material].update(counts, total_copied_photos=len(current),
            test_metadata_only=sum(r['group'] == material and r.get('reserved') and not r.get('copy_path') for r in rows))
        summary['classes'][material]['pending_user_import'] = material == 'papel'
        if material in sources:
            summary['classes'][material]['photos_with_human_boxes'] = sum(r.get('boxes_count', 0) > 0 for r in current)
            summary['classes'][material]['photos_without_boxes'] = sum(r.get('boxes_count', 0) == 0 for r in current)
            summary['classes'][material]['click_photos'] = 0
            summary['classes'][material]['trashnet_photos_already_present'] = len(current)
            assert counts == partitions[material]
        if material != 'papel':
            assert len(list((output/material).rglob('*.jpg'))) == len(current)
    summary.update(partition_policy='preserve_source_splits', source_partition_counts=partitions,
                   total_copied_photos=len(active), copied_photo_groups=dict(Counter(r['group'] for r in active)),
                   test_image_pixels_decoded=0, test_images_copied_for_storage=sum(r['split'] == 'test' for r in active),
                   paper_pending_user_import=True, training_started=False,
                   previous_organization_backup=str(before.relative_to(output)),
                   source_train_val_blocked_by_historical_B_test=sum(r.get('blocked_by_historical_B_test', False) for r in replacement),
                   source_test_seen_in_historical_B_train_val=sum(r['split'] == 'test' and r.get('historical_split') in ('train', 'val') for r in replacement))
    summary.pop('test_image_files_opened_or_copied', None)
    for obsolete in ('total_photos_including_excluded_copies', 'total_stored_photo_bytes_including_excluded',
                     'new_coco_test_overlaps_historical_train_val'):
        summary.pop(obsolete, None)
    summary['unmapped_photos_quarantined'] = [r['identity'] for r in active
                                             if r['group'] == 'revision' and r.get('proposed_material')]
    summary['copied_bytes'] = sum((output/r['copy_path']).stat().st_size for r in active)
    summary['test_protection_reference'] = 'Test original de cada fuente; cruces con B bloqueados antes de entrenar.'
    summary['limits'] = ['Las particiones originales se conservan; valid se llama val sin cambiar sus imágenes.',
                         'Las copias test son para almacenamiento y evaluación final: sin decodificar ni previsualizar.',
                         'Los cruces con el historial B siguen bloqueados; ninguna foto se aprueba automáticamente.',
                         'Papel se está aportando por el usuario: pendiente de verificación, sin modificar esa carpeta.',
                         'CLICK conserva su restricción por origen SAM 3; TrashNet sin cajas no se presenta como detección anotada.']
    save_csv(output/'inventario.csv', rows)
    save_csv(output/'conteos_por_categoria.csv', [dict(material=m, **v) for m, v in summary['classes'].items()])
    save_csv(output/'test_reservado/solo_metadatos.csv', [r for r in rows if r.get('reserved')])
    save_csv(output/'_auditoria/pendientes.csv', [r for r in rows if r.get('issues') and not r.get('excluded')])
    missing, whole = [], []
    for row in replacement:
        if row['split'] == 'test':
            continue
        record = load((output/row['copy_path']).with_suffix('.anotaciones.json'))
        if not record['objects']:
            missing.append(row)
        for obj in record['objects']:
            if obj['bbox'] == [0, 0, record['width'], record['height']]:
                whole.append({'identity': row['identity'], 'material': row['group'],
                              'image_path': row['copy_path'], 'annotation_id': obj['annotation_id'],
                              'bbox': obj['bbox'], 'blocked_by_historical_B_test': row['blocked_by_historical_B_test']})
    save_csv(output/'_auditoria/fotos_sin_cajas_train_val.csv', missing)
    save_csv(output/'_auditoria/cajas_imagen_completa_train_val.csv', whole)
    summary['safe_photos_without_boxes'] = len(missing)
    summary['safe_whole_image_boxes_for_human_review'] = len(whole)
    save_csv(job/'bloqueos_historicos_B.csv', [r for r in replacement if r.get('blocked_by_historical_B_test')])
    verification = {'source_splits_preserved': True, 'coco_images': 1314, 'coco_counts': partitions,
                    'test_photos_copied_for_storage': summary['test_images_copied_for_storage'],
                    'test_pixels_decoded': 0, 'training_started': False, 'paper_folder_modified': False,
                    'verified_utc': datetime.now(timezone.utc).isoformat()}
    save(output/'resumen.json', summary)
    save(output/'_auditoria/verificacion_final.json', verification)
    save(PROJECT_ROOT/'data/review/materiales_actualizados_20261124_v1.json',
         {'output': str(output), 'summary': summary, 'verification': verification})
    render(output, summary)
    policy_path = PROJECT_ROOT/'config/material_intake_policy.json'
    policy = load(policy_path)
    policy.update(partition_policy='preserve_source_splits', test_copy_purpose='almacenamiento; evaluación final',
                  paper_pending_user_import=True)
    save(policy_path, policy)
    save(job/'completado.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)


def render(output, summary):
    lines = ['# Organización con train, val y test originales', '',
             'Cartón, vidrio y metal conservan exactamente las particiones de las exportaciones entregadas. '
             'Solo se cambia el nombre valid por val. Cada partición contiene imágenes, anotaciones COCO '
             'originales, una versión COCO con los seis IDs de material y registros por foto.', '',
             '| Material | Train | Val | Test | Total |', '|---|---:|---:|---:|---:|']
    table = ''
    for material, counts in summary['classes'].items():
        values = [str(counts.get(s, 0)) for s in ('train', 'val', 'test')]
        total = str(counts['total_copied_photos'])
        if material == 'papel':
            values, total = ['Por verificar']*3, 'En preparación'
        lines.append('| '+material+' | '+' | '.join(values)+' | '+total+' |')
        table += '<tr><td><a href="'+material+'/">'+material+'</a></td>'+''.join('<td>'+v+'</td>' for v in values+[total])+'</tr>'
    lines += ['', '## Pendientes y procedencia', '',
              '- Papel: el usuario está colocando sus nuevos datos. Esa carpeta no se modifica ni se dan por verificados sus archivos.',
              '- Orgánicos: pendiente de aportar fotos y anotaciones nuevas.',
              '- Las particiones test están físicamente copiadas para conservar la estructura. No se decodificaron imágenes, '
              'no se previsualizaron y no se usaron para entrenar o elegir modelos.',
              '- Las exportaciones COCO de cartón, vidrio y metal tienen cajas humanas confirmadas por el usuario. '
              'Siete fotos de vidrio carecen de cajas; revisar las de train/val que no estén bloqueadas por el historial, '
              'sin asumir que son negativos.',
              '- Plástico conserva dos fuentes: CLICK con cajas de origen SAM 3, aún restringidas, y TrashNet con '
              'etiqueta de imagen pero sin cajas. Las tres carpetas no implican aprobación del contenido.',
              '- Las fotos mixtas y de material pendiente siguen separadas en mixtas y revision.',
              f'- {summary["source_train_val_blocked_by_historical_B_test"]} fotos del train/val de las exportaciones '
              'coinciden con test B: quedan registradas como bloqueadas para entrenamiento. '
              f'{summary["source_test_seen_in_historical_B_train_val"]} fotos del test entregado ya figuraban en train/val B '
              'y no son una prueba independiente de los modelos B anteriores.',
              '- La organización anterior se conserva en '+summary['previous_organization_backup']+'. '
              'Los originales, la exclusión del papel anterior y los modelos se conservan.', '',
              '## Archivos', '', '- inventario.csv: fuente, partición original, destino y restricciones.',
              '- conteos_por_categoria.csv y resumen.json: conteos actuales.',
              '- _auditoria/verificacion_final.json: comprobación de la estructura.',
              '- _auditoria/restauracion_particiones_20261125/bloqueos_historicos_B.csv: cruces que impiden entrenar directamente.', '',
              'No se inició entrenamiento.', '']
    (output/'LEEME.md').write_text('\n'.join(lines), encoding='utf-8')
    page = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Datos · train, val y test</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#183632}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#087a61}.note{background:#fff3d7;padding:16px}</style>
<h1>Particiones originales: train, val y test</h1><p>Cartón, vidrio y metal conservan las tres particiones entregadas. Test se almacena reservado para evaluación final.</p>
<table><tr><th>Material</th><th>Train</th><th>Val</th><th>Test</th><th>Total</th></tr>'''+table+'''
</table><p class="note">Papel está siendo aportado por el usuario y está pendiente de verificar. Orgánicos sigue pendiente. La estructura no aprueba automáticamente las anotaciones ni los cruces con la base histórica.</p>
<p><a href="LEEME.md">Detalle y pendientes</a> · <a href="inventario.csv">Inventario</a> · <a href="conteos_por_categoria.csv">Conteos</a></p>
<p>No se inició entrenamiento.</p></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')
    for material in ('plastico', 'carton', 'vidrio', 'metal', 'organicos'):
        (output/material/'LEEME.txt').write_text(
            f'{material}: carpetas train, val y test. Test reservado para evaluación final.\n'
            'Consulta el reporte principal para procedencia, cajas y restricciones.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    restore(args.source, args.output)
