"""Versión local de escenas oficiales; preserva referencias históricas y nunca abre test."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import combinations
from pathlib import Path, PurePosixPath
import zipfile

import imagehash
from PIL import Image, ImageOps

from scripts.download_mju import verify as verify_mju
from scripts.download_taco import verify as verify_taco
from scripts.restore_public_data import SerializedZipReader, digest, encode
from utils.config import PROJECT_ROOT
from utils.quality_route import file_hash, read_json, write_new_json


VERSION = 'official-scenes-local-v1'


def aligned_rgb(content, row):
    with Image.open(BytesIO(content)) as image:
        rgb = image.convert('RGB')
        expected = row['width'] / row['height']
        if abs(rgb.width / rgb.height / expected - 1) > .015:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
        if abs(rgb.width / rgb.height / expected - 1) > .015:
            raise ValueError(f"Orientación/dimensiones incompatibles: {row['id']}")
        return rgb


def prepare_content(content, row, method):
    rgb = aligned_rgb(content, row)
    distance = (int(str(imagehash.phash(rgb)), 16) ^ int(row['phash'], 16)).bit_count()
    alignment = 'raw_or_EXIF_dimension_alignment'
    if distance > 6:
        # El ZIP antiguo y la fuente histórica pueden tener distinta orientación.
        # Alinear contra la huella ya registrada, nunca contra una predicción del modelo.
        with Image.open(BytesIO(content)) as image:
            raw = image.convert('RGB')
        for name, operation in (('raw', None), ('rot90', Image.Transpose.ROTATE_90),
                                ('rot180', Image.Transpose.ROTATE_180), ('rot270', Image.Transpose.ROTATE_270),
                                ('flip_lr', Image.Transpose.FLIP_LEFT_RIGHT), ('flip_tb', Image.Transpose.FLIP_TOP_BOTTOM),
                                ('transpose', Image.Transpose.TRANSPOSE), ('transverse', Image.Transpose.TRANSVERSE)):
            view = raw if operation is None else raw.transpose(operation)
            if abs((view.width / view.height) / (row['width'] / row['height']) - 1) > .015:
                continue
            candidate_distance = (int(str(imagehash.phash(view)), 16) ^ int(row['phash'], 16)).bit_count()
            if candidate_distance < distance:
                rgb, distance, alignment = view, candidate_distance, name
            if distance == 0:
                break
    if distance > 6:
        raise ValueError(f"La foto oficial {row['id']} difiere visualmente de la referencia (pHash {distance}). Revisión humana necesaria.")
    if digest(content) == row['sha256']:
        prepared = content
    else:
        rgb.thumbnail((640, 640), Image.Resampling.LANCZOS)
        prepared = encode(rgb, 95 if row.get('source_dataset') == 'MJU-Waste' else 92)
    with Image.open(BytesIO(prepared)) as image:
        final = image.convert('RGB')
        updated = dict(row, sha256=digest(prepared), pixels_sha256=digest(final.tobytes()),
                       phash=str(imagehash.phash(final)), downloaded_size=list(final.size))
    updated['local_recovery'] = {'version': VERSION, 'method': method, 'source_sha256': digest(content),
                                'historical_sha256': row['sha256'], 'historical_pixels_sha256': row['pixels_sha256'],
                                'source_phash_distance_to_history': distance,
                                'orientation_alignment': alignment,
                                'historical_bytes_reproduced': updated['sha256'] == row['sha256']}
    return prepared, updated


def cross_split_candidates(rows):
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    candidates = []
    for first, second in combinations(splits, 2):
        for a in splits[first]:
            for b in splits[second]:
                if (a['group'] == b['group'] or a['sha256'] == b['sha256'] or a['pixels_sha256'] == b['pixels_sha256']
                        or (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count() <= 6):
                    candidates.append({'first': a['id'], 'second': b['id'], 'splits': [first, second]})
    return candidates


def recover(output, taco_path, mju_path, workers=4):
    output = Path(output)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Ya existe un manifiesto publicado en esta versión; usar otra salida.')
    reference_path = PROJECT_ROOT / 'data/reference/refine_D2/manifest.json'
    historical = read_json(reference_path)
    verify_taco(taco_path)
    verify_mju(mju_path)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(taco_path) as taco_file, zipfile.ZipFile(mju_path) as mju_file:
        taco, mju = SerializedZipReader(taco_file), SerializedZipReader(mju_file)
        members = {'/'.join(PurePosixPath(n).parts[-2:]): n for n in taco.namelist()
                   if PurePosixPath(n).parent.name.startswith('batch_')}

        def restore(row):
            if row['split'] == 'test':
                return dict(row), None
            try:
                cached = PROJECT_ROOT / 'data/detection/taco/images' / row['split'] / f"{row['id']:05d}.jpg"
                if row.get('source_dataset') == 'MJU-Waste':
                    content = mju.read('JPEGImages/' + PurePosixPath(row['file_name']).name)
                    if digest(content) != row['source_png_sha256']:
                        raise ValueError(f"PNG MJU distinto: {row['id']}")
                    method = 'official_MJU_archive_exact_PNG'
                elif cached.is_file() and file_hash(cached) == row['sha256']:
                    content, method = cached.read_bytes(), 'historical_exact_cache'
                else:
                    content, method = taco.read(members[row['file_name']]), 'official_TACO_archive'
                content, updated = prepare_content(content, row, method)
                target = output / 'images' / row['split'] / f"{row['id']:05d}.jpg"
                if target.exists():
                    if file_hash(target) != updated['sha256']:
                        raise ValueError(f'Archivo existente distinto: {target}')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(content)
                return updated, None
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                return None, {'image_id': row['id'], 'split': row['split'], 'error': str(exc)}

        rows, errors = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (row, error) in enumerate(pool.map(restore, historical), 1):
                if error:
                    errors.append(error)
                if row:
                    rows.append(row)
                if index % 100 == 0:
                    print(f'Escenas locales: {index}/{len(historical)}; incidencias {len(errors)}', flush=True)
    audit = {'dataset_version': VERSION, 'historical_reference_sha256': file_hash(reference_path),
             'source_archives_sha256': {'TACO': file_hash(taco_path), 'MJU': file_hash(mju_path)},
             'test_images_opened': 0, 'test_rows_unchanged': [r for r in rows if r['split'] == 'test'] == [r for r in historical if r['split'] == 'test'],
             'errors': errors, 'counts': dict(Counter(r['split'] for r in rows)),
             'historical_bytes_reproduced': sum(r.get('local_recovery', {}).get('historical_bytes_reproduced', False) for r in rows),
             'foreign_weights_or_foundation_annotations': False,
             'note': 'Nueva versión de imágenes; no presentar como reproducción de las métricas históricas. pHash es una auditoría heurística.'}
    if errors:
        write_new_json(output / 'recovery_errors.json', audit)
        raise ValueError(f'{len(errors)} fotos requieren revisión; no se publica un manifiesto parcial.')
    preserved_keys = ('id', 'split', 'group', 'width', 'height', 'boxes')
    audit['historical_annotations_and_splits_preserved'] = all(all(a[k] == b[k] for k in preserved_keys) for a, b in zip(rows, historical))
    if not audit['historical_annotations_and_splits_preserved'] or not audit['test_rows_unchanged']:
        raise ValueError('Cambió la anotación humana o la partición histórica.')
    audit['cross_split_candidates'] = cross_split_candidates(rows)
    audit['ready_for_training'] = not audit['cross_split_candidates']
    write_new_json(output / 'audit.json', audit)
    if not audit['ready_for_training']:
        raise ValueError('Hay candidatos de similitud entre particiones; revisar auditoría antes de publicar.')
    write_new_json(output / 'manifest.json', rows)
    print(f"Versión local publicada: {output}; {audit['counts']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--taco-archive', type=Path, default=Path('data/downloads/TACO-official-v1.zip'))
    parser.add_argument('--mju-archive', type=Path, default=Path('data/detection/mju/original.zip'))
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    recover(args.output, args.taco_archive, args.mju_archive, args.workers)
