"""Restore the locked training and validation partitions using public dataset files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import threading
from urllib.parse import urlparse
import zipfile

from PIL import Image, ImageOps
import requests

from training.data import read_manifest, verify_files
from utils.config import PROJECT_ROOT
from utils.dataset import sha256


DATA = PROJECT_ROOT / 'data'
REFERENCE = DATA / 'reference'
TACO_ARCHIVE = 'https://zenodo.org/records/3587843'


class SerializedZipReader:
    """Serializa seek/read compartidos; conserva CRC y controles de seguridad ZIP."""
    def __init__(self, archive):
        self.archive = archive
        self.lock = threading.Lock()

    def namelist(self):
        return self.archive.namelist()

    def read(self, name):
        with self.lock:
            return self.archive.read(name)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def copy_verified(source, destination, expected):
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f'Conserva el archivo existente y usa otra carpeta: {destination}')
        return
    if sha256(source) != expected:
        raise ValueError(f'Datos de origen distintos: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def classification_bytes(row):
    """Solo RealWaste de la revisión ya fijada; no solicita fotos test."""
    url = row.get('source_url', '')
    parsed = urlparse(url)
    prefix = '/sam-single/realwaste/156e8e9154eeb163df30fcfd767f55b19633bb06/RealWaste/'
    if (row['split'] not in ('train', 'val') or row.get('dataset') != 'RealWaste'
            or parsed.scheme != 'https' or parsed.netloc != 'raw.githubusercontent.com'
            or not parsed.path.startswith(prefix)):
        raise ValueError('Descarga no permitida: se exige una foto train/val de RealWaste con revisión fijada.')
    for attempt in range(3):
        try:
            with requests.get(url, timeout=30) as response:
                response.raise_for_status()
                content = response.content
            if digest(content) != row['sha256']:
                raise ValueError(f"Foto RealWaste distinta: {row['source_path']}")
            return content
        except requests.RequestException:
            if attempt == 2:
                raise


def restore_classification(experiment, raw_data, output, download_missing=False, workers=8):
    reference = REFERENCE / experiment
    rows, _ = read_manifest(reference)
    output.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row['split'] in ('train', 'val')]

    def restore(row):
        target = output / row['path']
        if target.exists():
            if sha256(target) != row['sha256']:
                raise ValueError(f'Archivo existente distinto: {target}')
            return
        source = (raw_data / row['raw_path']).resolve()
        if not source.is_relative_to(raw_data.resolve()):
            raise ValueError(f'Ruta de datos inválida: {row["raw_path"]}')
        if source.is_file():
            copy_verified(source, target, row['sha256'])
        elif download_missing:
            content = classification_bytes(row)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
        else:
            raise FileNotFoundError(f'Falta {source}. Usa --download-missing para RealWaste train/val.')

    errors = []
    def attempt(row):
        try:
            restore(row)
        except (OSError, ValueError, requests.RequestException) as error:
            return str(error)
        return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, error in enumerate(pool.map(attempt, selected), 1):
            if error:
                errors.append(error)
            if index % 100 == 0:
                print(f'Clasificación {experiment}: {index}/{len(selected)}', flush=True)
    if errors:
        raise ValueError(f'Faltan {len(errors)} fotos verificadas. Primeros errores:\n' + '\n'.join(errors[:8]))
    for name in ('manifest.csv', 'split_summary.json'):
        copy_verified(reference / name, output / name, sha256(reference / name))
    verify_files(output, rows)
    return sum(row['split'] != 'test' for row in rows)


def encode(rgb, quality):
    output = BytesIO()
    rgb.save(output, format='JPEG', quality=quality)
    return output.getvalue()


def taco_candidates(row, archive, members, allow_network=True):
    if archive and row['file_name'] in members:
        yield archive.read(members[row['file_name']])
    if not allow_network:
        return
    urls = dict.fromkeys(row.get(name) for name in ('downloaded_url', 'flickr_640_url', 'flickr_url', 'coco_url'))
    urls.pop(None, None)
    for url in urls:
        if url.startswith(('https://', 'http://')):
            try:
                response = requests.get(url, timeout=20)
                if response.ok:
                    yield response.content
            except requests.RequestException:
                continue


def taco_bytes(row, archive, members, allow_network=True):
    for payload in taco_candidates(row, archive, members, allow_network):
        if digest(payload) == row['sha256']:
            return payload
        try:
            with Image.open(BytesIO(payload)) as source:
                rgb = source.convert('RGB')
                if abs((rgb.width / rgb.height) / (row['width'] / row['height']) - 1) > .015:
                    rgb = ImageOps.exif_transpose(source).convert('RGB')
                rgb.thumbnail((640, 640), Image.Resampling.LANCZOS)
                content = encode(rgb, 92)
                if digest(content) == row['sha256']:
                    return content
        except (OSError, ValueError):
            continue
    raise ValueError(f"TACO {row['id']} no coincide con la edición histórica. ZIP oficial: {TACO_ARCHIVE}")


def restore_detector(dataset, output, taco_zip, mju_zip, workers, from_taco, offline=False):
    reference = REFERENCE / dataset
    manifest = json.loads((reference / 'manifest.json').read_text(encoding='utf-8'))
    rows = [r for r in manifest if r['split'] in ('train', 'val')]
    with ExitStack() as stack:
        taco = SerializedZipReader(stack.enter_context(zipfile.ZipFile(taco_zip))) if taco_zip else None
        members = {'/'.join(PurePosixPath(name).parts[-2:]): name for name in taco.namelist()
                   if PurePosixPath(name).parent.name.startswith('batch_')} if taco else {}
        if any(r.get('source_dataset') == 'MJU-Waste' for r in rows):
            from scripts.download_mju import download, verify
            if mju_zip.is_file():
                verify(mju_zip)
            elif offline:
                raise FileNotFoundError('Falta el ZIP MJU local; no se descarga en modo offline.')
            else:
                download(mju_zip)
            mju = SerializedZipReader(stack.enter_context(zipfile.ZipFile(mju_zip)))
        else:
            mju = None

        def restore(row):
            target = output / 'images' / row['split'] / f"{row['id']:05d}.jpg"
            if target.exists():
                if sha256(target) != row['sha256']:
                    raise ValueError(f'Archivo existente distinto: {target}')
                return
            if row.get('source_dataset') == 'MJU-Waste':
                name = PurePosixPath(row['file_name']).name
                payload = mju.read('JPEGImages/' + name)
                if digest(payload) != row['source_png_sha256']:
                    raise ValueError(f'MJU RGB modificado: {name}')
                with Image.open(BytesIO(payload)) as source:
                    content = encode(source.convert('RGB'), 95)
            else:
                cached = from_taco / 'images' / row['split'] / target.name
                if dataset == 'refine_D2' and cached.is_file():
                    copy_verified(cached, target, row['sha256'])
                    return
                content = taco_bytes(row, taco, members, allow_network=not offline)
            if digest(content) != row['sha256']:
                raise ValueError(f"JPEG preparado distinto para la foto {row['id']}. Comprueba la versión de Pillow.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        errors = []
        def attempt(row):
            try:
                restore(row)
                return None
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                return str(error)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, error in enumerate(pool.map(attempt, rows), 1):
                if error:
                    errors.append(error)
                if index % 100 == 0:
                    print(f'Escenas procesadas: {index}/{len(rows)}', flush=True)
        if errors:
            raise ValueError(f'Faltan {len(errors)} escenas verificadas. Primeros errores:\n' + '\n'.join(errors[:8]))
    for name in ('manifest.json', 'audit.json'):
        copy_verified(reference / name, output / name, sha256(reference / name))
    annotations = REFERENCE / 'taco/annotations.json'
    if dataset == 'taco':
        copy_verified(annotations, output / 'annotations.json', sha256(annotations))
    if dataset == 'refine_D2':
        for name in ('manifest.json', 'audit.json'):
            src = REFERENCE / 'mixed_D2' / name
            copy_verified(src, output / ('parent_' + name), sha256(src))
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('A', 'B', 'taco', 'refine_D2'))
    parser.add_argument('--raw-data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--taco-archive', type=Path)
    parser.add_argument('--from-taco', type=Path, default=DATA / 'detection/taco')
    parser.add_argument('--mju-archive', type=Path, default=DATA / 'detection/mju/original.zip')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--download-missing', action='store_true', help='Recuperar solo RealWaste train/val faltante de la revisión fijada')
    parser.add_argument('--offline', action='store_true', help='Usar solo fotos y archivos locales, sin solicitudes HTTP')
    args = parser.parse_args()
    if args.offline and args.download_missing:
        parser.error('--offline y --download-missing son incompatibles')
    if args.dataset in ('A', 'B'):
        output = args.output or DATA / 'experiments' / args.dataset / 'processed'
        count = restore_classification(args.dataset, args.raw_data, output, args.download_missing, args.workers)
    else:
        output = args.output or DATA / 'detection' / args.dataset
        count = restore_detector(args.dataset, output, args.taco_archive, args.mju_archive, args.workers, args.from_taco, args.offline)
    print(f'{count} imágenes de train/val verificadas en {output}.')
