"""Recupera el ZIP oficial TACO de Zenodo; no extrae ni abre fotografías."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import zipfile

import requests


URL = 'https://zenodo.org/records/3587843/files/TACO.zip?download=1'
MD5 = 'e9149407d883e21a8d224feef8210920'
MAX_BYTES = 3 * 1024**3
CHUNK_BYTES = 64 * 1024**2


def verify(path):
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'md5').hexdigest()
    if actual != MD5 or not zipfile.is_zipfile(path):
        raise ValueError('El archivo difiere del ZIP oficial TACO v1 de Zenodo. Se conserva para revisión.')


def download(destination):
    destination = Path(destination)
    if destination.exists():
        verify(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix('.zip.part')
    offset = part.stat().st_size if part.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise ValueError('Zenodo no entregó un ZIP; no se modifica el archivo parcial.')
        if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError('El servidor no permite reanudar este parcial. Conservarlo y usar otra ruta --output.')
        total, last = offset, offset
        with part.open('ab' if part.exists() else 'xb') as stream:
            for chunk in response.iter_content(4 * 1024**2):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError('El ZIP supera el tamaño máximo esperado. Se conserva el parcial.')
                stream.write(chunk)
                if total - last >= 128 * 1024**2:
                    print(f'TACO: {total // 1024**2} MiB descargados', flush=True)
                    last = total
    verify(part)
    part.replace(destination)
    return destination


def download_parallel(destination, connections=4):
    """Reanuda el prefijo ya descargado; rangos verificados sin truncar archivos."""
    if not 1 <= connections <= 8:
        raise ValueError('Usa entre 1 y 8 conexiones.')
    destination = Path(destination)
    if destination.exists():
        verify(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.zip.part')
    with requests.get(URL, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(30, 60)) as probe:
        probe.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', probe.headers.get('Content-Range', ''))
        if probe.status_code != 206 or not match:
            raise ValueError('Zenodo no admite rangos: conservar el parcial y usar la descarga secuencial.')
        total = int(match[1])
        identity = probe.headers.get('ETag') or probe.headers.get('Last-Modified')
    if not 0 < total <= MAX_BYTES:
        raise ValueError('Tamaño oficial inesperado.')
    folder = destination.with_suffix('.zip.ranges')
    state_path = folder / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['total'] != total or state['identity'] != identity or state['url'] != URL or state['md5'] != MD5:
            raise ValueError('El archivo remoto cambió respecto al plan de rangos. Se conservan los parciales.')
    else:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > total:
            raise ValueError('El parcial supera el archivo oficial.')
        folder.mkdir(exist_ok=False)
        state = {'url': URL, 'md5': MD5, 'total': total, 'identity': identity, 'offset': offset,
                 'ranges': [[start, min(total - 1, start + CHUNK_BYTES - 1)] for start in range(offset, total, CHUNK_BYTES)]}
        with state_path.open('x', encoding='utf-8') as stream:
            json.dump(state, stream)
    current = partial.stat().st_size if partial.exists() else 0
    if current < state['offset'] or current > total:
        raise ValueError('El prefijo local cambió o está incompleto.')
    if not partial.exists():
        partial.touch(exist_ok=False)

    def fetch(item):
        index, (start, end) = item
        if current > end:
            return index
        path = folder / f'{index:04d}.part'
        size = path.stat().st_size if path.exists() else 0
        expected = end - start + 1
        if size > expected:
            raise ValueError('Rango parcial demasiado grande; se conserva para revisión.')
        if size == expected:
            return index
        headers = {'Range': f'bytes={start + size}-{end}'}
        if identity:
            headers['If-Range'] = identity
        with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start + size}-{end}/{total}':
                raise ValueError('Respuesta HTTP Range incorrecta; no se modifica el parcial.')
            with path.open('ab' if path.exists() else 'xb') as stream:
                for block in response.iter_content(64 * 1024):
                    size += len(block)
                    if size > expected:
                        raise ValueError('El servidor entregó más bytes que el rango solicitado.')
                    stream.write(block)
        if size != expected:
            raise ValueError('Rango incompleto; ejecutar de nuevo para reanudar.')
        print(f'TACO rango {index + 1}/{len(state["ranges"])} completo', flush=True)
        return index

    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(fetch, enumerate(state['ranges'])))
    with partial.open('ab') as target:
        for index, (start, end) in enumerate(state['ranges']):
            if current > end:
                continue
            with (folder / f'{index:04d}.part').open('rb') as source:
                source.seek(max(0, current - start))
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    target.write(block)
                    current += len(block)
    verify(partial)
    partial.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/downloads/TACO-official-v1.zip'))
    parser.add_argument('--connections', type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    path = download_parallel(args.output, args.connections) if args.connections > 1 else download(args.output)
    print(f'ZIP TACO oficial verificado: {path}', flush=True)
