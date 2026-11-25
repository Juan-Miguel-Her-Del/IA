"""Download and check the public MJU-Waste archive published by its authors."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import zipfile

import requests

from utils.dataset import sha256


URL = 'https://drive.google.com/uc?export=download&id=1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny'
PUBLIC_PAGE = 'https://github.com/realwecan/mju-waste#download'
SHA256 = '2747be582aca60759138bc961ea89c1472ace69b9c617156678752748d041668'
MAX_BYTES = 1_500_000_000


class DownloadForm(HTMLParser):
    endpoint = None

    def __init__(self):
        super().__init__()
        self.fields = {}
        self.in_download_form = False

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == 'form' and values.get('id') == 'download-form':
            self.endpoint = values.get('action')
            self.in_download_form = True
        if self.in_download_form and tag == 'input' and values.get('name'):
            self.fields[values['name']] = values.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.in_download_form = False


def verify(path):
    if not zipfile.is_zipfile(path) or sha256(path) != SHA256:
        raise ValueError(f'El ZIP MJU difiere de la versión verificada. Fuente pública: {PUBLIC_PAGE}')


def download(destination):
    destination = Path(destination)
    if destination.exists():
        verify(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix('.zip.part')
    with requests.Session() as session:
        with session.get(URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                form = DownloadForm()
                form.feed(response.text)
                if urlparse(form.endpoint or '').hostname != 'drive.usercontent.google.com':
                    raise ValueError(f'La descarga pública de los autores no está disponible: {PUBLIC_PAGE}')
                endpoint, params = form.endpoint, form.fields
            else:
                endpoint, params = response.url, None
        with session.get(endpoint, params=params, stream=True, timeout=90) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError(f'El servidor público no entregó el ZIP. Descarga manual: {PUBLIC_PAGE}')
            with part.open('wb') as stream:
                total, last = 0, 0
                for chunk in response.iter_content(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError('La descarga supera el tamaño de la edición verificada.')
                    stream.write(chunk)
                    if total - last >= 128 * 1024 * 1024:
                        print(f'MJU-Waste: {total // 1048576} MiB descargados', flush=True)
                        last = total
    verify(part)
    part.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/detection/mju/original.zip'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
    else:
        download(args.output)
    print(f'MJU-Waste verificado: {args.output}')
