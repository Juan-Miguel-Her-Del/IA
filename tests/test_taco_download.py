"""El respaldo oficial no reemplaza archivos existentes ni acepta una descarga distinta."""
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from scripts.download_taco import download, download_parallel


def zip_bytes():
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('metadata.txt', 'unit-test-only')
    return stream.getvalue()


class TacoDownloadTests(unittest.TestCase):
    def test_download_verified_archive_and_reuse_without_network(self):
        content = zip_bytes()
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Content-Type': 'application/zip'}
        response.iter_content.return_value = [content]
        with tempfile.TemporaryDirectory() as folder, \
             patch('scripts.download_taco.MD5', hashlib.md5(content).hexdigest()), \
             patch('scripts.download_taco.requests.get', return_value=response) as get:
            out = Path(folder) / 'source.zip'
            download(out)
            download(out)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(out.read_bytes(), content)

    def test_wrong_existing_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch('scripts.download_taco.requests.get') as get:
            out = Path(folder) / 'source.zip'
            out.write_bytes(b'keep original')
            with self.assertRaises(ValueError):
                download(out)
            self.assertEqual(out.read_bytes(), b'keep original')
            get.assert_not_called()

    def test_server_ignoring_range_does_not_corrupt_partial(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.headers = {'Content-Type': 'application/zip'}
        with tempfile.TemporaryDirectory() as folder, patch('scripts.download_taco.requests.get', return_value=response):
            out = Path(folder) / 'source.zip'
            partial = out.with_suffix('.zip.part')
            partial.write_bytes(b'keep partial')
            with self.assertRaises(ValueError):
                download(out)
            self.assertEqual(partial.read_bytes(), b'keep partial')

    def test_parallel_recovery_preserves_prefix_and_verifies_ranges(self):
        content = zip_bytes()
        def get(url, headers, **kwargs):
            start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
            result = MagicMock()
            result.__enter__.return_value = result
            result.status_code = 206
            result.headers = {'Content-Range': f'bytes {start}-{end}/{len(content)}', 'ETag': 'fixture'}
            result.iter_content.return_value = [content[start:end+1]]
            return result
        with tempfile.TemporaryDirectory() as folder, \
             patch('scripts.download_taco.MD5', hashlib.md5(content).hexdigest()), \
             patch('scripts.download_taco.CHUNK_BYTES', 32), \
             patch('scripts.download_taco.requests.get', side_effect=get):
            out = Path(folder) / 'source.zip'
            out.with_suffix('.zip.part').write_bytes(content[:20])
            download_parallel(out, 4)
            self.assertEqual(out.read_bytes(), content)

    def test_empty_interrupted_range_can_be_resumed(self):
        content = zip_bytes()
        interrupted = [False]
        def get(url, headers, **kwargs):
            start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
            result = MagicMock()
            result.__enter__.return_value = result
            result.status_code = 206
            result.headers = {'Content-Range': f'bytes {start}-{end}/{len(content)}', 'ETag': 'fixture'}
            if end > 0 and not interrupted[0]:
                interrupted[0] = True
                result.iter_content.return_value = []
            else:
                result.iter_content.return_value = [content[start:end+1]]
            return result
        with tempfile.TemporaryDirectory() as folder, \
             patch('scripts.download_taco.MD5', hashlib.md5(content).hexdigest()), \
             patch('scripts.download_taco.requests.get', side_effect=get):
            out = Path(folder) / 'source.zip'
            with self.assertRaisesRegex(ValueError, 'incompleto'):
                download_parallel(out, 1)
            download_parallel(out, 1)
            self.assertEqual(out.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
