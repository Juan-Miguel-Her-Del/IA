"""La recuperación selectiva conserva hashes y no solicita fotografías test."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import time
from unittest.mock import MagicMock, patch

from scripts.restore_public_data import SerializedZipReader, classification_bytes, restore_classification, taco_candidates, taco_bytes


def fixture_row(split='val'):
    content = b'unit-test-image-bytes'
    return {'split': split, 'class': 'plastico', 'dataset': 'RealWaste',
            'path': f'{split}/plastico/photo.jpg', 'raw_path': 'raw/realwaste/missing.jpg',
            'source_path': 'RealWaste/Plastic/Plastic_1.jpg',
            'source_url': 'https://raw.githubusercontent.com/sam-single/realwaste/156e8e9154eeb163df30fcfd767f55b19633bb06/RealWaste/Plastic/Plastic_1.jpg',
            'sha256': hashlib.sha256(content).hexdigest()}


def response(content):
    result = MagicMock()
    result.__enter__.return_value = result
    result.content = content
    return result


class PublicRestoreTests(unittest.TestCase):
    def test_zip_reader_serializes_shared_seek_and_read(self):
        class Archive:
            busy = False
            def read(self, name):
                if self.busy:
                    raise RuntimeError('Concurrent archive cursor')
                self.busy = True
                time.sleep(.001)
                self.busy = False
                return name.encode()
        reader = SerializedZipReader(Archive())
        names = [str(i) for i in range(20)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            self.assertEqual(list(pool.map(reader.read, names)), [n.encode() for n in names])
    def test_offline_taco_uses_only_exact_local_archive_bytes(self):
        content = b'exact-archived-image'
        row = {'id': 1, 'file_name': 'batch_1/one.jpg', 'sha256': hashlib.sha256(content).hexdigest(),
               'downloaded_url': 'https://example.invalid/must-not-request'}
        archive = MagicMock()
        archive.read.return_value = content
        with patch('scripts.restore_public_data.requests.get') as get:
            self.assertEqual(taco_bytes(row, archive, {row['file_name']: 'archive/one.jpg'}, False), content)
            self.assertEqual(list(taco_candidates(row, None, {}, False)), [])
            get.assert_not_called()
    def test_test_split_foreign_sources_and_unpinned_revision_are_rejected(self):
        variants = [dict(fixture_row(), split='test'), dict(fixture_row(), dataset='Other'),
                    dict(fixture_row(), source_url='https://example.invalid/file.jpg'),
                    dict(fixture_row(), source_url=fixture_row()['source_url'].replace('156e8e9154eeb163df30fcfd767f55b19633bb06', 'main'))]
        with patch('scripts.restore_public_data.requests.get') as get:
            for row in variants:
                with self.assertRaises(ValueError):
                    classification_bytes(row)
            get.assert_not_called()

    def test_download_must_match_original_hash(self):
        with patch('scripts.restore_public_data.requests.get', return_value=response(b'changed')):
            with self.assertRaisesRegex(ValueError, 'distinta'):
                classification_bytes(fixture_row())

    def test_recovery_excludes_test_and_reuses_verified_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / 'reference/B'
            reference.mkdir(parents=True)
            for name in ('manifest.csv', 'split_summary.json'):
                (reference / name).write_text('fixture metadata', encoding='utf-8')
            rows = [fixture_row('val'), fixture_row('test')]
            out = root / 'processed'
            with patch('scripts.restore_public_data.REFERENCE', root / 'reference'), \
                 patch('scripts.restore_public_data.read_manifest', return_value=(rows, {})), \
                 patch('scripts.restore_public_data.requests.get', return_value=response(b'unit-test-image-bytes')) as get:
                self.assertEqual(restore_classification('B', root, out, True, 1), 1)
                self.assertEqual(get.call_count, 1)
                self.assertFalse((out / 'test').exists())
                self.assertEqual(restore_classification('B', root, out, True, 1), 1)
                self.assertEqual(get.call_count, 1)
                (out / rows[0]['path']).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'existente distinto'):
                    restore_classification('B', root, out, True, 1)
                self.assertEqual(get.call_count, 1)


if __name__ == '__main__':
    unittest.main()
