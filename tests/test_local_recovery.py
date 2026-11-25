"""La versión local declara diferencias de bytes y conserva anotaciones humanas."""
import copy
from io import BytesIO
import unittest

import imagehash
import numpy as np
from PIL import Image

from scripts.recover_local_scenes import aligned_rgb, cross_split_candidates, prepare_content
from scripts.restore_public_data import digest


def fixture():
    y, x = np.indices((96, 128))
    image = Image.fromarray(np.stack(((x * 2) % 256, (y * 3) % 256, (x + y) % 256), axis=2).astype('uint8'))
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=90)
    content = stream.getvalue()
    with Image.open(BytesIO(content)) as decoded:
        rgb = decoded.convert('RGB')
        row = {'id': 1, 'split': 'train', 'group': 'session-a', 'width': 128, 'height': 96,
               'sha256': digest(content), 'pixels_sha256': digest(rgb.tobytes()), 'phash': str(imagehash.phash(rgb)),
               'boxes': [{'annotation_id': 1, 'category': 'Plastic bottle', 'xyxy_normalized': [.1, .1, .9, .9]}]}
    return content, row


class LocalRecoveryTests(unittest.TestCase):
    def test_exact_historical_payload_is_kept(self):
        content, row = fixture()
        before = copy.deepcopy(row)
        prepared, updated = prepare_content(content, row, 'historical_exact_cache')
        self.assertEqual(prepared, content)
        self.assertTrue(updated['local_recovery']['historical_bytes_reproduced'])
        self.assertEqual(row, before)
        self.assertEqual(updated['boxes'], row['boxes'])

    def test_different_encoding_is_explicit_not_claimed_as_historical(self):
        content, row = fixture()
        row['sha256'] = digest(b'other historical encoding')
        _, updated = prepare_content(content, row, 'official_TACO_archive')
        self.assertFalse(updated['local_recovery']['historical_bytes_reproduced'])
        self.assertEqual(updated['local_recovery']['historical_sha256'], row['sha256'])
        self.assertEqual(updated['split'], row['split'])
        self.assertEqual(updated['boxes'], row['boxes'])

    def test_incompatible_dimensions_and_visual_identity_are_rejected(self):
        content, row = fixture()
        with self.assertRaisesRegex(ValueError, 'dimensiones'):
            aligned_rgb(content, dict(row, width=400, height=96))
        with self.assertRaisesRegex(ValueError, 'difiere visualmente'):
            prepare_content(content, dict(row, phash=f"{int(row['phash'],16) ^ ((1 << 64)-1):016x}"), 'official_TACO_archive')

    def test_cross_split_similarity_is_reported(self):
        _, row = fixture()
        second = dict(row, id=2, split='val', group='other')
        self.assertEqual(len(cross_split_candidates([row, second])), 1)
        self.assertEqual(cross_split_candidates([row, dict(second, split='train')]), [])

    def test_archive_orientation_is_matched_to_recorded_fingerprint(self):
        content, row = fixture()
        stream = BytesIO()
        with Image.open(BytesIO(content)) as image:
            image.transpose(Image.Transpose.ROTATE_180).save(stream, format='JPEG', quality=90)
        _, updated = prepare_content(stream.getvalue(), row, 'official_TACO_archive')
        self.assertEqual(updated['local_recovery']['orientation_alignment'], 'rot180')
        self.assertLessEqual(updated['local_recovery']['source_phash_distance_to_history'], 6)
        self.assertEqual(updated['boxes'], row['boxes'])


if __name__ == '__main__':
    unittest.main()
