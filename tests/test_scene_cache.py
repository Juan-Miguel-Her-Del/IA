"""La caché comprimida reduce duplicación de RAM sin cambiar entradas del modelo."""
from io import BytesIO
import random
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.train_scene_materials import MaterialSet, digest
from utils.config import PROJECT_ROOT
from utils.quality_route import read_json


class SceneCacheTests(unittest.TestCase):
    def test_both_caches_produce_identical_validation_and_seeded_training_tensors(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (110, 150, 3), dtype=np.uint8))
        stream = BytesIO()
        image.save(stream, format='JPEG', quality=90)
        content = stream.getvalue()
        archive = SimpleNamespace(read=lambda name: content)
        original_row = {'path': 'fixture.jpg', 'sha256': digest(content), 'class': 'plastico'}
        scene_row = {'paths': dict.fromkeys(('square', 'rect', 'tight', 'foreground'), 'fixture.jpg'), 'class': 'metal'}
        config = read_json(PROJECT_ROOT / 'config/train_scene_materials.json')
        for training in (False, True):
            with self.subTest(training=training):
                a = MaterialSet(archive, archive, [original_row, scene_row], dict(config, image_cache='decoded'), training)
                b = MaterialSet(archive, archive, [original_row, scene_row], dict(config, image_cache='compressed'), training)
                self.assertIsInstance(b.images[0]['original'], bytes)
                for index in range(2):
                    random.seed(27);np.random.seed(27)
                    expected, label_a = a[index]
                    random.seed(27);np.random.seed(27)
                    actual, label_b = b[index]
                    self.assertEqual(label_a, label_b)
                    self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
