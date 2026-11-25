"""Atención, CNN, checkpoints y COCO: pruebas técnicas, sin entrenar con el dataset del usuario."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from app.vision import InferenceService
from training.hybrid import HybridMaterialCNN, SpatialAttention, build_hybrid
from training.intake_data import collect_material_rows, IntakeMaterialDataset, assert_disjoint, reserve_partitions
from training import intake_training
from training.runtime import load_checkpoint
from utils.config import CLASSES, PROJECT_ROOT


def small_config(root=None):
    config = json.loads((PROJECT_ROOT/'config/train_intake_hybrid.json').read_text(encoding='utf-8'))
    config.update(image_size=32, cnn_channels=[8, 8, 16, 16], embedding_dim=16,
                  attention_heads=2, transformer_layers=1, mlp_ratio=1, dropout=0.,
                  batch_size=4, epochs=2, warmup_epochs=0, minimum_epochs=1,
                  cache_max_bytes=1024, protect_historical_test=False,
                  augmentation={'layout': 'clean', 'rotation_degrees': 0, 'brightness': 0,
                                'contrast': 0, 'color': 0, 'min_scale': 1., 'scale_probability': 0.})
    if root is not None:
        config['dataset_dir'] = str(root)
    return config


def fixture(root, extras=False):
    for index, material in enumerate(CLASSES):
        for split in ('train', 'val', 'test'):
            folder = root/material/split
            folder.mkdir(parents=True)
            if split == 'test':
                image = f'{material}_test.jpg'
                (folder/image).write_bytes(b'TEST PIXELS MUST NEVER BE READ OR DECODED')
                (folder/'_annotations.coco.json').write_text(json.dumps({
                    'images': [{'id': 1, 'file_name': image, 'width': 40, 'height': 32}],
                    'categories': [{'id': 1, 'name': material}], 'annotations': []}))
                continue
            name = f'{material}_{split}.jpg'
            Image.new('RGB', (40, 32), (30+index*25, 75, 160)).save(folder/name)
            document = {'images': [{'id': 1, 'file_name': name, 'width': 40, 'height': 32}],
                        'categories': [{'id': 1, 'name': material}],
                        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [2, 3, 25, 24]}]}
            if extras and split == 'train' and material in ('plastico', 'papel'):
                extra_name = f'extra_{material}.jpg'
                Image.new('RGB', (40, 32), (20, 90, 120)).save(folder/extra_name)
                extra = {'id': 2, 'file_name': extra_name, 'width': 40, 'height': 32}
                if material == 'plastico':
                    extra.update(annotation_status='image_label_only', material_label='plastico')
                document['images'].append(extra)
            (folder/'_annotations.coco.json').write_text(json.dumps(document), encoding='utf-8')


class HybridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_configuration_supports_100_and_144_spatial_tokens(self):
        config = json.loads((PROJECT_ROOT/'config/train_intake_hybrid.json').read_text(encoding='utf-8'))
        for size, tokens in ((160, 100), (192, 144)):
            config['image_size'] = size
            model = build_hybrid(config).eval()
            self.assertEqual(model.token_count, tokens)
            with torch.no_grad():
                logits = model(torch.zeros(1, 3, size, size))
            self.assertEqual(logits.shape, (1, 6))
            self.assertTrue(torch.isfinite(logits).all())

    def test_cnn_and_attention_both_receive_finite_gradients(self):
        model = build_hybrid(small_config())
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 6))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
        for parameter in (model.local_features[0].weight, model.context[0].attention.qkv.weight, model.position):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_evaluation_disables_attention_dropout_and_rejects_invalid_shapes(self):
        model = HybridMaterialCNN(image_size=32, dropout=.5, cnn_channels=(8, 8, 16, 16),
                                  embedding_dim=16, attention_heads=2, transformer_layers=1).eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(inputs), model(inputs)))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 48, 48))
        with self.assertRaises(ValueError):
            HybridMaterialCNN(embedding_dim=30, attention_heads=4)

    def test_own_checkpoint_roundtrip_and_app_preprocessing(self):
        config = small_config()
        model = build_hybrid(config).eval()
        payload = {'architecture': config['architecture'], 'classes': list(CLASSES), 'image_size': 32,
                   'config': config, 'model_state_dict': model.state_dict(), 'manifest_sha256': 'technical-fixture'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'hybrid.pth'
            torch.save(payload, path)
            loaded, _ = load_checkpoint(path)
            inputs = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(inputs), loaded(inputs)))
            service = InferenceService(path)
            import numpy as np
            image = np.full((60, 90, 3), 130, dtype=np.uint8)
            reading = service.classify(image)
            self.assertEqual(reading.image_size, 32)
            self.assertEqual(set(reading.probabilities), set(CLASSES))
            self.assertAlmostEqual(sum(reading.probabilities.values()), 1., places=5)
            payload['foreign_pretrained_weights'] = True
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)


class IntakeHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_local_category_ids_map_by_name_and_test_is_never_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'dataset'
            fixture(root, extras=True)
            config = small_config(root)
            original_read = Path.read_bytes
            def guarded_read(path):
                if 'test' in path.resolve().relative_to(root.resolve()).parts and path.suffix.lower() != '.json':
                    raise AssertionError('Se intentó leer test')
                return original_read(path)
            with patch.object(Path, 'read_bytes', guarded_read):
                train, stats = collect_material_rows(root, 'train', config, {})
                val, _ = collect_material_rows(root, 'val', config, {})
                assert_disjoint(train, val)
                dataset = IntakeMaterialDataset(train, config)
                for i in range(len(dataset)):
                    tensor, label = dataset[i]
                    self.assertEqual(tuple(tensor.shape), (3, 32, 32))
                    self.assertIn(label, range(6))
                    self.assertLessEqual(dataset.cache_bytes, config['cache_max_bytes'])
            self.assertEqual(stats['image_label_samples'], 1)
            self.assertEqual(stats['unverified_unboxed_images_skipped'], 1)
            self.assertEqual(len(train), 7)
            self.assertEqual({r['label'] for r in train}, set(range(6)))
            self.assertTrue((root/'papel/train/extra_papel.jpg').exists())
            with self.assertRaises(ValueError):
                collect_material_rows(root, 'test', config, {})

    def test_augmented_source_families_are_reserved_without_moving_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'dataset'
            fixture(root)
            training = [{'path': 'train/shared.jpg', 'group': 'organicos/shared'},
                        {'path': 'train/reserved.jpg', 'group': 'organicos/organicos_test.jpg'},
                        {'path': 'train/unique.jpg', 'group': 'organicos/train_unique'}]
            validation = [{'path': 'val/shared.jpg', 'group': 'organicos/shared'},
                          {'path': 'val/test_variant.jpg', 'group': 'organicos/organicos_test.jpg'}]
            selected_train, selected_val, stats, test = reserve_partitions(root, training, validation)
            self.assertEqual([r['path'] for r in selected_train], ['train/unique.jpg'])
            self.assertEqual([r['path'] for r in selected_val], ['val/shared.jpg'])
            self.assertEqual(stats['train_samples_reserved_for_test'], 1)
            self.assertEqual(stats['train_samples_reserved_for_val'], 1)
            self.assertIn('organicos/organicos_test.jpg', test)
            self.assertTrue((root/'organicos/train/organicos_train.jpg').exists())

    def test_historical_test_and_annotation_exception_are_enforced_without_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'dataset'
            fixture(root)
            config = small_config(root)
            ref = Path(folder)/'ref.csv'
            with ref.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['source_id', 'split', 'group_id'])
                writer.writeheader();writer.writerow({'source_id': 'vidrio_train', 'split': 'test', 'group_id': 'reserved'})
            config.update(protect_historical_test=True, historical_test_reference=str(ref))
            rows, stats = collect_material_rows(root, 'train', config, {})
            self.assertEqual(stats['historical_test_images_skipped'], 1)
            self.assertFalse(any(r['label'] == 3 for r in rows))
            p = root/'plastico/train/_annotations.coco.json'
            doc = json.loads(p.read_text());doc['images'][0].update(source_dataset='CLICK', annotation_origin='SAM3 reviewed')
            p.write_text(json.dumps(doc))
            without, _ = collect_material_rows(root, 'train', config, {})
            with_exception, _ = collect_material_rows(root, 'train', config, {'annotation_source_exceptions': [
                {'source': 'CLICK', 'allowed_for_current_training': True}]})
            self.assertEqual(len(with_exception), len(without)+1)
            self.assertTrue((root/'vidrio/train/vidrio_train.jpg').exists())

    def test_smoke_updates_weights_without_publishing_checkpoints_or_touching_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder)/'dataset', Path(folder)/'weights'
            fixture(root)
            config = small_config(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                result = intake_training.train_intake(config, output, device_name='cpu', smoke=True)
            self.assertTrue(result['parameters_changed'])
            self.assertEqual(result['checkpoints_written'], 0)
            self.assertFalse(output.exists())
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_resume_after_interruption_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'dataset'
            fixture(root)
            config = small_config(root)
            reference, interrupted = Path(folder)/'reference', Path(folder)/'interrupted'
            real_save = intake_training.atomic_save
            def interrupt_after_epoch(path, payload):
                real_save(path, payload)
                if path.name == 'last_training.pth' and payload['epoch'] == 1:
                    raise RuntimeError('Simulación de reinicio')
            with contextlib.redirect_stdout(io.StringIO()):
                intake_training.train_intake(config, reference, device_name='cpu')
                with patch.object(intake_training, 'atomic_save', interrupt_after_epoch):
                    with self.assertRaisesRegex(RuntimeError, 'Simulación'):
                        intake_training.train_intake(config, interrupted, device_name='cpu')
                intake_training.train_intake(config, interrupted, device_name='cpu', resume=True)
            expected = torch.load(reference/'last_training.pth', weights_only=True)
            actual = torch.load(interrupted/'last_training.pth', weights_only=True)
            for key, value in expected['model_state_dict'].items():
                self.assertTrue(torch.equal(value, actual['model_state_dict'][key]), key)
            self.assertEqual(expected['history'][-1]['val']['confusion_matrix'], actual['history'][-1]['val']['confusion_matrix'])
            model, _ = load_checkpoint(interrupted/'best_model.pth')
            self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 6))
            self.assertEqual({p.name for p in interrupted.iterdir()}, {'best_model.pth', 'last_training.pth'})


if __name__ == '__main__':
    unittest.main()
