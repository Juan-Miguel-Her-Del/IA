"""Protección de test y conservación de anotaciones al organizar datos aportados."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import organize_material_intake as intake


class MaterialIntakeTests(unittest.TestCase):
    def test_coco_rejects_out_of_bounds_boxes_and_unknown_images(self):
        doc = {'images': [{'id': 1, 'width': 100, 'height': 80}],
               'categories': [{'id': 1, 'name': 'metal'}],
               'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                'bbox': [10, 10, 20, 30], 'area': 600}]}
        self.assertEqual(intake.coco_errors(doc), [])
        doc['annotations'][0]['bbox'] = [90, 10, 20, 30]
        self.assertTrue(intake.coco_errors(doc))
        doc['annotations'][0]['image_id'] = 99
        self.assertIn('referencia inválida', intake.coco_errors(doc)[0])

    def test_source_paths_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(ValueError):
                intake.contained_path(name, '../elsewhere.jpg')

    def test_end_to_end_reserves_test_aliases_before_any_image_inspection(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, raw, project = root/'source', root/'raw', root/'project'
            output = root/'organized'
            historical = []
            raw.mkdir()
            (raw/'metal').mkdir()
            for index, split in enumerate(('train', 'train', 'val', 'test'), 1):
                p = raw/'metal'/f'metal{index}.jpg'
                Image.new('RGB', (32, 24), (index*20, 40, 80)).save(p)
                digest = intake.sha256(p)
                row = {'split': split, 'class': 'metal', 'dataset': 'TrashNet',
                       'source_id': p.stem, 'sha256': digest, 'pixel_sha256': 'fixture-'+str(index),
                       'group_id': 'group-'+str(index)}
                historical.append(row)
                target = project/'data/raw/trashnet/metal'/f'{digest}_{p.name}'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            # Alias exacto del test histórico: queda protegido aun sin una fila propia del manifiesto.
            alias = raw/'metal/metal5.jpg'
            shutil.copyfile(raw/'metal/metal4.jpg', alias)
            alias_meta = project/'data/raw/trashnet/metal'/f'{intake.sha256(alias)}_{alias.name}'
            shutil.copyfile(alias, alias_meta)
            # Una foto excluida o ausente en la referencia no se fuerza a una clase aprobada.
            Image.new('RGB', (32, 24), (10, 100, 200)).save(raw/'metal/metal6.jpg')
            reference = project/'data/reference/B/manifest.csv'
            intake.write_csv(reference, historical, list(historical[0]))
            reserved = {str((raw/'metal'/f'metal{i}.jpg').resolve()) for i in (1, 4, 5)}
            for split, indices in [('train', [2, 4]), ('test', [1])]:
                images, annotations = [], []
                for index in indices:
                    path = source/'Metal.coco'/split/f'export{index}.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(raw/'metal'/f'metal{index}.jpg', path)
                    if index in (1, 4):
                        reserved.add(str(path.resolve()))
                    images.append({'id': index, 'file_name': path.name, 'width': 32, 'height': 24,
                                   'extra': {'name': f'metal{index}.jpg'}})
                    annotations.append({'id': index, 'image_id': index, 'category_id': 1,
                                        'bbox': [1, 1, 20, 20], 'area': 400})
                intake.write_json(path.parent/'_annotations.coco.json',
                                  {'images': images, 'annotations': annotations,
                                   'categories': [{'id': 1, 'name': 'metal'}]})
            click_path = source/'images/train/click.jpg'
            click_path.parent.mkdir(parents=True)
            Image.new('RGB', (40, 30), (240, 0, 0)).save(click_path)
            feature = intake.inspect_image(click_path)[1]
            feature.update(split='train', file_name=click_path.name)
            docs, fingerprints = {}, {}
            for split in ('train', 'val', 'test'):
                path = source/'multi class/annotations'/f'instances_{split}.json'
                doc = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'Plastic Bottle'}]}
                if split == 'train':
                    doc['images'] = [{'id': 1, 'file_name': 'click.jpg', 'width': 40, 'height': 30}]
                intake.write_json(path, doc)
                fingerprints['multi class/'+split] = intake.sha256(path)
            catalog = [{'split': 'train', 'image_id': 1, 'file_name': 'click.jpg', 'width': 40, 'height': 30,
                        'source_path': str(click_path), 'objects': [{'annotation_id': 1,
                            'source_class': 'Plastic Bottle', 'material': 'plastico', 'bbox': [1, 1, 20, 20]}]}]
            review = project/'data/review/click_material_groups_v1'
            intake.write_json(review/'summary.json', {'annotations_sha256': fingerprints})
            intake.write_json(review/'catalog_train_val.json', catalog)
            intake.write_json(project/'runs/dataset_audits/click_20261124/train_val_fingerprints.json', [feature])
            intake.write_json(project/'config/click_material_mapping.json',
                              {'classes': {'Plastic Bottle': {'material': 'plastico'}}})
            (project/'docs').mkdir()
            (project/'docs/TRASHNET_LICENSE.txt').write_text('fixture license')
            inspected = []
            real_inspect = intake.inspect_image
            def guarded_inspect(path):
                self.assertNotIn(str(Path(path).resolve()), reserved, 'No abrir ninguna variante de test')
                inspected.append(str(path))
                return real_inspect(path)
            args = SimpleNamespace(source=source, trashnet=raw, output=output, workers=1,
                                   manual_boxes_confirmed=True)
            with patch.object(intake, 'PROJECT_ROOT', project), patch.object(intake, 'inspect_image', guarded_inspect), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = intake.organize(args)
            self.assertTrue(inspected)
            self.assertEqual(summary['test_image_files_opened_or_copied'], 0)
            self.assertEqual(summary['classes']['metal']['total_copied_photos'], 2)
            self.assertEqual(summary['classes']['metal']['photos_with_human_boxes'], 1)
            self.assertEqual(summary['new_coco_test_overlaps_historical_train_val'], 1)
            self.assertEqual(summary['total_copied_photos'], 4)
            self.assertEqual(summary['unmapped_photos_quarantined'], ['TrashNet/metal6'])
            copies = list(output.rglob('*.jpg'))
            self.assertEqual(len(copies), 4)
            unknown = list((output/'revision').rglob('metal6.anotaciones.json'))
            self.assertEqual(intake.read_json(unknown[0])['proposed_material'], 'metal')
            self.assertFalse(any('export1' in p.name or 'export4' in p.name or 'metal5' in p.name for p in copies))
            annotations = list(output.rglob('export2.anotaciones.json'))
            self.assertEqual(len(annotations), 1)
            self.assertEqual(intake.read_json(annotations[0])['objects'][0]['class_id'], 4)
            self.assertEqual(intake.read_json(annotations[0])['annotation_origin'],
                             'cajas humanas confirmadas por el usuario')
            with self.assertRaises(FileExistsError):
                intake.organize(args)


if __name__ == '__main__':
    unittest.main()
