"""Agrupación conservadora: clases ambiguas, fotos mixtas y test reservado."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.group_click_materials import group_dataset, image_group
from utils.config import PROJECT_ROOT, CLASSES
from utils.quality_route import read_json


class ClickGroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping_path = PROJECT_ROOT/'config/click_material_mapping.json'
        cls.mapping = read_json(cls.mapping_path)['classes']

    def test_caps_are_not_assumed_plastic_and_pure_photos_require_all_labels(self):
        self.assertEqual(image_group(['Bottle Cap'], self.mapping)[0], 'revision')
        self.assertEqual(image_group(['Plastic Bottle', 'Bottle Cap'], self.mapping)[0], 'revision')
        self.assertEqual(image_group(['Plastic Bag', 'Plastic Bottle'], self.mapping)[0], 'plastico')

    def test_mixed_materials_and_negative_photos_are_not_forced_into_a_class(self):
        self.assertEqual(image_group(['Plastic Bag', 'Crumbled Paper'], self.mapping)[0], 'mixtas')
        self.assertEqual(image_group([], self.mapping)[0], 'sin_objetos')
        self.assertEqual(image_group(['Others'], self.mapping)[0], 'revision')

    def test_catalog_preserves_sources_does_not_open_test_and_does_not_copy_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root, out = Path(folder)/'source', Path(folder)/'grouped'
            cases = {'train': [('one.jpg', [1]), ('mixed.jpg', [1,2])],
                     'val': [('cap.jpg', [3])], 'test': [('sealed.jpg', [1])]}
            sealed = root/'images/test/sealed.jpg'
            source_bytes = b'original fixture bytes, not model input'
            for split, items in cases.items():
                images, annotations = [], []
                for i, (name, categories) in enumerate(items, 1):
                    path = root/'images'/split/name
                    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(source_bytes)
                    images.append({'id': i, 'file_name': name, 'width': 100, 'height': 100})
                    for category in categories:
                        annotations.append({'id': len(annotations)+1, 'image_id': i,
                                            'category_id': category, 'bbox': [10,10,20,20]})
                path = root/'multi class/annotations'/f'instances_{split}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': [
                    {'id': 1, 'name': 'Plastic Bag'}, {'id': 2, 'name': 'Crumbled Paper'}, {'id': 3, 'name': 'Bottle Cap'}]}))
            original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.resolve() == sealed.resolve():
                    raise AssertionError('No abrir test')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded_open):
                report = group_dataset(root,out,self.mapping_path)
            catalog = read_json(out/'catalog_train_val.json')
            self.assertTrue(all(r['split'] != 'test' and not r['approved_for_training'] for r in catalog))
            self.assertNotIn('sealed.jpg', (out/'index.html').read_text(encoding='utf-8'))
            self.assertEqual(report['source_images_copied'], 0)
            self.assertFalse(list(out.rglob('*.jpg')))
            self.assertEqual((root/'images/train/one.jpg').read_bytes(), source_bytes)
            self.assertTrue(all((out/name/'objetos_train_val.csv').exists() for name in CLASSES))
            self.assertEqual(report['counts']['train']['images'], {'plastico':1,'mixtas':1})


if __name__ == '__main__':
    unittest.main()
