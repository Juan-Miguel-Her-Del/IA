"""Las tres particiones y sus cajas se conservan sin decodificar imágenes test."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.restore_intake_source_splits import stage_coco, reconcile_rows


class SourceSplitTests(unittest.TestCase):
    def test_previous_click_source_is_archived_instead_of_inflating_manual_export_counts(self):
        old = [{'identity': 'CLICK/train/1', 'group': 'metal', 'copy_path': 'metal/train/CLICK/one.jpg'},
               {'identity': 'TrashNet/metal1', 'group': 'metal', 'copy_path': 'metal/train/old.jpg'},
               {'identity': 'paper1', 'group': 'papel', 'copy_path': ''}]
        replacement = [{'identity': 'TrashNet/metal1', 'group': 'metal', 'copy_path': 'metal/test/new.jpg', 'boxes_count': 1}]
        rows = reconcile_rows(old, replacement, [], {'metal': 'source'}, Path('_audit/before'))
        archived = next(r for r in rows if r['identity'].startswith('CLICK'))
        self.assertEqual(archived['copy_path'], '')
        self.assertIn('materiales', archived['archived_copy_path'])
        active = [r for r in rows if r.get('copy_path')]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]['boxes_count'], 1)

    def test_preserves_all_three_splits_boxes_and_test_bytes_without_image_decoding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, staging = root/'source', root/'staging'
            for i, split in enumerate(('train', 'valid', 'test')):
                folder = source/split
                folder.mkdir(parents=True)
                # Estos bytes no son una imagen: copiar test no requiere interpretarlos.
                image = f'cardboard{i}.jpg'
                (folder/image).write_bytes(b'opaque reserved file '+str(i).encode())
                document = {'images': [{'id': 1, 'file_name': image, 'width': 40, 'height': 30,
                                        'extra': {'name': image}}],
                            'categories': [{'id': 1, 'name': 'carton'}],
                            'annotations': [{'id': 7, 'image_id': 1, 'category_id': 1,
                                             'bbox': [2, 3, 20, 15], 'area': 300}]}
                (folder/'_annotations.coco.json').write_text(json.dumps(document), encoding='utf-8')
            rows, counts = stage_coco(source, 'carton', staging, root/'final', {}, {})
            self.assertEqual(counts, {'train': 1, 'val': 1, 'test': 1})
            self.assertEqual([r['split'] for r in rows], ['train', 'val', 'test'])
            self.assertEqual([r['reserved'] for r in rows], [False, False, True])
            self.assertEqual((staging/'carton/test/cardboard2.jpg').read_bytes(),
                             (source/'test/cardboard2.jpg').read_bytes())
            normalized = json.loads((staging/'carton/test/_annotations.coco.json').read_text(encoding='utf-8'))
            self.assertEqual(normalized['annotations'][0]['bbox'], [2, 3, 20, 15])
            self.assertEqual(normalized['annotations'][0]['category_id'], 2)
            self.assertTrue((staging/'carton/test/_annotations.original.coco.json').exists())
            self.assertTrue(all(r['approved_for_training'] is False for r in rows))


if __name__ == '__main__':
    unittest.main()
