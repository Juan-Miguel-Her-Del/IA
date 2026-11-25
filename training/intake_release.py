"""Select only on val; freeze hashes before the final held-out evaluation."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from app.detector import NeuralDetector, crop_box
from app.localization import Region
from app.vision import InferenceService
from training.intake_data import collect_material_rows, IntakeMaterialDataset
from training.intake_training import prepare_data, atomic_save
from training.intake_detection import group_detection_rows, IntakeDetectionDataset
from training.detection_metrics import spatial_metrics, match
from training.prediction import model_probabilities
from training.runtime import load_checkpoint
from utils.config import CLASSES
from utils.dataset import sha256


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    matrix = np.bincount(labels*6+predictions, minlength=36).reshape(6, 6)
    tp = matrix.diagonal().astype(float)
    recall, precision = tp/np.maximum(matrix.sum(1), 1), tp/np.maximum(matrix.sum(0), 1)
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-12)
    return {'accuracy': float(tp.sum()/max(len(labels), 1)), 'macro_f1': float(f1.mean()),
            'samples': len(labels), 'class_order': list(CLASSES), 'confusion_matrix': matrix.tolist(),
            'per_class_recall': dict(zip(CLASSES, recall.tolist())),
            'glass_as_plastic': int(matrix[3, 0]), 'plastic_as_glass': int(matrix[0, 3])}


class RuntimeCropDataset(IntakeMaterialDataset):
    """Use the exact region geometry used by NeuralDetector and InferenceService."""
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(BytesIO(self._bytes(row))) as image:
            image.load()
            if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('Image dimensions/orientation mismatch.')
            image = image.convert('RGB')
        if row['crop'] is not None:
            x, y, w, h = row['crop']
            image = image.crop(crop_box((x, y, x+w, y+h), image.width, image.height))
        return self.transform(image), row['label']


def select_classifier(run):
    run = Path(run)
    output = run/'classifier_selection.pth'
    if output.exists(): raise ValueError('Classifier selection already exists; preserve it.')
    torch.set_num_threads(2)
    candidates, labels, fingerprints = {}, None, set()
    for name in ('hybrid', 'cnn'):
        path = run/name/'best_model.pth'
        model, ck = load_checkpoint(path, 'cpu')
        _, _, rows, provenance = prepare_data(ck['config'])
        if provenance['manifest_sha256'] != ck['manifest_sha256']:
            raise ValueError('Dataset changed since classifier training.')
        fingerprints.add(ck['manifest_sha256'])
        dataset = RuntimeCropDataset(rows, ck['config'])
        all_probs, actual = [], []
        with torch.inference_mode():
            for images, target in DataLoader(dataset, batch_size=32, num_workers=0):
                all_probs.append(model_probabilities(model, images, ck['config']).numpy())
                actual.extend(target.tolist())
        if labels is not None and labels != actual: raise ValueError('Different classifier validation order.')
        labels = actual
        candidates[name] = {'path': str(path.resolve()), 'sha256': sha256(path),
                            'probabilities': np.concatenate(all_probs), 'architecture': ck['architecture']}
    if len(fingerprints) != 1: raise ValueError('Classifiers do not use identical data.')
    comparisons = []
    for cnn_weight in (0., .25, .5, .75, 1.):
        probs = (1-cnn_weight)*candidates['hybrid']['probabilities']+cnn_weight*candidates['cnn']['probabilities']
        comparisons.append({'cnn_weight': cnn_weight, **classification_metrics(labels, probs.argmax(1))})
    # The user requires the CNN+transformer itself as the deployed classifier.
    # Independent CNN/ensemble results remain diagnostics, never runtime choices.
    best = next(r for r in comparisons if r['cnn_weight'] == 0.)
    settings = json.loads(Path('config/app.json').read_text(encoding='utf-8'))
    weight = best['cnn_weight']
    settings['checkpoint'] = candidates['cnn' if weight == 1 else 'hybrid']['path']
    settings['material_candidate_checkpoint'] = candidates['cnn']['path'] if 0 < weight < 1 else None
    settings['material_candidate_weight'] = weight if 0 < weight < 1 else None
    result = {'selection_split': 'val', 'test_images_opened': 0,
        'selection_rule': 'VisionHybrid-v1 alone required by user; checkpoint selected on validation before test',
        'architecture_required_by_user': 'VisionHybrid-v1', 'independent_cnn_in_runtime': False,
        'comparisons': comparisons, 'selected': best, 'settings': settings,
        'classifier_hashes': {v['path']: v['sha256'] for v in candidates.values()},
        'dataset_fingerprint': next(iter(fingerprints))}
    atomic_save(output, result)
    print(json.dumps({'phase': 'classifier_selection', 'comparisons': comparisons, 'selected': best}), flush=True)
    return result


def detector_records(rows, config, path, views, progress=True):
    dataset = IntakeDetectionDataset(rows, config)
    detector = NeuralDetector(path, 2, .27, .45, views)
    records = []
    for i, row in enumerate(rows):
        rgb = dataset.rgb(i)
        detections = detector.detect(rgb, confidence=.001)
        records.append({'image_id': i, 'size': [row['width'], row['height']],
            'truth': [{'box': box, 'category': CLASSES[label]} for box, label in zip(row['boxes'], row['materials'])],
            'predictions': [{'box': list(d.box), 'score': d.confidence} for d in detections]})
        if progress and (i+1) % 100 == 0: print(json.dumps({'detector_evaluation_images': i+1}), flush=True)
    return records


def previous_classifier_validation(rows, config, settings):
    service = InferenceService(settings['checkpoint'], settings['cpu_threads'],
        settings.get('material_candidate_checkpoint'),
        candidate_weight=settings.get('material_candidate_weight'))
    reader = IntakeMaterialDataset([], config)
    actual, predictions, previous_path, rgb = [], [], None, None
    for row in rows:
        if row['path'] != previous_path:
            with Image.open(BytesIO(reader._bytes(row))) as image:
                rgb = np.asarray(image.convert('RGB'))
            previous_path = row['path']
        region = None
        if row['crop'] is not None:
            x, y, w, h = row['crop']
            region = Region(crop_box((x, y, x+w, y+h), row['width'], row['height']), 'validation-box')
        reading = service.classify(rgb, region=region)
        actual.append(row['label']); predictions.append(CLASSES.index(reading.material))
    return classification_metrics(actual, predictions)


def freeze_release(run):
    run = Path(run)
    if (run/'frozen_release.pth').exists(): raise ValueError('A frozen release already exists.')
    selection = torch.load(run/'classifier_selection.pth', map_location='cpu', weights_only=True)
    config = json.loads(Path('config/train_intake_detector.json').read_text(encoding='utf-8'))
    _, _, val, provenance = prepare_data(config)
    previous_settings = json.loads(Path('config/app.json').read_text(encoding='utf-8'))
    previous_classification = selection.get('previous_classifier_validation')
    if previous_classification is None or selection.get('previous_settings') != previous_settings:
        previous_classification = previous_classifier_validation(val, config, previous_settings)
    print(json.dumps({'phase': 'previous_classifier_validation', 'validation': previous_classification}), flush=True)
    if selection['selected']['macro_f1'] < previous_classification['macro_f1']:
        raise ValueError('The new classifiers regress on validation; keep the current runtime.')
    rows = group_detection_rows(val)
    ck = torch.load(run/'detector/last.pth', map_location='cpu', weights_only=True)
    if not ck['complete'] or ck['provenance']['dataset']['manifest_sha256'] != provenance['manifest_sha256']:
        raise ValueError('Detector is incomplete or the dataset changed.')
    comparisons = []
    for name, path, views in (
        ('parent_three_views', Path('models/runtime/detector_D2.pth'), [[512, False], [640, False], [512, True]]),
        ('refined_single_view', run/'detector/best.pth', []),
        ('refined_three_views', run/'detector/best.pth', [[512, False], [640, False], [512, True]])):
        records = detector_records(rows, config, path, views)
        metrics = spatial_metrics(records, confidence=.27)
        comparisons.append({'name': name, 'path': str(path.resolve()), 'views': views,
                            'sha256': sha256(path), 'validation': metrics})
        print(json.dumps({'phase': 'detector_runtime_validation', **comparisons[-1]}), flush=True)
    best = max(comparisons, key=lambda r: r['validation']['mAP50_95'])
    settings = deepcopy(selection['settings'])
    settings.update(detector_checkpoint=best['path'], detection_views=best['views'],
                    detection_threshold=.27, detection_nms_iou=.45)
    hashes = {settings['checkpoint']: sha256(settings['checkpoint']), best['path']: best['sha256']}
    if settings.get('material_candidate_checkpoint'):
        hashes[settings['material_candidate_checkpoint']] = sha256(settings['material_candidate_checkpoint'])
    result = {'settings': settings, 'frozen_before_test': True, 'test_used_for_selection': False,
              'classifier_selection': selection, 'detector_comparisons': comparisons,
              'selected_detector': best['name'], 'checkpoint_hashes': hashes,
              'dataset_fingerprint': provenance['manifest_sha256'], 'detector_config': config,
              'previous_settings': previous_settings, 'previous_classifier_validation': previous_classification}
    atomic_save(run/'frozen_release.pth', result)
    (run/'camera_settings.json').write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'phase': 'release_frozen', 'classifier_weight': selection['selected']['cnn_weight'],
                      'detector': best['name'], 'test_images_opened': 0}), flush=True)
    return result


def final_test(run, resume=False):
    run = Path(run)
    result_path = run/'final_test.pth'
    if result_path.exists() or ((run/'test_started.pth').exists() and not resume):
        raise ValueError('Final test was already started; do not silently reuse it for selection.')
    frozen = torch.load(run/'frozen_release.pth', map_location='cpu', weights_only=True)
    if frozen.get('frozen_before_test') is not True or frozen.get('test_used_for_selection') is not False:
        raise ValueError('Freeze the complete runtime before opening any test pixels.')
    for path, expected in frozen['checkpoint_hashes'].items():
        if sha256(path) != expected: raise ValueError('A frozen checkpoint changed.')
    settings, config = frozen['settings'], frozen['detector_config']
    _, train, val, provenance = prepare_data(config)
    if provenance['manifest_sha256'] != frozen['dataset_fingerprint']:
        raise ValueError('Dataset changed after selection.')
    rows, stats = collect_material_rows(config['dataset_dir'], 'test', config, final_test=True)
    if {r['group'] for r in rows} & {r['group'] for r in train+val}:
        raise ValueError('Test shares source families with model development.')
    freeze_hash = sha256(run/'frozen_release.pth')
    if (run/'test_started.pth').exists():
        started = torch.load(run/'test_started.pth', map_location='cpu', weights_only=True)
        if started['frozen_release_sha256'] != freeze_hash:
            raise ValueError('Resuming evaluation requires the exact same frozen runtime.')
    else:
        atomic_save(run/'test_started.pth', {'frozen_release_sha256': freeze_hash,
                                          'purpose': 'one final evaluation; no retraining or threshold fitting'})
    service = InferenceService(settings['checkpoint'], 2, settings.get('material_candidate_checkpoint'),
                               candidate_weight=settings.get('material_candidate_weight'))
    # Classification-only images count for classification, never as detector negatives.
    actual, predicted, cache_path, rgb = [], [], None, None
    reader = IntakeMaterialDataset([], config)
    image_hashes = {}
    for row in rows:
        if row['path'] != cache_path:
            raw = reader._bytes(row)
            with Image.open(BytesIO(raw)) as image:
                image.load()
                if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                    raise ValueError('Test image dimensions/orientation mismatch.')
                rgb = np.asarray(image.convert('RGB'))
            cache_path = row['path']; image_hashes[cache_path] = sha256(cache_path)
        region = None
        if row['crop'] is not None:
            x, y, w, h = row['crop']
            region = Region(crop_box((x, y, x+w, y+h), row['width'], row['height']), 'human-test-box')
        reading = service.classify(rgb, region=region)
        actual.append(row['label']); predicted.append(CLASSES.index(reading.material))
    classification = classification_metrics(actual, predicted)
    detection_rows = group_detection_rows(rows)
    records = detector_records(detection_rows, config, settings['detector_checkpoint'], settings['detection_views'])
    detection = spatial_metrics(records, settings['detection_threshold'])
    # Actual app crop+classifier; joint recall counts missed objects as failures.
    dataset = IntakeDetectionDataset(detection_rows, config)
    joint_correct, matched_correct, matched_count, predicted_count = 0, 0, 0, 0
    timings = []
    for i, (row, record) in enumerate(zip(detection_rows, records)):
        rgb = dataset.rgb(i)
        predictions = [p for p in record['predictions'] if p['score'] >= settings['detection_threshold']]
        guesses = []
        tick = time.perf_counter()
        for pred in predictions:
            region = Region(crop_box(pred['box'], row['width'], row['height']), 'detector-test')
            guesses.append(service.classify(rgb, region=region).material)
        timings.append((time.perf_counter()-tick)*1000)
        matched = match(predictions, record['truth'], .5)
        matched_count += len(matched); predicted_count += len(predictions)
        correct = sum(guesses[p] == record['truth'][t]['category'] for p, t, _ in matched)
        matched_correct += correct; joint_correct += correct
    result = {'frozen_release_sha256': sha256(run/'frozen_release.pth'), 'test_used_for_selection': False,
              'classification_human_crops_and_explicit_labels': classification,
              'detection': detection, 'joint_IoU50_material_recall': joint_correct/max(detection['objects'], 1),
              'material_accuracy_on_matched_detections': matched_correct/max(matched_count, 1),
              'joint_precision': joint_correct/max(predicted_count, 1),
              'test_metadata_stats': stats, 'test_photo_hashes': image_hashes,
              'test_source_groups_disjoint': True, 'settings': settings}
    atomic_save(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('test_photo_hashes', 'settings')}), flush=True)
    return result


def activate_release(run, defer_camera=False):
    """Publish fresh weight names, retaining every historical checkpoint and rollback bytes."""
    run = Path(run)
    if (run/'activated_release.pth').exists(): raise ValueError('Release already activated.')
    frozen = torch.load(run/'frozen_release.pth', map_location='cpu', weights_only=True)
    evaluation = torch.load(run/'final_test.pth', map_location='cpu', weights_only=True)
    camera = torch.load(run/'camera_validation.pth', map_location='cpu', weights_only=True)
    if evaluation['frozen_release_sha256'] != sha256(run/'frozen_release.pth'):
        raise ValueError('Test does not correspond to this frozen release.')
    if not defer_camera and (camera.get('camera_opened') is not True or camera.get('stop_cleared_objects') is not True or
            camera.get('confirmed_expected', 0) < 1):
        raise ValueError('The camera did not confirm the user-labelled object; keep the current app.')
    if camera['checkpoint_hashes'] != frozen['checkpoint_hashes']:
        raise ValueError('The camera must test the exact frozen models.')
    if camera.get('settings') != frozen['settings']:
        raise ValueError('The camera must use the exact frozen runtime settings.')
    folder = Path('models/runtime')
    manifest_path, app_path = folder/'manifest.json', Path('config/app.json')
    previous_manifest = manifest_path.read_bytes()
    previous_settings = app_path.read_bytes()
    reserved_path = folder/'reserved_hashes.json'
    previous_reserved = reserved_path.read_bytes() if reserved_path.exists() else None
    manifest = json.loads(previous_manifest)
    settings = deepcopy(frozen['settings'])
    destinations = {}
    for key, filename in [('checkpoint', 'classifier_hybrid_intake.pth'),
                          ('material_candidate_checkpoint', 'classifier_cnn_intake.pth'),
                          ('detector_checkpoint', 'detector_intake_D2.pth')]:
        source = settings.get(key)
        if not source: continue
        if sha256(source) != frozen['checkpoint_hashes'][source]: raise ValueError('A frozen weight file changed.')
        destination = folder/filename
        if destination.exists():
            if sha256(destination) != frozen['checkpoint_hashes'][source]:
                raise ValueError('Refusing to overwrite a different runtime checkpoint.')
        else:
            temporary = destination.with_suffix('.pth.tmp')
            shutil.copyfile(source, temporary)
            if sha256(temporary) != frozen['checkpoint_hashes'][source]: raise ValueError('Runtime copy hash mismatch.')
            temporary.replace(destination)
        ck = torch.load(destination, map_location='cpu', weights_only=True)
        manifest[filename] = {'architecture': ck['architecture'], 'sha256': sha256(destination),
            'source_checkpoint': str(Path(source).as_posix()), 'trained_in_project': True,
            'foreign_pretrained_weights': False, 'epoch': ck['epoch'],
            'release_freeze_sha256': sha256(run/'frozen_release.pth'), 'test_used_for_selection': False}
        settings[key] = destination.as_posix()
        destinations[key] = destination.as_posix()
    # Verify actual runtime loading and probabilities before changing active settings.
    classifier = InferenceService(settings['checkpoint'], 2, settings.get('material_candidate_checkpoint'),
                                  candidate_weight=settings.get('material_candidate_weight'))
    detector = NeuralDetector(settings['detector_checkpoint'], 2, settings['detection_threshold'],
                              settings['detection_nms_iou'], settings['detection_views'])
    example = np.full((120, 160, 3), 128, dtype=np.uint8)
    detector.detect(example)
    reading = classifier.classify(example)
    if not np.isclose(sum(reading.probabilities.values()), 1., atol=1e-6):
        raise ValueError('Runtime probabilities failed verification.')
    backup_path = run/'activation_backup.pth'
    if backup_path.exists():
        backup = torch.load(backup_path, map_location='cpu', weights_only=True)
        if backup['new_settings'] != settings: raise ValueError('An activation backup belongs to another runtime.')
        previous_settings, previous_manifest = backup['app_bytes'], backup['manifest_bytes']
        previous_reserved = backup.get('reserved_hashes_bytes')
    else:
        atomic_save(backup_path, {'app_bytes': previous_settings, 'manifest_bytes': previous_manifest,
                                 'reserved_hashes_bytes': previous_reserved,
                                 'destinations': destinations, 'new_settings': settings})
    def write_json_atomic(path, value):
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
    try:
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(app_path, settings)
        if evaluation.get('test_photo_hashes'):
            reserved = set(json.loads(previous_reserved)) if previous_reserved else set()
            reserved.update(evaluation['test_photo_hashes'].values())
            write_json_atomic(reserved_path, sorted(reserved))
    except Exception:
        manifest_path.write_bytes(previous_manifest)
        app_path.write_bytes(previous_settings)
        if previous_reserved is not None: reserved_path.write_bytes(previous_reserved)
        raise
    atomic_save(run/'activated_release.pth', {'settings': settings, 'checkpoint_hashes': frozen['checkpoint_hashes'],
        'test': evaluation, 'camera': camera, 'previous_models_preserved': True,
        'camera_accuracy_validated': not defer_camera,
        'camera_validation_deferred_by_user': bool(defer_camera),
        'camera_deferral_reason': 'User will verify using a phone camera; preserve observed laptop failure.' if defer_camera else None})
    print(json.dumps({'activated': True, 'settings': settings, 'previous_models_preserved': True}), flush=True)
    return settings


def rollback_release(run):
    run = Path(run)
    backup = torch.load(run/'activation_backup.pth', map_location='cpu', weights_only=True)
    # Restore only the app's configuration and manifest; preserve all weight files.
    for path, key in ((Path('config/app.json'), 'app_bytes'),
                      (Path('models/runtime/manifest.json'), 'manifest_bytes')):
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_bytes(backup[key]); temporary.replace(path)
    if backup.get('reserved_hashes_bytes') is not None:
        Path('models/runtime/reserved_hashes.json').write_bytes(backup['reserved_hashes_bytes'])
    print(json.dumps({'previous_runtime_restored': True, 'weight_files_deleted': False}), flush=True)
