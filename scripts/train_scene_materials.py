"""Train our residual CNN from random weights on verified B and TACO train crops."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.compact import CompactCNN, build_transform
from training.runtime import choose_device
from utils.config import CLASSES


ORIGINAL_PREFIX = 'Vision-IA/data/experiments/B/processed/'
VIEW_MIX = {'square': .35, 'rect': .30, 'tight': .25, 'foreground': .10}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def original_rows(archive):
    with StringIO(archive.read(ORIGINAL_PREFIX + 'manifest.csv').decode('utf-8'), newline='') as stream:
        return list(csv.DictReader(stream))


@contextmanager
def open_original(path):
    """Read a prepared local B or the historical ZIP through the same interface."""
    path = Path(path).resolve()
    if path.is_dir():
        class LocalOriginal:
            def read(self, member):
                target = (path / member.removeprefix(ORIGINAL_PREFIX)).resolve()
                if not target.is_relative_to(path):
                    raise ValueError(f'Path outside dataset: {member}')
                return target.read_bytes()
        yield LocalOriginal()
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive


def cache_picture(content, limit=236):
    with Image.open(BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        return image


class MaterialSet(Dataset):
    def __init__(self, originals, scenes, rows, config, training=False):
        self.rows = rows
        self.training = training
        self.transform = build_transform(config, training)
        mode = config.get('image_cache', 'decoded')
        if mode not in ('decoded', 'compressed'):
            raise ValueError('Modo de caché de imágenes desconocido.')
        self.compressed = mode == 'compressed'
        self.images = []
        def store(content):
            return content if self.compressed else cache_picture(content)
        for row in rows:
            if 'paths' in row:
                views = VIEW_MIX if training else ('square', 'rect', 'tight')
                self.images.append({v: store(scenes.read(row['paths'][v])) for v in views})
            else:
                content = originals.read(ORIGINAL_PREFIX + row['path'])
                if digest(content) != row['sha256']:
                    raise ValueError(f"Image changed: {row['path']}")
                self.images.append({'original': store(content)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        images = self.images[index]
        def picture(key):
            return cache_picture(images[key]) if self.compressed else images[key]
        if not self.training and 'square' in images:
            views = torch.stack([self.transform(picture(key)) for key in ('square', 'rect', 'tight')])
            return views, CLASSES.index(self.rows[index]['class'])
        key = random.choices(list(VIEW_MIX), list(VIEW_MIX.values()))[0] if 'square' in images else 'original'
        return self.transform(picture(key)), CLASSES.index(self.rows[index]['class'])


def scores(actual, predicted):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (np.asarray(actual, dtype=np.intp), np.asarray(predicted, dtype=np.intp)), 1)
    support = matrix.sum(axis=1)
    true_positive = matrix.diagonal()
    per_class = 2 * true_positive / np.maximum(support + matrix.sum(axis=0), 1)
    return {'accuracy': float(true_positive.sum() / max(support.sum(), 1)),
            'macro_F1': float(per_class.mean()), 'macro_F1_present': float(per_class.sum() / max(np.count_nonzero(support), 1)),
            'per_class_recall': dict(zip(CLASSES, (true_positive / np.maximum(support, 1)).tolist())),
            'confusion_matrix': matrix.tolist(), 'objects': int(support.sum())}


def balanced_sampler(rows, config):
    counts = Counter(('paths' in r, r['class']) for r in rows)
    # Set the original/scene mixture independently of the number of crops per class.
    effective = {key: min(count, 50) / 50 if key[0] else 1 for key, count in counts.items()}
    total = {domain: sum(value for (d, _), value in effective.items() if d == domain) for domain in (True, False)}
    fraction = config['scene_fraction']
    weights = [(fraction if domain else 1 - fraction) * effective[domain, label] / (total[domain] * counts[domain, label])
               for domain, label in [('paths' in r, r['class']) for r in rows]]
    return WeightedRandomSampler(weights, config['epoch_samples'], replacement=True)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for images, labels in loader:
        truth.extend(labels.tolist())
        if images.ndim == 5:
            outputs = model(images.flatten(0, 1).to(device)).argmax(1).view(len(labels), -1)
        else:
            outputs = model(images.to(device)).argmax(1)[:, None]
        predictions.extend(outputs.cpu().tolist())
    predictions = np.array(predictions)
    names = ('square', 'rect', 'tight') if predictions.shape[1] == 3 else ('original',)
    return {key: scores(truth, predictions[:, index]) for index, key in enumerate(names)}


def save_checkpoint(path, model, config, epoch, report, provenance):
    payload = {'format_version': 1, 'architecture': config['architecture'], 'classes': list(CLASSES),
               'image_size': config['image_size'], 'config': config, 'epoch': epoch,
               'manifest_sha256': provenance['original_manifest_sha256'],
               'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
               'provenance': provenance, 'validation': report}
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def train(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['initialization'] != 'random_from_scratch' or config['test_used'] is not False:
        raise ValueError('Expected random initialization and reserved test')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = choose_device(args.device or ('cpu' if args.smoke else 'cuda'))
    torch.set_num_threads(2)
    with open_original(args.original) as original, zipfile.ZipFile(args.scenes) as scenes:
        b_rows = original_rows(original)
        b_manifest = original.read(ORIGINAL_PREFIX + 'manifest.csv')
        scene_manifest = scenes.read('manifest.json')
        scene_rows = json.loads(scene_manifest)
        scene_audit = json.loads(scenes.read('audit.json'))
        if scene_audit['test_images_opened'] != 0 or {row['split'] for row in scene_rows} != {'train', 'val'}:
            raise ValueError('Incompatible scene partitions')
        for key in ('image_id', 'group', 'source_sha256'):
            if {r[key] for r in scene_rows if r['split'] == 'train'} & {r[key] for r in scene_rows if r['split'] == 'val'}:
                raise ValueError(f'Scene split overlap on {key}')
        b_train, b_val = ([r for r in b_rows if r['split'] == s] for s in ('train', 'val'))
        scene_train, scene_val = ([r for r in scene_rows if r['split'] == s] for s in ('train', 'val'))
        if args.smoke:
            b_train, b_val, scene_train, scene_val = [r[:8] for r in (b_train, b_val, scene_train, scene_val)]
            config.update(epochs=1, batch_size=8, epoch_samples=16)
        provenance = {'initialization': config['initialization'], 'pretrained': False,
                      'original_manifest_sha256': digest(b_manifest), 'scene_manifest_sha256': digest(scene_manifest),
                      'counts': dict(zip(('original_train', 'original_val', 'scene_train', 'scene_val'),
                                         map(len, (b_train, b_val, scene_train, scene_val)))), 'test_used': False,
                      'GPU': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                      'smoke_test': bool(args.smoke)}
        train_rows = b_train + scene_train
        print('Caching', provenance, flush=True)
        training = MaterialSet(original, scenes, train_rows, config, training=True)
        original_val = MaterialSet(original, scenes, b_val, config)
        scene_validation = MaterialSet(original, scenes, scene_val, config)
    loaders = (DataLoader(original_val, batch_size=128), DataLoader(scene_validation, batch_size=64))
    workers = 0 if args.smoke else int(config.get('num_workers', 2))
    if workers < 0:
        raise ValueError('num_workers debe ser no negativo.')
    print('Data loader:', {'num_workers': workers, 'image_cache': config.get('image_cache', 'decoded')}, flush=True)
    loader = DataLoader(training, batch_size=config['batch_size'], sampler=balanced_sampler(train_rows, config),
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=device.type == 'cuda' and not args.smoke)
    model = CompactCNN(config['image_size'], config['dropout'], config['channels'], config['pooling']).to(device)
    provenance['initial_state_sha256'] = digest(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=.00008)
    criterion = nn.CrossEntropyLoss(label_smoothing=.035)
    amp = device.type == 'cuda'
    # La escala por defecto puede omitir todos los pasos de una prueba corta.
    # Contar pasos efectivos: completar una época no acredita actualizar pesos.
    initial_scale = float(config.get('amp_initial_scale', 512.))
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=initial_scale)
    provenance['amp_initial_scale'] = initial_scale if amp else None
    history = []
    best, balanced = -1., -1.
    for epoch in range(1, config['epochs'] + 1):
        tick = time.monotonic()
        model.train()
        losses = []
        optimizer_steps, skipped_steps = 0, 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Pérdida no finita; no se publica un checkpoint de esta época.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not amp and not torch.isfinite(norm):
                raise RuntimeError('Gradientes no finitos en float32.')
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                optimizer_steps += 1
            else:
                skipped_steps += 1
            losses.append(float(loss.detach()))
        if optimizer_steps == 0:
            raise RuntimeError('La época no actualizó pesos: AMP omitió todos los pasos. Revisar escala/gradientes.')
        scheduler.step()
        original = evaluate(model, loaders[0], device)['original']
        scene = evaluate(model, loaders[1], device)
        scene_score = np.mean([v['macro_F1_present'] for v in scene.values()]).item()
        report = {'epoch': epoch, 'loss': float(np.mean(losses)), 'original': original, 'scene': scene,
                  'scene_score': scene_score, 'seconds': time.monotonic() - tick,
                  'optimizer_steps': optimizer_steps, 'skipped_steps': skipped_steps,
                  'grad_scaler_scale': scaler.get_scale() if amp else None}
        history.append(report)
        if scene_score > best:
            best = scene_score
            save_checkpoint(args.output / 'best_scene.pth', model, config, epoch, report, provenance)
        if original['macro_F1'] >= config['minimum_original_macro_F1'] and scene_score > balanced:
            balanced = scene_score
            save_checkpoint(args.output / 'best_balanced.pth', model, config, epoch, report, provenance)
        temporary = args.output / 'history.tmp'
        temporary.write_text(json.dumps(history, indent=2), encoding='utf-8')
        temporary.replace(args.output / 'history.json')
        print(epoch, round(original['macro_F1'], 4), round(scene_score, 4), round(report['seconds'], 1), flush=True)
    status = {'completed_epochs': len(history), 'provenance': provenance}
    (args.output / 'completed.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=Path('data/experiments/B/processed'), help='B preparado o ZIP histórico')
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/train_scene_materials.json'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'))
    train(parser.parse_args())
