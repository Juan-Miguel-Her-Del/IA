"""Comparación CNN/híbrido sobre COCO nuevo; pesos fuera del dataset, sin reportes dentro."""
from collections import Counter
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.intake_data import collect_material_rows, assert_disjoint, reserve_partitions, IntakeMaterialDataset
from training.material_models import build_material_model
from training.runtime import choose_device, seed_everything
from utils.config import CLASSES


def validate_config(config):
    if config.get('initialization') != 'random_from_scratch' or config.get('test_used') is not False:
        raise ValueError('Se requiere inicialización aleatoria y test reservado.')
    for key in ('epochs', 'batch_size', 'early_stopping_patience', 'minimum_epochs'):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f'{key} debe ser un entero positivo.')
    if config.get('num_workers') != 0:
        raise ValueError('El perfil local usa num_workers=0 para limitar RAM y reanudar su estado aleatorio.')
    for key in ('learning_rate', 'gradient_clip', 'amp_initial_scale'):
        if not isinstance(config.get(key), (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} debe ser positivo y finito.')
    if not 0 <= config.get('warmup_epochs', -1) < config['epochs']:
        raise ValueError('warmup_epochs debe ser menor que epochs.')
    if not 0 <= config.get('label_smoothing', -1) < 1 or config.get('weight_decay', -1) < 0:
        raise ValueError('Regularización inválida.')


def prepare_data(config):
    root = Path(config['dataset_dir']).expanduser().resolve()
    train_rows, train_stats = collect_material_rows(root, 'train', config)
    val_rows, val_stats = collect_material_rows(root, 'val', config)
    train_rows, val_rows, reservations, test_groups = reserve_partitions(root, train_rows, val_rows)
    assert_disjoint(train_rows, val_rows)
    for rows, stats in ((train_rows, train_stats), (val_rows, val_stats)):
        if set(r['label'] for r in rows) != set(range(6)):
            raise ValueError('Train y val deben representar las seis clases tras las exclusiones de seguridad.')
        stats['source_samples_before_group_reservation'] = stats['samples']
        stats['samples'] = len(rows)
        stats['class_counts'] = dict(Counter(CLASSES[r['label']] for r in rows))
        stats['box_crops'] = sum(r['crop'] is not None for r in rows)
        stats['image_label_samples'] = sum(r['crop'] is None for r in rows)
        stats['source_groups'] = len({r['group'] for r in rows})
    fingerprint = hashlib.sha256(json.dumps({'train': train_stats['fingerprint'], 'val': val_stats['fingerprint'],
        'test_source_groups': sorted(test_groups), 'selected_samples': train_rows+val_rows}, sort_keys=True).encode()).hexdigest()
    return root, train_rows, val_rows, {'manifest_sha256': fingerprint, 'train': train_stats, 'val': val_stats,
        'group_reservations': reservations, 'test_used': False, 'test_metadata_read_for_split_protection': True,
        'sample_definition': 'caja recortada o etiqueta explícita de imagen'}


def run_epoch(model, loader, device, optimizer=None, scaler=None, config=None, verify_updates=False,
              max_batches=None):
    training = optimizer is not None
    model.train(training)
    matrix = torch.zeros((6, 6), dtype=torch.long)
    total_loss, samples, steps, skipped = 0., 0, 0, 0
    original = [p.detach().clone() for p in model.parameters()] if training and verify_updates else []
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'] if training else 0.)
    amp = device.type == 'cuda'
    with torch.set_grad_enabled(training):
        iterator = loader if max_batches is None else islice(loader, max_batches)
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Pérdida no finita; no guardar este modelo.')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                if not amp and not torch.isfinite(norm):
                    raise RuntimeError('Gradientes no finitos.')
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    steps += 1
                else:
                    skipped += 1
            prediction = logits.detach().argmax(1)
            matrix += torch.bincount((labels*6+prediction).cpu(), minlength=36).reshape(6, 6)
            total_loss += float(loss.detach())*len(labels)
            samples += len(labels)
    if not samples or (training and steps == 0):
        raise RuntimeError('La época no produjo muestras o no actualizó pesos.')
    changed = any(not torch.equal(p.detach(), old) for p, old in zip(model.parameters(), original)) if original else None
    if original and not changed:
        raise RuntimeError('Los parámetros no cambiaron en la prueba de actualización.')
    tp = matrix.diag().double()
    recall = tp/matrix.sum(1).clamp_min(1)
    precision = tp/matrix.sum(0).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-12)
    return {'loss': total_loss/samples, 'accuracy': float(tp.sum()/samples),
            'macro_f1': float(f1.mean()), 'per_class_recall': dict(zip(CLASSES, recall.tolist())),
            'confusion_matrix': matrix.tolist(), 'class_order': list(CLASSES), 'samples': samples,
            'glass_as_plastic': int(matrix[3, 0]), 'plastic_as_glass': int(matrix[0, 3]),
            'optimizer_steps': steps, 'skipped_steps': skipped, 'parameters_changed': changed}


def atomic_save(path, payload):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state(generator, device):
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(), 'loader': generator.get_state(),
            'numpy': (state[0], state[1].tolist(), *state[2:]),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def _restore_rng(state, generator, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    generator.set_state(state['loader'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])


def train_intake(config, output=None, device_name='cuda', smoke=False, resume=False):
    validate_config(config)
    root, train_rows, val_rows, provenance = prepare_data(config)
    if not smoke and output is None:
        raise ValueError('Indicar una carpeta nueva para los checkpoints fuera del dataset.')
    output = Path(output).expanduser().resolve() if output is not None else None
    if output is not None and (output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError('No se escriben pesos ni archivos de entrenamiento dentro del dataset.')
    if smoke and resume:
        raise ValueError('La prueba corta no reanuda ni publica modelos.')
    if not smoke and output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Salida ocupada: elegir otra o reanudar explícitamente.')
    device = choose_device(device_name)
    seed_everything(config['seed'])
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(config['seed'])
    train_set = IntakeMaterialDataset(train_rows, config, training=True)
    val_set = IntakeMaterialDataset(val_rows, config)
    counts = Counter(r['label'] for r in train_rows)
    sampler = WeightedRandomSampler([1/counts[r['label']] for r in train_rows],
                                    len(train_rows), replacement=True, generator=generator)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], sampler=sampler,
                              num_workers=0, generator=generator, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], num_workers=0)
    model = build_material_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    def schedule(index):
        warmup = config['warmup_epochs']
        if index < warmup:
            return (index+1)/max(1, warmup)
        return .04+.96*(1+math.cos(math.pi*(index-warmup)/max(1, config['epochs']-warmup)))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=config['amp_initial_scale'])
    history, best, best_epoch, bad_epochs, start = [], -1., 0, 0, 1
    if resume:
        state = torch.load(output/'last_training.pth', map_location='cpu', weights_only=True)
        if (state['config'] != config or state['manifest_sha256'] != provenance['manifest_sha256'] or
                state['classes'] != list(CLASSES)):
            raise ValueError('Reanudar requiere la misma arquitectura, datos, particiones y configuración.')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        _restore_rng(state['rng'], generator, device)
        history, best, best_epoch, bad_epochs = state['history'], state['best_f1'], state['best_epoch'], state['bad_epochs']
        start = state['epoch']+1
        if state.get('complete'):
            return {'best_epoch': best_epoch, 'best_macro_f1': best, 'epochs': len(history), 'test_used': False}
    print(json.dumps({'architecture': config['architecture'], 'parameters': sum(p.numel() for p in model.parameters()),
                      'device': str(device), 'data': provenance, 'smoke': smoke}), flush=True)
    if smoke:
        result = run_epoch(model, train_loader, device, optimizer, scaler, config, verify_updates=True, max_batches=2)
        result['validation'] = run_epoch(model, val_loader, device, config=config, max_batches=2)
        result.update(smoke_test=True, checkpoints_written=0, test_used=False)
        print(json.dumps(result), flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, config['epochs']+1):
        tick = time.monotonic()
        train_values = run_epoch(model, train_loader, device, optimizer, scaler, config)
        val_values = run_epoch(model, val_loader, device, config=config)
        row = {'epoch': epoch, 'train': train_values, 'val': val_values,
               'seconds': time.monotonic()-tick, 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(row)
        if val_values['macro_f1'] > best:
            best, best_epoch, bad_epochs = val_values['macro_f1'], epoch, 0
            atomic_save(output/'best_model.pth', {'format_version': 3, 'architecture': config['architecture'],
                'classes': list(CLASSES), 'image_size': config['image_size'], 'config': config,
                'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'manifest_sha256': provenance['manifest_sha256'], 'epoch': epoch,
                'val_macro_f1': best, 'validation': val_values, 'provenance': provenance,
                'initialization': 'random_from_scratch', 'foreign_pretrained_weights': False})
        else:
            bad_epochs += 1
        scheduler.step()
        complete = epoch == config['epochs'] or (epoch >= config['minimum_epochs'] and
                                                 bad_epochs >= config['early_stopping_patience'])
        atomic_save(output/'last_training.pth', {'architecture': config['architecture'], 'classes': list(CLASSES),
            'config': config, 'manifest_sha256': provenance['manifest_sha256'],
            'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'epoch': epoch, 'history': history, 'best_f1': best, 'best_epoch': best_epoch,
            'bad_epochs': bad_epochs, 'rng': _rng_state(generator, device), 'complete': complete})
        print(json.dumps(row), flush=True)
        if epoch >= config['minimum_epochs'] and bad_epochs >= config['early_stopping_patience']:
            break
    return {'best_epoch': best_epoch, 'best_macro_f1': best, 'epochs': len(history), 'test_used': False}
