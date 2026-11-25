"""Continúa el entrenamiento de D2 con las cajas aprobadas del conjunto COCO."""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.intake_data import IntakeMaterialDataset
from training.intake_training import prepare_data, atomic_save
from training.detection_data import letterbox, image_tensor, collate_detection
from training.detection_metrics import nms, spatial_metrics
from training.detector_own import ARCHITECTURE, VisionLocaliza, decode_predictions
from scripts.train_detector_refine import complete_object_loss


def group_detection_rows(rows):
    """Agrupa todas las cajas de una foto para entrenar con la escena completa."""
    grouped = {}
    for row in rows:
        # Las fotos etiquetadas solo por material se reservan para clasificación.
        if row['crop'] is None:
            continue
        key = row['path']
        if key not in grouped:
            grouped[key] = dict(row, boxes=[], materials=[])
        x, y, w, h = row['crop']
        # Pasa del formato COCO (origen y dimensiones) al de dos esquinas.
        grouped[key]['boxes'].append([x, y, x+w, y+h])
        grouped[key]['materials'].append(row['label'])
    return list(grouped.values())


class IntakeDetectionDataset(Dataset):
    """Entrega la imagen completa, sus cajas ajustadas y el índice de la muestra."""

    def __init__(self, rows, config, training=False):
        self.rows, self.config, self.training = rows, config, training
        self.reader = IntakeMaterialDataset([], dict(config, crop_margin=.12), False)

    def __len__(self):
        return len(self.rows)

    def rgb(self, index):
        row = self.rows[index]
        with Image.open(BytesIO(self.reader._bytes(row))) as image:
            image.load()
            if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('Image dimensions/orientation do not match its boxes.')
            return np.asarray(image.convert('RGB'))

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = self.rgb(index)
        boxes = np.asarray(row['boxes'], dtype=np.float32)
        canvas, boxes, _ = letterbox(rgb, self.config['image_size'], boxes,
                                     self.training, self.config)
        return image_tensor(canvas), torch.from_numpy(boxes.copy()), index


def detector_validation(model, loader, device, confidence=.27):
    """Compara las cajas predichas con las anotadas en el conjunto de validación."""
    model.eval()
    records = []
    with torch.inference_mode():
        for images, boxes, ids, _, _ in loader:
            # Se retienen puntuaciones bajas para construir la curva de AP.
            predictions = decode_predictions(model(images.to(device)), confidence=.001)
            size = images.shape[-1]
            for ident, actual, (pb, ps) in zip(ids, boxes, predictions):
                pb, ps = pb.cpu().numpy().clip(0, size), ps.cpu().numpy()
                valid = ((pb[:, 2:] - pb[:, :2]) >= 2).all(1) & np.isfinite(pb).all(1) & np.isfinite(ps)
                pb, ps = pb[valid], ps[valid]
                keep = nms(pb, ps, .45, 100)
                materials = loader.dataset.rows[ident]['materials']
                # El material se conserva solo para medir recobrado por categoría de origen.
                from utils.config import CLASSES
                truth = [{'box': b.tolist(), 'category': CLASSES[label]} for b, label in zip(actual, materials)]
                records.append({'image_id': ident, 'size': [size, size], 'truth': truth,
                                'predictions': [{'box': pb[i].tolist(), 'score': float(ps[i])} for i in keep]})
    return spatial_metrics(records, confidence)


def train_detector(config, parent, output, smoke=False):
    """Refina en CUDA los pesos propios de D2 y selecciona el mejor mAP en validación."""
    root, train, val, data_provenance = prepare_data(config)
    output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Training artifacts must be outside the dataset.')
    if output.exists() and any(output.iterdir()) and not smoke:
        raise ValueError('Refusing to overwrite an existing run.')
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the local detector continuation.')
    parent = Path(parent)
    # El detector activo continúa un D2 previo, originalmente entrenado desde cero.
    ck = torch.load(parent, map_location='cpu', weights_only=True)
    provenance = deepcopy(ck['provenance'])
    if (ck['architecture'] != ARCHITECTURE or provenance.get('pretrained') is not False or
            provenance.get('initialization') != 'random' or
            provenance.get('official_detector_components') is not False):
        raise ValueError('Only our own randomly initialized D2 parent is allowed.')
    provenance.update(own_parent_checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                      continuation_of_own_random_initialization=True, dataset=data_provenance,
                      test_used_for_selection=False, annotation_exception='Existing reviewed CLICK/SAM3 only')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed']); torch.set_num_threads(2); cv2.setNumThreads(1)
    device = torch.device('cuda')
    train_rows, val_rows = group_detection_rows(train), group_detection_rows(val)
    train_set = IntakeDetectionDataset(train_rows, config, True)
    val_set = IntakeDetectionDataset(val_rows, config)
    counts = Counter(r['label'] for r in train_rows)
    # Muestreo inverso a la frecuencia del material de la primera caja de cada foto.
    generator = torch.Generator().manual_seed(config['seed'])
    sampler = WeightedRandomSampler([1/counts[r['label']] for r in train_rows], len(train_rows),
                                    replacement=True, generator=generator)
    loader = DataLoader(train_set, batch_size=config['batch_size'], sampler=sampler,
                        num_workers=0, collate_fn=collate_detection, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], num_workers=0,
                            collate_fn=collate_detection)
    model = VisionLocaliza().to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    # EMA mantiene una copia suavizada de los pesos y se usa al validar.
    ema = deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # D2 entrena en FP32 para evitar pasos omitidos por AMP y estabilizar la pérdida de cajas.
    print(json.dumps({'phase': 'detector_data', 'train_images': len(train_rows),
                      'train_boxes': sum(len(r['boxes']) for r in train_rows),
                      'val_images': len(val_rows), 'val_boxes': sum(len(r['boxes']) for r in val_rows),
                      'parent_sha256': provenance['own_parent_checkpoint_sha256']}), flush=True)
    if smoke:
        # Dos pasos comprueban gradientes finitos y una actualización efectiva de los pesos.
        model.train()
        before = model.stem[0].weight.detach().clone()
        for step, (images, _, _, targets, collisions) in enumerate(loader):
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = complete_object_loss(model(images.to(device)), targets, collisions, config)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('Nonfinite smoke update.')
            optimizer.step()
            if step == 1: break
        changed = not torch.equal(before, model.stem[0].weight)
        if not changed: raise RuntimeError('Detector weights did not update.')
        result = {'smoke_passed': True, 'optimizer_steps': 2, 'parameters_changed': changed,
                  'test_images_opened': 0, 'checkpoints_saved': False}
        print(json.dumps(result), flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    baseline = detector_validation(ema, val_loader, device)
    # El modelo de partida también compite como época cero.
    best, best_epoch, bad_epochs, steps, history = baseline['mAP50_95'], 0, 0, 0, []
    def save_best(epoch, metrics):
        atomic_save(output/'best.pth', {'architecture': ARCHITECTURE,
            'state_dict': {k: v.detach().cpu() for k, v in ema.state_dict().items()},
            'config': config, 'provenance': provenance, 'image_size': config['image_size'],
            'epoch': epoch, 'validation': metrics})
    save_best(0, baseline)
    print(json.dumps({'phase': 'detector_baseline', 'validation': baseline}), flush=True)
    for epoch in range(1, config['epochs']+1):
        tick = time.monotonic(); model.train(); total, updates = 0., 0
        # Reduce gradualmente la tasa de aprendizaje con una curva coseno.
        lr = config['learning_rate']*(.1+.9*(1+np.cos(np.pi*(epoch-1)/config['epochs']))/2)
        for group in optimizer.param_groups: group['lr'] = float(lr)
        for images, _, _, targets, collisions in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss, _ = complete_object_loss(model(images.to(device)), targets, collisions, config)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite detector loss; previous weights preserved.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite detector gradients.')
            optimizer.step(); updates += 1; steps += 1; total += float(loss.detach())
            with torch.no_grad():
                decay = min(.995, 1-1/(steps+10))
                for a, b in zip(ema.state_dict().values(), model.state_dict().values()):
                    if a.dtype.is_floating_point: a.mul_(decay).add_(b.detach(), alpha=1-decay)
                    else: a.copy_(b)
        if not updates: raise RuntimeError('No effective detector updates.')
        metrics = detector_validation(ema, val_loader, device)
        # mAP50_95 premia detecciones correctas bajo varios niveles de ajuste espacial.
        if metrics['mAP50_95'] > best:
            best, best_epoch, bad_epochs = metrics['mAP50_95'], epoch, 0
            save_best(epoch, metrics)
        else: bad_epochs += 1
        complete = epoch == config['epochs'] or (epoch >= config['minimum_epochs'] and bad_epochs >= config['early_stopping_patience'])
        row = {'epoch': epoch, 'loss': total/updates, 'optimizer_steps': updates,
               'seconds': time.monotonic()-tick, 'validation': metrics}
        history.append(row)
        # best conserva los pesos promediados. last registra el estado de la última época.
        atomic_save(output/'last.pth', {'architecture': ARCHITECTURE, 'state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': config, 'provenance': provenance, 'history': history, 'epoch': epoch,
                    'best_epoch': best_epoch, 'best_mAP50_95': best, 'baseline': baseline,
                    'complete': complete})
        print(json.dumps(dict(row, best_epoch=best_epoch, complete=complete)), flush=True)
        if complete: break
    return {'best_epoch': best_epoch, 'best_mAP50_95': best, 'complete': True}
