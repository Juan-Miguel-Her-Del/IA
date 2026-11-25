"""Ejecuta CNN e híbrido secuencialmente; checkpoints fuera del dataset y estado reanudable."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from utils.config import PROJECT_ROOT


def update_status(**values):
    path = PROJECT_ROOT/'config/material_intake_policy.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    policy['training_preparation'].update(values)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def exclusive_training():
    path = PROJECT_ROOT/'runs/.intake_training.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'1')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Ya hay una secuencia de entrenamiento activa.') from exc
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_pair(run_dir, resume=False):
    """Entrena en secuencia la CNN residual y el híbrido con las mismas muestras."""
    run_dir = Path(run_dir).resolve()
    policy = json.loads((PROJECT_ROOT/'config/material_intake_policy.json').read_text(encoding='utf-8'))
    dataset = Path(policy['dataset_root']).resolve()
    if run_dir.is_relative_to(dataset) or dataset.is_relative_to(run_dir):
        raise ValueError('Los checkpoints deben guardarse fuera del dataset.')
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise ValueError('La carpeta ya está ocupada; usar otra o --resume.')
    with exclusive_training():
        run_dir.mkdir(parents=True, exist_ok=True)
        update_status(status='training_pair_running', training_started=True, auto_start=False,
                      active_run_dir=str(run_dir), controller_pid=os.getpid(), active_child_pid=None,
                      started_utc=datetime.now(timezone.utc).isoformat(), completed_models=[], last_error=None)
        completed, fingerprint = [], None
        child = None
        try:
            for name, config_name in (('cnn', 'train_intake_cnn.json'), ('hybrid', 'train_intake_hybrid.json')):
                output = run_dir/name
                args = [sys.executable, '-B', '-u', '-m', 'scripts.train_intake_classifier',
                        '--config', str(PROJECT_ROOT/'config'/config_name),
                        '--output', str(output), '--device', 'cuda']
                if resume and (output/'last_training.pth').exists():
                    args.append('--resume')
                update_status(status='training_'+name, current_model=name, active_output=str(output))
                print(f'START {name} -> {output}', flush=True)
                child = subprocess.Popen(args, cwd=PROJECT_ROOT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update_status(active_child_pid=child.pid)
                code = child.wait()
                update_status(active_child_pid=None)
                if code:
                    raise RuntimeError(f'{name} terminó con código {code}; no se inicia el siguiente modelo.')
                state = torch.load(output/'last_training.pth', map_location='cpu', weights_only=True)
                if not state.get('complete'):
                    raise RuntimeError(f'{name} no publicó una terminación válida.')
                current = state['manifest_sha256']
                # Una huella común permite comparar los modelos sobre el mismo conjunto.
                if fingerprint is not None and current != fingerprint:
                    raise RuntimeError('Los dos modelos no usaron las mismas muestras; no comparar sus resultados.')
                fingerprint = current
                completed.append({'model': name, 'output': str(output)})
                update_status(completed_models=completed)
                print(json.dumps({'model': name, 'epochs': state['epoch'], 'best_epoch': state['best_epoch'],
                                  'best_validation_macro_f1': state['best_f1'], 'test_used': False}), flush=True)
                del state
            update_status(status='classification_pair_complete', current_model=None, active_output=None,
                          active_child_pid=None, controller_pid=None,
                          finished_utc=datetime.now(timezone.utc).isoformat(), test_still_reserved=True)
            print('COMPLETE: CNN e híbrido terminados; modelo activo intacto y test reservado.', flush=True)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            update_status(status='training_pair_interrupted', active_child_pid=None, controller_pid=None,
                          last_error=str(exc), completed_models=completed)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    run_pair(arguments.run_dir, arguments.resume)
