"""Espera el par local y evalúa sus checkpoints; nunca cambia el runtime activo."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from utils.config import PROJECT_ROOT
from utils.quality_route import compare_reports, file_hash, read_json, write_new_json


def finish(args):
    plan = read_json(args.plan)
    baseline = read_json(args.baseline)
    selected = [r for r in plan['experiments'] if r['focus'] == 'material' and r['seed'] == args.seed]
    if len(selected) != 2 or {r['variant'] for r in selected} != {'control', 'candidate'}:
        raise ValueError('Se necesita un par control/candidata de la misma semilla.')
    folders = {r['name']: Path(r['training_args'][r['training_args'].index('--output') + 1]) for r in selected}
    deadline = time.monotonic() + args.timeout_hours * 3600
    print('Esperando ejecuciones completas:', list(folders), flush=True)
    while not all((folder / 'completed.json').is_file() for folder in folders.values()):
        if time.monotonic() > deadline:
            raise TimeoutError('No terminó el par dentro del plazo. Se conservan ejecuciones y checkpoints.')
        time.sleep(15)
    # Evitar comparar código/configuraciones modificadas durante la ejecución.
    snapshot = Path(args.plan).parent / 'source_snapshot/manifest.json'
    for name, expected in read_json(snapshot).items():
        if file_hash(PROJECT_ROOT / name) != expected:
            raise ValueError(f'El código o configuración cambió: {name}. Revisar y repetir la referencia si corresponde.')
    policy = read_json(PROJECT_ROOT / 'config/quality_route.json')
    for key, expected in baseline['checkpoint_hashes'].items():
        path = PROJECT_ROOT / baseline['config'][key]
        if file_hash(path) != expected:
            raise ValueError('Cambió un checkpoint de referencia; no se sustituye ni se mezcla la evidencia.')
    results, initial_states = [], set()
    for trial in selected:
        folder = folders[trial['name']]
        completion = read_json(folder / 'completed.json')
        config = read_json(trial['config'])
        provenance = completion['provenance']
        if (completion['completed_epochs'] != config['epochs'] or provenance.get('smoke_test') is not False
                or provenance.get('test_used') is not False or provenance.get('pretrained') is not False
                or provenance.get('initialization') != 'random_from_scratch'):
            raise ValueError(f"Ejecución incompleta o procedencia incorrecta: {trial['name']}")
        history = read_json(folder / 'history.json')
        if any(row['optimizer_steps'] < 1 or not math.isfinite(row['loss']) for row in history):
            raise ValueError('Una época no actualizó pesos o tiene pérdida no finita.')
        initial_states.add(provenance['initial_state_sha256'])
        for kind in ('best_scene', 'best_balanced'):
            checkpoint = folder / (kind + '.pth')
            if not checkpoint.is_file():
                continue
            name = trial['name'] + '_' + kind
            settings = dict(baseline['config'], material_candidate_checkpoint=str(checkpoint.resolve()))
            config_path, report_path = args.output / (name + '_app.json'), args.output / (name + '.json')
            write_new_json(config_path, settings)
            print('Evaluando', name, flush=True)
            subprocess.run([sys.executable, '-B', '-m', 'scripts.evaluate_runtime', '--config', str(config_path),
                            '--scenes', str(args.scenes), '--labels', str(args.labels), '--output', str(report_path)],
                           cwd=PROJECT_ROOT, check=True)
            report = read_json(report_path)
            comparison = compare_reports(baseline, report, policy, 'material')
            write_new_json(args.output / (name + '_comparison.json'), comparison)
            m = report['metrics']
            results.append({'name': name, 'report': str(report_path), 'offline_pass': comparison['offline_pass'],
                            'original_accuracy': m['original']['accuracy'], 'original_macro_f1': m['original']['macro_f1'],
                            'scene_accuracy': m['scene']['accuracy'], 'scene_macro_f1_present': m['scene']['macro_f1_present'],
                            'joint_recall': m['joint']['recall'], 'single_object_scenes': m['single_object_scenes']})
    if len(initial_states) != 1:
        raise ValueError('El control y la candidata no partieron del mismo estado aleatorio.')
    summary = {'seed': args.seed, 'same_initial_state': True, 'results': results,
               'runtime_replaced': False, 'test_used': False,
               'note': 'Piloto de una semilla sobre validación. Confirmar con más semillas y evaluación de campo antes de afirmar mejora general.'}
    write_new_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--timeout-hours', type=float, default=6.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        finish(args)
    except Exception as exc:
        write_new_json(args.output / 'error.json', {'error': str(exc), 'runtime_replaced': False})
        raise
