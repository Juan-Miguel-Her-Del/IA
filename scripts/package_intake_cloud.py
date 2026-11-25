"""Package only the active hybrid, D2 and inference source; never training data or secrets."""
import argparse
import json
from pathlib import Path
import zipfile

from utils.config import PROJECT_ROOT
from utils.dataset import sha256


def package(output):
    root = PROJECT_ROOT
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Choose a new archive path; existing exports are preserved.')
    settings = json.loads((root/'config/app.json').read_text(encoding='utf-8'))
    if settings.get('material_candidate_checkpoint') or settings.get('material_candidate_weight'):
        raise ValueError('Cloud release must use the hybrid alone.')
    paths = [Path(settings['checkpoint']), Path(settings['detector_checkpoint'])]
    if any(p.is_absolute() or '..' in p.parts for p in paths):
        raise ValueError('Cloud settings must use relative checkpoint paths.')
    manifest = json.loads((root/'models/runtime/manifest.json').read_text(encoding='utf-8'))
    if manifest[paths[0].name]['architecture'] != 'VisionHybrid-v1':
        raise ValueError('The hybrid has not been activated.')
    active_manifest = {}
    for path in paths:
        meta = manifest[path.name]
        if sha256(root/path) != meta['sha256'] or meta.get('foreign_pretrained_weights') is not False:
            raise ValueError('Active checkpoint integrity/provenance failed.')
        active_manifest[path.name] = meta
    files = set()
    for path in (root/'app').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.css', '.js', '.html'):
            files.add(path.relative_to(root).as_posix())
    for name in ('__init__', 'compact', 'data', 'detection_data', 'detection_metrics',
                 'detector_own', 'hybrid', 'model', 'prediction', 'runtime'):
        files.add(f'training/{name}.py')
    files.update(['utils/__init__.py', 'utils/config.py', 'utils/dataset.py', 'scripts/__init__.py',
                  'scripts/verify_runtime.py', 'config/app.json', 'models/runtime/reserved_hashes.json',
                  'requirements.txt', 'requirements-app.txt', 'requirements-data.txt', '.gitignore'])
    files.update(p.as_posix() for p in paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files): archive.write(root/name, name)
        archive.writestr('models/runtime/manifest.json', json.dumps(active_manifest, indent=2)+'\n')
        config = (root/'.streamlit/config.toml').read_text(encoding='utf-8')
        # Let the hosting platform choose its own network binding.
        config = '\n'.join(line for line in config.splitlines() if not line.lstrip().startswith('address ='))+'\n'
        archive.writestr('.streamlit/config.toml', config)
        archive.writestr('README.md', '# Vision-IA: híbrido CNN + transformer\n\n'
            'Sube el contenido de esta carpeta al repositorio que usarás en Streamlit Community Cloud.\n'
            'Archivo principal: `app/main.py`. Dependencias: `requirements.txt`.\n\n'
            'Los pesos propios del híbrido y D2 están en `models/runtime/`. '
            'La aplicación realiza inferencia en CPU y recibe la cámara desde el navegador. '
            'No requiere el dataset ni ejecutar entrenamientos.\n\n'
            'Verificación local: `python -m scripts.verify_runtime`.\n'
            'Inicio local: `python -m streamlit run app/main.py`.\n')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
                      'classifier': 'VisionHybrid-v1', 'weight_files': [p.as_posix() for p in paths],
                      'datasets_included': False, 'secrets_included': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package(args.output)
