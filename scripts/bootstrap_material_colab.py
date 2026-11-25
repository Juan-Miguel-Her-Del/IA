"""Flujo histórico. Recibe una carpeta con el paquete B y el suplemento previo."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch


def launch(root, work):
    root=Path(root).resolve()
    assert torch.cuda.is_available(),'PyTorch debe detectar una GPU CUDA'
    gpu=torch.cuda.get_device_name(0)
    source=root/'material_refinement/source_balanced_v2'
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    for directory in ['scripts','training']:
        shutil.copytree(source/directory,work/'source_balanced_v2'/directory,dirs_exist_ok=True)
    parent=source/'classifier_B_original.pth'
    assert hashlib.sha256(parent.read_bytes()).hexdigest()=='10f25c75048b45459247e3a7cc51204e2c6413ac7735be0a46821622cd4d931c'
    print('CUDA disponible. Preparando solo train/val...',flush=True)
    archive=root/'packages/B/613896da0e225f390081/Vision-IA_Colab_B.zip'
    with zipfile.ZipFile(archive) as z:
        prefix='Vision-IA/data/experiments/B/processed/'
        for name in z.namelist():
            if not name.startswith(prefix):continue
            rel=name[len(prefix):]
            if not rel or not (rel.startswith('train/') or rel.startswith('val/') or rel in ('manifest.csv','split_summary.json')):continue
            assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
            out=work/'data/original'/rel;out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():out.write_bytes(z.read(name))
    with zipfile.ZipFile(root/'material_refinement/material_crops.zip') as z:
        for name in z.namelist():
            assert '..' not in Path(name).parts and not Path(name).is_absolute()
            if not (name.startswith('train/') or name.startswith('val/') or name in ('manifest.json','audit.json')):raise ValueError(name)
            out=work/'data/supplement'/name;out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():out.write_bytes(z.read(name))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    runs=root/'material_refinement/runs';runs.mkdir(exist_ok=True)
    args=[sys.executable,'-u','-m','scripts.train_material_candidate','--root',str(work/'data'),'--parent',str(parent)]
    smoke=runs/('smoke_material_T4_balanced_'+stamp)
    subprocess.run(args+['--output',str(smoke),'--smoke'],cwd=work/'source_balanced_v2',check=True)
    result=json.loads((smoke/'smoke.json').read_text());assert result['passed'] and result['device']=='cuda'
    run=runs/('BMaterialsBalanced_'+stamp)
    log=runs/('BMaterialsBalanced_'+stamp+'.log')
    with log.open('w') as stream:
        process=subprocess.Popen(args+['--output',str(run)],cwd=work/'source_balanced_v2',stdout=stream,stderr=subprocess.STDOUT)
    record={'run':str(run),'log':str(log),'pid':process.pid,'GPU':gpu,'G_workspace_verified':True,'smoke':result,
            'original_B_unchanged':True,'test_used':False}
    (root/'material_refinement/launch_status_balanced.json').write_text(json.dumps(record,indent=2))
    print('ENTRENAMIENTO INICIADO',json.dumps(record),flush=True)
    return process


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root',required=True,type=Path)
    parser.add_argument('--work-dir',type=Path,default=Path('runs/historical_material_work'))
    args=parser.parse_args()
    launch(args.shared_root,args.work_dir)
