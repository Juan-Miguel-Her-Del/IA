"""Continúa exclusivamente nuestra CNN B en una candidata separada con repaso."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image,ImageEnhance,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from training.model import ARCHITECTURE,VisionCNN

CLASSES=('plastico','papel','carton','vidrio','metal','organicos')
PARENT_SHA='10f25c75048b45459247e3a7cc51204e2c6413ac7735be0a46821622cd4d931c'
OPTIONS={'seed':49,'epochs':16,'patience':5,'batch_size':64,'learning_rate':.0001,
         'supplement_fraction':.25,'supplement_sampling':'equal material mass, exclude two isolated organic crops',
         'max_original_macro_F1_drop':.005,'test_used':False}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(payload,path):
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


class MaterialData(Dataset):
    def __init__(self,root,rows,training=False,limit=None):
        self.rows=rows[:limit] if limit else rows;self.training=training;self.cache=[]
        for i,row in enumerate(self.rows):
            path=Path(root)/row['path']
            if sha(path)!=row['sha256']:raise ValueError('Imagen modificada: '+str(path))
            with Image.open(path) as im:
                self.cache.append(np.array(ImageOps.exif_transpose(im).convert('RGB').resize((128,128),Image.Resampling.BILINEAR)))
            if (i+1)%1000==0:print('cached',i+1,flush=True)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        image=Image.fromarray(self.cache[index])
        if self.training:
            if random.random()<.5:image=ImageOps.mirror(image)
            image=image.rotate(random.uniform(-10,10),resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
        tensor=torch.from_numpy(np.array(image).transpose(2,0,1).copy()).float()/127.5-1.
        return tensor,CLASSES.index(self.rows[index]['class'])


def metrics(truth,predicted):
    matrix=np.zeros((6,6),dtype=int)
    for t,p in zip(truth,predicted):matrix[t,p]+=1
    tp=matrix.diagonal();f1=2*tp/np.maximum(matrix.sum(0)+matrix.sum(1),1)
    return {'accuracy':float(tp.sum()/max(matrix.sum(),1)),'macro_F1':float(f1.mean()),
            'plastic_recall':float(tp[0]/max(matrix[0].sum(),1)),'images':int(matrix.sum()),
            'confusion_matrix':matrix.tolist()}


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();truth=[];predicted=[]
    for images,labels in loader:
        predicted.extend(model(images.to(device)).argmax(1).cpu().tolist());truth.extend(labels.tolist())
    result=metrics(truth,predicted)
    indices=[i for i,r in enumerate(loader.dataset.rows) if r.get('category')=='Clear plastic bottle']
    if indices:result['clear_bottle_accuracy']=sum(predicted[i]==truth[i] for i in indices)/len(indices)
    return result


def train(data_root,parent,output,smoke=False,resume=False,device=None):
    root=Path(data_root);parent=Path(parent);run=Path(output)
    if run.exists() and not resume:raise ValueError('No sobrescribir una ejecución anterior')
    if sha(parent)!=PARENT_SHA:raise ValueError('Se exige el B original verificado del proyecto')
    ck=torch.load(parent,map_location='cpu',weights_only=True)
    assert ck['architecture']==ARCHITECTURE and ck['classes']==list(CLASSES)
    assert ck['config'].get('initialization')=='random_from_scratch'
    assert sha(root/'original/manifest.csv')==ck['manifest_sha256']
    audit=json.loads((root/'supplement/audit.json').read_text())
    assert audit['test_images_opened']==0 and audit['foreign_annotations'] is False
    with (root/'original/manifest.csv').open() as stream:original=list(csv.DictReader(stream))
    supplement=json.loads((root/'supplement/manifest.json').read_text())
    assert {r['split'] for r in supplement}=={'train','val'}
    assert not ({r['image_id'] for r in supplement if r['split']=='train'} & {r['image_id'] for r in supplement if r['split']=='val'})
    from training.runtime import choose_device
    device=choose_device(device or ('auto' if smoke else 'cuda'))
    torch.set_num_threads(2);random.seed(OPTIONS['seed']);np.random.seed(OPTIONS['seed']);torch.manual_seed(OPTIONS['seed'])
    model=VisionCNN(dropout=ck['config']['dropout']).to(device);model.load_state_dict(ck['model_state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=OPTIONS['learning_rate'],weight_decay=1e-4)
    train_original=MaterialData(root/'original',[r for r in original if r['split']=='train'],True,2 if smoke else None)
    train_extra=MaterialData(root/'supplement',[r for r in supplement if r['split']=='train'],True,2 if smoke else None)
    data=torch.utils.data.ConcatDataset([train_original,train_extra])
    run.mkdir(parents=True,exist_ok=resume)
    provenance={'initialization':'random_from_scratch','continued_own_classifier_B':True,'own_parent_sha256':PARENT_SHA,
                'foreign_pretrained_weights':False,'official_architecture_imported':False,
                'supplement_manifest_sha256':sha(root/'supplement/manifest.json'),'test_used':False,'config':OPTIONS}
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    criterion=nn.CrossEntropyLoss(label_smoothing=.05)
    def payload(epoch,validation=None):
        result=deepcopy(ck);result['model_state_dict']={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        result.update(epoch=epoch,original_parent_epoch=ck['epoch'],provenance=provenance,refinement_validation=validation)
        if validation:result['val_macro_f1']=validation['original']['macro_F1']
        result['torch_version']=str(torch.__version__)
        return result
    if smoke:
        images,labels=next(iter(DataLoader(data,batch_size=4)))
        model.train();loss=criterion(model(images.to(device)),labels.to(device));loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        before=model.classifier[-1].weight.detach().clone();optimizer.step();assert not torch.equal(before,model.classifier[-1].weight)
        save(payload(0),run/'iteration.pth')
        restored=VisionCNN(dropout=ck['config']['dropout']);restored.load_state_dict(torch.load(run/'iteration.pth',map_location='cpu',weights_only=True)['model_state_dict'])
        assert all(torch.equal(t.cpu(),restored.state_dict()[n]) for n,t in model.state_dict().items())
        report={'passed':True,'device':str(device),'loss':float(loss.detach()),'checkpoint_reload_exact':True,'test_used':False}
        (run/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    smokes=list(run.parent.glob('smoke_material*/smoke.json'))
    assert any(json.loads(p.read_text()).get('passed') and json.loads((p.parent/'provenance.json').read_text())==provenance for p in smokes)
    val_original=MaterialData(root/'original',[r for r in original if r['split']=='val'])
    val_extra=MaterialData(root/'supplement',[r for r in supplement if r['split']=='val'])
    loaders={n:DataLoader(d,batch_size=128) for n,d in [('original',val_original),('objects',val_extra)]}
    def validation():return {n:evaluate(model,l,device) for n,l in loaders.items()}
    def score(m):return .5*m['original']['macro_F1']+.4*m['objects']['macro_F1']+.1*m['objects']['clear_bottle_accuracy']
    baseline=validation();best=score(baseline);best_epoch=0;start=0;history=[]
    if resume:
        last=torch.load(run/'last.pth',map_location=device,weights_only=True)
        assert last['provenance']==provenance
        model.load_state_dict(last['model_state_dict']);optimizer.load_state_dict(last['optimizer'])
        best=last['best_score'];best_epoch=last['best_epoch'];start=last['epoch']
        baseline=json.loads((run/'baseline.json').read_text());history=json.loads((run/'history.json').read_text())
        torch.set_rng_state(last['torch_rng'].cpu());random.setstate(last['python_rng'])
    else:
        (run/'baseline.json').write_text(json.dumps(baseline,indent=2));save(payload(0,baseline),run/'best.pth')
    original_weights=[(1-OPTIONS['supplement_fraction'])/len(train_original)]*len(train_original)
    focus=[2. if r['category'] in ('Clear plastic bottle','Other plastic bottle') else 1. for r in train_extra.rows]
    totals={label:sum(w for r,w in zip(train_extra.rows,focus) if r['class']==label) for label in CLASSES if label!='organicos'}
    weights=original_weights+[OPTIONS['supplement_fraction']*w/totals[r['class']]/len(totals)
                              if r['class'] in totals else 0. for r,w in zip(train_extra.rows,focus)]
    loader=DataLoader(data,batch_size=OPTIONS['batch_size'],sampler=WeightedRandomSampler(weights,len(train_original),replacement=True),num_workers=0)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True;print('Stopping after epoch',flush=True)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    print('BASELINE',json.dumps(baseline),flush=True)
    for epoch in range(start+1,OPTIONS['epochs']+1):
        began=time.monotonic();model.train();total=0
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        for images,labels in loader:
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(images.to(device)),labels.to(device))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; previous checkpoint retained')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
            optimizer.step();total+=float(loss.detach())
        current=validation()
        old_matrix=np.array(baseline['original']['confusion_matrix']);matrix=np.array(current['original']['confusion_matrix'])
        per_class_guard=bool(np.all(matrix.diagonal()/matrix.sum(1)>=old_matrix.diagonal()/old_matrix.sum(1)-.08))
        guard=(current['original']['macro_F1']>=baseline['original']['macro_F1']-OPTIONS['max_original_macro_F1_drop'] and per_class_guard and
               current['original']['plastic_recall']>=baseline['original']['plastic_recall']-.03)
        if guard and score(current)>best:
            best=score(current);best_epoch=epoch;save(payload(epoch,current),run/'best.pth')
        row={'epoch':epoch,'best_epoch':best_epoch,'score':best,'loss':total/len(loader),
             'seconds':time.monotonic()-began,'validation':current,'guard':guard}
        history.append(row);(run/'history.json').write_text(json.dumps(history,indent=2))
        last=payload(epoch,current);last.update(optimizer=optimizer.state_dict(),best_score=best,best_epoch=best_epoch,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate())
        save(last,run/'last.pth');(run/'status.json').write_text(json.dumps({'epoch':epoch,'best_epoch':best_epoch,'stopped':stopped,'complete':False}))
        print(json.dumps(row),flush=True)
        if stopped or epoch-best_epoch>=OPTIONS['patience']:break
    (run/'status.json').write_text(json.dumps({'epoch':epoch,'best_epoch':best_epoch,'stopped':stopped,'complete':not stopped}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('auto','cuda','cpu'))
    a=p.parse_args();train(a.root,a.parent,a.output,a.smoke,a.resume,a.device)
