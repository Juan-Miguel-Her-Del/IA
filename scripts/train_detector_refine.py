"""Continue ONLY our own D2 weights; full-object loss and two audited val domains."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler

from scripts.train_detector_own import atomic_save,fingerprint,validate
from training.detection_data import DetectionDataset,collate_detection
from training.detector_own import ARCHITECTURE,STRIDES,VisionLocaliza,decode_head,detection_loss,aligned_iou


def candidate_quality_loss(outputs,targets,topk=12):
    """La calidad de una propuesta mide IoU con el objeto entero, no con su dibujo."""
    total=sum(o.sum()*0 for o in outputs);count=0
    for raw,stride in zip(outputs,STRIDES):
        decoded=decode_head(raw.float(),stride).permute(0,2,3,1).detach()
        for i in range(len(raw)):
            truth=torch.cat([t['boxes'][i][t['positive'][i]] for t in targets])
            scores,indices=raw[i,0].float().sigmoid().detach().flatten().topk(min(topk,raw.shape[2]*raw.shape[3]))
            indices=indices[scores>.05]
            if not len(indices):continue
            boxes=decoded[i].reshape(-1,4)[indices]
            expected=boxes.new_zeros(len(boxes))
            if len(truth):
                intersection=(torch.minimum(boxes[:,None,2:],truth[None,:,2:])-torch.maximum(boxes[:,None,:2],truth[None,:,:2])).clamp(min=0).prod(-1)
                areas=(boxes[:,2:]-boxes[:,:2]).clamp(min=0).prod(-1)
                actual_areas=(truth[:,2:]-truth[:,:2]).clamp(min=0).prod(-1)
                expected=(intersection/(areas[:,None]+actual_areas[None,:]-intersection).clamp(min=1e-6)).max(1).values
            total=total+F.binary_cross_entropy_with_logits(raw[i,5].float().flatten()[indices],expected,reduction='sum')
            count+=len(indices)
    return total/max(count,1)


def complete_object_loss(outputs,targets,collisions,config=None):
    config=config or {}
    loss,parts=detection_loss(outputs,prepared_targets=(targets,collisions))
    edge=sum(o.sum()*0 for o in outputs);missing=edge;geometry=edge;negative_quality=edge
    count=max(1,sum(int(t['positive'].sum()) for t in targets))
    for raw,target,stride in zip(outputs,targets,STRIDES):
        positive=target['positive']
        if positive.any():
            predicted=decode_head(raw.float(),stride).permute(0,2,3,1)[positive]
            actual=target['boxes'][positive]
            size=(actual[:,2:]-actual[:,:2]).clamp(min=4)
            scale=torch.cat((size,size),1)
            edge=edge+F.smooth_l1_loss((predicted-actual)/scale,torch.zeros_like(actual),reduction='sum')
            deficits=torch.cat(((predicted[:,:2]-actual[:,:2]).clamp(min=0),
                                (actual[:,2:]-predicted[:,2:]).clamp(min=0)),1)/scale
            missing=missing+deficits.sum()
            geometry=geometry+(1-aligned_iou(predicted,actual,generalized=True)).sum()
        # Quality is also supervised on a few confident false center candidates.
        if config.get('negative_quality_weight',.2)>0:
            absent=(target['heat']<.05)&(~positive)
            candidates=raw[:,0].float().sigmoid().detach().masked_fill(~absent,0).flatten()
            k=min(candidates.numel(),max(4,int(positive.sum())*3))
            scores,indices=candidates.topk(k)
            indices=indices[scores>.05]
            if len(indices):
                logits=raw[:,5].float().flatten()[indices]
                negative_quality=negative_quality+F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='sum')
    extras=(config.get('extent_edge_weight',2)*edge+config.get('extent_missing_weight',1)*missing+
            config.get('extent_GIoU_weight',1.5)*geometry+config.get('negative_quality_weight',.2)*negative_quality)/count
    if config.get('candidate_quality_weight',0):
        extras=extras+config['candidate_quality_weight']*candidate_quality_loss(outputs,targets)
    return loss+extras,dict(parts,edge=float(edge.detach()/count),missing_extent=float(missing.detach()/count))


def domain_validation(model,loaders,device,config=None):
    metrics={name:validate(model,loader,device) for name,loader in loaders.items()}
    weights=(config or {}).get('selection_weights',{})
    metrics['balanced_AP95']=sum(weights.get(n,1)*m['mAP50_95'] for n,m in metrics.items())/sum(weights.get(n,1) for n in metrics)
    return metrics


def train(config_path,root,parent_path,output,smoke=False,resume=None,device=None):
    config=json.loads(Path(config_path).read_text());root=Path(root);run=Path(output)
    if run.exists() and not resume:raise ValueError('Do not overwrite another run')
    if resume and Path(resume).resolve().parent!=run.resolve():raise ValueError('Resume only this run')
    assert config['architecture']==ARCHITECTURE and config['pretrained'] is False
    from training.runtime import choose_device
    device=choose_device(device or ('auto' if smoke else 'cuda'))
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(2);cv2.setNumThreads(1)
    ck=torch.load(parent_path,map_location='cpu',weights_only=True)
    provenance=deepcopy(ck['provenance'])
    assert ck['architecture']==ARCHITECTURE and provenance['initialization']=='random'
    assert provenance['pretrained'] is False and provenance['official_detector_components'] is False
    audit=json.loads((root/'audit.json').read_text())
    assert audit['original_rows_preserved'] and audit['original_test_unchanged']
    assert audit['cross_split_perceptual_candidates']==0 and audit['minimum_polygon_mask_IoU']>=.95
    assert audit['test_images_opened']==0 and audit['foreign_weights_or_foundation_annotations'] is False
    rows=json.loads((root/'manifest.json').read_text())
    old=json.loads((root/'parent_manifest.json').read_text())
    assert rows[:len(old)]==old
    assert hashlib.sha256((root/'parent_manifest.json').read_bytes()).hexdigest()==audit['parent_manifest_sha256']
    assert [r for r in rows if r['split']=='test']==[r for r in old if r['split']=='test']
    run.mkdir(parents=True,exist_ok=bool(resume))
    provenance.update(own_parent_checkpoint_sha256=hashlib.sha256(Path(parent_path).read_bytes()).hexdigest(),
        continuation_of_own_random_initialization=True,config=config,
        dataset_audit_sha256=hashlib.sha256((root/'audit.json').read_bytes()).hexdigest(),
        selection=config['selection'],test_used_for_selection=False)
    model=VisionLocaliza().to(device);model.load_state_dict(ck['state_dict'])
    if config.get('train_box_quality_only',False):
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('heads.') and ('.box.' in name or '.quality.' in name))
        provenance['frozen_features_and_presence']=True
        provenance['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    amp=device.type=='cuda' and config.get('native_amp',False)
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=512.)
    resumed=None;start_epoch=0
    if resume:
        resumed=torch.load(resume,map_location=device,weights_only=True)
        assert resumed['architecture']==ARCHITECTURE and resumed['config']==config
        assert resumed['provenance']['dataset_audit_sha256']==provenance['dataset_audit_sha256']
        assert resumed['provenance']['own_parent_checkpoint_sha256']==provenance['own_parent_checkpoint_sha256']
        model.load_state_dict(resumed['state_dict']);ema.load_state_dict(resumed['ema_state_dict'])
        optimizer.load_state_dict(resumed['optimizer'])
        if amp and resumed['scaler']:scaler.load_state_dict(resumed['scaler'])
        start_epoch=resumed['epoch']
        if start_epoch>=config['epochs']:raise ValueError('This run already reached its epoch limit')
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    collate=collate_detection
    dataset=DetectionDataset(root,'train',config,limit=2 if smoke else None)
    if smoke:
        images,boxes,ids,targets,collisions=collate([dataset[0],dataset[1]])
        prepared=[{k:v.to(device) for k,v in t.items()} for t in targets]
        before=fingerprint(model);model.train(not config.get('train_box_quality_only',False))
        frozen_state=None
        if config.get('train_box_quality_only',False):
            trainable={n for n,p in model.named_parameters() if p.requires_grad}
            frozen_state={n:t.detach().clone() for n,t in model.state_dict().items() if n not in trainable}
            with torch.no_grad():presence_before=[o[:,0].clone() for o in model(images.to(device))]
        with torch.autocast(device.type,enabled=amp):outputs=model(images.to(device))
        loss,parts=complete_object_loss(outputs,prepared,collisions,config)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),10);scaler.step(optimizer);scaler.update()
        assert fingerprint(model)!=before
        if frozen_state:
            assert all(torch.equal(model.state_dict()[n],t) for n,t in frozen_state.items())
            with torch.no_grad():presence_after=[o[:,0] for o in model(images.to(device))]
            assert all(torch.equal(a,b) for a,b in zip(presence_before,presence_after))
        atomic_save({'architecture':ARCHITECTURE,'state_dict':model.state_dict(),'provenance':provenance,'config':config,'image_size':512},run/'iteration.pth')
        restored=VisionLocaliza().to(device);restored.load_state_dict(torch.load(run/'iteration.pth',map_location=device,weights_only=True)['state_dict'])
        assert fingerprint(restored)==fingerprint(model)
        result={'passed':True,'device':str(device),'loss':float(loss.detach()),'finite_gradients':True,
                'weights_updated':True,'checkpoint_reload_exact':True,'own_parent_checkpoint':True,'test_used':False}
        if frozen_state:result['features_and_presence_unchanged']=True
        (run/'smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    smokes=list(run.parent.glob('smoke_refine*/smoke.json'))
    assert any(json.loads(p.read_text()).get('passed') and json.loads((p.parent/'provenance.json').read_text())['config']==config for p in smokes)
    val=DetectionDataset(root,'val',config)
    loaders={}
    for name in ['TACO','MJU']:
        subset=deepcopy(val)
        indices=[i for i,r in enumerate(val.rows) if (r.get('source_dataset')=='MJU-Waste')==(name=='MJU')]
        subset.rows=[val.rows[i] for i in indices];subset.cache=[val.cache[i] for i in indices]
        loaders[name]=DataLoader(subset,batch_size=16,collate_fn=collate)
    for name,categories in config.get('validation_groups',{}).items():
        subset=deepcopy(val)
        indices=[i for i,r in enumerate(val.rows) if any(b['category'] in categories for b in r['boxes'])]
        if not indices:raise ValueError('Empty validation group: '+name)
        subset.rows=[val.rows[i] for i in indices];subset.cache=[val.cache[i] for i in indices]
        loaders[name]=DataLoader(subset,batch_size=16,collate_fn=collate)
    baseline=json.loads((run/'baseline.json').read_text()) if resumed else domain_validation(ema,loaders,device,config)
    floor=max(config.get('TACO_minimum_AP95',0),baseline['TACO']['mAP50_95']-config['max_TACO_AP95_drop'])
    if not resumed:(run/'baseline.json').write_text(json.dumps(baseline,indent=2))
    best=resumed['best_score'] if resumed else baseline['balanced_AP95']
    best_epoch=resumed['best_epoch'] if resumed else 0
    def save_best(epoch,metrics):
        atomic_save({'architecture':ARCHITECTURE,'state_dict':ema.state_dict(),'config':config,
            'provenance':provenance,'image_size':512,'epoch':epoch,'validation':metrics},run/'best.pth')
    if not resumed:save_best(0,baseline)
    focus=config.get('category_sampling_weights',{})
    weights=[max([config['MJU_sampling_weight'] if r.get('source_dataset')=='MJU-Waste' else 1.]+
                 [focus.get(b['category'],1.) for b in r['boxes']]) for r in dataset.rows]
    sampler=WeightedRandomSampler(weights,len(dataset),replacement=True)
    loader=DataLoader(dataset,batch_size=config['batch_size'],sampler=sampler,num_workers=config['workers'],
        collate_fn=collate,pin_memory=True,persistent_workers=config['workers']>0)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True;print('Stopping safely after epoch',flush=True)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    history=json.loads((run/'history.json').read_text()) if resumed else []
    steps=start_epoch*len(loader)
    if resumed:
        torch.set_rng_state(resumed['torch_rng'].cpu());random.setstate(resumed['python_rng'])
        torch.cuda.set_rng_state_all([s.cpu() for s in resumed['cuda_rng']])
    print('BASELINE',json.dumps({n:{k:v[k] for k in ['mAP50','mAP50_95']} for n,v in baseline.items() if isinstance(v,dict)}),flush=True)
    for epoch in range(start_epoch+1,config['epochs']+1):
        started=time.monotonic();model.train(not config.get('train_box_quality_only',False));total=0
        lr=config['learning_rate']*(.1+.9*(1+math.cos(math.pi*(epoch-1)/config['epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr
        for images,boxes,ids,targets,collisions in loader:
            prepared=[{k:v.to(device,non_blocking=True) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=amp):outputs=model(images.to(device,non_blocking=True))
            loss,parts=complete_object_loss(outputs,prepared,collisions,config)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; previous last checkpoint preserved')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients; previous last preserved')
            scaler.step(optimizer);scaler.update();steps+=1
            with torch.no_grad():
                decay=min(.995,1-1/(steps+10))
                for a,b in zip(ema.state_dict().values(),model.state_dict().values()):
                    if a.dtype.is_floating_point:a.mul_(decay).add_(b.detach(),alpha=1-decay)
                    else:a.copy_(b)
            total+=float(loss.detach())
        metrics=None
        if epoch==1 or epoch%config['validation_every']==0 or stopped or epoch==config['epochs']:
            metrics=domain_validation(ema,loaders,device,config)
            if (metrics['TACO']['mAP50_95']>=floor and metrics['MJU']['mAP50_95']>=config.get('MJU_minimum_AP95',0)
                    and metrics['balanced_AP95']>best):
                best=metrics['balanced_AP95'];best_epoch=epoch;save_best(epoch,metrics)
        atomic_save({'architecture':ARCHITECTURE,'state_dict':model.state_dict(),'ema_state_dict':ema.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'config':config,'provenance':provenance,
            'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},run/'last.pth')
        row={'epoch':epoch,'loss':total/len(loader),'seconds':time.monotonic()-started,'validation':metrics}
        history.append(row);(run/'history.json').write_text(json.dumps(history,indent=2))
        status={'epoch':epoch,'best_epoch':best_epoch,'balanced_AP95':best,'TACO_floor':floor,'complete':False,'stopped':stopped}
        (run/'status.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(dict(status,loss=row['loss'],seconds=row['seconds'],validation=None if not metrics else
            {n:{k:v[k] for k in ['mAP50','mAP50_95','precision','recall']} for n,v in metrics.items() if isinstance(v,dict)})),flush=True)
        if stopped or epoch-best_epoch>=config['patience']:break
    status['complete']=not stopped;(run/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/detector_D2_refine.json')
    p.add_argument('--root',default='data/detection/refine_D2');p.add_argument('--parent',default='models/runtime/detector_D2.pth')
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume')
    p.add_argument('--device',choices=('auto','cuda','cpu'))
    a=p.parse_args();train(a.config,a.root,a.parent,a.output,a.smoke,a.resume,a.device)
