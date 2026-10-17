import argparse
import csv
import json
import logging
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from torch.utils.data import DataLoader

from data import prepare, prepare_fold, PrefixDataset
from model import SpectralKT
from weakkc import WeakKCConfig, weak_recall_metrics


def metrics(y,p,c):
    auc=lambda a,b: float(roc_auc_score(a,b)) if len(set(a))==2 else None
    atomic=sorted({kc for concepts in c for kc in concepts})
    per=[]
    for kc in atomic:
        mask=np.array([kc in concepts for concepts in c])
        per.append(auc(np.array(y)[mask],np.array(p)[mask]))
    per=[v for v in per if v is not None]
    return dict(AUC=auc(y,p),ACC=float(accuracy_score(y,np.array(p)>=.5)),
                RMSE=float(mean_squared_error(y,p)**.5),
                Macro_KCAUC=float(np.mean(per)) if per else None,
                valid_concepts=len(per),n=len(y),positives=int(np.sum(y)),
                majority_ACC=float(max(np.mean(y),1-np.mean(y))))


def move(b,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}


@torch.no_grad()
def evaluate(model,loader,device,weak=False,recall_ks=(1,3,5,10)):
    model.eval(); y=[]; p=[]; c=[]; rows=[]; gates=[]
    for b in loader:
        b=move(b,device); logits,_,g=model(b,weak)
        yy=b['y'].cpu().tolist(); pp=logits.sigmoid().cpu().tolist()
        if weak:
            cc=[[x] for x in b['target_c'].cpu().tolist()]
        else:
            cc=[[x for x in row if x] for row in b['target_cs'].cpu().tolist()]
        y+=yy; p+=pp; c+=cc; gates.append(g.sum(0).cpu().numpy())
        rows.extend(dict(uid=u,concepts=k,y=a,p=s) for u,k,a,s in zip(b['uid'],cc,yy,pp))
    if not y: return {"n":0,"AUC":None,**{f"Recall@{k}":None for k in recall_ks}},rows
    result=metrics(y,p,c); result['expert_weights']=(np.sum(gates,axis=0)/len(y)).tolist()
    if weak:
        weak_rows=[dict(uid=row['uid'],concept=row['concepts'][0],y=row['y'],p=row['p']) for row in rows]
        result.update(weak_recall_metrics(weak_rows,recall_ks))
    return result,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='/home/u202310176043/project/pykt-toolkit/data/junyi2015/train_valid_quelevel.csv')
    parser.add_argument('--test-source',default=None)
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--fold-data',action='store_true')
    parser.add_argument('--data',default='data/junyi_small')
    parser.add_argument('--out',default='runs/prototype')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',
                        help='build the atomic-KC dataset and exit without constructing a model')
    parser.add_argument('--users',type=int,default=1600)
    parser.add_argument('--cap',type=int,default=200)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--min-delta',type=float,default=1e-3)
    parser.add_argument('--scheduler-patience',type=int,default=3)
    parser.add_argument('--lr-factor',type=float,default=.5)
    parser.add_argument('--min-lr',type=float,default=1e-6)
    parser.add_argument('--recall-ks',type=int,nargs='+',default=[1,3,5,10])
    parser.add_argument('--weak-history-ratio',type=float,default=.7)
    parser.add_argument('--weak-min-history',type=int,default=16)
    parser.add_argument('--weak-min-future',type=int,default=10)
    parser.add_argument('--weak-positive-threshold',type=float,default=.6)
    parser.add_argument('--weak-negative-threshold',type=float,default=.4)
    parser.add_argument('--lambda-weak',type=float,default=.2)
    parser.add_argument('--lambda-balance',type=float,default=.01)
    parser.add_argument('--weak-update-every',type=int,default=4)
    parser.add_argument('--context',type=int,default=32)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--weak-workers',type=int,default=None,
                        help='workers for sparse diagnostic loader; defaults to --workers')
    parser.add_argument('--gpu-memory-fraction',type=float,default=.9)
    parser.add_argument('--dim',type=int,default=32)
    parser.add_argument('--topk',type=int,default=2,choices=[1,2,3,4])
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--no-history-skip',action='store_true')
    parser.add_argument('--no-spectrum',action='store_true')
    parser.add_argument('--no-kc-conditioning',action='store_true')
    parser.add_argument('--uniform-fusion',action='store_true')
    parser.add_argument('--disable-expert',action='append',default=[],
                        choices=['attention','gru','cnn','mamba_official'])
    parser.add_argument('--training-mode',choices=['joint','kt_only','weak_head'],default='joint')
    parser.add_argument('--init-from',default=None,
                        help='initialize model weights; required for frozen weak_head stage')
    parser.add_argument('--item-prior',action='store_true')
    parser.add_argument('--test',action='store_true',help='evaluate held-out test only after validation selection')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--resume-from',default=None,
                        help='resume model weights from a checkpoint (legacy state_dict or full checkpoint)')
    parser.add_argument('--resume-epoch',type=int,default=0,
                        help='completed epoch represented by a legacy --resume-from checkpoint')
    parser.add_argument('--amp',action='store_true',help='use CUDA automatic mixed precision')
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.evaluate_only:
        saved=json.loads((out/'config.json').read_text())
        for name in ['data','context','dim','lr','batch','workers','gpu_memory_fraction','seed','device','fold','epochs',
                     'no_history_skip','no_spectrum','topk','recall_ks','weak_history_ratio',
                     'weak_min_history','weak_min_future','weak_positive_threshold',
                     'weak_negative_threshold','lambda_weak','lambda_balance','weak_update_every',
                     'no_kc_conditioning','uniform_fusion','disable_expert','training_mode']:
            if name in saved:
                setattr(args,name,saved[name])
        # Backward compatibility with runs created before multi-K reporting.
        if 'recall_ks' not in saved:
            args.recall_ks=[saved.get('recall_k',3)]
        args.item_prior=saved.get('item_prior',False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'train.log'),logging.StreamHandler()])
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,device=args.device)
    if args.prepare:
        if args.fold_data:
            if not args.test_source: parser.error('--fold-data requires --test-source')
            prepare_fold(args.source,args.test_source,args.data,args.fold,args.users,args.seed,args.cap)
        else: prepare(args.source,args.data,args.users,args.seed,args.cap)
        if args.prepare_only:
            print(json.dumps(json.loads((Path(args.data)/'manifest.json').read_text()),indent=2))
            return
    elif args.prepare_only:
        parser.error('--prepare-only requires --prepare')
    root=Path(args.data); students=json.loads((root/'students.json').read_text());meta=json.loads((root/'manifest.json').read_text())
    weak_config=WeakKCConfig(args.weak_history_ratio,args.weak_min_history,args.weak_min_future,
                             args.weak_positive_threshold,args.weak_negative_threshold,
                             tuple(sorted(set(args.recall_ks)))).validate()
    sets={s:PrefixDataset(students,s,args.context) for s in ['train','valid','test']}
    weaksets={s:PrefixDataset(students,s,args.context,weak=True,weak_config=weak_config) for s in sets}
    loader_args=dict(batch_size=args.batch,num_workers=args.workers,
                     pin_memory=args.device.startswith('cuda'),persistent_workers=args.workers>0)
    weak_workers=args.workers if args.weak_workers is None else args.weak_workers
    weak_loader_args=dict(batch_size=args.batch,num_workers=weak_workers,
                          pin_memory=args.device.startswith('cuda'),persistent_workers=weak_workers>0)
    loaders={s:DataLoader(ds,shuffle=s=='train',**loader_args) for s,ds in sets.items()}
    weakloaders={s:DataLoader(ds,shuffle=s=='train' and len(ds)>0,**weak_loader_args) for s,ds in weaksets.items()}
    logging.info('KT examples %s; diagnostic pairs %s', {s:len(d) for s,d in sets.items()},{s:len(d) for s,d in weaksets.items()})
    model=SpectralKT(meta['num_q'],meta['num_c'],args.context,args.dim,topk=args.topk,
                     history_skip=not args.no_history_skip,spectral=not args.no_spectrum,
                     item_prior=args.item_prior,kc_conditioning=not args.no_kc_conditioning,
                     uniform_fusion=args.uniform_fusion,
                     disabled_experts=tuple(args.disable_expert)).to(args.device)
    if args.init_from:
        initial=torch.load(args.init_from,map_location=args.device,weights_only=False)
        model.load_state_dict(initial['model'] if isinstance(initial,dict) and 'model' in initial else initial)
        logging.info('initialized model weights from %s',args.init_from)
    if args.training_mode=='weak_head':
        if not args.init_from and not args.evaluate_only:
            parser.error('--training-mode weak_head requires --init-from')
        for parameter in model.parameters(): parameter.requires_grad=False
        for parameter in model.weak.parameters(): parameter.requires_grad=True
        logging.info('weak_head stage: encoder frozen; trainable parameters=%s',
                     sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.item_prior:
        count=np.zeros(meta['num_q']);correct=np.zeros(meta['num_q'])
        for s in students:
            if s['split']=='train':
                np.add.at(count,s['q'],1);np.add.at(correct,s['q'],s['r'])
        rate=correct.sum()/count.sum()
        prior=np.clip((correct+30*rate)/(count+30),.01,.99)
        with torch.no_grad():
            model.item_bias.weight.copy_(torch.tensor(np.log(prior/(1-prior)),device=args.device,dtype=torch.float32)[:,None])
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='max',factor=args.lr_factor,patience=args.scheduler_patience,min_lr=args.min_lr)
    amp_enabled=args.amp and args.device.startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    if not args.evaluate_only:
        saved_config={**vars(args),'_command':shlex.join([sys.executable,*sys.argv])}
        (out/'config.json').write_text(json.dumps(saved_config,indent=2))
    best=-1; best_epoch=0; bad_epochs=0; history=[]; start_epoch=1
    if args.resume_from and not args.evaluate_only:
        checkpoint=torch.load(args.resume_from,map_location=args.device,weights_only=False)
        if isinstance(checkpoint,dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if checkpoint.get('scaler') and amp_enabled: scaler.load_state_dict(checkpoint['scaler'])
            history=checkpoint.get('history',[]);best=checkpoint.get('best',-1)
            best_epoch=checkpoint.get('best_epoch',0);bad_epochs=checkpoint.get('bad_epochs',0)
            start_epoch=checkpoint['epoch']+1
        else:
            model.load_state_dict(checkpoint)
            if (out/'history.json').exists():
                old=json.loads((out/'history.json').read_text())
                history=[r for r in old if r.get('epoch',0)<=args.resume_epoch]
            if history:
                ranked=[(r['validation']['AUC'],r['epoch']) for r in history
                        if r.get('validation',{}).get('AUC') is not None]
                best,best_epoch=max(ranked) if ranked else (-1,0)
                bad_epochs=max(0,args.resume_epoch-best_epoch)
                for r in history: scheduler.step(r['validation'].get('AUC',-1))
            start_epoch=args.resume_epoch+1
        logging.info('resume checkpoint=%s start_epoch=%s best_epoch=%s best_auc=%.8f amp=%s',
                     args.resume_from,start_epoch,best_epoch,best,amp_enabled)
    if not args.evaluate_only:
        for epoch in range(start_epoch,args.epochs+1):
            model.train(); losses=[]; weakiter=iter(weakloaders['train'])
            epoch_loader=weakloaders['train'] if args.training_mode=='weak_head' else loaders['train']
            for step,b in enumerate(epoch_loader):
                b=move(b,args.device); optimizer.zero_grad()
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp_enabled):
                    if args.training_mode=='weak_head':
                        logits,balance,_=model(b,weak=True)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['y'])
                    else:
                        logits,balance,_=model(b)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['y'])+args.lambda_balance*balance
                    # A zero WeakKC coefficient is the true "no joint diagnostic
                    # training" ablation: do not run the diagnostic branch and do
                    # not let its router-balance term update the shared encoder.
                    if args.training_mode=='joint' and args.lambda_weak > 0 and len(weaksets['train']) and step%args.weak_update_every==0:
                        try: wb=next(weakiter)
                        except StopIteration: weakiter=iter(weakloaders['train']);wb=next(weakiter)
                        wb=move(wb,args.device); wlogit,wbal,_=model(wb,weak=True)
                        loss=loss+args.lambda_weak*torch.nn.functional.binary_cross_entropy_with_logits(wlogit,wb['y'])+args.lambda_balance*wbal
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update()
                losses.append(loss.item())
                if step%100==0: logging.info('epoch=%s step=%s/%s loss=%.5f',epoch,step,len(epoch_loader),np.mean(losses))
            val,_=evaluate(model,loaders['valid'],args.device,recall_ks=weak_config.recall_ks)
            diagnostic,_=evaluate(model,weakloaders['valid'],args.device,True,weak_config.recall_ks)
            selection=diagnostic if args.training_mode=='weak_head' else val
            current=selection['AUC'] if selection['AUC'] is not None else -1
            scheduler.step(current); lr=optimizer.param_groups[0]['lr']
            record=dict(epoch=epoch,loss=float(np.mean(losses)),learning_rate=lr,
                        validation=val,weak_validation=diagnostic)
            history.append(record);logging.info('%s',json.dumps(record))
            if current>best+args.min_delta:
                best=current;best_epoch=epoch;bad_epochs=0
                torch.save(model.state_dict(),out/'best.pt')
                logging.info('new_best epoch=%s selection=%s valid_auc=%.8f',epoch,args.training_mode,current)
            else:
                bad_epochs+=1
            (out/'history.json').write_text(json.dumps(history,indent=2))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                            best=best,best_epoch=best_epoch,bad_epochs=bad_epochs,history=history),out/'last.pt')
            if bad_epochs>=args.patience:
                logging.info('early_stop epoch=%s best_epoch=%s best_auc=%.8f',epoch,best_epoch,best)
                break
    model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True))
    result=dict(protocol={**meta,'weakkc':weak_config.to_dict()},context=args.context,max_epochs=args.epochs,
                epochs_trained=len(history) if history else None,best_epoch=best_epoch or None,
                test_evaluated=args.test)
    for split in ['valid']+(['test'] if args.test else []):
        result[split],rows=evaluate(model,loaders[split],args.device,recall_ks=weak_config.recall_ks)
        result['weak_'+split],weakrows=evaluate(model,weakloaders[split],args.device,True,weak_config.recall_ks)
        for name,records in [(split,rows),('weak_'+split,weakrows)]:
            with (out/f'{name}_predictions.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['uid','concepts','y','p']);writer.writeheader();writer.writerows(records)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'results.csv').open('w') as f:
        recall_fields=[f'Recall@{k}' for k in weak_config.recall_ks]
        writer=csv.DictWriter(f,fieldnames=['split','AUC','ACC','RMSE','Macro_KCAUC',*recall_fields,'n']);writer.writeheader()
        for split in ['valid']+(['test'] if args.test else []):
            row={'split':split,**{k:result[split].get(k) for k in ['AUC','ACC','RMSE','Macro_KCAUC','n']}}
            row.update({name:result['weak_'+split].get(name) for name in recall_fields})
            writer.writerow(row)
    logging.info('RESULT %s',json.dumps(result))


if __name__=='__main__':main()
