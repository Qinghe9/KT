"""Jointly fine-tune a pretrained pyKT baseline with the shared WeakKC head."""
import argparse,ast,csv,hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pykt.datasets.init_dataset import init_dataset4train
from pykt.datasets.data_loader import KTDataset
from pykt.datasets.dimkt_dataloader import DIMKTDataset
from pykt.datasets.mockt_data_loader import MOCKTDataset
from pykt.datasets.weakkc_dataset import WeakKCDataset
from pykt.models import init_model
from pykt.models.evaluate_model import evaluate,evaluate_question
from pykt.models.train_model import model_forward
from pykt.models.weakkc_adapter import encode_history,encode_dimkt_history,state_dim
from pykt.models.weakkc_head import WeakKCHead


def move(batch,device):
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def recalls(rows,ks=(1,3,5,10)):
    users=defaultdict(list)
    for row in rows: users[row['uid']].append(row)
    vals={k:[] for k in ks}; candidates=[]
    for group in users.values():
        positives=sum(x['y'] for x in group)
        if not positives: continue
        ranked=sorted(group,key=lambda x:-x['p']);candidates.append(len(ranked))
        for k in ks: vals[k].append(sum(x['y'] for x in ranked[:k])/positives)
    return {**{f'Recall@{k}':float(np.mean(v)) if v else None for k,v in vals.items()},
            'recall_users':len(candidates),'mean_candidates':float(np.mean(candidates)) if candidates else None}


def difficulty_arrays(dpath,level,device):
    def load(name):
        frame=pd.read_csv(Path(dpath)/f'{name}_difficult_{level}.csv')
        values=frame.iloc[0].to_numpy(dtype=np.int64)
        return torch.tensor(values,device=device)
    return load('skills'),load('questions')


def question_test_loader(name,data,batch,diff_level):
    if 'test_question_file' not in data:
        raise RuntimeError('question-level test file is required for fair multi-KC evaluation')
    # Upstream pyKT dataset classes concatenate cache suffixes to file_path and
    # therefore require a string rather than pathlib.Path.
    path=str(Path(data['dpath'])/data['test_question_file'])
    if name=='dimkt': ds=DIMKTDataset(data['dpath'],path,data['input_type'],{-1},qtest=True,diff_level=diff_level)
    elif name=='mockt': ds=MOCKTDataset(path,data['input_type'],{-1},qtest=True)
    else: ds=KTDataset(path,data['input_type'],{-1},qtest=True)
    return DataLoader(ds,batch_size=batch,shuffle=False)


def macro_kcauc_from_prediction_file(path):
    groups=defaultdict(lambda:[[],[]])
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            row=ast.literal_eval(line)
            for kc,y,pred in zip(row[2],row[3],row[4]):
                groups[int(kc)][0].append(y);groups[int(kc)][1].append(pred)
    aucs=[roc_auc_score(y,p) for y,p in groups.values() if len(set(y))==2]
    return float(np.mean(aucs)) if aucs else None,len(aucs)


def question_rmse(path):
    ys=[];ps=[];groups=defaultdict(lambda:[[],[]])
    with open(path) as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            true=float(row['late_trues']);ys.append(true);ps.append(float(row['late_mean']))
            concepts=ast.literal_eval(row['concepts']);preds=ast.literal_eval(row['concept_preds'])
            # pyKT serializes one-element fields as scalars ("17", "0.19")
            # and multi-element fields as tuples ("9,7"). Normalize both.
            if not isinstance(concepts,(list,tuple)): concepts=[concepts]
            if not isinstance(preds,(list,tuple)): preds=[preds]
            for kc,pred in zip(concepts,preds):
                groups[int(kc)][0].append(true);groups[int(kc)][1].append(float(pred))
    aucs=[roc_auc_score(y,p) for y,p in groups.values() if len(set(y))==2]
    return (float(np.sqrt(np.mean((np.asarray(ys)-np.asarray(ps))**2))),len(ys),
            float(np.mean(aucs)) if aucs else None,len(aucs))


def weak_forward(model,head,batch,difficulty=None):
    if model.model_name=='dimkt': state=encode_dimkt_history(model,batch,*difficulty)
    else: state=encode_history(model,batch)
    return head(state,batch['target_c'])


@torch.no_grad()
def eval_weak(model,head,loader,device,difficulty=None):
    model.eval();head.eval();rows=[];ys=[];ps=[]
    for batch in loader:
        batch=move(batch,device); score=weak_forward(model,head,batch,difficulty).sigmoid()
        y=batch['label'].cpu().tolist();p=score.cpu().tolist();ys+=y;ps+=p
        rows.extend({'uid':u,'concept':int(c),'y':int(a),'p':float(b),'future_support':int(n)}
                    for u,c,a,b,n in zip(batch['uid'],batch['target_c'].cpu(),y,p,batch['future_support'].cpu()))
    auc=float(roc_auc_score(ys,ps)) if len(set(ys))==2 else None
    return {'AUC':auc,'n':len(ys),'positives':int(sum(ys)),**recalls(rows)},rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--weak-update-every',type=int,default=4);p.add_argument('--lambda-weak',type=float,default=.2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='cuda');p.add_argument('--check-only',action='store_true')
    p.add_argument('--evaluate-only',action='store_true',
                   help='load out/best.pt and only run validation/test export')
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device); ckpt=Path(args.checkpoint_dir);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    saved=json.loads((ckpt/'config.json').read_text());params=saved['params'];name=params['model_name'];emb=params['emb_type']
    data=saved['data_config'];data['dataset_name']=params['dataset_name'];fold=int(params['fold'])
    if fold!=0 or len(data.get('folds',[]))!=9:
        raise RuntimeError(f"formal protocol requires fold=0 and nine train/valid folds; got fold={fold}, folds={data.get('folds')}")
    model_cfg=dict(saved['model_config'])
    for key in ['use_wandb','learning_rate','add_uuid','l2','num_epochs']:
        model_cfg.pop(key,None)
    if name!='dimkt': model_cfg.pop('batch_size',None)
    if name in ('saint','sakt'): model_cfg['seq_len']=saved['train_config']['seq_len']
    model=init_model(name,model_cfg,data,emb,params['dataset_name']).to(device)
    model.load_state_dict(torch.load(ckpt/f'{emb}_model.ckpt',map_location=device,weights_only=True))
    head=WeakKCHead(state_dim(model),data['num_c'],64,.2).to(device)
    all_cfg={params['dataset_name']:data}
    diff_level=model_cfg.get('difficult_levels',100) if name=='dimkt' else None
    kt_train,kt_valid=init_dataset4train(params['dataset_name'],name,all_cfg,fold,args.batch_size,diff_level=diff_level)
    dpath=Path(data['dpath']); folds=set(data['folds'])
    # Reserve one position for SAINT's explicit diagnostic query.
    common=dict(maxlen=data['maxlen']-1 if name=='saint' else data['maxlen'])
    weak_train=WeakKCDataset(dpath/data['train_valid_original_file_quelevel'],folds-{fold},**common)
    weak_valid=WeakKCDataset(dpath/data['train_valid_original_file_quelevel'],{fold},**common)
    weak_test=WeakKCDataset(dpath/data['test_original_file_quelevel'],{-1},**common)
    kwargs=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda')
    wl={s:DataLoader(ds,shuffle=s=='train',**kwargs) for s,ds in [('train',weak_train),('valid',weak_valid),('test',weak_test)]}
    difficulty=difficulty_arrays(dpath,diff_level,device) if name=='dimkt' else None
    if args.check_only:
        sample=move(next(iter(wl['valid'])),device)
        with torch.no_grad(): logits=weak_forward(model,head,sample,difficulty)
        print(json.dumps({'status':'ok','model':name,'state_dim':state_dim(model),
                          'weak_train':len(weak_train),'weak_valid':len(weak_valid),
                          'weak_test':len(weak_test),'checked_batch':len(logits)}))
        return
    best=-1;history=[]
    if not args.evaluate_only:
        optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=args.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3,factor=.5,min_lr=1e-6)
        bad=0
        for epoch in range(1,args.epochs+1):
            model.train();head.train();weak_iter=iter(wl['train']);losses=[]
            for step,kt_batch in enumerate(kt_train):
                optimizer.zero_grad();loss=model_forward(model,kt_batch)
                if len(weak_train) and step%args.weak_update_every==0:
                    try: wb=next(weak_iter)
                    except StopIteration: weak_iter=iter(wl['train']);wb=next(weak_iter)
                    wb=move(wb,device);logits=weak_forward(model,head,wb,difficulty)
                    loss=loss+args.lambda_weak*torch.nn.functional.binary_cross_entropy_with_logits(logits,wb['label'])
                loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.0)
                optimizer.step();losses.append(float(loss.detach()))
            valid_auc,valid_acc=evaluate(model,kt_valid,name);weak_metrics,_=eval_weak(model,head,wl['valid'],device,difficulty)
            scheduler.step(valid_auc);record={'epoch':epoch,'loss':float(np.mean(losses)),'valid_auc':valid_auc,
                'valid_acc':valid_acc,'weak_valid':weak_metrics,'lr':optimizer.param_groups[0]['lr']}
            history.append(record);print(json.dumps(record),flush=True)
            if valid_auc>best+1e-3:
                best=valid_auc;bad=0;torch.save({'model':model.state_dict(),'weak_head':head.state_dict(),'epoch':epoch},out/'best.pt')
            else: bad+=1
            if bad>=args.patience: break
    elif not (out/'best.pt').exists():
        raise FileNotFoundError(f'--evaluate-only requires {out/"best.pt"}')
    state=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(state['model']);head.load_state_dict(state['weak_head'])
    if args.evaluate_only:
        best,_=evaluate(model,kt_valid,name)
        print(json.dumps({'evaluate_only':True,'best_epoch':state['epoch'],'valid_auc':best}),flush=True)
    prediction_path=out/'test_question_predictions.tsv'
    test_loader=question_test_loader(name,data,args.batch_size,diff_level)
    question_aucs,question_accs=evaluate_question(model,test_loader,name,save_path=str(prediction_path))
    test_auc,test_acc=question_aucs['late_mean'],question_accs['late_mean']
    test_rmse,test_n,macro_kcauc,macro_num=question_rmse(prediction_path)
    weak_result,rows=eval_weak(model,head,wl['test'],device,difficulty)
    result={'model':name+'+WeakHead','fold':fold,'seed':args.seed,'best_valid_auc':best,
            'best_epoch':state['epoch'],'test':{'AUC':test_auc,'ACC':test_acc,'RMSE':test_rmse,
                'Macro_KCAUC':macro_kcauc,'macro_kcauc_num_kcs':macro_num,
                'question_interactions':test_n,'aggregation':'late_mean'},
            'weak_test':weak_result,'history':history,
            'protocol':{'history_ratio':.7,'min_history':16,'min_future_per_kc':10,
                        'positive_error_rate':.6,'negative_error_rate':.4,'recall_ks':[1,3,5,10],
                        'future_use':'labels only','candidate_unit':'atomic KC',
                        'train_valid_sha256':file_sha256(dpath/data['train_valid_original_file_quelevel']),
                        'test_sha256':file_sha256(dpath/data['test_original_file_quelevel'])}}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'weak_test_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ['uid','concept','y','p','future_support']);w.writeheader();w.writerows(rows)
    with (out/'results.csv').open('w',newline='') as f:
        fields=['model','fold','seed','best_epoch','best_valid_auc','AUC','ACC','RMSE','Macro-KCAUC','WeakKC_AUC','Recall@1','Recall@3','Recall@5','Recall@10','recall_users','mean_candidates','n']
        row={'model':result['model'],'fold':fold,'seed':args.seed,'best_epoch':state['epoch'],
             'best_valid_auc':best,'AUC':test_auc,'ACC':test_acc,'RMSE':test_rmse,'Macro-KCAUC':macro_kcauc,
             'WeakKC_AUC':weak_result['AUC'],'n':weak_result['n'],
             'recall_users':weak_result['recall_users'],'mean_candidates':weak_result['mean_candidates']}
        row.update({f'Recall@{k}':weak_result[f'Recall@{k}'] for k in (1,3,5,10)})
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)

if __name__=='__main__': main()
