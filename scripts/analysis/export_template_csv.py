"""Export completed one-fold runs using the three supplied experiment-table schemas."""
import argparse
import csv
import json
import shlex
from datetime import date
from pathlib import Path


def write(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='append',required=True,metavar='DATASET=RUN_DIR')
    p.add_argument('--output-dir',default='final_csv')
    args=p.parse_args(); out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    today=date.today().isoformat(); summary=[]; table=[]; detail=[]
    for spec in args.run:
        dataset,run=spec.split('=',1); root=Path(run).resolve()
        m=json.loads((root/'metrics.json').read_text());cfg=json.loads((root/'config.json').read_text())
        if 'test' not in m: raise ValueError(f'{run}: test metrics missing')
        test=m['test']; weak=m.get('weak_test',{})
        recall_ks=cfg.get('recall_ks',[cfg.get('recall_k',3)])
        recalls={f'Recall@{k}':weak.get(f'Recall@{k}',weak.get('Recall_at_K') if k==cfg.get('recall_k',3) else None)
                 for k in recall_ks}
        hp={k:cfg.get(k) for k in ['dim','context','topk','batch','lr','epochs','patience',
                                    'scheduler_patience','lr_factor','min_lr','seed','recall_ks',
                                    'weak_history_ratio','weak_min_history','weak_min_future',
                                    'weak_positive_threshold','weak_negative_threshold',
                                    'lambda_weak','lambda_balance']}
        recall_text='; '.join(f'{name}={value if value is not None else "NA"}' for name,value in recalls.items())
        weak_auc=weak.get('AUC')
        weak_auc_text=f'{weak_auc:.6f}' if weak_auc is not None else 'NA'
        core=(f"AUC={test['AUC']:.6f}; ACC={test['ACC']:.6f}; RMSE={test['RMSE']:.6f}; "
              f"Macro-KCAUC={test['Macro_KCAUC']:.6f}; WeakKC-AUC={weak_auc_text}; {recall_text}")
        summary.append({'model':'KC-SRHE-KT','实验时间':today,'模型类型':'知识点条件化频谱异构专家',
          '实验问题':f'{dataset} one-fold; early stopping; LR scheduling','论文':'','代码仓库':str(Path.cwd()),
          '核心指标':core,'原论文指标范围':'','实验使用的超参数':json.dumps(hp,ensure_ascii=False),
          '原论文超参数':'','实验的数据集':json.dumps(m['protocol'],ensure_ascii=False),
          '原论文的数据集':'','论文发现的现象':'Attention+GRU+causal CNN+official Mamba; validation-selected checkpoint'})
        table.append({'model':f'KC-SRHE-KT-{dataset}','实验时间':today,'AUC':test['AUC'],
          'ACC':test['ACC'],'RMSE':test['RMSE'],'Macro-KCAUC':test['Macro_KCAUC'],
          'WeakKC-AUC':weak_auc,**recalls,'KC-MND（平方）%':'未计算','DOA':'未计算','KC-MND（绝对值）%':'未计算'})
        command='python -u train.py '+shlex.join([f'--{k.replace("_","-")}' if v is True else str(v)
                  for k,v in []])
        detail.append({'model':'KC-SRHE-KT','实验时间':today,'fold':cfg.get('fold',0),
          'batch_size':cfg['batch'],'ckpt_path':str(root/'best.pt'),'dataset':dataset,'emb_dim':cfg['dim'],
          'epochs':m.get('epochs_trained'),'lr':cfg['lr'],'seed':cfg['seed'],'test_acc':test['ACC'],
          'test_auc':test['AUC'],'test_recall':recalls.get('Recall@3'),'Command':cfg.get('_command',command),
          'test_rmse':test['RMSE'],'Macro-KCAUC':test['Macro_KCAUC'],'WeakKC-AUC':weak_auc,**recalls,
          'best_epoch':m.get('best_epoch'),'valid_auc':m['valid']['AUC'],'Parent items':''})
    write(out/'习题推荐基线模型实验记录表_pykt基线记录_Grid.csv',
          ['model','实验时间','模型类型','实验问题','论文','代码仓库','核心指标','原论文指标范围','实验使用的超参数',
           '原论文超参数','实验的数据集','原论文的数据集','论文发现的现象'],summary)
    write(out/'习题推荐基线模型实验记录表_Table_Grid.csv',
          ['model','实验时间','AUC','ACC','RMSE','Macro-KCAUC','WeakKC-AUC','Recall@1','Recall@3','Recall@5','Recall@10','KC-MND（平方）%','DOA','KC-MND（绝对值）%'],table)
    write(out/'习题推荐基线模型实验记录表_实验记录_Grid.csv',
          ['model','实验时间','fold','batch_size','ckpt_path','dataset','emb_dim','epochs','lr','seed','test_acc',
           'test_auc','test_recall','Command','test_rmse','Macro-KCAUC','WeakKC-AUC','Recall@1','Recall@3','Recall@5','Recall@10','best_epoch','valid_auc','Parent items'],detail)


if __name__=='__main__':main()
