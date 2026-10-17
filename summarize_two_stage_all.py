import csv
import json
from pathlib import Path

ROOT=Path('/home/u202310176043/project/demo')
FIELDS=('dataset','seed','training','best_epoch','epochs_trained','AUC','ACC','RMSE',
        'Macro-KCAUC','WeakKC_AUC','Recall@1','Recall@3','Recall@5','Recall@10',
        'test_interactions','weak_pairs')

def path_for(dataset,seed,training):
    if dataset in ('assist2009','ednet'):
        if training=='kt_only': return ROOT/f'runs/ablation_small_seed{seed}'/dataset/'no_weak'
        return ROOT/f'runs/complete_ablation_seed{seed}'/dataset/'two_stage'
    return ROOT/'runs/two_stage_remaining'/f'seed_{seed}'/dataset/training

rows=[]
for dataset in ('assist2009','assist2012','assist2017','ednet'):
    for seed in (3407,2026):
        for training in ('kt_only','two_stage'):
            path=path_for(dataset,seed,training)/'metrics.json'
            if not path.exists(): continue
            m=json.loads(path.read_text()); test=m['test']; weak=m['weak_test']
            rows.append(dict(dataset=dataset,seed=seed,training=training,
                best_epoch=m.get('best_epoch'),epochs_trained=m.get('epochs_trained'),
                AUC=test.get('AUC'),ACC=test.get('ACC'),RMSE=test.get('RMSE'),
                **{'Macro-KCAUC':test.get('Macro_KCAUC')},WeakKC_AUC=weak.get('AUC'),
                **{f'Recall@{k}':weak.get(f'Recall@{k}') for k in (1,3,5,10)},
                test_interactions=test.get('n'),weak_pairs=weak.get('n')))

out=ROOT/'runs/two_stage_all_datasets.csv'
with out.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
print(out)
