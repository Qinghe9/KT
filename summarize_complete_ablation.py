import argparse
import csv
import json
from pathlib import Path


ROOT = Path('/home/u202310176043/project/demo')
VARIANTS = (
    'full', 'no_spectrum', 'no_kc_conditioning', 'uniform_fusion', 'top4',
    'no_attention', 'no_gru', 'no_cnn', 'no_mamba', 'kt_only',
    'two_stage', 'joint_training',
)
FIELDS = ('dataset','variant','seed','best_epoch','epochs_trained','AUC','ACC','RMSE',
          'Macro-KCAUC','WeakKC_AUC','Recall@1','Recall@3','Recall@5','Recall@10',
          'test_interactions','weak_pairs')


def result_dir(seed, dataset, variant):
    if variant in ('full','joint_training'):
        return ROOT/f'runs/formal_own_model/seed_{seed}'/dataset
    if variant == 'no_spectrum':
        return ROOT/f'runs/ablation_small_seed{seed}'/dataset/'no_spectrum'
    if variant == 'top4':
        return ROOT/f'runs/ablation_small_seed{seed}'/dataset/'top4'
    if variant == 'kt_only':
        return ROOT/f'runs/ablation_small_seed{seed}'/dataset/'no_weak'
    return ROOT/f'runs/complete_ablation_seed{seed}'/dataset/variant


def row(seed, dataset, variant):
    path=result_dir(seed,dataset,variant)/'metrics.json'
    if not path.exists():
        return None
    value=json.loads(path.read_text()); test=value['test']; weak=value['weak_test']
    return dict(dataset=dataset,variant=variant,seed=seed,best_epoch=value.get('best_epoch'),
                epochs_trained=value.get('epochs_trained'),AUC=test.get('AUC'),ACC=test.get('ACC'),
                RMSE=test.get('RMSE'),**{'Macro-KCAUC':test.get('Macro_KCAUC')},
                WeakKC_AUC=weak.get('AUC'),**{f'Recall@{k}':weak.get(f'Recall@{k}') for k in (1,3,5,10)},
                test_interactions=test.get('n'),weak_pairs=weak.get('n'))


def write(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--all-seeds',type=int,nargs='*');a=p.parse_args()
    seeds=a.all_seeds or [a.seed]
    rows=[r for seed in seeds for dataset in ('assist2009','ednet')
          for variant in VARIANTS if (r:=row(seed,dataset,variant))]
    out=(ROOT/'runs'/('complete_ablation_all_seeds.csv' if a.all_seeds else f'complete_ablation_seed{a.seed}/ablation_results.csv'))
    write(rows,out);print(out)


if __name__=='__main__': main()
