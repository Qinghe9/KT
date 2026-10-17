import argparse,csv,json
from pathlib import Path

FIELDS=('dataset','stage','seed','best_epoch','epochs_trained','AUC','ACC','RMSE',
        'Macro-KCAUC','WeakKC_AUC','Recall@1','Recall@3','Recall@5','Recall@10',
        'test_interactions','weak_pairs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for seed in (3407,2026):
        for dataset in ('assist2012','assist2017'):
            for stage in ('kt_only','two_stage'):
                m=json.loads((a.root/f'seed_{seed}'/dataset/stage/'metrics.json').read_text())
                test,weak=m['test'],m['weak_test']
                row=dict(dataset=dataset,stage=stage,seed=seed,best_epoch=m.get('best_epoch'),
                         epochs_trained=m.get('epochs_trained'),AUC=test.get('AUC'),ACC=test.get('ACC'),
                         RMSE=test.get('RMSE'),**{'Macro-KCAUC':test.get('Macro_KCAUC')},
                         WeakKC_AUC=weak.get('AUC'),test_interactions=test.get('n'),weak_pairs=weak.get('n'))
                row.update({f'Recall@{k}':weak.get(f'Recall@{k}') for k in (1,3,5,10)})
                rows.append(row)
    out=a.root/'two_stage_results.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print(out)

if __name__=='__main__':main()
