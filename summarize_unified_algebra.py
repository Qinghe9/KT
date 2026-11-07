#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'demo/reproduction/canonical_weakkc_algebra2005'
FIELDS=['dataset','model','seed','AUC','ACC','RMSE','Macro_KCAUC','WeakKC_AUC',
        'Recall@1','Recall@3','Recall@5','Recall@10','KT_n','WeakKC_n',
        'KT_protocol_hash','WeakKC_protocol_hash']


def collect():
    weak_hash=json.loads((ROOT/'demo/data/canonical_weakkc_v1/algebra2005/manifest.json').read_text())['protocol_hash']
    kt_hash=json.loads((ROOT/'demo/data/canonical_kt_v1/algebra2005/manifest.json').read_text())['protocol_hash']
    rows=[]
    for seed in (2026,3407):
        base=ROOT/f'pykt-toolkit/reproduction/formal_weak_baselines_seed{seed}/algebra2005/weakhead'
        for path in sorted(base.glob('*/metrics.json')):
            value=json.loads(path.read_text()); test=value['test']; weak=value['weak_test']
            rows.append({'dataset':'Algebra I 2005-2006','model':path.parent.name,'seed':seed,
                         'AUC':test['AUC'],'ACC':test['ACC'],'RMSE':test['RMSE'],
                         'Macro_KCAUC':test['Macro_KCAUC'],'WeakKC_AUC':weak['AUC'],
                         **{f'Recall@{k}':weak[f'Recall@{k}'] for k in (1,3,5,10)},
                         'KT_n':test['question_interactions'],'WeakKC_n':weak['n'],
                         'KT_protocol_hash':kt_hash,'WeakKC_protocol_hash':weak_hash})
        own=ROOT/f'demo/runs/formal_own_model_algebra2005_canonical/seed_{seed}/algebra2005'
        kt=json.loads((own/'canonical_kt_metrics.json').read_text())
        weak=json.loads((own/'metrics.json').read_text())['weak_test']
        rows.append({'dataset':'Algebra I 2005-2006','model':'SpectralMoE-Mamba','seed':seed,
                     'AUC':kt['AUC'],'ACC':kt['ACC'],'RMSE':kt['RMSE'],
                     'Macro_KCAUC':kt['Macro_KCAUC'],'WeakKC_AUC':weak['AUC'],
                     **{f'Recall@{k}':weak[f'Recall@{k}'] for k in (1,3,5,10)},
                     'KT_n':kt['n'],'WeakKC_n':weak['n'],
                     'KT_protocol_hash':kt_hash,'WeakKC_protocol_hash':weak_hash})
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=collect()
    with (OUT/'unified_results_by_seed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    summary=[]
    for model in sorted({row['model'] for row in rows}):
        selected=[row for row in rows if row['model']==model]
        item={'dataset':'Algebra I 2005-2006','model':model,'seeds':'2026;3407'}
        for field in ['AUC','ACC','RMSE','Macro_KCAUC','WeakKC_AUC','Recall@1','Recall@3','Recall@5','Recall@10']:
            values=[float(row[field]) for row in selected]
            item[field+'_mean']=statistics.mean(values)
            item[field+'_std']=statistics.pstdev(values)
        item['KT_n']=selected[0]['KT_n'];item['WeakKC_n']=selected[0]['WeakKC_n']
        summary.append(item)
    summary_fields=list(summary[0])
    with (OUT/'unified_results_mean_std.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=summary_fields);writer.writeheader();writer.writerows(summary)
    print(f'wrote {len(rows)} seed rows and {len(summary)} summary rows to {OUT}')


if __name__=='__main__':main()
