"""Freeze validation-based choice, then evaluate that checkpoint on held-out test."""
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def paired_interval(first,second,repeats=200):
    with open(first) as f: a=list(csv.DictReader(f))
    with open(second) as f: b=list(csv.DictReader(f))
    assert [(r['uid'],r['concept'],r['y']) for r in a]==[(r['uid'],r['concept'],r['y']) for r in b]
    groups={}
    for i,row in enumerate(a):groups.setdefault(row['uid'],[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(3407)
    y=np.array([float(r['y']) for r in a]);pa=np.array([float(r['p']) for r in a]);pb=np.array([float(r['p']) for r in b])
    differences=[]
    for _ in range(repeats):
        idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[idx]))==2:
            differences.append(roc_auc_score(y[idx],pa[idx])-roc_auc_score(y[idx],pb[idx]))
    return dict(delta=float(roc_auc_score(y,pa)-roc_auc_score(y,pb)),
                student_bootstrap_95CI=np.quantile(differences,[.025,.975]).tolist(),repeats=repeats)


def main():
    runs=[Path(p) for p in sys.argv[1:]]
    records=[]
    for root in runs:
        m=json.loads((root/'metrics.json').read_text());cfg=json.loads((root/'config.json').read_text())
        records.append(dict(run=str(root),AUC=m['valid']['AUC'],ACC=m['valid']['ACC'],
                            spectral=not cfg['no_spectrum'],epochs=2))
    with open('runs/validation_comparison.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    best=max((r for r in records if r['spectral']),key=lambda r:r['AUC'])
    Path('runs/selection.json').write_text(json.dumps(dict(selected=best,criterion='validation KT AUC only; complete spectral models'),indent=2))
    subprocess.run([sys.executable,'train.py','--out',best['run'],'--evaluate-only','--test','--epochs','2'],check=True)
    m=json.loads((Path(best['run'])/'metrics.json').read_text())
    paired=paired_interval('runs/full_v1/valid_predictions.csv','runs/no_spectrum/valid_predictions.csv')
    Path('runs/spectral_ablation.json').write_text(json.dumps(paired,indent=2))
    report=['# 两轮小数据集实验结果','',f"固定1600名Junyi用户；最终模型 `{best['run']}` 按验证集作答AUC选择。",
            '', '| 实验 | 验证 AUC | 验证 ACC | 轮数 |','|---|---:|---:|---:|']
    report.extend(f"| {r['run']} | {r['AUC']:.6f} | {r['ACC']:.6f} | 2 |" for r in records)
    report.extend(['',f"最终测试：AUC={m['test']['AUC']:.6f}，ACC={m['test']['ACC']:.6f}，RMSE={m['test']['RMSE']:.6f}，Macro-KCAUC={m['test']['Macro_KCAUC']:.6f}。",
                   '',f"作答AUC >= 0.9：{'达到' if m['test']['AUC']>=.9 else '未达到'}。本次不根据测试结果继续调参。",
                   '', '诊断测试完整指标：','```json',json.dumps(m['weak_test'],ensure_ascii=False,indent=2),'```',
                   '', '专家顺序：Attention、GRU、MLP、Mamba-inspired SSM。',
                   f"同维度频谱消融验证AUC差值与学生级配对bootstrap：{json.dumps(paired)}",
                   '诊断有效样本少、候选少时Recall@3退化，因此不能把高诊断AUC/Recall当作下一题AUC或稳定诊断能力。',
                   '频谱贡献须看同维度消融；更大维度或训练题目先验的收益不能归因于频谱。',
                   '实现修正、公式与限制见 README.md；输出不是与旧EdNet多知识点展开评估直接可比的成绩。'])
    Path('RESULTS.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
