import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (
    'full', 'no_spectrum', 'no_kc_conditioning', 'uniform_fusion', 'top4',
    'no_attention', 'no_gru', 'no_cnn', 'no_mamba', 'kt_only',
    'two_stage', 'joint_training',
)
FIELDS = (
    'dataset', 'variant', 'seed', 'best_epoch', 'epochs_trained',
    'AUC', 'ACC', 'RMSE', 'Macro-KCAUC', 'WeakKC_AUC',
    'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10',
    'test_interactions', 'weak_pairs',
)


def result_dir(seed, variant):
    if variant in ('full', 'joint_training'):
        return ROOT / 'runs/formal_own_model_algebra2005' / f'seed_{seed}' / 'algebra2005'
    return ROOT / 'runs/algebra2005_ablation' / f'seed_{seed}' / variant


def load_row(seed, variant):
    path = result_dir(seed, variant) / 'metrics.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    test, weak = value['test'], value['weak_test']
    return {
        'dataset': 'algebra2005', 'variant': variant, 'seed': seed,
        'best_epoch': value.get('best_epoch'),
        'epochs_trained': value.get('epochs_trained'),
        'AUC': test.get('AUC'), 'ACC': test.get('ACC'), 'RMSE': test.get('RMSE'),
        'Macro-KCAUC': test.get('Macro_KCAUC'), 'WeakKC_AUC': weak.get('AUC'),
        **{f'Recall@{k}': weak.get(f'Recall@{k}') for k in (1, 3, 5, 10)},
        'test_interactions': test.get('n'), 'weak_pairs': weak.get('n'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-seeds', type=int, nargs='+', default=[3407, 2026])
    args = parser.parse_args()
    rows = [row for seed in args.all_seeds for variant in VARIANTS
            if (row := load_row(seed, variant)) is not None]
    out = ROOT / 'runs/algebra2005_ablation/ablation_results_all.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f'collected {len(rows)}/{len(args.all_seeds) * len(VARIANTS)} rows -> {out}')


if __name__ == '__main__':
    main()
