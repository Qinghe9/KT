import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "dataset", "variant", "seed", "best_epoch", "epochs_trained",
    "AUC", "ACC", "RMSE", "Macro-KCAUC", "WeakKC_AUC",
    "Recall@1", "Recall@3", "Recall@5", "Recall@10",
    "recall_users", "mean_candidates", "test_interactions", "weak_pairs",
]


def load_row(dataset, variant, path, seed):
    metrics = json.loads((path / "metrics.json").read_text())
    test = metrics["test"]
    weak = metrics["weak_test"]
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "best_epoch": metrics.get("best_epoch"),
        "epochs_trained": metrics.get("epochs_trained"),
        "AUC": test.get("AUC"),
        "ACC": test.get("ACC"),
        "RMSE": test.get("RMSE"),
        "Macro-KCAUC": test.get("Macro_KCAUC"),
        "WeakKC_AUC": weak.get("AUC"),
        "Recall@1": weak.get("Recall@1"),
        "Recall@3": weak.get("Recall@3"),
        "Recall@5": weak.get("Recall@5"),
        "Recall@10": weak.get("Recall@10"),
        "recall_users": weak.get("recall_users"),
        "mean_candidates": weak.get("mean_candidates"),
        "test_interactions": test.get("n"),
        "weak_pairs": weak.get("n"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        rows.append(load_row(dataset, "full", args.formal_root / dataset, args.seed))
        for variant in ("no_spectrum", "top4", "no_weak"):
            rows.append(load_row(dataset, variant, args.ablation_root / dataset / variant, args.seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
