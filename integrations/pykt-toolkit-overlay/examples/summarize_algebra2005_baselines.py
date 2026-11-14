"""Collect Algebra2005 standard and WeakKC baseline reports."""
import argparse
import csv
from pathlib import Path


ORDER = ["dkt", "dkt+", "dkvmn", "sakt", "saint", "akt", "simplekt",
         "dtransformer", "dimkt", "mockt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--weak-dir",
        default="weakhead",
        help="directory containing per-model WeakHead results (default: weakhead)",
    )
    args = parser.parse_args()
    rows = []
    for model in ORDER:
        path = args.root / args.weak_dir / model / "results.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    destination = args.root / f"weakhead_results_all_seed{args.seed}.csv"
    if rows:
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    standard = args.root / "standard" / "习题推荐基线模型实验记录表_Table_Grid.csv"
    if standard.exists():
        (args.root / "standard_results_all.csv").write_bytes(standard.read_bytes())
    # Produce one model-aligned report while keeping standard KT and the
    # jointly trained WeakHead evaluation explicitly separated.  They use
    # different prediction heads and must not be silently treated as the same
    # metric column.
    macro_path = args.root / "standard_macro_kcauc.csv"
    if rows and macro_path.exists():
        with macro_path.open(newline="") as handle:
            standard_rows = list(csv.DictReader(handle))
        weak_by_model = {
            row["model"].removesuffix("+WeakHead").lower(): row for row in rows
        }
        # DKT+ becomes ``dkt+`` after removing the suffix; the normalization
        # above therefore remains unambiguous despite the visual ``++`` in
        # the exported WeakHead display name.
        combined = []
        for standard_row in standard_rows:
            internal = standard_row["model"].lower()
            weak = weak_by_model.get(internal, {})
            combined.append({
                "model": standard_row["model"],
                "fold": standard_row.get("fold", ""),
                "seed": standard_row.get("seed", args.seed),
                "standard_AUC": standard_row.get("AUC", ""),
                "standard_ACC": standard_row.get("ACC", ""),
                "standard_RMSE": standard_row.get("RMSE", ""),
                "standard_Macro-KCAUC": standard_row.get("Macro-KCAUC", ""),
                "standard_test_interactions": standard_row.get("test_interactions", ""),
                "weakhead_AUC": weak.get("AUC", ""),
                "weakhead_ACC": weak.get("ACC", ""),
                "weakhead_RMSE": weak.get("RMSE", ""),
                "weakhead_Macro-KCAUC": weak.get("Macro-KCAUC", ""),
                "WeakKC_AUC": weak.get("WeakKC_AUC", ""),
                "Recall@1": weak.get("Recall@1", ""),
                "Recall@3": weak.get("Recall@3", ""),
                "Recall@5": weak.get("Recall@5", ""),
                "Recall@10": weak.get("Recall@10", ""),
                "weak_candidates": weak.get("n", ""),
            })
        combined_path = args.root / "experiment_results_all.csv"
        with combined_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
        print(f"combined {len(combined)}/{len(ORDER)} models -> {combined_path}")
    print(f"collected {len(rows)}/{len(ORDER)} WeakKC baselines -> {destination}")


if __name__ == "__main__":
    main()
