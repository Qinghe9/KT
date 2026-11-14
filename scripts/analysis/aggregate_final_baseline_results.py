#!/usr/bin/env python3
"""Build compact, paper-ready two-seed baseline summaries.

The four ASSIST/EdNet datasets are read from the two archived WeakHead exports.
Algebra I 2005-2006 is read from the canonical unified-evaluation summary.
Only the ten baseline models are retained; the proposed model is intentionally
excluded because this file is used for the appendix baseline inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


BASELINES = [
    "dkt",
    "dkt+",
    "dkvmn",
    "sakt",
    "saint",
    "akt",
    "simplekt",
    "dtransformer",
    "dimkt",
    "mockt",
]

DISPLAY = {
    "dkt": "DKT",
    "dkt+": "DKT+",
    "dkvmn": "DKVMN",
    "sakt": "SAKT",
    "saint": "SAINT",
    "akt": "AKT",
    "simplekt": "simpleKT",
    "dtransformer": "DTransformer",
    "dimkt": "DIMKT",
    "mockt": "MoCKT",
}

DATASET_DISPLAY = {
    "assist2009": "ASSISTments 2009",
    "assist2012": "ASSISTments 2012",
    "assist2017": "ASSISTments 2017",
    "ednet": "EdNet",
    "algebra": "Algebra I 2005-2006",
}

METRICS = [
    "AUC",
    "ACC",
    "RMSE",
    "Macro-KCAUC",
    "WeakKC-AUC",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Recall@10",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_model(name: str) -> str:
    suffix = "+WeakHead"
    value = name.strip()
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value.lower()


def read_archived(path: Path, store: dict[tuple[str, str], list[dict[str, float]]]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"].strip().lower()
            if dataset not in {"assist2009", "assist2012", "assist2017", "ednet"}:
                continue
            model = normalize_model(row["model"])
            if model not in BASELINES:
                continue
            store[(dataset, model)].append(
                {
                    "AUC": float(row["AUC"]),
                    "ACC": float(row["ACC"]),
                    "RMSE": float(row["RMSE"]),
                    "Macro-KCAUC": float(row["Macro-KCAUC"]),
                    "WeakKC-AUC": float(row["WeakKC_AUC"]),
                    "Recall@1": float(row["Recall@1"]),
                    "Recall@3": float(row["Recall@3"]),
                    "Recall@5": float(row["Recall@5"]),
                    "Recall@10": float(row["Recall@10"]),
                }
            )


def read_algebra(path: Path, store: dict[tuple[str, str], list[dict[str, float]]]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].strip().lower()
            if model not in BASELINES:
                continue
            store[("algebra", model)].append(
                {
                    "AUC": float(row["AUC_mean"]),
                    "ACC": float(row["ACC_mean"]),
                    "RMSE": float(row["RMSE_mean"]),
                    "Macro-KCAUC": float(row["Macro_KCAUC_mean"]),
                    "WeakKC-AUC": float(row["WeakKC_AUC_mean"]),
                    "Recall@1": float(row["Recall@1_mean"]),
                    "Recall@3": float(row["Recall@3_mean"]),
                    "Recall@5": float(row["Recall@5_mean"]),
                    "Recall@10": float(row["Recall@10_mean"]),
                }
            )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-2026", type=Path, required=True)
    parser.add_argument("--seed-3407", type=Path, required=True)
    parser.add_argument("--algebra-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    store: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    read_archived(args.seed_2026, store)
    read_archived(args.seed_3407, store)
    read_algebra(args.algebra_summary, store)

    kt_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    for dataset in ["assist2009", "assist2012", "assist2017", "ednet", "algebra"]:
        for model in BASELINES:
            runs = store.get((dataset, model), [])
            expected = 1 if dataset == "algebra" else 2
            if len(runs) != expected:
                raise RuntimeError(
                    f"Expected {expected} summarized source rows for {dataset}/{model}, got {len(runs)}"
                )
            averaged = {metric: mean([run[metric] for run in runs]) for metric in METRICS}
            common = {
                "dataset": DATASET_DISPLAY[dataset],
                "model": DISPLAY[model],
                "seeds": "2026;3407",
            }
            kt_rows.append(
                common
                | {key: f"{averaged[key]:.4f}" for key in ["AUC", "ACC", "RMSE", "Macro-KCAUC"]}
            )
            weak_rows.append(
                common
                | {
                    key: f"{averaged[key]:.4f}"
                    for key in ["WeakKC-AUC", "Recall@1", "Recall@3", "Recall@5", "Recall@10"]
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kt_path = args.output_dir / "full_10_baselines_kt_mean.csv"
    weak_path = args.output_dir / "full_10_baselines_weakkc_mean.csv"
    manifest_path = args.output_dir / "result_manifest.json"
    write_csv(kt_path, kt_rows, ["dataset", "model", "seeds", "AUC", "ACC", "RMSE", "Macro-KCAUC"])
    write_csv(
        weak_path,
        weak_rows,
        ["dataset", "model", "seeds", "WeakKC-AUC", "Recall@1", "Recall@3", "Recall@5", "Recall@10"],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": "student-level 8:1:1, fold 0, non-windowed test, atomic KC, seeds 2026 and 3407",
                "sources": {
                    str(args.seed_2026): sha256(args.seed_2026),
                    str(args.seed_3407): sha256(args.seed_3407),
                    str(args.algebra_summary): sha256(args.algebra_summary),
                },
                "outputs": {
                    kt_path.name: sha256(kt_path),
                    weak_path.name: sha256(weak_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(kt_path)
    print(weak_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
