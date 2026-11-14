"""Re-evaluate saved standard checkpoints and add interaction-level Macro-KCAUC.

This script never trains a model.  It restores each checkpoint listed in the
standard experiment record, runs the same non-overlapping ``test_sequences``
evaluation used by the original report, exports one flat prediction CSV per
model, and computes a macro average of per-atomic-KC ROC-AUC values.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wandb_predict import main as predict_main  # noqa: E402


DISPLAY_TO_INTERNAL = {
    "DKT": "dkt",
    "DKT+": "dkt+",
    "DKVMN": "dkvmn",
    "SAKT": "sakt",
    "SAINT": "saint",
    "AKT": "akt",
    "simpleKT": "simplekt",
    "DTransformer": "dtransformer",
    "DIMKT": "dimkt",
    "MoCKT": "mockt",
}


def read_run_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten_predictions(raw_path: Path, flat_path: Path):
    """Flatten pyKT's sequence-wise prediction text into atomic-KC rows."""
    groups = defaultdict(lambda: [[], []])
    labels, scores = [], []
    interaction_id = 0
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open(encoding="utf-8") as source, flat_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["interaction_id", "sequence_id", "position", "atomic_kc", "label", "prediction"],
        )
        writer.writeheader()
        sequence_id = 0
        for line in source:
            if not line.strip():
                continue
            # pyKT writes a bare ``nan`` for a sequence-level AUC when that
            # individual sequence contains only one class.  That trailing
            # diagnostic is not used here; normalize it so literal_eval can
            # still safely parse the aligned KC/label/prediction arrays.
            row = ast.literal_eval(line.replace("nan", "None"))
            # save_cur_predict_result stores target concept ids, labels and
            # probabilities at indices 2, 3 and 4 respectively.
            concepts, trues, predictions = row[2], row[3], row[4]
            if not (len(concepts) == len(trues) == len(predictions)):
                raise RuntimeError(f"unaligned prediction row in {raw_path}: sequence={sequence_id}")
            for position, (kc, label, prediction) in enumerate(zip(concepts, trues, predictions)):
                kc, label, prediction = int(kc), int(label), float(prediction)
                writer.writerow(
                    {
                        "interaction_id": interaction_id,
                        "sequence_id": sequence_id,
                        "position": position,
                        "atomic_kc": kc,
                        "label": label,
                        "prediction": prediction,
                    }
                )
                labels.append(label)
                scores.append(prediction)
                groups[kc][0].append(label)
                groups[kc][1].append(prediction)
                interaction_id += 1
            sequence_id += 1

    if not labels:
        raise RuntimeError(f"no predictions parsed from {raw_path}")
    per_kc = []
    for kc, (kc_labels, kc_scores) in sorted(groups.items()):
        if len(set(kc_labels)) < 2:
            continue
        per_kc.append((kc, len(kc_labels), float(roc_auc_score(kc_labels, kc_scores))))
    if not per_kc:
        raise RuntimeError("no atomic KC contains both positive and negative labels")

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "AUC": float(roc_auc_score(labels_array, scores_array)),
        "ACC": float(accuracy_score(labels_array, scores_array >= 0.5)),
        "RMSE": float(np.sqrt(np.mean((labels_array - scores_array) ** 2))),
        "Macro-KCAUC": float(np.mean([item[2] for item in per_kc])),
        "macro_kcauc_num_kcs": len(per_kc),
        "test_interactions": len(labels),
        "per_kc": per_kc,
    }


def assert_same_metric(model: str, name: str, actual: float, expected: str, tolerance: float):
    if expected in (None, ""):
        return
    expected_value = float(expected)
    if abs(actual - expected_value) > tolerance:
        raise RuntimeError(
            f"{model} {name} mismatch: re-evaluated={actual:.12f}, "
            f"original={expected_value:.12f}, tolerance={tolerance}"
        )


def write_reports(root: Path, completed: list[dict]):
    result_path = root / "standard_macro_kcauc.csv"
    pd.DataFrame(completed).to_csv(result_path, index=False)

    standard_path = root / "standard_results_all.csv"
    standard = pd.read_csv(standard_path, encoding="utf-8-sig")
    macro_map = {row["model"]: row["Macro-KCAUC"] for row in completed}
    count_map = {row["model"]: row["macro_kcauc_num_kcs"] for row in completed}
    standard["Macro-KCAUC"] = standard["model"].map(macro_map)
    standard["Macro-KCAUC有效KC数"] = standard["model"].map(count_map)
    enriched_path = root / "standard_results_all_with_macro_kcauc.csv"
    standard.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    return result_path, enriched_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reproduction" / "algebra2005_baselines_seed3407",
    )
    parser.add_argument("--models", nargs="*", choices=sorted(DISPLAY_TO_INTERNAL))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-5,
        help="metric reproducibility tolerance (covers one threshold-edge prediction)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    run_file = root / "standard" / "习题推荐基线模型实验记录表_实验记录_Grid.csv"
    records = read_run_records(run_file)
    selected = set(args.models or DISPLAY_TO_INTERNAL)
    output_dir = root / "standard_macro_kcauc"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    # Make subset retries resumable: retain already audited model reports and
    # replace only models explicitly evaluated in the current invocation.
    for metrics_path in sorted(output_dir.glob("*_metrics.json")):
        saved_row = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved_row.get("model") not in selected:
            completed.append(saved_row)

    for record in records:
        display = record["model"]
        if display not in selected:
            continue
        internal = DISPLAY_TO_INTERNAL[display]
        checkpoint = Path(record["ckpt_path"]).resolve()
        checkpoint_dir = checkpoint.parent
        if not checkpoint.is_file() or not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"missing standard checkpoint for {display}: {checkpoint}")

        print(f"[EVAL] {display}: {checkpoint_dir}", flush=True)
        result = predict_main(
            {
                "save_dir": str(checkpoint_dir),
                "bz": int(record["batch_size"]),
                "fusion_type": "early_fusion,late_fusion",
                "use_wandb": 0,
                "skip_question": 1,
                "skip_window": 1,
                "compute_rmse": 1,
                "save_predictions": 1,
            }
        )
        raw_path = Path(result["test_prediction_path"])
        flat_path = output_dir / f"{internal}_test_predictions.csv"
        metrics = flatten_predictions(raw_path, flat_path)
        assert_same_metric(display, "AUC", metrics["AUC"], record["test_auc"], args.tolerance)
        assert_same_metric(display, "ACC", metrics["ACC"], record["test_acc"], args.tolerance)
        assert_same_metric(display, "RMSE", metrics["RMSE"], record["test_rmse"], args.tolerance)

        per_kc_path = output_dir / f"{internal}_per_kc_auc.csv"
        pd.DataFrame(metrics.pop("per_kc"), columns=["atomic_kc", "n", "AUC"]).to_csv(
            per_kc_path, index=False
        )
        row = {
            "model": display,
            "fold": int(record["fold"]),
            "seed": int(record["seed"]),
            **metrics,
            "checkpoint": str(checkpoint),
            "prediction_file": str(flat_path),
            "per_kc_file": str(per_kc_path),
        }
        completed = [item for item in completed if item["model"] != display]
        completed.append(row)
        (output_dir / f"{internal}_metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_reports(root, completed)
        print(
            f"[DONE] {display}: AUC={metrics['AUC']:.6f}, "
            f"Macro-KCAUC={metrics['Macro-KCAUC']:.6f}, "
            f"valid_KCs={metrics['macro_kcauc_num_kcs']}",
            flush=True,
        )
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_path, enriched_path = write_reports(root, completed)
    print(f"Macro-KCAUC report: {result_path}")
    print(f"Enriched standard report: {enriched_path}")


if __name__ == "__main__":
    main()
