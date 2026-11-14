"""Evaluate saved standard pyKT checkpoints on the non-windowed question-level test set.

This script does not train or fine-tune a model.  It exists to make the standard
KT baselines use the same question-level aggregation and atomic-KC accounting as
the proposed model and the shared WeakKC evaluation protocol.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pykt.models import init_model
from pykt.models.evaluate_model import evaluate_question
from train_weakkc_baseline import question_rmse, question_test_loader


DISPLAY_TO_INTERNAL = {
    "DKT": "dkt", "DKT+": "dkt+", "DKVMN": "dkvmn", "SAKT": "sakt",
    "SAINT": "saint", "AKT": "akt", "simpleKT": "simplekt",
    "DTransformer": "dtransformer", "DIMKT": "dimkt", "MoCKT": "mockt",
}


def read_records(path):
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            checkpoint = row.get("ckpt_path") or row.get("checkpoint")
            if row.get("model") in DISPLAY_TO_INTERNAL and checkpoint:
                rows.append({**row, "checkpoint": checkpoint})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", nargs="*", choices=sorted(DISPLAY_TO_INTERNAL))
    args = parser.parse_args()

    root = args.root.resolve()
    run_file = root / "standard" / "习题推荐基线模型实验记录表_实验记录_Grid.csv"
    selected = set(args.models or DISPLAY_TO_INTERNAL)
    output = root / "standard_questionlevel"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for record in read_records(run_file):
        display = record["model"]
        if display not in selected:
            continue
        checkpoint = Path(record["checkpoint"]).resolve()
        checkpoint_dir = checkpoint.parent
        saved = json.loads((checkpoint_dir / "config.json").read_text())
        params = saved["params"]
        name = params["model_name"]
        emb = params["emb_type"]
        data = saved["data_config"]
        data["dataset_name"] = params["dataset_name"]
        model_cfg = dict(saved["model_config"])
        for key in ["use_wandb", "learning_rate", "add_uuid", "l2", "num_epochs"]:
            model_cfg.pop(key, None)
        if name != "dimkt":
            model_cfg.pop("batch_size", None)
        if name in ("saint", "sakt"):
            model_cfg["seq_len"] = saved["train_config"]["seq_len"]
        device = torch.device(args.device)
        model = init_model(name, model_cfg, data, emb, params["dataset_name"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        diff_level = model_cfg.get("difficult_levels", 100) if name == "dimkt" else None
        loader = question_test_loader(name, data, args.batch_size, diff_level)
        prediction_path = output / f"{DISPLAY_TO_INTERNAL[display]}_test_question_predictions.tsv"
        aucs, accs = evaluate_question(model, loader, name, save_path=str(prediction_path))
        rmse, n, macro, macro_n = question_rmse(prediction_path)
        result = {
            "model": display, "fold": int(params["fold"]), "seed": int(record["seed"]),
            "AUC": aucs["late_mean"], "ACC": accs["late_mean"], "RMSE": rmse,
            "Macro-KCAUC": macro, "macro_kcauc_num_kcs": macro_n,
            "question_interactions": n, "aggregation": "late_mean",
            "checkpoint": str(checkpoint), "prediction_file": str(prediction_path),
        }
        rows.append(result)
        pd.DataFrame(rows).to_csv(root / "standard_questionlevel_results.csv", index=False)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del model, loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
