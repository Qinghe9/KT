"""Train pyKT baselines and export local experiment CSVs.

Run this file from ``examples/`` after dataset preprocessing.  Every model gets
its own timestamped log, checkpoint directory and prediction file.  The three
CSV outputs intentionally use the same columns as the supplied record sheets.
"""

import argparse
import ast
import csv
import fcntl
import gc
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

# The upstream example scripts assume pykt-toolkit is installed as a package.
# Also support running directly from a source checkout under examples/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from wandb_predict import main as predict_main
from wandb_train import main as train_main


MODEL_CONFIGS = {
    "dkt": {
        "emb_type": "qid", "learning_rate": 1e-3, "dropout": 0.2,
        "emb_size": 200,
    },
    "dkt+": {
        "emb_type": "qid", "learning_rate": 1e-3, "dropout": 0.2,
        "emb_size": 200, "lambda_r": 0.01, "lambda_w1": 0.003,
        "lambda_w2": 3.0,
    },
    "dkvmn": {
        "emb_type": "qid", "learning_rate": 1e-3, "dropout": 0.2,
        "dim_s": 200, "size_m": 50,
    },
    "sakt": {
        "emb_type": "qid", "learning_rate": 1e-3, "dropout": 0.2,
        "emb_size": 256, "num_attn_heads": 8, "num_en": 1,
    },
    "saint": {
        "emb_type": "qid", "learning_rate": 1e-3, "dropout": 0.2,
        "emb_size": 256, "num_attn_heads": 8, "n_blocks": 4,
    },
    "akt": {
        "emb_type": "qid", "learning_rate": 1e-4, "dropout": 0.2,
        "d_model": 256, "d_ff": 512, "num_attn_heads": 8, "n_blocks": 4,
    },
    "simplekt": {
        "emb_type": "qid", "learning_rate": 1e-4, "dropout": 0.1,
        "d_model": 256, "d_ff": 256, "num_attn_heads": 4, "n_blocks": 2,
        "final_fc_dim": 256, "final_fc_dim2": 256, "num_layers": 2,
        "nheads": 4, "loss1": 0.5, "loss2": 0.5, "loss3": 0.5,
        "start": 50,
    },
    "dtransformer": {
        "emb_type": "qid_cl", "learning_rate": 1e-3, "dropout": 0.3,
        "d_model": 256, "d_ff": 256, "num_attn_heads": 8,
        "n_blocks": 4, "n_know": 16, "lambda_cl": 0.1, "window": 1,
        "proj": True, "hard_neg": False, "batch_size": 32,
    },
    "dimkt": {
        "emb_type": "qid", "learning_rate": 5e-4, "weight_decay": 1e-4,
        "dropout": 0.2, "emb_size": 128, "num_steps": 199,
        "difficult_levels": 100, "batch_size": 64,
    },
    "mockt": {
        "emb_type": "qid_conv_ker_noexp", "learning_rate": 1e-4,
        "dropout": 0.1, "d_model": 256, "d_ff": 64,
        "num_attn_heads": 4, "n_blocks": 4, "num_buckets": 256,
        "max_distance": 256, "bar_d": 64, "max_local_shift": 0.0,
        "n_hat": 0, "d_state": 16, "d_conv": 2, "expand": 1,
        "kernel_size": 4, "sequence1": 50, "sequence2": 150,
    },
    "qikt": {
        "emb_type": "iekt", "learning_rate": 1e-3, "dropout": 0.4,
        "emb_size": 300, "mlp_layer_num": 2,
        "other_config": {
            "output_mode": "an", "loss_q_all_lambda": 0,
            "loss_c_all_lambda": 0, "loss_q_next_lambda": 0,
            "loss_c_next_lambda": 0, "output_q_all_lambda": 1,
            "output_c_all_lambda": 1, "output_q_next_lambda": 0,
            "output_c_next_lambda": 1,
        },
    },
}

DETAIL_FIELDS = ["model", "实验时间", "模型类型", "实验问题", "论文", "代码仓库",
                 "核心指标", "原论文指标范围", "实验使用的超参数", "原论文超参数",
                 "实验的数据集", "原论文的数据集", "论文发现的现象"]
TABLE_FIELDS = ["model", "实验时间", "AUC", "ACC", "RMSE", "KC-MND（平方）%",
                "DOA", "KC-MND（绝对值）%"]
RUN_FIELDS = ["model", "实验时间", "fold", "batch_size", "ckpt_path", "dataset",
              "emb_dim", "epochs", "lr", "seed", "test_acc", "test_auc",
              "test_recall", "Command", "test_rmse", "Parent items"]


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def prediction_rmse(path):
    trues, scores = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = ast.literal_eval(line)
            trues.extend(row[3]); scores.extend(row[4])
    if not trues:
        raise RuntimeError(f"No predictions found in {path}")
    return float(np.sqrt(np.mean((np.asarray(trues) - np.asarray(scores)) ** 2)))


def upsert_csv(path, fields, row):
    # Separate lock files remain stable while the CSV is opened/truncated and
    # let independent GPU workers safely update the same report.
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = [item for item in csv.DictReader(handle) if item.get("model") != row["model"]]
        rows.append(row)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        fcntl.flock(lock, fcntl.LOCK_UN)


DISPLAY_NAMES = {
    "dkt": "DKT", "dkt+": "DKT+", "dkvmn": "DKVMN", "sakt": "SAKT",
    "saint": "SAINT", "akt": "AKT", "simplekt": "simpleKT",
    "dtransformer": "DTransformer", "dimkt": "DIMKT", "mockt": "MoCKT",
    "qikt": "QIKT",
}

MODEL_TYPES = {
    "dkt": "循环神经网络", "dkt+": "正则化循环神经网络",
    "dkvmn": "记忆网络", "sakt": "自注意力模型",
    "saint": "Transformer编码器-解码器", "akt": "单调注意力模型",
    "simplekt": "Transformer/注意力模型", "dtransformer": "诊断Transformer",
    "dimkt": "难度感知知识追踪模型", "mockt": "卷积/混合结构模型",
    "qikt": "题目级知识追踪模型",
}


def display_name(model):
    return DISPLAY_NAMES[model]


def dataset_display_name(dataset_name):
    return {"ednet": "EdNet", "junyi2015": "Junyi2015"}.get(
        dataset_name, dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="ednet")
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS, default=list(MODEL_CONFIGS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save-dir", default="../reproduction/ednet_baselines/checkpoints")
    parser.add_argument("--output-dir", default="../reproduction/ednet_baselines")
    parser.add_argument("--parallel", action="store_true",
                        help="run one model worker per GPU concurrently")
    parser.add_argument("--gpu-ids", default="0,1",
                        help="comma-separated physical GPU IDs used by --parallel")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).resolve().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    command = " ".join(shlex.quote(x) for x in sys.argv)

    with open("../configs/data_config.json", encoding="utf-8") as handle:
        data_cfg = json.load(handle).get(args.dataset_name)
    if not data_cfg:
        raise SystemExit(f"Dataset {args.dataset_name!r} is absent from configs/data_config.json; run preprocessing first.")
    if args.fold not in data_cfg.get("folds", []):
        raise SystemExit(
            f"Invalid fold {args.fold} for {args.dataset_name}; "
            f"available folds are {data_cfg.get('folds', [])}."
        )
    required = ["train_valid_sequences.csv", "test_sequences.csv",
                "train_valid_sequences_quelevel.csv", "test_sequences_quelevel.csv"]
    missing = [name for name in required if not (Path(data_cfg["dpath"]) / name).exists()]
    if missing:
        raise SystemExit(f"Preprocessed data missing: {missing}. Run examples/data_preprocess.py first.")

    if args.parallel:
        gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
        if not gpu_ids:
            raise SystemExit("--gpu-ids must contain at least one GPU ID")
        model_queue = Queue()
        for model in args.models:
            model_queue.put(model)

        def run_gpu_queue(gpu_id):
            while True:
                try:
                    model = model_queue.get_nowait()
                except Empty:
                    return
                child_cmd = [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    "--models", model, "--dataset-name", args.dataset_name,
                    "--fold", str(args.fold), "--seed", str(args.seed),
                    "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
                    "--save-dir", str(Path(args.save_dir).resolve()),
                    "--output-dir", str(output_dir),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(f"GPU {gpu_id} -> {model}: {' '.join(shlex.quote(x) for x in child_cmd)}", flush=True)
                completed = subprocess.run(child_cmd, env=env)
                if completed.returncode != 0:
                    raise RuntimeError(f"{model} failed on GPU {gpu_id} with exit code {completed.returncode}")
                model_queue.task_done()

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(run_gpu_queue, gpu) for gpu in gpu_ids]
            for future in futures:
                future.result()
        print("All parallel model workers completed successfully.")
        return

    for model in args.models:
        name = display_name(model)
        dataset_label = dataset_display_name(args.dataset_name)
        cfg = dict(MODEL_CONFIGS[model])
        params = {
            "dataset_name": args.dataset_name, "model_name": model,
            "save_dir": str(Path(args.save_dir).resolve()), "seed": args.seed,
            "fold": args.fold, "batch_size": args.batch_size,
            "num_epochs": args.epochs, "use_wandb": 0, "add_uuid": 0,
            **cfg,
        }
        log_path = log_dir / f"{model}_{stamp}.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as log, \
                redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"[{datetime.now().isoformat()}] START {name}\ncommand: {command}\nparams: {json.dumps(params, ensure_ascii=False)}")
            train = train_main(params)
            # train_main builds a second copy of the best model before it
            # returns.  On memory-heavy models (notably DTransformer), cached
            # allocations can otherwise make the immediate evaluation reload
            # fail even though the checkpoint itself is valid.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = predict_main({"save_dir": train["ckpt_path"], "bz": train["batch_size"],
                                   "fusion_type": "early_fusion,late_fusion", "use_wandb": 0,
                                   "skip_question": 1, "skip_window": 1, "compute_rmse": 1,
                                   "save_predictions": 0})
            rmse = float(result["test_rmse"])
            auc, acc = float(result["testauc"]), float(result["testacc"])
            now = datetime.now().strftime("%Y/%m/%d")
            hyper = {k: v for k, v in cfg.items() if k not in ("emb_type", "other_config")}
            actual_batch_size = int(train["batch_size"])
            hyper.update({"batch_size": actual_batch_size, "epochs": args.epochs})
            test_ratio = float(data_cfg.get("test_ratio", 0.2))
            valid_ratio = (1.0 - test_ratio) / len(data_cfg["folds"])
            train_ratio = 1.0 - test_ratio - valid_ratio
            split_desc = (f"user-level {train_ratio:.0%}/{valid_ratio:.0%}/"
                          f"{test_ratio:.0%} train/valid/test")
            stats = (f"{dataset_label} ({args.dataset_name}); num_q={data_cfg['num_q']}, "
                     f"num_c={data_cfg['num_c']}, min_seq_len={data_cfg['min_seq_len']}, "
                     f"maxlen={data_cfg['maxlen']}; {split_desc}")
            upsert_csv(output_dir / "习题推荐基线模型实验记录表_pykt基线记录_Grid.csv", DETAIL_FIELDS, {
                "model": name, "实验时间": now, "模型类型": MODEL_TYPES[model],
                "实验问题": f"{dataset_label} fold={args.fold} baseline; {split_desc}; standard non-window test",
                "代码仓库": str(Path.cwd().parent),
                "核心指标": f"AUC={auc:.6f}; ACC={acc:.6f}; RMSE={rmse:.6f}",
                "实验使用的超参数": json.dumps(hyper, ensure_ascii=False), "实验的数据集": stats,
            })
            upsert_csv(output_dir / "习题推荐基线模型实验记录表_Table_Grid.csv", TABLE_FIELDS, {
                "model": name, "实验时间": now, "AUC": auc, "ACC": acc, "RMSE": rmse,
                "KC-MND（平方）%": "未计算", "DOA": "未计算", "KC-MND（绝对值）%": "未计算",
            })
            upsert_csv(output_dir / "习题推荐基线模型实验记录表_实验记录_Grid.csv", RUN_FIELDS, {
                "model": name, "实验时间": now, "fold": args.fold, "batch_size": actual_batch_size,
                "ckpt_path": train["model_save_path"], "dataset": args.dataset_name,
                "emb_dim": cfg.get("d_model", cfg.get("emb_size", "")), "epochs": args.epochs,
                "lr": cfg["learning_rate"], "seed": args.seed, "test_acc": acc,
                "test_auc": auc, "Command": command, "test_rmse": rmse,
            })
            print(f"[{datetime.now().isoformat()}] DONE {name}: AUC={auc:.6f}, ACC={acc:.6f}, RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
