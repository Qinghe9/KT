#!/usr/bin/env bash
set -uo pipefail

ROOT=${PYKT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PYTHON:-python}
OUT=${OUT:-"$ROOT/reproduction/algebra2005_baselines_seed3407"}
STANDARD="$OUT/standard"
CHECKPOINTS="$OUT/checkpoints"
WEAK="$OUT/weakkc"
GPU_ID=${GPU_ID:-0}
cd "$ROOT/examples"
mkdir -p "$OUT/logs"

display_name() {
  case "$1" in
    dkt) echo DKT ;; dkt+) echo DKT+ ;; dkvmn) echo DKVMN ;;
    sakt) echo SAKT ;; saint) echo SAINT ;; akt) echo AKT ;;
    simplekt) echo simpleKT ;; dtransformer) echo DTransformer ;;
    dimkt) echo DIMKT ;; mockt) echo MoCKT ;;
  esac
}

checkpoint_dir() {
  local display=$1
  "$PY" - "$STANDARD/习题推荐基线模型实验记录表_实验记录_Grid.csv" "$display" <<'PY'
import csv,sys
from pathlib import Path
path=Path(sys.argv[1])
if path.exists():
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['model']==sys.argv[2]:
                print(Path(row['ckpt_path']).parent)
                break
PY
}

evaluate_saved() {
  local model=$1 batch=$2 display ckpt
  [[ -s "$WEAK/$model/results.csv" ]] && { echo "[$(date -Is)] EVAL SKIP model=$model"; return 0; }
  display=$(display_name "$model"); ckpt=$(checkpoint_dir "$display")
  echo "[$(date -Is)] EVAL SAVED model=$model batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u train_weakkc_baseline.py --checkpoint-dir "$ckpt" \
    --out "$WEAK/$model" --batch-size "$batch" --workers 1 \
    --seed 3407 --device cuda --evaluate-only \
    >"$OUT/logs/${model}_weakkc_eval.log" 2>&1
}

train_weak() {
  local model=$1 batch=$2 display ckpt
  [[ -s "$WEAK/$model/results.csv" ]] && { echo "[$(date -Is)] WEAK SKIP model=$model"; return 0; }
  display=$(display_name "$model"); ckpt=$(checkpoint_dir "$display")
  [[ -n "$ckpt" && -s "$ckpt/config.json" ]] || { echo "missing checkpoint model=$model"; return 1; }
  echo "[$(date -Is)] WEAK RETRY model=$model batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u train_weakkc_baseline.py --checkpoint-dir "$ckpt" \
    --out "$WEAK/$model" --epochs 30 --patience 10 --batch-size "$batch" \
    --weak-update-every 4 --lambda-weak 0.2 --lr 0.0001 \
    --workers 1 --seed 3407 --device cuda \
    >"$OUT/logs/${model}_weakkc_retry.log" 2>&1
}

# These models already reached early stopping; only export their saved best.
evaluate_saved dkt 128 || true
evaluate_saved dkt+ 128 || true
evaluate_saved sakt 64 || true
evaluate_saved simplekt 64 || true
evaluate_saved dimkt 64 || true
evaluate_saved mockt 64 || true

# AKT was the only missing standard checkpoint. Run alone with a safe batch.
if [[ -z "$(checkpoint_dir AKT)" ]]; then
  echo "[$(date -Is)] AKT STANDARD RETRY batch=32"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u run_ednet_baselines.py --models akt --dataset-name algebra2005 \
    --fold 0 --seed 3407 --batch-size 32 --epochs 200 \
    --output-dir "$STANDARD" --save-dir "$CHECKPOINTS" \
    >"$OUT/logs/akt_standard_retry.log" 2>&1
fi

# Previously OOM models now run one at a time with reduced batches.
train_weak dkvmn 64 || true
train_weak saint 32 || true
train_weak dtransformer 8 || true
train_weak akt 32 || true

"$PY" summarize_algebra2005_baselines.py --root "$OUT" --seed 3407
echo "[$(date -Is)] RECOVERY FINISHED output=$OUT"
