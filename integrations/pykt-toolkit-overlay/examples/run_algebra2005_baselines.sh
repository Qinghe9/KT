#!/usr/bin/env bash
set -uo pipefail

ROOT=${PYKT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PYTHON:-python}
OUT=${OUT:-"$ROOT/reproduction/algebra2005_baselines_seed3407"}
STANDARD="$OUT/standard"
CHECKPOINTS="$OUT/checkpoints"
WEAK="$OUT/weakkc"
SEED=3407
GPU_ID=${GPU_ID:-0}
STANDARD_EPOCHS=${STANDARD_EPOCHS:-200}
WEAK_EPOCHS=${WEAK_EPOCHS:-30}
BATCH=${BATCH:-128}

cd "$ROOT/examples"
mkdir -p "$STANDARD" "$CHECKPOINTS" "$WEAK" "$OUT/logs"

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

run_model() {
  local model=$1 display ckpt
  display=$(display_name "$model")
  if [[ -s "$WEAK/$model/results.csv" && -s "$WEAK/$model/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP completed model=$model"
    return 0
  fi
  ckpt=$(checkpoint_dir "$display")
  if [[ -z "$ckpt" || ! -s "$ckpt/config.json" ]]; then
    echo "[$(date -Is)] STANDARD START model=$model"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u run_ednet_baselines.py \
      --models "$model" --dataset-name algebra2005 --fold 0 --seed "$SEED" \
      --batch-size "$BATCH" --epochs "$STANDARD_EPOCHS" \
      --output-dir "$STANDARD" --save-dir "$CHECKPOINTS" \
      >"$OUT/logs/${model}_standard.log" 2>&1 || {
        echo "[$(date -Is)] STANDARD FAILED model=$model"; return 1; }
    ckpt=$(checkpoint_dir "$display")
  fi
  if [[ -z "$ckpt" || ! -s "$ckpt/config.json" ]]; then
    echo "[$(date -Is)] CHECKPOINT MISSING model=$model"; return 1
  fi
  mkdir -p "$WEAK/$model"
  echo "[$(date -Is)] WEAK START model=$model checkpoint=$ckpt"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u train_weakkc_baseline.py \
    --checkpoint-dir "$ckpt" --out "$WEAK/$model" \
    --epochs "$WEAK_EPOCHS" --patience 10 --batch-size "$BATCH" \
    --weak-update-every 4 --lambda-weak 0.2 --lr 0.0001 \
    --workers 1 --seed "$SEED" --device cuda \
    >"$OUT/logs/${model}_weakkc.log" 2>&1 || {
      echo "[$(date -Is)] WEAK FAILED model=$model"; return 1; }
  echo "[$(date -Is)] DONE model=$model"
}

run_queue() {
  local model
  for model in "$@"; do
    run_model "$model" || true
  done
}

run_queue dkt dkvmn saint simplekt dimkt & PID_A=$!
run_queue dkt+ sakt akt dtransformer mockt & PID_B=$!
echo "[$(date -Is)] queues launched pid_a=$PID_A pid_b=$PID_B gpu=$GPU_ID"
wait "$PID_A"
wait "$PID_B"

"$PY" summarize_algebra2005_baselines.py --root "$OUT" --seed "$SEED"
echo "[$(date -Is)] BASELINE PIPELINE FINISHED output=$OUT"
