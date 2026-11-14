#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data/atomic_fold0"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/runs/formal_main"}
GPU_IDS=${GPU_IDS:-0,1}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_IDS is empty" >&2; exit 2; }

datasets=(assist2009 assist2012 assist2017 ednet algebra2005)
seeds=(3407 2026)

batch_for() {
  case "$1" in
    assist2009) echo "${ASSIST2009_BATCH:-2048}" ;;
    assist2012) echo "${ASSIST2012_BATCH:-6144}" ;;
    assist2017|ednet|algebra2005) echo "${DEFAULT_BATCH:-4096}" ;;
  esac
}

run_one() {
  local gpu=$1 dataset=$2 seed=$3 batch out log
  batch=$(batch_for "$dataset")
  out="$RUN_ROOT/seed_${seed}/${dataset}"
  log="$RUN_ROOT/logs/${dataset}_seed${seed}.log"
  mkdir -p "$out" "$(dirname "$log")"
  if [[ -s "$out/metrics.json" && -s "$out/test_predictions.csv" ]]; then
    echo "[$(date -Is)] SKIP $dataset seed=$seed"
    return
  fi
  local resume=()
  [[ -s "$out/last.pt" ]] && resume=(--resume-from "$out/last.pt")
  echo "[$(date -Is)] START $dataset seed=$seed gpu=$gpu batch=$batch"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/train.py" \
    --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs 100 --patience 10 --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers 2 --weak-workers 2 --bucket-by-length \
    --gpu-memory-fraction 0.95 --amp --dim 32 --context 200 --topk 2 \
    --lr 0.003 --seed "$seed" --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-weak 0.2 --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test "${resume[@]}" >"$log" 2>&1
  echo "[$(date -Is)] DONE $dataset seed=$seed"
}

tasks=()
for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do tasks+=("$dataset:$seed"); done
done

pids=()
for i in "${!GPUS[@]}"; do
  (
    for j in "${!tasks[@]}"; do
      (( j % ${#GPUS[@]} == i )) || continue
      IFS=: read -r dataset seed <<< "${tasks[$j]}"
      run_one "${GPUS[$i]}" "$dataset" "$seed"
    done
  ) & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[$(date -Is)] all main experiments completed"
