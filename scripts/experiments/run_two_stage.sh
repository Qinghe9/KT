#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data/atomic_fold0"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/runs/two_stage"}
GPU_IDS=${GPU_IDS:-0,1}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
datasets=(assist2009 assist2012 assist2017 ednet algebra2005)
seeds=(3407 2026)

batch_for() { case "$1" in assist2009) echo 2048;; assist2012) echo 6144;; *) echo 4096;; esac; }

common_args() {
  local dataset=$1 seed=$2 out=$3
  printf '%q ' --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs 100 --patience 10 --min-delta 0.001 --scheduler-patience 3 \
    --lr-factor 0.5 --min-lr 0.000001 --batch "$(batch_for "$dataset")" \
    --workers 2 --weak-workers 2 --bucket-by-length --gpu-memory-fraction 0.95 \
    --amp --dim 32 --context 200 --topk 2 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 --weak-history-ratio 0.7 --weak-min-history 16 \
    --weak-min-future 10 --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --weak-update-every 4 --device cuda:0 --test
}

run_pair() {
  local gpu=$1 dataset=$2 seed=$3 base kt weak args
  base="$RUN_ROOT/seed_${seed}/${dataset}"; kt="$base/kt_only"; weak="$base/two_stage"
  mkdir -p "$kt" "$weak" "$RUN_ROOT/logs"
  if [[ ! -s "$kt/metrics.json" ]]; then
    eval "args=( $(common_args "$dataset" "$seed" "$kt") )"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/train.py" "${args[@]}" \
      --training-mode kt_only --lambda-weak 0 --lambda-balance 0.01 \
      >"$RUN_ROOT/logs/${dataset}_seed${seed}_kt.log" 2>&1
  fi
  [[ -s "$kt/best.pt" ]] || { echo "missing $kt/best.pt" >&2; exit 1; }
  if [[ ! -s "$weak/metrics.json" ]]; then
    eval "args=( $(common_args "$dataset" "$seed" "$weak") )"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/train.py" "${args[@]}" \
      --training-mode weak_head --init-from "$kt/best.pt" --lambda-weak 0 --lambda-balance 0 \
      >"$RUN_ROOT/logs/${dataset}_seed${seed}_weak.log" 2>&1
  fi
}

tasks=(); for seed in "${seeds[@]}"; do for d in "${datasets[@]}"; do tasks+=("$d:$seed"); done; done
pids=()
for i in "${!GPUS[@]}"; do
  ( for j in "${!tasks[@]}"; do (( j % ${#GPUS[@]} == i )) || continue; IFS=: read -r d s <<<"${tasks[$j]}"; run_pair "${GPUS[$i]}" "$d" "$s"; done ) & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
