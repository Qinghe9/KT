#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
RUN_ROOT="$ROOT/runs/two_stage_remaining"
GPU_ID=${GPU_ID:-0}
WORKERS=${WORKERS:-2}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"

batch_for() {
  case "$1" in
    assist2012) echo "${ASSIST2012_BATCH:-6144}" ;;
    assist2017) echo "${ASSIST2017_BATCH:-4096}" ;;
    *) return 2 ;;
  esac
}

run_stage() {
  local seed=$1 dataset=$2 stage=$3
  local batch; batch=$(batch_for "$dataset")
  local base="$RUN_ROOT/seed_${seed}/$dataset"
  local out="$base/$stage"
  local log="$RUN_ROOT/logs/${dataset}_seed${seed}_${stage}.log"
  local extra=()

  if [[ "$stage" == kt_only ]]; then
    extra=(--training-mode kt_only --lambda-weak 0 --lambda-balance 0.01)
  elif [[ "$stage" == two_stage ]]; then
    local source="$base/kt_only/best.pt"
    [[ -s "$source" ]] || { echo "missing KT checkpoint: $source" >&2; return 1; }
    extra=(--training-mode weak_head --init-from "$source" --lambda-weak 0 --lambda-balance 0)
  else
    return 2
  fi

  mkdir -p "$out"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP seed=$seed dataset=$dataset stage=$stage"
    return
  fi
  local resume=()
  if [[ -s "$out/last.pt" ]]; then resume=(--resume-from "$out/last.pt"); fi

  echo "[$(date -Is)] START seed=$seed dataset=$dataset stage=$stage batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs 100 --patience 10 --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --topk 2 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 --weak-history-ratio 0.7 --weak-min-history 16 \
    --weak-min-future 10 --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --weak-update-every 4 --device cuda:0 --test \
    "${extra[@]}" "${resume[@]}" > "$log" 2>&1
  echo "[$(date -Is)] DONE seed=$seed dataset=$dataset stage=$stage"
}

run_pair() {
  local seed=$1 dataset=$2
  run_stage "$seed" "$dataset" kt_only
  run_stage "$seed" "$dataset" two_stage
}

# One seed per queue. Two processes share the RTX 3090 at 47% allocator caps.
(
  run_pair 3407 assist2017
  run_pair 3407 assist2012
) & pid_a=$!
(
  run_pair 2026 assist2017
  run_pair 2026 assist2012
) & pid_b=$!

echo "[$(date -Is)] queues started pid3407=$pid_a pid2026=$pid_b"
wait "$pid_a"; wait "$pid_b"

"$PY" summarize_two_stage_all.py
echo "[$(date -Is)] ALL REMAINING TWO-STAGE RUNS COMPLETED"
