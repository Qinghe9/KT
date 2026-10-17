#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
RUN_ROOT="$ROOT/runs/two_stage_assist1217"
GPU_ID=${GPU_ID:-0}
WORKERS=${WORKERS:-2}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"

run_stage() {
  local seed=$1 dataset=$2 stage=$3
  local out="$RUN_ROOT/seed_${seed}/$dataset/$stage"
  local log="$RUN_ROOT/logs/${dataset}_seed${seed}_${stage}.log"
  local batch init=() mode lambda
  case "$dataset" in
    assist2012) batch=${ASSIST2012_BATCH:-6144} ;;
    assist2017) batch=${ASSIST2017_BATCH:-4096} ;;
    *) echo "unsupported dataset: $dataset" >&2; return 2 ;;
  esac
  case "$stage" in
    kt_only) mode=kt_only; lambda=0 ;;
    two_stage)
      mode=weak_head; lambda=0
      local source="$RUN_ROOT/seed_${seed}/$dataset/kt_only/best.pt"
      [[ -s "$source" ]] || { echo "missing KT-only checkpoint: $source" >&2; return 1; }
      init=(--init-from "$source")
      ;;
    *) return 2 ;;
  esac

  mkdir -p "$out"
  if [[ -s "$out/metrics.json" && -s "$out/results.csv" ]]; then
    echo "[$(date -Is)] SKIP seed=$seed dataset=$dataset stage=$stage"
    return
  fi
  local resume=()
  if [[ -s "$out/last.pt" ]]; then resume=(--resume-from "$out/last.pt"); fi
  echo "[$(date -Is)] START seed=$seed dataset=$dataset stage=$stage batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --topk 2 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --training-mode "$mode" --lambda-weak "$lambda" --lambda-balance 0.01 \
    --weak-update-every 4 --device cuda:0 --test \
    "${init[@]}" "${resume[@]}" > "$log" 2>&1
  echo "[$(date -Is)] DONE seed=$seed dataset=$dataset stage=$stage"
}

run_dataset_queue() {
  local dataset=$1
  for seed in 3407 2026; do
    run_stage "$seed" "$dataset" kt_only
    run_stage "$seed" "$dataset" two_stage
  done
}

run_dataset_queue assist2012 & pid_a=$!
run_dataset_queue assist2017 & pid_b=$!
echo "[$(date -Is)] queues started assist2012_pid=$pid_a assist2017_pid=$pid_b"
wait "$pid_a"; wait "$pid_b"

"$PY" summarize_two_stage_assist1217.py --root "$RUN_ROOT"
echo "[$(date -Is)] ALL TWO-STAGE ASSIST2012/2017 RUNS COMPLETED"
