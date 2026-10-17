#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
SEED=${SEED:-3407}
FORMAL_ROOT="$ROOT/runs/formal_own_model/seed_${SEED}"
RUN_ROOT="$ROOT/runs/ablation_small_seed${SEED}"

GPU_ID=${GPU_ID:-0}
WORKERS=${WORKERS:-2}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"

run_one() {
  local dataset=$1 variant=$2
  local out="$RUN_ROOT/$dataset/$variant"
  local log="$RUN_ROOT/logs/${dataset}_${variant}.log"
  local batch extra=()

  case "$dataset" in
    assist2009) batch=${ASSIST2009_BATCH:-2048} ;;
    ednet)      batch=${EDNET_BATCH:-4096} ;;
    *) echo "unsupported dataset: $dataset" >&2; return 2 ;;
  esac

  case "$variant" in
    no_spectrum) extra=(--no-spectrum --topk 2 --lambda-weak 0.2) ;;
    top4)        extra=(--topk 4 --lambda-weak 0.2) ;;
    no_weak)     extra=(--topk 2 --lambda-weak 0) ;;
    *) echo "unsupported ablation: $variant" >&2; return 2 ;;
  esac

  mkdir -p "$out"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP completed dataset=$dataset variant=$variant"
    return 0
  fi

  local resume_args=()
  if [[ -s "$out/last.pt" ]]; then
    resume_args=(--resume-from "$out/last.pt")
    echo "[$(date -Is)] RESUME dataset=$dataset variant=$variant"
  else
    echo "[$(date -Is)] START dataset=$dataset variant=$variant"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --lr 0.003 --seed "$SEED" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test "${extra[@]}" "${resume_args[@]}" \
    > "$log" 2>&1
  echo "[$(date -Is)] DONE dataset=$dataset variant=$variant"
}

# Two balanced queues share the 24 GiB GPU. Each process is capped at 47%.
(
  run_one assist2009 no_spectrum
  run_one ednet top4
  run_one assist2009 no_weak
) &
PID_A=$!

(
  run_one assist2009 top4
  run_one ednet no_spectrum
  run_one ednet no_weak
) &
PID_B=$!

echo "[$(date -Is)] ablation queues started: pid_a=$PID_A pid_b=$PID_B gpu=$GPU_ID"
wait "$PID_A"
wait "$PID_B"

for dataset in assist2009 ednet; do
  test -s "$FORMAL_ROOT/$dataset/metrics.json"
  for variant in no_spectrum top4 no_weak; do
    test -s "$RUN_ROOT/$dataset/$variant/metrics.json"
  done
done

"$PY" summarize_ablation.py \
  --formal-root "$FORMAL_ROOT" \
  --ablation-root "$RUN_ROOT" \
  --datasets assist2009 ednet \
  --seed "$SEED" \
  --output "$RUN_ROOT/ablation_results.csv"

echo "[$(date -Is)] ALL ABLATIONS COMPLETED; CSV=$RUN_ROOT/ablation_results.csv"
