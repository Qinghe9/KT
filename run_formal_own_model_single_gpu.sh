#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
RUN_ROOT="$ROOT/runs/formal_own_model"

GPU_ID=${GPU_ID:-0}
WORKERS=${WORKERS:-2}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
TOPK=${TOPK:-2}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
sha256sum -c "$DATA_ROOT/SHA256SUMS"

run_one() {
  local seed=$1 dataset=$2
  local out="$RUN_ROOT/seed_${seed}/${dataset}"
  local log="$RUN_ROOT/logs/${dataset}_seed${seed}_gpu${GPU_ID}.log"
  local batch
  case "$dataset" in
    assist2012) batch=${ASSIST2012_BATCH:-6144} ;;
    assist2017) batch=${ASSIST2017_BATCH:-4096} ;;
    ednet)      batch=${EDNET_BATCH:-4096} ;;
    assist2009) batch=${ASSIST2009_BATCH:-2048} ;;
    *) echo "unsupported formal dataset: $dataset" >&2; return 2 ;;
  esac
  mkdir -p "$out"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP completed dataset=$dataset seed=$seed"
    return 0
  fi
  local resume_args=()
  if [[ -s "$out/last.pt" ]]; then
    resume_args=(--resume-from "$out/last.pt")
    echo "[$(date -Is)] RESUME dataset=$dataset seed=$seed batch=$batch gpu=$GPU_ID"
  else
    echo "[$(date -Is)] START dataset=$dataset seed=$seed batch=$batch gpu=$GPU_ID"
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    --fold-data --fold 0 \
    --data "$DATA_ROOT/$dataset" \
    --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --topk "$TOPK" --lr 0.003 \
    --seed "$seed" --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-weak 0.2 --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test "${resume_args[@]}" \
    > "$log" 2>&1
  echo "[$(date -Is)] DONE dataset=$dataset seed=$seed gpu=$GPU_ID"
}

# Two balanced queues share the single GPU.  Each process is allocator-limited
# to 47%, so concurrent KT+WeakKC backward passes cannot exhaust the card.
(
  for dataset in assist2009 ednet assist2017 assist2012; do
    run_one 3407 "$dataset"
  done
) &
PID_A=$!

(
  for dataset in assist2009 ednet assist2017 assist2012; do
    run_one 2026 "$dataset"
  done
) &
PID_B=$!

echo "[$(date -Is)] queues started: seed3407_pid=$PID_A seed2026_pid=$PID_B gpu=$GPU_ID"
wait "$PID_A"
wait "$PID_B"

for seed in 3407 2026; do
  for dataset in assist2009 ednet assist2017 assist2012; do
    test -s "$RUN_ROOT/seed_${seed}/${dataset}/results.csv"
    test -s "$RUN_ROOT/seed_${seed}/${dataset}/metrics.json"
  done
done

"$PY" export_template_csv.py \
  --run assist2012-seed3407="$RUN_ROOT/seed_3407/assist2012" \
  --run ednet-seed3407="$RUN_ROOT/seed_3407/ednet" \
  --run assist2009-seed3407="$RUN_ROOT/seed_3407/assist2009" \
  --run assist2017-seed3407="$RUN_ROOT/seed_3407/assist2017" \
  --run assist2012-seed2026="$RUN_ROOT/seed_2026/assist2012" \
  --run ednet-seed2026="$RUN_ROOT/seed_2026/ednet" \
  --run assist2009-seed2026="$RUN_ROOT/seed_2026/assist2009" \
  --run assist2017-seed2026="$RUN_ROOT/seed_2026/assist2017" \
  --output-dir "$RUN_ROOT/final_csv"

echo "[$(date -Is)] ALL FORMAL OWN-MODEL RUNS COMPLETED; CSV=$RUN_ROOT/final_csv"
