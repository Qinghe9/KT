#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
RUN_ROOT="$ROOT/runs/formal_own_model"

BATCH=${BATCH:-512}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
sha256sum -c "$DATA_ROOT/SHA256SUMS"

run_one() {
  local gpu=$1 seed=$2 dataset=$3
  local out="$RUN_ROOT/seed_${seed}/${dataset}"
  local log="$RUN_ROOT/logs/${dataset}_seed${seed}_gpu${gpu}.log"
  mkdir -p "$out"
  echo "[$(date -Is)] START dataset=$dataset seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train.py \
    --fold-data --fold 0 \
    --data "$DATA_ROOT/$dataset" \
    --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$BATCH" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction 0.95 --amp \
    --dim 32 --context 200 --topk 4 --lr 0.003 \
    --seed "$seed" --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-weak 0.2 --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test \
    > "$log" 2>&1
  echo "[$(date -Is)] DONE dataset=$dataset seed=$seed gpu=$gpu"
}

# Junyi is kept sequential because loading its 25.9M interactions consumes a
# large amount of host RAM. GPU1 processes the other datasets meanwhile.
(
  run_one 0 3407 junyi2015
  run_one 0 2026 junyi2015
) &
PID0=$!

(
  for seed in 3407 2026; do
    run_one 1 "$seed" assist2012
    run_one 1 "$seed" ednet
    run_one 1 "$seed" assist2009
    run_one 1 "$seed" assist2017
  done
) &
PID1=$!

echo "GPU0 queue PID=$PID0; GPU1 queue PID=$PID1"
wait "$PID0" "$PID1"
echo "[$(date -Is)] ALL FORMAL OWN-MODEL RUNS COMPLETED"
