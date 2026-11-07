#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA="$ROOT/data/atomic_fold0/algebra2005"
RUN_ROOT="$ROOT/runs/formal_own_model_algebra2005"
GPU_ID=${GPU_ID:-0}
BATCH=${BATCH:-4096}
WORKERS=${WORKERS:-1}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/final_csv"
test -s "$DATA/students.json"
test -s "$DATA/manifest.json"

run_one() {
  local seed=$1
  local out="$RUN_ROOT/seed_${seed}/algebra2005"
  local log="$RUN_ROOT/logs/algebra2005_seed${seed}_gpu${GPU_ID}.log"
  mkdir -p "$out"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP completed seed=$seed"
    return 0
  fi
  local resume=()
  if [[ -s "$out/last.pt" ]]; then
    resume=(--resume-from "$out/last.pt")
    echo "[$(date -Is)] RESUME seed=$seed gpu=$GPU_ID"
  else
    echo "[$(date -Is)] START seed=$seed gpu=$GPU_ID"
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    --fold-data --fold 0 --bucket-by-length \
    --data "$DATA" --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$BATCH" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --topk 2 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-weak 0.2 --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test "${resume[@]}" >"$log" 2>&1
  echo "[$(date -Is)] DONE seed=$seed"
}

# Length bucketing leaves short gaps between FFT/Mamba dispatch groups.  Two
# independent seeds share the GPU to fill those gaps. Each process is capped at
# 47% of VRAM and uses one loader worker to bound host/shared-memory pressure.
run_one 3407 & PID_A=$!
run_one 2026 & PID_B=$!
echo "[$(date -Is)] concurrent seeds started: seed3407_pid=$PID_A seed2026_pid=$PID_B gpu=$GPU_ID"
wait "$PID_A"
wait "$PID_B"

for seed in 3407 2026; do
  test -s "$RUN_ROOT/seed_${seed}/algebra2005/results.csv"
  test -s "$RUN_ROOT/seed_${seed}/algebra2005/metrics.json"
done

"$PY" export_template_csv.py \
  --run algebra2005-seed3407="$RUN_ROOT/seed_3407/algebra2005" \
  --run algebra2005-seed2026="$RUN_ROOT/seed_2026/algebra2005" \
  --output-dir "$RUN_ROOT/final_csv"

echo "[$(date -Is)] ALGEBRA2005 COMPLETED; CSV=$RUN_ROOT/final_csv"
