#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT=/home/u202310176043/project/pykt-toolkit/data
USERS=${USERS:-0}       # 0 = all students
CAP=${CAP:-0}           # 0 = complete sequence
BATCH=${BATCH:-1024}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}

cd "$ROOT"
mkdir -p runs/fold0/logs data/fold0

run_one() {
  local gpu=$1 dataset=$2
  echo "[$(date -Is)] START $dataset GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train.py \
    --prepare --fold-data --fold 0 \
    --source "$DATA_ROOT/$dataset/train_valid_quelevel.csv" \
    --test-source "$DATA_ROOT/$dataset/test_quelevel.csv" \
    --data "data/fold0/$dataset" \
    --out "runs/fold0/$dataset" \
    --users "$USERS" --cap "$CAP" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$BATCH" --workers "$WORKERS" --gpu-memory-fraction 0.9 \
    --dim 32 --context 200 --topk 4 --lr 0.003 --seed 3407 \
    --recall-ks 1 3 5 10 --device cuda:0 --test \
    > "runs/fold0/logs/$dataset.log" 2>&1
  echo "[$(date -Is)] DONE $dataset GPU=$gpu"
}

# Independent queues keep both cards occupied without cross-model contention.
(run_one 0 junyi2015; run_one 0 assist2012; run_one 0 ednet) &
PID0=$!
(run_one 1 assist2009; run_one 1 assist2017) &
PID1=$!
echo "GPU0 queue PID=$PID0; GPU1 queue PID=$PID1"
wait "$PID0" "$PID1"

"$PY" export_template_csv.py \
  --run junyi2015=runs/fold0/junyi2015 \
  --run assist2009=runs/fold0/assist2009 \
  --run assist2012=runs/fold0/assist2012 \
  --run assist2017=runs/fold0/assist2017 \
  --run ednet=runs/fold0/ednet \
  --output-dir runs/fold0/final_csv
echo "[$(date -Is)] ALL DONE; CSV: $ROOT/runs/fold0/final_csv"
