#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT=/home/u202310176043/project/pykt-toolkit/data
JUNYI_PID=${JUNYI_PID:?set JUNYI_PID to the active Junyi training PID}
OLD_QUEUE_PID=${OLD_QUEUE_PID:-}
OLD_LAUNCHER_PID=${OLD_LAUNCHER_PID:-}

cd "$ROOT"
mkdir -p runs/fold0/logs data/fold0

run_one() {
  local dataset=$1
  echo "[$(date -Is)] START $dataset GPU=1"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u train.py \
    --prepare --fold-data --fold 0 \
    --source "$DATA_ROOT/$dataset/train_valid_quelevel.csv" \
    --test-source "$DATA_ROOT/$dataset/test_quelevel.csv" \
    --data "data/fold0/$dataset" \
    --out "runs/fold0/$dataset" \
    --users 0 --cap 0 \
    --epochs 100 --patience 10 --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch 1024 --workers 4 --gpu-memory-fraction 0.9 \
    --dim 32 --context 200 --topk 4 --lr 0.003 --seed 3407 \
    --recall-ks 1 3 5 10 --device cuda:0 --test \
    > "runs/fold0/logs/$dataset.log" 2>&1
  echo "[$(date -Is)] DONE $dataset GPU=1"
}

run_one assist2012
run_one ednet

echo "[$(date -Is)] GPU1 queue complete; waiting for Junyi PID=$JUNYI_PID"
while kill -0 "$JUNYI_PID" 2>/dev/null; do
  sleep 30
done

for dataset in junyi2015 assist2009 assist2012 assist2017 ednet; do
  test -s "runs/fold0/$dataset/results.csv"
  test -s "runs/fold0/$dataset/metrics.json"
done

"$PY" export_template_csv.py \
  --run junyi2015=runs/fold0/junyi2015 \
  --run assist2009=runs/fold0/assist2009 \
  --run assist2012=runs/fold0/assist2012 \
  --run assist2017=runs/fold0/assist2017 \
  --run ednet=runs/fold0/ednet \
  --output-dir runs/fold0/final_csv

echo "[$(date -Is)] ALL DONE; CSV: $ROOT/runs/fold0/final_csv"

# The original static controller was intentionally stopped before this script
# started so it could not launch duplicate ASSIST2012/EdNet jobs.
if [[ -n "$OLD_QUEUE_PID" ]]; then kill -TERM "$OLD_QUEUE_PID" 2>/dev/null || true; fi
if [[ -n "$OLD_LAUNCHER_PID" ]]; then kill -TERM "$OLD_LAUNCHER_PID" 2>/dev/null || true; fi
