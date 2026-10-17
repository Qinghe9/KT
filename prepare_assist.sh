#!/usr/bin/env bash
set -euo pipefail

PY=/home/u202310176043/.conda/envs/pykt/bin/python
ROOT=/home/u202310176043/project/pykt-toolkit
LOG=/home/u202310176043/project/demo/runs/preprocess
mkdir -p "$LOG"
cd "$ROOT/examples"

for DATASET in assist2009 assist2017 assist2012; do
  "$PY" -u data_preprocess.py \
    --dataset_name "$DATASET" \
    --min_seq_len 3 \
    --maxlen 200 \
    --kfold 9 \
    --test-ratio 0.1 \
    --no-window \
    > "$LOG/$DATASET.log" 2>&1
done
