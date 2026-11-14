#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}; PYKT_DATA=${PYKT_DATA:?set PYKT_DATA to pyKT data directory}
OUT=${DATA_ROOT:-"$ROOT/data/atomic_fold0"}; mkdir -p "$OUT"

prepare() {
  local name=$1 source=$2 test=$3
  "$PYTHON" "$ROOT/train.py" --prepare --prepare-only --fold-data --fold 0 \
    --source "$PYKT_DATA/$name/$source" --test-source "$PYKT_DATA/$name/$test" \
    --data "$OUT/$name" --users 0 --cap 0 --seed 1024
}
prepare assist2009 train_valid_sequences_quelevel.csv test_sequences_quelevel.csv
prepare assist2012 train_valid_sequences_quelevel.csv test_sequences_quelevel.csv
prepare assist2017 train_valid_sequences_quelevel.csv test_sequences_quelevel.csv
prepare ednet train_valid_sequences_quelevel.csv test_sequences_quelevel.csv
prepare algebra2005 train_valid_sequences_quelevel.csv test_sequences_quelevel.csv

echo "Atomic-KC data written to $OUT"
