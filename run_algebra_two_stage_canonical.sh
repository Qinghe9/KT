#!/usr/bin/env bash
set -euo pipefail

PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
ROOT=/home/u202310176043/project/demo
DATA="$ROOT/data/atomic_fold0/algebra2005"
WEAK="$ROOT/data/canonical_weakkc_v1/algebra2005"
KT="$ROOT/data/canonical_kt_v1/algebra2005"
OUT="$ROOT/runs/algebra2005_two_stage_canonical"
LOG="$OUT/logs"

mkdir -p "$LOG"
cd "$ROOT"

run_stage() {
  local seed="$1"
  local mode="$2"
  local out="$OUT/seed_${seed}/${mode}"
  local extra=()

  if [[ "$mode" == "kt_only" ]]; then
    extra=(--training-mode kt_only --lambda-weak 0 --lambda-balance 0.01)
  else
    extra=(--training-mode weak_head --init-from "$OUT/seed_${seed}/kt_only/best.pt" --lambda-weak 0 --lambda-balance 0)
  fi

  "$PY" -u train.py \
    --fold-data --fold 0 --bucket-by-length \
    --data "$DATA" --canonical-weakkc "$WEAK" \
    --out "$out" \
    --epochs 100 --patience 10 --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch 4096 --workers 1 --weak-workers 1 \
    --gpu-memory-fraction 0.9 --amp \
    --dim 32 --context 200 --topk 2 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --weak-update-every 4 --device cuda:0 --test \
    "${extra[@]}" \
    2>&1 | tee "$LOG/seed${seed}_${mode}.log"

  "$PY" -u evaluate_canonical_kt.py \
    --manifest-dir "$KT" --data "$DATA" \
    --checkpoint "$out/best.pt" --config "$out/config.json" \
    --out "$out/canonical_kt_metrics.json" \
    --device cuda:0 --batch 4096 \
    2>&1 | tee "$LOG/seed${seed}_${mode}_canonical_kt.log"
}

for seed in 2026 3407; do
  run_stage "$seed" kt_only
  run_stage "$seed" two_stage
done

date -Iseconds > "$OUT/COMPLETE"
