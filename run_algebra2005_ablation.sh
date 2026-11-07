#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA="$ROOT/data/atomic_fold0/algebra2005"
FORMAL="$ROOT/runs/formal_own_model_algebra2005"
RUN_ROOT="$ROOT/runs/algebra2005_ablation"
GPU_ID=${GPU_ID:-0}
BATCH=${BATCH:-8192}
WORKERS=${WORKERS:-2}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.90}
SEEDS=${SEEDS:-"3407 2026"}

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
test -s "$DATA/students.json"
test -s "$DATA/manifest.json"

run_one() {
  local seed=$1 variant=$2
  local out="$RUN_ROOT/seed_${seed}/$variant"
  local log="$RUN_ROOT/logs/seed${seed}_${variant}.log"
  local extra=() source=""

  case "$variant" in
    no_spectrum)        extra=(--no-spectrum --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_kc_conditioning) extra=(--no-kc-conditioning --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    uniform_fusion)     extra=(--uniform-fusion --training-mode joint --topk 4 --lambda-weak 0.2) ;;
    top4)               extra=(--training-mode joint --topk 4 --lambda-weak 0.2) ;;
    no_attention)       extra=(--disable-expert attention --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_gru)             extra=(--disable-expert gru --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_cnn)             extra=(--disable-expert cnn --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_mamba)           extra=(--disable-expert mamba_official --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    kt_only)            extra=(--training-mode kt_only --topk 2 --lambda-weak 0) ;;
    two_stage)
      source="$RUN_ROOT/seed_${seed}/kt_only/best.pt"
      test -s "$source"
      extra=(--training-mode weak_head --init-from "$source" --topk 2 --lambda-weak 0 --lambda-balance 0)
      ;;
    *) echo "unsupported variant: $variant" >&2; return 2 ;;
  esac

  mkdir -p "$out"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP seed=$seed variant=$variant"
    return 0
  fi
  local resume=()
  if [[ -s "$out/last.pt" ]]; then
    resume=(--resume-from "$out/last.pt")
    echo "[$(date -Is)] RESUME seed=$seed variant=$variant"
  else
    echo "[$(date -Is)] START seed=$seed variant=$variant"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -u train.py \
    --fold-data --fold 0 --bucket-by-length --data "$DATA" --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$BATCH" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-balance 0.01 --weak-update-every 4 \
    --device cuda:0 --test "${extra[@]}" "${resume[@]}" >"$log" 2>&1
  echo "[$(date -Is)] DONE seed=$seed variant=$variant"
}

run_seed() {
  local seed=$1
  # Two-stage is intentionally last because it consumes the KT-only checkpoint.
  for variant in no_spectrum no_kc_conditioning uniform_fusion top4 \
                 no_attention no_gru no_cnn no_mamba kt_only two_stage; do
    run_one "$seed" "$variant"
  done
}

# A single high-throughput process avoids the host/CUDA initialization race
# observed with two concurrent Mamba jobs. Batch 8192 and a 90% VRAM cap use
# the available GPU more efficiently while preserving deterministic run order.
echo "[$(date -Is)] Algebra2005 ablation started in high-throughput sequential mode"
for seed in $SEEDS; do
  run_seed "$seed"
done

"$PY" summarize_algebra2005_ablation.py --all-seeds 3407 2026
echo "[$(date -Is)] ALGEBRA2005 ABLATION FINISHED: $RUN_ROOT/ablation_results_all.csv"
