#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u202310176043/project/demo
PY=/home/u202310176043/.conda/envs/ktmamba26/bin/python
DATA_ROOT="$ROOT/data/atomic_fold0"
GPU_ID=${GPU_ID:-0}
WORKERS=${WORKERS:-2}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.47}
SEEDS=${SEEDS:-"3407 2026"}
WAIT_FOR_TMUX=${WAIT_FOR_TMUX:-ablation2026}

cd "$ROOT"

# The suite is safe to enqueue immediately: it waits for the currently running
# second-seed ablation and skips every completed result.
if [[ -n "$WAIT_FOR_TMUX" ]]; then
  while tmux has-session -t "$WAIT_FOR_TMUX" 2>/dev/null; do
    echo "[$(date -Is)] WAIT current suite=$WAIT_FOR_TMUX"
    sleep 60
  done
fi

common_args() {
  local dataset=$1 seed=$2 out=$3 batch=$4
  printf '%s\0' \
    --fold-data --fold 0 --data "$DATA_ROOT/$dataset" --out "$out" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --min-delta 0.001 \
    --scheduler-patience 3 --lr-factor 0.5 --min-lr 0.000001 \
    --batch "$batch" --workers "$WORKERS" --weak-workers "$WORKERS" \
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION" --amp \
    --dim 32 --context 200 --lr 0.003 --seed "$seed" \
    --recall-ks 1 3 5 10 \
    --weak-history-ratio 0.7 --weak-min-history 16 --weak-min-future 10 \
    --weak-positive-threshold 0.6 --weak-negative-threshold 0.4 \
    --lambda-balance 0.01 --weak-update-every 4 --device cuda:0 --test
}

run_one() {
  local seed=$1 dataset=$2 variant=$3
  local root="$ROOT/runs/complete_ablation_seed${seed}"
  local out="$root/$dataset/$variant" log="$root/logs/${dataset}_${variant}.log"
  local batch extra=() source=""
  [[ "$dataset" == assist2009 ]] && batch=${ASSIST2009_BATCH:-2048} || batch=${EDNET_BATCH:-4096}

  case "$variant" in
    no_kc_conditioning) extra=(--no-kc-conditioning --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    uniform_fusion)     extra=(--uniform-fusion --training-mode joint --topk 4 --lambda-weak 0.2) ;;
    no_attention)       extra=(--disable-expert attention --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_gru)             extra=(--disable-expert gru --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_cnn)             extra=(--disable-expert cnn --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    no_mamba)           extra=(--disable-expert mamba_official --training-mode joint --topk 2 --lambda-weak 0.2) ;;
    two_stage)
      source="$ROOT/runs/ablation_small_seed${seed}/$dataset/no_weak/best.pt"
      [[ -s "$source" ]] || { echo "missing KT-only checkpoint: $source" >&2; return 1; }
      extra=(--training-mode weak_head --init-from "$source" --topk 2 --lambda-weak 0 --lambda-balance 0)
      ;;
    *) echo "unsupported variant: $variant" >&2; return 2 ;;
  esac

  mkdir -p "$out" "$root/logs"
  if [[ -s "$out/results.csv" && -s "$out/metrics.json" ]]; then
    echo "[$(date -Is)] SKIP seed=$seed dataset=$dataset variant=$variant"
    return
  fi
  local resume=()
  if [[ -s "$out/last.pt" ]]; then resume=(--resume-from "$out/last.pt"); fi
  mapfile -d '' args < <(common_args "$dataset" "$seed" "$out" "$batch")
  echo "[$(date -Is)] START seed=$seed dataset=$dataset variant=$variant"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -u train.py \
    "${args[@]}" "${extra[@]}" "${resume[@]}" > "$log" 2>&1
  echo "[$(date -Is)] DONE seed=$seed dataset=$dataset variant=$variant"
}

# Two concurrent jobs fill the single 24-GiB GPU. The queues alternate the
# larger EdNet runs so host RAM, disk and GPU pressure stay bounded.
for seed in $SEEDS; do
  (
    for variant in no_kc_conditioning no_attention no_cnn two_stage; do
      run_one "$seed" assist2009 "$variant"
      run_one "$seed" ednet "$variant"
    done
  ) & pid_a=$!
  (
    for variant in uniform_fusion no_gru no_mamba; do
      run_one "$seed" ednet "$variant"
      run_one "$seed" assist2009 "$variant"
    done
  ) & pid_b=$!
  wait "$pid_a"; wait "$pid_b"
  "$PY" summarize_complete_ablation.py --seed "$seed"
done

"$PY" summarize_complete_ablation.py --seed 3407 --all-seeds $SEEDS
echo "[$(date -Is)] COMPLETE ABLATION SUITE FINISHED"
