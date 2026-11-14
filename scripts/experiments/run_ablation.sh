#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}; DATA_ROOT=${DATA_ROOT:-"$ROOT/data/atomic_fold0"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/runs/ablation"}; GPU_IDS=${GPU_IDS:-0,1}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
datasets=(assist2009 ednet); seeds=(3407 2026)
variants=(full no_spectrum no_kc uniform top4 no_attention no_gru no_cnn no_mamba)

extra_for() {
  case "$1" in
    full) :;; no_spectrum) echo --no-spectrum;; no_kc) echo --no-kc-conditioning;;
    uniform) echo --uniform-fusion;; top4) echo --topk 4;;
    no_attention) echo --disable-expert attention;; no_gru) echo --disable-expert gru;;
    no_cnn) echo --disable-expert cnn;; no_mamba) echo --disable-expert mamba_official;;
  esac
}

run_one() {
  local gpu=$1 d=$2 s=$3 v=$4 out="$RUN_ROOT/seed_${s}/${d}/${v}"
  [[ -s "$out/metrics.json" ]] && return
  mkdir -p "$out" "$RUN_ROOT/logs"
  read -r -a extra <<< "$(extra_for "$v")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$ROOT/train.py" \
    --fold-data --fold 0 --data "$DATA_ROOT/$d" --out "$out" --epochs 100 \
    --patience 10 --min-delta 0.001 --scheduler-patience 3 --lr-factor 0.5 \
    --min-lr 0.000001 --batch 2048 --workers 2 --weak-workers 2 --bucket-by-length \
    --gpu-memory-fraction 0.95 --amp --dim 32 --context 200 --topk 2 --lr 0.003 \
    --seed "$s" --recall-ks 1 3 5 10 --weak-history-ratio 0.7 \
    --weak-min-history 16 --weak-min-future 10 --weak-positive-threshold 0.6 \
    --weak-negative-threshold 0.4 --lambda-weak 0.2 --lambda-balance 0.01 \
    --weak-update-every 4 --device cuda:0 --test "${extra[@]}" \
    >"$RUN_ROOT/logs/${d}_${v}_seed${s}.log" 2>&1
}

tasks=(); for s in "${seeds[@]}"; do for d in "${datasets[@]}"; do for v in "${variants[@]}"; do tasks+=("$d:$s:$v"); done; done; done
pids=(); for i in "${!GPUS[@]}"; do ( for j in "${!tasks[@]}"; do ((j % ${#GPUS[@]} == i)) || continue; IFS=: read -r d s v <<<"${tasks[$j]}"; run_one "${GPUS[$i]}" "$d" "$s" "$v"; done ) & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p"; done
