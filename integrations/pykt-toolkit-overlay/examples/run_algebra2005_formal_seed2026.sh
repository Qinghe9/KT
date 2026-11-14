#!/usr/bin/env bash
# Formal Algebra I 2005--2006 pyKT baseline pipeline for the local RTX 3090.
# Protocol matches the other formal baselines: user split 8:1:1, fold 0,
# configurable seed, maxlen 200, non-window test, and the common WeakHead protocol.
set -uo pipefail

ROOT=${PYKT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PYTHON:-python}
SEED=${SEED:-2026}
OUT=${OUT:-"$ROOT/reproduction/formal_weak_baselines_seed${SEED}/algebra2005"}
STANDARD="$OUT/standard"
CHECKPOINTS="$OUT/checkpoints"
WEAK="$OUT/weakhead"
LOGS="$OUT/logs"
GPU_ID=${GPU_ID:-0}
STANDARD_EPOCHS=${STANDARD_EPOCHS:-200}
WEAK_EPOCHS=${WEAK_EPOCHS:-30}

cd "$ROOT/examples"
mkdir -p "$STANDARD" "$CHECKPOINTS" "$WEAK" "$LOGS"

MODELS=(dkt dkt+ dkvmn sakt saint akt simplekt dtransformer dimkt mockt)

display_name() {
  case "$1" in
    dkt) echo DKT ;; dkt+) echo DKT+ ;; dkvmn) echo DKVMN ;;
    sakt) echo SAKT ;; saint) echo SAINT ;; akt) echo AKT ;;
    simplekt) echo simpleKT ;; dtransformer) echo DTransformer ;;
    dimkt) echo DIMKT ;; mockt) echo MoCKT ;;
  esac
}

# Safe batches for one 24-GiB RTX 3090.  DTransformer and DIMKT also carry
# model-level batch overrides in run_ednet_baselines.py; these values document
# and enforce the intended formal run configuration.
standard_batch() {
  case "$1" in
    akt) echo 64 ;;
    saint) echo 64 ;;
    dkvmn) echo 128 ;;
    dtransformer) echo 32 ;;
    dimkt) echo 64 ;;
    *) echo 128 ;;
  esac
}

weak_batch() {
  case "$1" in
    dtransformer) echo 8 ;;
    saint|akt) echo 32 ;;
    dkvmn|sakt|simplekt|dimkt|mockt) echo 64 ;;
    *) echo 128 ;;
  esac
}

checkpoint_dir() {
  local display=$1
  "$PY" - "$STANDARD/习题推荐基线模型实验记录表_实验记录_Grid.csv" "$display" <<'PY'
import csv,sys
from pathlib import Path
path=Path(sys.argv[1])
if path.exists():
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('model') == sys.argv[2]:
                checkpoint=Path(row['ckpt_path'])
                print(checkpoint.parent if checkpoint.suffix else checkpoint)
                break
PY
}

write_protocol_manifest() {
  "$PY" - "$OUT/protocol.json" "$SEED" "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[3]).resolve()
data=root/'data/algebra2005'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
files=['train_valid_sequences.csv','test_sequences.csv',
       'train_valid_sequences_quelevel.csv','test_sequences_quelevel.csv',
       'train_valid_quelevel.csv','test_quelevel.csv','test_question_sequences.csv']
missing=[name for name in files if not (data/name).is_file()]
if missing: raise SystemExit(f'missing Algebra files: {missing}')
cfg=json.loads((root/'configs/data_config.json').read_text())['algebra2005']
manifest={
  'dataset':'algebra2005','fold':0,'seed':int(sys.argv[2]),'split_seed':1024,
  'test_ratio':0.1,'kfold':9,'min_seq_len':3,'maxlen':200,
  'evaluation':'non-window','atomic_kc':True,
  'weak_protocol':{'history_ratio':0.7,'min_history':16,'min_future':10,
                   'positive_error_rate':0.6,'negative_error_rate':0.4,
                   'recall_ks':[1,3,5,10]},
  'data_config':cfg,
  'sha256':{name:sha(data/name) for name in files},
}
Path(sys.argv[1]).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'status':'ok','protocol':sys.argv[1],'hashes':manifest['sha256']},ensure_ascii=False))
PY
}

standard_done() {
  local display=$1 ckpt
  ckpt=$(checkpoint_dir "$display")
  [[ -n "$ckpt" && -s "$ckpt/config.json" ]]
}

run_standard() {
  local model=$1 display batch
  display=$(display_name "$model")
  batch=$(standard_batch "$model")
  if standard_done "$display"; then
    echo "[$(date -Is)] STANDARD SKIP completed model=$model"
    return 0
  fi
  echo "[$(date -Is)] STANDARD START model=$model batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u run_ednet_baselines.py \
      --models "$model" --dataset-name algebra2005 --fold 0 --seed "$SEED" \
      --batch-size "$batch" --epochs "$STANDARD_EPOCHS" \
      --output-dir "$STANDARD" --save-dir "$CHECKPOINTS" \
      >"$LOGS/${model}_standard.log" 2>&1
}

run_standard_audit() {
  if [[ "${SKIP_STANDARD_AUDIT:-0}" == "1" && \
        -s "$OUT/standard_macro_kcauc.csv" && \
        -s "$OUT/standard_results_all_with_macro_kcauc.csv" ]]; then
    echo "[$(date -Is)] STANDARD RE-EVALUATION SKIP existing complete reports"
    return 0
  fi
  # Refresh on every retry so a previously interrupted partial report can
  # never survive after more standard models have completed.
  cp "$STANDARD/习题推荐基线模型实验记录表_Table_Grid.csv" "$OUT/standard_results_all.csv"
  echo "[$(date -Is)] STANDARD RE-EVALUATION START"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u evaluate_standard_macro_kcauc.py --root "$OUT" \
      >"$LOGS/standard_macro_kcauc.log" 2>&1
  echo "[$(date -Is)] STANDARD RE-EVALUATION DONE"
}

run_weak() {
  local model=$1 display ckpt batch
  if [[ -s "$WEAK/$model/results.csv" && -s "$WEAK/$model/metrics.json" ]]; then
    echo "[$(date -Is)] WEAK SKIP completed model=$model"
    return 0
  fi
  display=$(display_name "$model")
  ckpt=$(checkpoint_dir "$display")
  batch=$(weak_batch "$model")
  if [[ -z "$ckpt" || ! -s "$ckpt/config.json" ]]; then
    echo "[$(date -Is)] missing checkpoint model=$model" >&2
    return 1
  fi
  mkdir -p "$WEAK/$model"
  echo "[$(date -Is)] WEAK START model=$model batch=$batch"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u train_weakkc_baseline.py \
      --checkpoint-dir "$ckpt" --out "$WEAK/$model" \
      --epochs "$WEAK_EPOCHS" --patience 10 --batch-size "$batch" \
      --weak-update-every 4 --lambda-weak 0.2 --lr 0.0001 \
      --workers 1 --seed "$SEED" --device cuda \
      >"$LOGS/${model}_weakhead.log" 2>&1
}

write_protocol_manifest

echo "[$(date -Is)] PHASE 1/3: standard baselines"
for model in "${MODELS[@]}"; do
  if ! run_standard "$model"; then
    echo "[$(date -Is)] STANDARD FAILED model=$model; inspect $LOGS/${model}_standard.log" >&2
    exit 1
  fi
done

echo "[$(date -Is)] PHASE 2/3: prediction export and Macro-KCAUC"
run_standard_audit

echo "[$(date -Is)] PHASE 3/3: common WeakHead"
for model in "${MODELS[@]}"; do
  if ! run_weak "$model"; then
    echo "[$(date -Is)] WEAK FAILED model=$model; inspect $LOGS/${model}_weakhead.log" >&2
    exit 1
  fi
done

"$PY" summarize_algebra2005_baselines.py \
  --root "$OUT" --seed "$SEED" --weak-dir weakhead \
  >"$LOGS/summarize.log" 2>&1

echo "[$(date -Is)] ALGEBRA FORMAL PIPELINE FINISHED output=$OUT"
