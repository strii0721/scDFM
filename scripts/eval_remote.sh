#!/usr/bin/env bash
# K5: evaluate saved checkpoints on the holdout line (K562).
# One GPU per checkpoint, parallel. Usage: bash scripts/eval_remote.sh <run_dir>
set -euo pipefail
PROJ="$HOME/Projects/scDFM"
OUTPUT_ROOT="$PROJ/output"
RUN_DIR="${1:?usage: eval_remote.sh <run_dir>}"
EXP_DIR=$(ls -d "$RUN_DIR"/results/*/ | head -1)
echo "exp dir: $EXP_DIR"

COMMON="--data_name=vcc \
  --max_test_perts=${MAX_PERTS:-20}"

for cp in ${CHECKPOINTS:-0 2000 4000}; do
  gpu=$(($cp / 2000))
  out="$RUN_DIR/eval_ckpt_$cp"
  mkdir -p "$out"
  echo "launching eval of checkpoint $cp on gpu $gpu"
  nohup env CUDA_VISIBLE_DEVICES=$gpu "$PROJ/.venv/bin/python" "$PROJ/scripts/eval_checkpoint.py" \
    --checkpoint_path "$EXP_DIR/iteration_$cp/checkpoint.pt" \
    --result_path "$out" \
    $COMMON > "$RUN_DIR/eval_ckpt_$cp.log" 2>&1 &
done
wait
echo "all evals done"
