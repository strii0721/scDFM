#!/usr/bin/env bash
# VCC-2026 scDFM training on vcc-worker02, multi-GPU DDP via torchrun.
# Usage: GPUS=8 STEPS=5000 bash scripts/run_vcc_remote_ddp.sh [extra tyro args]
set -euo pipefail
export PYTHONPATH="$HOME/Projects/scDFM"

OUTPUT_ROOT="$HOME/Projects/scDFM/output"
RUN_DIR="$OUTPUT_ROOT/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR"
cd "$RUN_DIR"

BF16_FLAG=""
if [ "${USE_BF16:-False}" = "True" ]; then BF16_FLAG="--use_bf16"; fi
MMD_FLAG=""
if [ "${USE_MMD:-False}" = "True" ]; then MMD_FLAG="--use_mmd_loss"; fi

exec "$HOME/Projects/scDFM/.venv/bin/torchrun" --nproc_per_node="${GPUS:-8}" "$HOME/Projects/scDFM/src/script/run.py" \
  --data_name=vcc \
  --steps="${STEPS:-5000}" \
  --print_every="${PRINT_EVERY:-1000}" \
  --gamma="${GAMMA:-0.5}" \
  $BF16_FLAG \
  $MMD_FLAG \
  --no-do_eval \
  --result_path="$RUN_DIR/results" \
  "$@"
