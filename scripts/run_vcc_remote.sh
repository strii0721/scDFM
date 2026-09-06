#!/usr/bin/env bash
# VCC-2026 scDFM training on vcc-worker02 (8x A800)
# Usage: STEPS=200 bash scripts/run_vcc_remote.sh [extra tyro args]
set -euo pipefail
cd "$HOME/Projects/scDFM"
export PYTHONPATH=.

OUTPUT_ROOT="$HOME/Projects/scDFM/output"
mkdir -p "$OUTPUT_ROOT/results"

exec ~/.local/bin/uv run python src/script/run.py \
  --data_name=vcc \
  --steps="${STEPS:-100}" \
  --print_every="${PRINT_EVERY:-50}" \
  --result_path="$OUTPUT_ROOT/results" \
  "$@"
