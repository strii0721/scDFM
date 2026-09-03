#!/usr/bin/env bash
# VCC-2026 scDFM training on vcc-worker02 (8x A800)
# Usage: STEPS=200 bash scripts/run_vcc_remote.sh [extra tyro args]
set -euo pipefail
cd "$HOME/Projects/scDFM"
export PYTHONPATH=.

OUTPUT_ROOT="$HOME/Projects/scDFM/output"
mkdir -p "$OUTPUT_ROOT/data" "$OUTPUT_ROOT/results"

exec ~/.local/bin/uv run python src/script/run.py \
  --data_name=vcc \
  --data_path="$OUTPUT_ROOT/data" \
  --corpus_path=/ssd1/PubData/vcc_val1_pretrain.aligned18533.ctrl400.min20.v2.h5ad \
  --panel_path="/home/ict2/Projects/vcc-2026/resources/PubData/vcc2026-val-1/pert_counts.csv" \
  --holdout_line="${HOLDOUT_LINE:-K562}" \
  --line_col=cell_line \
  --crispr_type_col=crispr_type \
  --crispr_type_value=CRISPRi \
  --n_top_genes=5000 \
  --infer_top_gene=1000 \
  --batch_size="${BATCH_SIZE:-48}" \
  --lr=5e-5 \
  --steps="${STEPS:-100}" \
  --print_every="${PRINT_EVERY:-50}" \
  --max_test_perts="${MAX_TEST_PERTS:-20}" \
  --split_method=leave_line_out \
  --topk=30 \
  --noise_type=Gaussian \
  --result_path="$OUTPUT_ROOT/results" \
  "$@"
