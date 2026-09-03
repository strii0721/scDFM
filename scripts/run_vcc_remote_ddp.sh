#!/usr/bin/env bash
# VCC-2026 scDFM training on vcc-worker02, multi-GPU DDP via torchrun.
# Usage: GPUS=8 STEPS=5000 bash scripts/run_vcc_remote_ddp.sh [extra tyro args]
set -euo pipefail
export PYTHONPATH="$HOME/Projects/scDFM"

OUTPUT_ROOT="$HOME/Projects/scDFM/output"
RUN_DIR="$OUTPUT_ROOT/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" "$OUTPUT_ROOT/data"
echo "run dir: $RUN_DIR"
cd "$RUN_DIR"

BF16_FLAG=""
if [ "${USE_BF16:-False}" = "True" ]; then BF16_FLAG="--use_bf16"; fi
MMD_FLAG=""
if [ "${USE_MMD:-False}" = "True" ]; then MMD_FLAG="--use_mmd_loss"; fi

exec "$HOME/Projects/scDFM/.venv/bin/torchrun" --nproc_per_node="${GPUS:-8}" "$HOME/Projects/scDFM/src/script/run.py" \
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
  --num_workers="${NUM_WORKERS:-4}" \
  --gamma="${GAMMA:-0.5}" \
  $BF16_FLAG \
  $MMD_FLAG \
  --no-do_eval \
  --split_method="${SPLIT_METHOD:-leave_line_out}" \
  --topk=30 \
  --noise_type=Gaussian \
  --result_path="$RUN_DIR/results" \
  "$@"
