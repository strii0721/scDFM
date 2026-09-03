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
  --data_path=$OUTPUT_ROOT/data \
  --corpus_path=/ssd1/PubData/vcc_val1_pretrain.aligned18533.ctrl400.min20.v2.h5ad \
  --panel_path=/home/ict2/Projects/vcc-2026/resources/PubData/vcc2026-val-1/pert_counts.csv \
  --holdout_line=K562 --line_col=cell_line --crispr_type_col=crispr_type --crispr_type_value=CRISPRi \
  --n_top_genes=5000 --infer_top_gene=1000 --batch_size=48 \
  --split_method=leave_line_out --topk=30 --noise_type=Gaussian \
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
