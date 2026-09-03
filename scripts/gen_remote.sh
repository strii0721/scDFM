#!/usr/bin/env bash
# K6: generate the VCC submission h5ad from a trained checkpoint on 8 GPUs.
# Usage: bash scripts/gen_remote.sh <run_dir> <checkpoint_iter>
# Env: NUM_SHARDS (default 8), MAX_PAIRS (smoke: generate only N pairs/shard)
set -euo pipefail
PROJ="$HOME/Projects/scDFM"
OUTPUT_ROOT="$PROJ/output"
RUN_DIR="${1:?usage: gen_remote.sh <run_dir> <checkpoint_iter>}"
ITER="${2:?checkpoint iter}"
EXP_DIR=$(ls -d "$RUN_DIR"/results/*/ | head -1)
GEN_DIR="$RUN_DIR/gen_ckpt_$ITER"
mkdir -p "$GEN_DIR/partials"
echo "exp dir: $EXP_DIR"
echo "gen dir: $GEN_DIR"

COMMON="--data_name=vcc \
  --data_path=$OUTPUT_ROOT/data \
  --panel_path=/home/ict2/Projects/vcc-2026/resources/PubData/vcc2026-val-1/pert_counts.csv \
  --controls_dir=/ssd1/PubData/vcc2026-val-1 \
  --n_top_genes=5000 --infer_top_gene=1000 --top_infer_genes=1000 \
  --batch_size=128 --ode_steps=${ODE_STEPS:-12} --split_method=leave_line_out --topk=30 \
  --noise_type=Gaussian --checkpoint_path $EXP_DIR/iteration_$ITER/checkpoint.pt"

EXTRA=""
[ -n "${MAX_PAIRS:-}" ] && EXTRA="--max_pairs $MAX_PAIRS"
EXTRA="$EXTRA --mask_fname=${MASK_FNAME:-mask_fold_0topk_30leave_line_out.pt}"

for s in $(seq 0 $(( ${NUM_SHARDS:-8} - 1 ))); do
  nohup env CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PROJ/.venv/bin/python" "$PROJ/scripts/generate_submission.py" \
    $COMMON --shard_id $s --num_shards ${NUM_SHARDS:-8} --out_dir "$GEN_DIR/partials" $EXTRA \
    > "$GEN_DIR/shard_$s.log" 2>&1 &
done
wait
echo "all shards done"
