#!/usr/bin/env bash
# 单卡重启一个 benchmark 分片（用户为腾卡单独 kill 某片后，在此重启）。
# 与 benchmark_run.sh 同一套分片口径：全局片号 GIDX → 该片负责的 perturbations
# 切片 = perts[GIDX*CHUNK : (GIDX+1)*CHUNK]，CHUNK = ceil(N_P / TOTAL_SHARDS)。
#
# 用法:
#   bash scripts/bench_shard.sh <全局片号> <本机GPU号> <heldout_line> <checkpoint.pt> [tag]
#   例: bash scripts/bench_shard.sh 5 5 RPE1 output/train/<ts>/iteration_100000/checkpoint.pt replogle_rpe1
# 环境变量: TOTAL_SHARDS(默认8，须与 driver 启动一致) BATCH_SIZE(默认3)
#   ODE_STEPS(默认100) SPLIT(默认whole)
set -euo pipefail

GIDX="${1:?usage: bench_shard.sh <global_shard_idx> <local_gpu> <heldout_line> <checkpoint.pt> [tag]}"
GPU="${2:?usage: bench_shard.sh <global_shard_idx> <local_gpu> <heldout_line> <checkpoint.pt> [tag]}"
LINE="${3:?}"
CKPT="${4:?}"
TAG="${5:-$(echo "$LINE" | tr '[:upper:]' '[:lower:]')}"
OUT="output/benchmark/$TAG"
TOT_SHARDS="${TOTAL_SHARDS:-8}"
BATCH="${BATCH_SIZE:-3}"
ODE="${ODE_STEPS:-100}"
SPLIT="${SPLIT:-whole}"
PY=".venv/bin/python"
SCRIPT="src/script/benchmark_line_holdout.py"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$OUT/perts.txt" ] || { echo "$OUT/perts.txt missing, run prep first" >&2; exit 1; }
mapfile -t PERTS < "$OUT/perts.txt"
N_P="${#PERTS[@]}"
CHUNK=$(( (N_P + TOT_SHARDS - 1) / TOT_SHARDS ))
START=$((GIDX * CHUNK))
SLICE=$(IFS=,; echo "${PERTS[*]:START:CHUNK}")
[ -n "$SLICE" ] || { echo "shard $GIDX out of range (N_P=$N_P, CHUNK=$CHUNK)" >&2; exit 1; }

tmux kill-session -t "bench-shard$GIDX" 2>/dev/null || true
tmux new-session -d -s "bench-shard$GIDX" \
  "cd '$(pwd)' && env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   '$PY' -u '$SCRIPT' --checkpoint_path '$CKPT' --split_method='$SPLIT' \
   --heldout_line='$LINE' --out_dir '$OUT' --no_eval --reuse_real \
   --batch_size='$BATCH' --ode_steps='$ODE' \
   --perts='$SLICE' --pred_tag='shard$GIDX' \
   > 'logs/bench_${TAG}_shard$GIDX.log' 2>&1"
echo "relaunched shard$GIDX on GPU $GPU ($((CHUNK)) genes: $SLICE)"
