#!/usr/bin/env bash
# 留一系/训练内系官方 benchmark：prep -> 8 卡分片 pred -> eval 三件套（baseline/run --anchor/score）。
# 与 train.sh/inference.sh 一致：远程服务器【项目根目录】执行，全程相对路径。
#
# 用法:
#   bash scripts/benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]
#   HCT116 留系:    bash scripts/benchmark_run.sh HCT116  output/train/<ts>/iteration_100000/checkpoint.pt
#   训练内系(HEK293T): bash scripts/benchmark_run.sh HEK293T output/train/<ts>/iteration_100000/checkpoint.pt hek293t_seen
# 环境变量:
#   NUM_SHARDS(默认8) BATCH_SIZE(默认200, 分片 worker) ODE_STEPS(默认100=config)
# 产物: output/benchmark/<tag>/ real.h5ad pred_shard*.h5ad pred.h5ad baseline/ run/ scores.csv
# 日志: logs/bench_<tag>_{prep,shard0..7,eval}.log
set -euo pipefail

PY=".venv/bin/python"
SCRIPT="src/script/benchmark_line_holdout.py"
LINE="${1:?usage: benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]}"
CKPT="${2:?usage: benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]}"
TAG="${3:-$(echo "$LINE" | tr '[:upper:]' '[:lower:]')}"
OUT="output/benchmark/$TAG"
N_SHARDS="${NUM_SHARDS:-8}"
BATCH="${BATCH_SIZE:-200}"
ODE="${ODE_STEPS:-100}"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
mkdir -p "$OUT" "logs"
rm -f "$OUT"/pred_shard*.h5ad
echo "== prep (real.h5ad + perts.txt) =="
"$PY" -u "$SCRIPT" --checkpoint_path "$CKPT" --split_method=single_line \
  --heldout_line="$LINE" --out_dir "$OUT" --no_eval \
  > "logs/bench_${TAG}_prep.log" 2>&1
grep -q PREP_DONE "logs/bench_${TAG}_prep.log" || { echo "prep failed, see logs/bench_${TAG}_prep.log" >&2; exit 1; }

mapfile -t PERTS < "$OUT/perts.txt"
N_P="${#PERTS[@]}"
[ "$N_P" -gt 0 ] || { echo "no perts in $OUT/perts.txt" >&2; exit 1; }
CHUNK=$(( (N_P + N_SHARDS - 1) / N_SHARDS ))
echo "== $N_P perts -> $CHUNK/shard =="

for s in $(seq 0 $((N_SHARDS - 1))); do
  START=$((s * CHUNK))
  SLICE=$(IFS=,; echo "${PERTS[*]:START:CHUNK}")
  [ -n "$SLICE" ] || continue
  nohup env CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$SCRIPT" --checkpoint_path "$CKPT" --split_method=single_line \
    --heldout_line="$LINE" --out_dir "$OUT" --no_eval --reuse_real \
    --batch_size="$BATCH" --ode_steps="$ODE" \
    --perts="$SLICE" --pred_tag="shard$s" \
    > "logs/bench_${TAG}_shard$s.log" 2>&1 &
done
wait
N_GOT=$(ls "$OUT"/pred_shard*.h5ad 2>/dev/null | wc -l)
N_EXP=$((N_P + CHUNK - 1) / CHUNK)
if [ "$N_GOT" -lt "$N_EXP" ]; then
  echo "shard failure: $N_GOT/$N_EXP partials produced (see logs/bench_${TAG}_shard*.log)" >&2
  exit 1
fi
echo "== all shards done ($N_GOT partials), eval trio =="
"$PY" -u "$SCRIPT" --checkpoint_path "$CKPT" --split_method=single_line \
  --heldout_line="$LINE" --out_dir "$OUT" --eval_only \
  > "logs/bench_${TAG}_eval.log" 2>&1
echo "== done: $OUT/scores.csv =="
