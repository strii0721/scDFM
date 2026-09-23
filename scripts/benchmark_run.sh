#!/usr/bin/env bash
# 留一系/训练内系官方 benchmark：prep -> 每片独立 tmux 会话分片 pred -> 轮询齐片 -> eval 三件套
# （baseline/run --anchor/score）-> write_benchmark_report.py 生成 6 指标 raw+scaled 表格 md。
# 与 train.sh/inference.sh 一致：远程服务器【项目根目录】执行，全程相对路径。
#
# 用法:
#   bash scripts/benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]
#   replogle RPE1（whole 切分，真实参考=独立测试文件 test_corpus_path）:
#     bash scripts/benchmark_run.sh RPE1 output/train/<ts>/iteration_100000/checkpoint.pt replogle_rpe1
#   单文件语料留系（原口径，--split_method=single_line）:
#     bash scripts/benchmark_run.sh HCT116 output/train/<ts>/iteration_100000/checkpoint.pt
# 环境变量:
#   NUM_SHARDS   本机片数（默认8；每片占一张本机卡，CUDA_VISIBLE_DEVICES=片内序号）
#   TOTAL_SHARDS 全局片数（默认=NUM_SHARDS；双机 14 卡时=14，chunk 按全局算）
#   SHARD_OFFSET 本机片号起点（默认0；双机第二台=7 → 本机片名 shard7..13）
#   SKIP_PREP=1  跳过 prep（双机第二台设 1，避免两机同写 real.h5ad/perts.txt）
#   SKIP_EVAL=1  跳过 eval（双机第二台设 1；eval 由第一台轮询 14 片齐后跑一次）
#   BATCH_SIZE(默认3; 全轴 L=11,071 下 B=12+ 即越显存墙——fp32 attention (B,2H,L,L)
#     B=4 时 ≈63GB 已贴 80GB 上限，B=3 ≈47GB 留余量)
#   ODE_STEPS(默认100=config)  SPLIT(默认whole，须与训练一致以命中 mask 派生键)
# 每片=独立 tmux 会话 bench-shard<全局片号>（2026-09-23 定案，为单独腾卡）：
#   tmux kill-session -t bench-shard5     # 停第 5 片对应卡上的推理，其余片不受影响
#   事后重启单卡：bash scripts/bench_shard.sh 5 5 RPE1 <ckpt> replogle_rpe1
# 产物: output/benchmark/<tag>/ real.h5ad pred_shard*.h5ad pred.h5ad baseline/ run/ scores.csv
#       output/benchmark/<时间戳>_<tag>.md
# 日志: logs/bench_<tag>_{prep,shard0..13,eval}.log
set -euo pipefail

PY=".venv/bin/python"
SCRIPT="src/script/benchmark_line_holdout.py"
LINE="${1:?usage: benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]}"
CKPT="${2:?usage: benchmark_run.sh <heldout_line> <checkpoint.pt> [tag]}"
TAG="${3:-$(echo "$LINE" | tr '[:upper:]' '[:lower:]')}"
OUT="output/benchmark/$TAG"
N_SHARDS="${NUM_SHARDS:-8}"
TOT_SHARDS="${TOTAL_SHARDS:-$N_SHARDS}"
OFFSET="${SHARD_OFFSET:-0}"
BATCH="${BATCH_SIZE:-3}"
ODE="${ODE_STEPS:-100}"
SPLIT="${SPLIT:-whole}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
mkdir -p "$OUT" "logs"

if [ "$SKIP_PREP" != "1" ]; then
  rm -f "$OUT"/pred_shard*.h5ad
  echo "== prep (real.h5ad + perts.txt) =="
  "$PY" -u "$SCRIPT" --checkpoint_path "$CKPT" --split_method="$SPLIT" \
    --heldout_line="$LINE" --out_dir "$OUT" --no_eval \
    > "logs/bench_${TAG}_prep.log" 2>&1
  grep -q PREP_DONE "logs/bench_${TAG}_prep.log" || { echo "prep failed, see logs/bench_${TAG}_prep.log" >&2; exit 1; }
fi

mapfile -t PERTS < "$OUT/perts.txt"
N_P="${#PERTS[@]}"
[ "$N_P" -gt 0 ] || { echo "no perts in $OUT/perts.txt" >&2; exit 1; }
CHUNK=$(( (N_P + TOT_SHARDS - 1) / TOT_SHARDS ))
echo "== $N_P perts, global $TOT_SHARDS shards -> $CHUNK/shard; this host launches shard$OFFSET..$((OFFSET+N_SHARDS-1)) =="

for s in $(seq 0 $((N_SHARDS - 1))); do
  GIDX=$((OFFSET + s))
  START=$((GIDX * CHUNK))
  SLICE=$(IFS=,; echo "${PERTS[*]:START:CHUNK}")
  [ -n "$SLICE" ] || continue
  tmux kill-session -t "bench-shard$GIDX" 2>/dev/null || true
  tmux new-session -d -s "bench-shard$GIDX" \
    "cd '$(pwd)' && env CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     '$PY' -u '$SCRIPT' --checkpoint_path '$CKPT' --split_method='$SPLIT' \
     --heldout_line='$LINE' --out_dir '$OUT' --no_eval --reuse_real \
     --batch_size='$BATCH' --ode_steps='$ODE' \
     --perts='$SLICE' --pred_tag='shard$GIDX' \
     > 'logs/bench_${TAG}_shard$GIDX.log' 2>&1"
done

echo "== waiting for $TOT_SHARDS global shard outputs =="
while :; do
  N_GOT=$(ls "$OUT"/pred_shard*.h5ad 2>/dev/null | wc -l)
  N_LIVE=$(tmux ls 2>/dev/null | grep -c '^bench-shard' || true)
  if [ "$N_GOT" -ge "$TOT_SHARDS" ]; then
    echo "== all $N_GOT shard outputs present =="
    break
  fi
  if [ "$N_LIVE" -eq 0 ]; then
    echo "no live shard sessions but only $N_GOT/$TOT_SHARDS outputs -> abort (restart missing shards with scripts/bench_shard.sh)" >&2
    exit 1
  fi
  echo "waiting: $N_GOT/$TOT_SHARDS shard outputs ($N_LIVE live sessions)"
  sleep 60
done

if [ "$SKIP_EVAL" = "1" ]; then
  echo "== SKIP_EVAL=1: shards done, eval left to the primary host =="
  exit 0
fi

echo "== eval trio =="
"$PY" -u "$SCRIPT" --checkpoint_path "$CKPT" --split_method="$SPLIT" \
  --heldout_line="$LINE" --out_dir "$OUT" --eval_only \
  > "logs/bench_${TAG}_eval.log" 2>&1
echo "== done: $OUT/scores.csv =="
"$PY" -u src/script/write_benchmark_report.py --out_dir "$OUT" --line "$LINE" \
  >> "logs/bench_${TAG}_eval.log" 2>&1
