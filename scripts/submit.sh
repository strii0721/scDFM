#!/usr/bin/env bash
# VCC 提交：上传 .vcc 到 virtualcellchallenge.org（每 UTC 日限 2 次）。
# 与 train.sh/inference.sh 一致：远程服务器【项目根目录】执行。
#
# 用法:
#   bash scripts/submit.sh                  # 提交 output/inference/prediction.vcc
#   bash scripts/submit.sh <file.vcc>       # 指定文件
# 环境变量:
#   MODEL_NAME="scDFM-K6"  模型名（默认 scDFM）
#   RESUME=1               中断续传（aborted upload 会锁团队槽位，必须 --resume 解除）
#   NO_WAIT=1              不阻塞等评分（默认 --wait）
# 坑: 远程实验室网络上传 GCS 会卡死（2.7GB .vcc 实测中途断）；卡住时把 .vcc rsync 回家用
#     --resume 续传（见 skill vcc-2026 → references/vcc-cli-pipeline.md 跨机续传配方）。
set -euo pipefail

VCC_FILE="${1:-output/inference/prediction.vcc}"
[ -f "$VCC_FILE" ] || { echo "missing: $VCC_FILE (先跑 scripts/gen-vcc.sh)" >&2; exit 1; }

MODEL_NAME="${MODEL_NAME:-scDFM}"

ARGS=()
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)
[ "${NO_WAIT:-0}" != "1" ] && ARGS+=(--wait)

vcc submit "$VCC_FILE" -m "$MODEL_NAME" "${ARGS[@]}"
