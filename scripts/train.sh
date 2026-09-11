#!/usr/bin/env bash
# VCC-2026 scDFM 训练。远程服务器【项目根目录】执行，先激活 uv 虚拟环境：
#   source .venv/bin/activate
#
# 用法:
#   bash scripts/train.sh              # 默认 8 卡 DDP（torchrun）
#   GPUS=1 bash scripts/train.sh       # 单卡
#   STEPS=200 bash scripts/train.sh    # 覆盖步数
# 训练超参/数据路径全部走 config/config_flow.py 默认值（VCC 主线已收口），
# 这里只传运行态参数；其他覆盖直接追加 tyro 参数:
#   bash scripts/train.sh --gamma=1.0 --max_test_perts=0
# 产物: output/train/{YYYY-MM-DD_HH-MM}/iteration_N/checkpoint.pt（见 config.make_path，时间戳即实验名）
set -euo pipefail
export PYTHONPATH=.

GPUS="${GPUS:-8}"

# 论文全局 batch=96（附录 A.4.3）；多卡 DDP 时按卡数分摊到每 rank
BATCH_TOTAL="${BATCH_TOTAL:-96}"
RANK_BATCH=$((BATCH_TOTAL / GPUS))

ARGS=(--data_name=vcc --batch_size="$RANK_BATCH")
[ -n "${STEPS:-}" ] && ARGS+=(--steps="$STEPS")
[ -n "${PRINT_EVERY:-}" ] && ARGS+=(--print_every="$PRINT_EVERY")

if [ "$GPUS" -gt 1 ]; then
  # 预处理缓存/共表达 mask/vocab 必须先单进程预构建：8 个 DDP rank 并发写
  # 同一 h5ad 会在 NFS 上撞 h5py 文件锁（实测 BlockingIOError errno 11，
  # 缓存写坏 + 全体退出）。预构建后各 rank 只读。
  python src/script/build_vcc_cache.py "${ARGS[@]}" "$@"
  # DDP：循环内 eval 会 NCCL 死锁，config 默认 do_eval=False，这里再显式关一次
  torchrun --nproc_per_node="$GPUS" src/script/run.py "${ARGS[@]}" --no-do_eval "$@"
else
  python src/script/run.py "${ARGS[@]}" "$@"
fi
