#!/usr/bin/env bash
# VCC-2026 提交物推理：checkpoint → 8 GPU 分片生成 → 合并 360k×18,533 提交 h5ad。
# 与 train.sh 一致：在远程服务器【项目根目录】执行，全程相对路径。
#
# 用法:
#   bash scripts/inference.sh [checkpoint.pt]   # 缺省自动取 output/train 下最新 checkpoint.pt
# 环境变量: NUM_SHARDS(默认8) MAX_PAIRS(冒烟,只生成每分片N对) ODE_STEPS MASK_FNAME
# 产物: output/inference/partials/*.h5ad → output/inference/prediction.h5ad
# 打包: vcc prep output/inference/prediction.h5ad -g <gene_names.csv> --perts <pert_counts.csv> -o prediction.vcc
set -euo pipefail

PY=".venv/bin/python"
GEN="scripts/generate_submission.py"
OUT_ROOT="output/inference"
PARTIALS="$OUT_ROOT/partials"

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(find output/train -name checkpoint.pt 2>/dev/null | sort | tail -1)
  [ -n "$CKPT" ] || { echo "no checkpoint under output/train; pass one explicitly" >&2; exit 1; }
fi
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
echo "checkpoint: $CKPT"

mkdir -p "$PARTIALS"
rm -f "$PARTIALS"/partial_s*.h5ad

COMMON="--data_name=vcc --batch_size=128 --ode_steps=${ODE_STEPS:-12} \
  --checkpoint_path $CKPT --mask_fname=${MASK_FNAME:-mask_fold_0topk_30leave_line_out.pt}"
EXTRA=""
[ -n "${MAX_PAIRS:-}" ] && EXTRA="--max_pairs $MAX_PAIRS"

for s in $(seq 0 $(( ${NUM_SHARDS:-8} - 1 ))); do
  nohup env CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$GEN" $COMMON --shard_id $s --num_shards ${NUM_SHARDS:-8} --out_dir "$PARTIALS" $EXTRA \
    > "$OUT_ROOT/shard_$s.log" 2>&1 &
done
wait
echo "all shards done"

# ---- merge shards（原 merge_submission.py 逻辑） ----
INFER_PARTIALS="$PARTIALS" INFER_OUT="$OUT_ROOT/prediction.h5ad" "$PY" - <<'PYEOF'
import os
import numpy as np
from scipy import sparse
import anndata as ad

partial_dir = os.environ['INFER_PARTIALS']
out_path = os.environ['INFER_OUT']
parts = sorted(f for f in os.listdir(partial_dir) if f.startswith('partial_s') and f.endswith('.h5ad'))
assert parts, 'no partial h5ad found'
adatas = [ad.read_h5ad(os.path.join(partial_dir, p)) for p in parts]
merged = ad.concat(adatas, join='outer', merge='first')
# enforce var order == first shard's (official gene order)
merged = merged[:, adatas[0].var_names]
merged.obs = merged.obs[['target_gene', 'context']]
merged.obs['target_gene'] = merged.obs['target_gene'].astype(str)
merged.obs['context'] = merged.obs['context'].astype(str)
merged.write_h5ad(out_path)
X = merged.X.tocsr() if sparse.issparse(merged.X) else sparse.csr_matrix(merged.X)
print(f'wrote {out_path}: {merged.shape[0]} x {merged.shape[1]}, {X.nnz} nnz '
      f'({X.nnz/merged.shape[0]:.0f}/cell), max {X.data.max():.0f} UMIs')
print('per (context, target) cells:')
print(merged.obs.groupby(['context', 'target_gene']).size().groupby('context').agg(['min', 'max']))
PYEOF
