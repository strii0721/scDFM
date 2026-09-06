#!/usr/bin/env bash
# VCC 打包：output/inference/prediction.h5ad → prediction.vcc（vcc prep 校验+打包）。
# 与 train.sh/inference.sh 一致：远程服务器【项目根目录】执行。
#
# 用法:
#   bash scripts/gen-vcc.sh               # 正式打包
#   bash scripts/gen-vcc.sh --dry-run     # 只校验不产出
# 环境变量: CONTROLS_DIR（默认官方数据目录）、OUT_VCC、IN_H5AD
# 注意: prep 内存需求 ~31GB（远程 1TB RAM 无压力）；产物 .vcc 约 2.7GB
set -euo pipefail

CONTROLS_DIR="${CONTROLS_DIR:-/ssd1/ict2/Projects/vcc-2026/resources/datasets/controls}"
IN_H5AD="${IN_H5AD:-output/inference/prediction.h5ad}"
OUT_VCC="${OUT_VCC:-output/inference/prediction.vcc}"

[ -f "$IN_H5AD" ] || { echo "missing: $IN_H5AD (先跑 scripts/inference.sh)" >&2; exit 1; }
[ -f "$CONTROLS_DIR/gene_names.csv" ] || { echo "missing: $CONTROLS_DIR/gene_names.csv" >&2; exit 1; }

# -g 要求无表头的基因列表（官方 gene_names.csv 首行是 'gene_name'）
GENE_CSV=$(mktemp /tmp/vcc_gene_names.XXXXXX.csv)
trap 'rm -f "$GENE_CSV"' EXIT
tail -n +2 "$CONTROLS_DIR/gene_names.csv" > "$GENE_CSV"

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

vcc prep -i "$IN_H5AD" -g "$GENE_CSV" --perts "$CONTROLS_DIR/pert_counts.csv" -o "$OUT_VCC" $DRY
echo "done: $OUT_VCC"
