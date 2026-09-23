#!/usr/bin/env python3
"""训练侧 DEG 索引（PerturbQA 论文 A.2 口径，2026-09-24 用户定案）。

口径：Log(TP10k+1) 归一化（MWU 是秩检验，单调变换后 p 值不变，raw counts 直接
等价）→ 每细胞系内每扰动 vs 同系 non-targeting（NTC 每系抽 n_ctrl 个，论文 K562*
同款）→ Mann-Whitney U（pdex = cell-eval2 官方 DE 后端同款）→ BH 校正 →
DE 对 = padj < 0.01（严格）/ p < 0.01（论文无重复系口径），方向 = log2FC 符号。

产物（out_dir）：
  deg_long_<line>.parquet   该系 p<0.05 的 (pert, gene, pvalue, padj, log2fc, pct_change)
  deg_sets.csv              跨系并集 padj<0.01：(pert, gene, dir, padj, n_lines, conflict)
  summary.csv               每扰动：n_cells、padj<0.01 DEG 数、p<0.01 DEG 数、是否 panel

用法（远程项目根）:
  .venv/bin/python -u src/script/build_deg_index.py \
      --adata_path /home/ict2/Projects/vcc-2026/resources/datasets/replogle/replogle_k562_jurkat_hepg2.h5ad \
      --out_dir output/deg_index --n_ctrl 5000
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import anndata as ad
import pdex


def _col(df, *keys):
    for c in df.columns:
        for k in keys:
            if k in c.lower():
                return c
    raise KeyError(f'columns {list(df.columns)} lack {keys}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adata_path', required=True)
    ap.add_argument('--out_dir', default='output/deg_index')
    ap.add_argument('--n_ctrl', type=int, default=5000)
    ap.add_argument('--padj_cutoff', type=float, default=0.01)
    ap.add_argument('--p_cutoff', type=float, default=0.01)
    ap.add_argument('--threads', type=int, default=0, help='0=全部核')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    a = ad.read_h5ad(args.adata_path, backed='r')
    tg_all = a.obs['target_gene'].astype(str).to_numpy()
    line_all = a.obs['cell_line_name'].astype(str).to_numpy()
    lines = sorted(set(line_all))
    print(f'[deg] corpus {a.shape}, lines={lines}', flush=True)

    # panel 标记（来自 config 的 pert_counts.csv，仅做标记不剔除）
    panel = set()
    panel_path = '/home/ict2/Projects/vcc-2026/resources/datasets/replogle/pert_counts.csv'
    if os.path.exists(panel_path):
        panel = set(pd.read_csv(panel_path)['target_gene'].astype(str))

    rng = np.random.default_rng(42)
    deg_sets = []  # DataFrame 片段：(pert, gene, dir, padj, line)
    summaries = []
    for line in lines:
        t0 = time.time()
        lm = line_all == line
        cells = np.nonzero(lm)[0]
        tg_line = tg_all[cells]
        ntc_idx = cells[tg_line == 'non-targeting']
        if len(ntc_idx) > args.n_ctrl:
            ntc_idx = np.sort(rng.choice(ntc_idx, size=args.n_ctrl, replace=False))
        pert_cells = cells[tg_line != 'non-targeting']
        sub = a[np.sort(np.concatenate([pert_cells, ntc_idx]))]
        n_groups = len(set(tg_line[tg_line != 'non-targeting']))
        print(f'[{line}] {len(sub)} cells (ntc={len(ntc_idx)}), {n_groups} groups, '
              f'pdex...', flush=True)
        res = pdex.pdex(sub, groupby='target_gene', mode='ref', reference='non-targeting',
                        is_log1p=False, threads=args.threads)
        if hasattr(res, 'to_pandas'):
            res = res.to_pandas()
        print(f'[{line}] pdex done {time.time()-t0:.0f}s, cols={list(res.columns)}', flush=True)
        c_pert = _col(res, 'target', 'group')
        c_gene = _col(res, 'gene')
        c_p = _col(res, 'pvalue', 'p_value')
        c_padj = _col(res, 'padj', 'fdr', 'qvalue', 'q_value')
        c_lfc = _col(res, 'log2')
        keep = res[c_p] < 0.05
        df = res.loc[keep, [c_pert, c_gene, c_p, c_padj, c_lfc]].rename(columns={
            c_pert: 'pert', c_gene: 'gene', c_p: 'pvalue', c_padj: 'padj', c_lfc: 'log2fc'})
        df['line'] = line
        df = df[df['pert'] != 'non-targeting']
        df.to_parquet(os.path.join(args.out_dir, f'deg_long_{line}.parquet'), index=False)
        deg = df[(df['padj'] < args.padj_cutoff) | (df['pvalue'] < args.p_cutoff)].copy()
        deg['dir'] = np.sign(deg['log2fc']).astype(int)
        deg_sets.append(deg[['pert', 'gene', 'dir', 'padj', 'line']])
        n_cells = pd.Series(tg_line[tg_line != 'non-targeting']).value_counts()
        s = df.groupby('pert').agg(n_genes_p005=('pvalue', 'size'),
                                   n_deg_padj=('padj', lambda x: (x < args.padj_cutoff).sum()),
                                   n_deg_p=('pvalue', lambda x: (x < args.p_cutoff).sum()))
        s['line'] = line
        s['n_cells'] = s.index.map(lambda p: n_cells.get(p, 0))
        s['is_panel'] = [p in panel for p in s.index]
        summaries.append(s.reset_index().rename(columns={'index': 'pert'}))
        print(f'[{line}] {len(deg)} DEG rows saved, {time.time()-t0:.0f}s total', flush=True)

    deg_all = pd.concat(deg_sets, ignore_index=True)
    # 跨系并集：dir 取 padj 最小系的符号；多系方向冲突标记
    deg_all = deg_all.sort_values('padj')
    g = deg_all.groupby(['pert', 'gene'], as_index=False).agg(
        padj=('padj', 'min'),
        dir=('dir', lambda x: x.iloc[0]),
        n_lines=('line', 'nunique'),
        lines=('line', lambda x: ','.join(sorted(set(x)))),
        dir_conflict=('dir', lambda x: int(len(set(x)) > 1)),
    )
    g = g[g['padj'] < args.padj_cutoff]
    g.to_csv(os.path.join(args.out_dir, 'deg_sets.csv'), index=False)
    pd.concat(summaries, ignore_index=True).to_csv(os.path.join(args.out_dir, 'summary.csv'),
                                                   index=False)
    print(f'[deg] done: {len(g)} DEG pairs across {g["pert"].nunique()} perturbations '
          f'(padj<{args.padj_cutoff}) -> {args.out_dir}/deg_sets.csv', flush=True)


if __name__ == '__main__':
    main()
