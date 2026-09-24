#!/usr/bin/env python3
"""训练侧 DEG 索引（PerturbQA 论文 A.2 口径，2026-09-24 用户定案）。

口径：Log(TP10k+1) 归一化（MWU 是秩检验，单调变换后 p 值不变，raw counts 直接
等价）→ 每细胞系内每扰动 vs 同系 non-targeting（NTC 每系抽 n_ctrl 个，论文 K562*
同款）→ Mann-Whitney U（numba_mwu/pdex = cell-eval2 官方 DE 后端同款）→ BH 校正 →
DE 对 = padj < 0.01（严格）/ p < 0.01（论文无重复系口径），方向 = log2FC 符号。

并行：pdex.pdex 的 group 循环是串行的（2026-09-24 实测单核）——本脚本用
multiprocessing fork（COW 共享系内 csr 矩阵，ref 亦 COW）+ 每 worker 串行 mwu
（numba 线程池置 1 防过订阅），64 worker 下全量 ~10-20 分钟。

产物（out_dir）：
  deg_long_<line>.parquet   该系 p<0.05 的 (pert, gene, pvalue, padj, log2fc, line)
  deg_sets.csv              跨系并集 padj<0.01：(pert, gene, dir, padj, n_lines, lines, conflict)
  summary.csv               每扰动：n_cells、padj<0.01 DEG 数、p<0.01 DEG 数、是否 panel

用法（远程项目根）:
  .venv/bin/python -u src/script/build_deg_index.py \
      --adata_path /home/ict2/Projects/vcc-2026/resources/datasets/replogle/replogle_k562_jurkat_hepg2.h5ad \
      --out_dir output/deg_index --n_ctrl 5000 --n_workers 64
"""
import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import anndata as ad

_SHARED_X = None     # 系内 csr（fork COW）
_REF_X = None        # 参考矩阵（fork COW）
_REF_MEAN = None     # 参考逐基因均值（count 空间）


def _one_group(rows):
    import pdex as _p
    _p.set_numba_threadpool(1)  # 防 numba prange 与进程池过订阅
    x = _SHARED_X[rows]
    pv = np.asarray(_p.mwu(x, _REF_X).pvalue).clip(0, 1)
    x_mean = np.asarray(x.mean(axis=0)).ravel()
    lfc = np.asarray(_p.log2_fold_change(x_mean, _REF_MEAN, epsilon=1.0)).ravel()
    return pv, lfc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adata_path', required=True)
    ap.add_argument('--out_dir', default='output/deg_index')
    ap.add_argument('--n_ctrl', type=int, default=5000)
    ap.add_argument('--padj_cutoff', type=float, default=0.01)
    ap.add_argument('--p_cutoff', type=float, default=0.01)
    ap.add_argument('--n_workers', type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    a = ad.read_h5ad(args.adata_path, backed='r')
    tg_all = a.obs['target_gene'].astype(str).to_numpy()
    line_all = a.obs['cell_line_name'].astype(str).to_numpy()
    genes = np.asarray(a.var_names)
    lines = sorted(set(line_all))
    print(f'[deg] corpus {a.shape}, lines={lines}, workers={args.n_workers}', flush=True)

    panel = set()
    panel_path = '/home/ict2/Projects/vcc-2026/resources/datasets/replogle/pert_counts.csv'
    if os.path.exists(panel_path):
        panel = set(pd.read_csv(panel_path)['target_gene'].astype(str))

    rng = np.random.default_rng(42)
    deg_sets, summaries = [], []
    ctx = mp.get_context('fork')
    for line in lines:
        t0 = time.time()
        lm = line_all == line
        cells = np.nonzero(lm)[0]
        tg_line = tg_all[cells]
        ntc_idx = cells[tg_line == 'non-targeting']
        if len(ntc_idx) > args.n_ctrl:
            ntc_idx = np.sort(rng.choice(ntc_idx, size=args.n_ctrl, replace=False))
        pert_cells = cells[tg_line != 'non-targeting']
        keep_idx = np.sort(np.concatenate([pert_cells, ntc_idx]))
        sub = a[keep_idx]
        tg_sub = tg_all[keep_idx]
        X = sub.X.tocsr().astype(np.float32)
        ref_pos = np.searchsorted(keep_idx, ntc_idx)
        ref_X = X[ref_pos]
        ref_mean = np.asarray(ref_X.mean(axis=0)).ravel()
        n_cells = pd.Series(tg_sub[tg_sub != 'non-targeting']).value_counts()
        groups = [(g, np.nonzero(tg_sub == g)[0])
                  for g in sorted(set(tg_sub) - {'non-targeting'})]
        print(f'[{line}] {len(sub)} cells (ntc={len(ntc_idx)}), {len(groups)} groups', flush=True)

        global _SHARED_X, _REF_X, _REF_MEAN
        _SHARED_X, _REF_X, _REF_MEAN = X, ref_X, ref_mean
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
            futs = [pool.submit(_one_group, rows) for _, rows in groups]
            for i, f in enumerate(futs):
                f.result()
                if (i + 1) % 500 == 0 or i + 1 == len(futs):
                    print(f'[{line}] {i+1}/{len(groups)} groups done, '
                          f'{time.time()-t0:.0f}s', flush=True)
            results = [f.result() for f in futs]

        pv_mat = np.vstack([r[0] for r in results])
        lfc_mat = np.vstack([r[1] for r in results])
        padj = pdex.false_discovery_control(pv_mat, axis=1, method='bh')

        keep = pv_mat < 0.05
        ri, ci = np.nonzero(keep)
        df = pd.DataFrame({
            'pert': [groups[i][0] for i in ri],
            'gene': genes[ci],
            'pvalue': pv_mat[keep],
            'padj': padj[keep],
            'log2fc': lfc_mat[keep],
            'line': line,
        })
        df.to_parquet(os.path.join(args.out_dir, f'deg_long_{line}.parquet'), index=False)
        deg = df[(df['padj'] < args.padj_cutoff) | (df['pvalue'] < args.p_cutoff)].copy()
        deg['dir'] = np.sign(deg['log2fc']).astype(int)
        deg_sets.append(deg[['pert', 'gene', 'dir', 'padj', 'line']])
        s = df.groupby('pert').agg(n_genes_p005=('pvalue', 'size'),
                                   n_deg_padj=('padj', lambda x: (x < args.padj_cutoff).sum()),
                                   n_deg_p=('pvalue', lambda x: (x < args.p_cutoff).sum()))
        s['line'] = line
        s['n_cells'] = s.index.map(lambda p: int(n_cells.get(p, 0)))
        s['is_panel'] = [p in panel for p in s.index]
        summaries.append(s.reset_index().rename(columns={'index': 'pert'}))
        print(f'[{line}] {len(deg)} DEG rows saved, {time.time()-t0:.0f}s total', flush=True)
        del _SHARED_X, _REF_X, _REF_MEAN, X, ref_X, sub

    deg_all = pd.concat(deg_sets, ignore_index=True).sort_values('padj')
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
    import pdex  # noqa: F401  （BH 校正 + worker import 预热）
    main()
