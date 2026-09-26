#!/usr/bin/env python3
"""训练范式二（残差目标）离线常量构建（2026-09-26 用户定案）。

口径：
  每组合 (c=细胞系 context, p=扰动基因)：
    扰动细胞与全量同系 NTC 各自逐细胞 CPM(1e6) → 组内算术均值 →
    r[c,p][g] = log2((x̄_pert+1)/(x̄_ctrl+1))（+1 在 CPM 尺度防零）
  r̄_c = 系内组合均值；r̄_p = 跨系组合均值；ḡ = 全组合均值（每组合一行）
  Res[c,p] = r − r̄_c − r̄_p + ḡ   ← 训练目标 x₁（每组合一个向量，同组合细胞共享）

推理恢复（用户定案）：r̂ = 0 + r̄_p − ḡ + Reŝ（r̄_c(RPE1) 取 0，r̄_p/ḡ 用训练集常量）
单系扰动的 Res = ḡ − r̄_c 为常量（退化，用户接受）。

产物（output/residual_targets/，列对齐训练缓存 11,371 基因序）：
  combos.csv          (line, pert, row) —— row = res_<line>.npy 行号
  res_<line>.npy      float32 (n_perts_line, 11371)
  rbar_p.npy          float32 (n_perts_unique, 11371)；rbar_p_perts.csv 行名
  rbar_c.npy          float32 (n_lines, 11371)；rbar_c_lines.csv 行名
  gbar.npy            float32 (11371,)
  genes_cache.csv     缓存列基因名（对齐序）

运行日志自动落 logs/residual_targets_<ts>/build.log（2026-09-26 目录规范，脚本自建）。

用法（远程项目根）:
  .venv/bin/python -u src/script/build_residual_targets.py \
      --adata_path /home/ict2/Projects/vcc-2026/resources/datasets/replogle/replogle_k562_jurkat_hepg2.h5ad \
      --cache_meta tmp/vcc/processed_n11919_replogle_k562_jurkat_hepg2_all.h5ad.meta.h5ad \
      --out_dir output/residual_targets
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad


def _self_log(task: str) -> None:
    """自建日志目录 logs/<task>_<ts>/build.log 并重定向 stdout/stderr（2026-09-26 目录规范）。"""
    log_dir = os.path.join('logs', f'{task}_{time.strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(log_dir, exist_ok=True)
    f = open(os.path.join(log_dir, 'build.log'), 'a', buffering=1)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    sys.stdout = f
    sys.stderr = f


def cpm_col_mean(X, rows):
    """逐细胞 CPM(1e6) 后组内算术均值 = 列加权均值（精确口径）。"""
    sub = X[rows]
    tot = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
    w = 1e6 / np.maximum(tot, 1.0)
    return np.asarray(sub.T @ w).ravel() / len(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adata_path', required=True)
    ap.add_argument('--cache_meta', required=True)
    ap.add_argument('--out_dir', default='output/residual_targets')
    args = ap.parse_args()
    _self_log('residual_targets')
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    a = ad.read_h5ad(args.adata_path, backed='r')
    tg_all = a.obs['target_gene'].astype(str).to_numpy()
    ctx_all = a.obs['context'].astype(str).to_numpy()
    genes_full = list(a.var_names)
    lines = sorted(set(ctx_all))
    print(f'[res] corpus {a.shape}, lines={lines}', flush=True)

    # 缓存列对齐（11,371 基因序）
    meta = ad.read_h5ad(args.cache_meta)
    cache_genes = list(meta.var_names)
    pos_full = {g: i for i, g in enumerate(genes_full)}
    keep_pos = np.array([pos_full[g] for g in cache_genes], dtype=np.int64)
    assert len(keep_pos) == len(cache_genes), 'cache meta genes not found in corpus var'
    print(f'[res] cache genes {len(cache_genes)}, aligned to corpus var', flush=True)

    # ---- 第一遍：逐系算 r[c,p]（全轴 11,919，每组合一行）
    r_rows = []          # float32 (n_combo, 11919)
    combos = []          # (line, pert)
    ctl_mean = {}        # line -> (11919,)
    for line in lines:
        t1 = time.time()
        lm = ctx_all == line
        cells = np.nonzero(lm)[0]
        lo, hi = cells.min(), cells.max() + 1
        block = a[lo:hi]
        Xb = block.X.tocsr().astype(np.float32)
        tg_b = tg_all[lo:hi]
        ntc_rows = np.nonzero(tg_b == 'non-targeting')[0]
        cm = cpm_col_mean(Xb, ntc_rows)
        ctl_mean[line] = cm
        print(f'[{line}] ctl mean done ({len(ntc_rows)} NTC), {time.time()-t1:.0f}s', flush=True)
        for p in sorted(set(tg_b) - {'non-targeting'}):
            rows = np.nonzero(tg_b == p)[0]
            pm = cpm_col_mean(Xb, rows)
            r = np.log2((pm + 1.0) / (cm + 1.0)).astype(np.float32)
            r_rows.append(r)
            combos.append((line, p))
        print(f'[{line}] {sum(1 for c in combos if c[0] == line)} combos, '
              f'{time.time()-t1:.0f}s', flush=True)
        del block, Xb

    r_mat = np.vstack(r_rows)                      # (n_combo, 11919)
    combos_arr = np.array(combos)
    print(f'[res] r computed {r_mat.shape} in {time.time()-t0:.0f}s', flush=True)

    # ---- 主效应与全局均值（每组合一行等权）
    n = len(combos)
    rbar_c = np.vstack([r_mat[combos_arr[:, 0] == L].mean(axis=0) for L in lines])
    rbar_p = {p: r_mat[combos_arr[:, 1] == p].mean(axis=0) for p in sorted(set(combos_arr[:, 1]))}
    gbar = r_mat.mean(axis=0)
    rbar_p_arr = np.vstack([rbar_p[p] for p in sorted(rbar_p)])
    rbar_p_perts = sorted(rbar_p)
    # Res[c,p] = r − r̄_c − r̄_p + ḡ
    res = r_mat - rbar_c[[lines.index(c) for c in combos_arr[:, 0]]] \
          - np.vstack([rbar_p[p] for p in combos_arr[:, 1]]) + gbar[None, :]
    print(f'[res] Res stats: mean={res.mean():.3e} std={res.std():.4f} '
          f'max_abs={np.abs(res).max():.3f}', flush=True)
    # 单系扰动退化验证：Res == ḡ − r̄_c（逐基因）
    n_lines_per = {p: int((combos_arr[:, 1] == p).sum()) for p in rbar_p_perts}
    single = [p for p, k in n_lines_per.items() if k == 1]
    if single:
        p0 = single[0]
        i0 = np.nonzero(combos_arr[:, 1] == p0)[0][0]
        exp = gbar - rbar_c[lines.index(combos_arr[i0, 0])]
        print(f'[res] single-line check {p0}: max|Res−(ḡ−r̄_c)|='
              f'{np.abs(res[i0] - exp).max():.2e}（应≈0）', flush=True)

    # ---- 对齐缓存列后落盘
    res = res[:, keep_pos].astype(np.float32)
    rbar_p_arr = rbar_p_arr[:, keep_pos].astype(np.float32)
    rbar_c = rbar_c[:, keep_pos].astype(np.float32)
    gbar = gbar[keep_pos].astype(np.float32)
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(combos_arr, columns=['line', 'pert']).to_csv(
        os.path.join(args.out_dir, 'combos.csv'), index=False)
    for L in lines:
        idx = np.nonzero(combos_arr[:, 0] == L)[0]
        np.save(os.path.join(args.out_dir, f'res_{L}.npy'), res[idx])
    np.save(os.path.join(args.out_dir, 'rbar_p.npy'), rbar_p_arr)
    pd.DataFrame({'pert': rbar_p_perts}).to_csv(
        os.path.join(args.out_dir, 'rbar_p_perts.csv'), index=False)
    np.save(os.path.join(args.out_dir, 'rbar_c.npy'), rbar_c)
    pd.DataFrame({'line': lines}).to_csv(os.path.join(args.out_dir, 'rbar_c_lines.csv'),
                                         index=False)
    np.save(os.path.join(args.out_dir, 'gbar.npy'), gbar)
    pd.DataFrame({'gene': cache_genes}).to_csv(os.path.join(args.out_dir, 'genes_cache.csv'),
                                               index=False)
    print(f'[res] saved to {args.out_dir}: {len(combos)} combos, '
          f'{len(rbar_p_perts)} perts, {time.time()-t0:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
