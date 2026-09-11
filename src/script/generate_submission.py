#!/usr/bin/env python3
"""VCC-2026 submission generator（log1p 单空间，2026-09-11 重构）。

加载训练 ckpt，从官方 A/B/C 对照流式生成 300×3×(400 细胞) 的提交物，
分片并行（SHARD_ID / NUM_SHARDS env），由 scripts/inference.sh 编排合并。

每 (context, perturbation)：
  source = 400 个对照细胞（稳定种子抽样）
  pred_log1p = ODE(x0~noise, cond = source 建模基因 + 扰动基因)
  full_log1p = 建模基因 <- pred、其余基因 <- 对照 log1p(CP10k)
  lam = expm1(clip(full_log1p)) → 按源细胞深度缩放 → Poisson 出整数计数

缓存/词表/共表达图一律【只读】cache/vcc/ 与 src/tokenizer/（与训练同名派生，
缺文件直接报错提示先跑 scripts/train.sh 或 build_vcc_cache.py）；本脚本不写
任何缓存文件，输出仅 partial h5ad 到 --out_dir。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hashlib
import h5py
import numpy as np
import pandas as pd
import torch
import tyro
import torchdiffeq
from scipy import sparse
from dataclasses import dataclass

from config.config_flow import FlowConfig, VCC_REMOTE_CONTROLS_DIR
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import make_lognorm_poisson_noise

ODEDEF_STEPS = 20


@dataclass
class GenConfig(FlowConfig):
    controls_dir: str = VCC_REMOTE_CONTROLS_DIR  # dir with context_{A,B,C}.h5ad + gene_names.csv
    mask_fname: str = ''  # '' = 按训练同款公式派生（cache/vcc/mask_fold_...）
    out_dir: str = ''           # partial h5ad 输出目录
    shard_id: int = 0
    num_shards: int = 1
    batch_size: int = 96
    seed: int = 42
    ode_steps: int = ODEDEF_STEPS
    top_infer_genes: int = 1000  # top-HVG 建模基因数（+ 强制 panel）
    max_pairs: int = 0  # smoke: cap pairs per shard (0 = all)


def corpus_stem(config) -> str:
    return os.path.splitext(os.path.basename(str(config.corpus_path)))[0]


def artifact_paths(config):
    """与 data.py/process_vocab 完全一致的派生规则（只读）。"""
    stem = corpus_stem(config)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo/src
    cache_dir = os.path.join(config.data_path, config.data_name)  # cache/vcc
    cache = os.path.join(cache_dir, f'processed_n{config.n_top_genes}_{stem}.h5ad')
    if config.mask_fname:
        mask = os.path.join(cache_dir, config.mask_fname)
    else:
        mask = os.path.join(cache_dir,
                            f'mask_fold_{config.fold}topk_{config.topk}{config.split_method}_{stem}.pt')
    vocab = os.path.join(src_dir, 'tokenizer',
                         f'{config.data_name}_{config.n_top_genes}_{stem}_highly_vocab.json')
    for p, what in [(cache, 'processed cache'), (mask, 'coexpression mask'), (vocab, 'vocab')]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{what} missing: {p}\n'
                f'（训练/预构建会生成它：先跑 scripts/train.sh 或 '
                f'python src/script/build_vcc_cache.py --data_name=vcc）')
    return cache, mask, vocab


def get_work_pairs(config: GenConfig):
    pairs = []
    for ctx in ['A', 'B', 'C']:
        pert_csv = os.path.join(config.controls_dir, 'pert_counts.csv')
        panel = pd.read_csv(pert_csv)['target_gene'].astype(str).tolist()
        for p in panel:
            pairs.append((ctx, p))
    pairs = pairs[config.shard_id::config.num_shards]
    if config.max_pairs:
        pairs = pairs[:config.max_pairs]
    return pairs


def stable_seed(ctx, pert, base):
    h = hashlib.md5(f'{ctx}:{pert}'.encode()).hexdigest()
    return (base + int(h[:8], 16)) % (2**31)


def read_var_names(f: h5py.File):
    var = f['var']
    if '_index' in var:
        ix = var['_index']
        if 'values' in ix:
            vals = ix['values'][:].astype(str)
            mask = ix['mask'][:] if 'mask' in ix else None
            return [v for v, m in zip(vals, mask)] if mask is not None else list(vals)
        return [x.decode() if isinstance(x, bytes) else str(x) for x in ix[:]]
    if 'gene_name' in var:
        g = var['gene_name']
        if 'categories' in g:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g['categories'][:]]
            return [cats[c] for c in g['codes'][:]]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in g[:]]
    raise KeyError('could not read gene names from processed cache var')


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)
    config = tyro.cli(GenConfig, description=__doc__)
    assert config.checkpoint_path and os.path.exists(config.checkpoint_path)
    assert config.controls_dir and os.path.isdir(config.controls_dir)
    os.makedirs(config.out_dir, exist_ok=True)

    device = torch.device('cuda:0')
    torch.manual_seed(config.seed)

    # 1) 只读缓存/词表/mask（与训练同名派生；缺文件报错，不生成）
    cache, mask_path, vocab_path = artifact_paths(config)
    vocab = GeneVocab.from_file(vocab_path)
    with h5py.File(cache, 'r') as f:
        names = read_var_names(f)
        disp = np.asarray(f['var']['dispersions_norm'][:])
    rank = np.argsort(-disp)
    panel = pd.read_csv(config.panel_path, header=None)[0].astype(str).tolist()
    panel = [g for g in panel if g in set(names)]
    top = [names[i] for i in rank[:config.top_infer_genes]]
    modeled = [g for g in top if g not in set(panel)] + panel
    modeled = [g for g in modeled if g in vocab]
    print(f'modeled genes: {len(modeled)} ({config.top_infer_genes} HVG + panel)', flush=True)

    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    L = len(modeled)

    # 2) model
    vf = instantiate_model(
        config.model_type, ntoken=config.ntoken, d_model=config.d_model,
        d_perturbation=config.d_model, fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function, mask_path=mask_path,
    )
    ckpt = torch.load(config.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    # 3) official axis
    gene_names_csv = os.path.join(config.controls_dir, 'gene_names.csv')
    official_genes = pd.read_csv(gene_names_csv)['gene_name'].astype(str).tolist()
    modeled_idx = np.array([official_genes.index(g) for g in modeled], dtype=np.int64)

    def ode_forward(t, x, source_b, pert_id_b):
        t_ = t.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return vf(gene_ids.repeat(x.shape[0], 1), x, t_, source_b, pert_id_b,
                      gene_ids.repeat(x.shape[0], 1))

    # 4) generate per (context, pert)
    pairs = get_work_pairs(config)
    print(f'shard {config.shard_id}/{config.num_shards}: {len(pairs)} pairs', flush=True)

    out_rows, out_obs = [], []
    for ci, (ctx, pert) in enumerate(pairs):
        adata_path = os.path.join(config.controls_dir, f'context_{ctx}.h5ad')
        # controls loaded once per context and cached on this worker
        if not hasattr(main, '_ctl_cache') or main._ctl_cache[0] != ctx:
            import anndata as ad
            ctl = ad.read_h5ad(adata_path)
            raw = ctl.X.tocsr().astype(np.float32)
            # log1p space：normalize_total(1e4) + log1p，与训练一致
            norm = raw.copy()
            totals = np.asarray(norm.sum(axis=1)).ravel()
            norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
            norm.data = np.log1p(norm.data)
            main._ctl_cache = (ctx, raw, norm)
        _, raw, norm = main._ctl_cache

        rng = np.random.default_rng(stable_seed(ctx, pert, config.seed))
        src_idx = np.sort(rng.choice(raw.shape[0], 400, replace=False))

        src_raw = raw[src_idx]            # (400, 18533) 原始计数：深度 + 非建模基因来源
        src_norm = norm[src_idx]          # (400, 18533) log1p(CP10k) 对照
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_idx].toarray()).float().to(device)
        pert_id_b = torch.tensor(vocab.encode([pert]), dtype=torch.long, device=device).repeat(1, 1)

        preds = []
        with torch.no_grad():
            for s in range(0, 400, config.batch_size):
                src_b = src_modeled[s:s + config.batch_size].contiguous()
                pid_b = pert_id_b.repeat(src_b.shape[0], 1)
                # 噪声源与训练 train_step 一致（Gaussian/Poisson 二选一）
                if config.noise_type == 'Poisson':
                    noise = make_lognorm_poisson_noise(
                        target_log=src_b,
                        alpha=getattr(config, 'poisson_alpha', 0.8),
                        per_cell_L=getattr(config, 'poisson_target_sum', 1e4),
                    )
                else:
                    noise = torch.randn(src_b.shape[0], L, device=device)
                traj = torchdiffeq.odeint(
                    lambda t, x: ode_forward(t, x, src_b, pid_b),
                    noise,
                    torch.linspace(0, 1, config.ode_steps, device=device),
                    atol=1e-4, rtol=1e-4, method='rk4',
                )
                preds.append(torch.clamp(traj[-1], min=0).float())
        pred_modeled = torch.cat(preds, dim=0).cpu().numpy()  # (400, L)

        # 5) full vector + counts（log1p 桥）
        full_log = src_norm.toarray()                  # (400, 18533) 对照 log1p(CP10k)
        full_log[:, modeled_idx] = pred_modeled        # 建模基因 <- 模型
        lam = np.expm1(np.clip(full_log, 0, 60))       # 回 CP10k 线性标度
        scale = depths / np.maximum(lam.sum(axis=1), 1.0)
        lam = lam * scale[:, None]
        counts = np.random.default_rng(stable_seed(ctx, pert, config.seed + 7)).poisson(lam)
        counts = counts.astype(np.int32)

        out_rows.append(sparse.csr_matrix(counts))
        out_obs.append(pd.DataFrame({
            'target_gene': [pert] * 400,
            'context': [ctx] * 400,
        }))
        if (ci + 1) % 20 == 0:
            print(f'[{config.shard_id}] {ci+1}/{len(pairs)} done', flush=True)

    # 6) save partial h5ad
    import anndata as ad
    X = sparse.vstack(out_rows).tocsr()
    obs = pd.concat(out_obs, ignore_index=True)
    var = pd.DataFrame(index=official_genes)
    partial = ad.AnnData(X=X, obs=obs, var=var)
    out_path = os.path.join(config.out_dir, f'partial_s{config.shard_id}.h5ad')
    partial.write_h5ad(out_path)
    print(f'wrote {partial.shape} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
