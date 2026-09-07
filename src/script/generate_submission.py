#!/usr/bin/env python3
"""VCC-2026 submission generator (K6).

Loads a trained scDFM checkpoint, flow-generates perturbed cells for the
official A/B/C controls, converts to raw integer counts, and writes partial
h5ad slices. Sharded across GPUs via SHARD_ID / NUM_SHARDS env vars (each
shard takes interleaved (context, pert) pairs). Orchestrated + merged by
scripts/inference.sh.

Pipeline per (context, perturbation):
  source = 400 control cells (fixed seed)
  data_space='counts': pred = ODE(solve, x0 ~ Gaussian noise, cond = pert gene) on
      modeled genes; full = source raw counts with modeled genes overwritten;
      counts ~ Poisson(clip(pred, 0)) -> int32
  data_space='log1p':  pred_log1p = ODE(...); full_log1p = modeled <- pred, other <-
      source normalized values; lambda = expm1(full_log1p); scale so E[total] =
      source cell's raw depth; counts ~ Poisson(lambda) -> int32, CSR
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

MASK_FNAME_DEFAULT = 'mask_fold_0topk_30leave_line_out.pt'
ODEDEF_STEPS = 20


@dataclass
class GenConfig(FlowConfig):
    controls_dir: str = VCC_REMOTE_CONTROLS_DIR  # dir with context_{A,B,C}.h5ad + gene_names.csv
    mask_fname: str = MASK_FNAME_DEFAULT  # co-expression mask (must match training run)
    out_dir: str = ''           # partial h5ad output dir
    shard_id: int = 0
    num_shards: int = 1
    batch_size: int = 96
    seed: int = 42
    ode_steps: int = ODEDEF_STEPS
    top_infer_genes: int = 1000  # top-HVG genes to model (plus panel)
    max_pairs: int = 0  # smoke: cap pairs per shard (0 = all)


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


def main():
    config = tyro.cli(GenConfig, description=__doc__)
    assert config.checkpoint_path and os.path.exists(config.checkpoint_path)
    assert config.controls_dir and os.path.isdir(config.controls_dir)
    os.makedirs(config.out_dir, exist_ok=True)

    device = torch.device('cuda:0')
    torch.manual_seed(config.seed)

    # 1) vocab + modeled gene set (top-HVG by dispersions_norm + panel forced)
    vocab = GeneVocab.from_file(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     'src', 'tokenizer', f'vcc_{config.n_top_genes}_highly_vocab.json'))
    cache = os.path.join(config.data_path, config.data_name, config.processed_cache_fname)
    with h5py.File(cache, 'r') as f:
        var = f['var']
        # var index: nullable-string-array group (values+mask) or plain dataset
        names = None
        if '_index' in var:
            ix = var['_index']
            if 'values' in ix:
                vals = ix['values'][:].astype(str)
                mask = ix['mask'][:] if 'mask' in ix else None
                names = [v for v, m in zip(vals, mask)] if mask is not None else list(vals)
            else:
                names = [x.decode() for x in ix[:]]
        if names is None and 'gene_name' in var:
            g = var['gene_name']
            if 'categories' in g:
                cats = [c.decode() for c in g['categories'][:]]
                names = [cats[c] for c in g['codes'][:]]
            else:
                names = [x.decode() for x in g[:]]
        assert names, 'could not read gene names from processed cache'
        disp = np.asarray(var['dispersions_norm'][:])
        rank = np.argsort(-disp)
        panel = pd.read_csv(config.panel_path, header=None)[0].astype(str).tolist()
        top = [names[i] for i in rank[:config.top_infer_genes]]
        modeled = [g for g in top if g not in set(panel)] + panel
        modeled = [g for g in modeled if g in vocab]
    print(f'modeled genes: {len(modeled)} ({config.top_infer_genes} HVG + panel)', flush=True)

    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    L = len(modeled)

    # 2) model
    mask_path = os.path.join(config.data_path, config.data_name, config.mask_fname)
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
            if config.data_space == 'counts':
                # counts space: conditioning source + non-modeled genes use raw counts
                norm = raw
            else:
                # normalize on full axis, same convention as training
                from scanpy.preprocessing import normalize_total
                norm = raw.copy()
                totals = np.asarray(norm.sum(axis=1)).ravel()
                norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
                norm.data = np.log1p(norm.data)
            main._ctl_cache = (ctx, raw, norm)
        _, raw, norm = main._ctl_cache

        rng = np.random.default_rng(stable_seed(ctx, pert, config.seed))
        src_idx = rng.choice(raw.shape[0], 400, replace=False)
        src_idx = np.sort(src_idx)

        src_raw = raw[src_idx]            # (400, 18533) depth + non-modeled source
        src_norm = norm[src_idx]          # (400, 18533) normalized source
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_idx].toarray()).float().to(device)
        pert_id = torch.tensor(vocab.encode([pert]), dtype=torch.long, device=device)
        pert_id_b = pert_id.repeat(1, 1)  # (1,1)

        preds = []
        with torch.no_grad():
            for s in range(0, 400, config.batch_size):
                src_b = src_modeled[s:s + config.batch_size].contiguous()
                pid_b = pert_id_b.repeat(src_b.shape[0], 1)
                noise = torch.randn(src_b.shape[0], L, device=device)
                traj = torchdiffeq.odeint(
                    lambda t, x: ode_forward(t, x, src_b, pid_b),
                    noise,
                    torch.linspace(0, 1, config.ode_steps, device=device),
                    atol=1e-4, rtol=1e-4, method='rk4',
                )
                preds.append(torch.clamp(traj[-1], min=0).float())
        pred_modeled = torch.cat(preds, dim=0).cpu().numpy()  # (400, L)

        # 5) full vector + counts
        if config.data_space == 'counts':
            # counts space: modeled genes <- predicted counts, non-modeled genes
            # keep control raw counts; Poisson noise on top; no expm1, no rescale
            full = src_norm.toarray()                      # (400, 18533) copy of control counts
            full[:, modeled_idx] = pred_modeled            # modeled genes <- model
            lam = np.clip(full, 0.0, None)
            counts = np.random.default_rng(stable_seed(ctx, pert, config.seed + 7)).poisson(lam)
        else:
            full_log = src_norm.toarray()                  # (400, 18533) copy of control
            full_log[:, modeled_idx] = pred_modeled        # modeled genes <- model
            lam = np.expm1(np.clip(full_log, 0, 60))       # CP10k-space
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
