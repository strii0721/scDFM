#!/usr/bin/env python3
"""留一系官方 benchmark（cell-eval2 vcc2026 preset，2026-09-14）。

heldout_line（默认 HCT116）整体留出：
  real = 该系真实细胞 raw counts（扰动全量 + 对照子采样 n_ctrl_cells）
  pred = 模型从该系对照生成：normalize(1e4)+log1p → 单槽条件 ODE → log1p 桥 → counts
然后跑官方三件套：
  1) baseline  -ar real --preset vcc2026            -> baseline_agg.csv（b，mean-response）
  2) run       -ap pred -ar real --anchor           -> agg_results.csv（u）+ anchor_agg.parquet（r 锚点）
  3) score     --user-agg u --baseline-agg b --anchor r -> s=(u-b)/(r-b)，官方榜单同标度
所有 CLI 显式 --set de.backend=pdex（远程无 gpudge GPU 后端，auto 会拒绝静默回退）。

用法（远程项目根，先 source .venv）：
  .venv/bin/python -u src/script/benchmark_line_holdout.py \
      --checkpoint_path output/train/<ts>/iteration_N/checkpoint.pt \
      --split_method=single_line --heldout_line=HCT116 \
      --out_dir output/benchmark/<tag> [--max_perts=20]
注意：与生成/训练同源派生缓存（build_vcc_cache 产物）只读复用；
单对 (pert,400 细胞) ODE 共租实测 ~2.4-3.5 min，全 300 基因是长任务（tmux 跑）。
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tyro
from scipy import sparse

from config.config_flow import FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.script.generate_submission import (
    artifact_paths,
    log1p_bridge_to_counts,
    ode_predict,
    select_modeled_genes,
    stable_seed,
)

if hasattr(ad.settings, "allow_write_nullable_strings"):
    ad.settings.allow_write_nullable_strings = True


@dataclass
class BenchConfig(FlowConfig):
    out_dir: str = ''
    n_ctrl_cells: int = 4000   # real 侧对照细胞数（DE 参考组；官方 context=18400，取子集控时长）
    n_pred_cells: int = 400    # 每扰动预测细胞数（官方 400）
    min_real_cells: int = 20   # real 侧每扰动最少细胞数（低于则跳过该基因）
    max_perts: int = 0         # 冒烟上限（0=全部）
    seed: int = 42
    de_backend: str = 'pdex'   # 无 gpudge 时显式 CPU DE 后端
    # 提交侧同款（generate_submission.GenConfig 亦有此二项）
    top_infer_genes: int = 1000
    ode_steps: int = 100


def _cli_bin() -> str:
    return str(Path(sys.executable).parent / 'cell-eval2')


def _run_cli(args: list[str]) -> None:
    cmd = [_cli_bin()] + args
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_real(cfg: BenchConfig) -> ad.AnnData:
    """heldout_line 真实细胞：对照子采样 + 全部（≥min_real_cells 的）扰动，raw counts。"""
    a = sc.read_h5ad(cfg.corpus_path, backed='r')
    obs = a.obs
    line_mask = (obs[cfg.line_col].astype(str) == cfg.heldout_line).to_numpy()
    tg = obs['target_gene'].astype(str).to_numpy()
    ctl_mask = line_mask & (tg == 'non-targeting')
    pert_mask = line_mask & (tg != 'non-targeting')

    rng = np.random.default_rng(cfg.seed)
    ctl_idx = np.nonzero(ctl_mask)[0]
    ctl_sel = np.sort(rng.choice(ctl_idx, size=min(cfg.n_ctrl_cells, len(ctl_idx)), replace=False))

    perts, counts = np.unique(tg[pert_mask], return_counts=True)
    keep_perts = [p for p, c in zip(perts, counts) if c >= cfg.min_real_cells]
    if cfg.max_perts:
        keep_perts = keep_perts[:cfg.max_perts]
    pert_idx = np.nonzero(pert_mask & np.isin(tg, list(keep_perts)))[0]

    real_mask = np.zeros(a.n_obs, dtype=bool)
    real_mask[ctl_sel] = True
    real_mask[pert_idx] = True
    real = a[real_mask].to_memory().copy()
    real.obs['context'] = cfg.heldout_line
    real.obs['target'] = real.obs['target_gene'].astype(str)
    print(f'real: {real.shape[0]} cells = {len(ctl_sel)} ctl + {len(pert_idx)} pert '
          f'({len(keep_perts)} genes)', flush=True)
    return real


def _norm_log1p(raw: sparse.csr_matrix) -> sparse.csr_matrix:
    """normalize_total(1e4) + log1p，与 data.py / generate_submission 预处理一致。"""
    norm = raw.copy().tocsr()
    totals = np.asarray(norm.sum(axis=1)).ravel()
    norm = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ norm
    norm.data = np.log1p(norm.data)
    return norm


def build_pred(cfg: BenchConfig, vf, gene_ids, vocab: GeneVocab, modeled: list[str],
               real: ad.AnnData, device) -> ad.AnnData:
    """从 heldout_line 对照生成扰动预测：ODE（单槽条件）→ log1p 桥 → counts。"""
    ctl_idx_all = np.nonzero((real.obs['target_gene'].astype(str) == 'non-targeting').to_numpy())[0]
    ctl_raw = real.X[ctl_idx_all].tocsr()          # raw counts 子矩阵
    ctl_norm = _norm_log1p(ctl_raw)                # log1p(CP10k)

    gene_axis_pos = {g: i for i, g in enumerate(real.var_names)}
    # 对照子矩阵的列轴 = 全 18,533 官方轴（real 未做过列过滤）
    modeled_pos_full = np.array([gene_axis_pos[g] for g in modeled], dtype=np.int64)

    perts = sorted(p for p in real.obs['target_gene'].astype(str).unique() if p != 'non-targeting')
    rng = np.random.default_rng(cfg.seed)
    rows, obs_rows = [], []
    for pert in perts:
        src_idx = np.sort(rng.choice(len(ctl_idx_all), size=min(cfg.n_pred_cells, len(ctl_idx_all)),
                                     replace=False))
        src_raw = ctl_raw[src_idx]                        # (n, 18533)
        src_norm = ctl_norm[src_idx]                      # log1p 对照
        depths = np.asarray(src_raw.sum(axis=1)).ravel()

        src_modeled = torch.from_numpy(src_norm[:, modeled_pos_full].toarray()).float().to(device)
        pert_id_b = torch.tensor([vocab.encode(pert)], dtype=torch.long,
                                 device=device).repeat(1, 1)
        pred_modeled = ode_predict(
            vf, gene_ids, src_modeled, pert_id_b, cfg.batch_size, cfg.ode_steps,
            cfg.noise_type, getattr(cfg, 'poisson_alpha', 0.8),
            getattr(cfg, 'poisson_target_sum', 1e4), device,
        ).cpu().numpy()

        counts = log1p_bridge_to_counts(pred_modeled, src_norm.toarray(), depths,
                                        modeled_pos_full, stable_seed(cfg.heldout_line, pert, cfg.seed + 7))
        rows.append(sparse.csr_matrix(counts))
        obs_rows.append(pd.DataFrame({'target_gene': [pert] * counts.shape[0],
                                      'context': [cfg.heldout_line] * counts.shape[0],
                                      'target': [pert] * counts.shape[0]}))
        if (len(rows) % 5) == 0:
            print(f'pred: {len(rows)}/{len(perts)} genes done', flush=True)

    X = sparse.vstack(rows).tocsr()
    obs_df = pd.concat(obs_rows, ignore_index=True)
    pred = ad.AnnData(X=X.astype(np.float32), obs=obs_df,
                      var=pd.DataFrame(index=real.var_names))
    print(f'pred: {X.shape[0]} cells x {X.shape[1]} genes ({len(perts)} genes)', flush=True)
    return pred


def main() -> None:
    cfg = tyro.cli(BenchConfig, description=__doc__)
    assert cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path), 'checkpoint_path required'
    assert cfg.out_dir, '--out_dir required'
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)

    cache, mask_path, vocab_path = artifact_paths(cfg)
    vocab = GeneVocab.from_file(vocab_path)
    modeled = select_modeled_genes(cache, cfg.panel_path, cfg.top_infer_genes, vocab)
    gene_ids = torch.tensor(vocab.encode(modeled), dtype=torch.long, device=device)
    print(f'modeled genes: {len(modeled)}', flush=True)

    vf = instantiate_model(cfg.model_type, ntoken=cfg.ntoken, d_model=cfg.d_model,
                           d_perturbation=cfg.d_model, fusion_method=cfg.fusion_method,
                           perturbation_function=cfg.perturbation_function, mask_path=mask_path)
    ckpt = torch.load(cfg.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    vf = vf.to(device).eval()

    real = build_real(cfg)
    real_path = os.path.join(cfg.out_dir, 'real.h5ad')
    real.write_h5ad(real_path)

    pred = build_pred(cfg, vf, gene_ids, vocab, modeled, real, device)
    pred_path = os.path.join(cfg.out_dir, 'pred.h5ad')
    pred.write_h5ad(pred_path)

    base_flags = ['--preset', 'vcc2026', '--input-type', 'counts',
                  '--pert-col', 'target_gene', '--control', 'non-targeting',
                  '--set', f'de.backend={cfg.de_backend}']
    bdir = os.path.join(cfg.out_dir, 'baseline')
    rdir = os.path.join(cfg.out_dir, 'run')
    _run_cli(['baseline', '-ar', real_path, *base_flags, '-o', bdir])
    _run_cli(['run', '-ap', pred_path, '-ar', real_path, *base_flags, '--anchor', '-o', rdir])

    user_agg = os.path.join(rdir, 'agg_results.csv')
    base_agg = os.path.join(bdir, 'baseline_agg.csv')
    anchor = os.path.join(rdir, 'anchor_agg.parquet')
    for p, what in [(user_agg, 'run agg'), (base_agg, 'baseline agg'), (anchor, 'anchor')]:
        assert os.path.exists(p), f'{what} missing: {p}'
    score_path = os.path.join(cfg.out_dir, 'scores.csv')
    _run_cli(['score', '--user-agg', user_agg, '--baseline-agg', base_agg,
              '--anchor', anchor, '-o', score_path])

    scores = pd.read_csv(score_path)
    print('\n===== vcc2026 scaled scores (s=(u-b)/(r-b), 1 = replicate level) =====', flush=True)
    print(scores.to_string(index=False), flush=True)
    print(f'\nartifacts in {cfg.out_dir}: real.h5ad pred.h5ad baseline/ run/ scores.csv', flush=True)


if __name__ == '__main__':
    main()
