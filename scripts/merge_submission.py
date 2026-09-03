#!/usr/bin/env python3
"""Merge partial_*.h5ad shards into the final VCC submission h5ad."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata as ad
import tyro
from dataclasses import dataclass


@dataclass
class MergeConfig:
    partial_dir: str = ''
    out_path: str = ''


def main():
    cfg = tyro.cli(MergeConfig)
    parts = sorted(f for f in os.listdir(cfg.partial_dir) if f.startswith('partial_s') and f.endswith('.h5ad'))
    assert parts, 'no partial h5ad found'
    adatas = [ad.read_h5ad(os.path.join(cfg.partial_dir, p)) for p in parts]
    merged = ad.concat(adatas, join='outer', merge='first')
    # enforce var order == first shard's (official gene order)
    merged = merged[:, adatas[0].var_names]
    # minimal obs
    merged.obs = merged.obs[['target_gene', 'context']]
    merged.obs['target_gene'] = merged.obs['target_gene'].astype(str)
    merged.obs['context'] = merged.obs['context'].astype(str)
    merged.write_h5ad(cfg.out_path)
    n_cells, n_genes = merged.shape
    import numpy as np
    from scipy import sparse
    X = merged.X.tocsr() if sparse.issparse(merged.X) else sparse.csr_matrix(merged.X)
    nz = X.nnz
    print(f'wrote {cfg.out_path}: {n_cells} x {n_genes}, {nz} nnz '
          f'({nz/n_cells:.0f}/cell), max {X.data.max():.0f} UMIs')
    print('per (context, target):')
    print(merged.obs.groupby(['context', 'target_gene']).size().groupby('context').agg(['min', 'max']))


if __name__ == '__main__':
    main()
