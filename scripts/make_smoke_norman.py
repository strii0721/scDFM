#!/usr/bin/env python3
"""Generate a tiny synthetic 'norman'-style h5ad for scDFM smoke testing.

Shape: ~1200 cells x 5200 genes (dense float32, log1p-normalized space).
obs: condition ("ctrl" | "G{i}+ctrl" | "G{i}+G{j}"), control (0/1)
Perturbed genes G0..G47 are in var_names; singles G0..G23, combos G24..G47.
"""
import numpy as np
import scanpy as sc
import pandas as pd

rng = np.random.default_rng(42)

N_GENES = 5200
N_CONTROL = 200
N_PER_SINGLE = 30
N_PER_COMBO = 15

gene_names = [f"G{i}" for i in range(N_GENES)]
single_genes = gene_names[:24]
combo_pairs = [(gene_names[24 + 2 * i], gene_names[24 + 2 * i + 1]) for i in range(12)]

def cell_profile(which):
    """Return a mean vector in log1p space per cell group."""
    base = rng.lognormal(mean=0.0, sigma=0.6, size=N_GENES)  # baseline "expression"
    if which.startswith("pert"):
        # perturbed groups: some genes deviate
        idx = rng.integers(0, N_GENES, size=120)
        base[idx] *= rng.uniform(0.4, 2.5, size=120)
    return np.clip(base, 0.0, 8.0)

rows = []
obs = []
for i in range(N_CONTROL):
    mu = cell_profile("ctrl")
    # ~55% dropout
    mask = rng.random(N_GENES) > 0.55
    x = mu * mask
    rows.append(x)
    obs.append(("ctrl", 1))

for g in single_genes:
    for i in range(N_PER_SINGLE):
        mu = cell_profile("pert")
        mu[gene_names.index(g)] *= 0.35  # knockdown of the target
        mask = rng.random(N_GENES) > 0.55
        rows.append(mu * mask)
        obs.append((f"{g}+ctrl", 0))

for a, b in combo_pairs:
    for i in range(N_PER_COMBO):
        mu = cell_profile("pert")
        mu[gene_names.index(a)] *= 0.35
        mu[gene_names.index(b)] *= 0.35
        mask = rng.random(N_GENES) > 0.55
        rows.append(mu * mask)
        obs.append((f"{a}+{b}", 0))

X = np.asarray(rows, dtype=np.float32)
obs_df = pd.DataFrame(obs, columns=["condition", "control"])
obs_df.index = [f"cell{i}" for i in range(X.shape[0])]
var_df = pd.DataFrame(index=gene_names)

adata = sc.AnnData(X=X, obs=obs_df, var=var_df)
adata.write("/home/lynchpin/Projects/scDFM/data/norman.h5ad")
print(f"wrote {adata.shape} -> data/norman.h5ad")
print("conditions:", adata.obs.condition.value_counts().to_dict())
