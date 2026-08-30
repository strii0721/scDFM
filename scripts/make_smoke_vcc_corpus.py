#!/usr/bin/env python3
"""Synthetic mini VCC-2026 corpus for smoke-testing the 'vcc' data branch.

Mirrors the real pretrain corpus schema:
- raw integer UMI counts, float32 CSR
- obs: target_gene (panel genes + 'non-targeting'), cell_line, crispr_type, study
- var: synthetic genes G0..G{N_GENES-1}
Panel = first N_PANEL genes; knockdown on the target gene in its pert cells.
"""
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import sparse

rng = np.random.default_rng(7)

N_GENES = 3000
N_PANEL = 8
LINES = ["L1", "L2", "L3", "L4"]
N_NTC_PER_LINE = 120
N_CELLS_PER_PERT = 30

genes = [f"G{i}" for i in range(N_GENES)]
panel = genes[:N_PANEL]

# base mean expression rate per gene (log-normal), per line (slight line shift)
base_rate = {}
for line in LINES:
    shift = rng.normal(0, 0.15)
    base_rate[line] = np.clip(rng.lognormal(mean=1.2 + shift, sigma=0.5, size=N_GENES), 0.01, 60)

rows, obs_rows = [], []
def add_cells(line, target, n, kd_factor=1.0):
    rate = base_rate[line].copy()
    if target != "non-targeting":
        rate[genes.index(target)] *= kd_factor  # knockdown target gene
    depth = rng.poisson(lam=9000, size=n)  # per-cell depth
    lam = rate[None, :] * (depth[:, None] / 1000.0)
    x = rng.poisson(lam).astype(np.float32)
    rows.append(x)
    for _ in range(n):
        obs_rows.append((target, line))

for line in LINES:
    add_cells(line, "non-targeting", N_NTC_PER_LINE)
    for g in panel:
        add_cells(line, g, N_CELLS_PER_PERT, kd_factor=0.3)
        if line == "L1" and g == panel[0]:
            # a few CRISPR KO cells to test the filter
            add_cells(line, g, 6, kd_factor=0.0)

X = np.vstack(rows)
n_cells = X.shape[0]
crispr_type = np.full(n_cells, "CRISPRi", dtype=object)
# mark the extra KO cells: last 6 cells of L1/G0 block
ko_idx = []
study = np.empty(n_cells, dtype=object)
obs_df = pd.DataFrame(obs_rows, columns=["target_gene", "cell_line"])
obs_df.index = [f"cell{i}" for i in range(n_cells)]

# assign crispr_type: the 6 extra L1 cells for panel[0] are KO
is_l1_g0 = (obs_df.cell_line == "L1") & (obs_df.target_gene == panel[0])
idx = np.where(is_l1_g0.to_numpy())[0]
obs_df["crispr_type"] = "CRISPRi"
obs_df.loc[obs_df.index[idx[-6:]], "crispr_type"] = "CRISPR KO"
obs_df["study"] = obs_df.cell_line.map({l: f"study_{l}" for l in LINES})

var_df = pd.DataFrame(index=genes)
adata = sc.AnnData(X=sparse.csr_matrix(X), obs=obs_df, var=var_df)
adata.write("/home/lynchpin/Projects/scDFM/data/vcc_corpus_smoke.h5ad")
with open("/home/lynchpin/Projects/scDFM/data/vcc_panel_genes.csv", "w") as f:
    f.write("\n".join(panel) + "\n")
print(f"wrote {adata.shape} -> data/vcc_corpus_smoke.h5ad")
print(adata.obs.groupby(["cell_line", "crispr_type"]).size())
