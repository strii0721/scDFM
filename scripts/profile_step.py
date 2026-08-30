#!/usr/bin/env python3
"""Profile: dataset __getitem__ vs model forward+backward, norman smoke config."""
import time, os, sys
sys.path.insert(0, os.getcwd())
import torch
from src.data_process.data import Data, PerturbationDataset
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import process_vocab

DATA_PATH = './data'
config_d = dict(
    data_name='norman', data_path=DATA_PATH, n_top_genes=5000, infer_top_gene=300,
    split_method='additive', fold=0, use_negative_edge=False, topk=30,
    model_type='origin', fusion_method='differential_perceiver', perturbation_function='crisper',
    mask_subsample=0, max_test_perts=2, batch_size=32, d_model=128,
)

dm = Data(DATA_PATH)
dm.load_data('norman')
dm.process_data(n_top_genes=5000, infer_top_gene=300, split_method='additive', fold=0,
                use_negative_edge=False, k=30)
train_sampler, test_sampler, _ = dm.load_flow_data(batch_size=32)
ds = PerturbationDataset(train_sampler, 32)

# --- dataset timing ---
t0 = time.time()
for i in range(20):
    b = ds[i]
dt = (time.time() - t0) / 20
print(f"dataset getitem: {dt*1000:.1f} ms")

# --- model setup ---
mask_path = os.path.join(DATA_PATH, 'norman', 'mask_fold_0topk_30additive.pt')
vf = instantiate_model('origin', ntoken=512, d_model=512, d_perturbation=512,
                       fusion_method='differential_perceiver', perturbation_function='crisper',
                       mask_path=mask_path).cuda()
vocab = GeneVocab.from_file(f'src/tokenizer/norman_5000_highly_vocab.json')
gene_ids = torch.tensor(vocab.encode(list(dm.adata.var_names)), dtype=torch.long).cuda()
B = 32
gene = gene_ids.repeat(B, 1).cuda()
infer_gene = 300
opt = torch.optim.Adam(vf.parameters(), lr=5e-5)

# warmup
src = torch.randn(B, infer_gene).cuda()
tgt = torch.randn(B, infer_gene).cuda()
t = torch.rand(B).cuda()
pid = torch.randint(0, 200, (B, 2)).cuda()  # (B,2) gene token ids
gene_in = gene[:, :infer_gene]
for _ in range(2):
    v = vf(gene_in, src, t, tgt, pid)
    v.sum().backward()
    opt.zero_grad()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    v = vf(gene_in, src, t, tgt, pid)
    v.sum().backward()
    opt.zero_grad()
torch.cuda.synchronize()
print(f"model fwd+bwd (B={B}, genes={infer_gene}): {(time.time()-t0)/5:.2f} s")
print(f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
