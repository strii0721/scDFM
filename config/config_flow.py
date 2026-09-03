
from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'

    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper' 
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 5000
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'differential_perceiver' # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15

    # VCC-2026 mode (data_name='vcc')
    data_path: str = './data'
    corpus_path: str = './data/vcc_corpus.h5ad'
    panel_path: str = './data/vcc_panel_genes.csv'
    holdout_line: str = ''  # leave-one-line-out validation line
    line_col: str = 'cell_line'
    crispr_type_col: str = 'crispr_type'
    crispr_type_value: str = 'CRISPRi'
    mask_subsample: int = 50000  # cells for co-expression graph (0 = all)
    cells_per_target: int = 400
    max_test_perts: int = 20  # cap on perturbations evaluated per checkpoint (0 = all)
    num_workers: int = 8  # DataLoader workers per rank
    use_bf16: bool = False  # bf16 autocast for forward/backward
    do_eval: bool = True  # in-loop eval; MUST be False under multi-GPU DDP (deadlocks NCCL)
    
    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()

    def make_path(self):
        exp_name = '-'.join(['flow', 
                             f'fusion_{self.fusion_method}',
                            f'{self.data_name}', 
                            self.model_type, 
                            self.mode, 
                            f'gamma_{self.gamma}',
                            f'perturbation_function_{self.perturbation_function}',
                            f'lr_{self.lr}', 
                            f'dim_model_{self.d_model}', 
                            f'infer_top_gene_{self.infer_top_gene}',
                            f'split_method_{self.split_method}',
                            f'use_mmd_loss_{self.use_mmd_loss}',
                            f'fold_{self.fold}',
                            f'use_negative_edge_{self.use_negative_edge}',
                            f'topk_{self.topk}',
                            ])
        return os.path.join(self.result_path, exp_name)