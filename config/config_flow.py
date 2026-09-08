
from dataclasses import dataclass
import os

VCC_REMOTE_RESOURCE_ROOT = '/ssd1/ict2/Projects/vcc-2026/resources/datasets'
VCC_REMOTE_CORPUS_PATH = os.path.join(
    VCC_REMOTE_RESOURCE_ROOT, 'combine', 'vcc_val1_pretrain.aligned18533.ctrl400.min20.v2.h5ad'
)
VCC_REMOTE_CONTROLS_DIR = os.path.join(VCC_REMOTE_RESOURCE_ROOT, 'controls')
VCC_REMOTE_PANEL_PATH = os.path.join(VCC_REMOTE_CONTROLS_DIR, 'pert_counts.csv')

@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'

    # Flow Matching specific parameters
    batch_size: int = 48
    ntoken: int = 512
    d_model: int = 512
    lr: float = 5e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "vcc"
    perturbation_function: str = 'crisper' 
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 1000
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = 'output/train'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'differential_perceiver' # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.5
    split_method: str = 'leave_line_out'
    use_mmd_loss: bool = True
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 30

    # VCC-2026 mode (data_name='vcc')
    data_path: str = VCC_REMOTE_RESOURCE_ROOT
    corpus_path: str = VCC_REMOTE_CORPUS_PATH
    panel_path: str = VCC_REMOTE_PANEL_PATH
    # 'cpm'    = CP10k linear: normalize_total(target_sum=1e4), NO log1p. Rows
    #            comparable across cells, values stay on linear count-per-10k scale.
    # 'counts' = train/predict raw UMI counts directly (no normalize_total/log1p);
    # 'log1p'  = legacy log1p(CP10k) space.
    # Cache/mask files are space-suffixed so spaces never share artifacts.
    data_space: str = 'cpm'
    holdout_line: str = 'K562'  # leave-one-line-out validation line
    line_col: str = 'cell_line'
    crispr_type_col: str = 'crispr_type'
    crispr_type_value: str = 'CRISPRi'
    mask_subsample: int = 50000  # cells for co-expression graph (0 = all)
    cells_per_target: int = 400
    max_test_perts: int = 20  # cap on perturbations evaluated per checkpoint (0 = all)
    num_workers: int = 4  # DataLoader workers per rank
    use_bf16: bool = True  # bf16 autocast for forward/backward
    do_eval: bool = False  # in-loop eval; MUST be False under multi-GPU DDP (deadlocks NCCL)
    
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
                            f'space_{getattr(self, "data_space", "log1p")}',
                            ])
        return os.path.join(self.result_path, exp_name)

    @property
    def processed_cache_fname(self) -> str:
        """Per-space processed cache name (data/vcc/<name>.h5ad)."""
        suffix = {'counts': '_counts', 'cpm': '_cpm'}.get(getattr(self, 'data_space', 'log1p'), '')
        return f'processed_n{self.n_top_genes}{suffix}.h5ad'

    @property
    def coexpr_mask_fname(self) -> str:
        """Per-space co-expression mask name (data/vcc/<name>.pt).

        The graph is built from the space's own processed data (pearson on counts
        vs cpm vs log1p differs), so spaces must not share a mask file.
        """
        neg = '_negative_edge' if getattr(self, 'use_negative_edge', False) else ''
        suffix = {'counts': '_counts', 'cpm': '_cpm'}.get(getattr(self, 'data_space', 'log1p'), '')
        return f'mask_fold_{self.fold}topk_{self.topk}{self.split_method}{neg}{suffix}.pt'