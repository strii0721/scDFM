
from dataclasses import dataclass
import os
from datetime import datetime

VCC_REMOTE_RESOURCE_ROOT = '/ssd1/ict2/Projects/vcc-2026/resources/datasets'
VCC_REMOTE_CORPUS_PATH = os.path.join(
    VCC_REMOTE_RESOURCE_ROOT, 'train_merged', 'train_merged_panel.h5ad'
)
VCC_REMOTE_CONTROLS_DIR = os.path.join(VCC_REMOTE_RESOURCE_ROOT, 'controls')
VCC_REMOTE_PANEL_PATH = os.path.join(VCC_REMOTE_CONTROLS_DIR, 'pert_counts.csv')

@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'

    # Flow Matching specific parameters（默认=论文附录 A.4.3 口径）
    batch_size: int = 96          # 论文全局 batch=96；8 卡 DDP 时 train.sh 按 96/GPUS 分摊到每 rank
    ntoken: int = 512
    d_model: int = 512
    lr: float = 5e-5              # 论文 Adam lr=5e-5 余弦衰减
    steps: int = 100000           # 论文 100,000 优化步
    eta_min: float = 1e-6         # 论文衰减下界 ηmin=1e-6
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
    gamma: float = 0.5             # 论文 MMD λ=0.5
    split_method: str = 'single'
    use_mmd_loss: bool = True      # 论文带 MMD 分布正则（动态多核 RBF）
    fold: int = 0
    use_negative_edge: bool = True # 论文 kNN k=30 带符号相关（signed mask）
    topk: int = 30                 # 论文 k=30

    # VCC-2026 mode (data_name='vcc')
    # data_path = 缓存/共表达图/split 产物根目录，相对项目根（每次在项目根运行）；
    # 语料数据集本身走 corpus_path（绝对路径）。勿放 resources/（那是数据集目录）。
    data_path: str = 'cache'
    corpus_path: str = VCC_REMOTE_CORPUS_PATH
    panel_path: str = VCC_REMOTE_PANEL_PATH
    line_col: str = 'cell_line'
    crispr_type_col: str = 'crispr_type'
    crispr_type_value: str = 'CRISPRi'
    mask_subsample: int = 50000  # cells for co-expression graph (0 = all)
    max_test_perts: int = 20  # cap on perturbations evaluated per checkpoint (0 = all)
    num_workers: int = 4  # DataLoader workers per rank
    use_bf16: bool = True  # bf16 autocast for forward/backward
    do_eval: bool = False  # in-loop eval; MUST be False under multi-GPU DDP (deadlocks NCCL)
    # same-line pairing (TrainSampler): a (line, gene) pool must have at least this
    # many cells for the line to be eligible; measured 2026-09-11 on the merged
    # corpus (iscr12g+xatlas+replogle): >=20 keeps ~2,011 eligible combos
    min_tgt_cells: int = 20

    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()

    def make_path(self):
        # timestamp IS the experiment name: output/train/{YYYY-MM-DD_HH-MM}/
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
        return os.path.join(self.result_path, ts)
