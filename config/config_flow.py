
from dataclasses import dataclass
import os
from datetime import datetime

VCC_REMOTE_RESOURCE_ROOT = '/ssd1/ict2/Projects/vcc-2026/resources/datasets'
VCC_REMOTE_CORPUS_PATH = os.path.join(
    VCC_REMOTE_RESOURCE_ROOT, 'train_merged', 'train_merged_panel.h5ad'
)
VCC_REMOTE_CONTROLS_DIR = os.path.join(VCC_REMOTE_RESOURCE_ROOT, 'controls')
VCC_REMOTE_PANEL_PATH = os.path.join(VCC_REMOTE_CONTROLS_DIR, 'pert_counts.csv')

# replogle train/test 语料（2026-09-17 起 scDFM 主线切到此组数据）：
# train = K562+Jurkat+HepG2 全量；test = RPE1 独立文件（留系 benchmark 用）。
# 两文件 var 轴一致 = /ssd1/ict2/Projects/vcc-2026/resources/datasets/replogle/gene_names.csv 的 11,919 基因。
REPLOGLE_DATA_DIR = '/ssd1/ict2/Projects/vcc-2026/resources/datasets/replogle'
REPLOGLE_TRAIN_PATH = os.path.join(REPLOGLE_DATA_DIR, 'replogle_k562_jurkat_hepg2.h5ad')
REPLOGLE_TEST_PATH = os.path.join(REPLOGLE_DATA_DIR, 'replogle_rpe1.h5ad')
REPLOGLE_PANEL_PATH = os.path.join(REPLOGLE_DATA_DIR, 'pert_counts.csv')

@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'

    # Flow Matching specific parameters（默认=论文附录 A.4.3 口径）
    batch_size: int = 96          # 论文全局 batch=96；8 卡 DDP 时 train.sh 按 96/GPUS 分摊到每 rank
    # replogle 全轴建模（2026-09-17 用户定案）：缓存保留全部 11,919 基因列，
    # vocab = 11,919 + 4 specials = 11,923。instantiate_model 已透传 ntoken/d_model
    # （此前恒 6000 是死参数，全轴 vocab 会索引越界）。
    ntoken: int = 11923
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
    infer_top_gene: int = 3000
    # 2026-09-19 用户定案（第二次）：训练每步从 common_hvg 清单 − panel 池随机抽
    # 3000（首版=同池抽 1000；中间尝试过全轴−panel 抽 3000 已回退——评估集不变、
    # 训练 6.4× 慢且 benchmark 覆盖天花板不动）。缓存列 = train_pool_path 清单
    # ∩ 语料 var（~29.5GB）。n_top_genes 仅作缓存/vocab 键组分。
    n_top_genes: int = 11919
    checkpoint_path: str = ''
    gamma: float = 0.5             # 论文 MMD λ=0.5
    # replogle 留系（2026-09-17 用户定案）：train 文件（K562/Jurkat/HepG2）全量训练，
    # 无内部留出；测试语料 = 独立文件 test_corpus_path（RPE1），benchmark 专用。
    # 原方案仍可用 --split_method=single_line（heldout_line 系留出）或 single（基因留出）。
    split_method: str = 'whole'
    use_mmd_loss: bool = True      # 论文带 MMD 分布正则（动态多核 RBF）
    fold: int = 0
    use_negative_edge: bool = True # 论文 kNN k=30 带符号相关（signed mask）
    topk: int = 30                 # 论文 k=30

    # VCC-2026 mode (data_name='vcc')
    # data_path = 缓存/共表达图/split 产物根目录。2026-09-19 用户定案：缓存放项目
    # 文件夹下 /ssd1/ict2/Projects/scDFM/tmp（该路径经 /ssd1/ict2/Projects 软链接
    # 实际落在 /ssd2 本地盘，/ssd1 NFS 98% 满不占 NFS；语料数据集走 corpus_path）。
    data_path: str = '/ssd1/ict2/Projects/scDFM/tmp'
    corpus_path: str = REPLOGLE_TRAIN_PATH
    panel_path: str = REPLOGLE_PANEL_PATH
    test_corpus_path: str = REPLOGLE_TEST_PATH  # split_method='whole' 的独立测试语料（benchmark real 侧）
    train_pool_path: str = os.path.join(REPLOGLE_DATA_DIR, 'common_hvg.csv')  # 训练每步采样池（基因清单，表头 gene_name，∩ 语料 var − panel）；空串=整个基因轴 − panel
    line_col: str = 'context'   # replogle train 文件 context=K562/Jurkat/HepG2（obs 无 cell_line 列）
    heldout_line: str = 'RPE1'  # whole 切分下仅作 benchmark 的 context 标签
    crispr_type_col: str = ''   # replogle 文件无 crispr_type 列；留空直接跳过 CRISPRi 过滤
    crispr_type_value: str = 'CRISPRi'  # 仅当 crispr_type_col 非空时使用（data.py 过滤分支）
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
