# -*- coding: utf-8 -*-
"""
LWM LLR 预测项目 —— 全局配置
"""
import os

# ================= 系统参数（3GPP 兼容 OFDM，Sionna PUSCH） =================
N_ANT = 8                # 基站（gNB）接收天线数 num_rx
N_SC = 120               # 子载波数（10 RB × 12）—— Sionna PUSCH 网格
N_SYMB = 14              # OFDM 符号数（1 slot）
N_DATA_RE = 1440         # 数据 RE 数（12 数据符号 × 120 子载波，DMRS 符号 2/11 全导频）
SCS_KHZ = 15             # 子载波间隔 15kHz（NR mu=0，Sionna 默认）
FFT_SIZE = 120           # FFT 大小
PILOT_SPACING = 4        # （保留，Sionna 用 DMRS type1 替代）
MAX_LLR = 20.0           # LLR 裁剪上限
CHANNEL_MODEL = "TDL-A"  # 3GPP TR 38.901 TDL-A（Sionna 信道建模）
CARRIER_FREQUENCY = 3.5e9
DELAY_SPREAD = 30e-9

# ================= 调制 =================
MOD_ORDERS = [4, 16, 64, 256]   # QPSK / 16QAM / 64QAM / 256QAM
MAX_BITS = 8                    # 最大 log2M（256QAM）
MOD_ONHOT_DIM = 4

# ================= 数据 =================
SNR_RANGE_DB = (-5.0, 25.0)     # 训练 SNR 范围

# ================= 路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LWM_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), "LWM")
LWM_OFFICIAL_CKPT = os.path.join(LWM_REPO_DIR, "model_weights.pth")

# 训练产物
CKPT_PRETRAIN = os.path.join(WEIGHTS_DIR, "lwm_continued.pt")   # 阶段1: MCM 继续预训练
CKPT_LLR = os.path.join(WEIGHTS_DIR, "lwm_llr.pt")              # 阶段2: LLR 微调
CKPT_LLR_NO_PT = os.path.join(WEIGHTS_DIR, "lwm_llr_no_pretrain.pt")  # 对照: 无继续预训练

# ================= CNN LLR Decoder（参考 NNreceiver） =================
CNN_SEP_CONV = True      # 残差块使用深度可分离卷积（depthwise 3x3 + pointwise 1x1）
CNN_TRANSPOSE = True     # 首层用转置卷积（Conv2DTranspose 风格，padding 保持网格尺寸）
CNN_GROUP_NORM = True    # 残差块内使用 GroupNorm

# ================= 训练超参数（GPU 大规模版） =================
# 阶段 1: MCM 继续预训练
PT_N_SAMPLES = 3000        # 预训练样本数（每样本 14 符号 = 14 个 MCM 序列）
PT_EPOCHS = 30
PT_BATCH = 16              # GPU 显存限制（6GB）
PT_GRAD_ACCUM = 4          # 梯度累积（等效 batch 64）
PT_LR = 1e-5
PT_MASK_RATIO = 0.15
PT_SEED = 0

# 阶段 2: LLR 微调
FT_TRAIN_N = 3000
FT_VAL_N = 300
FT_EPOCHS = 30
FT_BATCH = 16
FT_GRAD_ACCUM = 4
FT_LR = 1e-4              # decoder 学习率
FT_LR_BACKBONE = 1e-6     # LWM 骨干学习率（微调）
FT_FREEZE_BACKBONE = False
FT_SEED = 1

# 数据缓存（Sionna 生成一次，训练多次复用）
CACHE_PT = os.path.join(DATA_DIR, "pusch_pt_train.npz")
CACHE_FT_TRAIN = os.path.join(DATA_DIR, "pusch_ft_train.npz")
CACHE_FT_VAL = os.path.join(DATA_DIR, "pusch_ft_val.npz")

# 评估
EVAL_N = 300              # 评估样本数（6 SNR × 50）
EVAL_SEED = 42
EVAL_NSC_LIST = [120]   # 固定 120 子载波（Sionna PUSCH 10 RB）
EVAL_SNR_LIST = [-5, 0, 5, 10, 15, 20]

# ================= v2 多配置升级（一个模型适配多种系统参数） =================
# 固定系统带宽：BWP（1~10 RB）嵌入 1024-FFT 系统网格（≈10MHz 载波，15.36MHz 采样）
SYS_FFT = 1024                 # 系统 FFT 大小（10MHz 级载波，标准 NR 常用 1024）
SYS_SCS_HZ = 15e3              # 子载波间隔 15kHz
SYS_CP = 72                    # 常规循环前缀（144/2048 × 1024，≈4.7µs）
# 信道场景多样性：TDL-A/B/C/D × 时延 {30,100,300}ns × 多普勒（速度 {0,5,30}m/s，
# @3.5GHz 对应 0/58/350Hz，Sionna TDL 原生 min/max_speed Jakes 采样）
V2_CARRIER_FREQUENCY = 3.5e9
V2_RX_ANTS = [1, 2, 4, 8]                  # 接收天线数
V2_RB_RANGE = (1, 10)                      # 子载波按 RB 分配：1~10 RB（12~120 子载波）
V2_SYMB_RANGE = (3, 14)                    # OFDM 符号数 3~14（3 符号用 mapping type B，Sionna 原生）
V2_DMRS_APS = [0, 1, 2]                    # DMRS 模式 {1} / {1+1} / {1+2}
V2_TDL_MODELS = ["A", "B", "C", "D"]       # 3GPP TR 38.901 信道模型
V2_DELAY_SPREADS = [30e-9, 100e-9, 300e-9] # 时延扩展
V2_MAX_SPEEDS = [0.0, 5.0, 30.0]           # UE 速度 m/s（多普勒频偏）

# v2 训练（大规模版：每配置组合 8 个不同样本，覆盖多配置 + 充分训练）
V2_GROUP_SIZE = 8          # 每个配置组合生成的样本数（不同信道/噪声/比特实现）
V2_PT_N = 2000            # 阶段1 MCM 样本数（250 组合 × 8）
V2_TRAIN_N = 2400         # 阶段2 训练样本数（300 组合 × 8）
V2_VAL_N = 160            # 验证样本数（20 组合 × 8）
V2_PT_EPOCHS = 15
V2_FT_EPOCHS = 40
V2_BATCH = 8
V2_GRAD_ACCUM = 8
V2_LR = 1e-4
V2_LR_BACKBONE = 1e-6
V2_SEED = 7
# 模型配置元数据通道（接收机已知的系统参数，小数据下帮助模型区分配置）：
# [n_rx onehot(4), n_sc/120, n_symb/14, dmrs_ap onehot(3), tdl onehot(4), speed/30]
CFG_DIM = 14
V2_EVAL_PER_SNR = 8      # 评估：每 SNR 样本数（多配置随机采样）
V2_EVAL_SNR_LIST = [-5, 0, 5, 10, 15, 20]
V2_EVAL_SEED = 42

# v2 缓存与权重（大规模版）
CACHE_V2_PT = os.path.join(DATA_DIR, "pusch_v2l_pt.pkl")
CACHE_V2_TRAIN = os.path.join(DATA_DIR, "pusch_v2l_train.pkl")
CACHE_V2_VAL = os.path.join(DATA_DIR, "pusch_v2l_val.pkl")
CKPT_PRETRAIN_V2 = os.path.join(WEIGHTS_DIR, "lwm_continued_v2.pt")
CKPT_LLR_V2 = os.path.join(WEIGHTS_DIR, "lwm_llr_v2.pt")
CKPT_LLR_NO_PT_V2 = os.path.join(WEIGHTS_DIR, "lwm_llr_no_pretrain_v2.pt")

for d in (DATA_DIR, WEIGHTS_DIR):
    os.makedirs(d, exist_ok=True)
