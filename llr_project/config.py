# -*- coding: utf-8 -*-
"""
LWM LLR 预测项目 —— 全局配置（多配置自适应版）
"""
import os

# ================= 系统参数（固定带宽 + BWP 多配置） =================
SYS_FFT = 1024                 # 系统 FFT 大小（≈10MHz 载波，标准 NR 常用 1024）
SYS_SCS_HZ = 15e3              # 子载波间隔 15kHz
SYS_CP = 72                    # 常规循环前缀（144/2048 × 1024，≈4.7µs）
MAX_LLR = 20.0                 # LLR 裁剪上限

# ================= 调制 =================
MOD_ORDERS = [4, 16, 64, 256]   # QPSK / 16QAM / 64QAM / 256QAM
MAX_BITS = 8                    # 最大 log2M（256QAM）
MOD_ONHOT_DIM = 4

# ================= 多配置空间（一个模型适配所有系统参数） =================
CARRIER_FREQUENCY = 3.5e9
RX_ANTS = [1, 2, 4, 8]                  # 接收天线数
RB_RANGE = (1, 10)                      # 子载波按 RB 分配：1~10 RB（12~120 子载波）
SYMB_RANGE = (3, 14)                    # OFDM 符号数 3~14（3 符号用 mapping type B，Sionna 原生）
DMRS_APS = [0, 1, 2]                    # DMRS 模式 {1} / {1+1} / {1+2}
TDL_MODELS = ["A", "B", "C", "D"]       # 3GPP TR 38.901 信道模型
DELAY_SPREADS = [30e-9, 100e-9, 300e-9] # 时延扩展
MAX_SPEEDS = [0.0, 5.0, 30.0]           # UE 速度 m/s（多普勒 0/58/350Hz @3.5GHz）

# ================= 模型 =================
# CNN Decoder（参考 NNreceiver）
CNN_SEP_CONV = True      # 残差块使用深度可分离卷积（depthwise 3x3 + pointwise 1x1）
CNN_TRANSPOSE = True     # 首层用转置卷积（Conv2DTranspose 风格，padding 保持网格尺寸）
CNN_GROUP_NORM = True    # 残差块内使用 GroupNorm
# 配置元数据通道（接收机已知的系统参数，帮助模型区分配置）：
# [n_rx onehot(4), n_sc/120, n_symb/14, dmrs_ap onehot(3), tdl onehot(4), speed/30]
CFG_DIM = 14

# ================= 路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LWM_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), "LWM")
LWM_OFFICIAL_CKPT = os.path.join(LWM_REPO_DIR, "model_weights.pth")

# 训练产物
CKPT_PRETRAIN = os.path.join(WEIGHTS_DIR, "lwm_continued.pt")   # 阶段1: MCM 继续预训练
CKPT_LLR = os.path.join(WEIGHTS_DIR, "lwm_llr.pt")              # 阶段2: LLR 微调（主）
CKPT_LLR_NO_PT = os.path.join(WEIGHTS_DIR, "lwm_llr_no_pretrain.pt")  # 阶段2: 对照（无预训练）

# ================= 训练（多配置大规模版：每配置组合 8 个不同样本） =================
GROUP_SIZE = 8           # 每个配置组合生成的样本数（不同信道/噪声/比特实现）
PT_N = 2000              # 阶段1 MCM 样本数（250 组合 × 8）
TRAIN_N = 2400           # 阶段2 训练样本数（300 组合 × 8）
VAL_N = 160              # 验证样本数（20 组合 × 8）
PT_EPOCHS = 15
FT_EPOCHS = 40
BATCH = 8
GRAD_ACCUM = 8
PT_LR = 1e-5             # MCM 预训练学习率
PT_MASK_RATIO = 0.15     # MCM mask 比例
LR = 1e-4                # decoder 学习率
LR_BACKBONE = 1e-6       # LWM 骨干学习率（微调）
SEED = 7

# 评估
EVAL_PER_SNR = 8         # 每 SNR 样本数（多配置循环采样）
EVAL_SNR_LIST = [-5, 0, 5, 10, 15, 20]
EVAL_SEED = 42

# 数据缓存（Sionna 生成一次，训练复用）
CACHE_PT = os.path.join(DATA_DIR, "pusch_pt.pkl")
CACHE_TRAIN = os.path.join(DATA_DIR, "pusch_train.pkl")
CACHE_VAL = os.path.join(DATA_DIR, "pusch_val.pkl")

for d in (DATA_DIR, WEIGHTS_DIR):
    os.makedirs(d, exist_ok=True)
