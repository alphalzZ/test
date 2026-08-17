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

for d in (DATA_DIR, WEIGHTS_DIR):
    os.makedirs(d, exist_ok=True)
