# -*- coding: utf-8 -*-
"""
LWM LLR 预测项目 —— 全局配置
"""
import os

# ================= 系统参数（3GPP 兼容 OFDM） =================
N_ANT = 8                # 基站天线数
N_SC = 128               # 训练块大小（子载波数，= BLOCK_SIZE）
SCS_KHZ = 30             # 子载波间隔 30kHz（NR mu=1）
FFT_SIZE = 4096          # FFT 大小（覆盖最大 3276 子载波）
PILOT_SPACING = 4        # DM-RS comb-4
MAX_LLR = 20.0           # LLR 裁剪上限
CHANNEL_MODEL = "TDL-C"  # 3GPP TR 38.900 TDL-C

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

# ================= 训练超参数 =================
# 阶段 1: MCM 继续预训练
PT_N_SAMPLES = 3000        # 预训练样本数（每样本 128 子载波 = 1 块）
PT_EPOCHS = 15
PT_BATCH = 64
PT_LR = 1e-5
PT_MASK_RATIO = 0.15
PT_SEED = 0

# 阶段 2: LLR 微调
FT_TRAIN_N = 4000
FT_VAL_N = 500
FT_EPOCHS = 25
FT_BATCH = 64
FT_LR = 1e-4              # decoder 学习率
FT_LR_BACKBONE = 1e-6     # LWM 骨干学习率（微调）
FT_FREEZE_BACKBONE = False
FT_SEED = 1

# 评估
EVAL_N = 400              # 评估样本数
EVAL_SEED = 42
EVAL_NSC_LIST = [32, 128, 512, 2048]   # 可变子载波数测试（含大带宽分块）
EVAL_SNR_LIST = [-5, 0, 5, 10, 15, 20]

for d in (DATA_DIR, WEIGHTS_DIR):
    os.makedirs(d, exist_ok=True)
