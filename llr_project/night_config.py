# -*- coding: utf-8 -*-
"""
夜间大规模训练方案配置（约 12 小时预算，实测 ~8.5 小时，RTX 3060 Laptop 6GB）

通过 config.py 末尾的可选覆盖机制加载（删除本文件即恢复默认小规模配置）。
核心思路：**扩大训练数据规模**（约 20 倍）增强模型性能。

数据规模（每配置组合 8 个不同样本）：
  - MCM 继续预训练：20000 样本（2500 组合 × 8）
  - LLR 训练：      50000 样本（6250 组合 × 8）
  - LLR 验证：      1000 样本（125 组合 × 8）

训练：
  - 阶段1 MCM：20 epoch（lr=1e-5）
  - 阶段2 LLR 两阶段微调：24（冻结骨干）+ 24（联合微调，骨干 lr=1e-4）= 48 epoch

时间预算（实测基准外推）：
  - 数据生成 ~1.7h（71000 样本，分集缓存、可断点续跑）
  - 阶段1 MCM ~0.5h   阶段2 主模型 ~2.9h   阶段2 对照 ~2.9h   评估 ~0.1h
  - 合计 ~8.5h（执行脚本 train_night.sh，完成后产物为 *_night.pt / pusch_night_*.pkl，
    不影响白天的小规模产物，可并行对比）

注意：磁盘占用约 3GB（data/pusch_night_*.pkl，git 已忽略）。
"""
import os

# ================= 数据规模 =================
GROUP_SIZE = 8            # 每配置组合样本数（不同信道/噪声/比特实现）
PT_N = 20000              # 阶段1 MCM 样本数（2500 组合 × 8）
TRAIN_N = 50000           # 阶段2 训练样本数（6250 组合 × 8）
VAL_N = 1000              # 验证样本数（125 组合 × 8）

# ================= 训练超参 =================
PT_EPOCHS = 20            # MCM 预训练轮数（数据 10 倍，轮数相应增加）
PT_LR = 1e-5
FT_EPOCHS = 48            # LLR 两阶段总轮数（24 冻结 + 24 联合）
FT_FREEZE_EPOCHS = 24     # 阶段2a 冻结骨干训 decoder 轮数
BATCH = 8
GRAD_ACCUM = 8
LR = 1e-4                 # decoder 学习率
LR_BACKBONE = 1e-4        # 骨干学习率（阶段2b 联合微调）
SEED = 7

# ================= 评估（加大样本量更稳） =================
EVAL_PER_SNR = 16         # 每 SNR 16 样本（共 96 样本）
EVAL_SNR_LIST = [-5, 0, 5, 10, 15, 20]
EVAL_SEED = 42

# ================= 独立产物路径（不覆盖白天小规模产物） =================
CACHE_PT = os.path.join(DATA_DIR, "pusch_night_pt.pkl")
CACHE_TRAIN = os.path.join(DATA_DIR, "pusch_night_train.pkl")
CACHE_VAL = os.path.join(DATA_DIR, "pusch_night_val.pkl")
CKPT_PRETRAIN = os.path.join(WEIGHTS_DIR, "lwm_continued_night.pt")
CKPT_LLR = os.path.join(WEIGHTS_DIR, "lwm_llr_night.pt")
CKPT_LLR_NO_PT = os.path.join(WEIGHTS_DIR, "lwm_llr_no_pretrain_night.pt")
