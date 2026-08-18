# -*- coding: utf-8 -*-
"""
夜间大规模训练方案配置 v2（实测 ~2.8 小时，RTX 3060 Laptop 6GB）

通过 config.py 末尾的可选覆盖机制加载（删除本文件即恢复默认小规模配置）。
v2 改动（基于夜间版 v1 性能分析结论）：
  1. **砍掉 MCM 继续预训练**（USE_PRETRAIN=False）：实测预训练收益≈噪声级
     （主 vs 对照 val loss 差 0.8%、逐样本均值差 -0.001），省 ~0.5h 训练 + 0.8GB PT 数据。
  2. **砍训练轮数**：FT_EPOCHS 48→22（冻结 14 + 联合 8）。v1 实测 2a 在 ep12~14 即收敛、
     2b 主在联合第 11 ep 最优、对照第 5 ep 最优，后半段 val 震荡白跑（省 ~2.8h/两模型）。
  3. 主模型 = 官方权重直接两阶段微调（与 v1 对照等价），不再训练第二个对照模型。

数据规模（每配置组合 8 个不同样本）：
  - LLR 训练：50000 样本（6250 组合 × 8）
  - LLR 验证：1000 样本（125 组合 × 8）

训练：
  - 阶段2 LLR 两阶段微调：14（冻结骨干）+ 8（联合微调，骨干 lr=1e-4）= 22 epoch

时间预算（v1 实测基准外推）：
  - 数据生成 ~1.4h（51000 样本，分片缓存、可断点续跑）
  - 阶段2 微调 ~1.3h（22 epoch，单模型）  评估 ~0.1h
  - 合计 ~2.8h（执行脚本 train_night.sh，完成后产物为 *_night.pt / pusch_night_*.pkl，
    不影响白天的小规模产物，可并行对比）

注意：磁盘占用约 2.2GB（data/pusch_night_*.pkl，git 已忽略）；
旧版 PT 数据 data/pusch_night_pt.pkl.* 已不再生成，可手动删除省 ~0.8GB。
"""
import os

# ================= 数据规模（不再生成 PT 数据） =================
GROUP_SIZE = 8            # 每配置组合样本数（不同信道/噪声/比特实现）
PT_N = 0                  # MCM 数据量（USE_PRETRAIN=False，置 0 跳过生成）
TRAIN_N = 50000           # 阶段2 训练样本数（6250 组合 × 8）
VAL_N = 1000              # 验证样本数（125 组合 × 8）

# ================= 训练超参 =================
USE_PRETRAIN = False      # 砍掉阶段1 MCM 继续预训练，主模型直接用官方权重
PT_EPOCHS = 0             # 不再使用
PT_LR = 1e-5              # 不再使用
FT_EPOCHS = 22            # LLR 两阶段总轮数（14 冻结 + 8 联合，v1 实测后半段白跑）
FT_FREEZE_EPOCHS = 14     # 阶段2a 冻结骨干训 decoder 轮数
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
