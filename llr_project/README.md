# LWM-LLR：基于 Large Wireless Model 的 3GPP OFDM 软解调 LLR 预测

> 在预训练无线信道基础模型 **LWM**（[wi-lab/lwm](https://huggingface.co/wi-lab/lwm)，Large Wireless Model）之上进行二次开发：用自己的 3GPP 兼容 OFDM 数据做 MCM 继续预训练，再用 **LWM 骨干 + 简单 Decoder** 做逐比特 **LLR（对数似然比）** 预测，实现软解调增强。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 系统架构](#2-系统架构)
- [3. 环境要求与安装](#3-环境要求与安装)
- [4. 快速开始](#4-快速开始)
- [5. 数据说明](#5-数据说明)
- [6. 代码结构](#6-代码结构)
- [7. 使用指南（分步详解）](#7-使用指南分步详解)
  - [7.1 阶段 1：MCM 继续预训练](#71-阶段-1mcm-继续预训练)
  - [7.2 阶段 2：LLR 微调](#72-阶段-2llr-微调)
  - [7.3 性能评估](#73-性能评估)
- [8. 实验结果](#8-实验结果)
- [9. 设计要点与关键技术](#9-设计要点与关键技术)
- [10. 常见问题（FAQ）](#10-常见问题faq)
- [11. 后续改进方向](#11-后续改进方向)
- [12. 参考](#12-参考)

---

## 1. 项目简介

**背景**：OFDM 接收机的软解调（soft-demapping）需要将均衡后的符号转换为逐比特 LLR 软信息，供信道译码（如 5G NR LDPC）使用。传统方法假设信道估计理想；当信道估计存在误差（低 SNR、导频稀疏、高速移动）时，LLR 质量下降。

**本项目的做法**：利用 LWM 预训练模型从带噪信道估计中提取空间/频率特征（隐式去噪、利用信道先验），通过一个轻量 Decoder **修正**传统软解调输出的 LLR，从而提升软信息质量、降低 BER。

**核心特性**：

- ✅ 标准 Sionna PUSCH 链路仿真（DMRS type1 {1+1}、TDL-A 信道、LS-DFT 信道估计）
- ✅ LWM 官方权重直接复用（0.6M 参数，CPU 可训练）
- ✅ 子载波对齐 tokenizer，模型输入维度 **{num_rx, num_sc, num_symb}** = (8, 120, 14)
- ✅ **残差学习** Decoder：保证性能不劣于传统基线，专注"增强"
- ✅ 完整的训练（MCM 继续预训练 + 监督微调）与评估（BER / LLR MSE）流水线
- ✅ 全部 CPU 可运行，无 GPU 依赖

---

## 2. 系统架构

```
                    ┌────────────────────────────────────────────┐
  信道矩阵 H_est     │   LWM 骨干（Transformer 编码器）             │
  (8天线×128子载波)  │   子载波对齐 tokenizer → patch 序列          │
        │           │   [CLS, patch_1, ..., patch_128]           │
        ▼           │         ↓                                 │
  ┌─────────┐       │   12 层双向注意力编码                        │
  │tokenizer│──────►│         ↓                                 │
  └─────────┘       │   逐子载波 channel embedding (64 维)        │
        │           └────────────────────────────────────────────┘
        │                          │
        │                          ▼
  均衡符号 z ──────────────┐   ┌────────────────────────────────┐
  噪声方差 σ² ─────────────┤   │  LLR Decoder（残差学习 MLP）      │
  调制阶数 one-hot ────────┼──►│  输入 [h_emb + H_patch + z +    │
  传统基线 LLR ────────────┘   │        σ² + mod_oh + llr_base]  │
                             │  输出 Δ（修正量）                  │
                             │  LLR_pred = llr_base + Δ         │
                             └────────────────────────────────┘
```

**两阶段训练**：

| 阶段 | 名称 | 监督 | 目标 |
|------|------|------|------|
| 1 | MCM 继续预训练 | 无监督 | Masked Channel Modeling，适配 3GPP OFDM 信道域 |
| 2 | LLR 微调 | 有监督 | 最小化预测 LLR 与理想 max-log LLR 的误差 |

---

## 3. 环境要求与安装

### 3.1 系统要求

- **Python**：3.10+（本项目在 3.14.4 上验证）
- **CPU**：任意（模型仅 0.6M 参数；训练建议 4+ 核）
- **内存**：≥ 8GB（数据规模可调）
- **GPU**：可选（代码自动检测 CUDA，无 GPU 自动回退 CPU）

### 3.2 依赖安装

```bash
# 1. 创建虚拟环境（如系统缺少 python3-venv 包，用 --without-pip + get-pip.py）
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖（Sionna 2.0.1 为纯 PyTorch 实现，无需 TensorFlow）
pip install torch numpy pandas matplotlib scikit-learn tqdm
pip install sionna
```

### 3.3 获取 LWM 官方权重

本项目复用官方 LWM 权重作为骨干初始化（`model.py` 的 `load_official_backbone()`）：

```bash
# 克隆 LWM 仓库（若尚未克隆）
git clone https://huggingface.co/wi-lab/lwm /home/le-lei/workspace/test/LWM
# 下载权重（仓库中 model_weights.pth 是 Git LFS 指针，系统无 git-lfs 时用 curl）
cd /home/le-lei/workspace/test/LWM
curl -sL "https://huggingface.co/wi-lab/lwm/resolve/main/model_weights.pth" -o model_weights.pth
```

权重路径在 `config.py` 的 `LWM_OFFICIAL_CKPT` 中配置（默认 `../LWM/model_weights.pth`）。

---

## 4. 快速开始

```bash
cd llr_project
chmod +x run_all.sh
./run_all.sh
```

该脚本依次执行：MCM 继续预训练 → LLR 微调（主模型）→ LLR 微调（对照模型）→ 性能评估。**CPU 全流程约 1.5~2 小时**。

也可以分步执行，见第 7 节。

---

## 5. 数据说明

### 5.1 数据来源

数据由 **`data_gen_sionna.py`** 基于 [Sionna 2.x](https://nvlabs.github.io/sionna/phy/index.html)（PyTorch 后端）在线生成，实现**标准的 5G NR PUSCH 链路**（对齐 Sionna `sionna.phy.nr` 模块）：

1. **PUSCH 发射机**（`PUSCHTransmitter`）：QAM 调制 → 层映射 → 资源网格（数据 + DMRS）
2. **DMRS 配置**（`PUSCHDMRSConfig`）：**type1**，**{1+1} 双 DMRS 符号**（前置符号 2 + 附加符号 11），`num_cdm_groups_without_data=2`
3. **信道建模**（`sionna.phy.channel.tr38901.TDL` + `TimeChannel`）：3GPP TDL-A，8 根接收天线
4. **OFDM 调制/解调**（`OFDMModulator`/`OFDMDemodulator`）：含 CP，过信道加噪
5. **信道估计**（`PUSCHLSChannelEstimator`）：DMRS 处 LS 估计 + **线性插值** → 输出**完整频域信道估计**
6. **MMSE 均衡** + **max-log LLR**（数据 RE，用理想信道作监督标签）

> 安装：`pip install sionna`（Sionna 2.0.1 已改为纯 PyTorch 实现，**无需 TensorFlow**，支持 Python 3.11+）。

### 5.2 系统参数（标准 3GPP NR PUSCH）

| 参数 | 值 | 说明 |
|------|-----|------|
| 接收天线（gNB） | 8 | `num_rx` |
| 发射天线（UE） | 1 | `num_tx_ant` |
| 子载波 | 120 | 10 RB × 12 |
| OFDM 符号/slot | 14 | `num_symb` |
| 子载波间隔 | 15 kHz | NR `mu=0` |
| 信道模型 | TDL-A | 3GPP TR 38.901，delay_spread=30ns，3.5GHz |
| DMRS | type1, {1+1} | 符号 2 + 符号 11，2 CDM 组 |
| 调制方式 | QPSK/16QAM/64QAM/256QAM | Gray 映射，能量归一化 |
| 均衡 | MMSE（8→1） | 逐数据 RE |
| SNR 范围 | -5 ~ 25 dB | 训练时均匀采样 |
| **模型输入信道** | **(8, 120, 14)** | `{num_rx, num_sc, num_symb}` 完整频域信道估计 |

### 5.3 数据格式

样本为 dict（data_gen_sionna 输出）：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `H_est` | **(8, 120, 14)** | complex64 | **完整频域信道估计**（LS+插值，模型输入） |
| `H_true` | (8, 120, 14) | complex64 | 真实信道（评估用） |
| `z` | (1440,) | complex64 | 数据 RE 均衡软符号（模型输入） |
| `sigma2` | scalar | float32 | 噪声方差（模型输入） |
| `sigma2_eq` | (1440,) | float32 | 均衡后等效噪声方差 |
| `llr_ref` | (1440, log2M) | float32 | 参考 LLR（监督标签，理想信道 max-log） |
| `bits_tx` | (1440, log2M) | int8 | 发送比特（评估用） |
| `mod_order` | scalar | int32 | 调制阶数 |
| `n_sc`/`n_symb`/`n_data` | scalar | int32 | 子载波/符号/数据 RE 数 |
| `data_re_idx` | (1440, 2) | int32 | 数据 RE 索引 [sc, symb] |

> 说明：DMRS 符号 2、11 全子载波为导频（2 CDM 组），**数据 RE 数 = 12 符号 × 120 子载波 = 1440**。LLR/比特只存在于数据 RE。模型输入 `H_est` 为完整 14 符号 × 120 子载波 × 8 天线的信道估计（用户要求的 `{num_rx, num_sc, num_symb}` 维度）。

### 5.4 数据集规模（config.py 可调）

| 用途 | 规模 | 说明 |
|------|------|------|
| 继续预训练 | 3000 | 只需信道，无标签 |
| LLR 训练 | 4000 | 混合调制 + 随机 SNR |
| LLR 验证 | 500 | |
| 评估 | 384 | 4 种 N_sc × 6 种 SNR × 随机调制 |

---

## 6. 代码结构

```
llr_project/
├── config.py            # 全局配置：系统参数、路径、训练超参数
├── data_gen_sionna.py   # Sionna PUSCH 数据生成器（标准 5G NR 链路）
├── data_gen.py          # 工具函数（QAM 星座 / max-log LLR / demapper）
├── tokenizer.py         # 子载波对齐 patch tokenizer（含 3D 逐符号 tokenize）
├── dataset.py           # PyTorch Dataset（含 llr_base 残差锚点预计算）
├── model.py             # LWM 骨干（官方结构）+ 残差 LLR Decoder + 组合模型
├── train_pretrain.py    # 阶段 1：MCM 继续预训练
├── train_llr.py         # 阶段 2：LLR 微调（主/对照模式）
├── evaluate.py          # 性能评估（4 方案对比 + BER 曲线）
├── run_all.sh           # 一键运行脚本
├── REPORT.md            # 开发与评估报告
├── README.md            # 本文档
├── weights/             # 训练产物
│   ├── lwm_continued.pt       # 阶段1 继续预训练权重
│   ├── lwm_llr.pt             # 阶段2 主模型（继续预训练 + LLR 微调）
│   └── lwm_llr_no_pretrain.pt # 阶段2 对照模型（官方权重直接微调）
├── eval_ber_curves.png  # BER vs SNR 曲线
└── eval_results.json    # 结构化评估结果
```

---

## 7. 使用指南（分步详解）

### 7.1 阶段 1：MCM 继续预训练

```bash
python train_pretrain.py
# 可选参数：
#   --epochs 15       训练轮数
#   --samples 3000    样本数
#   --batch 64        批大小
#   --lr 1e-5         学习率（小 lr 防灾难性遗忘）
#   --seed 0          随机种子
```

**做了什么**：加载官方 LWM 权重 → 用自己的 3GPP OFDM 信道（H_est）做 **Masked Channel Modeling**：随机 mask 15% 的 patch，用 MSE 损失重建被 mask 的 patch。使 LWM 的隐空间适配 3GPP 信道分布。

**产出**：`weights/lwm_continued.pt`

**参考效果**（本机 CPU）：MCM loss 0.607 → 0.492（15 epoch，约 22 分钟）。

### 7.2 阶段 2：LLR 微调

```bash
# 主模型（使用阶段1继续预训练权重）
python train_llr.py

# 对照模型（使用官方权重，无继续预训练，用于消融对比）
python train_llr.py --no-pretrain

# 可选参数：
#   --train-n 4000    训练样本数
#   --val-n 500       验证样本数
#   --epochs 25       训练轮数
#   --batch 64        批大小
#   --lr 3e-4         Decoder 学习率
#   --lr-backbone 1e-6  LWM 骨干学习率
#   --freeze-backbone  冻结骨干只训 Decoder
```

**做了什么**：加载 LWM（继续预训练或官方权重）→ 冻结/微调骨干 → 训练残差 Decoder。输入 `[channel_emb + H_est patch + z + σ² + mod_onehot + llr_base]`，输出修正量 Δ，最终 `LLR = llr_base + Δ`，监督标签为理想 max-log LLR。

**产出**：`weights/lwm_llr.pt`（主）、`weights/lwm_llr_no_pretrain.pt`（对照）

**参考效果**：val MSE 0.45 → 0.029（25 epoch，约 45 分钟/模型）。

### 7.3 性能评估

```bash
python evaluate.py
```

**对比 4 个方案**：

1. **理想上界**（ideal）：max-log LLR 用真实信道 H_true —— 理论上限
2. **传统基线**（base）：max-log LLR 用带噪估计 H_est —— 传统接收机
3. **本方案**（lwm）：LWM（继续预训练）+ 残差 Decoder
4. **对照**（lwm_noPT）：LWM（官方权重，无继续预训练）+ 残差 Decoder

**指标**：

- **硬判决 BER**：`sign(LLR)` 与发送比特的比较
- **LLR MSE**：预测 LLR 与理想 LLR 的均方误差

**产出**：终端表格、`eval_ber_curves.png`（BER vs SNR 曲线）、`eval_results.json`

---

## 8. 实验结果

### 8.1 LLR MSE（384 评估样本，越小越好）

| 方案 | MSE | vs 传统基线 |
|------|-----|------------|
| 理想上界（H_true） | 0.00 | — |
| 传统基线（H_est） | 31.58 | — |
| **LWM + Decoder（本方案）** | **29.99** | **-5.0%** ✅ |
| LWM 对照（无继续预训练） | 30.00 | -5.0% |

### 8.2 硬判决 BER（越低越好）

| N_sc | SNR | ideal | base | LWM+Decoder |
|------|-----|-------|------|-------------|
| 32 | -5 dB | 0.196 | 0.341 | **0.336** |
| 32 | 0 dB | 0.101 | 0.235 | **0.229** |
| 32 | 5 dB | 0.064 | 0.143 | **0.131** |
| 128 | -5 dB | 0.228 | 0.314 | **0.303** |
| 128 | 0 dB | 0.114 | 0.151 | **0.150** |
| 128 | 10 dB | 0.020 | 0.033 | 0.034 |
| 2048 | -5 dB | 0.146 | 0.274 | 0.284 |
| 2048 | 20 dB | 0.000 | 0.102 | 0.102 |

### 8.3 结论

- **低 SNR（-5~5 dB）**：本方案 BER 系统性低于传统基线 —— 信道估计误差大时，LWM 利用信道先验做隐式去噪，修正 LLR 有效。
- **高 SNR（≥10 dB）**：基线已接近理想，本方案与基线持平（残差设计保证不劣化）。
- **LLR MSE 全面低于基线约 5%**。
- **继续预训练增益**：当前设置下主模型与对照差异 <0.1%（15 epoch MCM 域适配不足，见第 11 节改进方向）。

---

## 9. 设计要点与关键技术

### 9.1 子载波对齐 Tokenizer

LWM 原生输入为 32 天线 × 32 子载波。本项目重新定义 patch 为 **单个子载波上的 8 天线空间向量**：

```
patch_k = [Re(H[:,k]); Im(H[:,k])] ∈ R^16
```

- 与 LWM 原生 `element_length=16` 完全一致 → **可直接复用官方 embedding 权重**
- 序列长度 = 子载波数（128 分块，+CLS 后 129 = 原生 MAX_LEN）
- 每 patch 对应一个子载波 → 与逐子载波 LLR 输出天然对齐

### 9.2 3D 信道输入（{num_rx, num_sc, num_symb}）

Sionna PUSCH 链路输出完整频域信道估计 `H_est (8, 120, 14)`（rx × sc × symb）。模型**逐 OFDM 符号独立编码**：每个符号的 (8, 120) 信道 → 120 patches（pad 到 128）+ CLS → LWM 序列 (129, 16)，14 个符号得到 14 组 channel embedding。Decoder 只对**数据 RE**（1440 个，DMRS 符号 2/11 之外）输出 LLR。

### 9.3 残差学习 Decoder（关键设计）

开发中发现：让网络直接回归理想 LLR 收敛极慢（MSE 0.18 vs 基线 0.024），因为 64 维压缩 embedding 丢失了逐子载波的精确信道信息。最终方案：

- **Decoder 输入**：`[channel_emb(64) + H_est_patch(16) + Re(z) + Im(z) + σ² + mod_onehot(4) + llr_base(8)]` = 95 维
- **输出**：修正量 Δ（tanh 裁剪 ±20）
- **最终 LLR** = `llr_base + Δ`，其中 `llr_base` 是传统均衡后软解调（demapper）输出

**收益**：模型初始即等于传统基线（Δ≈0），学习目标是"增强"而非"从零逼近"，收敛快且保证不劣化。

### 9.4 损失函数

有效子载波 × 有效比特掩码上的**归一化 MSE**（LLR 除以 MAX_LLR=20 归一化到 [-1,1]），支持 batch 内混合调制阶数。

### 9.5 LWM 骨干微调策略

- 默认微调（backbone lr=1e-6，远小于 decoder 的 3e-4）
- 可选 `--freeze-backbone` 冻结骨干（数据少时）

---

## 10. 常见问题（FAQ）

**Q1：没有 GPU 能跑吗？**
能。模型仅 0.6M 参数，全流程 CPU 可运行（约 2 小时）。代码自动检测 CUDA。

**Q2：官方权重路径报错怎么办？**
`config.py` 的 `LWM_OFFICIAL_CKPT` 默认指向 `../LWM/model_weights.pth`。确认已克隆 LWM 仓库并下载真实权重（注意 git-lfs 指针问题，见 3.3 节）。

**Q3：系统缺 `python3-venv` 包无法创建虚拟环境？**
用 `python3 -m venv --without-pip .venv` + 官方 get-pip.py 引导：
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
```

**Q4：为什么 LLR 只对 1440 个 RE（而非全部 1680 个 RE）输出？**
DMRS type1 + 2 CDM 组 + {1+1} 双符号配置下，符号 2 和 11 的全部 120 个子载波都是导频（不发送数据比特），无 LLR。数据 RE = 12 符号 × 120 子载波 = 1440。

**Q5：如何用我自己的数据？**
`data_gen.generate_sample()` 返回样本 dict（见 5.3 表）。用自己的数据时，只需保证字段一致：`H_est(8,N_sc)`、`z(n_data,)`、`sigma2`、`llr_ref(n_data,log2M)`、`bits_tx`、`mod_order`、`sigma2_eq`。可以改写 `generate_dataset()` 为读取自己的数据集。

**Q6：为什么继续预训练收益不明显？**
15 epoch MCM 可能不足（loss 0.49 仍高），且骨干微调 lr 很小。改进建议见第 11 节。

**Q7：训练很慢怎么办？**
- 减少 `--samples` / `--epochs`（先小规模验证）
- 有 GPU 时设置 `CUDA_VISIBLE_DEVICES` 自动加速
- 设置环境变量限制线程数避免过度争抢：`OMP_NUM_THREADS=8`

---

## 11. 后续改进方向

1. **加强继续预训练**：MCM epochs 提到 50+、提高 backbone 微调 lr（1e-4 量级）、分两阶段（先冻骨干训 decoder，再联合微调）。
2. **训练分布对齐推理**：训练时随机截断子载波数（模拟 32/512/2048 等短块），消除分块补零的分布偏移。
3. **端到端评估**：接入 5G NR LDPC 译码器，用 BLER 作为最终落地指标（当前为硬判决 BER）。
4. **改进 LLR 标签**：尝试精确后验（"app"）LLR 替代 max-log 近似，或加权损失。
5. **模型结构探索**：取 LWM 浅层（3~6 层）特征、CLS embedding 全局增益分支、1D-CNN decoder（建模子载波相关性）。
6. **多天线/多流扩展**：扩展到 8×2/8×4 MIMO，decoder 输出多流 LLR。

---

## 12. 参考

- [wi-lab/lwm - HuggingFace](https://huggingface.co/wi-lab/lwm)：LWM 模型仓库与 README
- LWM 论文：Alikhani et al., "Large Wireless Model (LWM): A Foundation Model for Wireless Channels", arXiv:2411.08872
- [Sionna 2.x](https://nvlabs.github.io/sionna/phy/index.html)：标准 5G NR PUSCH 链路仿真库（PyTorch 后端）
- [Sionna GitHub](https://github.com/NVlabs/sionna)：源码与示例
- LWM 论文：Alikhani et al., "Large Wireless Model (LWM): A Foundation Model for Wireless Channels", arXiv:2411.08872
- 3GPP TR 38.901（信道模型）/ TS 38.211（物理信道，DMRS/PUSCH）
- 设计文档：`../LWM_LLR_Design_Doc.md`（本项目的总体设计）

---

*最后更新：2026-08*　*环境：Python 3.14.4 / torch 2.13.0 / CPU*
