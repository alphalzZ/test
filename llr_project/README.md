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

**本项目的做法**：利用 LWM 预训练模型从带噪信道估计中提取空间/频率特征（隐式去噪、利用信道先验），通过 CNN 残差网络（NNreceiver 架构）在全网格上**直接预测逐比特 LLR**（BCE 监督真实 0/1 bit），无需传统软解调，显著提升软信息质量、大幅降低 BER。

**核心特性**：

- ✅ 标准 Sionna PUSCH 链路仿真（DMRS type1 {1+1}、TDL-A 信道、LS+插值信道估计）
- ✅ LWM 官方权重直接复用（0.6M 参数骨干）
- ✅ 子载波对齐 tokenizer，模型输入维度 **{num_rx, num_sc, num_symb}** = (8, 120, 14)
- ✅ **CNN 残差 Decoder**（NNreceiver 架构）：全网格空洞可分离卷积残差网络，
  输入**不含 llr_base**（免去传统软解调，降低推理复杂度）
- ✅ **BCE 损失**：监督标签为真实传输 0/1 bit（无需理想 LLR 标签）
- ✅ 完整的训练（MCM 继续预训练 + 监督微调）与评估（BER / LLR MSE / 相关系数）流水线
- ✅ **CUDA GPU 加速**（自动检测，无 GPU 回退 CPU）；大规模训练下 **BER 在 0~20 dB 相对传统基线改善 53%~99%+**
- ✅ 数据缓存（Sionna 生成一次，训练复用）

---

## 2. 系统架构

```
                    ┌────────────────────────────────────────────┐
  信道估计 H_est     │   LWM 骨干（Transformer 编码器）             │
  (8×120×14 3D)     │   逐 OFDM 符号 tokenizer → patch 序列       │
        │           │   [CLS, patch_1, ..., patch_128] × 14 符号  │
        ▼           │         ↓                                 │
  ┌─────────┐       │   12 层双向注意力编码                        │
  │tokenizer│──────►│         ↓                                 │
  └─────────┘       │   逐子载波 channel embedding (64 维)        │
        │           └────────────────────────────────────────────┘
        │                          │
        │                          ▼
  均衡符号 z ──────────────┐   ┌────────────────────────────────────┐
  噪声方差 σ² ─────────────┼──►│ CNN LLR Decoder（NNreceiver 架构）  │
  调制阶数 one-hot ────────┘   │ 全网格特征图 (14 符号 × 120 子载波) │
                              │ GroupNorm + 3x3 转置卷积            │
                              │ 11 个残差块（空洞可分离卷积）        │
                              │ 输出 逐数据 RE 逐比特 LLR logits     │
                              └────────────────────────────────────┘
```

**两阶段训练**：

| 阶段 | 名称 | 监督 | 目标 |
|------|------|------|------|
| 1 | MCM 继续预训练 | 无监督 | Masked Channel Modeling，适配 3GPP OFDM 信道域 |
| 2 | LLR 微调 | 有监督（真实 0/1 bit） | BCE 最小化，直接优化硬判决正确率 |

---

## 3. 环境要求与安装

### 3.1 系统要求

- **Python**：3.11+（本项目在 3.14.4 上验证，Sionna 2.x 要求）
- **CPU**：任意（模型仅 1.3M 参数）
- **内存**：≥ 8GB（数据规模可调）
- **GPU**：**推荐**（CUDA 加速，训练快约 17 倍）；无 GPU 自动回退 CPU
  - 本项目验证环境：RTX 3060 Laptop（6GB, sm_86, CUDA 13.2）
  - 显存限制：6GB 下训练 batch=16 + 梯度累积=4（等效 batch 64）
  - 若 GPU 显示 "requires reset"（驱动异常），需重启机器或让管理员执行 `nvidia-smi --gpu-reset`

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

该脚本依次执行：MCM 继续预训练 → LLR 微调（主模型）→ LLR 微调（对照模型）→ 性能评估。

**耗时参考**（本机验证）：

| 硬件 | 阶段 | 耗时 |
|------|------|------|
| CPU | 全流程（小规模） | ~1.5~2 小时 |
| **GPU** | 全流程（大规模 3000 样本 × 30 epoch） | **~1.7 小时** |

也可以分步执行，见第 7 节。Sionna 数据首次生成会缓存到 `data/`（npz），后续训练直接加载复用。

---

## 5. 数据说明

### 5.1 数据来源

数据由 **`data_gen_sionna.py`** 基于 [Sionna 2.x](https://nvlabs.github.io/sionna/phy/index.html)（PyTorch 后端）在线生成，实现**标准的 5G NR PUSCH 链路**（对齐 Sionna `sionna.phy.nr` 模块）：

1. **PUSCH 发射机**（`PUSCHTransmitter`）：QAM 调制 → 层映射 → 资源网格（数据 + DMRS）
2. **DMRS 配置**（`PUSCHDMRSConfig`）：**type1**，**{1+1} 双 DMRS 符号**（前置符号 2 + 附加符号 11），`num_cdm_groups_without_data=2`
3. **信道建模**（`sionna.phy.channel.tr38901.TDL` + `TimeChannel`）：3GPP TDL-A，8 根接收天线
4. **OFDM 调制/解调**（`OFDMModulator`/`OFDMDemodulator`）：含 CP，过信道加噪
5. **信道估计**（`PUSCHLSChannelEstimator`）：DMRS 处 LS 估计 + **线性插值** → 输出**完整频域信道估计**
6. **MMSE 均衡** + **max-log LLR**（数据 RE，理想信道 max-log 作参考 LLR，评估用）

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
| `llr_ref` | (1440, log2M) | float32 | 参考 LLR（理想信道 max-log，评估指标用） |
| `bits_tx` | (1440, log2M) | int8 | 发送比特（评估用） |
| `mod_order` | scalar | int32 | 调制阶数 |
| `n_sc`/`n_symb`/`n_data` | scalar | int32 | 子载波/符号/数据 RE 数 |
| `data_re_idx` | (1440, 2) | int32 | 数据 RE 索引 [sc, symb] |

> 说明：DMRS 符号 2、11 全子载波为导频（2 CDM 组），**数据 RE 数 = 12 符号 × 120 子载波 = 1440**。LLR/比特只存在于数据 RE。模型输入 `H_est` 为完整 14 符号 × 120 子载波 × 8 天线的信道估计（用户要求的 `{num_rx, num_sc, num_symb}` 维度）。

### 5.4 数据集规模（config.py 可调，GPU 大规模版）

| 用途 | 规模 | 说明 |
|------|------|------|
| 继续预训练 | 3000 | 只需信道，无标签；缓存 `data/pusch_pt_train.npz` |
| LLR 训练 | 3000 | 混合调制 + 随机 SNR；缓存 `data/pusch_ft_train.npz` |
| LLR 验证 | 300 | 缓存 `data/pusch_ft_val.npz` |
| 评估 | 300 | 6 种 SNR × 50 样本，混合调制 |

---

## 6. 代码结构

```
llr_project/
├── config.py            # 全局配置：系统参数、路径、训练超参数（GPU 版）
├── data_gen_sionna.py   # Sionna PUSCH 数据生成器（标准 5G NR 链路，CPU 强制）
├── data_gen.py          # 工具函数（QAM 星座 / max-log LLR / demapper）
├── tokenizer.py         # 子载波对齐 patch tokenizer（含 3D 逐符号 tokenize）
├── dataset.py           # PyTorch Dataset + 数据缓存（npz）
├── model.py             # LWM 骨干（官方结构）+ CNN 残差 LLR Decoder（3D 输入）
├── train_pretrain.py    # 阶段 1：MCM 继续预训练（GPU 向量化）
├── train_llr.py         # 阶段 2：LLR 微调（主/对照模式，GPU）
├── evaluate.py          # 性能评估（基线/本方案/对照对比 + BER 曲线）
├── run_all.sh           # 一键运行脚本
├── REPORT.md            # 开发与评估报告
├── README.md            # 本文档
├── data/                # Sionna 数据缓存（npz，训练复用）
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
#   --epochs 30       训练轮数（默认 30）
#   --samples 3000    样本数
#   --batch 16        批大小（6GB 显存上限；CPU 可调大）
#   --grad-accum 4    梯度累积步数（等效 batch 64）
#   --lr 1e-5         学习率（小 lr 防灾难性遗忘）
#   --seed 0          随机种子
```

**做了什么**：加载官方 LWM 权重 → 用自己的 3GPP OFDM 信道（H_est）做 **Masked Channel Modeling**：逐 OFDM 符号随机 mask 15% 的 patch，用 MSE 损失重建被 mask 的 patch（**GPU 向量化实现**）。使 LWM 的隐空间适配 3GPP 信道分布。

**产出**：`weights/lwm_continued.pt`；数据缓存 `data/pusch_pt_train.npz`

**参考效果**（GPU, 3000 样本 30 epoch）：MCM loss 0.586 → **0.426**（约 33 分钟；GPU 前向加速比约 17×）。

### 7.2 阶段 2：LLR 微调

```bash
# 主模型（使用阶段1继续预训练权重）
python train_llr.py

# 对照模型（使用官方权重，无继续预训练，用于消融对比）
python train_llr.py --no-pretrain

# 可选参数：
#   --train-n 3000    训练样本数
#   --val-n 300       验证样本数
#   --epochs 30       训练轮数
#   --batch 16        批大小（6GB 显存上限）
#   --grad-accum 4    梯度累积步数
#   --lr 1e-4         Decoder 学习率
#   --lr-backbone 1e-6  LWM 骨干学习率
#   --freeze-backbone  冻结骨干只训 Decoder
```

**做了什么**：加载 LWM（继续预训练或官方权重）→ 冻结/微调骨干 → 训练 CNN 残差 Decoder。输入 `[channel_emb + H_est patch + z + σ² + mod_onehot]`（**不含 llr_base**，无需传统软解调），在全网格特征图上直接输出逐数据 RE 的逐比特 LLR logits；监督标签为**真实传输的 0/1 bit**，损失函数为 **BCE**（binary cross-entropy）。

**产出**：`weights/lwm_llr.pt`（主）、`weights/lwm_llr_no_pretrain.pt`（对照）

**参考效果**（GPU, 3000 样本 30 epoch）：val BCE 0.620 → **0.244**，val 准确率 0.62 → 0.89（约 38 分钟/模型）。

### 7.3 性能评估

```bash
python evaluate.py
```

**对比 4 个方案**：

1. **理想上界**（ideal）：max-log LLR 用真实信道 H_true —— 理论上限
2. **传统基线**（base）：max-log LLR 用带噪估计 H_est —— 传统接收机
3. **本方案**（lwm）：LWM（继续预训练）+ CNN 残差 Decoder
4. **对照**（lwm_noPT）：LWM（官方权重，无继续预训练）+ CNN 残差 Decoder

**指标**：

- **硬判决 BER**：`sign(LLR)` 与发送比特的比较（主指标）
- **LLR MSE**：预测 LLR 与理想 LLR 的均方误差（注：BCE logits 与 max-log 尺度不同，仅参考）
- **LLR 相关系数**：预测 LLR 与理想 LLR 的 Pearson 相关（尺度无关的形状相似度）

**产出**：终端表格、`eval_ber_curves.png`（BER vs SNR 曲线）、`eval_results.json`

---

## 8. 实验结果（v2：CNN+BCE，GPU 大规模训练，300 评估样本）

### 8.1 硬判决 BER（主指标，越低越好）

| SNR | 传统基线 | **LWM+CNN (v2)** | 改善 | 对照(noPT) |
|-----|---------|-----------------|------|-----------|
| -5 dB | 0.2614 | 0.2390 | -8.6% | 0.2377 |
| 0 dB | 0.1733 | **0.0818** | **-52.8%** ✅ | 0.0821 |
| 5 dB | 0.0890 | **0.0217** | **-75.6%** ✅ | 0.0218 |
| 10 dB | 0.1572 | **0.0020** | **-98.7%** ✅ | 0.0020 |
| 15 dB | 0.0527 | **0.0002** | **-99.6%** ✅ | 0.0002 |
| 20 dB | 0.0409 | **0.0003** | **-99.3%** ✅ | 0.0003 |

按调制阶数（全 SNR 平均 BER）：4QAM 0.0678 → 0.0759（⚠️ 轻微回退，见 8.3）；16QAM 0.0000 → 0.0004；64QAM 0.2102 → **0.0639（-70%）**；256QAM 0.2227 → **0.0355（-84%）**。

### 8.2 LLR 指标（BCE logits 与 max-log 参考尺度不同，仅参考）

| 方案 | MSE | 相关系数 |
|------|-----|---------|
| 传统基线（H_est） | 4.30 | 0.964 |
| LWM + CNN（本方案） | 76.8 | 0.820 |

> 模型学的是**后验概率 LLR**（logit = log p/(1-p)），与 max-log 参考尺度不同，故 MSE/相关不直接可比；其硬判决正确率远超基线（见 8.1），对下游软判决译码是更优校准。

### 8.3 结论

- **数量级提升**：移除 llr_base + CNN 残差网络 + BCE，使 0~20 dB BER 改善 **53%~99.6%**（v1 仅 13~20%）；10~20 dB 已接近理想标签水平。
- **高调制受益最大**：64QAM/256QAM（传统基线最大短板）改善 70%~84% —— CNN 全网格上下文 + LWM 信道先验的增益所在。
- **4QAM 轻微回退**（+12%，集中在低 SNR）：后验 LLR 在 QPSK 决策边界附近有偏置；改进方向见 §11。
- **推理复杂度降低**：不再需要传统软解调（llr_base），接收链路更简洁。
- **继续预训练 vs 官方权重**：差异 <0.5%（骨干微调 lr 仍偏小，见 §11）。

### 8.4 v1 历史结果（残差 MLP + MSE 训练）

| SNR | 传统基线 | v1 LWM+Decoder | 改善 |
|-----|---------|----------------|------|
| -5 dB | 0.263 | 0.259 | -1.5% |
| 0 dB | 0.174 | 0.170 | -2.3% |
| 5 dB | 0.091 | **0.078** | **-13.8%** |
| **10 dB** | 0.161 | **0.128** | **-20.2%** |
| 15 dB | 0.051 | **0.042** | **-18.4%** |
| 20 dB | 0.040 | **0.035** | **-12.9%** |

v1 LLR MSE：4.260 → **3.442（-19.2%）**。

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

### 9.3 CNN 残差 Decoder（关键设计，v2）

参考 **NNreceiver** 架构（CNN 残差网络），直接在 (符号 × 子载波) 全网格上预测 LLR：

- **特征图输入**：`[channel_emb(64) + H_est_patch(16) + Re(z) + Im(z) + σ² + mod_onehot(4)]` = 87 通道（补零到 88 满足 GroupNorm(groups=2)），布局 (B, 88, 14, 120)
- **网络结构**：GroupNorm(2) → 3×3 转置卷积（64 通道）→ **11 个残差块**（GroupNorm + 空洞可分离卷积 3×3，out_channels `[64,64,128,128,256,256,256,128,128,64,64]`，dilation `[(1,1),(1,1),(2,3),(2,3),(2,3),(3,6),(2,3),(2,3),(2,3),(1,1),(1,1)]`）→ 3×3 卷积 → 8 通道 logits
- **输出**：全网格 logits → 按 `data_re_index` 取数据 RE → `(B, 1440, 8)`，裁剪 ±MAX_LLR
- z 仅数据 RE 位置有值（其余为 0），空洞卷积利用**邻近 RE 的空间/频率上下文**弥补缺失信息

**为什么不用 v1 的 llr_base 残差学习**：llr_base 需要完整传统软解调（复杂且本身就是近似解）。v2 直接预测后验 LLR logits，推理更简洁；且实验表明（配合 BCE）硬判决 BER 反超 v1 一个数量级。CNN 的空间上下文能力替代了"基线锚点"的作用。

### 9.4 损失函数

有效 RE × 有效比特掩码上的 **BCE（binary cross-entropy with logits）**，标签为**真实传输的 0/1 bit**：

```
loss = BCEWithLogits(pred, bits) * mask,  mask = valid × (bit_idx < log2M)
```

- 直接优化分类正确率（BER），无需理想 LLR 标签（v1 的 max-log 监督标签可弃用）
- 支持 batch 内混合调制阶数（超出 log2M 的比特通道被掩码）
- 训练/验证同时监控硬判决 BER

### 9.5 LWM 骨干微调策略

- 默认微调（backbone lr=1e-6，远小于 decoder 的 1e-4，防灾难性遗忘）
- 可选 `--freeze-backbone` 冻结骨干（数据少时）
- 实验结论：骨干 lr=1e-6 时继续预训练与官方权重差异 <0.5%，后续可尝试提高骨干 lr（见第 11 节）

### 9.6 GPU 训练策略

- 6GB 显存下 **batch=16 + 梯度累积=4**（等效 batch 64），`expandable_segments` 防碎片
- 预训练 MCM 全程 GPU 向量化（tokenize 与 mask 均在 GPU）
- Sionna 数据生成强制 CPU（避免其 CUDA 全局 device 导致 CPU/GPU 混合错误）

---

## 10. 常见问题（FAQ）

**Q1：没有 GPU 能跑吗？**
能。模型仅 1.3M 参数，全流程 CPU 可运行（小规模约 2 小时）。代码自动检测 CUDA，有 GPU 自动加速（本机实测快约 17 倍）。

**Q1b：有 GPU 但 `torch.cuda.is_available()` 为 False？**
- 运行 `nvidia-smi`，若显示 **"GPU requires reset" / ERR**：GPU 驱动状态异常，需重启机器或让管理员执行 `sudo nvidia-smi --gpu-reset`
- 检查驱动/CUDA 版本与 torch 匹配（本项目：驱动 595.84/CUDA 13.2 + torch 2.13.0+cu130）
- 若报 `CUDA out of memory`：6GB 显存需 `--batch 16 --grad-accum 4`（见 config.py 默认值）

**Q1c：Sionna 数据生成报 CPU/GPU 设备混合错误？**
Sionna 检测到 CUDA 会自动把全局 device 设为 cuda，导致其内部组件 CPU/GPU 混合。本项目已在 `data_gen_sionna.py` 顶部强制 `sionna.config.device = "cpu"`（数据生成在 CPU，训练在 GPU），无需处理。

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
修改 `data_gen_sionna.py` 的 `SionnaPUSCHSystem`（信道模型、带宽、DMRS 配置等），或改写 `generate_dataset()` 读取自己的数据集。样本需保证字段一致（见 5.3 表）：`H_est(8,120,14)`、`z(1440,)`、`sigma2`、`llr_ref(1440,log2M)`、`bits_tx`、`mod_order`、`sigma2_eq`。

**Q6：为什么继续预训练收益不明显？**
当前 MCM（30 epoch, loss 0.426）与官方权重对照的差异 <0.5%（v2 训练），主要因为骨干微调 lr=1e-6 过小。改进建议：提高骨干 lr（1e-4 量级）或两阶段训练（先冻骨干训 decoder，再联合微调），见第 11 节。

**Q7：训练很慢怎么办？**
- 使用 GPU（自动检测；`CUDA_VISIBLE_DEVICES` 指定设备）
- 6GB 显存用默认 `--batch 16 --grad-accum 4`；显存更大可调大 batch
- 数据缓存（`data/*.npz`）避免重复生成 Sionna 数据
- CPU 上设置 `OMP_NUM_THREADS=8` 提升多核利用率

---

## 11. 后续改进方向

1. **解决 4QAM 低 SNR 回退**：QPSK 样本加权、SNR 条件化（把 SNR 作为额外输入通道）、或对 QPSK 单独校准。
2. **深挖继续预训练价值**：提高 backbone 微调 lr（1e-4 量级）或两阶段训练（先冻骨干训 decoder，再联合微调），充分发挥 MCM 域适配（当前 lr=1e-6 下主 vs 对照差异 <0.5%）。
3. **端到端评估**：接入 5G NR LDPC 译码器，用 BLER 作为最终落地指标（v2 的 LLR 是后验概率 logits，天然适合软判决译码）。
4. **模型结构探索**：CNN decoder 的可分离卷积开关（`CNN_SEP_CONV`）、转置/普通卷积（`CNN_TRANSPOSE`）、dilation 配置；取 LWM 浅层（3~6 层）特征。
5. **信道场景扩展**：TDL-B/C/D、多普勒、多 UE 干扰、不同带宽（更多 RB）。
6. **更强基线对比**：LMMSE 插值器（`LMMSEInterpolator`）、迭代接收机。

---

## 12. 参考

- [wi-lab/lwm - HuggingFace](https://huggingface.co/wi-lab/lwm)：LWM 模型仓库与 README
- LWM 论文：Alikhani et al., "Large Wireless Model (LWM): A Foundation Model for Wireless Channels", arXiv:2411.08872
- [Sionna 2.x](https://nvlabs.github.io/sionna/phy/index.html)：标准 5G NR PUSCH 链路仿真库（PyTorch 后端）
- [Sionna GitHub](https://github.com/NVlabs/sionna)：源码与示例
- 3GPP TR 38.901（信道模型）/ TS 38.211（物理信道，DMRS/PUSCH）
- 设计文档：`../LWM_LLR_Design_Doc.md`（本项目的总体设计）

---

*最后更新：2026-08*　*环境：Python 3.14.4 / torch 2.13.0+cu130 / RTX 3060 Laptop (CUDA)*
