# LWM 二次开发设计文档

## 基于继续预训练的 LWM 做 3GPP OFDM 软解调 LLR 预测

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 目标模型 | [wi-lab/lwm](https://huggingface.co/wi-lab/lwm)（Large Wireless Model，无线信道基础模型） |
| 下游任务 | 软解调增强：给定信道估计 + 均衡符号，预测逐比特 LLR |
| 数据域 | 3GPP 兼容 OFDM（8 天线 × 4~3276 子载波，可变） |
| 交付物 | 设计文档 + 关键代码骨架 |

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统模型与任务定义](#2-系统模型与任务定义)
3. [数据准备与生成方案](#3-数据准备与生成方案)
4. [LWM 输入适配（Tokenizer 重新设计）](#4-lwm-输入适配tokenizer-重新设计)
5. [继续训练方案（MCM 域适配）](#5-继续训练方案mcm-域适配)
6. [Decoder 设计（LLR 预测头）](#6-decoder-设计llr-预测头)
7. [两阶段训练流程](#7-两阶段训练流程)
8. [评估方案](#8-评估方案)
9. [工程实施步骤与里程碑](#9-工程实施步骤与里程碑)
10. [风险与注意事项](#10-风险与注意事项)
11. [附录：代码骨架](#11-附录代码骨架)

---

## 1. 项目概述

### 1.1 背景

LWM 是一个面向无线信道的 Transformer 基础模型，通过 **Masked Channel Modeling (MCM)** 自监督预训练，从 32 天线 × 32 子载波的复数信道矩阵中提取高质量特征（CLS 整体语义 + Channel 逐位置空间/频率特征）。它不依赖标签，通用性强，适合作为下游无线任务的骨干。

本项目对 LWM 进行二次开发，目标场景是 **3GPP 兼容 OFDM 系统的软解调（soft-demapping）**：利用 LWM 提取的信道特征辅助计算每个调制符号每个比特的 **对数似然比（LLR）**，为后续信道译码（LDPC）提供软信息。

### 1.2 目标

1. **数据**：设计并生成 3GPP 兼容 OFDM 仿真数据集（信道矩阵、接收符号、LLR 标签）。
2. **继续预训练**：用自己的域内信道数据对 LWM 做 MCM 继续预训练，使特征适应 3GPP OFDM 信道分布。
3. **下游微调**：在继续预训练后的 LWM 之上接一个**简单 decoder**，端到端预测逐比特 LLR。
4. **评估**：以传统软解调为基线，从 LLR 精度、译码 BER/BLER 两个层面验证收益。

### 1.3 总体架构

```
                    ┌────────────────────────────────────────────┐
  信道矩阵 H         │   LWM 骨干（Transformer，12 层双向注意力）     │
  (8天线×N_sc子载波)  │   输入 tokenizer 重新适配（子载波对齐 patch）  │
        │           │                                            │
        ▼           │   H → [CLS, patch_1, ..., patch_T]         │
  ┌─────────┐       │        ↓                                   │
  │tokenizer│ ──────┼──► 双向 Transformer 编码                   │
  └─────────┘       │        ↓                                   │
        │           │   逐 patch 的 channel embedding h_k (64 维) │
        ▼           └────────────────────────────────────────────┘
  均衡符号 z ───────────────┐
  (每子载波)                │ 拼接
  噪声方差 σ² ──────────────┤
                           ▼
              ┌─────────────────────────────┐
              │  LLR Decoder（简单 MLP / 1D-CNN）│
              │  [h_k ; Re(z_k); Im(z_k); σ²]   │
              │        ↓                       │
              │  每子载波每比特的 LLR 估计值       │
              └─────────────────────────────┘
```

**两阶段训练**：

- **阶段 1（继续预训练，无监督）**：MCM，mask 15% 的 patch 并预测，让 LWM 适配 3GPP 域。
- **阶段 2（LLR 微调，有监督）**：冻结/微调 LWM 骨干，训练 decoder，用 max-log LLR 作标签做回归。

### 1.4 术语与符号

| 符号 | 含义 |
|------|------|
| `N_ant` | 基站天线数（本项目 = 8） |
| `N_sc` | 有效子载波数（4 ~ 3276 可变） |
| `H` | 频域信道矩阵，$\mathbf{H}\in\mathbb{C}^{N_{ant}\times N_{sc}}$ |
| `x` | 发送调制符号（QPSK/16QAM/64QAM/256QAM） |
| `y` | 频域接收符号，$\mathbf{y}=\mathbf{H}\mathbf{x}+\mathbf{n}$ |
| `z` | 均衡后软符号（MMSE/ZF），$\mathbf{z}\in\mathbb{C}^{N_{sc}}$ |
| `σ²` | 噪声方差 |
| `LLR` | 对数似然比，$L(b)=\ln\frac{P(b=0\mid y,H)}{P(b=1\mid y,H)}$ |
| `patch` | tokenizer 输出的最小序列单元 |
| `h_k` | 第 k 个子载波对应 patch 的 channel embedding（64 维） |

---

## 2. 系统模型与任务定义

### 2.1 3GPP OFDM 系统模型

采用单用户（或单流）下行 OFDM 链路，物理层参数对齐 3GPP NR（TS 38.211）：

- 子载波间隔（SCS）：30 kHz（`mu=1`）
- 每 PRB 12 子载波；100 MHz 带宽对应 273 PRB = **3276 子载波**（与需求上限一致）
- 调制方式：QPSK / 16QAM / 64QAM / 256QAM（每符号 2/4/6/8 比特）
- 基站天线：8（如 8T8R）；UE 天线：1
- 参考信号：DM-RS（用于信道估计，位置按 3GPP 规范）
- 信道编码：LDPC（5G NR 数据信道，仅用于端到端 BLER 评估）

接收端频域模型（对第 k 个子载波，单流）：

$$
\mathbf{y}_k = \mathbf{h}_k x_k + \mathbf{n}_k,\quad \mathbf{h}_k\in\mathbb{C}^{N_{ant}},\ \mathbf{n}_k\sim\mathcal{CN}(0,\sigma^2\mathbf{I})
$$

其中 $\mathbf{h}_k = H[:,k]$ 是第 k 子载波上的 8 天线空间向量。

均衡（MMSE 或 ZF）后得到标量软符号：

$$
z_k = \mathbf{w}_k^H \mathbf{y}_k,\quad \mathbf{w}_k = \left(\mathbf{h}_k\mathbf{h}_k^H + \sigma^2\mathbf{I}\right)^{-1}\mathbf{h}_k \ \text{(MMSE)}
$$

### 2.2 LLR 定义与软解调

对第 k 子载波第 i 比特（$i=0,\dots,\log_2 M-1$），LLR 定义为：

$$
L(b_i)=\ln\frac{\sum_{x\in\mathcal{X}_i^0}p(y_k\mid x,\mathbf{h}_k)}{\sum_{x\in\mathcal{X}_i^1}p(y_k\mid x,\mathbf{h}_k)}
$$

其中 $\mathcal{X}_i^b$ 为第 i 比特取值为 b 的星座点子集。**max-log 近似**（工程常用、也是本项目的监督标签）：

$$
L(b_i)\approx\frac{1}{\sigma^2}\left[\min_{x\in\mathcal{X}_i^1}|y_k-\mathbf{h}_k x|^2-\min_{x\in\mathcal{X}_i^0}|y_k-\mathbf{h}_k x|^2\right]
$$

### 2.3 任务形式化

**输入**：

- 信道估计矩阵 $\mathbf{H}\in\mathbb{C}^{8\times N_{sc}}$（由 DM-RS 估计得到，可含估计误差）
- 均衡软符号 $\mathbf{z}\in\mathbb{C}^{N_{sc}}$
- 噪声方差 $\sigma^2$（或等效均衡后噪声方差 $\sigma_z^2$）

**输出**：

- 逐子载波逐比特 LLR 估计 $\hat{L}\in\mathbb{R}^{N_{sc}\times\log_2 M}$

**学习目标**：最小化 $\hat{L}$ 与 max-log 参考 LLR $L^{ref}$ 之间的回归误差（MSE），或在端到端意义下最大化译码成功率（BLER）。

**为什么 LWM 有用**：传统软解调假设信道估计**完美**。当信道估计存在误差、存在残余干扰或低精度接收时，LLR 质量下降。LWM 能从信道矩阵中学习**空间/频率相关性先验**，对噪声信道估计进行"隐式去噪/增强"，从而输出更可靠的 LLR。这是"软解调增强"的核心动机。

---

## 3. 数据准备与生成方案

### 3.1 数据生成工具

推荐使用 **[Sionna](https://nvlabs.github.io/sionna/)**（NVIDIA 开源的物理层仿真库，原生支持 TensorFlow，并有 3GPP NR 模块），理由：

- 内置 3GPP 信道模型（CDL-A/B/C/D/E、TDL），符合 TR 38.901
- 内置 OFDM / 5G NR 资源网格、DM-RS 生成
- 内置 `Mapper` / `Demapper`，可直接计算参考 LLR（`"app"` 精确 / `"maxlog"` 近似）
- 支持 GPU 加速批量仿真

> 备选：Matlab 5G Toolbox / Communication Toolbox（若团队更熟悉 Matlab）。本方案以 Sionna 为主，数据格式与工具解耦。

### 3.2 系统参数配置（生成侧）

| 参数 | 取值 | 说明 |
|------|------|------|
| SCS | 30 kHz | NR `mu=1` |
| FFT 大小 | 4096 | 覆盖 3276 子载波 |
| 有效子载波数 `N_sc` | 4 ~ 3276 均匀/随机采样 | 覆盖小调度到全带宽 |
| PRB 数 | `ceil(N_sc/12)` | 对齐资源网格 |
| 基站天线 `N_ant` | 8 | |
| UE 天线 | 1 | |
| 调制阶数 `M` | 4/16/64/256 | 覆盖不同 SNR 工作区 |
| 信道模型 | CDL-C / TDL-C | 可选 UMi/UMa 射线追踪 |
| 多普勒/时延扩展 | 按 CDL 默认 | 可加移动性 |
| 信道估计 | LS / 理想 + 加噪 | 引入估计误差以体现 LWM 价值 |
| SNR 范围 | -5 ~ 30 dB | 对数均匀采样 |
| 噪声 | AWGN | 复高斯 |

### 3.3 数据生成流程

```
1. 配置系统参数（N_sc, M, SNR, 信道模型, 是否含估计误差）
2. 生成频域资源网格：数据符号 + DM-RS
3. 生成信道 H ∈ C^(N_ant × N_sc)（CDL 模型 + 时频采样）
4. OFDM 调制 → 过信道 → 加噪声 → OFDM 解调
5. 信道估计 Ĥ（LS：基于 DM-RS；或理想 H 叠加估计噪声）
6. 均衡得到软符号 z（MMSE，用 Ĥ 与 σ²）
7. 用理想 H 与 σ² 计算参考 LLR（Sionna Demapper "maxlog"）
8. 打包存储（见 3.4）
```

### 3.4 数据格式规范（核心交付）

统一使用 **HDF5（`.h5`）** 存储，按字段组织；支持大规模数据与随机访问。单样本 schema 如下：

```text
数据集文件: dataset_<split>.h5
├── attrs:
│     n_ant          : 8
│     scs_khz        : 30
│     fft_size       : 4096
│     mod_order      : [4, 16, 64, 256]  (本文件包含的调制阶数)
│     channel_model  : "CDL-C"
│     max_llr        : 20.0             (LLR 裁剪上限)
│
├── 每样本组 sample_000000/
│     ├── H_est       : complex64  (8, N_sc)     # 信道估计（含误差，模型输入）
│     ├── H_true      : complex64  (8, N_sc)     # 真实信道（仅评估用，可选）
│     ├── z           : complex64  (N_sc,)       # 均衡软符号（模型输入）
│     ├── sigma2      : float32    scalar        # 噪声方差（模型输入）
│     ├── sigma2_eq   : float32    (N_sc,)       # 均衡后等效噪声方差（可选）
│     ├── llr_ref     : float32    (N_sc, log2M)# 参考 LLR（监督标签）
│     ├── bits_tx     : uint8      (N_sc, log2M)# 发送比特（评估用）
│     ├── mod_order   : int32      scalar        # 本样本调制阶数
│     └── n_sc        : int32      scalar        # 本样本子载波数
```

**约定**：

- `z`、`H_est` 均按**星座平均能量归一化**（对 QAM，令 $E[|x|^2]=1$）。
- `llr_ref` 已做裁剪：$\text{clip}(L, -\text{max\_llr}, +\text{max\_llr})$，避免训练时离群值。
- 同一 HDF5 文件内样本的 `n_sc` 可变；训练时按 batch 内最大 `n_sc` 做 padding（见第 4 节）。
- 推荐用 `h5py` 写入，`torch.utils.data.Dataset` 惰性读取（避免一次性载入内存）。

**数据集规模建议**：

| 用途 | 样本数 | 说明 |
|------|--------|------|
| 继续预训练（阶段 1） | ≥ 100k | 只需 `H_est`/`H_true`，无标签，越多越好 |
| LLR 微调（阶段 2） | ≥ 200k | 需要 `(H_est, z, σ², llr_ref)` |
| 验证集 | 20k | 与训练分布独立（不同信道实现/SNR 种子） |
| 测试集 | 20k | 含**未见过的**调制阶数组合与信道场景 |

> 数据生成是最大的工程量。建议先跑通小规模（如 5k）端到端管道，再批量扩到百万级。

---

## 4. LWM 输入适配（Tokenizer 重新设计）

### 4.1 原生 LWM 数据格式回顾

原生 LWM 的 tokenizer（`input_preprocess.py`）处理 **32 天线 × 32 子载波**复数信道：

1. `flatten`：$\mathbf{H}\in\mathbb{C}^{32\times 32}\to\mathbb{C}^{1024}$（C 序，即逐天线遍历其 32 子载波）
2. 实虚分离拼接：$\mathbb{C}^{1024}\to\mathbb{R}^{2048}=[\Re(\cdot),\Im(\cdot)]$
3. patch 化：按 16 维切分 → **128 个 patch × 16 维**（每个 patch = 8 个复数值）
4. 加 CLS token → 输入序列 `(129, 16)`
5. MCM 掩码：mask 15% 的 patch（实虚部成对），模型预测被 mask 的 patch

**关键事实**：原生 patch 是**无物理语义的固定长度切分**，且维度（32×32、MAX_LEN=129、element_length=16）被硬编码在 `lwm_model.py` 与 `input_preprocess.py` 中。

### 4.2 8 天线 × 可变子载波的适配挑战

| 维度 | 原生 LWM | 本项目 | 冲突 |
|------|----------|--------|------|
| 天线数 | 32 | 8 | patch 语义需重定义 |
| 子载波数 | 32（固定） | 4~3276（可变） | 序列长度远超 MAX_LEN=129 |
| 序列长度 | 129 | 5~3277 | 位置编码、attention 复杂度 |

因此**必须重新设计 tokenizer**，但**保留 LWM 骨干的权重价值**（即 64 维 embedding、12 层 Transformer 可复用，仅输入投影层与位置编码需重新适配）。

### 4.3 子载波对齐 patch 方案（推荐）

重新定义 patch 为**「单个子载波上的 8 天线空间向量」**，物理语义清晰：

$$
\text{patch}_k = \big[\Re(\mathbf{h}_k),\ \Im(\mathbf{h}_k)\big]\in\mathbb{R}^{16},\quad \mathbf{h}_k = H[:,k]\in\mathbb{C}^{8}
$$

- **element_length = 16**：8 实部 + 8 虚部，**与原生 LWM 的 embedding 投影维度完全一致**，可直接复用预训练 embedding 权重（`Embedding.proj` 是 16→64 的线性层）。
- **序列长度 = N_sc**：每子载波一个 patch，序列语义为「沿频率方向的 8 天线空间签名序列」。
- **位置编码**：子载波索引天然有序，位置编码即频率位置信息（沿用正弦/学习式位置编码均可，推荐**学习式**并在继续预训练中微调）。

> 为什么这样切分最合适：OFDM 各子载波在无 ICI 时正交，子载波间的信息交互主要在**信道频率相关性**（同一信道的相邻子载波高度相关）。把 patch 对齐到子载波，让 Transformer 的注意力天然建模「频率方向的空间相关性」，且天然支持逐子载波输出 LLR（每个 patch 位置对应一个子载波），与下游 decoder 无缝衔接。

### 4.4 可变长度处理

针对 `N_sc ∈ [4, 3276]` 的三种情形：

**方案 A：固定窗口 + padding（首选，最简单可靠）**

- 设定 `MAX_LEN = 256`（或 512，权衡显存与覆盖）
- `N_sc ≤ MAX_LEN`：直接作为序列，右侧 pad 到 MAX_LEN，配合 **attention mask** 忽略 padding
- `N_sc > MAX_LEN`：按 `MAX_LEN` 滑动切块，块间重叠 `overlap = 16` 子载波，每块独立过 LWM，输出按原位置拼回（重叠区取平均）

**方案 B：分块 + 无 padding（适合大子载波）**

- 固定块大小 128 子载波，非重叠切块（LLR 逐子载波独立，块边界几乎无损）
- 每块加 CLS，得到 `(129, 16)`，与原生 LWM 序列长度一致，**直接复用原生位置编码**

**方案 C：可变序列 + 位置编码插值（进阶）**

- 动态序列长度，对学习式位置编码做**线性插值**以适配任意 N_sc
- 实现复杂，仅当追求极致效率时采用

> **推荐**：继续预训练与微调阶段统一采用 **方案 B（128 子载波分块）**——它与原生 LWM 的 MAX_LEN=129 完全一致，复用度最高、实现最简单；对 4 子载波等超短序列则 pad 到 128。若你的典型场景是整带宽 3276 子载波一次处理，再评估方案 A 的大窗口版本。

### 4.5 归一化策略

原生 LWM 对 DeepMIMO 信道乘了 `1e6`（因射线追踪信道幅度极小）。3GPP CDL 信道幅度为 0 dB 量级，需重新设计归一化：

1. **逐样本 Frobenius 归一化**（推荐）：$\hat{\mathbf{H}} = \mathbf{H} / \|\mathbf{H}\|_F$，消除路径损耗与发射功率差异，让模型专注空间/频率**形状**特征。
2. **全局统计标准化**：训练集统计 $\mu,\sigma$，对 `H_est`、`z` 分别做 z-score；推理时复用训练集统计量（须持久化保存）。
3. **LLR 标签不归一化**，但做 ±20 裁剪。

> 注意：归一化会丢失绝对功率信息，而 LLR 与 σ² 强相关。因此 **σ² 作为显式输入**喂给 decoder（见第 6 节），补偿被归一化掉的功率信息。

---

## 5. 继续训练方案（MCM 域适配）

### 5.1 继续训练目标

LWM 在 DeepMIMO 射线追踪信道（32 天线、3.5 GHz）上预训练。你的 3GPP CDL 信道在**统计分布、天线数、频率相关性**上均有差异。继续预训练的目的是**领域自适应**：让 LWM 的隐空间适应 3GPP OFDM 信道分布。

### 5.2 MCM 掩码策略

沿用 LWM 原生 MCM，但适配新 tokenizer：

- **掩码比例**：15% 的 patch（与原生一致）
- **实虚成对掩码**：由于 patch 已把实虚部打包在同一 patch 内（patch_k = [Re, Im]），对 patch 的掩码天然成对，无需额外处理
- **掩码方式**：80% 置为 `[MASK]` 向量（如 0.1 向量）、10% 随机向量、10% 保持原样（沿用 BERT 掩码技巧，可简化）
- **CLS 位置**：不掩码
- **预测头**：复用原生 `lwm.decoder`（`Linear(64→16)` + bias），输出被掩码 patch 的 16 维重建值

### 5.3 训练配置

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| 损失 | MSE（重建 patch） | 与原生一致，可选除以其方差归一化 |
| 优化器 | AdamW | |
| 学习率 | `1e-5`（继续预训练） | 远小于从头训练 `1e-4`，避免灾难性遗忘 |
| weight decay | `1e-5` | |
| 调度器 | CosineAnnealing 或 ReduceLROnPlateau | |
| batch size | 256（CPU）/ 512（GPU） | 分块后每块 128 patch，显存很小 |
| epochs | 10~50 | 以验证 MCM loss 收敛为准 |
| 早停 | patience=5 | 监控验证 loss |
| 混合精度 | 可选（AMP） | CPU 上可省略 |

### 5.4 冻结策略讨论

| 策略 | 适用 | 优缺点 |
|------|------|--------|
| **全参数微调**（推荐） | 数据量充足（≥100k） | 适配最充分；需小 lr 防遗忘 |
| 冻结 embedding + 底层，只训高层 | 数据较少 | 保留底层通用特征，减少过拟合 |
| 冻结全部，仅训 decoder | 数据极少（<10k） | 最快；但域差异未消除 |

> 建议默认**全参数微调 + 小学习率**，并在验证集上对比冻结策略，选最优。

---

## 6. Decoder 设计（LLR 预测头）

### 6.1 输入输出定义

- **输入**：对每个子载波 k，拼接
  - channel embedding $\mathbf{h}_k^{emb}\in\mathbb{R}^{64}$（LWM 第 k 个 patch 的隐状态）
  - 均衡软符号 $\Re(z_k),\Im(z_k)\in\mathbb{R}^{2}$
  - 噪声方差 $\sigma^2$（或 $\sigma^2_{eq,k}$）
  - 可选：调制阶数 one-hot（支持多阶调制混合训练）
- **输出**：$\hat{L}_k\in\mathbb{R}^{\log_2 M}$（该子载波 $\log_2 M$ 个比特的 LLR）

### 6.2 方案 A：逐子载波 MLP（"简单 decoder"首选）

```text
输入 (64 + 2 + 1 + 4) = 71 维  [h_k ; Re(z_k) ; Im(z_k) ; σ² ; mod_onehot]
   ↓
Linear(71 → 128) + GELU
   ↓
Linear(128 → 64) + GELU
   ↓
Linear(64 → log2M)            # 直接回归 LLR（无激活或 tanh·max_llr）
```

- 逐子载波**共享权重**（1D 卷积 view 或循环展开），参数量约 ~20k，符合"简单 decoder"要求。
- 天然支持可变 `N_sc` 与不同调制阶数（输出维度由 `mod_order` 决定）。

### 6.3 方案 B：轻量 1D-CNN（复用官方 res1dcnn 思想）

若需要建模**子载波间相关性**，在 channel embedding 序列上先做轻量时序建模，再逐子载波输出：

```text
channel embedding 序列 (T, 64)
   ↓
1D 残差 CNN（kernel=3, 通道 64→64，2 个残差块）   # 沿频率方向
   ↓
[refined_h_k ; Re(z_k); Im(z_k); σ²]  → 逐子载波 MLP → LLR
```

- 参考官方 `utils/res1dcnn.py` 的 ResidualBlock 设计，但大幅精简（2 层而非 3 层）。
- 适用：大带宽、信道频率选择性强的场景。

### 6.4 损失函数设计

| 方案 | 形式 | 适用 |
|------|------|------|
| **MSE 回归**（推荐） | $\mathcal{L}=\frac{1}{N_{sc}\log_2 M}\sum\|\hat L - L^{ref}\|^2$ | 直接对齐 max-log LLR，训练稳定 |
| 加权 MSE | 对 $\|L^{ref}\|$ 小的比特降权 | 减少高置信比特主导 |
| BCE（sigmoid + 真实比特） | $\mathcal{L}=-\sum b\log\hat p+(1-b)\log(1-\hat p)$ | 端到端语义，等价于 LLR 的 sigmoid 表示 |

> 推荐从 **MSE 回归到 max-log LLR** 起步，简单稳定；后续可尝试 BCE（用真实发送比特作标签，输出经 `sigmoid` 映射后取 log 得到 LLR），更贴合译码器需求。

### 6.5 与 LWM 的拼接方式

- **阶段 2 中 LWM 是否冻结**：推荐**小学习率微调**（`1e-6~1e-5`），让骨干与 decoder 协同适应 LLR 任务；数据少时冻结。
- **梯度回传**：LWM 输出 `output`（含 CLS 与全部 patch 的隐状态），取 `output[:, 1:, :]` 即逐子载波 embedding（分块时需按位置拼回）。
- **CLS embedding** 可用于可选的整体特征（如 SNR 感知），但 LLR 预测主要依赖逐 patch 的 channel embedding。

---

## 7. 两阶段训练流程

### 7.1 阶段 1：继续预训练（MCM，无监督）

```
输入：H_est 分块 → tokenizer → (B, 129, 16)
流程：
  for epoch in 1..E1:
      mask 15% patches
      logits_lm, _ = lwm(input_ids, masked_pos)
      loss = MSE(logits_lm, masked_patch_gt)
      loss.backward(); optimizer.step()
保存：lwm_continued.pth（骨干 + 预测头）
```

### 7.2 阶段 2：LLR 微调（有监督）

```
输入：(H_est, z, σ²) + 标签 llr_ref
流程：
  for epoch in 1..E2:
      h = lwm.encode(H_est)            # (B, T, 64)，冻结或小 lr
      z_cat = concat(h, Re(z), Im(z), σ²)
      llr_hat = decoder(z_cat)          # (B, T, log2M)
      loss = MSE(llr_hat, llr_ref)
      loss.backward(); optimizer.step()
保存：lwm_llr_full.pt（骨干 + decoder）
```

### 7.3 训练脚本结构建议

```text
project/
├── data/                     # HDF5 数据集
├── src/
│   ├── config.py             # 全局配置（系统参数、超参数）
│   ├── generate_data.py      # Sionna 数据生成
│   ├── dataset.py            # HDF5 Dataset + collate（分块/padding）
│   ├── tokenizer.py          # 子载波对齐 tokenizer（第 4 节）
│   ├── lwm_backbone.py       # 复用/改造 lwm_model.py
│   ├── decoder.py            # LLR 预测头（第 6 节）
│   ├── train_pretrain.py     # 阶段 1 MCM
│   ├── train_llr.py          # 阶段 2 微调
│   └── evaluate.py           # 评估（第 8 节）
└── weights/                  # 模型权重
```

---

## 8. 评估方案

### 8.1 指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **LLR MSE** | $\mathbb{E}\|\hat L - L^{ref}\|^2$ | 直接度量回归精度 |
| **NMI** | 预测 LLR 与发送比特的互信息（归一化） | 度量软信息质量 |
| **BER / BLER** | 将 $\hat L$ 送入 LDPC 译码，测误码率/误块率 | 端到端最终指标（最重要） |
| **AUC（比特判决）** | $\text{sign}(\hat L)$ 与真实比特的 AUC | 硬判决视角 |

### 8.2 基线对比（必须包含）

1. **传统软解调（基准）**：MMSE 均衡 + max-log demapper（用 `H_est` 与 σ² 计算），即 `llr_ref` 的生成路径本身。
2. **LWM（继续预训练后）+ decoder**：本项目方案。
3. **LWM（未继续预训练，原始权重）+ decoder**：隔离"继续预训练"的贡献。
4. **Raw H + decoder**：不用 LWM，直接把 `H_est` 展平接同样 decoder，隔离"LWM 特征提取"的贡献。
5. **理想软解调（上界）**：用真实 H（而非 H_est）计算 LLR。

### 8.3 消融实验建议

- 冻结 vs 微调骨干
- MLP decoder vs 1D-CNN decoder
- 有无 σ² 输入、有无调制 one-hot
- 信道估计误差大小对收益的影响（LWM 价值应在估计误差大时更明显）
- 不同调制阶数、不同 SNR 区间的分档表现

---

## 9. 工程实施步骤与里程碑

| 阶段 | 任务 | 交付物 | 预估 |
|------|------|--------|------|
| M1 | 数据管道 | `generate_data.py` + 5k 样本 HDF5 | 1 周 |
| M2 | Tokenizer 适配 | `tokenizer.py`，分块/padding 通过单测 | 3 天 |
| M3 | 继续预训练 | `train_pretrain.py` + `lwm_continued.pth` | 1 周（含调参） |
| M4 | Decoder + 微调 | `decoder.py` + `train_llr.py` | 1 周 |
| M5 | 评估 + 基线 | `evaluate.py`，BER/BLER 对比报告 | 1 周 |
| M6 | 大规模数据 + 调优 | 百万级数据、超参搜索 | 2~4 周 |

---

## 10. 风险与注意事项

1. **维度硬编码**：原生 `lwm_model.py` 中 `MAX_LEN=129`、`element_length=16` 是模块常量，改造 tokenizer 时需同步核对位置编码长度；分块方案（块=128）可完全规避该改动。
2. **归一化与 σ² 一致性**：归一化会改变功率尺度，务必把 σ² 作为 decoder 显式输入，否则 LLR 幅值会系统性偏置。
3. **灾难性遗忘**：继续预训练 lr 过高会破坏预训练特征；建议 `1e-5` 起步，监控阶段 2 验证指标。
4. **计算资源**：当前机器 GPU 处于 requires-reset 状态、仅 CPU 可用；LWM 极小（0.6M 参数），CPU 也能跑，但百万级数据生成与训练建议恢复 GPU。
5. **可变长度效率**：分块会放大 batch 中的样本数，注意 `collate_fn` 的正确 padding 与 mask。
6. **LLR 范围**：LLR 动态范围大，务必裁剪（±20）并做梯度裁剪（`clip_grad_norm=1.0`）。
7. **3GPP 合规性**：若最终用于标准评测，需严格对齐 NR 帧结构、DM-RS 位置、LDPC 码率；本方案仅需"兼容 3GPP 的 OFDM 调制"，可先简化。

---

## 11. 附录：代码骨架

### A. Tokenizer（子载波对齐 patch + 分块）

```python
import numpy as np

ELEMENT_LENGTH = 16   # 8 天线复数 = 16 实数
BLOCK_SIZE = 128      # 分块大小（子载波数），对应 MAX_LEN=129

def channel_to_patches(H):
    """H: (N_ant=8, N_sc) 复数 -> (N_sc, 16) 实数 patch 序列"""
    H = np.asarray(H)                       # (8, N_sc)
    real = H.real.T                         # (N_sc, 8)
    imag = H.imag.T
    return np.concatenate([real, imag], axis=1)   # (N_sc, 16)

def tokenize(H, norm=True):
    """H -> list of blocks, 每块 (129, 16)，含 CLS token"""
    if norm:
        H = H / (np.linalg.norm(H) + 1e-9)  # Frobenius 归一化
    patches = channel_to_patches(H)         # (N_sc, 16)
    N_sc = patches.shape[0]
    blocks = []
    for start in range(0, N_sc, BLOCK_SIZE):
        blk = patches[start:start+BLOCK_SIZE]
        if blk.shape[0] < BLOCK_SIZE:       # 尾块 padding
            pad = np.zeros((BLOCK_SIZE - blk.shape[0], ELEMENT_LENGTH), dtype=blk.dtype)
            blk = np.concatenate([blk, pad], axis=0)
        cls = 0.2 * np.ones((1, ELEMENT_LENGTH), dtype=blk.dtype)  # 与原生一致
        blocks.append(np.concatenate([cls, blk], axis=0))           # (129, 16)
    return blocks
```

### B. Decoder（逐子载波 MLP）

```python
import torch
import torch.nn as nn

class LLRDecoder(nn.Module):
    def __init__(self, d_emb=64, max_bits=8, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_emb + 2 + 1 + 4, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, max_bits),          # 输出 log2M 个 LLR
        )

    def forward(self, h_emb, z, sigma2, mod_onehot):
        """
        h_emb: (B, T, 64)  channel embedding（逐子载波）
        z    : (B, T)      复数软符号
        sigma2: (B, 1)     噪声方差
        mod_onehot: (B, T, 4)
        """
        z_re, z_im = z.real.unsqueeze(-1), z.imag.unsqueeze(-1)
        s2 = sigma2.unsqueeze(-1).expand(-1, z_re.shape[1], 1)
        x = torch.cat([h_emb, z_re, z_im, s2, mod_onehot], dim=-1)  # (B, T, 71)
        return self.mlp(x)                    # (B, T, max_bits)
```

### C. 阶段 1 继续预训练（MCM 核心循环）

```python
# 复用 lwm_model.py 的 lwm 类；device 视 CUDA 可用性选择
model = lwm.from_pretrained('model_weights.pth', device='cpu')  # 加载官方权重
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

for epoch in range(epochs):
    for H_batch in dataloader:                 # H_batch: (B, 8, N_sc)
        blocks = [tokenize(H) for H in H_batch]  # 每样本 -> 若干 (129,16)
        # 组装 batch、构造 masked_pos（15% 随机）、masked_tokens(ground truth)
        input_ids, masked_tokens, masked_pos = make_mcm_batch(blocks)
        logits_lm, _ = model(input_ids, masked_pos)
        loss = nn.functional.mse_loss(logits_lm, masked_tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
```

### D. Sionna 参考 LLR 计算

```python
# Sionna 计算参考 LLR（阶段 2 标签 & 评估基线）
from sionna.phy.mapping import Mapper, Demapper

mapper = Mapper("qam", num_bits_per_symbol=log2M)
demapper = Demapper("maxlog", "qam", num_bits_per_symbol=log2M)
# y: (..., N_ant) 接收, h: (..., N_ant) 信道, no: 噪声方差
llr_ref = demapper([y, h, no])       # 直接对多天线接收计算 LLR
```

> 说明：以上为骨架示意，完整实现需与 Sionna 版本（0.18/0.19 与 v2 API 有差异）对齐，并按第 3 节的数据 schema 落地。

---

*文档完。如需我把第 4~6 节的 tokenizer / decoder / 训练脚本直接实现为可运行代码，或先搭建 M1 数据生成管道，请告知。*
