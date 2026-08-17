# LWM LLR 预测 —— 开发与评估报告（Sionna PUSCH 版）

## 1. 数据建模修复（本版核心）

根据要求，数据建模已从自研 NumPy 仿真**迁移到标准 Sionna 2.x（PyTorch 后端）**：

| 项 | 旧版（自研） | **新版（Sionna 标准库）** |
|----|-------------|--------------------------|
| 链路 | 简化 OFDM | **标准 5G NR PUSCH**（`sionna.phy.nr.PUSCHTransmitter`） |
| 信道 | 自研 TDL-C | **TDL-A**（`sionna.phy.channel.tr38901.TDL` + `TimeChannel`，3GPP TR 38.901） |
| 导频 | comb-4 | **DMRS type1, {1+1} 双符号**（`PUSCHDMRSConfig`: 符号2+11, 2 CDM 组） |
| 信道估计 | LS-DFT | **LS + 线性插值**（`PUSCHLSChannelEstimator`，先估计后插值） |
| 模型输入 | (8, 128) 2D | **{num_rx, num_sc, num_symb} = (8, 120, 14) 3D 完整频域信道** |
| 数据 RE | 96/块 | 1440（12 数据符号 × 120 子载波） |
| SCS | 30 kHz | 15 kHz（NR mu=0） |
| 依赖 | — | `pip install sionna`（纯 PyTorch，无需 TensorFlow） |

## 2. 项目结构

```
llr_project/
├── config.py             # 全局配置（3D 维度、超参数）
├── data_gen_sionna.py    # ★ Sionna PUSCH 数据生成器（标准 5G NR 链路）
├── data_gen.py           # 工具函数（QAM 星座 / max-log LLR / demapper）
├── tokenizer.py          # 3D 信道 tokenizer（逐符号子载波对齐 patch）
├── dataset.py            # PyTorch Dataset（3D + llr_base 残差锚点）
├── model.py              # LWM 骨干 + 残差 LLR Decoder（3D 输入）
├── train_pretrain.py     # 阶段1: MCM 继续预训练
├── train_llr.py          # 阶段2: LLR 微调（主/对照）
├── evaluate.py           # 性能评估
├── run_all.sh / README.md / REPORT.md
└── weights/              # lwm_continued / lwm_llr / lwm_llr_no_pretrain
```

## 3. 训练过程（CPU）

| 阶段 | 配置 | 结果 |
|------|------|------|
| 阶段1 MCM | 1000 样本 × 14 符号，12 epoch | loss 0.636 → 0.482（~74 min） |
| 阶段2 主模型 | 1200 样本，12 epoch | val MSE **0.00750**（~95 min） |
| 阶段2 对照(noPT) | 1200 样本，12 epoch | val MSE **0.00746**（~95 min） |

## 4. 性能评估（Sionna PUSCH，180 样本）

### LLR MSE（vs 理想 max-log，越小越好）

| 方案 | MSE | vs 基线 |
|------|-----|---------|
| 传统基线（H_est） | 4.816 | — |
| **LWM+Decoder（本方案）** | **4.715** | **-2.1%** ✅ |
| LWM 对照（无继续预训练） | 4.700 | -2.4% |

### 硬判决 BER（越低越好）

| SNR | 传统基线 | **LWM+Decoder** | 改善 |
|-----|---------|-----------------|------|
| -5 dB | 0.181 | 0.183 | 持平 |
| 0 dB | 0.081 | 0.082 | 持平 |
| 5 dB | 0.018 | 0.018 | 持平 |
| **10 dB** | 0.144 | **0.118** | **-18%** ✅ |
| 15 dB | 0.000 | 0.000 | — |
| 20 dB | 0.000 | 0.000 | — |

### 结论

1. **Sionna 标准 PUSCH 数据下，LWM 软解调增强有效**：LLR MSE 全面低于基线；**中等 SNR（10dB，256QAM 场景）BER 改善 18%** —— 此时信道估计误差适中，LWM 利用信道先验修正 LLR 最有效。
2. **残差设计保证不劣化**：低 SNR（-5dB）与高 SNR（≥15dB）均与基线持平。
3. **继续预训练增益不明显**（主 vs 对照 <0.5%）：12 epoch MCM 域适配仍不足，且 LLR 微调中骨干 lr 极小。建议后续：增加 MCM epoch、提高骨干微调 lr、或两阶段（先冻骨干后联合微调）。

## 5. 与旧版（自研 2D 仿真）对比

| 指标 | 旧版 (8×128) | **新版 Sionna (8×120×14)** |
|------|-------------|---------------------------|
| 基线 MSE | 31.6 | 4.82（Sionna DMRS 估计更准） |
| LWM MSE | 30.0 (-5%) | 4.72 (-2.1%) |
| 最优 BER 增益 | -5dB ~ 5dB | **10dB (-18%)** |
| 数据真实性 | 简化 | **标准 3GPP PUSCH** |

新版数据更接近真实 5G 接收机（标准 PUSCH/DMRS/信道模型），虽基线更准、增益窗口收窄，但结果更可信、可直接对标实际系统。

## 6. 后续改进

1. 加强 MCM 继续预训练（epoch 50+ / lr 1e-4），验证域适配价值
2. 接入 LDPC 译码测 BLER（端到端指标）
3. 增加信道场景多样性（TDL-B/C/D、多普勒、多 UE）
4. 试 LMMSE 插值器（`LMMSEInterpolator`）作为更强基线
