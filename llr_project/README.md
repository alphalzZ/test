# LWM-LLR：基于大无线模型（LWM）的无线信道 LLR 预测

将 [LWM（Large Wireless Model，0.6M Transformer）](https://github.com/wi-lab/lwm) 与 CNN 残差 LLR
解码器结合，学习 5G NR PUSCH 数据 RE 的逐比特 LLR，在多种系统配置（接收天线 1/2/4/8、
带宽 1~10 RB、OFDM 符号 3~14、DMRS {1}/{1+1}/{1+2}、TDL-A/B/C/D、多普勒 0/5/30 m/s）下
**一个模型适配所有配置**，并显著优于 LS 信道估计 + MMSE 均衡的传统基线
（高 SNR 段 BER 降低 90%+）。

- 数据：Sionna 2.x（PyTorch 后端）标准 5G NR PUSCH 链路仿真，固定带宽（1024-FFT）嵌入
- 训练：MCM 继续预训练（可选）→ LLR 两阶段微调（冻结骨干训 decoder → 联合微调，BCE 损失）
- 参考实现：Sionna 官方 [Neural Receiver 教程](notebooks/Neural_Receiver.ipynb)

## 目录结构

```
llr_project/
├── configs/                 # JSON 配置（仿真/模型/训练/实验/路径；LLR_CFG=night 覆盖）
├── data/                    # Sionna 生成的数据缓存（分片 pkl，git 忽略）
├── src/                     # 源代码
│   ├── models/              # LWM 骨干（lwm.py）、CNN LLR 解码器（llr_decoder.py）、
│   │                        #   组合模型 LWMLLR（lwm_llr.py）
│   ├── datasets/            # 数据生成（Sionna PUSCH）、QAM/demap 工具、tokenizer、
│   │                        #   分片缓存与桶式加载（loader.py）
│   ├── trainers/            # 阶段1 MCM 预训练（pretrain.py）、阶段2 LLR 微调（train_llr.py）
│   ├── simulation/          # Sionna PUSCH 链路仿真（pusch.py）
│   ├── evaluation/          # 性能评估（evaluate.py）、NeuralReceiver 风格实验（experiment.py）、
│   │                        #   结果分析（analyze.py）
│   └── utils/               # 配置加载器（config.py，读 configs/*.json）
├── scripts/                 # 顶层运行脚本（train_all.sh 白天 / train_night.sh 夜间大规模）
├── tests/                   # 单元测试（unittest）
├── notebooks/               # Jupyter Notebook（Sionna Neural Receiver 教程参考）
├── experiments/             # 实验输出
│   ├── logs/                # 训练/仿真日志
│   ├── checkpoints/         # 模型权重（*.pt，git 忽略）
│   └── results/             # 评估结果（eval_results.json、*.png 曲线）
└── docs/                    # 项目文档（README.md 完整版、REPORT.md 设计报告）
```

## 快速开始

```bash
# 一键全流程（数据生成 -> MCM 预训练 -> LLR 微调 主+对照 -> 评估，默认小规模配置）
./scripts/train_all.sh

# 分步（在项目根目录）
python -m src.trainers.pretrain                 # 阶段1：MCM 继续预训练
python -m src.trainers.train_llr                # 阶段2：LLR 微调（主，继续预训练权重）
python -m src.trainers.train_llr --no-pretrain  # 阶段2：对照（官方权重）
python -m src.evaluation.evaluate               # 多配置性能评估
python -m src.evaluation.experiment --per-snr 32   # NeuralReceiver 风格实验（即时生成数据）
python -m src.evaluation.analyze                # 评估结果分析

# 夜间大规模训练（约 2.8 小时，配置 configs/night.json）
./scripts/train_night.sh
```

## 文档

- [完整 README（设计、方法、结果、FAQ）](docs/README.md)
- [设计报告（二次开发设计与优化历程）](docs/REPORT.md)
