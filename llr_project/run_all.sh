#!/usr/bin/env bash
# LWM LLR 预测 —— 一键运行：数据生成 -> 继续预训练 -> LLR 微调(主+对照, BCE+CNN) -> 评估
# 用法: ./run_all.sh   (GPU 大规模约 2 小时；CPU 小规模约 2 小时)
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "========== [1/5] 阶段1: MCM 继续预训练 =========="
$PY train_pretrain.py

echo "========== [2/5] 阶段2: LLR 微调（继续预训练权重） =========="
$PY train_llr.py

echo "========== [3/5] 阶段2: LLR 微调（对照：官方权重，无继续预训练） =========="
$PY train_llr.py --no-pretrain

echo "========== [4/5] 性能评估 =========="
$PY evaluate.py

echo "========== [5/5] 完成 =========="
