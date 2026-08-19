#!/usr/bin/env bash
# =============================================================================
# LWM LLR 预测（多配置自适应版）—— 一键运行（默认小规模配置）：
#   数据生成（多配置混合: 天线1/2/4/8 × RB1~10 × 符号3~14 × DMRS × TDL/多普勒）
#   -> 继续预训练 -> LLR 微调(主+对照, BCE+CNN) -> 评估
# 用法: ./scripts/train_all.sh   （GPU 全流程约 30 分钟；CPU 小规模约 2 小时）
# 配置: configs/*.json（默认小规模；夜间大规模方案用 ./scripts/train_night.sh）
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # 项目根
PY=/home/le-lei/workspace/test/.venv/bin/python

echo "========== [1/5] 阶段1: MCM 继续预训练 =========="
$PY -m src.trainers.pretrain

echo "========== [2/5] 阶段2: LLR 微调（继续预训练权重） =========="
$PY -m src.trainers.train_llr

echo "========== [3/5] 阶段2: LLR 微调（对照：官方权重，无继续预训练） =========="
$PY -m src.trainers.train_llr --no-pretrain

echo "========== [4/5] 性能评估 =========="
$PY -m src.evaluation.evaluate

echo "========== [5/5] 完成 =========="
echo "   - 权重: experiments/checkpoints/"
echo "   - 评估: experiments/results/（eval_results.json + eval_ber_curves.png）"
