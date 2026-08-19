#!/usr/bin/env bash
# =============================================================================
# 夜间大规模训练方案 v2（约 5 小时，实测基准外推）
#   数据生成（分片缓存，断点续跑，sub_batch=2 控制内存峰值）-> 两阶段微调
#   （官方权重，14 冻结 + 8 联合）-> 评估
# v2 改动（基于 v1 性能分析）：砍掉 MCM 预训练（收益≈噪声级）+ 砍训练轮数 48→22
# v3 改动（OOM 修复）：Sionna TDL 采样瞬时峰值 9~15GB 导致 15GB 机器被杀；
#   组内生成拆小批量（SUB_BATCH=2，峰值降至 ~3GB）+ 分片参数指纹（参数变更
#   自动重生成）+ 每片后归还 glibc 内存。
# 用法: ./scripts/train_night.sh   （可重复执行：已完成步骤自动跳过）
# 配置: configs/night.json（通过环境变量 LLR_CFG=night 加载，见 src/utils/config.py）
# 产物: experiments/checkpoints/lwm_llr_night.pt,
#       data/pusch_night_{train,val}.pkl.*, experiments/results/ 等
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # 项目根
PY=/home/le-lei/workspace/test/.venv/bin/python
W=experiments/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LLR_CFG=night              # 加载 configs/night.json

echo "========== [0/4] 配置确认 =========="
$PY -c "
from src.utils import config
print(f'  USE_PRETRAIN={config.USE_PRETRAIN}  TRAIN_N={config.TRAIN_N}  VAL_N={config.VAL_N}')
print(f'  FT_EPOCHS={config.FT_EPOCHS} (freeze {config.FT_FREEZE_EPOCHS}+joint)')
print(f'  ckpt: {config.CKPT_LLR}')
print(f'  cache: {config.CACHE_TRAIN}')
"

echo "========== [1/4] 数据生成（分片缓存，断点续跑） =========="
$PY -c "
from src.datasets.loader import build_data
from src.utils import config
build_data(config.TRAIN_N, config.VAL_N, config.PT_N, config.SEED,
           config.CACHE_TRAIN, config.CACHE_VAL, config.CACHE_PT)
print('  数据全部就绪')
"

echo "========== [2/4] LLR 两阶段微调（官方权重，22 epoch，约 1.3 小时） =========="
[ -f "$W/lwm_llr_night.pt" ] || $PY -m src.trainers.train_llr

echo "========== [3/4] 性能评估（96 样本，约 1 分钟） =========="
$PY -m src.evaluation.evaluate

echo "=============================================================="
echo " 夜间训练全部完成！结果:"
echo "   - 权重: $W/lwm_llr_night.pt"
echo "   - 评估: experiments/results/eval_results.json + eval_ber_curves.png"
echo "=============================================================="
