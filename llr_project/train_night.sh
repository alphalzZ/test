#!/usr/bin/env bash
# =============================================================================
# 夜间大规模训练方案 v2（约 2.8 小时，实测基准外推）
#   数据生成（分片缓存，断点续跑）-> 两阶段微调（官方权重，14 冻结 + 8 联合）-> 评估
# v2 改动（基于 v1 性能分析）：砍掉 MCM 预训练（收益≈噪声级）+ 砍训练轮数 48→22
# 用法: ./train_night.sh          （可重复执行：已完成步骤自动跳过）
# 依赖: llr_project/night_config.py（已创建；删除它即恢复白天小规模配置）
# 产物: weights/lwm_llr_night.pt, data/pusch_night_{train,val}.pkl.*, eval_results.json 等
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/le-lei/workspace/test/.venv/bin/python
W=weights
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== [0/4] 配置确认 =========="
$PY -c "
import config
print(f'  USE_PRETRAIN={config.USE_PRETRAIN}  TRAIN_N={config.TRAIN_N}  VAL_N={config.VAL_N}')
print(f'  FT_EPOCHS={config.FT_EPOCHS} (freeze {config.FT_FREEZE_EPOCHS}+joint)')
print(f'  ckpt: {config.CKPT_LLR}')
print(f'  cache: {config.CACHE_TRAIN}')
"

echo "========== [1/4] 数据生成（分片缓存，断点续跑，约 1.4 小时） =========="
$PY -c "
from dataset import build_data
import config
build_data(config.TRAIN_N, config.VAL_N, config.PT_N, config.SEED,
           config.CACHE_TRAIN, config.CACHE_VAL, config.CACHE_PT)
print('  数据全部就绪')
"

echo "========== [2/4] LLR 两阶段微调（官方权重，22 epoch，约 1.3 小时） =========="
[ -f "$W/lwm_llr_night.pt" ] || $PY train_llr.py

echo "========== [3/4] 性能评估（96 样本，约 1 分钟） =========="
$PY evaluate.py

echo "=============================================================="
echo " 夜间训练全部完成！结果:"
echo "   - 权重: $W/lwm_llr_night.pt"
echo "   - 评估: eval_results.json + eval_ber_curves.png"
echo "=============================================================="
