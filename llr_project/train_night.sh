#!/usr/bin/env bash
# =============================================================================
# 夜间大规模训练方案（约 12 小时预算，实测 ~8.5 小时）
#   数据生成（分集缓存，断点续跑）-> MCM 预训练 -> 两阶段微调(主+对照) -> 评估
# 用法: ./train_night.sh          （可重复执行：已完成步骤自动跳过）
# 依赖: llr_project/night_config.py（已创建；删除它即恢复白天小规模配置）
# 产物: weights/*_night.pt, data/pusch_night_*.pkl, eval_ber_curves.png 等
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/le-lei/workspace/test/.venv/bin/python
W=weights
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== [0/5] 配置确认 =========="
$PY -c "
import config
print(f'  PT_N={config.PT_N}  TRAIN_N={config.TRAIN_N}  VAL_N={config.VAL_N}')
print(f'  PT_EPOCHS={config.PT_EPOCHS}  FT_EPOCHS={config.FT_EPOCHS} (freeze {config.FT_FREEZE_EPOCHS}+joint)')
print(f'  ckpt: {config.CKPT_LLR}')
print(f'  cache: {config.CACHE_TRAIN}')
"

echo "========== [1/5] 数据生成（分集缓存，约 1.5~2 小时，一次性） =========="
$PY << 'EOF'
import os
import config
from data_gen_sionna import generate_dataset
from dataset import save_samples_pkl

def gen(name, n, seed, cache):
    if os.path.exists(cache):
        print(f"  [{name}] 缓存已存在: {cache}")
        return
    print(f"  [{name}] 生成 {n} 样本 (seed={seed}, group_size={config.GROUP_SIZE}) ...")
    samples = generate_dataset(n, seed=seed, group_size=config.GROUP_SIZE)
    save_samples_pkl(cache, samples)
    print(f"  [{name}] 完成 -> {cache}")

gen("train", config.TRAIN_N, config.SEED,       config.CACHE_TRAIN)
gen("val",   config.VAL_N,   config.SEED + 1000, config.CACHE_VAL)
gen("pt",    config.PT_N,    config.SEED + 2000, config.CACHE_PT)
print("  数据全部就绪")
EOF

echo "========== [2/5] 阶段1: MCM 继续预训练（约 0.5 小时） =========="
[ -f "$W/lwm_continued_night.pt" ] || $PY train_pretrain.py

echo "========== [3/5] 阶段2: LLR 两阶段微调 - 主模型（约 2.9 小时） =========="
[ -f "$W/lwm_llr_night.pt" ] || $PY train_llr.py

echo "========== [4/5] 阶段2: LLR 两阶段微调 - 对照（官方权重，无预训练；约 2.9 小时） =========="
[ -f "$W/lwm_llr_no_pretrain_night.pt" ] || $PY train_llr.py --no-pretrain

echo "========== [5/5] 性能评估（96 样本，约 1 分钟） =========="
$PY evaluate.py

echo "=============================================================="
echo " 夜间训练全部完成！结果:"
echo "   - 权重: $W/lwm_llr_night.pt (主) / $W/lwm_llr_no_pretrain_night.pt (对照)"
echo "   - 评估: eval_results.json + eval_ber_curves.png"
echo "=============================================================="
