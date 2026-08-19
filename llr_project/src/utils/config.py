# -*- coding: utf-8 -*-
"""
全局配置加载器（src.utils.config）

从 configs/*.json 加载系统/模型/训练/实验/路径参数到模块属性，import 后直接
`config.XXX` 访问（接口与旧版 config.py 完全一致）。

配置分层：
  1. 默认配置：simulation.json + model.json + training.json + experiment.json + paths.json
  2. 可选覆盖：环境变量 `LLR_CFG=<name>` 时加载 configs/<name>.json 覆盖默认值，
     例如 `LLR_CFG=night` 对应夜间大规模训练方案（configs/night.json），
     由 scripts/train_night.sh 设置；不设置时使用默认小规模配置。
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")


def _load(fname):
    with open(os.path.join(CONFIG_DIR, fname), encoding="utf-8") as f:
        return json.load(f)


# ---- 默认配置 ----
for _f in ("simulation.json", "model.json", "training.json",
           "experiment.json", "paths.json"):
    globals().update(_load(_f))

# ---- 可选覆盖（LLR_CFG 环境变量，如 LLR_CFG=night） ----
_override = os.environ.get("LLR_CFG")
if _override:
    globals().update(_load(f"{_override}.json"))
    print(f"[config] 已加载覆盖配置: configs/{_override}.json", file=sys.stderr)

# ---- 派生路径（覆盖配置在路径计算前生效） ----
DATA_DIR = os.path.join(BASE_DIR, DATA_DIR_REL)
WEIGHTS_DIR = os.path.join(BASE_DIR, CKPT_DIR_REL)
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR_REL)
LOGS_DIR = os.path.join(BASE_DIR, LOGS_DIR_REL)
LWM_REPO_DIR = os.path.normpath(os.path.join(BASE_DIR, LWM_REPO_DIR_REL))
LWM_OFFICIAL_CKPT = os.path.join(LWM_REPO_DIR, LWM_CKPT_FILENAME)
CKPT_PRETRAIN = os.path.join(WEIGHTS_DIR, CKPT_PRETRAIN_FILE)
CKPT_LLR = os.path.join(WEIGHTS_DIR, CKPT_LLR_FILE)
CKPT_LLR_NO_PT = os.path.join(WEIGHTS_DIR, CKPT_LLR_NO_PT_FILE)
CACHE_PT = os.path.join(DATA_DIR, CACHE_PT_FILE)
CACHE_TRAIN = os.path.join(DATA_DIR, CACHE_TRAIN_FILE)
CACHE_VAL = os.path.join(DATA_DIR, CACHE_VAL_FILE)
EVAL_RESULTS = os.path.join(RESULTS_DIR, EVAL_RESULTS_FILE)
EVAL_CURVES = os.path.join(RESULTS_DIR, EVAL_CURVES_FILE)

for _d in (DATA_DIR, WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)
