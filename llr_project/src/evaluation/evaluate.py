# -*- coding: utf-8 -*-
"""
性能评估（多配置自适应版，Sionna PUSCH 3D 数据）

评估一个模型在不同系统参数组合下的 LLR 预测性能：
  1. 传统基线  ：max-log LLR 用带噪信道估计 H_est（仅作对比，模型推理不再需要）
  2. LWM+CNN   ：本项目（继续预训练 + LLR 微调，BCE 训练，输入不含 llr_base）
  3. 对照模型  ：LWM（官方权重）+ CNN decoder
  4. 理想上界  ：max-log LLR 用真实信道 H_true（标签）

评估覆盖维度：SNR、接收天线数（1/2/4/8）、RB（1~10）、符号数（3~14）、
DMRS 模式（{1}/{1+1}/{1+2}）、TDL（A/B/C/D）、时延/多普勒。
指标：硬判决 BER（主）、LLR MSE、LLR 相关系数。
用法：python evaluate.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils import config
from src.datasets.demap import qam_constellation, demap_llr
from src.simulation.pusch import SionnaPUSCHSystem
from src.models.lwm_llr import LWMLLR, load_official_backbone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_llr_model(ckpt, device="cpu"):
    """加载阶段2微调权重（完整状态字典）"""
    bb = load_official_backbone(device=device)
    model = LWMLLR(bb).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def pearson_corr(a, b):
    """尺度无关的 LLR 形状相似度（BCE 训练 logits 与 max-log 参考尺度不同）"""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def sample_cfg_vec(s):
    """单样本 -> 配置元数据向量 (CFG_DIM,)（与 loader.build_cfg_vec 一致）。
    只含接收端可感知参数：n_rx onehot(4) + n_sc/120 + n_symb/14 + dmrs_ap onehot(3)，
    不含信道模型信息（TDL/多普勒，接收端不感知）。"""
    v = np.zeros(config.CFG_DIM, dtype=np.float32)
    ants = [1, 2, 4, 8]
    v[0:4] = [int(s["n_rx"]) == a for a in ants]
    v[4] = int(s["n_sc"]) / 120.0
    v[5] = int(s["n_symb"]) / 14.0
    v[6:9] = [int(s["dmrs_ap"]) == a for a in [0, 1, 2]]
    return v


def eval_sample(sample, model=None, model_nopt=None):
    """对单样本计算各方案 LLR 与指标（数据 RE）"""
    H_est = sample["H_est"]       # (n_rx, n_sc, n_symb)
    sigma2 = float(sample["sigma2"])
    mod_order = int(sample["mod_order"])
    bits = sample["bits_tx"]      # (n_data, k)
    llr_ref = sample["llr_ref"]   # (n_data, k)
    dri = sample["data_re_idx"]
    cfg = sample_cfg_vec(sample)

    X, btab = qam_constellation(mod_order)
    k = btab.shape[1]
    # 基线 = Sionna 标准 LS+MMSE+APP demapper（generate_batch 的 llr_base）；
    # 旧缓存（无 llr_base 字段）回退手写 max-log
    if "llr_base" in sample:
        llr_base = np.asarray(sample["llr_base"], dtype=np.float32)
    else:
        llr_base = demap_llr(sample["z"], sample["sigma2_eq"], X, btab, config.MAX_LLR)

    out = {"snr_db": -10 * np.log10(sigma2), "mod": mod_order, "k": k,
           "n_rx": int(sample["n_rx"]), "n_sc": int(sample["n_sc"]),
           "n_symb": int(sample["n_symb"]), "dmrs_ap": int(sample["dmrs_ap"]),
           "tdl": str(sample["tdl"]),
           "delay_spread": float(sample["delay_spread"]),
           "max_speed": float(sample["max_speed"])}
    out["mse_base"] = float(np.mean((llr_base - llr_ref) ** 2))
    out["corr_base"] = pearson_corr(llr_base, llr_ref)
    hard = (llr_base > 0).astype(int)
    out["ber_base"] = float(np.mean(hard != bits))

    # ofdm_rx 风格传统接收机基线（generate_batch 的 llr_legacy，仅新数据）
    if "llr_legacy" in sample:
        llr_leg = np.asarray(sample["llr_legacy"], dtype=np.float32)
        out["mse_legacy"] = float(np.mean((llr_leg - llr_ref) ** 2))
        out["corr_legacy"] = pearson_corr(llr_leg, llr_ref)
        out["ber_legacy"] = float(np.mean(((llr_leg > 0).astype(int)) != bits))

    if model is not None:
        llr_lwm = model.infer_llr(H_est, sample["z"], sigma2, mod_order, dri, cfg)
        out["mse_lwm"] = float(np.mean((llr_lwm - llr_ref) ** 2))
        out["corr_lwm"] = pearson_corr(llr_lwm, llr_ref)
        hard = (llr_lwm > 0).astype(int)
        out["ber_lwm"] = float(np.mean(hard != bits))
    if model_nopt is not None:
        llr_np = model_nopt.infer_llr(H_est, sample["z"], sigma2, mod_order, dri, cfg)
        out["mse_lwm_nopt"] = float(np.mean((llr_np - llr_ref) ** 2))
        out["corr_lwm_nopt"] = pearson_corr(llr_np, llr_ref)
        hard = (llr_np > 0).astype(int)
        out["ber_lwm_nopt"] = float(np.mean(hard != bits))
    return out


def eval_cfg(rng, i, snr):
    """确定性配置循环（覆盖各维度）：天线 1/2/4/8 × RB 1~10 × 符号 3~14
    × DMRS 0/1/2 × TDL A/B/C/D × 速度 0/5/30，随样本序号循环偏移。
    ★ 共线修复：tdl/speed 的取模偏移改用 SNR（si=int(snr)）引入第二自由度——
    旧版 tdl 偏移 +4≡0(mod4) 与天线完全共线、speed 偏移 +6≡0(mod3) 与 DMRS 完全共线，
    导致"按 TDL"表 ≡ "按天线"表、"按速度"表 ≡ "按 DMRS"表，维度无法独立解读。
    现 tdl 偏移 si%4∈{3,0,1,2}、speed 偏移 si%3∈{1,0,2} 覆盖全部相位，解除了完全共线。"""
    ants = config.RX_ANTS
    rbs = [1, 2, 3, 4, 6, 8, 10]
    symbs = [3, 5, 7, 10, 14]
    aps = [0, 1, 2]
    tdls = ["A", "B", "C", "D"]
    speeds = [0.0, 5.0, 30.0]
    si = int(round(snr))
    return {
        "num_rx_ant": ants[(i + 0) % len(ants)],
        "n_size_grid": rbs[(i + 1) % len(rbs)],
        "num_ofdm_symbols": symbs[(i + 2) % len(symbs)],
        "dmrs_ap": aps[(i + 3) % len(aps)],
        "channel_model": tdls[(i + si) % len(tdls)],
        "delay_spread": config.DELAY_SPREADS[(i + 5) % 3],
        "max_speed": speeds[(i + si) % len(speeds)],
        "carrier_frequency": config.CARRIER_FREQUENCY,
    }


def main():
    print("=" * 80)
    print("LWM LLR 预测性能评估（多配置：天线/RB/符号/DMRS/信道场景）")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    model_nopt = None
    if os.path.exists(config.CKPT_LLR):
        print(f"加载主模型: {config.CKPT_LLR}")
        model = load_llr_model(config.CKPT_LLR, device)
    else:
        print("警告: 未找到 lwm_llr.pt，仅评估基线")
    if os.path.exists(config.CKPT_LLR_NO_PT):
        print(f"加载对照模型: {config.CKPT_LLR_NO_PT}")
        model_nopt = load_llr_model(config.CKPT_LLR_NO_PT, device)

    rng = np.random.default_rng(config.EVAL_SEED)
    results = []
    t0 = time.time()
    for snr in config.EVAL_SNR_LIST:
        for i in range(config.EVAL_PER_SNR):
            cfg = eval_cfg(rng, i, snr)
            sys_ = SionnaPUSCHSystem(**cfg)
            mod = [4, 16, 64, 256][(i + snr) % 4]
            batch = sys_.generate_batch(1, snr, mod, seed=config.EVAL_SEED + snr + i)
            s = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 and k != "data_re_idx"
                     else v) for k, v in batch.items()}
            s["snr_db"] = float(snr)
            r = eval_sample(s, model, model_nopt)
            results.append(r)
    print(f"评估样本数: {len(results)} ({time.time()-t0:.1f}s)")

    # ============ BER vs SNR ============
    print("\n" + "=" * 100)
    print("硬判决 BER（越低越好）  [base=传统基线  lwm=本方案  lwm_noPT=对照]")
    print("=" * 100)
    hdr = f"{'SNR':>6} | {'base':>8} {'lwm':>8}"
    if model_nopt:
        hdr += f" {'lwm_noPT':>10}"
    print(hdr)
    print("-" * 100)
    xs, ys_base, ys_lwm = [], [], []
    for snr in config.EVAL_SNR_LIST:
        rs = [r for r in results if abs(r["snr_db"] - snr) < 1e-6]
        if not rs:
            continue
        avg = lambda key: float(np.mean([r[key] for r in rs]))
        line = f"{snr:>6} | {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
        if model_nopt:
            line += f" {avg('ber_lwm_nopt'):>10.4f}"
        print(line)
        xs.append(snr)
        ys_base.append(avg("ber_base"))
        ys_lwm.append(avg("ber_lwm"))

    # ============ 按配置维度分档 ============
    def dim_table(title, key, fmt=None):
        if fmt is None:
            fmt = lambda v: f"{v}"
        print(f"\n按 {title}（BER, 全 SNR 平均）:")
        hdr = f"{'值':>12} {'n':>4} {'base':>8} {'lwm':>8}"
        if model_nopt:
            hdr += f" {'lwm_noPT':>10}"
        print(hdr)
        vals = sorted({r[key] for r in results})
        for v in vals:
            rs = [r for r in results if r[key] == v]
            avg = lambda k2: float(np.mean([r[k2] for r in rs]))
            line = f"{fmt(v):>12} {len(rs):>4} {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
            if model_nopt:
                line += f" {avg('ber_lwm_nopt'):>10.4f}"
            print(line)

    dim_table("接收天线数 n_rx", "n_rx")
    dim_table("RB 数（子载波）", "n_sc", fmt=lambda v: f"{v}sc({v//12}RB)")
    dim_table("OFDM 符号数", "n_symb")
    dim_table("DMRS 模式", "dmrs_ap",
              fmt=lambda v: {0: "{1}", 1: "{1+1}", 2: "{1+2}"}[v])
    dim_table("TDL 信道模型", "tdl")
    dim_table("UE 速度（多普勒）", "max_speed", fmt=lambda v: f"{v}m/s")

    # ============ LLR 指标汇总 ============
    print("\n" + "=" * 70)
    print("LLR 指标（MSE vs 理想 max-log 越小越好；相关系数越接近 1 越好）")
    print("注: BCE 训练的 logits 与 max-log 参考尺度不同，MSE 仅作参考，BER 为主指标")
    print("=" * 70)
    for key, name in [("mse_base", "base"), ("mse_lwm", "lwm"),
                      ("mse_lwm_nopt", "lwm_noPT")]:
        if any(key in r for r in results):
            v = float(np.mean([r[key] for r in results]))
            c = float(np.mean([r[key.replace("mse", "corr")] for r in results]))
            print(f"  {name:<10}: MSE={v:.4f}  corr={c:.4f}")

    # ============ 绘图 ============
    if model is not None and xs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(xs, np.clip(ys_base, 1e-6, 1), "r--s", label="Baseline (H_est)")
        ax.semilogy(xs, np.clip(ys_lwm, 1e-6, 1), "b-^", label="LWM+CNN (multi-config)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title("LWM LLR Prediction: BER vs SNR (multi-config PUSCH)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        png = config.EVAL_CURVES
        plt.savefig(png, dpi=130)
        print(f"\n图已保存: {png}")

    with open(config.EVAL_RESULTS, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"结果已保存: {config.EVAL_RESULTS}")


if __name__ == "__main__":
    main()
