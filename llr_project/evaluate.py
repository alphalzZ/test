# -*- coding: utf-8 -*-
"""
性能评估（设计文档第 8 节）

对比三路 LLR：
  1. 传统基线  ：max-log LLR 用带噪信道估计 H_est
  2. LWM+Decoder：本项目（继续预训练 + LLR 微调）
  3. 理想上界  ：max-log LLR 用真实信道 H_true（= 监督标签）

指标：LLR MSE、硬判决 BER；按 SNR / 子载波数 / 调制阶数分档汇总。
用法：python evaluate.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_gen import (generate_sample, qam_constellation, maxlog_llr,
                      data_subcarrier_idx)
from model import LWMLLR, load_official_backbone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_llr_model(ckpt, device="cpu"):
    """加载阶段2微调权重"""
    bb = load_official_backbone(device=device)
    model = LWMLLR(bb).to(device)
    sd = torch.load(ckpt, map_location=device)
    # 兼容：可能保存了完整 LWMLLR state_dict
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        bb.load_state_dict(sd)
        model = LWMLLR(bb).to(device)
    model.eval()
    return model


def eval_sample(sample, model=None, model_nopt=None):
    """对单样本计算各方案 LLR 与指标（数据子载波）"""
    H_true = sample["H_true"]
    H_est = sample["H_est"]
    y = sample["y"]
    sigma2 = float(sample["sigma2"])
    mod_order = int(sample["mod_order"])
    bits = sample["bits_tx"]
    llr_ref = sample["llr_ref"]

    X, btab = qam_constellation(mod_order)
    k = btab.shape[1]

    # 数据子载波
    n_sc = int(sample["n_sc"])
    data_idx = data_subcarrier_idx(n_sc, config.PILOT_SPACING)
    y_d = y[:, data_idx]
    H_true_d = H_true[:, data_idx]
    H_est_d = H_est[:, data_idx]

    # 1) 理想上界（真实信道，标签同源）
    llr_ideal = maxlog_llr(y_d, H_true_d, sigma2, X, btab, config.MAX_LLR)
    # 2) 传统基线（带噪信道估计）
    llr_base = maxlog_llr(y_d, H_est_d, sigma2, X, btab, config.MAX_LLR)

    out = {"n_sc": n_sc, "snr_db": -10 * np.log10(sigma2),
           "mod": mod_order, "k": k}
    for name, llr in [("ideal", llr_ideal), ("base", llr_base)]:
        out[f"mse_{name}"] = float(np.mean((llr - llr_ref) ** 2))
        hard = (llr > 0).astype(int)
        out[f"ber_{name}"] = float(np.mean(hard != bits))

    # 3) LWM + decoder（残差增强）
    if model is not None:
        llr_lwm = model.infer_llr(H_est, sample["z"], sigma2, mod_order,
                                  sample["sigma2_eq"])[:, :k]
        out["mse_lwm"] = float(np.mean((llr_lwm - llr_ref) ** 2))
        hard = (llr_lwm > 0).astype(int)
        out["ber_lwm"] = float(np.mean(hard != bits))
    # 4) 对照：LWM（无继续预训练）+ decoder
    if model_nopt is not None:
        llr_np = model_nopt.infer_llr(H_est, sample["z"], sigma2, mod_order,
                                      sample["sigma2_eq"])[:, :k]
        out["mse_lwm_nopt"] = float(np.mean((llr_np - llr_ref) ** 2))
        hard = (llr_np > 0).astype(int)
        out["ber_lwm_nopt"] = float(np.mean(hard != bits))
    return out


def main():
    print("=" * 70)
    print("LWM LLR 预测性能评估")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    model_nopt = None
    if os.path.exists(config.CKPT_LLR):
        print(f"加载主模型: {config.CKPT_LLR}")
        model = load_llr_model(config.CKPT_LLR, device)
    else:
        print("警告: 未找到 lwm_llr.pt，仅评估基线与理想上界")
    if os.path.exists(config.CKPT_LLR_NO_PT):
        print(f"加载对照模型(无继续预训练): {config.CKPT_LLR_NO_PT}")
        model_nopt = load_llr_model(config.CKPT_LLR_NO_PT, device)

    rng = np.random.default_rng(config.EVAL_SEED)
    results = []
    per_nsc = {nsc: {s: [] for s in config.EVAL_SNR_LIST} for nsc in config.EVAL_NSC_LIST}
    t0 = time.time()

    n_per_combo = max(1, config.EVAL_N // (len(config.EVAL_NSC_LIST) * len(config.EVAL_SNR_LIST)))
    total = 0
    for nsc in config.EVAL_NSC_LIST:
        for snr in config.EVAL_SNR_LIST:
            for _ in range(n_per_combo):
                mod = config.MOD_ORDERS[rng.integers(0, len(config.MOD_ORDERS))]
                s = generate_sample(rng, config.N_ANT, nsc, mod, float(snr))
                r = eval_sample(s, model, model_nopt)
                results.append(r)
                per_nsc[nsc][snr].append(r)
                total += 1
    print(f"评估样本数: {total} ({time.time()-t0:.1f}s)")

    # ============ 汇总表：按 N_sc 分组，BER vs SNR ============
    print("\n" + "=" * 100)
    print("硬判决 BER（越低越好）  [ideal=理想上界  base=传统基线  lwm=本方案  lwm_noPT=无继续预训练对照]")
    print("=" * 100)
    hdr = f"{'N_sc':>6} {'SNR':>6} | {'ideal':>8} {'base':>8} {'lwm':>8}"
    if model_nopt:
        hdr += f" {'lwm_noPT':>10}"
    print(hdr)
    print("-" * 100)

    plot_data = {}
    for nsc in config.EVAL_NSC_LIST:
        xs, ys_ideal, ys_base, ys_lwm = [], [], [], []
        for snr in config.EVAL_SNR_LIST:
            rs = per_nsc[nsc][snr]
            if not rs:
                continue
            avg = lambda key: float(np.mean([r[key] for r in rs if key in r]))
            line = f"{nsc:>6} {snr:>6} | {avg('ber_ideal'):>8.4f} {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
            if model_nopt:
                line += f" {avg('ber_lwm_nopt'):>10.4f}"
            print(line)
            xs.append(snr)
            ys_ideal.append(avg("ber_ideal"))
            ys_base.append(avg("ber_base"))
            ys_lwm.append(avg("ber_lwm"))
        plot_data[nsc] = (xs, ys_ideal, ys_base, ys_lwm)

    # ============ LLR MSE 汇总 ============
    print("\n" + "=" * 70)
    print("LLR MSE（vs 理想 max-log LLR，越小越好）")
    print("=" * 70)
    for key, name in [("mse_ideal", "ideal"), ("mse_base", "base"),
                      ("mse_lwm", "lwm"), ("mse_lwm_nopt", "lwm_noPT")]:
        if any(key in r for r in results):
            v = float(np.mean([r[key] for r in results]))
            print(f"  {name:<10}: {v:.4f}")

    # 按调制阶数分档 BER
    print("\n按调制阶数（BER, 全 SNR 平均）:")
    hdr = f"{'mod':>8} {'ideal':>8} {'base':>8} {'lwm':>8}"
    if model_nopt:
        hdr += f" {'lwm_noPT':>10}"
    print(hdr)
    for mod in config.MOD_ORDERS:
        rs = [r for r in results if r["mod"] == mod]
        if rs:
            avg = lambda key: float(np.mean([r[key] for r in rs]))
            line = f"{mod:>6}QAM {avg('ber_ideal'):>8.4f} {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
            if model_nopt:
                line += f" {avg('ber_lwm_nopt'):>10.4f}"
            print(line)

    # ============ 绘图 ============
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (nsc, (xs, yi, yb, yl)) in zip(axes.flat, plot_data.items()):
        ax.semilogy(xs, np.clip(yi, 1e-6, 1), "g-o", label="Ideal (H_true)")
        ax.semilogy(xs, np.clip(yb, 1e-6, 1), "r--s", label="Baseline (H_est)")
        ax.semilogy(xs, np.clip(yl, 1e-6, 1), "b-^", label="LWM+Decoder")
        ax.set_title(f"N_sc = {nsc}")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("LWM LLR 预测: BER vs SNR", fontsize=14)
    plt.tight_layout()
    png = os.path.join(config.BASE_DIR, "eval_ber_curves.png")
    plt.savefig(png, dpi=130)
    print(f"\n图已保存: {png}")

    with open(os.path.join(config.BASE_DIR, "eval_results.json"), "w") as f:
        json.dump({"results": results, "summary": {
            "mse_ideal": float(np.mean([r["mse_ideal"] for r in results])),
            "mse_base": float(np.mean([r["mse_base"] for r in results])),
            "mse_lwm": float(np.mean([r["mse_lwm"] for r in results])) if model else None,
        }}, f, indent=2)
    print("结果已保存: eval_results.json")


if __name__ == "__main__":
    main()
