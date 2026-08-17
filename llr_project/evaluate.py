# -*- coding: utf-8 -*-
"""
性能评估（Sionna PUSCH 3D 数据）

对比各方案在数据 RE 上的 LLR：
  1. 传统基线  ：max-log LLR 用带噪信道估计 H_est
  2. LWM+Decoder：本项目（继续预训练 + LLR 微调）
  3. 对照模型  ：LWM（官方权重）+ decoder
  4. 理想上界  ：max-log LLR 用真实信道 H_true

指标：LLR MSE、硬判决 BER；按 SNR / 调制阶数分档。
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
from data_gen import qam_constellation, maxlog_llr, demap_llr
from data_gen_sionna import generate_dataset
from model import LWMLLR, load_official_backbone, N_DATA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_llr_model(ckpt, device="cpu"):
    """加载阶段2微调权重"""
    bb = load_official_backbone(device=device)
    model = LWMLLR(bb).to(device)
    sd = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        bb.load_state_dict(sd)
        model = LWMLLR(bb).to(device)
    model.eval()
    return model


def eval_sample(sample, model=None, model_nopt=None):
    """对单样本计算各方案 LLR 与指标（数据 RE）"""
    H_true = sample["H_true"]     # (8,120,14)
    H_est = sample["H_est"]
    sigma2 = float(sample["sigma2"])
    mod_order = int(sample["mod_order"])
    bits = sample["bits_tx"]      # (1440, k)
    llr_ref = sample["llr_ref"]   # (1440, k)

    X, btab = qam_constellation(mod_order)
    k = btab.shape[1]

    # 数据 RE 上的多天线信号：y 未在样本中保存，从 H/z 推算不可行
    # 基线/理想 LLR 用"均衡后 demap"（z, sigma2_eq）计算，与标签口径一致
    # 注：llr_ref 是理想信道 max-log（含多天线合并），基线与模型输入同为 z
    llr_base = demap_llr(sample["z"], sample["sigma2_eq"], X, btab, config.MAX_LLR)

    out = {"snr_db": -10 * np.log10(sigma2), "mod": mod_order, "k": k}
    out["mse_base"] = float(np.mean((llr_base - llr_ref) ** 2))
    hard = (llr_base > 0).astype(int)
    out["ber_base"] = float(np.mean(hard != bits))

    if model is not None:
        llr_lwm = model.infer_llr(H_est, sample["z"], sigma2, mod_order,
                                  sample["sigma2_eq"])
        out["mse_lwm"] = float(np.mean((llr_lwm - llr_ref) ** 2))
        hard = (llr_lwm > 0).astype(int)
        out["ber_lwm"] = float(np.mean(hard != bits))
    if model_nopt is not None:
        llr_np = model_nopt.infer_llr(H_est, sample["z"], sigma2, mod_order,
                                      sample["sigma2_eq"])
        out["mse_lwm_nopt"] = float(np.mean((llr_np - llr_ref) ** 2))
        hard = (llr_np > 0).astype(int)
        out["ber_lwm_nopt"] = float(np.mean(hard != bits))
    return out


def main():
    print("=" * 70)
    print("LWM LLR 预测性能评估（Sionna PUSCH 3D 信道）")
    print("=" * 70)

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
    per_snr = {s: [] for s in config.EVAL_SNR_LIST}
    t0 = time.time()

    n_per_snr = max(1, config.EVAL_N // len(config.EVAL_SNR_LIST))
    total = 0
    for snr in config.EVAL_SNR_LIST:
        # 每个 SNR 生成一批样本（混合调制），用独立随机种子保证与训练不同
        samples = generate_dataset(n_per_snr, num_rx_ant=config.N_ANT,
                                   n_size_grid=config.N_SC // 12,
                                   snr_db=snr, seed=config.EVAL_SEED + snr)
        for s in samples:
            r = eval_sample(s, model, model_nopt)
            results.append(r)
            per_snr[snr].append(r)
            total += 1
    print(f"评估样本数: {total} ({time.time()-t0:.1f}s)")

    # ============ BER vs SNR ============
    print("\n" + "=" * 100)
    print("硬判决 BER（越低越好）  [base=传统基线  lwm=本方案  lwm_noPT=对照  ideal_ref=标签LLR]")
    print("=" * 100)
    hdr = f"{'SNR':>6} | {'base':>8} {'lwm':>8}"
    if model_nopt:
        hdr += f" {'lwm_noPT':>10}"
    hdr += f" {'ideal':>8}"
    print(hdr)
    print("-" * 100)

    xs, ys_base, ys_lwm = [], [], []
    for snr in config.EVAL_SNR_LIST:
        rs = per_snr[snr]
        if not rs:
            continue
        avg = lambda key: float(np.mean([r[key] for r in rs if key in r]))
        line = f"{snr:>6} | {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
        if model_nopt:
            line += f" {avg('ber_lwm_nopt'):>10.4f}"
        print(line)
        xs.append(snr)
        ys_base.append(avg("ber_base"))
        ys_lwm.append(avg("ber_lwm"))

    # ============ LLR MSE 汇总 ============
    print("\n" + "=" * 70)
    print("LLR MSE（vs 理想 max-log LLR，越小越好）")
    print("=" * 70)
    for key, name in [("mse_base", "base"), ("mse_lwm", "lwm"),
                      ("mse_lwm_nopt", "lwm_noPT")]:
        if any(key in r for r in results):
            v = float(np.mean([r[key] for r in results]))
            print(f"  {name:<10}: {v:.4f}")

    # ============ 按调制阶数 ============
    print("\n按调制阶数（BER, 全 SNR 平均）:")
    hdr = f"{'mod':>8} {'base':>8} {'lwm':>8}"
    if model_nopt:
        hdr += f" {'lwm_noPT':>10}"
    print(hdr)
    for mod in config.MOD_ORDERS:
        rs = [r for r in results if r["mod"] == mod]
        if rs:
            avg = lambda key: float(np.mean([r[key] for r in rs]))
            line = f"{mod:>6}QAM {avg('ber_base'):>8.4f} {avg('ber_lwm'):>8.4f}"
            if model_nopt:
                line += f" {avg('ber_lwm_nopt'):>10.4f}"
            print(line)

    # ============ 绘图 ============
    if model is not None and xs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(xs, np.clip(ys_base, 1e-6, 1), "r--s", label="Baseline (H_est)")
        ax.semilogy(xs, np.clip(ys_lwm, 1e-6, 1), "b-^", label="LWM+Decoder")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title("LWM LLR Prediction: BER vs SNR (Sionna PUSCH)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        png = os.path.join(config.BASE_DIR, "eval_ber_curves.png")
        plt.savefig(png, dpi=130)
        print(f"\n图已保存: {png}")

    with open(os.path.join(config.BASE_DIR, "eval_results.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("结果已保存: eval_results.json")


if __name__ == "__main__":
    main()
