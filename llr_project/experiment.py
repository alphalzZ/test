# -*- coding: utf-8 -*-
"""
experiment.py — 参考 Sionna 官方 "5G NR PUSCH Neural Receiver" 教程的实验脚本

教程: https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Neural_Receiver.html

与教程一致的思路：
  1. 数据不预先生成落盘：实验循环内用 Sionna **即时生成**（SionnaPUSCHSystem.generate_batch），
     支持任意配置组合（接收天线 / RB / OFDM 符号 / DMRS / TDL / 多普勒 / 调制 / SNR），
     对应教程的 generate_batch + 逐 SNR 循环评估；
  2. 接收机对比（对应教程的多接收机对比）：
       base    : 传统基线  = LS 信道估计 + MMSE 均衡 + max-log LLR（教程的 LS+MMSE 基线）
       lwm     : 本方案    = LWM（大无线模型）+ CNN LLR decoder（教程的神经接收机）
       lwm_noPT: 对照      = 官方权重微调（无 MCM 预训练，可选）
       ref     : 理想上界  = 真实信道 max-log LLR（教程的 perfect-CSI 参考）
  3. 指标：硬判决 BER vs SNR（semilogy 曲线，教程 Fig 风格）+ 逐样本明细 +
     按配置维度分档表（天线/RB/符号/DMRS/TDL/速度）。

用法:
  python experiment.py                                       # 默认: 确定性配置循环（96 样本）
  python experiment.py --snrs -5 0 5 10 15 20 --per-snr 32
  python experiment.py --fix-rb 6 --fix-symb 14 --snrs 0 10 20 --per-snr 64   # 固定配置扫 SNR
  python experiment.py --mods 4 16 --ckpt weights/lwm_llr_night.pt --tag night
输出:
  experiment_results.json        逐样本结果（结构同 eval_results.json，可用 analyze_night.py 分析）
  experiment_ber_curves.png      BER vs SNR 总曲线
  experiment_ber_by_mod.png      按调制阶数分组的 BER vs SNR 曲线
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_gen_sionna import SionnaPUSCHSystem
from evaluate import eval_sample, load_llr_model   # 复用基线/模型的 LLR 与 BER 计算

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_cfg(i, snr, args):
    """第 i 个样本的系统配置。
    --fix-* 固定某维度时该维度取固定值（教程风格：单配置扫 SNR）；
    未固定时按确定性循环覆盖全维度（偏移设计同 evaluate.eval_cfg，
    tdl/speed 用 SNR 作第二自由度避免与天线/DMRS 完全共线）。"""
    ants = [args.fix_ant] if args.fix_ant else config.RX_ANTS
    rbs = [args.fix_rb] if args.fix_rb else [1, 2, 3, 4, 6, 8, 10]
    symbs = [args.fix_symb] if args.fix_symb else [3, 5, 7, 10, 14]
    aps = ([args.fix_dmrs] if args.fix_dmrs is not None
           else [0, 1, 2])
    tdls = [args.fix_tdl] if args.fix_tdl else ["A", "B", "C", "D"]
    speeds = ([args.fix_speed] if args.fix_speed is not None
              else [0.0, 5.0, 30.0])
    si = int(round(snr))
    return {
        "num_rx_ant": ants[i % len(ants)],
        "n_size_grid": rbs[(i + 1) % len(rbs)],
        "num_ofdm_symbols": symbs[(i + 2) % len(symbs)],
        "dmrs_ap": aps[(i + 3) % len(aps)],
        "channel_model": tdls[(i + si) % len(tdls)],
        "delay_spread": config.DELAY_SPREADS[(i + 5) % 3],
        "max_speed": speeds[(i + si) % len(speeds)],
        "carrier_frequency": config.CARRIER_FREQUENCY,
    }


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EXP] device={device}, seed={args.seed}")

    # ---- 加载模型（存在才加载；缺省时只评估基线与理想上界） ----
    model = load_llr_model(args.ckpt, device) if os.path.exists(args.ckpt) else None
    if model is None:
        print(f"[EXP] 未找到主模型 {args.ckpt}，仅评估基线与理想上界")
    else:
        print(f"[EXP] 主模型: {args.ckpt}")
    model_nopt = None
    if args.ckpt_nopt and os.path.exists(args.ckpt_nopt):
        model_nopt = load_llr_model(args.ckpt_nopt, device)
        print(f"[EXP] 对照模型: {args.ckpt_nopt}")

    mods = args.mods or config.MOD_ORDERS
    snrs = list(args.snrs)
    n_total = len(snrs) * args.per_snr
    print(f"[EXP] 配置: {args.per_snr} 样本/SNR × {len(snrs)} 个 SNR = {n_total} 样本, "
          f"调制 {mods}")

    # ---- 教程风格数据生成循环：Sionna 即时生成，每样本一个配置 ----
    results = []
    t0 = time.time()
    for snr in snrs:
        for i in range(args.per_snr):
            cfg = sample_cfg(i, snr, args)
            sys_ = SionnaPUSCHSystem(**cfg)
            mod = mods[(i + int(round(snr))) % len(mods)]
            batch = sys_.generate_batch(1, snr, mod, seed=args.seed + int(round(snr)) + i)
            # 单样本解包（data_re_idx 为配置级共享，不按 batch 维切片）
            s = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 and k != "data_re_idx"
                     else v) for k, v in batch.items()}
            s["snr_db"] = float(snr)
            r = eval_sample(s, model, model_nopt)
            # 理想上界 BER：真实信道 max-log LLR 硬判决
            hard = (np.asarray(s["llr_ref"]) > 0).astype(int)
            r["ber_ref"] = float(np.mean(hard != np.asarray(s["bits_tx"])))
            r["cfg"] = dict(cfg)
            results.append(r)
        print(f"  [EXP] SNR {snr}dB 完成（累计 {time.time() - t0:.1f}s）")
    print(f"[EXP] 评估样本数: {len(results)} ({time.time() - t0:.1f}s)")

    # ---- 汇总输出 ----
    summary = summarize(results, args)
    print_summary(summary)

    # ---- 保存 ----
    tag = f"_{args.tag}" if args.tag else ""
    out_json = f"experiment{tag}_results.json"
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"[EXP] 结果已保存: {out_json}")
    plot_ber_curves(results, args, f"experiment{tag}_ber_curves.png",
                    f"experiment{tag}_ber_by_mod.png")
    print(f"[EXP] 曲线已保存: experiment{tag}_ber_curves.png / "
          f"experiment{tag}_ber_by_mod.png")


def summarize(results, args):
    """按 SNR / 调制 / 配置维度聚合（返回 dict，供打印与存档）"""
    def agg(key, filt=None):
        rs = [r for r in results if filt(r)] if filt else results
        if not rs:
            return None
        return float(np.mean([r[key] for r in rs]))

    out = {"by_snr": [], "by_mod": [], "by_ant": [], "by_rb": [],
           "by_symb": [], "by_dmrs": [], "by_tdl": [], "by_speed": []}
    for snr in sorted({r["snr_db"] for r in results}):
        out["by_snr"].append({"snr": snr, "n": sum(1 for r in results if r["snr_db"] == snr),
                              "ber_base": agg("ber_base", lambda r, s=snr: r["snr_db"] == s),
                              "ber_lwm": agg("ber_lwm", lambda r, s=snr: r["snr_db"] == s),
                              "ber_lwm_nopt": agg("ber_lwm_nopt", lambda r, s=snr: r["snr_db"] == s),
                              "ber_ref": agg("ber_ref", lambda r, s=snr: r["snr_db"] == s)})
    for mod in sorted({r["mod"] for r in results}):
        out["by_mod"].append({"mod": mod, "n": sum(1 for r in results if r["mod"] == mod),
                              "ber_base": agg("ber_base", lambda r, m=mod: r["mod"] == m),
                              "ber_lwm": agg("ber_lwm", lambda r, m=mod: r["mod"] == m),
                              "ber_lwm_nopt": agg("ber_lwm_nopt", lambda r, m=mod: r["mod"] == m),
                              "ber_ref": agg("ber_ref", lambda r, m=mod: r["mod"] == m)})
    for key, outkey in [("n_rx", "by_ant"), ("n_sc", "by_rb"), ("n_symb", "by_symb"),
                        ("dmrs_ap", "by_dmrs"), ("tdl", "by_tdl"), ("max_speed", "by_speed")]:
        for v in sorted({r[key] for r in results}):
            out[outkey].append({"value": v, "n": sum(1 for r in results if r[key] == v),
                                "ber_base": agg("ber_base", lambda r, k=key, v=v: r[k] == v),
                                "ber_lwm": agg("ber_lwm", lambda r, k=key, v=v: r[k] == v),
                                "ber_lwm_nopt": agg("ber_lwm_nopt", lambda r, k=key, v=v: r[k] == v)})
    # LLR 指标（全样本）
    out["llr"] = {k: agg(k) for k in
                  ("mse_base", "corr_base", "mse_lwm", "corr_lwm",
                   "mse_lwm_nopt", "corr_lwm_nopt")}
    return out


def print_summary(s):
    """控制台表格（教程风格：BER vs SNR 主表 + 配置分档）"""
    def fmt_val(v):
        return "    -" if v is None else f"{v:8.4f}"

    print("\n" + "=" * 100)
    print("硬判决 BER（越低越好）  [base=LS+MMSE基线  lwm=本方案  lwm_noPT=对照  ref=理想上界]")
    print("=" * 100)
    print(f"{'SNR':>6} | {'base':>8} {'lwm':>8} {'lwm_noPT':>10} {'ref':>8}")
    print("-" * 100)
    for row in s["by_snr"]:
        print(f"{row['snr']:>6.0f} | {fmt_val(row['ber_base'])} {fmt_val(row['ber_lwm'])} "
              f"{fmt_val(row['ber_lwm_nopt']):>10} {fmt_val(row['ber_ref'])}")

    def dim(title, rows, fmt=None):
        print(f"\n按 {title}（BER, 全 SNR 平均）:")
        print(f"{'值':>12} {'n':>4} {'base':>8} {'lwm':>8} {'lwm_noPT':>10}")
        for r in rows:
            v = fmt(r["value"]) if fmt else r["value"]
            print(f"{str(v):>12} {r['n']:>4} {fmt_val(r['ber_base'])} "
                  f"{fmt_val(r['ber_lwm'])} {fmt_val(r['ber_lwm_nopt']):>10}")

    dim("接收天线数 n_rx", s["by_ant"])
    dim("RB 数（子载波）", s["by_rb"], fmt=lambda v: f"{v}sc({v // 12}RB)")
    dim("OFDM 符号数", s["by_symb"])
    dim("DMRS 模式", s["by_dmrs"], fmt=lambda v: {0: "{1}", 1: "{1+1}", 2: "{1+2}"}[v])
    dim("TDL 信道模型", s["by_tdl"])
    dim("UE 速度（多普勒）", s["by_speed"], fmt=lambda v: f"{v}m/s")

    print("\n" + "=" * 70)
    print("LLR 指标（MSE vs 理想 max-log 越小越好；相关系数越接近 1 越好）")
    print("注: BCE 训练的 logits 与 max-log 参考尺度不同，MSE 仅作参考，BER 为主指标")
    print("=" * 70)
    for k, name in [("mse_base", "base"), ("mse_lwm", "lwm"), ("mse_lwm_nopt", "lwm_noPT")]:
        if s["llr"].get(k) is not None:
            c = s["llr"].get(k.replace("mse", "corr"))
            print(f"  {name:>9}: MSE={s['llr'][k]:.4f}  corr={c:.4f}")


def plot_ber_curves(results, args, png_total, png_mod):
    """BER vs SNR 曲线（教程 Fig 风格：semilogy，每接收机一条线）"""
    snrs = sorted({r["snr_db"] for r in results})

    def series(key, label, marker):
        ys = []
        for s in snrs:
            rs = [r for r in results if abs(r["snr_db"] - s) < 1e-6 and key in r]
            ys.append(float(np.mean([r[key] for r in rs])) if rs else None)
        return key, snrs, ys, label, marker

    lines = [series("ber_ref", "理想上界 (perfect CSI)", "d--"),
             series("ber_base", "LS+MMSE 基线", "o-"),
             series("ber_lwm", "LWM+CNN (本方案)", "s-"),
             series("ber_lwm_nopt", "LWM+CNN (对照)", "^-")]

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, snrs_, ys, label, marker in lines:
        if any(y is not None for y in ys):
            ax.semilogy(snrs_, ys, marker, label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs SNR (多配置混合)" + (f"  [{args.tag}]" if args.tag else ""))
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_total, dpi=150)
    plt.close(fig)

    # 按调制阶数分组
    mods = sorted({r["mod"] for r in results})
    fig, axes = plt.subplots(1, len(mods), figsize=(5.2 * len(mods), 4.2), squeeze=False)
    for ax, m in zip(axes[0], mods):
        rs = [r for r in results if r["mod"] == m]
        for key, snrs_, _, label, marker in lines:
            ys_m = [float(np.mean([r[key] for r in rs if abs(r["snr_db"] - s) < 1e-6]))
                    if any(abs(r["snr_db"] - s) < 1e-6 for r in rs) else None
                    for s in snrs]
            if any(y is not None for y in ys_m):
                ax.semilogy(snrs_, ys_m, marker, label=label)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"QAM{m}")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("BER vs SNR 按调制阶数" + (f"  [{args.tag}]" if args.tag else ""))
    fig.tight_layout()
    fig.savefig(png_mod, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sionna Neural Receiver 风格实验："
                                            "即时生成多配置数据，对比模型与基线")
    p.add_argument("--snrs", type=float, nargs="+", default=config.EVAL_SNR_LIST)
    p.add_argument("--per-snr", type=int, default=config.EVAL_PER_SNR,
                   help="每 SNR 样本数（默认取 config.EVAL_PER_SNR）")
    p.add_argument("--mods", type=int, nargs="+", default=None,
                   help="调制阶数集合（默认 config.MOD_ORDERS，逐样本循环分配）")
    p.add_argument("--fix-ant", type=int, default=None, help="固定接收天线数")
    p.add_argument("--fix-rb", type=int, default=None, help="固定 RB 数（1~10）")
    p.add_argument("--fix-symb", type=int, default=None, help="固定 OFDM 符号数（3~14）")
    p.add_argument("--fix-dmrs", type=int, default=None, help="固定 DMRS 模式 0/1/2")
    p.add_argument("--fix-tdl", type=str, default=None, help="固定 TDL 信道 A/B/C/D")
    p.add_argument("--fix-speed", type=float, default=None, help="固定 UE 速度 m/s")
    p.add_argument("--ckpt", type=str, default=config.CKPT_LLR, help="主模型权重")
    p.add_argument("--ckpt-nopt", type=str, default=config.CKPT_LLR_NO_PT,
                   help="对照模型权重（默认路径不存在则跳过）")
    p.add_argument("--tag", type=str, default="", help="输出文件后缀")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_experiment(args)
