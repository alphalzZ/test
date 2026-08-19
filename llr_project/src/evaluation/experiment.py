# -*- coding: utf-8 -*-
"""
experiment.py — 参考 Sionna 官方 "Neural Receiver for OFDM SIMO Systems" 教程架构

教程: ipynb/Neural_Receiver.ipynb
      https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Neural_Receiver.html

架构对齐（与教程一一对应）：
  1. E2ESystem 类（教程 "End-to-end System" 一节）：
     统一封装 发射机 -> 信道 -> 接收机；system 字符串选择接收机
       'baseline-perfect-csi'   : 真实信道 max-log LLR（教程 perfect CSI 基线）
       'baseline-ls-estimation' : LS 信道估计 + MMSE 均衡 + max-log LLR（教程 LS 估计基线）
       'neural-receiver'        : LWM+CNN LLR 预测器（教程 NeuralReceiver）
       'neural-receiver-nopt'   : 对照（官方权重微调，无 MCM 预训练）
     forward(batch_size, snr_db, mod) 每次调用即时采样新信道/噪声/比特并计算 LLR
     （教程 forward 风格）；compute_llr(batch, mod) 对给定样本只执行接收机，
     供多个接收机在同一批数据上公平对比（配对比较）。
  2. 评估（教程 "Evaluation of the Baselines" 风格）：扫 SNR，逐样本蒙特卡洛，
     聚合硬判决 BER，输出 BER vs SNR 半对数曲线（教程 sim_ber + 绘图风格）。
  3. BMD rate（教程训练目标）：提供 bmd_rate(bits, llr) 工具
     R = 1 - BCE(bits, LLR)/ln2；本项目的模型训练在 train_llr.py 完成，
     脚本专注"训练好的模型 + 基线"的性能对比。

多配置说明：教程是单配置（固定 CDL/天线）扫 SNR；本项目核心是多配置自适应
（天线/RB/符号/DMRS/TDL/速度），默认确定性循环覆盖全维度并输出分档表，
--fix-* 全指定时即为教程式的单配置评估。

用法:
  python experiment.py                                       # 默认: 多配置循环（96 样本）
  python experiment.py --snrs -5 0 5 10 15 20 --per-snr 32
  python experiment.py --fix-ant 4 --fix-rb 6 --fix-symb 14 --snrs 0 10 20 --per-snr 64
  python experiment.py --mods 4 16 --ckpt weights/lwm_llr_night.pt --tag night
输出:
  experiment_results.json         逐样本结果（结构同 eval_results.json，可用 analyze_night.py 分析）
  experiment_ber_curves.png       BER vs SNR 总曲线
  experiment_ber_by_mod.png       按调制阶数分组的 BER vs SNR 曲线
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils import config
from src.datasets.demap import qam_constellation, demap_llr
from src.simulation.pusch import SionnaPUSCHSystem
from src.evaluation.evaluate import load_llr_model, sample_cfg_vec, pearson_corr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_cjk_font():
    """设置 matplotlib 中文字体（按系统可用字体自动选择），修复中文变方块。
    图例/标题含中文（理想上界/基线/本方案/对照 等），默认 DejaVu Sans 无中文字形。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
                 "Noto Sans CJK HK", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "SimHei", "Microsoft YaHei", "PingFang SC", "Droid Sans Fallback",
                 "AR PL UMing CN", "AR PL UKai CN"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII 连字符，避免字体缺失


_setup_cjk_font()


# =============================================================================
# E2ESystem：端到端系统（教程 "End-to-end System" 架构）
# =============================================================================
class E2ESystem:
    """
    端到端系统：封装 发射机 -> 信道 -> 接收机（LLR 计算）。

    与教程 E2ESystem 对应：``system`` 字符串选择接收机，``forward`` 每次调用
    即时采样一批新信道/噪声/比特（教程 forward(batch_size, ebno_db) 风格）。
    本项目数据生成基于 Sionna PUSCH 链路（SionnaPUSCHSystem，固定带宽嵌入
    1024-FFT 系统网格），接收机支持：

      - 'baseline-perfect-csi'   : 真实信道 max-log LLR（教程 perfect CSI 上界）
      - 'baseline-ls-estimation' : LS 信道估计 + MMSE 均衡 + max-log LLR（教程 LS 基线）
      - 'neural-receiver'        : LWM+CNN LLR 预测器（教程神经接收机）
      - 'neural-receiver-nopt'   : 对照（官方权重微调，无 MCM 预训练）

    用法:
      sys_ = E2ESystem('neural-receiver', cfg, model)     # 单配置
      out  = sys_.forward(8, 10.0, 16)                    # 生成 8 样本并算 LLR
      out  = sys_.compute_llr(batch, 16)                  # 只算接收机（共享数据）
    """

    def __init__(self, system, cfg, model=None):
        assert system in ("baseline-perfect-csi", "baseline-ls-estimation",
                          "neural-receiver", "neural-receiver-nopt"), system
        self.system = system
        self.cfg = dict(cfg) if cfg else None
        self.model = model
        self._sys = None   # SionnaPUSCHSystem 懒构建（仅 forward 需要）

    def _build(self):
        if self._sys is None:
            self._sys = SionnaPUSCHSystem(**self.cfg)
        return self._sys

    def forward(self, batch_size, snr_db, mod_order, seed=None):
        """教程 forward 风格：即时采样新信道/噪声/比特并计算 LLR。
        返回 {"bits": (B,n_data,k), "llr": (B,n_data,k), "llr_ref": (B,n_data,k)}"""
        batch = self._build().generate_batch(batch_size, snr_db, mod_order, seed=seed)
        return self.compute_llr(batch, mod_order)

    def compute_llr(self, batch, mod_order):
        """对给定生成样本执行接收机（同一批数据上多接收机公平对比用）。"""
        B = int(np.asarray(batch["bits_tx"]).shape[0])
        if self.system == "baseline-perfect-csi":
            llr = np.asarray(batch["llr_ref"])
        elif self.system == "baseline-ls-estimation":
            # 基线 = Sionna 标准 LS+MMSE+APP demapper（generate_batch 的 llr_base）；
            # 旧缓存（无 llr_base）回退手写 max-log
            if "llr_base" in batch:
                llr = np.asarray(batch["llr_base"], dtype=np.float32)
            else:
                X, btab = qam_constellation(mod_order)
                llr = np.stack([demap_llr(np.asarray(batch["z"])[bb],
                                          np.asarray(batch["sigma2_eq"])[bb],
                                          X, btab, config.MAX_LLR)
                                for bb in range(B)])
        else:
            # LWM+CNN（神经接收机）：输入 LS 信道估计 + 均衡符号 + 噪声功率
            assert self.model is not None, "neural-receiver 需要加载模型权重"
            cfg_v = sample_cfg_vec(self._sample0(batch))
            llr = np.stack([
                self.model.infer_llr(np.asarray(batch["H_est"])[bb],
                                     np.asarray(batch["z"])[bb],
                                     float(batch["sigma2"]), mod_order,
                                     batch["data_re_idx"], cfg_v)
                for bb in range(B)])
        return {"bits": np.asarray(batch["bits_tx"]), "llr": llr,
                "llr_ref": np.asarray(batch["llr_ref"])}

    @staticmethod
    def _sample0(batch):
        """batch dict -> 单样本 dict（data_re_idx 为配置级共享，不按 batch 维切片）"""
        return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 and k != "data_re_idx"
                    else v) for k, v in batch.items()}


def bmd_rate(bits, llr):
    """教程训练目标：BMD rate（bit/channel use），R = 1 - BCE/ln2"""
    b = torch.as_tensor(bits, dtype=torch.float32)
    l = torch.as_tensor(llr, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(l, b)
    return 1.0 - float(bce) / math.log(2.0)


# =============================================================================
# 配置采样与评估
# =============================================================================
def sample_cfg(i, snr, args):
    """第 i 个样本的系统配置。
    --fix-* 固定某维度时该维度取固定值（教程风格：单配置扫 SNR）；
    未固定时按确定性循环覆盖全维度（偏移设计同 evaluate.eval_cfg，
    tdl/speed 用 SNR 作第二自由度避免与天线/DMRS 完全共线）。"""
    ants = [args.fix_ant] if args.fix_ant else config.RX_ANTS
    rbs = [args.fix_rb] if args.fix_rb else [1, 2, 3, 4, 6, 8, 10]
    symbs = [args.fix_symb] if args.fix_symb else [3, 5, 7, 10, 14]
    aps = ([args.fix_dmrs] if args.fix_dmrs is not None else [0, 1, 2])
    tdls = [args.fix_tdl] if args.fix_tdl else ["A", "B", "C", "D"]
    speeds = ([args.fix_speed] if args.fix_speed is not None else [0.0, 5.0, 30.0])
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

    # ---- 加载模型（教程：神经接收机权重；缺省时只评估两个基线） ----
    model = load_llr_model(args.ckpt, device) if os.path.exists(args.ckpt) else None
    model_nopt = None
    if args.ckpt_nopt and os.path.exists(args.ckpt_nopt):
        model_nopt = load_llr_model(args.ckpt_nopt, device)
    # 参与对比的接收机（教程：分别实例化 E2ESystem 并评估）
    # 内部 tag 与结果键一致（ber_ref/ber_base/ber_lwm/ber_lwm_nopt，
    # 兼容 evaluate.py 与 analyze_night.py），显示名在 print/plot 中映射。
    receivers = [("ref", E2ESystem("baseline-perfect-csi", None)),
                 ("base", E2ESystem("baseline-ls-estimation", None))]
    if model is not None:
        print(f"[EXP] 主模型: {args.ckpt}")
        receivers.append(("lwm", E2ESystem("neural-receiver", None, model)))
    else:
        print(f"[EXP] 未找到主模型 {args.ckpt}，仅评估基线与理想上界")
    if model_nopt is not None:
        print(f"[EXP] 对照模型: {args.ckpt_nopt}")
        receivers.append(("lwm_nopt", E2ESystem("neural-receiver-nopt", None, model_nopt)))

    mods = args.mods or config.MOD_ORDERS
    snrs = list(args.snrs)
    n_total = len(snrs) * args.per_snr
    print(f"[EXP] 配置: {args.per_snr} 样本/SNR × {len(snrs)} 个 SNR = {n_total} 样本, "
          f"调制 {mods}，接收机 [{', '.join(t for t, _ in receivers)}]")

    # ---- 评估循环（教程 sim_ber 风格：扫 SNR，蒙特卡洛采样） ----
    # 数据只生成一次，所有接收机在同一批数据上对比（配对比较，更公平）
    cache = {}                      # 配置 key -> SionnaPUSCHSystem（同配置复用）
    records = []
    t0 = time.time()
    for snr in snrs:
        for i in range(args.per_snr):
            cfg = sample_cfg(i, snr, args)
            mod = mods[(i + int(round(snr))) % len(mods)]
            key = tuple(sorted(cfg.items()))
            if key not in cache:
                cache[key] = SionnaPUSCHSystem(**cfg)
            batch = cache[key].generate_batch(1, snr, mod,
                                              seed=args.seed + int(round(snr)) + i)
            bits = np.asarray(batch["bits_tx"])[0]
            ref = np.asarray(batch["llr_ref"])[0]
            rec = {"snr_db": float(snr), "mod": mod, "k": int(math.log2(mod)),
                   "n_rx": int(batch["n_rx"]), "n_sc": int(batch["n_sc"]),
                   "n_symb": int(batch["n_symb"]), "dmrs_ap": int(batch["dmrs_ap"]),
                   "tdl": str(batch["tdl"]), "max_speed": float(batch["max_speed"]),
                   "cfg": dict(cfg)}
            for tag, sys_ in receivers:
                out = sys_.compute_llr(batch, mod)
                llr = out["llr"][0]
                rec[f"ber_{tag}"] = float(np.mean(((llr > 0).astype(int)) != bits))
                rec[f"mse_{tag}"] = float(np.mean((llr - ref) ** 2))
                rec[f"corr_{tag}"] = pearson_corr(llr, ref)
                rec[f"bmd_{tag}"] = bmd_rate(bits, llr)
            records.append(rec)
        print(f"  [EXP] SNR {snr}dB 完成（累计 {time.time() - t0:.1f}s）")
    print(f"[EXP] 评估样本数: {len(records)} ({time.time() - t0:.1f}s)")

    # ---- 汇总输出 ----
    summary = summarize(records, args)
    print_summary(summary, receivers)

    # ---- 保存 ----
    tag = f"_{args.tag}" if args.tag else ""
    out_json = os.path.join(config.RESULTS_DIR, f"experiment{tag}_results.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "results": records}, f, indent=2)
    print(f"[EXP] 结果已保存: {out_json}")
    plot_ber_curves(records, args,
                    os.path.join(config.RESULTS_DIR, f"experiment{tag}_ber_curves.png"),
                    os.path.join(config.RESULTS_DIR, f"experiment{tag}_ber_by_mod.png"))
    print(f"[EXP] 曲线已保存: experiments/results/experiment{tag}_ber_curves.png / "
          f"experiment{tag}_ber_by_mod.png")


def summarize(results, args):
    """按 SNR / 调制 / 配置维度聚合（返回 dict，供打印与存档）"""
    def agg(key, filt=None):
        rs = [r for r in results if filt(r)] if filt else results
        if not rs or key not in rs[0]:
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
            entry = {"value": v, "n": sum(1 for r in results if r[key] == v)}
            for tag in ("ref", "base", "lwm", "lwm_nopt"):
                entry[f"ber_{tag}"] = agg(f"ber_{tag}",
                                          lambda r, k=key, v=v: r[k] == v)
            out[outkey].append(entry)
    out["llr"] = {k: agg(k) for k in
                  ("mse_base", "corr_base", "mse_lwm", "corr_lwm",
                   "mse_lwm_nopt", "corr_lwm_nopt")}
    # BMD rate（教程训练目标 R=1-BCE/ln2，全样本平均）
    out["bmd"] = {}
    for tag in ("ref", "base", "lwm", "lwm_nopt"):
        if f"bmd_{tag}" in results[0]:
            out["bmd"][tag] = float(np.mean([r[f"bmd_{tag}"] for r in results]))
    return out


DISPLAY = {"ref": "ref", "base": "base", "lwm": "lwm", "lwm_nopt": "lwm_noPT"}


def print_summary(s, receivers):
    """控制台表格（教程风格：BER vs SNR 主表 + 配置分档）"""
    def fmt_val(v):
        return "    -" if v is None else f"{v:8.4f}"

    names = {t: {"ref": "ref", "base": "base", "lwm": "lwm", "lwm_nopt": "lwm_noPT"}[t]
             for t, _ in receivers}
    has = {t: f"ber_{t}" in s["by_snr"][0] for t, _ in receivers}

    print("\n" + "=" * 100)
    print("硬判决 BER（越低越好）  [base=LS+MMSE基线  lwm=本方案  "
          "lwm_noPT=对照  ref=理想上界]")
    print("=" * 100)
    print(f"{'SNR':>6} | " + " ".join(f"{names[t]:>8}" for t, _ in receivers if has[t]))
    print("-" * 100)
    for row in s["by_snr"]:
        line = f"{row['snr']:>6.0f} | "
        for t, _ in receivers:
            if has[t]:
                line += f"{fmt_val(row[f'ber_{t}'])} "
        print(line)

    def dim(title, rows, fmt=None):
        print(f"\n按 {title}（BER, 全 SNR 平均）:")
        print(f"{'值':>12} {'n':>4} " +
              " ".join(f"{names[t]:>8}" for t, _ in receivers if has[t]))
        for r in rows:
            v = fmt(r["value"]) if fmt else r["value"]
            line = f"{str(v):>12} {r['n']:>4} "
            for t, _ in receivers:
                if has[t]:
                    line += f"{fmt_val(r[f'ber_{t}'])} "
            print(line)

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
    for t, _ in receivers:
        m = s["llr"].get(f"mse_{t}")
        if m is not None:
            print(f"  {t:>9}: MSE={m:.4f}  corr={s['llr'][f'corr_{t}']:.4f}")

    print("\nBMD rate（教程训练目标 R=1-BCE/ln2，全样本平均，越高越好）:")
    for t in ("ref", "base", "lwm", "lwm_nopt"):
        v = s["bmd"].get(t)
        if v is not None:
            print(f"  {DISPLAY[t]:>9}: {v:.4f} bit")


def plot_ber_curves(results, args, png_total, png_mod):
    """BER vs SNR 曲线（教程 Fig 风格：semilogy，每接收机一条线）"""
    snrs = sorted({r["snr_db"] for r in results})
    tags = [t for t in ("ref", "base", "lwm", "lwm_nopt") if f"ber_{t}" in results[0]]
    labels = {"ref": "理想上界 (perfect CSI)", "base": "LS+MMSE 基线",
              "lwm": "LWM+CNN (本方案)", "lwm_nopt": "LWM+CNN (对照)"}
    markers = {"ref": "d--", "base": "o-", "lwm": "s-", "lwm_nopt": "^-"}

    def ber_series(rs, tag):
        """按 SNR 聚合 BER；某 (SNR, 子集) 无样本时置 None（避免空均值警告）"""
        ys = []
        for s in snrs:
            vals = [r[f"ber_{tag}"] for r in rs if abs(r["snr_db"] - s) < 1e-6]
            ys.append(float(np.mean(vals)) if vals else None)
        return ys

    def plot_line(ax, ys, marker, label):
        xs = [s for s, y in zip(snrs, ys) if y is not None]
        ok = [y for y in ys if y is not None]
        if ok:
            ax.semilogy(xs, ok, marker, label=label)

    fig, ax = plt.subplots(figsize=(7, 5))
    for t in tags:
        plot_line(ax, ber_series(results, t), markers[t], labels[t])
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
        for t in tags:
            plot_line(ax, ber_series(rs, t), markers[t], labels[t])
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
