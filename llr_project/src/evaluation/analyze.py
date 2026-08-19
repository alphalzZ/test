# -*- coding: utf-8 -*-
"""性能分析：从实验/评估结果 JSON 计算关键指标（只读，不改动产物）。
用法: python -m src.evaluation.analyze [--results experiments/results/eval_results.json]"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils import config

p = argparse.ArgumentParser()
p.add_argument("--results", default=config.EVAL_RESULTS,
               help="评估结果 JSON（默认 config.EVAL_RESULTS）")
_args = p.parse_args()
with open(_args.results) as f:
    data = json.load(f)
R = data["results"]
N = len(R)
print(f"评估样本总数: {N}")

def avg(xs):
    return sum(xs) / len(xs)

# ---------- 1. 按 SNR ----------
print("\n===== 1. 按 SNR =====")
print(f"{'SNR':>5} {'n':>3} {'base':>8} {'lwm':>8} {'noPT':>8} {'lwm改善%':>9} {'noPT改善%':>9}")
by_snr = defaultdict(list)
for r in R:
    by_snr[round(r["snr_db"])].append(r)
for snr in sorted(by_snr):
    rs = by_snr[snr]
    b, l, p = avg([r["ber_base"] for r in rs]), avg([r["ber_lwm"] for r in rs]), avg([r["ber_lwm_nopt"] for r in rs])
    print(f"{snr:>5} {len(rs):>3} {b:>8.4f} {l:>8.4f} {p:>8.4f} {(1-l/b)*100:>8.1f}% {(1-p/b)*100:>8.1f}%")

# ---------- 2. 逐样本：lwm 败给 base 的样本 ----------
print("\n===== 2. 逐样本失败点（lwm BER > base BER 的样本）=====")
wins = losses = 0
loss_list = []
for r in R:
    if r["ber_lwm"] < r["ber_base"] - 1e-9:
        wins += 1
    elif r["ber_lwm"] > r["ber_base"] + 1e-9:
        losses += 1
        loss_list.append(r)
print(f"lwm 胜出 {wins}/{N}，败给 base {losses}/{N}")
for r in sorted(loss_list, key=lambda r: r["ber_lwm"] - r["ber_base"], reverse=True)[:12]:
    print(f"  SNR{r['snr_db']:>4.0f} {r['n_rx']}ant {r['n_sc']}sc {r['n_symb']}sym dmrs{r['dmrs_ap']} TDL-{r['tdl']} {r['max_speed']}m/s "
          f"mod{r['mod']}k{r['k']}: base={r['ber_base']:.4f} lwm={r['ber_lwm']:.4f} d={r['ber_lwm']-r['ber_base']:+.4f} "
          f"(corr base {r['corr_base']:.3f}/lwm {r['corr_lwm']:.3f})")

# noPT 失败点
losses_np = sum(1 for r in R if r["ber_lwm_nopt"] > r["ber_base"] + 1e-9)
print(f"lwm_noPT 败给 base {losses_np}/{N}")

# ---------- 3. 主 vs 对照 ----------
print("\n===== 3. 主模型 vs 对照（逐样本差值统计）=====")
diffs = [r["ber_lwm"] - r["ber_lwm_nopt"] for r in R]
print(f"lwm - noPT 均值 {avg(diffs):+.5f}（负=主更优），中位 {st.median(diffs):+.5f}")
main_better = sum(1 for d in diffs if d < -1e-9)
print(f"主更优 {main_better}/{N}，对照更优 {sum(1 for d in diffs if d > 1e-9)}/{N}")
# 差距较大的样本
print("主 vs 对照差距最大样本:")
for r in sorted(R, key=lambda r: abs(r["ber_lwm"] - r["ber_lwm_nopt"]), reverse=True)[:6]:
    print(f"  SNR{r['snr_db']:>4.0f} {r['n_rx']}ant {r['n_sc']}sc {r['n_symb']}sym TDL-{r['tdl']} {r['max_speed']}m/s: "
          f"lwm={r['ber_lwm']:.4f} noPT={r['ber_lwm_nopt']:.4f}")

# ---------- 4. 按配置维度：lwm 相对 base 的改善率 ----------
print("\n===== 4. 按配置维度改善率（全 SNR 平均 BER 降幅）=====")
def dim_gain(key, fmt=None):
    groups = defaultdict(list)
    for r in R:
        groups[r[key]].append(r)
    out = []
    for v in sorted(groups):
        rs = groups[v]
        b, l = avg([r["ber_base"] for r in rs]), avg([r["ber_lwm"] for r in rs])
        out.append((fmt(v) if fmt else v, len(rs), b, l, (1 - l / b) * 100))
    return out

for title, key, fmt in [
    ("天线", "n_rx", None), ("RB", "n_sc", lambda v: f"{v//12}RB"),
    ("符号", "n_symb", None), ("DMRS", "dmrs_ap", lambda v: {0:'{1}',1:'{1+1}',2:'{1+2}'}[v]),
    ("TDL", "tdl", None), ("速度", "max_speed", None)]:
    print(f"-- {title} --")
    for v, n, b, l, g in dim_gain(key, fmt):
        print(f"   {str(v):>8} n={n:>3} base={b:.4f} lwm={l:.4f} 改善 {g:>5.1f}%")

# ---------- 5. LLR 相关性按 SNR ----------
print("\n===== 5. LLR 相关系数（base vs lwm）按 SNR =====")
for snr in sorted(by_snr):
    rs = by_snr[snr]
    cb, cl, cp = avg([r["corr_base"] for r in rs]), avg([r["corr_lwm"] for r in rs]), avg([r["corr_lwm_nopt"] for r in rs])
    print(f"  SNR{snr:>3}: base={cb:.4f} lwm={cl:.4f} noPT={cp:.4f}")

# ---------- 6. 高 SNR 错误残留 ----------
print("\n===== 6. 高 SNR 残差（15/20dB 仍出错样本）=====")
cnt = 0
for r in R:
    if r["snr_db"] >= 15 and r["ber_lwm"] > 1e-9:
        cnt += 1
        print(f"  SNR{r['snr_db']:.0f} {r['n_rx']}ant {r['n_sc']}sc {r['n_symb']}sym dmrs{r['dmrs_ap']} TDL-{r['tdl']} "
              f"{r['max_speed']}m/s mod{r['mod']}k{r['k']}: base={r['ber_base']:.4f} lwm={r['ber_lwm']:.4f}")
print(f"  15/20dB 残差样本数: {cnt}/32")
