# -*- coding: utf-8 -*-
"""
传统 OFDM 接收机基线（移植自 wirelessLearning/src/ofdm_rx.py 的接收处理链）

参考实现（wirelessLearning/src/ofdm_rx.py）的接收处理流程，在频域资源网格上实现：
  1. 信道估计：导频处 LS + 子载波线性插值 + **滑动平均平滑**（窗口按调制阶数）
     —— 输入 h_est 为已插值的 LS 估计（PUSCHLSChannelEstimator 输出），
        本模块在其上做 ofdm_rx 风格的子载波滑动平均平滑；
  2. 噪声方差估计：**导频残差** |y - h·x_pilot|² → 子载波线性插值 + 滑动平均平滑
     + 符号维线性插值（ofdm_rx 的 noise_var_estimate，而非理想 AWGN 假设）；
  3. MMSE 均衡：逐数据 RE，W = h/(|h|²+σ²)，x̂ = Wᴴy，σ_eq² = σ²·|h|²/(|h|²+σ²)²
     （ofdm_rx 的 channel_equalization，MMSE 方法）；
  4. 软解调：**APP LLR**（logsumexp 精确 MAP，ofdm_rx 的 qam_demodulation
     return_llr=True），星座/比特约定与本项目 qam_constellation 一致（LLR>0 → bit1）。

与 Sionna 标准基线（LMMSEEqualizer + maxlog）的差异：
  - 信道估计多一步滑动平均平滑（经典接收机做法）；
  - 噪声由导频残差估计（而非 err_var/理想噪声模型）；
  - LLR 用 APP 精确解调（而非 max-log 近似）。

输入（单个样本，与 generate_batch 的批量维度对应）：
  y_rx      : (n_rx, n_symb, fft) complex  接收频域网格（数据 RE 含噪声）
  h_est     : (n_rx, n_symb, fft) complex  LS 信道估计（导频 LS + 插值后）
  pilot_mask: (n_symb, fft) 0/1           导频位置掩码
  x_pilots  : (n_pilots,) complex         导频发射符号（与 pilot 位置对应）
  data_re_idx: (n_data, 2) int [sc, symb] 数据 RE 索引（本项目 sc-major 约定）
  mod_order : int                          调制阶数 4/16/64/256
返回：
  llr : (n_data, log2M) float32   APP LLR，LLR>0 → bit1
"""
import numpy as np

from src.datasets.demap import qam_constellation


def _logsumexp(a, axis):
    """数值稳定 logsumexp（ofdm_rx.qam_demodulation 同款）"""
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(a_max, axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis))


def _sliding_average(x, half, extra):
    """子载波维滑动平均：对每个 j 取 [j-half, j+half+extra) 窗口均值（ofdm_rx 平滑）"""
    n = x.shape[-1]
    out = np.zeros_like(x)
    for j in range(n):
        start = max(0, j - half)
        end = min(n, j + half + extra)
        out[..., j] = np.mean(x[..., start:end], axis=-1)
    return out


def _interp_pilot_to_grid(values_pilot, pilot_sc, n_sc, pilot_sym_indices,
                          n_symb, interp_sym=True):
    """
    ofdm_rx 风格：导频子载波值 -> 全子载波线性插值 -> 滑动平均前的网格。
    values_pilot: (n_pilot_sym, n_pilot_sc)
    返回 (n_symb, n_sc)（符号维线性插值到全部符号；导频符号数=1 时保持）。
    """
    n_ps = len(pilot_sym_indices)
    grid_pilot_sym = np.zeros((n_ps, n_sc), dtype=values_pilot.dtype)
    for i, sc in enumerate(pilot_sc):
        # 该子载波出现在每个导频符号
        grid_pilot_sym[:, sc] = values_pilot[:, i]
    # 非导频子载波线性插值
    full = np.zeros((n_ps, n_sc), dtype=values_pilot.dtype)
    for sc in range(n_sc):
        if sc in pilot_sc:
            full[:, sc] = grid_pilot_sym[:, sc]
        else:
            left = pilot_sc[pilot_sc < sc]
            right = pilot_sc[pilot_sc > sc]
            if len(left) and len(right):
                l, r = left[-1], right[0]
                alpha = (sc - l) / (r - l)
                full[:, sc] = ((1 - alpha) * grid_pilot_sym[:, l]
                               + alpha * grid_pilot_sym[:, r])
            elif len(left):
                full[:, sc] = grid_pilot_sym[:, left[-1]]
            else:
                full[:, sc] = grid_pilot_sym[:, right[0]]
    if not interp_sym or n_ps == 1 or n_ps == n_symb:
        if n_ps == n_symb:
            return full
        return np.repeat(full, n_symb // n_ps, axis=0)[:n_symb] \
            if n_symb % n_ps == 0 else np.repeat(full, n_symb, axis=0)[:n_symb]
    # 符号维线性插值（ofdm_rx interp_method='linear'）
    symbol_range = np.arange(n_symb)
    left_idx = np.searchsorted(pilot_sym_indices, symbol_range, side="right") - 1
    right_idx = np.clip(left_idx + 1, 0, n_ps - 1)
    left_idx = np.clip(left_idx, 0, n_ps - 1)
    left_pos = np.asarray(pilot_sym_indices)[left_idx]
    right_pos = np.asarray(pilot_sym_indices)[right_idx]
    denom = right_pos - left_pos
    denom[denom == 0] = 1
    alpha = ((symbol_range - left_pos) / denom)[:, None]
    return ((1 - alpha) * full[left_idx] + alpha * full[right_idx])


def legacy_ofdm_rx_llr(y_rx, h_est, pilot_mask, x_pilots, data_re_idx,
                       mod_order, pilot_spacing=2, win_size=None):
    """
    ofdm_rx 风格传统接收机：信道平滑 + 导频残差噪声估计 + MMSE + APP LLR。
    """
    n_rx, n_symb, n_sc = y_rx.shape
    pm = np.asarray(pilot_mask)
    # 导频位置 [symb, sc]（与 x_pilots 一一对应，按 symb-major 顺序）
    pil_idx = np.argwhere(pm > 0)                    # (n_pilots, 2) = [symb, sc]
    assert len(pil_idx) == len(x_pilots), "导频位置与导频符号数量不一致"
    pilot_sym_indices = sorted(set(pil_idx[:, 0]))
    n_ps = len(pilot_sym_indices)
    pilot_sc = np.array([sc for _, sc in pil_idx[:len(set(pil_idx[:, 1]))]])
    # 注意：不同导频符号的导频子载波集合相同（type1 DMRS 每符号同位置）
    k = int(np.log2(mod_order))
    qm = k
    if win_size is None:
        win_size = {2: 8, 4: 4}.get(qm, 2)          # ofdm_rx cfg.win_size=[8,4,2]

    # ---- 1. 信道估计：子载波滑动平均平滑（ofdm_rx estimate_channel 的平滑步） ----
    h_smooth = _sliding_average(h_est, pilot_spacing, win_size)

    # ---- 2. 噪声方差估计：导频残差 -> 插值 -> 平滑（ofdm_rx noise_var_estimate） ----
    noise_pilot = np.zeros((n_ps, len(pilot_sc)), dtype=np.float64)
    xp = np.asarray(x_pilots)
    for i, sp in enumerate(pilot_sym_indices):
        sel = pil_idx[:, 0] == sp                    # 该导频符号的导频子载波
        scs = pil_idx[sel, 1]
        xs = xp[sel]
        res = y_rx[:, sp, scs] - h_est[:, sp, scs] * xs[None, :]   # (n_rx, n_pilot_sc)
        noise_pilot[i] = np.mean(np.abs(res) ** 2, axis=0)
    noise_grid = _interp_pilot_to_grid(
        noise_pilot, pilot_sc, n_sc, pilot_sym_indices, n_symb)
    noise = _sliding_average(noise_grid, pilot_spacing, win_size)
    noise = np.maximum(noise, 1e-12)

    # ---- 3. MMSE 均衡（逐数据 RE，ofdm_rx channel_equalization） ----
    n_data = len(data_re_idx)
    z = np.zeros(n_data, dtype=np.complex128)
    sig2_eq = np.zeros(n_data, dtype=np.float64)
    for i, (sc, sy) in enumerate(data_re_idx):
        h = h_smooth[:, sy, sc]                      # (n_rx,)
        r = y_rx[:, sy, sc]                          # (n_rx,)
        n0 = noise[sy, sc]
        h2 = float(np.sum(np.abs(h) ** 2))           # |h|²（SIMO 合并功率）
        denom = h2 + n0
        z[i] = np.vdot(h, r) / denom                 # x̂ = hᴴy/(|h|²+σ²)（MMSE 有偏）
        sig2_eq[i] = n0 * h2 / denom ** 2            # σ_eq² = σ²|h|²/(|h|²+σ²)²

    # ---- 4. APP LLR（ofdm_rx qam_demodulation return_llr=True） ----
    X, btab = qam_constellation(mod_order)
    d2 = np.abs(z[:, None] - X[None, :]) ** 2 / sig2_eq[:, None]   # (n_data, M)
    metric = -d2
    llr = np.zeros((n_data, k), dtype=np.float32)
    for b in range(k):
        m0 = _logsumexp(metric[:, btab[:, b] == 0], axis=1)
        m1 = _logsumexp(metric[:, btab[:, b] == 1], axis=1)
        llr[:, b] = m1 - m0                          # 正 LLR -> bit1（本项目约定）
    return np.clip(llr, -20.0, 20.0).astype(np.float32)
