# -*- coding: utf-8 -*-
"""
3GPP 兼容 OFDM 数据生成器

生成流程（对齐设计文档第 3 节）：
  1. TDL-C 多径信道 -> 频域响应 H ∈ C^(N_ant × N_sc)
  2. QAM 调制发送符号 x（Gray 编码）
  3. y = H·x + n（AWGN，SNR 给定 -> σ²）
  4. DM-RS(comb-4) LS 信道估计 + 线性插值 -> H_est（含估计误差）
  5. MMSE 均衡 -> 软符号 z
  6. 用理想 H 与 σ² 计算 max-log 参考 LLR（监督标签）

说明：H 按样本 Frobenius 归一化到 E||H||²_F=1，σ²=10^(-SNR/10) 与之匹配。
"""
import numpy as np

# ---- 3GPP TR 38.900 TDL-C 抽头（时延 µs，功率 dB） ----
TDL_C_DELAYS_US = np.array([0.0, 0.2673, 0.8019, 1.2029, 2.4048, 3.4737])
TDL_C_POWERS_DB = np.array([0.0, -0.6, -4.9, -8.0, -9.8, -13.9])


# ================= QAM 星座（Gray 映射，能量归一化） =================

def _gray(n):
    return n ^ (n >> 1)


def _ungray(n):
    m = n
    while n >> 1:
        n >>= 1
        m ^= n
    return m


def qam_constellation(m):
    """
    生成 Gray 映射 QAM 星座。
    返回 (X, bits):
      X    : (M,) 复数星座点，E[|x|²]=1
      bits : (M, log2M) 0/1 标签（每符号的比特）
    符号索引 s 的比特向量 = s 的二进制（高位在前）；几何上相邻星座点 Gray 相邻。
    """
    k = int(np.log2(m))
    side = int(np.sqrt(m))
    hk = k // 2
    X = np.zeros(m, dtype=np.complex128)
    bits = np.zeros((m, k), dtype=np.int8)
    for s in range(m):
        i_idx = s >> hk          # I 轴 gray 索引
        q_idx = s & (side - 1)   # Q 轴 gray 索引
        i = _ungray(i_idx)
        q = _ungray(q_idx)
        X[s] = (2 * i - side + 1) + 1j * (2 * q - side + 1)
        for b in range(k):
            bits[s, k - 1 - b] = (s >> b) & 1
    X /= np.sqrt(np.mean(np.abs(X) ** 2))  # 能量归一化
    return X, bits


# ================= 信道 =================

def gen_channel(n_ant, n_sc, rng, scs_khz=30):
    """
    TDL-C 频域信道响应 H ∈ C^(n_ant × n_sc)。
    H[a, f] = sum_p g_p[a] * exp(-j2π f Δf τ_p)
    """
    powers = 10.0 ** (TDL_C_POWERS_DB / 10.0)
    powers /= powers.sum()
    n_taps = len(powers)
    g = np.sqrt(powers)[:, None] * (
        rng.standard_normal((n_taps, n_ant)) + 1j * rng.standard_normal((n_taps, n_ant))
    ) / np.sqrt(2.0)
    freqs = np.arange(n_sc) * (scs_khz * 1e3)  # Hz
    H = np.zeros((n_ant, n_sc), dtype=np.complex128)
    for p in range(n_taps):
        phase = np.exp(-2j * np.pi * freqs * TDL_C_DELAYS_US[p] * 1e-6)
        H += g[p][:, None] * phase[None, :]
    # 归一化 E|H|² = 1
    H /= np.sqrt(np.mean(np.abs(H) ** 2))
    return H


# ================= 信道估计（DM-RS comb-4 + DFT 去噪插值） =================

def pilot_estimation(y, pilot_spacing=4, n_taps=8):
    """
    基于 DM-RS 的 LS-DFT 信道估计（经典 OFDM 信道估计，5G 接收机标准做法）：
      1. 导频处 LS 估计（导频符号 = 1）: H_p = y_pilot
      2. IDFT 到时域，截断保留前 n_taps 个抽头（去噪）
      3. DFT 插值回全子载波
    相比线性插值，利用信道时延有限性，估计误差显著更低、更贴近真实系统。
    y: (N_ant, N_sc) complex -> H_est (N_ant, N_sc) complex
    """
    n_ant, n_sc = y.shape
    n_p = n_sc // pilot_spacing
    pilot_idx = np.arange(0, n_sc, pilot_spacing)
    H_p = y[:, pilot_idx]                       # (n_ant, n_p)
    H_est = np.zeros_like(y)
    for a in range(n_ant):
        h_t = np.fft.ifft(H_p[a])               # 时域 (n_p,)
        h_t[n_taps:] = 0.0                      # 去噪（截断抽头）
        h_ext = np.zeros(n_sc, dtype=np.complex128)
        h_ext[:n_p] = h_t
        H_est[a] = np.fft.fft(h_ext)            # DFT 插值（导频处严格等于 LS 估计）
    return H_est


# ================= MMSE 均衡 =================

def mmse_equalize(y, H_est, sigma2):
    """
    MMSE 均衡（逐子载波，8 天线 -> 1 流）。
    z_k = w_k^H y_k,  w_k = (h h^H + σ²I)^{-1} h
    返回 z (N_sc,), sigma2_eq (N_sc,)
    """
    n_ant, n_sc = y.shape
    eye = np.eye(n_ant, dtype=np.complex128)
    z = np.zeros(n_sc, dtype=np.complex128)
    sig2_eq = np.zeros(n_sc)
    for k in range(n_sc):
        h = H_est[:, k]
        hh = np.outer(h, h.conj())
        w = np.linalg.solve(hh + sigma2 * eye, h)
        z[k] = np.vdot(w, y[:, k])
        sig2_eq[k] = sigma2 * np.vdot(w, w).real
    return z, sig2_eq


# ================= max-log LLR =================

def maxlog_llr(y, h, sigma2, X, bits, max_llr=20.0):
    """
    多天线 max-log LLR（直接对接收信号，无需均衡）：
    L(b_i) ≈ (1/σ²)[min_{x∈X_i^1}||y-hx||² - min_{x∈X_i^0}||y-hx||²]
    y: (N_ant, N_sc); h: (N_ant, N_sc); X: (M,); bits: (M, k)
    返回 LLR (N_sc, k)
    """
    n_ant, n_sc = y.shape
    M, k = bits.shape
    hX = h[..., None] * X[None, None, :]          # (N_ant, N_sc, M)
    d = np.abs(y[..., None] - hX) ** 2            # (N_ant, N_sc, M)
    d = d.sum(axis=0)                             # (N_sc, M)
    llr = np.zeros((n_sc, k))
    for b in range(k):
        d0 = d[:, bits[:, b] == 0]
        d1 = d[:, bits[:, b] == 1]
        llr[:, b] = (d0.min(axis=1) - d1.min(axis=1)) / sigma2
    return np.clip(llr, -max_llr, max_llr)


def demap_llr(z, sigma2_eq, X, bits, max_llr=20.0):
    """
    均衡后软解调 max-log LLR（传统接收机标准 demapper）：
    L(b_i) ≈ (1/σ_z²)[min_{x∈X_i^1}|z-x|² - min_{x∈X_i^0}|z-x|²]
    z: (N_sc,) 均衡软符号; sigma2_eq: (N_sc,) 均衡后等效噪声方差
    X: (M,); bits: (M, k)
    """
    M, k = bits.shape
    z = np.asarray(z)
    d = np.abs(z[:, None] - X[None, :]) ** 2      # (N_sc, M)
    llr = np.zeros((z.shape[0], k))
    for b in range(k):
        d0 = d[:, bits[:, b] == 0]
        d1 = d[:, bits[:, b] == 1]
        llr[:, b] = (d0.min(axis=1) - d1.min(axis=1)) / np.maximum(sigma2_eq, 1e-12)
    return np.clip(llr, -max_llr, max_llr)


# ================= 单样本生成 =================

def data_subcarrier_idx(n_sc, pilot_spacing):
    """数据子载波索引（非导频位置）；导频在 k % spacing == 0"""
    return np.array([k for k in range(n_sc) if k % pilot_spacing != 0], dtype=int)


def generate_sample(rng, n_ant, n_sc, mod_order, snr_db, pilot_spacing=4,
                    n_taps=8, max_llr=20.0):
    """
    生成一个完整样本（3GPP 兼容 OFDM 链路仿真）：
      - 导频子载波（comb-spacing）发送已知符号 1，用于 LS-DFT 信道估计
      - 数据子载波发送 QAM 符号，MMSE 均衡 + max-log LLR 只在数据子载波计算
    返回 dict：
      H_est   : (n_ant, n_sc) complex  信道估计（全子载波，模型输入）
      H_true  : (n_ant, n_sc) complex  真实信道（评估用）
      z       : (n_data,) complex      数据子载波的均衡软符号（模型输入）
      sigma2  : float                  噪声方差（模型输入）
      llr_ref : (n_data, log2M) float  参考 LLR（监督标签，仅数据子载波）
      bits_tx : (n_data, log2M) int8   发送比特（评估用）
      mod_order: int
      n_sc    : int
      n_data  : int                    数据子载波数
    """
    X, bits = qam_constellation(mod_order)
    k = bits.shape[1]

    H_true = gen_channel(n_ant, n_sc, rng)
    nrm = np.sqrt(np.mean(np.abs(H_true) ** 2))   # 每样本归一化（与 σ² 匹配）
    H_true = H_true / nrm

    # 发送符号：导频位置 = 1，数据位置 = QAM
    data_idx = data_subcarrier_idx(n_sc, pilot_spacing)
    n_data = len(data_idx)
    sym_idx = rng.integers(0, mod_order, size=n_data)
    x = np.ones(n_sc, dtype=np.complex128)
    x[data_idx] = X[sym_idx]
    bits_tx = bits[sym_idx]               # (n_data, k)

    # 噪声
    sigma2 = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal((n_ant, n_sc)) + 1j * rng.standard_normal((n_ant, n_sc))
    )
    y = H_true * x[None, :] + noise       # (n_ant, n_sc)

    # 信道估计（导频 LS + DFT 去噪插值，含估计误差）
    H_est = pilot_estimation(y, pilot_spacing, n_taps)

    # MMSE 均衡 + 参考 LLR：仅数据子载波
    y_d = y[:, data_idx]
    H_est_d = H_est[:, data_idx]
    H_true_d = H_true[:, data_idx]
    z, sigma2_eq = mmse_equalize(y_d, H_est_d, sigma2)
    llr_ref = maxlog_llr(y_d, H_true_d, sigma2, X, bits, max_llr)

    return {
        "H_est": H_est.astype(np.complex64),
        "H_true": H_true.astype(np.complex64),
        "y": y.astype(np.complex64),
        "z": z.astype(np.complex64),
        "sigma2_eq": sigma2_eq.astype(np.float32),
        "sigma2": np.float32(sigma2),
        "llr_ref": llr_ref.astype(np.float32),
        "bits_tx": bits_tx.astype(np.int8),
        "mod_order": np.int32(mod_order),
        "n_sc": np.int32(n_sc),
        "n_data": np.int32(n_data),
        "data_idx": data_idx.astype(np.int32),
    }


# ================= 批量生成 =================

def generate_dataset(n_samples, n_sc=128, mod_orders=None, snr_db=None,
                     seed=0, pilot_spacing=4):
    """
    批量生成样本列表。SNR 与调制阶数随机采样（若未指定）。
    """
    rng = np.random.default_rng(seed)
    if mod_orders is None:
        mod_orders = [4, 16, 64, 256]
    samples = []
    for _ in range(n_samples):
        m = mod_orders[rng.integers(0, len(mod_orders))]
        if snr_db is None:
            s = rng.uniform(-5, 25)
        else:
            s = snr_db
        samples.append(generate_sample(rng, 8, n_sc, m, float(s), pilot_spacing))
    return samples


if __name__ == "__main__":
    # 自测
    s = generate_sample(np.random.default_rng(0), 8, 128, 16, 10.0)
    print("sample keys:", sorted(s.keys()))
    print("H_est:", s["H_est"].shape, s["H_est"].dtype)
    print("z:", s["z"].shape, "sigma2:", s["sigma2"])
    print("llr_ref:", s["llr_ref"].shape, "range:", s["llr_ref"].min(), s["llr_ref"].max())
    print("bits_tx:", s["bits_tx"].shape, s["bits_tx"].dtype)
    X, bits = qam_constellation(16)
    print("16QAM energy:", np.mean(np.abs(X) ** 2))
    # 验证 LLR 符号与比特一致性（高 SNR 下硬判决应接近）
    s2 = generate_sample(np.random.default_rng(7), 8, 64, 4, 25.0)
    hard = (s2["llr_ref"] > 0).astype(int)
    acc = np.mean(hard == s2["bits_tx"])
    print("QPSK@25dB LLR hard-decision accuracy:", acc)
    print("OK")
