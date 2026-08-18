# -*- coding: utf-8 -*-
"""
工具函数：QAM 星座（Gray 映射）与均衡后软解调 max-log LLR（传统基线 demapper）
"""
import numpy as np


def _ungray(n):
    x = n
    while n >> 1:
        n >>= 1
        x ^= n
    return x


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


def demap_llr(z, sigma2_eq, X, bits, max_llr=20.0):
    """
    均衡后软解调 max-log LLR（传统接收机标准 demapper，评估基线）：
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


if __name__ == "__main__":
    X, bits = qam_constellation(16)
    print("16QAM energy:", np.mean(np.abs(X) ** 2))
    z = np.array([0.1 + 0.2j, 1.0 + 1.0j])
    llr = demap_llr(z, np.array([0.1, 0.1]), X, bits)
    print("demap_llr:", llr.shape)
    print("OK")
