# -*- coding: utf-8 -*-
"""
Tokenizer：信道矩阵 -> 子载波对齐 patch 序列

- 每个 patch = 单个子载波上的天线空间向量 [Re(H[:,k]); Im(H[:,k])] ∈ R^16
  （天线不足 8 补零，维度恒为 16，与原生 LWM element_length 一致）
- 序列 = [CLS] + patches，长度 = n_sc + 1（自适应，≤121 ≤ LWM MAX_LEN=129）
- 多配置自适应：1~10 RB（12~120 子载波）直接编码，无需 padding
"""
import numpy as np

import config

ELEMENT_LENGTH = 16
CLS_TOKEN = 0.2 * np.ones((ELEMENT_LENGTH,), dtype=np.float32)

# Sionna PUSCH 数据 RE 索引的默认网格（data_re_index 仅用于工具/测试）
N_SC_SIONNA = 120
N_SYMB_SIONNA = 14
DMRS_SYMBS = (2, 11)


def data_re_index(n_sc=N_SC_SIONNA, n_symb=N_SYMB_SIONNA, dmrs_symbs=DMRS_SYMBS):
    """数据 RE 索引 [(sc, symb), ...]（仅作工具/冒烟测试；训练数据用 Sionna pilot mask）"""
    idx = [(sc, sy) for sy in range(n_symb) for sc in range(n_sc)
           if sy not in dmrs_symbs]
    return np.array(idx, dtype=np.int32)   # (n_data, 2)


def tokenize_3d_var(H):
    """
    H: (N_ant, N_sc, N_symb) complex（N_ant<=8, N_sc<=120, N_symb<=14）
    -> (n_symb, n_sc+1, 16) float32，序列 = [CLS] + 逐子载波 patch
    （不补零到 128，长度 n_sc+1 <= 121 <= LWM MAX_LEN=129，适配任意 RB 数）
    """
    H = np.asarray(H)
    n_ant, n_sc, n_symb = H.shape
    real = H.real.T                        # (n_sc, n_ant)
    imag = H.imag.T
    patches = np.concatenate([real, imag], axis=1).astype(np.float32)  # (n_sc, 2*n_ant)
    if n_ant < 8:                          # 天线补零到 8（16 维 patch，复用官方 embedding）
        pad = np.zeros((n_sc, 16 - 2 * n_ant), dtype=np.float32)
        patches = np.concatenate([patches, pad], axis=1)
    seq = np.concatenate([CLS_TOKEN[None, :], patches], axis=0)     # (n_sc+1, 16)
    return np.stack([seq] * n_symb)                  # (n_symb, n_sc+1, 16)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 多配置测试：1 天线 / 1 RB / 3 符号 与 8 天线 / 10 RB / 14 符号
    for n_ant, n_sc, n_symb in [(1, 12, 3), (8, 120, 14), (4, 48, 7)]:
        H = rng.standard_normal((n_ant, n_sc, n_symb)) + 1j * rng.standard_normal((n_ant, n_sc, n_symb))
        blocks = tokenize_3d_var(H)
        print(f"{n_ant}rx/{n_sc}sc/{n_symb}symb -> {blocks.shape}")
    print("tokenizer OK")
