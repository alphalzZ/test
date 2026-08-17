# -*- coding: utf-8 -*-
"""
Tokenizer：信道矩阵 -> 子载波对齐 patch 序列（设计文档第 4 节）

- 每个 patch = 单个子载波上的 8 天线空间向量 [Re(h_k); Im(h_k)] ∈ R^16
- element_length = 16 与原生 LWM 一致，可直接复用官方预训练 embedding 权重
- 分块：BLOCK_SIZE=128 子载波/块（+CLS token 后序列长 129 = 原生 MAX_LEN）
- 长度不足补零；分块输出按块保存，下游逐子载波输出可拼回

3D 版本（Sionna PUSCH）：
  H (8, N_sc, N_symb) -> 逐 OFDM 符号独立编码（每符号一个 (129,16) 序列）
"""
import numpy as np

import config

ELEMENT_LENGTH = 16
# LWM 原生序列长度：128 子载波 + CLS = 129（位置编码上限）
# Sionna PUSCH 的 120 子载波会自动 pad 到 128
BLOCK_SIZE = 128
CLS_TOKEN = 0.2 * np.ones((ELEMENT_LENGTH,), dtype=np.float32)

# Sionna PUSCH 数据 RE 索引（num_cdm_groups_without_data=2, DMRS 符号 2/11 全导频）
N_SC_SIONNA = 120
N_SYMB_SIONNA = 14
DMRS_SYMBS = (2, 11)


def data_re_index(n_sc=N_SC_SIONNA, n_symb=N_SYMB_SIONNA, dmrs_symbs=DMRS_SYMBS):
    """数据 RE 索引 [(sc, symb), ...]"""
    idx = [(sc, sy) for sy in range(n_symb) for sc in range(n_sc)
           if sy not in dmrs_symbs]
    return np.array(idx, dtype=np.int32)   # (n_data, 2)


def channel_to_patches(H):
    """
    H: (N_ant, N_sc) complex -> patches (N_sc, 16) float32
    patch_k = [Re(H[:,k]), Im(H[:,k])]
    """
    H = np.asarray(H)
    real = H.real.T   # (N_sc, N_ant)
    imag = H.imag.T
    return np.concatenate([real, imag], axis=1).astype(np.float32)


def tokenize_blocks(H, block_size=BLOCK_SIZE):
    """
    H: (N_ant, N_sc) complex -> list of blocks, 每块 (129, 16) float32（含 CLS）
    尾块不足补零。返回 (blocks, masks)，masks 标记每块真实子载波数（不含 CLS）。
    """
    patches = channel_to_patches(H)          # (N_sc, 16)
    n_sc = patches.shape[0]
    blocks, masks = [], []
    for start in range(0, n_sc, block_size):
        blk = patches[start:start + block_size]
        n_real = blk.shape[0]
        if n_real < block_size:
            pad = np.zeros((block_size - n_real, ELEMENT_LENGTH), dtype=np.float32)
            blk = np.concatenate([blk, pad], axis=0)
        seq = np.concatenate([CLS_TOKEN[None, :], blk], axis=0)   # (129, 16)
        blocks.append(seq)
        m = np.ones((block_size,), dtype=np.float32)
        m[n_real:] = 0.0
        masks.append(m)
    return blocks, masks


def tokenize_3d(H, block_size=BLOCK_SIZE):
    """
    H: (N_ant, N_sc, N_symb) complex -> (blocks, symb_of_block)
    逐 OFDM 符号独立 tokenize（每符号 1 块）。
    blocks: (n_symb, 129, 16)
    """
    H = np.asarray(H)
    n_symb = H.shape[2]
    blocks = []
    for s in range(n_symb):
        blk, _ = tokenize_blocks(H[:, :, s], block_size)
        blocks.append(blk[0])
    return np.stack(blocks)   # (n_symb, 129, 16)


def make_mcm_batch(blocks, mask_ratio=0.15, rng=None):
    """
    构造 MCM 训练 batch。
    blocks: list of (129, 16)
    返回 (input_ids (B,129,16), masked_tokens (B,n_mask,16), masked_pos (B,n_mask))
    掩码策略（对齐原生 LWM）：每块随机 mask 15% 的 patch（不 mask CLS）。
    """
    if rng is None:
        rng = np.random.default_rng()
    B = len(blocks)
    n_patches = BLOCK_SIZE
    n_mask = int(np.floor(mask_ratio * n_patches))
    input_ids = np.stack(blocks).astype(np.float32)          # (B, 129, 16)
    masked_tokens = np.zeros((B, n_mask, ELEMENT_LENGTH), dtype=np.float32)
    masked_pos = np.zeros((B, n_mask), dtype=np.int64)
    word2mask = 0.1 * np.ones((ELEMENT_LENGTH,), dtype=np.float32)
    for b in range(B):
        pos = rng.choice(np.arange(1, n_patches + 1), size=n_mask, replace=False)
        pos.sort()
        masked_pos[b] = pos
        masked_tokens[b] = input_ids[b, pos]
        for i, p in enumerate(pos):
            r = rng.random()
            if r < 0.8:
                input_ids[b, p] = word2mask
            elif r < 0.9:
                input_ids[b, p] = rng.random(ELEMENT_LENGTH).astype(np.float32)
    return input_ids, masked_tokens, masked_pos


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H = rng.standard_normal((8, 128)) + 1j * rng.standard_normal((8, 128))
    blocks, masks = tokenize_blocks(H)
    print("blocks:", len(blocks), "block shape:", blocks[0].shape)
    inp, mt, mp = make_mcm_batch(blocks, rng=rng)
    print("input_ids:", inp.shape, "masked_tokens:", mt.shape, "masked_pos:", mp.shape)
    assert np.all(mp >= 1) and np.all(mp <= 128)

    # 3D 测试
    H3 = rng.standard_normal((8, 120, 14)) + 1j * rng.standard_normal((8, 120, 14))
    blocks3 = tokenize_3d(H3)
    print("3D blocks:", blocks3.shape)
    idx = data_re_index()
    print("data_re_idx:", idx.shape, "n_data:", len(idx))
    print("tokenizer OK")

