# -*- coding: utf-8 -*-
"""
PyTorch Dataset / collate：样本列表 -> 训练 batch
（设计文档第 3.4 节 schema 的 torch 落地）

训练样本固定 n_sc=128（= BLOCK_SIZE），数据子载波数 n_data = 96（comb-4）。
z / llr / bits 只含数据子载波（96）；H_est 含全部 128 子载波（供 LWM 建模）。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from data_gen import (data_subcarrier_idx, qam_constellation, demap_llr)

DATA_IDX_128 = data_subcarrier_idx(config.N_SC, config.PILOT_SPACING)  # 96 个
N_DATA = len(DATA_IDX_128)


def pack_samples(samples, block_size=config.N_SC):
    """
    把样本列表打包成便于训练的 numpy 数组。
    返回 dict（形状）：
      H_est (n, 8, 128) complex, z (n, 96) complex,
      llr (n, 96, 8), llr_base (n, 96, 8), bits (n, 96, 8),
      mod_oh (n, 4), sigma2 (n,), valid (n, 96), nbits (n,)
    llr_base: 传统均衡后软解调基线（残差学习锚点）
    """
    n = len(samples)
    H_est = np.zeros((n, config.N_ANT, block_size), dtype=np.complex64)
    z = np.zeros((n, N_DATA), dtype=np.complex64)
    llr = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.float32)
    llr_base = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.float32)
    bits = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.int8)
    mod_oh = np.zeros((n, config.MOD_ONHOT_DIM), dtype=np.float32)
    sigma2 = np.zeros(n, dtype=np.float32)
    valid = np.zeros((n, N_DATA), dtype=np.float32)
    nbits = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(samples):
        n_sc = int(s["n_sc"])
        assert n_sc == block_size, "pack_samples 要求 n_sc == block_size(128)"
        n_data = int(s["n_data"])
        assert n_data == N_DATA
        H_est[i] = s["H_est"]
        z[i, :n_data] = s["z"]
        k = s["llr_ref"].shape[1]
        llr[i, :n_data, :k] = s["llr_ref"]
        bits[i, :n_data, :k] = s["bits_tx"]
        # 基线 LLR：均衡后软解调（传统 demapper）
        mod = int(s["mod_order"])
        X, btab = qam_constellation(mod)
        llr_base[i, :n_data, :k] = demap_llr(s["z"], s["sigma2_eq"], X, btab, config.MAX_LLR)
        mod_oh[i, config.MOD_ORDERS.index(mod)] = 1.0
        sigma2[i] = s["sigma2"]
        valid[i, :n_data] = 1.0
        nbits[i] = k

    return {
        "H_est": H_est, "z": z, "llr": llr, "llr_base": llr_base, "bits": bits,
        "mod_oh": mod_oh, "sigma2": sigma2, "valid": valid, "nbits": nbits,
    }


def pack_pretrain(samples, block_size=config.N_SC):
    """预训练只需信道矩阵 -> (n, 8, 128) 复数数组（用带噪估计，与下游输入域一致）"""
    n = len(samples)
    H = np.zeros((n, config.N_ANT, block_size), dtype=np.complex64)
    for i, s in enumerate(samples):
        H[i] = s["H_est"]
    return H


class LLRDataset(Dataset):
    """阶段 2 微调数据集"""
    def __init__(self, packed):
        self.H_est = torch.tensor(packed["H_est"])
        self.z = torch.tensor(packed["z"])
        self.llr = torch.tensor(packed["llr"])
        self.llr_base = torch.tensor(packed["llr_base"])
        self.bits = torch.tensor(packed["bits"])
        self.mod_oh = torch.tensor(packed["mod_oh"])
        self.sigma2 = torch.tensor(packed["sigma2"])
        self.valid = torch.tensor(packed["valid"])
        self.nbits = torch.tensor(packed["nbits"])

    def __len__(self):
        return self.H_est.shape[0]

    def __getitem__(self, i):
        return (self.H_est[i], self.z[i], self.llr[i], self.llr_base[i],
                self.bits[i], self.mod_oh[i], self.sigma2[i],
                self.valid[i], self.nbits[i])


def llr_collate(batch):
    return tuple(torch.stack([b[j] for b in batch]) for j in range(len(batch[0])))


class PretrainDataset(Dataset):
    """阶段 1 继续预训练数据集（仅信道）"""
    def __init__(self, H_arr):
        self.H = torch.tensor(H_arr)

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, i):
        return self.H[i]
