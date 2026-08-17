# -*- coding: utf-8 -*-
"""
PyTorch Dataset / collate：Sionna PUSCH 样本 -> 训练 batch

样本格式（data_gen_sionna 输出，3D 信道）：
  H_est(8,120,14)complex, H_true(8,120,14)complex, z(n_data,)complex,
  sigma2, sigma2_eq(n_data,), llr_ref(n_data,log2M), bits_tx(n_data,log2M),
  mod_order, n_sc, n_symb, n_data, data_re_idx

n_data = 1440（12 个数据符号 × 120 子载波，DMRS 符号 2/11 全导频）。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from data_gen import qam_constellation, demap_llr
from tokenizer import data_re_index

DATA_RE_IDX = data_re_index()     # (1440, 2) [sc, symb]
N_DATA = len(DATA_RE_IDX)
N_SC_3D = 120
N_SYMB_3D = 14


def pack_samples(samples):
    """
    把 Sionna 样本列表打包成 numpy 数组。
    返回 dict：
      H_est (n, 8, 120, 14) complex, z (n, 1440) complex,
      llr (n, 1440, 8), llr_base (n, 1440, 8), bits (n, 1440, 8),
      mod_oh (n, 4), sigma2 (n,), valid (n, 1440), nbits (n,)
    llr_base: 传统均衡后软解调基线（残差学习锚点）
    """
    n = len(samples)
    H_est = np.zeros((n, config.N_ANT, N_SC_3D, N_SYMB_3D), dtype=np.complex64)
    z = np.zeros((n, N_DATA), dtype=np.complex64)
    llr = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.float32)
    llr_base = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.float32)
    bits = np.zeros((n, N_DATA, config.MAX_BITS), dtype=np.int8)
    mod_oh = np.zeros((n, config.MOD_ONHOT_DIM), dtype=np.float32)
    sigma2 = np.zeros(n, dtype=np.float32)
    valid = np.ones((n, N_DATA), dtype=np.float32)
    nbits = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(samples):
        H_est[i] = s["H_est"]
        z[i] = s["z"]
        k = s["llr_ref"].shape[1]
        llr[i, :, :k] = s["llr_ref"]
        bits[i, :, :k] = s["bits_tx"]
        mod = int(s["mod_order"])
        X, btab = qam_constellation(mod)
        llr_base[i, :, :k] = demap_llr(s["z"], s["sigma2_eq"], X, btab, config.MAX_LLR)
        mod_oh[i, config.MOD_ORDERS.index(mod)] = 1.0
        sigma2[i] = s["sigma2"]
        nbits[i] = k

    return {
        "H_est": H_est, "z": z, "llr": llr, "llr_base": llr_base, "bits": bits,
        "mod_oh": mod_oh, "sigma2": sigma2, "valid": valid, "nbits": nbits,
    }


def pack_pretrain(samples):
    """预训练只需信道矩阵 -> (n, 8, 120, 14) 复数数组"""
    n = len(samples)
    H = np.zeros((n, config.N_ANT, N_SC_3D, N_SYMB_3D), dtype=np.complex64)
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
    """阶段 1 继续预训练数据集（仅信道，3D）"""
    def __init__(self, H_arr):
        self.H = torch.tensor(H_arr)

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, i):
        return self.H[i]
