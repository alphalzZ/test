# -*- coding: utf-8 -*-
"""
PyTorch Dataset / collate：Sionna PUSCH 样本 -> 训练 batch

样本格式（data_gen_sionna 输出，3D 信道）：
  H_est(8,120,14)complex, H_true(8,120,14)complex, z(n_data,)complex,
  sigma2, sigma2_eq(n_data,), llr_ref(n_data,log2M), bits_tx(n_data,log2M),
  mod_order, n_sc, n_symb, n_data, data_re_idx

n_data = 1440（12 个数据符号 × 120 子载波，DMRS 符号 2/11 全导频）。
"""
import os

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
    llr: 理想信道 max-log 参考 LLR（评估指标用）
    llr_base: 传统均衡后软解调基线（仅评估对比用；模型输入已不含 llr_base）
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


# ================= 数据缓存（Sionna 生成一次，训练复用） =================

def save_packed(path, packed):
    """packed dict of numpy arrays -> npz"""
    np.savez(path, **packed)


def load_packed(path):
    """npz -> dict of numpy arrays"""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def save_pretrain_cache(path, H_arr):
    np.savez(path, H_est=H_arr)


def load_pretrain_cache(path):
    with np.load(path) as d:
        return d["H_est"]


# ================= v2 多配置数据集（一个模型适配多种系统参数） =================

def build_cfg_vec(batch):
    """配置元数据向量 (B, CFG_DIM)：接收机已知的系统参数（帮助小数据下区分配置）"""
    n = len(batch)
    v = np.zeros((n, config.CFG_DIM), dtype=np.float32)
    ants = [1, 2, 4, 8]
    tdls = ["A", "B", "C", "D"]
    for i, s in enumerate(batch):
        v[i, 0:4] = [int(s["n_rx"]) == a for a in ants]
        v[i, 4] = int(s["n_sc"]) / 120.0
        v[i, 5] = int(s["n_symb"]) / 14.0
        v[i, 6:9] = [int(s["dmrs_ap"]) == a for a in [0, 1, 2]]
        v[i, 9:13] = [str(s["tdl"]) == t for t in tdls]
        v[i, 13] = float(s["max_speed"]) / 30.0
    return v


def collate_v2(batch):
    """
    合并一批**同 (n_sc, n_symb) 配置**的样本 dict -> 训练 batch dict（numpy）。
    比特/LLR 按 MAX_BITS=8 补齐，nbits 记录真实比特数（损失掩码用）。
    """
    B = len(batch)
    s0 = batch[0]
    n_data = s0["z"].shape[0]
    n_rx = s0["H_est"].shape[0]
    n_sc = s0["H_est"].shape[1]
    n_symb = s0["H_est"].shape[2]

    H_est = np.stack([s["H_est"] for s in batch])          # (B,n_rx,n_sc,n_symb)
    z = np.stack([s["z"] for s in batch])                  # (B,n_data)
    sigma2_eq = np.stack([s["sigma2_eq"] for s in batch])  # (B,n_data)
    llr = np.zeros((B, n_data, config.MAX_BITS), dtype=np.float32)
    bits = np.zeros((B, n_data, config.MAX_BITS), dtype=np.int8)
    mod_oh = np.zeros((B, config.MOD_ONHOT_DIM), dtype=np.float32)
    sigma2 = np.zeros(B, dtype=np.float32)
    nbits = np.zeros(B, dtype=np.int32)
    for i, s in enumerate(batch):
        k = int(np.log2(s["mod_order"]))
        llr[i, :, :k] = s["llr_ref"]
        bits[i, :, :k] = s["bits_tx"]
        mod_oh[i, config.MOD_ORDERS.index(int(s["mod_order"]))] = 1.0
        sigma2[i] = s["sigma2"]
        nbits[i] = k

    out = {
        "H_est": H_est, "z": z, "llr": llr, "bits": bits,
        "mod_oh": mod_oh, "sigma2": sigma2, "sigma2_eq": sigma2_eq,
        "valid": np.ones((B, n_data), dtype=np.float32),
        "nbits": nbits,
        "data_re_idx": np.stack([s["data_re_idx"] for s in batch]),
        "cfg_v": build_cfg_vec(batch),
        "n_sc": np.int32(n_sc), "n_symb": np.int32(n_symb),
        "n_rx": np.int32(n_rx), "n_data": np.int32(n_data),
    }
    for key in ("snr_db", "dmrs_ap", "tdl", "delay_spread", "max_speed", "mod_order"):
        if key in s0:
            out[key] = np.array([s[key] for s in batch])
    return out


class BucketedLoader:
    """
    多配置训练的桶式加载器：按 (n_sc, n_symb) 分组，batch 内同配置
    （CNN 特征图尺寸一致），epoch 内轮转各组。返回 collate_v2 后的 dict。
    """

    def __init__(self, samples, batch_size, shuffle=True, seed=0):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.groups = {}
        for i, s in enumerate(samples):
            # 组键必须包含决定张量尺寸的所有维度（CNN 特征图/stack 需同形状）
            key = (int(s["n_sc"]), int(s["n_symb"]), int(s["n_rx"]), int(s["n_data"]))
            self.groups.setdefault(key, []).append(i)
        self.n_groups = len(self.groups)

    def _make_batches(self):
        batches = []
        for key, idxs in self.groups.items():
            idxs = list(idxs)
            if self.shuffle:
                self.rng.shuffle(idxs)
            if self.shuffle and len(idxs) < self.batch_size:
                # 配置组样本不足时：有放回采样凑满 batch（训练稳定；验证用无放回）
                idxs = [int(i) for i in self.rng.choice(idxs, size=self.batch_size, replace=True)]
                batches.append((key, idxs))
            else:
                for i in range(0, len(idxs), self.batch_size):
                    batches.append((key, idxs[i:i + self.batch_size]))
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        for key, idxs in self._make_batches():
            yield collate_v2([self.samples[i] for i in idxs])

    def __len__(self):
        n = 0
        for idxs in self.groups.values():
            n += int(np.ceil(len(idxs) / self.batch_size))
        return n


def save_samples_pkl(path, samples):
    """样本列表 -> pickle（v2 缓存，样本形状随配置变化，不能用固定 npz）"""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples_pkl(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def build_v2_data(n_train, n_val, n_pt, seed, cache_tr, cache_va, cache_pt,
                  group_size=None):
    """
    生成或加载 v2 多配置数据缓存（train/val/pt，大规模版：每组合 group_size 样本）。
    返回 (tr_samples, va_samples, pt_samples)。
    """
    from data_gen_sionna import generate_dataset_v2
    if all(os.path.exists(p) for p in (cache_tr, cache_va, cache_pt)):
        print("[V2] 加载缓存数据")
        return (load_samples_pkl(cache_tr), load_samples_pkl(cache_va),
                load_samples_pkl(cache_pt))
    import time
    t0 = time.time()
    gs = group_size or config.V2_GROUP_SIZE
    print(f"[V2] 生成多配置数据: train={n_train}, val={n_val}, pt={n_pt} "
          f"(seed={seed}, group_size={gs}) ...")
    tr = generate_dataset_v2(n_train, seed=seed, group_size=gs)
    va = generate_dataset_v2(n_val, seed=seed + 1000, group_size=gs)
    pt = generate_dataset_v2(n_pt, seed=seed + 2000, group_size=gs)
    save_samples_pkl(cache_tr, tr)
    save_samples_pkl(cache_va, va)
    save_samples_pkl(cache_pt, pt)
    print(f"[V2] 数据就绪并缓存 ({time.time()-t0:.1f}s)")
    return tr, va, pt
