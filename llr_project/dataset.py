# -*- coding: utf-8 -*-
"""
PyTorch Dataset 工具：多配置样本 -> 桶式训练 batch

样本格式（data_gen_sionna 输出，多配置自适应）：
  H_est(n_rx,n_sc,n_symb)complex, H_true, z(n_data,)complex,
  sigma2, sigma2_eq(n_data,), llr_ref(n_data,log2M), bits_tx(n_data,log2M),
  mod_order, n_rx/n_sc/n_symb/n_data, dmrs_ap/tdl/delay_spread/max_speed,
  data_re_idx(n_data,2), snr_db, cfg

n_data 随配置变化（如 14 符号 {1+1} × 120 子载波 = 1440；3 符号 {1} × 12 = 24）。
"""
import os
import pickle

import numpy as np

import config


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


def collate_batch(batch):
    """
    合并一批**同 (n_sc, n_symb, n_rx, n_data) 配置**的样本 dict -> 训练 batch dict（numpy）。
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
    多配置训练的桶式加载器：按 (n_sc, n_symb, n_rx, n_data) 分组，batch 内同配置
    （CNN 特征图尺寸一致），epoch 内轮转各组。返回 collate_batch 后的 dict。
    组内样本不足 batch_size 时有放回采样凑满（训练稳定；验证用无放回）。
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
            yield collate_batch([self.samples[i] for i in idxs])

    def __len__(self):
        n = 0
        for idxs in self.groups.values():
            n += int(np.ceil(len(idxs) / self.batch_size))
        return n


# ================= 数据缓存（Sionna 生成一次，训练复用） =================

def save_samples_pkl(path, samples):
    """样本列表 -> pickle（形状随配置变化，不能用固定 npz）"""
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_data(n_train, n_val, n_pt, seed, cache_tr, cache_va, cache_pt,
               group_size=None):
    """
    生成或加载多配置数据缓存（train/val/pt，大规模版：每组合 group_size 样本）。
    返回 (tr_samples, va_samples, pt_samples)。
    """
    from data_gen_sionna import generate_dataset
    if all(os.path.exists(p) for p in (cache_tr, cache_va, cache_pt)):
        print("[DATA] 加载缓存数据")
        return (load_samples_pkl(cache_tr), load_samples_pkl(cache_va),
                load_samples_pkl(cache_pt))
    import time
    t0 = time.time()
    gs = group_size or config.GROUP_SIZE
    print(f"[DATA] 生成多配置数据: train={n_train}, val={n_val}, pt={n_pt} "
          f"(seed={seed}, group_size={gs}) ...")
    tr = generate_dataset(n_train, seed=seed, group_size=gs)
    va = generate_dataset(n_val, seed=seed + 1000, group_size=gs)
    pt = generate_dataset(n_pt, seed=seed + 2000, group_size=gs)
    save_samples_pkl(cache_tr, tr)
    save_samples_pkl(cache_va, va)
    save_samples_pkl(cache_pt, pt)
    print(f"[DATA] 数据就绪并缓存 ({time.time()-t0:.1f}s)")
    return tr, va, pt
