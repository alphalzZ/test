# -*- coding: utf-8 -*-
"""
PyTorch Dataset 工具：多配置样本 -> 桶式训练 batch

样本格式（src/simulation/pusch.py 的 SionnaPUSCHSystem 输出，多配置自适应）：
  H_est(n_rx,n_sc,n_symb)complex, H_true, z(n_data,)complex,
  sigma2, sigma2_eq(n_data,), llr_ref(n_data,log2M), bits_tx(n_data,log2M),
  mod_order, n_rx/n_sc/n_symb/n_data, dmrs_ap/tdl/delay_spread/max_speed,
  data_re_idx(n_data,2), snr_db, cfg

n_data 随配置变化（如 14 符号 {1+1} × 120 子载波 = 1440；3 符号 {1} × 12 = 24）。
"""
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import config


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
    """样本列表 -> 单文件 pickle（小规模缓存用）"""
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_shard(samples, path, shard_index):
    """增量写一个分片（大规模生成时边生成边落盘，控制内存峰值）"""
    with open(f"{path}.{shard_index:03d}", "wb") as f:
        pickle.dump(samples, f)


def save_samples_shards(samples, path, shard_size=None):
    """整列表分片保存（path.000/001/... + path.manifest 记录片数）"""
    ss = shard_size or config.SHARD_SIZE
    n_shards = int(np.ceil(len(samples) / ss))
    for si in range(n_shards):
        save_shard(samples[si * ss:(si + 1) * ss], path, si)
    with open(path + ".manifest", "w") as f:
        f.write(str(n_shards))


def load_samples_shards(path):
    """加载分片缓存（无 manifest 时回退单文件 pkl，兼容旧缓存）"""
    if not os.path.exists(path + ".manifest"):
        return load_samples_pkl(path)
    with open(path + ".manifest") as f:
        n_shards = int(f.read().strip())
    samples = []
    for si in range(n_shards):
        with open(f"{path}.{si:03d}", "rb") as f:
            samples.extend(pickle.load(f))
    return samples


def _cache_ready(path):
    """缓存是否已就绪（单文件；或分片缓存：manifest 存在且全部分片文件完整）"""
    if os.path.exists(path):
        return True
    mf = path + ".manifest"
    if not os.path.exists(mf):
        return False
    with open(mf) as f:
        n_shards = int(f.read().strip())
    return all(os.path.exists(f"{path}.{si:03d}") for si in range(n_shards))


def build_data(n_train, n_val, n_pt, seed, cache_tr, cache_va, cache_pt,
               group_size=None, shard_size=None):
    """
    生成或加载多配置数据缓存（train/val/pt，大规模版：每组合 group_size 样本）。
    ★ 分片生成：每个数据集按 SHARD_SIZE 分片，边生成边落盘（内存峰值=单片），
      断点续跑粒度到分片级；训练时 load_samples_shards 合并加载。
    返回 (tr_samples, va_samples, pt_samples)。
    """
    from src.simulation.pusch import generate_dataset
    if all(_cache_ready(p) for p in (cache_tr, cache_va, cache_pt)):
        print("[DATA] 加载缓存数据")
        return (load_samples_shards(cache_tr), load_samples_shards(cache_va),
                load_samples_shards(cache_pt))
    import gc
    import time
    t0 = time.time()
    gs = group_size or config.GROUP_SIZE
    ss = shard_size or config.SHARD_SIZE
    print(f"[DATA] 生成多配置数据: train={n_train}, val={n_val}, pt={n_pt} "
          f"(seed={seed}, group_size={gs}, shard_size={ss}) ...")

    def gen_and_save(name, n, seed_, cache):
        if _cache_ready(cache):
            print(f"  [{name}] 缓存已存在，跳过: {cache}")
            return
        n_shards = int(np.ceil(n / ss))
        for si in range(n_shards):
            sp = f"{cache}.{si:03d}"
            if os.path.exists(sp):
                continue                     # 断点续跑：跳过已完成分片
            m = min(ss, n - si * ss)
            part = generate_dataset(m, seed=seed_ + si, group_size=gs)
            save_shard(part, cache, si)
            del part
            gc.collect()
            print(f"  [{name}] 分片 {si + 1}/{n_shards} ({m} 样本) 完成，"
                  f"累计 {time.time() - t0:.0f}s")
        with open(cache + ".manifest", "w") as f:
            f.write(str(n_shards))
        print(f"  [{name}] 全部 {n} 样本就绪 -> {cache}.*")

    gen_and_save("train", n_train, seed, cache_tr)
    gen_and_save("val", n_val, seed + 1000, cache_va)
    gen_and_save("pt", n_pt, seed + 2000, cache_pt)
    print(f"[DATA] 数据就绪 ({time.time() - t0:.1f}s)")
    return (load_samples_shards(cache_tr), load_samples_shards(cache_va),
            load_samples_shards(cache_pt))
