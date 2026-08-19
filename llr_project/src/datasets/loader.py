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
import hashlib
import json
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import config


def build_cfg_vec(batch):
    """配置元数据向量 (B, CFG_DIM)：**接收端可感知**的系统参数。
    不含信道模型信息（TDL/多普勒）——接收端不知道信道模型，只含：
    n_rx onehot(4) + n_sc/120 + n_symb/14 + dmrs_ap onehot(3)。"""
    n = len(batch)
    v = np.zeros((n, config.CFG_DIM), dtype=np.float32)
    ants = [1, 2, 4, 8]
    for i, s in enumerate(batch):
        v[i, 0:4] = [int(s["n_rx"]) == a for a in ants]
        v[i, 4] = int(s["n_sc"]) / 120.0
        v[i, 5] = int(s["n_symb"]) / 14.0
        v[i, 6:9] = [int(s["dmrs_ap"]) == a for a in [0, 1, 2]]
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

# 生成参数指纹：任何影响样本内容的参数变化（配置/seed/分组/子批/库版本）都会
# 改变指纹，从而让旧分片缓存自动失效重生成，避免新旧参数样本混用。
_FP_KEYS = ("GROUP_SIZE", "SUB_BATCH", "SYS_FFT", "SYS_SCS_HZ", "SYS_CP",
            "CARRIER_FREQUENCY", "RX_ANTS", "RB_RANGE", "SYMB_RANGE",
            "DMRS_APS", "TDL_MODELS", "DELAY_SPREADS", "MAX_SPEEDS",
            "MOD_ORDERS")


def _params_fingerprint(seed, group_size=None, sub_batch=None):
    vals = {k: getattr(config, k, None) for k in _FP_KEYS}
    vals.update({"seed": int(seed), "group_size": int(group_size or config.GROUP_SIZE),
                 "sub_batch": int(sub_batch or getattr(config, "SUB_BATCH", 2))})
    raw = json.dumps(vals, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _read_fingerprint(path):
    fp_path = path + ".fp"
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            return f.read().strip()
    return None


def _write_fingerprint(path, fp):
    with open(path + ".fp", "w") as f:
        f.write(fp)


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


def _cache_ready(path, fp=None):
    """缓存是否已就绪。
    - fp 给定（build_data 场景）：必须是指纹匹配的完整分片缓存；单文件旧缓存
      或无指纹分片视为未就绪（参数无法校验，保守重生成）。
    - fp 为 None（通用/兼容场景）：单文件或分片缓存任一完整即可。"""
    if fp is not None:
        if _read_fingerprint(path) != fp:
            return False
        mf = path + ".manifest"
        if not os.path.exists(mf):
            return False
        with open(mf) as f:
            n_shards = int(f.read().strip())
        return all(os.path.exists(f"{path}.{si:03d}") for si in range(n_shards))
    if os.path.exists(path):
        return True
    mf = path + ".manifest"
    if not os.path.exists(mf):
        return False
    with open(mf) as f:
        n_shards = int(f.read().strip())
    return all(os.path.exists(f"{path}.{si:03d}") for si in range(n_shards))


def _remove_stale_shards(path):
    """删除 path 的旧分片/manifest/指纹（参数变更后失效重生成用）。
    返回是否实际删除了文件。"""
    removed = False
    d = os.path.dirname(path) or "."
    base = os.path.basename(path)
    for f in os.listdir(d):
        if f.startswith(base) and f != base:
            os.remove(os.path.join(d, f))
            removed = True
    return removed


def _trim_memory():
    """归还 glibc 内存池/释放 torch CPU 缓存，降低长时间生成的内存水位"""
    import gc
    import ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_data(n_train, n_val, n_pt, seed, cache_tr, cache_va, cache_pt,
               group_size=None, shard_size=None, sub_batch=None):
    """
    生成或加载多配置数据缓存（train/val/pt，大规模版：每组合 group_size 样本）。
    ★ 分片生成：每个数据集按 SHARD_SIZE 分片，边生成边落盘（内存峰值=单片），
      断点续跑粒度到分片级；训练时 load_samples_shards 合并加载。
    ★ 内存安全：组内生成按 sub_batch 小批量执行（Sionna TDL 采样峰值与批量
      成正比，sub_batch=2 时峰值降至 ~1/4），每片后归还 glibc 内存。
    ★ 参数指纹：manifest 旁写 .fp 指纹，配置/seed/分组参数变化时旧分片自动
      失效删除并重生成，杜绝新旧参数样本混用。
    返回 (tr_samples, va_samples, pt_samples)。
    """
    from src.simulation.pusch import generate_dataset
    gs = group_size or config.GROUP_SIZE
    ss = shard_size or config.SHARD_SIZE
    sb = sub_batch if sub_batch is not None else getattr(config, "SUB_BATCH", 2)
    fp = _params_fingerprint(seed, gs, sb)
    # 参数变更/无指纹（旧版缓存或中断残留）：删除旧分片，从新参数开始生成
    for p in (cache_tr, cache_va, cache_pt):
        if _read_fingerprint(p) != fp:
            if _remove_stale_shards(p):
                print(f"[DATA] 清除旧缓存（参数/指纹不匹配）: {p}.*")
    if all(_cache_ready(p, fp) for p in (cache_tr, cache_va, cache_pt)):
        print("[DATA] 加载缓存数据")
        return (load_samples_shards(cache_tr), load_samples_shards(cache_va),
                load_samples_shards(cache_pt))
    import time
    t0 = time.time()
    print(f"[DATA] 生成多配置数据: train={n_train}, val={n_val}, pt={n_pt} "
          f"(seed={seed}, group_size={gs}, shard_size={ss}, sub_batch={sb}) ...")

    def gen_and_save(name, n, seed_, cache):
        if _cache_ready(cache, fp):
            print(f"  [{name}] 缓存已存在，跳过: {cache}")
            return
        # 先写指纹：中断后重跑可凭指纹续跑；无指纹的旧分片（参数已变）会被覆盖重生成
        _write_fingerprint(cache, fp)
        n_shards = int(np.ceil(n / ss))
        for si in range(n_shards):
            sp = f"{cache}.{si:03d}"
            if os.path.exists(sp) and _read_fingerprint(cache) == fp:
                continue                     # 断点续跑：跳过已完成分片
            m = min(ss, n - si * ss)
            part = generate_dataset(m, seed=seed_ + si, group_size=gs, sub_batch=sb)
            save_shard(part, cache, si)
            del part
            _trim_memory()
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
