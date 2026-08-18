# -*- coding: utf-8 -*-
"""
Sionna 数据生成器（3GPP 5G NR PUSCH 标准实现，v2 多配置版）

基于 Sionna 2.x（PyTorch 后端）的链路级仿真，支持一个模型适配多种系统参数：
  1. 标准 PUSCH 发射机（sionna.phy.nr.PUSCHTransmitter）
  2. 信道场景多样性：TDL-A/B/C/D（sionna.phy.channel.tr38901.TDL + TimeChannel，
     原生 min_speed/max_speed Jakes 多普勒采样、delay_spread 时延扩展）
  3. 接收天线数：1/2/4/8
  4. 子载波按 RB 分配：1~10 RB（12~120 子载波）
  5. OFDM 符号数：3~14（3 符号用 mapping type B，4~14 用 type A，Sionna 原生）
  6. DMRS：type1 单符号，{1}/{1+1}/{1+2}（additional_position 0/1/2）
  7. ★ 固定系统带宽：BWP（1~10 RB）嵌入 1024-FFT 系统网格（≈10MHz 载波，
     15.36MHz 采样），信道按系统带宽建模（OFDM 循环前缀/信道抽头始终正确）
  8. 接收：OFDM 解调 -> LS 信道估计（先估计后插值，PUSCHLSChannelEstimator）
  9. MMSE 均衡 + max-log LLR（数据 RE，数据 RE 索引取自 Sionna pilot mask）

模型输入信道估计维度：{num_rx, num_sc, num_symb}（随配置变化，模型内部天线补零到 8）
"""
import numpy as np
import torch

# 数据生成统一在 CPU 上进行（Sionna 检测到 CUDA 会自动设全局 device=cuda，
# 导致其内部组件 CPU/GPU 混合报错；数据生成非瓶颈，训练时数据再转 GPU）
from sionna.phy import config as _sionna_cfg
_sionna_cfg.device = "cpu"

from sionna.phy.nr import (CarrierConfig, PUSCHDMRSConfig, PUSCHConfig,
                           PUSCHTransmitter, PUSCHLSChannelEstimator)
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator, ResourceGrid
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import TimeChannel, time_to_ofdm_channel

import config


def qam_constellation(m):
    """Gray 映射 QAM 星座（能量归一化），与 data_gen.py 一致"""
    def _ungray(n):
        x = n
        while n >> 1:
            n >>= 1
            x ^= n
        return x
    k = int(np.log2(m))
    side = int(np.sqrt(m))
    hk = k // 2
    X = np.zeros(m, dtype=np.complex128)
    bits = np.zeros((m, k), dtype=np.int8)
    for s in range(m):
        i_idx = s >> hk
        q_idx = s & (side - 1)
        i = _ungray(i_idx)
        q = _ungray(q_idx)
        X[s] = (2 * i - side + 1) + 1j * (2 * q - side + 1)
        for b in range(k):
            bits[s, k - 1 - b] = (s >> b) & 1
    X /= np.sqrt(np.mean(np.abs(X) ** 2))
    return X, bits


class SionnaPUSCHSystem:
    """
    封装完整的 Sionna PUSCH 链路（一个配置实例，可批量生成样本）。
    v2：配置可任意组合 num_rx_ant / RB / 符号数 / DMRS 模式 / 信道场景。
    """

    def __init__(self, num_rx_ant=8, n_size_grid=10, num_ofdm_symbols=14,
                 dmrs_ap=1, channel_model="A", delay_spread=30e-9,
                 max_speed=0.0, carrier_frequency=3.5e9, device="cpu"):
        self.num_rx_ant = num_rx_ant
        self.device = device

        # ---- 载波配置 ----
        self.carrier = CarrierConfig()
        self.carrier.n_size_grid = n_size_grid   # RB 数（1~10）
        self.carrier.n_cell_id = 1

        # ---- DMRS 配置：type1 单符号，{1}/{1+1}/{1+2} ----
        dmrs = PUSCHDMRSConfig()
        dmrs.config_type = 1                    # DMRS type 1
        dmrs.length = 1                         # 前置单符号 DMRS
        dmrs.additional_position = dmrs_ap      # 0={1}, 1={1+1}, 2={1+2}
        dmrs.num_cdm_groups_without_data = 2    # 2 CDM 组（DMRS 符号全导频）

        # ---- PUSCH 配置 ----
        self.pusch = PUSCHConfig(carrier_config=self.carrier,
                                 pusch_dmrs_config=dmrs)
        self.pusch.num_antenna_ports = 1        # UE 发射天线
        self.pusch.num_layers = 1
        self.pusch.n_size_bwp = n_size_grid
        self.pusch.modulation_order = 4         # 默认 16QAM，可逐样本改
        # 符号分配：3 符号用 mapping type B（l0=0），4~14 用 type A（l0=2）
        self.pusch.mapping_type = "B" if num_ofdm_symbols == 3 else "A"
        self.pusch.symbol_allocation = [0, num_ofdm_symbols]

        # ---- 发射机 ----
        self.pusch_tx = PUSCHTransmitter(self.pusch, return_bits=True,
                                         output_domain="freq", device=device)
        self.rg = self.pusch_tx.resource_grid
        self.fft_size = self.rg.fft_size          # BWP FFT 大小（= 活动子载波数）
        self.cp = self.rg.cyclic_prefix_length    # BWP CP（样本数）
        self.num_symb = self.rg.num_ofdm_symbols
        self.num_sc = self.pusch.num_subcarriers   # n_size_bwp * 12

        # ---- ★ 固定系统带宽：BWP 嵌入系统网格（1024-FFT ≈ 10MHz 载波） ----
        self.sys_fft = config.SYS_FFT
        self.sys_cp = config.SYS_CP
        self.bandwidth = self.sys_fft * config.SYS_SCS_HZ   # 15.36 MHz 采样
        self.k0 = (self.sys_fft - self.fft_size) // 2       # BWP 起始子载波（居中）
        self.ofdm_mod_sys = OFDMModulator(self.sys_cp, device=device)
        self.rg_sys = ResourceGrid(
            num_ofdm_symbols=self.num_symb,
            fft_size=self.sys_fft,
            subcarrier_spacing=config.SYS_SCS_HZ,
            cyclic_prefix_length=self.sys_cp,
            device=device,
        )

        # ---- 信道（TDL + TimeChannel，延迟到 build 后创建） ----
        self.channel_model = channel_model
        self.delay_spread = delay_spread
        self.max_speed = max_speed
        self.carrier_frequency = carrier_frequency
        self.channel = None

        # ---- LS 信道估计器（先估计后插值 -> 完整频域信道） ----
        self.estimator = PUSCHLSChannelEstimator(
            self.rg,
            dmrs_length=1,
            dmrs_additional_position=dmrs_ap,
            num_cdm_groups_without_data=2,
            interpolation_type="lin",           # 线性插值
            device=device,
        )

        # ---- 数据 RE 索引（resource grid 导频掩码 -> 1-mask，[sc, symb]） ----
        # 注意：pusch.dmrs_mask 是 14 符号全帧掩码（短 PUSCH 时不匹配），
        # 必须用 pilot_pattern.mask（按实际分配符号数 num_ofdm_symbols）。
        pm = self.rg.pilot_pattern.mask
        if torch.is_tensor(pm):
            pm = pm.cpu().numpy()
        pm = np.asarray(pm)[0, 0]                       # (num_symb, num_sc)
        data_mask = 1 - pm                              # 数据 RE = 非导频
        self.data_re_idx = np.argwhere(data_mask.T > 0)  # (n_data, 2) = [sc, symb]
        self.n_data = self.data_re_idx.shape[0]
        self.data_sc = self.data_re_idx[:, 0]           # sc 索引
        self.data_symb = self.data_re_idx[:, 1]         # symb 索引
        self.dmrs_symbs = sorted(set(np.argwhere(pm > 0)[:, 0]))

    def _channel_lags(self):
        """
        计算适配 OFDM 的信道时延参数 (tdl, l_min, l_max)。
        Sionna 默认 l_min=-6 / l_max=ceil(3µs·W)+6 的裕量在窄带（小 RB，CP 短）
        下会超过循环前缀长度，破坏 OFDM 循环卷积（LLR 标签系统性错误）。
        这里按 TDL 真实最大时延计算 l_max，并限制在 CP 内：l_min=0，l_max<=cp。
        """
        tdl = TDL(model=self.channel_model,
                  delay_spread=self.delay_spread,
                  carrier_frequency=self.carrier_frequency,
                  num_rx_ant=self.num_rx_ant,
                  min_speed=0.0,
                  max_speed=self.max_speed)
        _, tau = tdl(batch_size=1, num_time_steps=1, sampling_frequency=self.bandwidth)
        if torch.is_tensor(tau):
            tau = tau.cpu().numpy()
        l_max = int(np.ceil(float(np.asarray(tau).max()) * self.bandwidth))
        l_min = 0
        # OFDM 循环前缀约束：信道时延扩展必须落在 CP 内（l_max <= cp）
        l_max = min(l_max, self.cp)
        return tdl, l_min, l_max

    def _build_channel(self, num_time_samples, batch_size):
        if self.channel is None or self.channel.num_time_samples != num_time_samples:
            tdl, l_min, l_max = self._channel_lags()
            self.channel = TimeChannel(
                channel_model=tdl,
                bandwidth=self.bandwidth,
                num_time_samples=num_time_samples,
                l_min=l_min,
                l_max=l_max,
                return_channel=True,
                device=self.device,
            )
        return self.channel

    def generate_batch(self, batch_size, snr_db, mod_order, seed=None):
        """
        生成一批 PUSCH 样本。

        返回 dict（均为 numpy）：
          H_est   : (B, num_rx, num_sc, num_symb) complex  信道估计（模型输入）
          H_true  : (B, num_rx, num_sc, num_symb) complex  真实信道（评估）
          z       : (B, n_data) complex      数据 RE 均衡软符号
          sigma2  : (B,) float
          sigma2_eq: (B, n_data) float       均衡后等效噪声方差
          llr_ref : (B, n_data, log2M) float 参考 LLR（理想信道 max-log）
          bits_tx : (B, n_data, log2M) int8  发送比特
          mod_order: int
          n_sc / n_symb / n_rx / n_data / dmrs_ap / tdl / delay_spread / max_speed : int/float
          data_re_idx: (n_data, 2) int32 [sc, symb]
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.pusch.modulation_order = mod_order
        k = int(np.log2(mod_order))
        X, btab = qam_constellation(mod_order)

        # ---- 发射 ----
        self.pusch_tx = PUSCHTransmitter(self.pusch, return_bits=True,
                                         output_domain="freq", device=self.device)
        x, b = self.pusch_tx(batch_size)      # x: (B,1,1,S,fft_bwp)

        # ---- 嵌入系统网格（固定带宽）-> OFDM 调制 -> 信道 -> 解调 ----
        B = batch_size
        S = self.num_symb
        x_sys = torch.zeros(B, 1, 1, S, self.sys_fft, dtype=x.dtype)
        x_sys[..., self.k0:self.k0 + self.fft_size] = x      # BWP 居中嵌入
        x_t = self.ofdm_mod_sys(x_sys)        # (B,1,1,N_t)
        channel = self._build_channel(x_t.shape[-1], batch_size)
        no = torch.tensor(10.0 ** (-snr_db / 10.0), dtype=torch.float32)
        y_t, h_time = channel(x_t, no)        # y_t: (B,1,n_rx,N_t+l); h_time 时域信道
        ofdm_demod = OFDMDemodulator(self.sys_fft, channel.l_min, self.sys_cp,
                                     device=self.device)
        y = ofdm_demod(y_t)                   # (B,1,n_rx,S',sys_fft)
        # 短 PUSCH 时 l_min<0 会使解调输出 S' > 实际符号数（帧尾残留），裁剪对齐
        if y.shape[-2] > self.num_symb:
            y = y[..., :self.num_symb, :]
        y = y[..., self.k0:self.k0 + self.fft_size]   # 提取 BWP 子载波

        # ---- LS 信道估计（先估计后插值，完整频域信道，BWP 网格） ----
        h_ls, err_var = self.estimator(y, no)  # (B,1,n_rx,1,1,S,fft_bwp)
        if h_ls.shape[-2] > self.num_symb:
            h_ls = h_ls[..., :self.num_symb, :]
        H_est = h_ls[:, 0, :, 0, 0]            # (B,n_rx,S,fft_bwp) [rx, symb, sc]
        H_est = H_est.permute(0, 1, 3, 2)      # -> (B,n_rx,sc,symb) = {num_rx,num_sc,num_symb}

        # ---- 真实信道（时域 -> 频域，系统网格 -> 提取 BWP） ----
        h_freq = time_to_ofdm_channel(h_time, self.rg_sys, channel.l_min)  # (B,1,n_rx,1,S,sys_fft)
        h_freq = h_freq[:, 0, :, 0, 0, :, self.k0:self.k0 + self.fft_size]  # (B,n_rx,S,fft_bwp)
        H_true = h_freq.permute(0, 1, 3, 2)    # (B,n_rx,sc,symb)

        # ---- 数据 RE 提取 ----
        B = batch_size
        y_re = y[:, 0, :, self.data_symb, self.data_sc]        # (B,n_rx,n_data)
        y_re = y_re.permute(0, 2, 1)                           # (B,n_data,n_rx)
        H_est_re = H_est[:, :, self.data_sc, self.data_symb]   # (B,n_rx,n_data)
        H_est_re = H_est_re.permute(0, 2, 1)                   # (B,n_data,n_rx)
        H_true_re = H_true[:, :, self.data_sc, self.data_symb]
        H_true_re = H_true_re.permute(0, 2, 1)

        # ---- MMSE 均衡（批量 n_rx x 1） ----
        z = torch.zeros(B, self.n_data, dtype=torch.complex64)
        sig2_eq = torch.zeros(B, self.n_data, dtype=torch.float32)
        h_c = H_est_re.to(torch.complex64)
        y_c = y_re.to(torch.complex64)
        no_t = no.to(torch.float32)
        eye = torch.eye(self.num_rx_ant, dtype=torch.complex64)
        for bb in range(B):
            Hm = h_c[bb].unsqueeze(-1)                        # (n_data,n_rx,1)
            Hh = Hm @ Hm.conj().transpose(-1, -2)             # (n_data,n_rx,n_rx)
            w = torch.linalg.solve(Hh + no_t * eye, Hm)       # (n_data,n_rx,1)
            z[bb] = (w.conj().transpose(-1, -2) @ y_c[bb].unsqueeze(-1))[:, 0, 0]
            sig2_eq[bb] = no_t * (w.conj() * w).sum(dim=-2)[:, 0].real

        # ---- 参考 LLR（理想信道 max-log，数据 RE） ----
        llr = torch.zeros(B, self.n_data, k, dtype=torch.float32)
        bits = torch.zeros(B, self.n_data, k, dtype=torch.int8)
        Xt = torch.tensor(X, dtype=torch.complex64)
        btab_t = torch.tensor(btab, dtype=torch.int64)
        M = len(X)
        for bb in range(B):
            hX = H_true_re[bb].unsqueeze(-1) * Xt.unsqueeze(0).unsqueeze(0)  # (n_data,n_rx,M)
            d = (y_re[bb].unsqueeze(-1) - hX).abs() ** 2
            d = d.sum(dim=1)                                  # (n_data, M)
            for bt in range(k):
                d0 = d[:, btab_t[:, bt] == 0].min(dim=1).values
                d1 = d[:, btab_t[:, bt] == 1].min(dim=1).values
                llr[bb, :, bt] = (d0 - d1) / no_t
        llr = torch.clamp(llr, -20.0, 20.0)

        # ---- 发送比特（数据 RE 上映射的符号索引） ----
        x_grid = x[:, 0, 0]                                    # (B,S,fft)
        x_re = x_grid[:, self.data_symb, self.data_sc]         # (B,n_data)
        for bb in range(B):
            dist = (x_re[bb].unsqueeze(-1) - Xt.unsqueeze(0)).abs() ** 2  # (n_data,M)
            idx = dist.argmin(dim=1)                           # (n_data,)
            bits[bb] = btab_t[idx]

        return {
            "H_est": H_est.cpu().numpy().astype(np.complex64),
            "H_true": H_true.cpu().numpy().astype(np.complex64),
            "z": z.cpu().numpy().astype(np.complex64),
            "sigma2": float(no_t.item()),
            "sigma2_eq": sig2_eq.cpu().numpy().astype(np.float32),
            "llr_ref": llr.cpu().numpy().astype(np.float32),
            "bits_tx": bits.cpu().numpy().astype(np.int8),
            "mod_order": np.int32(mod_order),
            "n_sc": np.int32(self.num_sc),
            "n_symb": np.int32(self.num_symb),
            "n_rx": np.int32(self.num_rx_ant),
            "dmrs_ap": np.int32(0 if self.pusch.dmrs.additional_position is None
                                else self.pusch.dmrs.additional_position),
            "tdl": self.channel_model,
            "delay_spread": np.float32(self.delay_spread),
            "max_speed": np.float32(self.max_speed),
            "n_data": np.int32(self.n_data),
            "data_re_idx": self.data_re_idx.astype(np.int32),
        }


def sample_config(rng, rx_ants=None, rb_range=None, symb_range=None,
                  dmrs_aps=None, tdl_models=None, delay_spreads=None,
                  max_speeds=None):
    """随机采样一个系统配置 dict（v2 多配置空间）"""
    import config
    return {
        "num_rx_ant": int(rng.choice(rx_ants or config.V2_RX_ANTS)),
        "n_size_grid": int(rng.integers(*(rb_range or config.V2_RB_RANGE))),
        "num_ofdm_symbols": int(rng.integers(*(symb_range or config.V2_SYMB_RANGE))),
        "dmrs_ap": int(rng.choice(dmrs_aps or config.V2_DMRS_APS)),
        "channel_model": str(rng.choice(tdl_models or config.V2_TDL_MODELS)),
        "delay_spread": float(rng.choice(delay_spreads or config.V2_DELAY_SPREADS)),
        "max_speed": float(rng.choice(max_speeds or config.V2_MAX_SPEEDS)),
        "carrier_frequency": config.V2_CARRIER_FREQUENCY,
    }


def generate_dataset_v2(n_samples, seed=0, snr_db=None, batch_size=32,
                        cfg_sampler=None, group_size=1):
    """
    v2：多配置混合样本生成（大规模版）。
    先采样 n_combos = ceil(n_samples/group_size) 个系统配置，每个配置组合生成
    group_size 个**不同**样本（不同信道实现/噪声/比特），同配置样本一组生成
    （组件复用）。SNR/调制逐组随机（snr_db 给定时固定 SNR）。
    """
    rng = np.random.default_rng(seed)
    import config
    mod_orders = config.MOD_ORDERS
    if cfg_sampler is None:
        cfg_sampler = sample_config
    n_combos = int(np.ceil(n_samples / group_size))
    cfgs = [cfg_sampler(rng) for _ in range(n_combos)]
    groups = {}
    for ci, c in enumerate(cfgs):
        key = tuple(sorted(c.items()))
        groups.setdefault(key, []).extend([ci] * group_size)
    samples = [None] * n_samples
    idx = 0
    for gi, (key, cis) in enumerate(groups.items()):
        cfg = dict(key)
        sys = SionnaPUSCHSystem(**cfg)
        n_here = min(len(cis), n_samples - idx)
        mod = int(mod_orders[rng.integers(0, len(mod_orders))])
        s = float(rng.uniform(-5, 25)) if snr_db is None else float(snr_db)
        batch = sys.generate_batch(n_here, s, mod, seed=seed + gi)
        for j in range(n_here):
            # 注意：data_re_idx 是配置级共享（batch 内相同），不能按 batch 维切片
            d = {k: (v[j] if isinstance(v, np.ndarray) and v.ndim > 0 and k != "data_re_idx"
                     else v) for k, v in batch.items()}
            d["snr_db"] = s
            d["cfg"] = dict(cfg)
            samples[idx] = d
            idx += 1
    return samples


def generate_dataset(n_samples, num_rx_ant=8, n_size_grid=10, mod_orders=None,
                     snr_db=None, seed=0, batch_size=32):
    """
    固定配置批量生成样本（列表，v1 兼容）。SNR/调制随机采样。
    """
    rng = np.random.default_rng(seed)
    if mod_orders is None:
        mod_orders = [4, 16, 64, 256]
    sys = SionnaPUSCHSystem(num_rx_ant=num_rx_ant, n_size_grid=n_size_grid)
    samples = []
    n_batches = int(np.ceil(n_samples / batch_size))
    for bi in range(n_batches):
        bs = min(batch_size, n_samples - bi * batch_size)
        if bs <= 0:
            break
        mod = int(mod_orders[rng.integers(0, len(mod_orders))])
        if snr_db is None:
            s = float(rng.uniform(-5, 25))
        else:
            s = float(snr_db)
        batch = sys.generate_batch(bs, s, mod, seed=seed + bi)
        for i in range(bs):
            samples.append({k: v[i] if isinstance(v, np.ndarray) and v.ndim > 0
                            else v for k, v in batch.items()})
    return samples


if __name__ == "__main__":
    import time
    t0 = time.time()
    # 极端小配置：1 天线 / 1 RB / 3 符号（mapping B）/ DMRS {1} / TDL-D 300ns / 30m/s
    sys = SionnaPUSCHSystem(num_rx_ant=1, n_size_grid=1, num_ofdm_symbols=3, dmrs_ap=0,
                            channel_model="D", delay_spread=300e-9, max_speed=30.0)
    s = sys.generate_batch(2, 10.0, 16)
    print(f"batch 生成耗时: {time.time()-t0:.2f}s")
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")
    # 验证 LLR 硬判决正确率（高 SNR，全配置应 ≈1.0）
    for cfg, mod in [(dict(num_rx_ant=1, n_size_grid=1, num_ofdm_symbols=3, dmrs_ap=0,
                           channel_model="D", delay_spread=300e-9, max_speed=30.0), 16),
                     (dict(num_rx_ant=8, n_size_grid=10, num_ofdm_symbols=14, dmrs_ap=2,
                           channel_model="A", delay_spread=30e-9, max_speed=0.0), 64)]:
        s2 = SionnaPUSCHSystem(**cfg).generate_batch(4, 25.0, mod)
        hard = (s2["llr_ref"] > 0).astype(int)
        acc = np.mean(hard == s2["bits_tx"])
        print(f"{cfg['num_rx_ant']}rx/{cfg['n_size_grid']}RB/{cfg['num_ofdm_symbols']}symb "
              f"QAM{mod}@25dB LLR 硬判决准确率: {acc:.4f}")
