# -*- coding: utf-8 -*-
"""
Sionna 数据生成器（3GPP 5G NR PUSCH 标准实现）

基于 Sionna 2.x（PyTorch 后端）的链路级仿真：
  1. 标准 PUSCH 发射机（sionna.phy.nr.PUSCHTransmitter）
  2. DMRS 配置：type1，{1+1} 双 DMRS 符号（前置符号2 + 附加符号11）
  3. 信道建模：3GPP TDL-A（sionna.phy.channel.tr38901.TDL + TimeChannel）
  4. 接收：OFDM 解调 -> LS 信道估计（先估计后插值，PUSCHLSChannelEstimator）
  5. MMSE 均衡 + max-log LLR（数据 RE）

模型输入信道估计维度：{num_rx, num_sc, num_symb} = (8, 120, 14)
"""
import numpy as np
import torch

from sionna.phy.nr import (CarrierConfig, PUSCHDMRSConfig, PUSCHConfig,
                           PUSCHTransmitter, PUSCHLSChannelEstimator)
from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import TimeChannel, time_to_ofdm_channel


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
    封装完整的 Sionna PUSCH 链路（配置固定，可批量生成样本）。
    """

    def __init__(self, num_rx_ant=8, n_size_grid=10, num_tx_ant=1,
                 channel_model="A", delay_spread=30e-9, carrier_frequency=3.5e9,
                 device="cpu"):
        self.num_rx_ant = num_rx_ant
        self.device = device

        # ---- 载波配置 ----
        self.carrier = CarrierConfig()
        self.carrier.n_size_grid = n_size_grid   # 10 RB = 120 子载波
        self.carrier.n_cell_id = 1

        # ---- DMRS 配置：type1, {1+1} 两个 DMRS 符号 ----
        dmrs = PUSCHDMRSConfig()
        dmrs.config_type = 1                    # DMRS type 1
        dmrs.length = 1                         # 前置单符号 DMRS
        dmrs.additional_position = 1            # 1 个附加 DMRS -> {1+1}
        dmrs.num_cdm_groups_without_data = 2    # 2 CDM 组（DMRS 符号全导频）

        # ---- PUSCH 配置 ----
        self.pusch = PUSCHConfig(carrier_config=self.carrier,
                                 pusch_dmrs_config=dmrs)
        self.pusch.num_antenna_ports = num_tx_ant   # UE 发射天线
        self.pusch.num_layers = 1
        self.pusch.n_size_bwp = n_size_grid
        self.pusch.modulation_order = 4             # 默认 16QAM，可逐样本改

        # ---- 发射机 ----
        self.pusch_tx = PUSCHTransmitter(self.pusch, return_bits=True,
                                         output_domain="freq", device=device)
        self.rg = self.pusch_tx.resource_grid
        self.fft_size = self.rg.fft_size
        self.cp = self.rg.cyclic_prefix_length
        self.num_symb = self.rg.num_ofdm_symbols    # 14
        self.num_sc = self.pusch.num_subcarriers    # 120

        # ---- OFDM 调制/解调 ----
        self.ofdm_mod = OFDMModulator(self.cp, device=device)
        self.bandwidth = self.fft_size * self.carrier.subcarrier_spacing * 1e3

        # ---- 信道（TDL + TimeChannel，延迟到 build 后创建） ----
        self.channel_model = channel_model
        self.delay_spread = delay_spread
        self.carrier_frequency = carrier_frequency
        self.channel = None

        # ---- LS 信道估计器（先估计后插值 -> 完整频域信道） ----
        self.estimator = PUSCHLSChannelEstimator(
            self.rg,
            dmrs_length=1,
            dmrs_additional_position=1,
            num_cdm_groups_without_data=2,
            interpolation_type="lin",           # 线性插值
            device=device,
        )

        # ---- 数据 RE 索引 ----
        dmrs_mask = self.pusch.dmrs_mask
        if torch.is_tensor(dmrs_mask):
            dmrs_mask = dmrs_mask.cpu().numpy()
        data_mask = 1 - np.asarray(dmrs_mask)   # (num_sc, num_symb)
        self.data_re_idx = np.argwhere(data_mask > 0)   # (n_data, 2) = [sc, symb]
        self.n_data = self.data_re_idx.shape[0]
        self.data_sc = self.data_re_idx[:, 0]   # sc 索引
        self.data_symb = self.data_re_idx[:, 1]  # symb 索引

    def _build_channel(self, num_time_samples, batch_size):
        if self.channel is None or self.channel.num_time_samples != num_time_samples:
            self.channel = TimeChannel(
                channel_model=TDL(model=self.channel_model,
                                  delay_spread=self.delay_spread,
                                  carrier_frequency=self.carrier_frequency,
                                  num_rx_ant=self.num_rx_ant),
                bandwidth=self.bandwidth,
                num_time_samples=num_time_samples,
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
          n_sc / n_symb / n_data : int
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
        x, b = self.pusch_tx(batch_size)      # x: (B,1,1,14,fft)

        # ---- OFDM 调制 -> 信道 -> OFDM 解调 ----
        x_t = self.ofdm_mod(x)                # (B,1,1,N_t)
        channel = self._build_channel(x_t.shape[-1], batch_size)
        no = torch.tensor(10.0 ** (-snr_db / 10.0), dtype=torch.float32)
        y_t, h_time = channel(x_t, no)        # y_t: (B,1,8,N_t+l); h_time 时域信道
        ofdm_demod = OFDMDemodulator(self.fft_size, channel.l_min, self.cp,
                                     device=self.device)
        y = ofdm_demod(y_t)                   # (B,1,8,14,fft)

        # ---- LS 信道估计（先估计后插值，完整频域信道） ----
        h_ls, err_var = self.estimator(y, no)  # (B,1,8,1,1,14,fft)
        H_est = h_ls[:, 0, :, 0, 0]            # (B,8,14,fft) [rx, symb, sc]
        H_est = H_est.permute(0, 1, 3, 2)      # -> (B,8,sc,symb) = {num_rx,num_sc,num_symb}

        # ---- 真实信道（时域 -> 频域） ----
        h_freq = time_to_ofdm_channel(h_time, self.rg, channel.l_min)  # (B,1,8,1,14,fft)? 验证
        # h_time: (B, num_rx, num_rx_ant, num_tx_ant, N_t + l_max - l_min)
        h_freq = h_freq[:, 0, :, 0, 0]         # (B,8,14,fft)
        H_true = h_freq.permute(0, 1, 3, 2)    # (B,8,sc,symb)

        # ---- 数据 RE 提取 ----
        # y 数据 RE: (B, n_data) complex（rx 维保留做多天线 LLR）
        B = batch_size
        y_re = y[:, 0, :, self.data_symb, self.data_sc]        # (B,8,n_data)
        y_re = y_re.permute(0, 2, 1)                           # (B,n_data,8)
        H_est_re = H_est[:, :, self.data_sc, self.data_symb]   # (B,8,n_data)
        H_est_re = H_est_re.permute(0, 2, 1)                   # (B,n_data,8)
        H_true_re = H_true[:, :, self.data_sc, self.data_symb]
        H_true_re = H_true_re.permute(0, 2, 1)

        # ---- MMSE 均衡（批量 8x1） ----
        z = torch.zeros(B, self.n_data, dtype=torch.complex64)
        sig2_eq = torch.zeros(B, self.n_data, dtype=torch.float32)
        h_c = H_est_re.to(torch.complex64)
        y_c = y_re.to(torch.complex64)
        no_t = no.to(torch.float32)
        eye = torch.eye(self.num_rx_ant, dtype=torch.complex64)
        for bb in range(B):
            Hm = h_c[bb].unsqueeze(-1)                        # (n_data,8,1)
            Hh = Hm @ Hm.conj().transpose(-1, -2)             # (n_data,8,8)
            w = torch.linalg.solve(Hh + no_t * eye, Hm)       # (n_data,8,1)
            z[bb] = (w.conj().transpose(-1, -2) @ y_c[bb].unsqueeze(-1))[:, 0, 0]
            sig2_eq[bb] = no_t * (w.conj() * w).sum(dim=-2)[:, 0].real

        # ---- 参考 LLR（理想信道 max-log，数据 RE） ----
        llr = torch.zeros(B, self.n_data, k, dtype=torch.float32)
        bits = torch.zeros(B, self.n_data, k, dtype=torch.int8)
        Xt = torch.tensor(X, dtype=torch.complex64)
        btab_t = torch.tensor(btab, dtype=torch.int64)
        M = len(X)
        for bb in range(B):
            hX = H_true_re[bb].unsqueeze(-1) * Xt.unsqueeze(0).unsqueeze(0)  # (n_data,8,M)
            d = (y_re[bb].unsqueeze(-1) - hX).abs() ** 2
            d = d.sum(dim=1)                                  # (n_data, M)
            for bt in range(k):
                d0 = d[:, btab_t[:, bt] == 0].min(dim=1).values
                d1 = d[:, btab_t[:, bt] == 1].min(dim=1).values
                llr[bb, :, bt] = (d0 - d1) / no_t
        llr = torch.clamp(llr, -20.0, 20.0)

        # ---- 发送比特（数据 RE 上映射的符号索引） ----
        # x 网格中数据 RE 的符号索引 -> 需要从发射机网格取。简化：随机比特不直接映射
        # 这里从 x 的网格提取数据 RE 的符号索引以获取真实比特
        # x: (B,1,1,14,fft)，数据 RE 处是调制符号；用最小距离映射回索引
        x_grid = x[:, 0, 0]                                    # (B,14,fft)
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
            "n_data": np.int32(self.n_data),
            "data_re_idx": self.data_re_idx.astype(np.int32),
        }


def generate_dataset(n_samples, num_rx_ant=8, n_size_grid=10, mod_orders=None,
                     snr_db=None, seed=0, batch_size=32):
    """
    批量生成样本（列表）。SNR/调制随机采样。
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
    sys = SionnaPUSCHSystem(num_rx_ant=8, n_size_grid=10)
    s = sys.generate_batch(2, 10.0, 16)
    print(f"batch 生成耗时: {time.time()-t0:.2f}s")
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")
    # 验证 LLR 硬判决正确率（高 SNR）
    s2 = sys.generate_batch(4, 25.0, 4)
    hard = (s2["llr_ref"] > 0).astype(int)
    acc = np.mean(hard == s2["bits_tx"])
    print("QPSK@25dB LLR 硬判决准确率:", acc)
