# -*- coding: utf-8 -*-
"""
组合模型 LWMLLR：LWM 骨干 + CNN LLR Decoder（多配置自适应版）。
输入 (H, z, sigma2, mod_onehot, data_re_idx, cfg) -> (B, T, 8) LLR。
适配：n_rx 1/2/4/8（补零到 8）、n_sc 1~10 RB（序列长度自适应）、
符号 3~14、DMRS {1}/{1+1}/{1+2}（数据 RE 索引随样本传入）。
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import config
from src.datasets.demap import qam_constellation
from src.models.lwm import lwm
from src.models.llr_decoder import CNNLLRDecoder, FEAT_CH


# ================= 组合模型 =================


class LWMLLR(nn.Module):
    """LWM 骨干 + CNN LLR decoder（多配置自适应版）。
    输入 3D 信道 H (B, n_rx, n_sc, n_symb) -> 数据 RE 的逐比特 LLR。
    适配：n_rx∈{1,2,4,8}（补零到 8）、n_sc∈{12,...,120}（1~10 RB，序列长度自适应）、
    n_symb∈{3,...,14}、DMRS {1}/{1+1}/{1+2}（数据 RE 索引随样本传入）。
    输入不含 llr_base（无需传统软解调），decoder 为 CNN 残差网络。
    """

    MAX_RX_ANT = 8   # patch 维度 16 与官方 LWM embedding 对齐，天线不足补零

    def __init__(self, backbone=None, freeze_backbone=False, device="cpu"):
        super().__init__()
        self.backbone = backbone if backbone is not None else lwm()
        self.decoder = CNNLLRDecoder()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _tokenize_3d(H):
        """
        H: (B, n_rx, n_sc, n_symb) complex（n_rx<=8）-> input_ids (B*n_symb, n_sc+1, 16)
        逐 OFDM 符号 tokenize：天线补零到 8 -> 每符号 n_sc patches + CLS（长度自适应，
        n_sc+1 <= 121 <= LWM MAX_LEN=129）。
        """
        B, n_rx, n_sc, S = H.shape
        if n_rx < LWMLLR.MAX_RX_ANT:
            pad = torch.zeros(B, LWMLLR.MAX_RX_ANT - n_rx, n_sc, S,
                              dtype=H.dtype, device=H.device)
            H = torch.cat([H, pad], dim=1)
        real = H.real.permute(0, 3, 2, 1)   # (B, S, n_sc, 8)
        imag = H.imag.permute(0, 3, 2, 1)
        patches = torch.cat([real, imag], dim=-1)          # (B, S, n_sc, 16)
        cls = 0.2 * torch.ones(B, S, 1, 16, dtype=patches.dtype, device=H.device)
        seq = torch.cat([cls, patches], dim=2)             # (B, S, n_sc+1, 16)
        return seq.reshape(B * S, n_sc + 1, 16)

    @staticmethod
    def _build_feat(h_emb, patch, shallow_feat, z, sigma2, mod_oh, data_re_idx, cfg):
        """
        构建 CNN 输入特征图 (B, FEAT_CH, n_symb, n_sc)：
          通道 = channel_emb(64) + H_patch(16) + Re(z) + Im(z) + σ² + mod_oh(4)
               + LWM 浅层特征 SHALLOW_LAYERS×64 + 配置元数据 cfg(CFG_DIM)，
               补零到偶数满足 GroupNorm(2)。
          z 仅在数据 RE 位置有值（data_re_idx 为 (n_data,2) [sc,symb]），其余为 0。
          cfg: (B, CFG_DIM) 接收机已知的系统参数（天线/RB/符号/DMRS/TDL/速度）。
        """
        B = h_emb.shape[0]
        S, SC = h_emb.shape[1], h_emb.shape[2]
        dev = h_emb.device
        feat = torch.cat([h_emb, patch, shallow_feat], dim=-1)  # (B, S, SC, 80+shallow)
        z_re = torch.zeros(B, S, SC, device=dev, dtype=torch.float32)
        z_im = torch.zeros_like(z_re)
        n_data = len(data_re_idx)
        b_idx = torch.arange(B, device=dev)[:, None].expand(B, n_data).reshape(-1)
        s_idx = torch.as_tensor(data_re_idx[:, 1], device=dev)[None, :].expand(B, n_data).reshape(-1)
        c_idx = torch.as_tensor(data_re_idx[:, 0], device=dev)[None, :].expand(B, n_data).reshape(-1)
        z_re.index_put_((b_idx, s_idx, c_idx), z.real.reshape(-1))
        z_im.index_put_((b_idx, s_idx, c_idx), z.imag.reshape(-1))
        feat = torch.cat([feat, z_re[..., None], z_im[..., None]], dim=-1)  # (B, S, SC, 82+shallow)
        s2 = sigma2.reshape(B, 1, 1, 1).expand(B, S, SC, 1)
        mo = mod_oh.reshape(B, 1, 1, -1).expand(B, S, SC, -1)
        cfg_v = cfg.reshape(B, 1, 1, -1).expand(B, S, SC, -1)
        feat = torch.cat([feat, s2, mo, cfg_v], dim=-1)     # (B, S, SC, 87+CFG_DIM+shallow)
        n_pad = FEAT_CH - feat.shape[-1]
        if n_pad > 0:
            feat = F.pad(feat, (0, n_pad))                  # 补零到偶数
        # 显式 contiguous：torch 2.13 的 CPU GroupNorm backward 对非连续输入段错误
        return feat.permute(0, 3, 1, 2).contiguous()        # (B, FEAT_CH, S, SC)

    def forward(self, H, z, sigma2, mod_oh, data_re_idx, cfg):
        """
        H: (B, n_rx, n_sc, n_symb) complex（Sionna PUSCH 3D 信道，含 DMRS 符号）
        z: (B, n_data) complex（数据 RE 均衡符号）
        sigma2: (B,)
        mod_oh: (B, 4)
        data_re_idx: (n_data, 2) int [sc, symb]（本样本配置的数据 RE 索引）
        cfg: (B, CFG_DIM) 系统配置元数据
        -> llr (B, n_data, 8)（LLR logits，正=bit1，裁剪到 ±MAX_LLR）
        """
        B, _, n_sc, n_symb = H.shape
        input_ids = self._tokenize_3d(H)               # (B*n_symb, n_sc+1, 16)
        output, shallow = self.backbone.encode_with_shallow(
            input_ids, shallow_layers=config.SHALLOW_LAYERS)
        h_emb = output[:, 1:1 + n_sc, :].reshape(B, n_symb, n_sc, -1)   # (B, S, SC, 64)
        patch = input_ids[:, 1:1 + n_sc, :].reshape(B, n_symb, n_sc, -1)  # (B, S, SC, 16)
        # 浅层特征拼接：每层 (B*S, n_sc+1, 64) -> (B, S, SC, 64) -> 沿通道拼接
        shallow_feat = torch.cat([
            s[:, 1:1 + n_sc, :].reshape(B, n_symb, n_sc, -1) for s in shallow
        ], dim=-1)                                      # (B, S, SC, len*64)
        feat = self._build_feat(h_emb, patch, shallow_feat, z, sigma2, mod_oh,
                                data_re_idx, cfg)       # (B,FEAT_CH,S,SC)
        logits = self.decoder(feat)                    # (B, 8, S, SC)
        logits = logits.permute(0, 2, 3, 1)            # (B, S, SC, 8)
        dr = np.asarray(data_re_idx)
        llr = logits[:, dr[:, 1], dr[:, 0], :]         # (B, n_data, 8)
        return torch.clamp(llr, -config.MAX_LLR, config.MAX_LLR)

    def infer_llr(self, H, z, sigma2, mod_order, data_re_idx, cfg):
        """
        推理入口（多配置自适应）。
        H: (n_rx, n_sc, n_symb) complex
        z: (n_data,) complex（数据 RE 均衡符号）
        sigma2: float
        mod_order: int
        data_re_idx: (n_data, 2) int [sc, symb]
        cfg: (CFG_DIM,) 系统配置元数据
        -> llr (n_data, log2M) float32
        """
        self.eval()
        dev = next(self.parameters()).device
        H = np.asarray(H)
        assert H.ndim == 3 and H.shape[0] <= 8, H.shape
        mod_oh = np.zeros((1, config.MOD_ONHOT_DIM), dtype=np.float32)
        mod_oh[0, config.MOD_ORDERS.index(mod_order)] = 1.0
        X, btab = qam_constellation(mod_order)
        with torch.no_grad():
            H_t = torch.tensor(H[None], dtype=torch.complex64, device=dev)
            z_t = torch.tensor(z[None], dtype=torch.complex64, device=dev)
            s2_t = torch.tensor([sigma2], dtype=torch.float32, device=dev)
            mo_t = torch.tensor(mod_oh, dtype=torch.float32, device=dev)
            cfg_t = torch.tensor(np.asarray(cfg, dtype=np.float32)[None], device=dev)
            llr = self(H_t, z_t, s2_t, mo_t, data_re_idx, cfg_t)[0].cpu().numpy()
        return llr[:, :btab.shape[1]].astype(np.float32)


def load_official_backbone(device="cpu"):
    """加载官方 LWM 权重"""
    ckpt = config.LWM_OFFICIAL_CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"官方权重不存在: {ckpt}。请先克隆 LWM 仓库（含 model_weights.pth）")
    return lwm.from_pretrained(ckpt, device=device)


if __name__ == "__main__":
    # 冒烟测试：官方权重加载 + 多配置 3D 前向（无 llr_base，自适应维度）
    bb = load_official_backbone()
    model = LWMLLR(bb).eval()
    from src.datasets.tokenizer import data_re_index
    cfg = torch.zeros(2, config.CFG_DIM)
    # 配置1: 4 天线, 4 RB, 7 符号, DMRS {1}（符号2）
    dri = data_re_index(n_sc=48, n_symb=7, dmrs_symbs=(2,))
    H = torch.randn(2, 4, 48, 7, dtype=torch.complex64)
    z = torch.randn(2, len(dri), dtype=torch.complex64)
    s2 = torch.tensor([0.1, 0.2])
    mo = torch.zeros(2, 4); mo[:, 1] = 1.0
    out = model(H, z, s2, mo, dri, cfg)
    print("config1 (4rx,48sc,7symb):", tuple(out.shape))
    # 配置2: 8 天线, 10 RB, 14 符号, DMRS {1+2}（符号 2/7/11）
    dri2 = data_re_index(n_sc=120, n_symb=14, dmrs_symbs=(2, 7, 11))
    H2 = torch.randn(2, 8, 120, 14, dtype=torch.complex64)
    z2 = torch.randn(2, len(dri2), dtype=torch.complex64)
    out2 = model(H2, z2, s2, mo, dri2, cfg)
    print("config2 (8rx,120sc,14symb):", tuple(out2.shape))
    # 配置3: 1 天线, 1 RB, 3 符号, DMRS {1}（符号0, mapping B）
    dri3 = data_re_index(n_sc=12, n_symb=3, dmrs_symbs=(0,))
    H3 = torch.randn(2, 1, 12, 3, dtype=torch.complex64)
    z3 = torch.randn(2, len(dri3), dtype=torch.complex64)
    out3 = model(H3, z3, s2, mo, dri3, cfg)
    print("config3 (1rx,12sc,3symb):", tuple(out3.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print("total params:", n_params)
    print("model OK")
