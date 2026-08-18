# -*- coding: utf-8 -*-
"""
模型定义：
  1. LWM 骨干：与官方 lwm_model.py 结构完全一致（保证可加载官方权重），
     新增 encode() 返回逐 patch 隐状态。
  2. CNNLLRDecoder：CNN 残差网络（参考 NNreceiver 架构）。
     输入 = 全网格特征图 [channel_emb(64) + H_patch(16) + Re(z) + Im(z)
            + σ² + mod_onehot(4) + 配置元数据 cfg]（不含 llr_base，免传统软解调）
     输出 = 逐数据 RE 的逐比特 LLR logits（正=bit1）。
  3. LWMLLR：组合模型，输入 (H, z, σ², mod_onehot, data_re_idx, cfg) -> (B, T, 8) LLR。
     多配置自适应：接收天线 1/2/4/8（补零到 8）、子载波 1~10 RB（序列长度自适应）、
     符号数 3~14、DMRS {1}/{1+1}/{1+2}（数据 RE 索引随样本传入）。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config

# ================= LWM 骨干（官方结构 + encode） =================
# 常量与官方 lwm_model.py 保持一致
ELEMENT_LENGTH = 16
D_MODEL = 64
MAX_LEN = 129
N_LAYERS = 12
N_HEADS = 12
D_FF = D_MODEL * 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
DROPOUT = 0.1


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x[:, :, 0])
        tok_emb = self.proj(x.float())
        embedding = tok_emb + self.pos_embed(pos)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS)
        self.linear = nn.Linear(N_HEADS * D_V, D_MODEL)
        self.norm = LayerNormalization(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_V)
        output = self.linear(output)
        return residual + self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return x + self.dropout(output)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, enc_inputs):
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_outputs = self.norm(attn_outputs)
        enc_outputs = self.pos_ffn(attn_outputs)
        return enc_outputs, attn


class lwm(nn.Module):
    """与官方 wi-lab/lwm 的 lwm_model.lwm 完全一致（可加载官方权重），新增 encode()。"""

    def __init__(self, element_length=16, d_model=64, max_len=129, n_layers=12):
        super().__init__()
        self.embedding = Embedding(element_length, d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        d_model, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    @classmethod
    def from_pretrained(cls, ckpt_name, device="cpu"):
        model = cls().to(device)
        sd = torch.load(ckpt_name, map_location=device)
        model.load_state_dict(sd)
        print(f"[LWM] loaded official weights from {ckpt_name}")
        return model

    def encode(self, input_ids):
        """input_ids: (B, T, 16) -> output: (B, T, 64)（逐 patch 隐状态，含 CLS）"""
        output = self.embedding(input_ids)
        for layer in self.layers:
            output, _ = layer(output)
        return output

    def forward(self, input_ids, masked_pos):
        output = self.encode(input_ids)
        masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.relu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, output


# ================= LLR Decoder（CNN 残差网络，参考 NNreceiver） =================
# 优化点：不再输入 llr_base（传统 max-log 软解调结果），直接由 CNN 在
# (符号 × 子载波) 全网格上预测逐比特 LLR logits，降低推理复杂度。

# 特征通道数：64(emb)+16(H patch)+2(Re/Im z)+1(σ²)+4(mod_oh)=87
# + 配置元数据 CFG_DIM=14（接收机已知的系统参数）-> 101，补零到 102 满足 GroupNorm(groups=2)
FEAT_CH = 87 + config.CFG_DIM + (1 if (87 + config.CFG_DIM) % 2 else 0)


class CNNResidualBlock(nn.Module):
    """
    残差块（NNreceiver residualBlock 风格）：
      GroupNorm(2) -> 3x3 空洞卷积 -> ReLU -> GroupNorm(2) -> 3x3 空洞卷积 -> + 捷径
    sep_conv=True 时卷积替换为深度可分离（depthwise 3x3 空洞 + pointwise 1x1）。
    通道变化时捷径为 1x1 卷积，否则恒等。
    """

    def __init__(self, in_channels, out_channels, dilation, group_norm=True, sep_conv=True):
        super().__init__()
        self.group_norm = group_norm
        self.dilation = tuple(dilation)
        if group_norm:
            self.gn1 = nn.GroupNorm(2, in_channels)
            self.gn2 = nn.GroupNorm(2, out_channels)
        if sep_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=self.dilation,
                          dilation=self.dilation, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=self.dilation,
                          dilation=self.dilation, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=self.dilation,
                                   dilation=self.dilation)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=self.dilation,
                                   dilation=self.dilation)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        z = self.gn1(x) if self.group_norm else x
        z = self.act(self.conv1(z))
        z = self.gn2(z) if self.group_norm else z
        z = self.conv2(z)
        return residual + z


class CNNLLRDecoder(nn.Module):
    """
    CNN 残差 LLR 解码器（NNreceiver 架构移植）：
      init_norm: GroupNorm(groups=2)
      conv1    : 3x3 转置卷积 stride=1（padding=1 保持网格尺寸）-> 64 通道
      11 个残差块: out_channels=[64,64,128,128,256,256,256,128,128,64,64]
                   dilation=[(1,1),(1,1),(2,3),(2,3),(2,3),(3,6),(2,3),(2,3),(2,3),(1,1),(1,1)]
      outconv  : 3x3 卷积 -> num_bits_per_symbol（最大 8 bit，256QAM）
    输入 (B, C, n_symb, n_sc) -> 输出 (B, n_bits, n_symb, n_sc) LLR logits。
    """

    def __init__(self, in_channels=FEAT_CH, num_bits_per_symbol=config.MAX_BITS,
                 group_norm=config.CNN_GROUP_NORM, sep_conv=config.CNN_SEP_CONV,
                 transpose=config.CNN_TRANSPOSE):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_res_blok = 11
        self.dilation = [(1, 1), (1, 1), (2, 3), (2, 3), (2, 3),
                         (3, 6), (2, 3), (2, 3), (2, 3), (1, 1), (1, 1)]
        self.out_channels = [64, 64, 128, 128, 256, 256, 256, 128, 128, 64, 64]
        self.transpose = transpose
        self.init_norm = nn.GroupNorm(2, in_channels)
        if transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=3,
                                            stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        prev = 64
        for i in range(self.num_res_blok):
            self.blocks.append(CNNResidualBlock(prev, self.out_channels[i],
                                                self.dilation[i], group_norm, sep_conv))
            prev = self.out_channels[i]
        if transpose:
            self.outconv = nn.Conv2d(prev, num_bits_per_symbol, kernel_size=3, padding=1)
        else:
            self.outconv = nn.Conv2d(prev, num_bits_per_symbol, kernel_size=1)

    def forward(self, x):
        z = self.init_norm(x)
        z = self.conv1(z)
        for blk in self.blocks:
            z = blk(z)
        z = self.outconv(z)
        return z


# ================= 组合模型 =================

from data_gen import qam_constellation


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
    def _build_feat(h_emb, patch, z, sigma2, mod_oh, data_re_idx, cfg):
        """
        构建 CNN 输入特征图 (B, FEAT_CH, n_symb, n_sc)：
          通道 = channel_emb(64) + H_patch(16) + Re(z) + Im(z) + σ² + mod_oh(4)
               + 配置元数据 cfg(CFG_DIM) = 87+CFG_DIM，补零到偶数满足 GroupNorm(2)。
          z 仅在数据 RE 位置有值（data_re_idx 为 (n_data,2) [sc,symb]），其余为 0。
          cfg: (B, CFG_DIM) 接收机已知的系统参数（天线/RB/符号/DMRS/TDL/速度）。
        """
        B = h_emb.shape[0]
        S, SC = h_emb.shape[1], h_emb.shape[2]
        dev = h_emb.device
        feat = torch.cat([h_emb, patch], dim=-1)            # (B, S, SC, 80)
        z_re = torch.zeros(B, S, SC, device=dev, dtype=torch.float32)
        z_im = torch.zeros_like(z_re)
        n_data = len(data_re_idx)
        b_idx = torch.arange(B, device=dev)[:, None].expand(B, n_data).reshape(-1)
        s_idx = torch.as_tensor(data_re_idx[:, 1], device=dev)[None, :].expand(B, n_data).reshape(-1)
        c_idx = torch.as_tensor(data_re_idx[:, 0], device=dev)[None, :].expand(B, n_data).reshape(-1)
        z_re.index_put_((b_idx, s_idx, c_idx), z.real.reshape(-1))
        z_im.index_put_((b_idx, s_idx, c_idx), z.imag.reshape(-1))
        feat = torch.cat([feat, z_re[..., None], z_im[..., None]], dim=-1)  # (B, S, SC, 82)
        s2 = sigma2.reshape(B, 1, 1, 1).expand(B, S, SC, 1)
        mo = mod_oh.reshape(B, 1, 1, -1).expand(B, S, SC, -1)
        cfg_v = cfg.reshape(B, 1, 1, -1).expand(B, S, SC, -1)
        feat = torch.cat([feat, s2, mo, cfg_v], dim=-1)     # (B, S, SC, 87+CFG_DIM)
        n_pad = FEAT_CH - feat.shape[-1]
        if n_pad > 0:
            feat = F.pad(feat, (0, n_pad))                  # 补零到偶数
        return feat.permute(0, 3, 1, 2)                     # (B, FEAT_CH, S, SC)

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
        output = self.backbone.encode(input_ids)       # (B*n_symb, n_sc+1, 64)
        h_emb = output[:, 1:1 + n_sc, :].reshape(B, n_symb, n_sc, -1)   # (B, S, SC, 64)
        patch = input_ids[:, 1:1 + n_sc, :].reshape(B, n_symb, n_sc, -1)  # (B, S, SC, 16)
        feat = self._build_feat(h_emb, patch, z, sigma2, mod_oh, data_re_idx, cfg)  # (B,FEAT_CH,S,SC)
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
    from tokenizer import data_re_index
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
