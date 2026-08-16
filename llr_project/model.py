# -*- coding: utf-8 -*-
"""
模型定义（设计文档第 5、6 节）：
  1. LWM 骨干：与官方 lwm_model.py 结构完全一致（保证可加载官方权重），
     新增 encode() 返回逐 patch 隐状态。
  2. LLRDecoder：逐子载波 MLP（方案 A）。
     输入 [channel_emb(64) + Re(z) + Im(z) + σ² + mod_onehot(4)] -> 输出 log2M 个 LLR
  3. LWMLLR：组合模型，输入 (H, z, σ², mod_onehot) -> (B, T, 8) LLR
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


# ================= LLR Decoder（残差学习：修正传统 demapper 输出） =================

class LLRDecoder(nn.Module):
    """
    输入 (B, T, 64) channel_emb + (B, T, 16) H_est_patch + (B, T) complex z
        + (B,) σ² + (B, 4) mod_oh + (B, T, 8) llr_base
    -> (B, T, 8) Δ（对传统软解调基线 LLR 的修正量）
    最终 LLR = llr_base + Δ。模型至少达到传统基线水平，学习"信道先验增强"。
    """

    def __init__(self, d_emb=64, d_patch=ELEMENT_LENGTH, hidden=128,
                 out_bits=config.MAX_BITS, mod_onhot_dim=config.MOD_ONHOT_DIM):
        super().__init__()
        in_dim = d_emb + d_patch + 2 + 1 + mod_onhot_dim + out_bits   # 64+16+2+1+4+8=95
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, out_bits),
        )
        self.max_llr = config.MAX_LLR

    def forward(self, h_emb, h_patch, z, sigma2, mod_oh, llr_base):
        """
        h_emb  : (B, T, 64)
        h_patch: (B, T, 16)
        z      : (B, T) complex
        sigma2 : (B,) or (B,1)
        mod_oh : (B, mod_dim)
        llr_base: (B, T, out_bits)
        -> (B, T, out_bits) 修正后的 LLR = llr_base + Δ
        """
        z_re = z.real.unsqueeze(-1)      # (B, T, 1)
        z_im = z.imag.unsqueeze(-1)
        s2 = sigma2.reshape(-1, 1, 1).expand(-1, z_re.shape[1], 1)
        m = mod_oh.unsqueeze(1).expand(-1, z_re.shape[1], -1)
        x = torch.cat([h_emb, h_patch, z_re, z_im, s2, m, llr_base], dim=-1)  # (B, T, 95)
        delta = torch.tanh(self.mlp(x)) * self.max_llr
        llr = torch.clamp(llr_base + delta, -self.max_llr, self.max_llr)
        return llr


# ================= 组合模型 =================

# 128 子载波块内的数据子载波索引（comb-4 导频 -> 96 个数据子载波）
from data_gen import data_subcarrier_idx, qam_constellation, demap_llr
DATA_IDX_128 = data_subcarrier_idx(config.N_SC, config.PILOT_SPACING)
N_DATA_128 = len(DATA_IDX_128)


class LWMLLR(nn.Module):
    """LWM 骨干 + LLR decoder。输入信道矩阵 H -> 数据子载波的逐比特 LLR。"""

    def __init__(self, backbone=None, freeze_backbone=False, device="cpu"):
        super().__init__()
        self.backbone = backbone if backbone is not None else lwm()
        self.decoder = LLRDecoder()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _tokenize(H):
        """
        H: (B, 8, 128) complex -> input_ids (B, 129, 16) float
        patch_k = [Re(H[:,k]); Im(H[:,k])]，加 CLS token。
        """
        B, _, T = H.shape
        real = H.real.transpose(1, 2)      # (B, T, 8)
        imag = H.imag.transpose(1, 2)
        patches = torch.cat([real, imag], dim=-1)          # (B, T, 16)
        cls = 0.2 * torch.ones(B, 1, 16, dtype=patches.dtype, device=H.device)
        return torch.cat([cls, patches], dim=1)            # (B, 129, 16)

    def forward(self, H, z, sigma2, mod_oh, llr_base):
        """
        H: (B, 8, 128) complex（128 子载波块，含导频）
        z: (B, 96) complex（数据子载波均衡符号）
        sigma2: (B,)
        mod_oh: (B, 4)
        llr_base: (B, 96, 8) 传统均衡后软解调基线 LLR
        -> llr (B, 96, 8)（基线 + 修正量，数据子载波逐比特 LLR）
        """
        input_ids = self._tokenize(H)                 # (B, 129, 16)
        output = self.backbone.encode(input_ids)      # (B, 129, 64)
        h_emb = output[:, 1:, :]                      # (B, 128, 64) 逐子载波
        h_data = h_emb[:, DATA_IDX_128, :]            # (B, 96, 64) 仅数据子载波
        patch_all = input_ids[:, 1:, :]               # (B, 128, 16) H_est patch
        patch_data = patch_all[:, DATA_IDX_128, :]    # (B, 96, 16)
        return self.decoder(h_data, patch_data, z, sigma2, mod_oh, llr_base)

    def infer_llr(self, H, z, sigma2, mod_order, sigma2_eq=None):
        """
        推理入口：支持任意 N_sc（自动分块，每块 128 子载波，输出按数据位置拼回）。
        H: (N_ant, N_sc) complex
        z: (n_data_total,) complex（全部数据子载波的均衡符号）
        sigma2: float
        mod_order: int
        sigma2_eq: (n_data_total,) 均衡后等效噪声方差（计算基线 LLR 用）
        -> llr (n_data_total, log2M) float32
        """
        self.eval()
        block = config.N_SC
        n_sc = H.shape[1]
        data_idx = data_subcarrier_idx(n_sc, config.PILOT_SPACING)
        mod_oh = np.zeros((1, config.MOD_ONHOT_DIM), dtype=np.float32)
        mod_oh[0, config.MOD_ORDERS.index(mod_order)] = 1.0
        X, btab = qam_constellation(mod_order)
        llr_blocks = []
        z_idx = 0
        with torch.no_grad():
            for start in range(0, n_sc, block):
                end = min(start + block, n_sc)
                H_b = H[:, start:end]
                blk_data_rel = data_idx[(data_idx >= start) & (data_idx < end)] - start
                n_d = len(blk_data_rel)
                z_b = z[z_idx:z_idx + n_d]
                s2eq_b = sigma2_eq[z_idx:z_idx + n_d] if sigma2_eq is not None else None
                z_idx += n_d
                if end - start < block:
                    pad = np.zeros((config.N_ANT, block - (end - start)), dtype=np.complex64)
                    H_b = np.concatenate([H_b, pad], axis=1)
                # 基线 LLR（均衡后 demap），补零位置填充 0
                llr_base = np.zeros((1, N_DATA_128, config.MAX_BITS), dtype=np.float32)
                if s2eq_b is not None:
                    llr_base[0, :n_d, :btab.shape[1]] = demap_llr(
                        z_b, s2eq_b, X, btab, config.MAX_LLR)
                H_t = torch.tensor(H_b[None], dtype=torch.complex64)
                z_t = torch.zeros((1, N_DATA_128), dtype=torch.complex64)
                z_t[0, :n_d] = torch.tensor(z_b, dtype=torch.complex64)
                s2_t = torch.tensor([sigma2], dtype=torch.float32)
                mo_t = torch.tensor(mod_oh, dtype=torch.float32)
                lb_t = torch.tensor(llr_base, dtype=torch.float32)
                llr_b = self(H_t, z_t, s2_t, mo_t, lb_t)[0, :n_d].cpu().numpy()
                llr_blocks.append(llr_b)
        llr = np.concatenate(llr_blocks, axis=0)
        return llr.astype(np.float32)


def load_official_backbone(device="cpu"):
    """加载官方 LWM 权重"""
    ckpt = config.LWM_OFFICIAL_CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"官方权重不存在: {ckpt}。请先克隆 LWM 仓库（含 model_weights.pth）")
    return lwm.from_pretrained(ckpt, device=device)


if __name__ == "__main__":
    # 冒烟测试：官方权重加载 + 前向
    bb = load_official_backbone()
    model = LWMLLR(bb).eval()
    H = torch.randn(2, 8, 128, dtype=torch.complex64)
    z = torch.randn(2, 128, dtype=torch.complex64)
    s2 = torch.tensor([0.1, 0.2])
    mo = torch.zeros(2, 4)
    mo[:, 1] = 1.0
    out = model(H, z, s2, mo)
    print("LWMLLR output:", tuple(out.shape), out.dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print("total params:", n_params)
    print("model OK")
