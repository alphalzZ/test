# -*- coding: utf-8 -*-
"""
CNN LLR Decoder（NNreceiver 架构移植）：残差卷积网络。
输入 = 全网格特征图（LWM 隐状态 + 信道 patch + 均衡符号 + 噪声 + 调制/配置元数据），
输出 = 逐数据 RE 的逐比特 LLR logits（正=bit1）。
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
import torch.nn as nn

from src.utils import config


# ================= LLR Decoder（CNN 残差网络，参考 NNreceiver） =================
# 优化点：不再输入 llr_base（传统 max-log 软解调结果），直接由 CNN 在
# (符号 × 子载波) 全网格上预测逐比特 LLR logits，降低推理复杂度。

# 特征通道数：64(emb)+16(H patch)+2(Re/Im z)+1(σ²)+4(mod_oh)=87
# + 配置元数据 CFG_DIM=9（接收端可感知：n_rx/n_sc/n_symb/dmrs，无信道模型信息）
#   + LWM 浅层特征 len(SHALLOW_LAYERS)*64，
# 补零到偶数满足 GroupNorm(groups=2)
SHALLOW_FEAT_DIM = len(config.SHALLOW_LAYERS) * 64
FEAT_CH = 87 + config.CFG_DIM + SHALLOW_FEAT_DIM + (
    1 if (87 + config.CFG_DIM + SHALLOW_FEAT_DIM) % 2 else 0)


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
