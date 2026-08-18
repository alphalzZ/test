# -*- coding: utf-8 -*-
"""
阶段 2：LLR 微调（多配置版，GPU 加速）
在 LWM（继续预训练或官方权重）之上训练 CNN LLR decoder。
监督标签：真实传输的 0/1 bit（bits_tx），损失函数：BCE（binary cross-entropy）。
模型输入不含 llr_base（无需传统软解调），decoder 为 CNN 残差网络（NNreceiver 风格）。

多配置：一个模型适配 天线 1/2/4/8 × RB 1~10 × 符号 3~14 × DMRS {1}/{1+1}/{1+2}
× TDL-A/B/C/D × 时延/多普勒，训练数据为各种配置的混合（每配置组合多样本）。
训练时按 (n_sc, n_symb, n_rx, n_data) 分桶（batch 内同配置，CNN 特征图尺寸一致）。

用法：
  python train_llr.py                       # 用阶段1权重微调（主模型）
  python train_llr.py --no-pretrain         # 用官方权重微调（对照模型）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import (BucketedLoader, build_data,
                     load_samples_pkl, save_samples_pkl)
from model import LWMLLR, load_official_backbone


def bit_mask_like(nbits, shape, device):
    """有效比特掩码 (B, T, K)：仅 log2M 以内的比特参与计算"""
    B, T, K = shape
    return (torch.arange(K, device=device)[None, None, :] <
            nbits[:, None, None]).float()


def bce_loss(pred, bits, valid, nbits):
    """有效 RE × 有效比特上的 BCE（标签为真实 0/1 bit）"""
    mask = valid.unsqueeze(-1) * bit_mask_like(nbits, pred.shape, pred.device)
    loss = F.binary_cross_entropy_with_logits(pred, bits.float(), reduction="none")
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def val_ber(pred, bits, valid, nbits):
    """硬判决 BER（LLR>0 -> bit1，越低越好）"""
    mask = valid.unsqueeze(-1) * bit_mask_like(nbits, pred.shape, pred.device)
    hard = (pred > 0).float()
    correct = ((hard == bits.float()).float() * mask).sum() / (mask.sum() + 1e-6)
    return 1.0 - correct


def to_tensors(b, device):
    """collate_batch 的 numpy dict -> GPU tensors（data_re_idx 保留 numpy）"""
    return {
        "H_est": torch.tensor(b["H_est"], device=device),
        "z": torch.tensor(b["z"], device=device),
        "bits": torch.tensor(b["bits"], device=device),
        "mod_oh": torch.tensor(b["mod_oh"], device=device),
        "sigma2": torch.tensor(b["sigma2"], device=device),
        "valid": torch.tensor(b["valid"], device=device),
        "nbits": torch.tensor(b["nbits"], device=device),
        "cfg_v": torch.tensor(b["cfg_v"], device=device),
        "data_re_idx": b["data_re_idx"],
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[FT] device={device}")

    tr, va, _ = build_data(args.train_n, args.val_n, args.pt_n,
                              seed=args.seed,
                              cache_tr=config.CACHE_TRAIN,
                              cache_va=config.CACHE_VAL,
                              cache_pt=config.CACHE_PT)
    tr_loader = BucketedLoader(tr, batch_size=args.batch, shuffle=True, seed=args.seed)
    va_loader = BucketedLoader(va, batch_size=args.batch, shuffle=False, seed=0)
    print(f"[FT] 训练样本 {len(tr)}（{tr_loader.n_groups} 种 (n_sc,n_symb) 配置），"
          f"验证样本 {len(va)}（{va_loader.n_groups} 种配置）")

    # 骨干初始化
    if args.no_pretrain:
        backbone = load_official_backbone(device=device)
        ckpt_out = config.CKPT_LLR_NO_PT
        print("[FT] 对照模式：使用官方权重（无继续预训练）")
    else:
        if not os.path.exists(config.CKPT_PRETRAIN):
            raise FileNotFoundError(f"未找到继续预训练权重 {config.CKPT_PRETRAIN}，请先运行 train_pretrain.py")
        bb = load_official_backbone(device=device)
        bb.load_state_dict(torch.load(config.CKPT_PRETRAIN, map_location=device))
        backbone = bb
        ckpt_out = config.CKPT_LLR
        print("[FT] 主模式：加载继续预训练权重")

    model = LWMLLR(backbone, freeze_backbone=args.freeze_backbone).to(device)
    print(f"[FT] total params={sum(p.numel() for p in model.parameters())/1e3:.1f}K, "
          f"batch={args.batch}, grad_accum={args.grad_accum}")

    # 分组学习率
    if args.freeze_backbone:
        params = [{"params": model.decoder.parameters(), "lr": args.lr}]
    else:
        params = [
            {"params": model.decoder.parameters(), "lr": args.lr},
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-5)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        n_batch = 0
        optimizer.zero_grad()
        for b in tr_loader:
            t = to_tensors(b, device)
            # batch 内同配置，data_re_idx 取第 0 个即可（(n_data, 2)）
            pred = model(t["H_est"], t["z"], t["sigma2"], t["mod_oh"],
                         t["data_re_idx"][0], t["cfg_v"])
            loss = bce_loss(pred, t["bits"], t["valid"], t["nbits"]) / args.grad_accum
            loss.backward()
            if (n_batch + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            run += loss.item() * args.grad_accum
            n_batch += 1

        # 验证
        model.eval()
        vrun = 0.0
        vber = 0.0
        n_vb = 0
        with torch.no_grad():
            for b in va_loader:
                t = to_tensors(b, device)
                pred = model(t["H_est"], t["z"], t["sigma2"], t["mod_oh"],
                             t["data_re_idx"][0], t["cfg_v"])
                vrun += bce_loss(pred, t["bits"], t["valid"], t["nbits"]).item()
                vber += val_ber(pred, t["bits"], t["valid"], t["nbits"]).item()
                n_vb += 1
        tl = run / max(n_batch, 1)
        vl = vrun / max(n_vb, 1)
        vb = vber / max(n_vb, 1)
        print(f"[FT] epoch {epoch}/{args.epochs}  train={tl:.5f}  val={vl:.5f}  "
              f"valBER={vb:.4f}  ({time.time()-t0:.1f}s)")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_out)
            print(f"      -> saved {ckpt_out} (val {vl:.5f}, BER {vb:.4f})")

    print(f"[FT] 完成，最佳 val loss={best_val:.5f}，权重 -> {ckpt_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=config.TRAIN_N)
    p.add_argument("--val-n", type=int, default=config.VAL_N)
    p.add_argument("--pt-n", type=int, default=config.PT_N)
    p.add_argument("--epochs", type=int, default=config.FT_EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--lr-backbone", type=float, default=config.LR_BACKBONE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-pretrain", action="store_true", help="对照：官方权重直接微调")
    p.add_argument("--freeze-backbone", action="store_true", help="冻结骨干只训 decoder")
    train(p.parse_args())
