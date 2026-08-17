# -*- coding: utf-8 -*-
"""
阶段 2：LLR 微调（设计文档第 6、7 节，GPU 加速）
在 LWM（继续预训练或官方权重）之上训练 LLR decoder，监督标签为 max-log 参考 LLR。

- GPU 可用时自动使用 CUDA（batch 16 + 梯度累积等效 64）
- Sionna 数据生成一次缓存到 data/，训练直接加载

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_gen_sionna import generate_dataset
from dataset import (LLRDataset, llr_collate, pack_samples,
                     save_packed, load_packed)
from model import LWMLLR, load_official_backbone


def build_ft_data(n_train, n_val, seed, cache_tr, cache_va):
    """生成或加载缓存"""
    if os.path.exists(cache_tr) and os.path.exists(cache_va):
        print(f"[FT] 加载缓存数据")
        return load_packed(cache_tr), load_packed(cache_va)
    print(f"[FT] 生成微调数据: train={n_train}, val={n_val} (seed={seed}) ...")
    t0 = time.time()
    tr = generate_dataset(n_train, num_rx_ant=config.N_ANT,
                          n_size_grid=config.N_SC // 12, seed=seed)
    va = generate_dataset(n_val, num_rx_ant=config.N_ANT,
                          n_size_grid=config.N_SC // 12, seed=seed + 1000)
    tr_p = pack_samples(tr)
    va_p = pack_samples(va)
    save_packed(cache_tr, tr_p)
    save_packed(cache_va, va_p)
    print(f"[FT] 数据就绪并缓存 ({time.time()-t0:.1f}s)")
    return tr_p, va_p


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[FT] device={device}")

    tr_p, va_p = build_ft_data(args.train_n, args.val_n, seed=args.seed,
                               cache_tr=config.CACHE_FT_TRAIN,
                               cache_va=config.CACHE_FT_VAL)
    tr_ds = LLRDataset(tr_p)
    va_ds = LLRDataset(va_p)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                                            collate_fn=llr_collate)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                                            collate_fn=llr_collate)

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
    criterion = nn.MSELoss(reduction="none")

    def loss_fn(pred, llr, valid, nbits):
        """有效 RE × 有效比特掩码上的归一化 MSE（LLR 归一到 [-1,1]）"""
        B, T, K = pred.shape
        bit_mask = (torch.arange(K, device=pred.device)[None, None, :] <
                    nbits[:, None, None]).float()
        mask = valid.unsqueeze(-1) * bit_mask          # (B, T, K)
        scale = config.MAX_LLR
        loss = (criterion(pred / scale, llr / scale) * mask).sum() / (mask.sum() + 1e-6)
        return loss

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        optimizer.zero_grad()
        for bi, (H, z, llr, llr_base, bits, mo, s2, valid, nbits) in enumerate(tr_loader):
            H, z, llr, llr_base, mo, s2, valid, nbits = (
                H.to(device), z.to(device), llr.to(device), llr_base.to(device),
                mo.to(device), s2.to(device), valid.to(device), nbits.to(device))
            pred = model(H, z, s2, mo, llr_base)
            loss = loss_fn(pred, llr, valid, nbits) / args.grad_accum
            loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            run += loss.item() * args.grad_accum

        # 验证
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for H, z, llr, llr_base, bits, mo, s2, valid, nbits in va_loader:
                H, z, llr, llr_base, mo, s2, valid, nbits = (
                    H.to(device), z.to(device), llr.to(device), llr_base.to(device),
                    mo.to(device), s2.to(device), valid.to(device), nbits.to(device))
                pred = model(H, z, s2, mo, llr_base)
                vrun += loss_fn(pred, llr, valid, nbits).item()
        tl = run / len(tr_loader)
        vl = vrun / len(va_loader)
        print(f"[FT] epoch {epoch}/{args.epochs}  train={tl:.5f}  val={vl:.5f}  ({time.time()-t0:.1f}s)")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_out)
            print(f"      -> saved {ckpt_out} (val {vl:.5f})")

    print(f"[FT] 完成，最佳 val loss={best_val:.5f}，权重 -> {ckpt_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=config.FT_TRAIN_N)
    p.add_argument("--val-n", type=int, default=config.FT_VAL_N)
    p.add_argument("--epochs", type=int, default=config.FT_EPOCHS)
    p.add_argument("--batch", type=int, default=config.FT_BATCH)
    p.add_argument("--grad-accum", type=int, default=config.FT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=config.FT_LR)
    p.add_argument("--lr-backbone", type=float, default=config.FT_LR_BACKBONE)
    p.add_argument("--seed", type=int, default=config.FT_SEED)
    p.add_argument("--no-pretrain", action="store_true", help="对照：官方权重直接微调")
    p.add_argument("--freeze-backbone", action="store_true", help="冻结骨干只训 decoder")
    train(p.parse_args())
