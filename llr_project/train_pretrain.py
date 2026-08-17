# -*- coding: utf-8 -*-
"""
阶段 1：MCM 继续预训练（Sionna PUSCH 3D 信道，GPU 加速）
逐 OFDM 符号独立做 MCM：mask 15% 的子载波 patch 并重建。

- GPU 可用时自动使用 CUDA（6GB 显存限制下用 batch 16 + 梯度累积）
- Sionna 数据生成一次并缓存到 data/，后续训练直接加载

用法：python train_pretrain.py [--epochs 30] [--samples 3000]
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
from dataset import pack_pretrain, save_pretrain_cache, load_pretrain_cache
from tokenizer import make_mcm_batch, tokenize_3d
from model import load_official_backbone, LWMLLR


def build_pretrain_data(n_samples, seed):
    """生成或加载缓存"""
    cache = config.CACHE_PT
    if os.path.exists(cache):
        print(f"[PT] 加载缓存数据: {cache}")
        return load_pretrain_cache(cache)
    print(f"[PT] 生成 {n_samples} 个 Sionna PUSCH 信道样本 (seed={seed}) ...")
    t0 = time.time()
    samples = generate_dataset(n_samples, num_rx_ant=config.N_ANT,
                               n_size_grid=config.N_SC // 12, seed=seed)
    H = pack_pretrain(samples)   # (N, 8, 120, 14)
    save_pretrain_cache(cache, H)
    print(f"[PT] 数据就绪: {H.shape} 已缓存 ({time.time()-t0:.1f}s)")
    return H


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[PT] device={device}")

    H = build_pretrain_data(args.samples, seed=args.seed)
    ds = torch.utils.data.TensorDataset(torch.tensor(H))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    backbone = load_official_backbone(device=device)
    backbone.train()

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    print(f"[PT] params={sum(p.numel() for p in backbone.parameters())/1e6:.2f}M, "
          f"batch={args.batch}, grad_accum={args.grad_accum}")

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        n_iter = 0
        optimizer.zero_grad()
        for bi, (H_b,) in enumerate(loader):
            H_b = H_b.to(device)                      # (B, 8, 120, 14) complex
            B = H_b.shape[0]
            # GPU 向量化 tokenize（逐符号）：(B*14, 129, 16)
            input_ids = LWMLLR._tokenize_3d(H_b)
            B_seq = input_ids.shape[0]
            # MCM mask：每序列随机 mask 15% 的 patch（1..128，不含 CLS）
            n_mask = int(np.floor(config.PT_MASK_RATIO * 128))
            perm = torch.argsort(torch.rand(B_seq, 128, device=device), dim=1)
            masked_pos = perm[:, :n_mask] + 1                       # (B_seq, n_mask)
            masked_tokens = input_ids.gather(
                1, masked_pos.unsqueeze(-1).expand(-1, -1, 16))    # (B_seq, n_mask, 16)
            # 应用 mask（80% [MASK]=0.1, 10% 随机, 10% 保留）
            input_ids_masked = input_ids.clone()
            rows = torch.arange(B_seq, device=device).unsqueeze(1).expand(-1, n_mask)
            r = torch.rand(B_seq, n_mask, device=device)
            r_3d = r.unsqueeze(-1)                     # (B_seq, n_mask, 1)
            mask_val = torch.where(r_3d < 0.8, torch.full_like(masked_tokens, 0.1),
                                   torch.rand_like(masked_tokens))
            keep = (r_3d >= 0.9)
            input_ids_masked[rows, masked_pos, :] = torch.where(keep, masked_tokens, mask_val)

            logits_lm, _ = backbone(input_ids_masked, masked_pos)
            loss = criterion(logits_lm, masked_tokens) / args.grad_accum
            loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            running += loss.item() * args.grad_accum
            n_iter += 1
        avg = running / max(n_iter, 1)
        print(f"[PT] epoch {epoch}/{args.epochs}  loss={avg:.5f}  ({time.time()-t0:.1f}s)")

    torch.save(backbone.state_dict(), config.CKPT_PRETRAIN)
    print(f"[PT] 保存继续预训练权重 -> {config.CKPT_PRETRAIN}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.PT_EPOCHS)
    p.add_argument("--samples", type=int, default=config.PT_N_SAMPLES)
    p.add_argument("--batch", type=int, default=config.PT_BATCH)
    p.add_argument("--grad-accum", type=int, default=config.PT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=config.PT_LR)
    p.add_argument("--seed", type=int, default=config.PT_SEED)
    train(p.parse_args())
