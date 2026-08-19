# -*- coding: utf-8 -*-
"""
阶段 1：MCM 继续预训练（多配置版，GPU 加速）
逐 OFDM 符号独立做 MCM：mask 15% 的子载波 patch 并重建。
训练数据为多配置混合（天线 1/2/4/8 × RB 1~10 × 符号 3~14 × DMRS 3 模式
× TDL-A/B/C/D × 时延/多普勒），按 (n_sc, n_rx, n_symb) 分桶保证 batch 内等形。

- GPU 可用时自动使用 CUDA（6GB 显存限制下用 batch 8 + 梯度累积）
- Sionna 数据生成一次并缓存到 data/，后续训练直接加载

用法：python train_pretrain.py [--epochs 15] [--samples 2000]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils import config
from src.datasets.loader import build_data
from src.models.lwm_llr import load_official_backbone, LWMLLR


def make_pt_batches(samples, batch_size, seed):
    """按 (n_sc, n_rx, n_symb) 分桶 -> 每批 (B, n_rx, n_sc, n_symb) H 张量。
    组内不足 batch_size 时有放回采样凑满（多配置数据组合多、每组样本少）。"""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(samples):
        sh = s["H_est"].shape
        groups.setdefault((int(sh[1]), int(sh[0]), int(sh[2])), []).append(s)
    batches = []
    for key, ss in groups.items():
        idx = list(range(len(ss)))
        rng.shuffle(idx)
        if len(idx) < batch_size:
            idx = [int(i) for i in rng.choice(idx, size=batch_size, replace=True)]
            sub = [ss[j] for j in idx]
            H = np.stack([s["H_est"] for s in sub])
            batches.append(torch.tensor(H))
        else:
            for i in range(0, len(idx), batch_size):
                sub = [ss[j] for j in idx[i:i + batch_size]]
                H = np.stack([s["H_est"] for s in sub])   # (B, n_rx, n_sc, n_symb)
                batches.append(torch.tensor(H))
    rng.shuffle(batches)
    return batches


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[PT] device={device}")

    _, _, pt = build_data(args.samples, args.val_n, args.samples,
                             seed=args.seed,
                             cache_tr=config.CACHE_TRAIN,
                             cache_va=config.CACHE_VAL,
                             cache_pt=config.CACHE_PT)
    batches = make_pt_batches(pt, batch_size=args.batch, seed=args.seed)
    n_scs = sorted({int(s["H_est"].shape[1]) for s in pt})
    print(f"[PT] 预训练样本 {len(pt)}，子载波配置 {n_scs} 种，batch {args.batch}")

    backbone = load_official_backbone(device=device)
    backbone.train()

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    print(f"[PT] params={sum(p.numel() for p in backbone.parameters())/1e6:.2f}M, "
          f"batch={args.batch}, grad_accum={args.grad_accum}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        n_iter = 0
        optimizer.zero_grad()
        for H_b in batches:
            H_b = H_b.to(device)                      # (B, n_rx, n_sc, n_symb)
            B, _, n_sc, n_symb = H_b.shape
            # GPU 向量化 tokenize（逐符号）：(B*n_symb, n_sc+1, 16)
            input_ids = LWMLLR._tokenize_3d(H_b)
            B_seq = input_ids.shape[0]
            # MCM mask：每序列随机 mask 15% 的 patch（1..n_sc，不含 CLS）
            n_mask = max(1, int(np.floor(config.PT_MASK_RATIO * n_sc)))
            perm = torch.argsort(torch.rand(B_seq, n_sc, device=device), dim=1)
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
            if (n_iter + 1) % args.grad_accum == 0:
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
    p.add_argument("--samples", type=int, default=config.PT_N)
    p.add_argument("--val-n", type=int, default=config.VAL_N)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=config.PT_LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    train(p.parse_args())
