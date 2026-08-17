# -*- coding: utf-8 -*-
"""
阶段 1：MCM 继续预训练（Sionna PUSCH 3D 信道）
用自己的 3GPP OFDM 信道数据（H_est, 8×120×14）对 LWM 做 MCM 域适配。
逐 OFDM 符号独立做 MCM：mask 15% 的子载波 patch 并重建。

用法：python train_pretrain.py [--epochs 15] [--samples 1500]
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
from dataset import pack_pretrain
from tokenizer import make_mcm_batch, tokenize_3d
from model import load_official_backbone


def build_pretrain_data(n_samples, seed):
    print(f"[PT] 生成 {n_samples} 个 Sionna PUSCH 信道样本 (seed={seed}) ...")
    t0 = time.time()
    samples = generate_dataset(n_samples, num_rx_ant=config.N_ANT,
                               n_size_grid=config.N_SC // 12, seed=seed)
    H = pack_pretrain(samples)   # (N, 8, 120, 14)
    print(f"[PT] 数据就绪: {H.shape} ({time.time()-t0:.1f}s)")
    return H


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    H = build_pretrain_data(args.samples, seed=args.seed)
    ds = torch.utils.data.TensorDataset(torch.tensor(H))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    backbone = load_official_backbone()
    backbone.train()

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)
    print(f"[PT] device={device}, params={sum(p.numel() for p in backbone.parameters())/1e6:.2f}M")

    rng = np.random.default_rng(args.seed)
    n_symb = config.N_SYMB
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        n_iter = 0
        for (H_b,) in loader:
            H_b = H_b.numpy()                     # (B, 8, 120, 14)
            B = H_b.shape[0]
            # 逐符号 tokenize -> (B*14, 129, 16)
            blocks = []
            for i in range(B):
                blk3 = tokenize_3d(H_b[i])
                blocks.extend(list(blk3))
            input_ids, masked_tokens, masked_pos = make_mcm_batch(
                blocks, mask_ratio=config.PT_MASK_RATIO, rng=rng)
            input_ids = torch.tensor(input_ids, device=device)
            masked_tokens = torch.tensor(masked_tokens, device=device)
            masked_pos = torch.tensor(masked_pos, device=device)

            optimizer.zero_grad()
            logits_lm, _ = backbone(input_ids, masked_pos)
            loss = criterion(logits_lm, masked_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
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
    p.add_argument("--lr", type=float, default=config.PT_LR)
    p.add_argument("--seed", type=int, default=config.PT_SEED)
    train(p.parse_args())
