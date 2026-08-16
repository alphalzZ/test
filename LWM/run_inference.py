# -*- coding: utf-8 -*-
"""
LWM (Large Wireless Model) 本地部署验证脚本
- 加载预训练权重
- 用 DeepMIMO fortworth 场景数据做推理
- 输出 CLS / channel embeddings
"""
import os
import sys
import time
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from input_preprocess import tokenizer
from lwm_model import lwm
from inference import lwm_inference

def main():
    t0 = time.time()

    # 1. 设备选择（GPU 不可用时回退 CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[1/4] Device: {device}")

    # 2. 数据预处理（fortworth 场景，tutorial 默认场景）
    scenario_names = np.array([
        "city_18_denver", "city_15_indianapolis", "city_19_oklahoma",
        "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
    ])
    selected_scenario_names = scenario_names[3]  # city_12_fortworth
    print(f"[2/4] Tokenizing scenario: {selected_scenario_names} ...")

    preprocessed_chs = tokenizer(
        selected_scenario_names=selected_scenario_names,
        manual_data=None,
        gen_raw=True,
        snr_db=None
    )
    print(f"      -> {len(preprocessed_chs)} samples preprocessed ({time.time()-t0:.1f}s)")

    # 3. 加载预训练模型
    print("[3/4] Loading LWM model ...")
    model = lwm.from_pretrained(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      -> {n_params/1e6:.2f}M parameters loaded ({time.time()-t0:.1f}s)")

    # 4. 推理：提取 channel embeddings
    print("[4/4] Running inference (channel_emb) ...")
    input_types = ['cls_emb', 'channel_emb', 'raw']
    selected_input_type = input_types[1]

    dataset = lwm_inference(preprocessed_chs, selected_input_type, model, device)

    print("\n========== 推理完成 ==========")
    print(f"Embedding shape : {tuple(dataset.shape)}")
    print(f"Embedding dtype : {dataset.dtype}")
    print(f"Embedding range : [{dataset.min().item():.4f}, {dataset.max().item():.4f}]")
    print(f"Total time      : {time.time()-t0:.1f}s")
    print("================================\n")

    # 5. 简单验证：保存前几个样本到文件
    np.save('lwm_embeddings_sample.npy', dataset[:16].cpu().numpy())
    print("Saved 16 sample embeddings -> LWM/lwm_embeddings_sample.npy")

if __name__ == '__main__':
    main()
