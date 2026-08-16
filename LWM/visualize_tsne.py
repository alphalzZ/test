# -*- coding: utf-8 -*-
"""
LWM embedding 可视化：PCA/t-SNE 2D 投影，按 LoS/NLoS 着色
对比 raw 数据 vs LWM cls_emb，直观展示特征提取效果
"""
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from input_preprocess import tokenizer, DeepMIMO_data_gen, label_gen
from lwm_model import lwm
from inference import lwm_inference, create_raw_dataset

SCENARIO = "city_18_denver"

def main():
    device = 'cpu'
    model = lwm.from_pretrained(device=device)
    model.eval()

    # 数据
    preprocessed_chs = tokenizer(selected_scenario_names=SCENARIO, gen_raw=True)
    data = DeepMIMO_data_gen(SCENARIO)
    y = np.array(label_gen('LoS/NLoS Classification', data, SCENARIO))

    X_cls  = lwm_inference(preprocessed_chs, 'cls_emb', model, device).cpu().numpy()
    X_raw  = create_raw_dataset(preprocessed_chs, device).cpu().numpy().reshape(len(preprocessed_chs), -1)

    # 采样（t-SNE 太慢，取 400 个样本）
    rng = np.random.RandomState(0)
    idx = rng.choice(len(y), min(400, len(y)), replace=False)
    X_cls_s, X_raw_s, y_s = X_cls[idx], X_raw[idx], y[idx]

    # 投影
    def project(X):
        pca = PCA(n_components=50, random_state=0).fit_transform(X)
        return TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(pca)

    print("t-SNE on raw ...")
    raw_2d = project(X_raw_s)
    print("t-SNE on cls_emb ...")
    cls_2d = project(X_cls_s)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X2d, title in [(axes[0], raw_2d, 'Raw channels (32x32)'),
                           (axes[1], cls_2d, 'LWM CLS embeddings')]:
        for label, color, name in [(0, '#d62728', 'NLoS'), (1, '#1f77b4', 'LoS')]:
            m = y_s == label
            ax.scatter(X2d[m, 0], X2d[m, 1], c=color, s=8, alpha=0.7, label=name)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.axis('off')
    fig.suptitle(f'LWM feature extraction on {SCENARIO} (t-SNE)', fontsize=14)
    plt.tight_layout()
    plt.savefig('lwm_tsne_comparison.png', dpi=150)
    print("Saved -> LWM/lwm_tsne_comparison.png")

if __name__ == '__main__':
    main()
