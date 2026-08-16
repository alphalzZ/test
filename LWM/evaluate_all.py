# -*- coding: utf-8 -*-
"""
LWM 多场景效果评估脚本
对 6 个 DeepMIMO 城市场景做 LWM 推理，评估:
  1. LoS/NLoS 二分类 (用 cls_emb / channel_emb / raw 三种输入)
  2. Beam Prediction 多分类 (n_beams=16)
用 KNN 分类器对比 LWM embedding 与原始数据的分类准确率
"""
import os
import sys
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'  # 无显示环境
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from input_preprocess import tokenizer, DeepMIMO_data_gen, label_gen
from lwm_model import lwm
from inference import lwm_inference, create_raw_dataset

SCENARIOS = [
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma",
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego",
]

def eval_knn(X, y, n_neighbors=5, test_frac=0.3, seed=42):
    """KNN 分类评估，返回 (acc, f1)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted', zero_division=0)

def get_labels(scenario, task, n_beams=16):
    data = DeepMIMO_data_gen(scenario)
    return np.array(label_gen(task, data, scenario, n_beams=n_beams))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    model = lwm.from_pretrained(device=device)
    model.eval()

    results = []
    t_all = time.time()

    for sc in SCENARIOS:
        t0 = time.time()
        print(f"=== {sc} ===")
        # 1. tokenize
        preprocessed_chs = tokenizer(selected_scenario_names=sc, manual_data=None, gen_raw=True)
        n_samples = len(preprocessed_chs)
        print(f"  samples: {n_samples}")

        # 2. 三种输入
        X_cls  = lwm_inference(preprocessed_chs, 'cls_emb',     model, device).cpu().numpy()
        X_chan = lwm_inference(preprocessed_chs, 'channel_emb', model, device).cpu().numpy().reshape(n_samples, -1)
        X_raw  = create_raw_dataset(preprocessed_chs, device).cpu().numpy().reshape(n_samples, -1)
        print(f"  shapes: cls={X_cls.shape}, chan={X_chan.shape}, raw={X_raw.shape}")

        # 3. 标签
        y_los  = get_labels(sc, 'LoS/NLoS Classification')
        y_beam = get_labels(sc, 'Beam Prediction', n_beams=16)
        print(f"  LoS classes: {np.unique(y_los)}, Beam classes: {len(np.unique(y_beam))}")

        # 4. 评估
        row = {'scenario': sc, 'n': n_samples}
        for name, X in [('raw', X_raw), ('cls_emb', X_cls), ('channel_emb', X_chan)]:
            for task, y in [('los', y_los), ('beam', y_beam)]:
                acc, f1 = eval_knn(X, y)
                row[f'{task}_{name}_acc'] = acc
                row[f'{task}_{name}_f1'] = f1
        results.append(row)
        print(f"  done in {time.time()-t0:.1f}s\n")

    # 汇总
    print("=" * 100)
    print("LWM 效果评估汇总 (KNN, 30% 测试集)")
    print("=" * 100)
    hdr = f"{'scenario':<22}{'n':>5} | {'los_raw':>8}{'los_cls':>9}{'los_chan':>10} | {'beam_raw':>9}{'beam_cls':>10}{'beam_chan':>11}"
    print(hdr)
    print("-" * 100)
    for r in results:
        print(f"{r['scenario']:<22}{r['n']:>5} | "
              f"{r['los_raw_acc']:>8.3f}{r['los_cls_emb_acc']:>9.3f}{r['los_channel_emb_acc']:>10.3f} | "
              f"{r['beam_raw_acc']:>9.3f}{r['beam_cls_emb_acc']:>10.3f}{r['beam_channel_emb_acc']:>11.3f}")
    print("-" * 100)

    # 平均
    def avg(k): return np.mean([r[k] for r in results])
    print(f"{'AVERAGE':<22}{'':>5} | "
          f"{avg('los_raw_acc'):>8.3f}{avg('los_cls_emb_acc'):>9.3f}{avg('los_channel_emb_acc'):>10.3f} | "
          f"{avg('beam_raw_acc'):>9.3f}{avg('beam_cls_emb_acc'):>10.3f}{avg('beam_channel_emb_acc'):>11.3f}")
    print(f"\nTotal time: {time.time()-t_all:.1f}s")

    import json
    with open('lwm_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved results -> LWM/lwm_eval_results.json")

if __name__ == '__main__':
    main()
