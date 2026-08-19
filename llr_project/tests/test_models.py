# -*- coding: utf-8 -*-
"""模型冒烟测试：官方权重加载 + 多配置 3D 前向（CPU，no_grad）。

注意：需要 ../LWM/model_weights.pth 官方权重存在（configs/paths.json 配置）。
"""
import os
import sys
import unittest

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import config
from src.datasets.tokenizer import data_re_index
from src.models.lwm_llr import LWMLLR, load_official_backbone


@unittest.skipUnless(os.path.exists(config.LWM_OFFICIAL_CKPT),
                     f"官方权重不存在: {config.LWM_OFFICIAL_CKPT}")
class TestLWMLLR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LWMLLR(load_official_backbone(device="cpu")).eval()

    def _forward(self, n_rx, n_sc, n_symb, dmrs_symbs):
        dri = data_re_index(n_sc=n_sc, n_symb=n_symb, dmrs_symbs=dmrs_symbs)
        cfg = torch.zeros(1, config.CFG_DIM)
        H = torch.randn(1, n_rx, n_sc, n_symb, dtype=torch.complex64)
        z = torch.randn(1, len(dri), dtype=torch.complex64)
        s2 = torch.tensor([0.1])
        mo = torch.zeros(1, config.MOD_ONHOT_DIM)
        mo[:, 1] = 1.0                                   # 16QAM
        with torch.no_grad():
            return self.model(H, z, s2, mo, dri, cfg)    # (1, n_data, 8)

    def test_multi_config_shapes(self):
        cases = [(4, 48, 7, (2,)), (8, 120, 14, (2, 7, 11)), (1, 12, 3, (0,))]
        for n_rx, n_sc, n_symb, dmrs in cases:
            out = self._forward(n_rx, n_sc, n_symb, dmrs)
            n_data = len(data_re_index(n_sc=n_sc, n_symb=n_symb, dmrs_symbs=dmrs))
            self.assertEqual(tuple(out.shape), (1, n_data, config.MAX_BITS),
                             f"{n_rx}rx/{n_sc}sc/{n_symb}symb 输出形状错误")

    def test_llr_bounded(self):
        out = self._forward(4, 48, 7, (2,))
        self.assertTrue(torch.all(out.abs() <= config.MAX_LLR + 1e-5))


if __name__ == "__main__":
    unittest.main()
