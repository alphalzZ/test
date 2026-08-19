# -*- coding: utf-8 -*-
"""Tokenizer 测试：多配置 3D 信道 -> 子载波对齐 patch 序列。"""
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets.tokenizer import tokenize_3d_var, data_re_index


class TestTokenizer(unittest.TestCase):
    def test_tokenize_shapes(self):
        """序列 = [CLS] + n_sc patches，每 patch 16 维（天线补零到 8）"""
        rng = np.random.default_rng(0)
        for n_ant, n_sc, n_symb in [(1, 12, 3), (4, 48, 7), (8, 120, 14)]:
            H = (rng.standard_normal((n_ant, n_sc, n_symb))
                 + 1j * rng.standard_normal((n_ant, n_sc, n_symb)))
            blocks = tokenize_3d_var(H)
            self.assertEqual(blocks.shape, (n_symb, n_sc + 1, 16))

    def test_cls_token(self):
        H = np.zeros((4, 12, 3), dtype=np.complex64)
        blocks = tokenize_3d_var(H)
        # CLS = 0.2 常量
        self.assertTrue(np.allclose(blocks[:, 0], 0.2))

    def test_data_re_index(self):
        idx = data_re_index(n_sc=12, n_symb=3, dmrs_symbs=(0,))
        self.assertEqual(len(idx), 24)          # 2 个数据符号 × 12 sc
        self.assertTrue(all(sy != 0 for _, sy in idx))
        self.assertEqual(idx.shape[1], 2)       # [sc, symb]


if __name__ == "__main__":
    unittest.main()
