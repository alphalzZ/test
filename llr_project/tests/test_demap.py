# -*- coding: utf-8 -*-
"""QAM 星座与 max-log demapper 测试。"""
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets.demap import qam_constellation, demap_llr


class TestQAMConstellation(unittest.TestCase):
    def test_shapes_and_energy(self):
        for m in (4, 16, 64, 256):
            X, bits = qam_constellation(m)
            self.assertEqual(len(X), m)
            self.assertEqual(bits.shape, (m, int(np.log2(m))))
            self.assertAlmostEqual(float(np.mean(np.abs(X) ** 2)), 1.0, places=5)

    def test_gray_labels_unique(self):
        for m in (4, 16, 64):
            X, bits = qam_constellation(m)
            # 每符号的比特标签应互不相同（双射）
            self.assertEqual(len({tuple(b) for b in bits}), m)


class TestDemapLLR(unittest.TestCase):
    def test_high_snr_correct(self):
        """无噪声时硬判决应完全正确"""
        for m in (4, 16, 64, 256):
            X, bits = qam_constellation(m)
            s = np.arange(m)
            z = X[s]
            llr = demap_llr(z, np.ones(m), X, bits, max_llr=20.0)
            hard = (llr > 0).astype(int)
            self.assertTrue(np.array_equal(hard, bits[s]),
                            f"QAM{m} 高 SNR 硬判决错误")

    def test_clip(self):
        X, bits = qam_constellation(4)
        llr = demap_llr(np.zeros(2), np.ones(2), X, bits, max_llr=20.0)
        self.assertTrue(np.all(np.abs(llr) <= 20.0))


if __name__ == "__main__":
    unittest.main()
