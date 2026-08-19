# -*- coding: utf-8 -*-
"""数据分片缓存测试：分片 roundtrip、断点续跑判定、旧单文件兼容。"""
import os
import pickle
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets.loader import (save_samples_shards, load_samples_shards,
                                 _cache_ready)


class TestShards(unittest.TestCase):
    def _samples(self, n):
        return [{"idx": i, "val": [float(i), float(i + 1)]} for i in range(n)]

    def test_roundtrip(self):
        samples = self._samples(25)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "cache.pkl")
            save_samples_shards(samples, p, shard_size=10)   # 3 片
            self.assertTrue(_cache_ready(p))
            loaded = load_samples_shards(p)
            self.assertEqual(len(loaded), 25)
            self.assertEqual(loaded[0]["idx"], 0)
            self.assertEqual(loaded[24]["idx"], 24)

    def test_missing_shard_not_ready(self):
        samples = self._samples(10)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "cache.pkl")
            save_samples_shards(samples, p, shard_size=5)    # 2 片
            os.remove(f"{p}.001")
            self.assertFalse(_cache_ready(p))

    def test_legacy_single_file(self):
        """无 manifest 时回退单文件（旧缓存兼容）"""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "cache.pkl")
            with open(p, "wb") as f:
                pickle.dump([{"x": 1}], f)
            loaded = load_samples_shards(p)
            self.assertEqual(loaded, [{"x": 1}])
            self.assertTrue(_cache_ready(p))


if __name__ == "__main__":
    unittest.main()
