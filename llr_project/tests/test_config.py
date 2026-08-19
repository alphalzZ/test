# -*- coding: utf-8 -*-
"""配置加载器测试：默认配置、派生路径、LLR_CFG=night 覆盖。"""
import os
import subprocess
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import config


class TestConfig(unittest.TestCase):
    def test_simulation_defaults(self):
        self.assertEqual(config.SYS_FFT, 1024)
        self.assertAlmostEqual(config.SYS_SCS_HZ, 15e3)
        self.assertEqual(config.SYS_CP, 72)
        self.assertEqual(config.RX_ANTS, [1, 2, 4, 8])
        self.assertEqual(config.RB_RANGE, [1, 10])
        self.assertEqual(config.TDL_MODELS, ["A", "B", "C", "D"])

    def test_model_defaults(self):
        self.assertEqual(config.MOD_ORDERS, [4, 16, 64, 256])
        self.assertEqual(config.MAX_BITS, 8)
        self.assertEqual(config.CFG_DIM, 9)            # 接收端可感知参数（无 TDL/速度）
        self.assertEqual(config.SHALLOW_LAYERS, [3, 4, 5, 6])

    def test_training_defaults(self):
        self.assertTrue(config.USE_PRETRAIN)          # 默认小规模配置
        self.assertEqual(config.PT_N, 2000)
        self.assertEqual(config.TRAIN_N, 2400)
        self.assertEqual(config.FT_EPOCHS, 40)
        self.assertEqual(config.FT_FREEZE_EPOCHS, 20)

    def test_derived_paths(self):
        self.assertTrue(config.DATA_DIR.endswith("data"))
        self.assertTrue(config.EVAL_RESULTS.endswith(
            os.path.join("experiments", "results", "eval_results.json")))
        self.assertTrue(config.CKPT_LLR.endswith("lwm_llr.pt"))

    def test_night_override(self):
        """子进程设置 LLR_CFG=night，验证 configs/night.json 覆盖生效"""
        code = ("from src.utils import config; "
                "print(config.TRAIN_N, config.FT_EPOCHS, config.USE_PRETRAIN, "
                "config.CKPT_LLR)")
        env = {**os.environ, "LLR_CFG": "night"}
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, cwd=_ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.strip().split()
        self.assertEqual(out[0], "50000")                     # TRAIN_N
        self.assertEqual(out[1], "22")                        # FT_EPOCHS
        self.assertEqual(out[2], "False")                     # USE_PRETRAIN
        self.assertTrue(out[3].endswith("lwm_llr_night.pt"))  # CKPT_LLR


if __name__ == "__main__":
    unittest.main()
