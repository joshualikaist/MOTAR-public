"""New transfer harness regressions; frozen RC-R5 is never imported or edited."""
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from renderer_validation.transfer_strategies import TransferPlan, full_host, hashes, rss_mib


class TransferContracts(unittest.TestCase):
    def buffers(self):
        return {"rgb": torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 18,
                "depth_m": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
                "normal_world": torch.ones((1, 2, 3, 3)),
                "face_id": torch.tensor([[[-1, 0, 1], [2, 3, 4]]], dtype=torch.int32),
                "instance_id": torch.tensor([[[-1, 0, 0], [1, 1, 1]]], dtype=torch.int32),
                "valid": torch.tensor([[[False, True, True], [True, True, True]]])}

    def test_rss_closes_on_early_return(self):
        stream = io.StringIO("Name: test\nVmRSS: 2048 kB\nTail: value\n")
        with mock.patch.object(Path, "open", return_value=stream):
            self.assertEqual(rss_mib(Path("ignored")), 2.0)
        self.assertTrue(stream.closed)

    def test_rss_closes_when_unavailable(self):
        stream = io.StringIO("Name: test\n")
        with mock.patch.object(Path, "open", return_value=stream):
            self.assertIsNone(rss_mib(Path("ignored")))
        self.assertTrue(stream.closed)

    def test_rss_repeated_calls_without_resourcewarning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            for _ in range(30):
                self.assertGreater(rss_mib(), 0)
        self.assertFalse([w for w in caught if issubclass(w.category, ResourceWarning)])

    def test_rss_closes_on_parse_error(self):
        stream = io.StringIO("VmRSS: invalid kB\n")
        with mock.patch.object(Path, "open", return_value=stream):
            with self.assertRaises(ValueError):
                rss_mib(Path("ignored"))
        self.assertTrue(stream.closed)

    def test_grouped_transfer_preserves_every_dtype_shape_and_hash(self):
        buffers = self.buffers()
        self.assertEqual(hashes(full_host(buffers)), hashes(TransferPlan("A2", buffers).transfer(buffers)))

    def test_grouped_cpu_result_survives_source_mutation(self):
        buffers = self.buffers()
        result = TransferPlan("A2", buffers).transfer(buffers)
        old = hashes(result)
        buffers["rgb"].zero_()
        self.assertEqual(hashes(result), old)

    def test_a0_owns_copy(self):
        buffers = self.buffers()
        result = TransferPlan("A0", buffers).transfer(buffers)
        buffers["rgb"].zero_()
        self.assertGreater(result["rgb"].sum(), 0)

    def test_resident_arm_returns_only_scalars_without_full_cpu_transfer(self):
        buffers = self.buffers()
        with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("full host copy")):
            result = TransferPlan("A1", buffers).transfer(buffers)
        self.assertEqual(set(result), {"valid_pixels", "rgb_sum"})
        self.assertEqual(result["valid_pixels"], 5)
        self.assertIsInstance(result["rgb_sum"], float)

    def test_a3_refuses_cpu_source(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            TransferPlan("A3", self.buffers())

    def test_shape_drift_is_rejected(self):
        buffers = self.buffers()
        plan = TransferPlan("A2", buffers)
        buffers["rgb"] = buffers["rgb"].reshape(-1)
        with self.assertRaisesRegex(ValueError, "contract changed"):
            plan.transfer(buffers)

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            TransferPlan("A4", self.buffers())

    def test_historical_reader_bytes_still_match_recorded_manifest(self):
        import json, hashlib
        record = json.loads((ROOT / "results/renderer_characterization_2026-09-13/R5_performance/source_manifest.json").read_text())
        name = "tools/benchmark_renderer_characterization.py"
        self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), record["source_sha256"][name])


if __name__ == "__main__":
    unittest.main()
