import importlib.util
from pathlib import Path
import unittest

import torch


_MODULE_PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/bar_probe.py"
_SPEC = importlib.util.spec_from_file_location("navrl_bar_probe_standalone", _MODULE_PATH)
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


class NavRLBarProbeTest(unittest.TestCase):
    def test_same_bearing_different_range_does_not_false_match(self):
        # The old +/-15-degree matcher picked slot 0 because both tokens have exactly the hit
        # bearing.  Slot 0 belongs to a different, nearer bar; slot 1 is the struck bar's surface.
        tokens = torch.tensor([[[1.0, 0.0], [2.65, 0.05]]])
        valid = torch.tensor([[True, True]])
        bars = torch.tensor([[[1.25, 0.0], [3.0, 0.0]]])

        result = _PROBE.associate_surface_tokens_to_bars(tokens, valid, bars)

        self.assertEqual(result["bar_index"].tolist(), [[0, 1]])
        self.assertTrue(result["associated"].all())

    def test_unrelated_token_on_same_ray_is_rejected(self):
        tokens = torch.tensor([[[1.0, 0.0]]])
        valid = torch.tensor([[True]])
        bars = torch.tensor([[[3.0, 0.0]]])

        result = _PROBE.associate_surface_tokens_to_bars(tokens, valid, bars)

        self.assertFalse(bool(result["associated"][0, 0]))

    def test_surface_offset_is_split_into_lateral_and_radial_components(self):
        tokens = torch.tensor([[[2.6, 0.0]]])
        valid = torch.tensor([[True]])
        bars = torch.tensor([[[3.0, 0.2]]])

        result = _PROBE.associate_surface_tokens_to_bars(tokens, valid, bars)

        self.assertTrue(bool(result["associated"][0, 0]))
        self.assertAlmostEqual(float(result["cross_track"][0, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(result["radial_gap"][0, 0]), 0.4, places=5)
        self.assertAlmostEqual(
            float(result["center_offset"][0, 0]), (0.2**2 + 0.4**2) ** 0.5, places=5
        )

    def test_invalid_token_is_never_associated(self):
        result = _PROBE.associate_surface_tokens_to_bars(
            torch.tensor([[[2.6, 0.0]]]),
            torch.tensor([[False]]),
            torch.tensor([[[3.0, 0.0]]]),
        )
        self.assertFalse(bool(result["associated"][0, 0]))


if __name__ == "__main__":
    unittest.main()
