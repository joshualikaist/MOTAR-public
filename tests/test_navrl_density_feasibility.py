import importlib.util
from pathlib import Path
import random
import unittest

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/analyze_navrl_density_feasibility.py"
)
_SPEC = importlib.util.spec_from_file_location("density_feasibility", _MODULE_PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


class DensityFeasibilityTest(unittest.TestCase):
    def test_component_flood_fill_does_not_create_singletons(self):
        free = np.ones((4, 5), dtype=bool)
        labels, sizes = _AUDIT._components(free)
        self.assertEqual(sizes.tolist(), [20])
        self.assertTrue(np.all(labels == 0))

    def test_navrl_band_never_leaves_forbidden_center_distance(self):
        sizes = [(0.4, 0.4), (0.8, 0.8)]
        bars, _, _, _ = _AUDIT.place_bars_navrl_band(
            80,
            random.Random(7),
            sizes,
            (0.0, 40.0, 0.0, 40.0),
            touch=0.4,
            gap=1.6,
        )
        for i, (x, y, _, _) in enumerate(bars):
            for px, py, _, _ in bars[:i]:
                distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                self.assertFalse(0.4 < distance < 1.6)

    def test_empty_arena_is_connected(self):
        crossing, largest, paired, free = _AUDIT.topology_metrics(
            [], arena=10.0, resolution=0.5, side_clearance=0.2
        )
        self.assertTrue(crossing)
        self.assertEqual(largest, 1.0)
        self.assertEqual(paired, 1.0)
        self.assertEqual(free, 1.0)


if __name__ == "__main__":
    unittest.main()
