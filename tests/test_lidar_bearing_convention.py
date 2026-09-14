"""Pin the LiDAR bin->bearing convention and the camera far-plane fusion remap.

The mirror bug this guards against: navrl_perception used to assume bin bearings INCREASE with the
bin index while the warp ray generator emits them DECREASING (warp_lidar.py:67). The mirror image
put every obstacle token on the wrong side of the drone. Physically adjudicated on 2026-07-29 with
tools/probe_lidar_bearing.py (increasing table: 13.9% of returns land on a GT bar; decreasing:
94.8%). These tests recompute the warp formula independently so EITHER side changing breaks them
loudly instead of silently reintroducing the mirror.

CPU-only: no Isaac Gym, no warp import needed.
"""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch

# Load navrl_perception standalone (same pattern as the other perception tests): importing the
# aerial_gym package would pull in isaacgym, which refuses to load after torch.
_TASK_DIR = Path(__file__).parents[1] / "aerial_gym/task/navrl_task"
for _package in ("aerial_gym", "aerial_gym.task", "aerial_gym.task.navrl_task"):
    sys.modules.setdefault(_package, types.ModuleType(_package))
_CORRIDOR_NAME = "aerial_gym.task.navrl_task.navrl_corridor"
_CORRIDOR_SPEC = importlib.util.spec_from_file_location(
    _CORRIDOR_NAME, _TASK_DIR / "navrl_corridor.py"
)
_CORRIDOR = importlib.util.module_from_spec(_CORRIDOR_SPEC)
sys.modules[_CORRIDOR_NAME] = _CORRIDOR
_CORRIDOR_SPEC.loader.exec_module(_CORRIDOR)
_SEARCH_NAME = "aerial_gym.task.navrl_task.navrl_search_state"
_SEARCH_SPEC = importlib.util.spec_from_file_location(
    _SEARCH_NAME, _TASK_DIR / "navrl_search_state.py"
)
_SEARCH = importlib.util.module_from_spec(_SEARCH_SPEC)
sys.modules[_SEARCH_NAME] = _SEARCH
_SEARCH_SPEC.loader.exec_module(_SEARCH)

_MODULE_PATH = _TASK_DIR / "navrl_perception.py"
_SPEC = importlib.util.spec_from_file_location("navrl_perception_bearing_test", _MODULE_PATH)
_PERCEPTION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PERCEPTION)
HBEAMS = _PERCEPTION.HBEAMS
VBEAMS = _PERCEPTION.VBEAMS
camera_no_return_to_lidar_range = _PERCEPTION.camera_no_return_to_lidar_range
lidar_bin_bearings = _PERCEPTION.lidar_bin_bearings


def warp_reference_azimuths(width):
    """Recompute the ray azimuths exactly as warp_lidar.initialize_ray_vectors does.

    Deliberately duplicated from aerial_gym/sensors/warp/warp_lidar.py (azimuth = hfov_max -
    span * j/(W-1)) with the navrl_lidar_config span (hfov_max=180, hfov_min=-180+360/W): the
    duplication IS the tripwire -- if the sensor formula or the config span ever changes, this
    test fails and forces the perception table to be re-verified rather than silently diverging.
    """
    hfov_max = math.radians(180.0)
    hfov_min = math.radians(-180.0 + 360.0 / width)
    return [
        hfov_max - (hfov_max - hfov_min) * (j / (width - 1)) for j in range(width)
    ]


class TestBearingConvention(unittest.TestCase):
    def test_table_matches_warp_ray_generator(self):
        table = lidar_bin_bearings()
        ref = warp_reference_azimuths(HBEAMS)
        self.assertEqual(table.numel(), HBEAMS)
        for j in (0, 1, HBEAMS // 4, HBEAMS // 2, HBEAMS - 2, HBEAMS - 1):
            self.assertAlmostEqual(
                float(table[j]), ref[j], places=5,
                msg=f"bin {j}: perception table diverges from the warp ray generator",
            )

    def test_convention_is_decreasing_not_mirrored(self):
        """The historical bug in one assertion: bearings must DECREASE with the bin index."""
        table = lidar_bin_bearings()
        self.assertGreater(float(table[0]), float(table[1]))
        self.assertAlmostEqual(float(table[0]), math.pi, places=5)
        self.assertAlmostEqual(
            float(table[-1]), -math.pi + 2.0 * math.pi / HBEAMS, places=5
        )
        # The mirrored (increasing) table satisfies assumed = bin - true; make sure we are not it.
        mirrored_first = -math.pi + 2.0 * math.pi / HBEAMS
        self.assertNotAlmostEqual(float(table[0]), mirrored_first, places=3)

    def test_bin_spacing_uniform(self):
        table = lidar_bin_bearings()
        diffs = table[:-1] - table[1:]
        expected = 2.0 * math.pi / HBEAMS
        self.assertTrue(
            torch.allclose(diffs, torch.full_like(diffs, expected), atol=1e-5),
            "bin spacing must be uniform 360/HBEAMS",
        )


class TestCameraFarPlaneRemap(unittest.TestCase):
    def test_far_plane_becomes_no_return(self):
        cam_max, lidar_max = 10.0, 12.0
        cols = torch.tensor([9.5, 10.0, 9.999, 3.2, 10.0])
        out = camera_no_return_to_lidar_range(cols, cam_max, lidar_max)
        self.assertAlmostEqual(float(out[0]), 9.5)      # real return preserved
        self.assertAlmostEqual(float(out[1]), 12.0)     # far plane -> lidar no-return
        self.assertAlmostEqual(float(out[2]), 12.0)     # within epsilon of far plane
        self.assertAlmostEqual(float(out[3]), 3.2)      # near obstacle preserved
        self.assertAlmostEqual(float(out[4]), 12.0)

    def test_legacy_short_lidar_unaffected(self):
        """With lidar range <= camera range (the old 4/8 m recipes), fusion stays a no-op."""
        cam_max, lidar_max = 10.0, 8.0
        cols = torch.tensor([10.0, 7.5])
        out = camera_no_return_to_lidar_range(cols, cam_max, lidar_max)
        # 10.0 (no return) maps to 8.0; min(scan<=8, 8.0) never alters the scan.
        self.assertAlmostEqual(float(out[0]), 8.0)
        self.assertAlmostEqual(float(out[1]), 7.5)

    def test_vbeams_positive(self):
        self.assertGreaterEqual(VBEAMS, 1)


if __name__ == "__main__":
    unittest.main()
