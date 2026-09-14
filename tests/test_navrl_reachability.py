"""CPU regression tests for the verification-4 static reachability oracle."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/analyze_navrl_v2_reachability.py"
SPEC = importlib.util.spec_from_file_location("navrl_reachability", SCRIPT)
REACH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REACH)


class ReachabilityCoordinateContractTest(unittest.TestCase):
    def test_arena_uses_dump_coordinate_frame_zero_to_forty(self):
        self.assertEqual(REACH.ARENA_MIN_M, 0.0)
        self.assertEqual(REACH.ARENA_MAX_M, 40.0)

    def test_empty_scene_connects_points_in_upper_half_of_arena(self):
        bars = np.array([[0.0, 0.0]], dtype=np.float64)
        self.assertTrue(
            REACH.episode_connected(bars, np.array([25.0, 30.0]), np.array([35.0, 30.0]), 0.4)
        )

    def test_full_height_wall_separates_spawn_and_goal(self):
        # A dense x=20 wall spans the real 0..40 m y extent. The legacy -20..20 oracle clipped
        # both endpoints and could not faithfully test this scene.
        ys = np.arange(0.0, 40.0001, 0.20)
        bars = np.column_stack([np.full_like(ys, 20.0), ys])
        self.assertFalse(
            REACH.episode_connected(bars, np.array([5.0, 30.0]), np.array([35.0, 30.0]), 0.4)
        )


if __name__ == "__main__":
    unittest.main()
