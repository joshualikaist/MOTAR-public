"""CPU tests for the threat-ranked (TTC) obstacle-token selector.

Loaded standalone so the suite never imports the aerial_gym package (which requires isaacgym).
"""

import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest

import torch

_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _ROOT / "aerial_gym/task/navrl_task/navrl_perception.py"

# The module reads its geometry from the environment at import time.
os.environ.setdefault("NAVRL_LIDAR_HBEAMS", "72")
os.environ.setdefault("NAVRL_LIDAR_VBEAMS", "4")
os.environ.setdefault("NAVRL_MAX_OBSTACLES", "8")


def _install_package_stubs():
    """Let navrl_perception import without pulling in the real aerial_gym package.

    navrl_perception does `from aerial_gym.task.navrl_task.navrl_corridor import ...`, and
    aerial_gym/__init__.py imports isaacgym, which refuses to load after torch. The corridor
    module itself is pure torch, so we pre-register empty parent packages and load the real
    corridor file directly into the expected slot.
    """
    for name in ("aerial_gym", "aerial_gym.task", "aerial_gym.task.navrl_task"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # mark as a package so submodule imports resolve
            sys.modules[name] = mod
    key = "aerial_gym.task.navrl_task.navrl_corridor"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            key, _ROOT / "aerial_gym/task/navrl_task/navrl_corridor.py"
        )
        corridor = importlib.util.module_from_spec(spec)
        sys.modules[key] = corridor
        spec.loader.exec_module(corridor)
    search_key = "aerial_gym.task.navrl_task.navrl_search_state"
    if search_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            search_key, _ROOT / "aerial_gym/task/navrl_task/navrl_search_state.py"
        )
        search = importlib.util.module_from_spec(spec)
        sys.modules[search_key] = search
        spec.loader.exec_module(search)


_install_package_stubs()
_SPEC = importlib.util.spec_from_file_location("navrl_perception_ttc_standalone", _MODULE_PATH)
_P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_P)

HBEAMS = _P.HBEAMS
MAX_RANGE = 12.0
# Same convention as the runtime: bearings span (-pi, pi].
BEARINGS = torch.linspace(-math.pi + 2 * math.pi / HBEAMS, math.pi, HBEAMS)


def _blank(batch=1):
    return torch.full((batch, HBEAMS), MAX_RANGE)


def _bearing_index(theta):
    return int(torch.argmin((BEARINGS - theta).abs()).item())


def _put(scan, theta, rng, rays=1):
    """Place a surface on the scan, centred on a bearing.

    ``rays=1`` by default because that is what a real bar looks like at range: a 0.6 m bar at 8 m
    subtends 4.3 degrees, less than the 5-degree beam pitch, so it returns a single ray. Widening
    it artificially would also fragment it, since adjacent endpoints at range d are
    2*d*sin(2.5deg) apart and exceed the 0.45 m cluster gap beyond d ~ 5.2 m -- a property of the
    shared clustering rule, not of the ranking under test.

    Returns the centre index. Assertions identify a chosen surface by its RANGE, not this index:
    the selector legitimately returns whichever ray of a cluster minimises the ranking score.
    """
    i = _bearing_index(theta)
    span = range(-(rays // 2), rays // 2 + 1) if rays > 1 else (0,)
    for off in span:
        scan[0, (i + off) % HBEAMS] = rng
    return i


def _picked_ranges(r, v):
    return sorted(round(float(x), 2) for x in r[0][v[0]])


def _select(scan, vel, **kw):
    return _P.select_ttc_obstacles(
        scan,
        BEARINGS,
        torch.tensor([vel], dtype=torch.float32),
        max_range=MAX_RANGE,
        max_obstacles=kw.pop("max_obstacles", 8),
        cluster_gap_m=kw.pop("cluster_gap_m", 0.45),
        **kw,
    )


class TTCSelectorTests(unittest.TestCase):
    def test_selector_provenance_matches_executed_candidate_set(self):
        cluster = _P.obstacle_selector_provenance("cluster_sector", 240.0)
        self.assertEqual(cluster["effective_fov_deg"], 240.0)
        self.assertFalse(cluster["suppress_active"])

        ttc = _P.obstacle_selector_provenance("ttc_sector", 240.0)
        self.assertEqual(ttc["effective_fov_deg"], 360.0)
        self.assertFalse(ttc["suppress_active"])

        greedy = _P.obstacle_selector_provenance("greedy_suppress", 180.0)
        self.assertEqual(greedy["effective_fov_deg"], 180.0)
        self.assertTrue(greedy["suppress_active"])

    def test_selector_provenance_rejects_unknown_selector(self):
        with self.assertRaises(ValueError):
            _P.obstacle_selector_provenance("invented", 240.0)

    def test_shapes_and_dtypes_match_contract(self):
        scan = _blank()
        _put(scan, 0.0, 5.0)
        r, i, v = _select(scan, (2.0, 0.0))
        self.assertEqual(r.shape, (1, 8))
        self.assertEqual(i.shape, (1, 8))
        self.assertEqual(v.shape, (1, 8))
        self.assertEqual(v.dtype, torch.bool)
        self.assertEqual(i.dtype, torch.long)

    def test_empty_scan_yields_no_valid_tokens(self):
        r, i, v = _select(_blank(), (2.0, 0.0))
        self.assertFalse(bool(v.any()))

    def test_approaching_bar_outranks_closer_receding_bar(self):
        """The core claim: threat, not proximity or bearing, decides the first slot."""
        scan = _blank()
        _put(scan, math.pi, 3.0)   # 3 m BEHIND, drone moving away from it
        _put(scan, 0.0, 8.0)       # 8 m AHEAD, drone closing on it
        r, i, v = _select(scan, (2.0, 0.0))
        self.assertEqual(_picked_ranges(r, v), [3.0, 8.0], "both surfaces fit in 8 slots")
        # Shrink capacity to 1 so only the highest-ranked cluster survives: proximity would keep
        # the 3 m bar behind, threat ranking must keep the 8 m bar ahead.
        r1, i1, v1 = _select(scan, (2.0, 0.0), max_obstacles=1)
        self.assertTrue(bool(v1[0, 0]))
        self.assertEqual(_picked_ranges(r1, v1), [8.0], "closing bar must win the only slot")

    def test_rear_bar_is_selected_when_drone_reverses_into_it(self):
        """The 23.6% failure mode: a bar behind the nose that the velocity points at."""
        scan = _blank()
        _put(scan, math.pi, 4.0)    # rear
        _put(scan, 0.0, 6.0)        # front, deliberately FARTHER so range cannot decide
        # moving backwards in body frame -> the rear bar is the one being approached
        r, i, v = _select(scan, (-2.0, 0.0), max_obstacles=1)
        self.assertTrue(bool(v[0, 0]))
        self.assertEqual(_picked_ranges(r, v), [4.0], "rear bar being reversed into wins")

    def test_two_clusters_in_one_sector_can_both_be_tokenized(self):
        """The 11.7% failure mode: cluster_sector reserves only one cluster per sector."""
        scan = _blank()
        _put(scan, math.radians(4.0), 5.0)
        _put(scan, math.radians(16.0), 6.0)
        r, i, v = _select(scan, (2.0, 0.0), max_obstacles=2)
        self.assertEqual(int(v[0].sum()), 2)
        self.assertEqual(_picked_ranges(r, v), [5.0, 6.0],
                         "two clusters inside one 30-degree sector both get tokens")

    def test_one_wide_surface_does_not_consume_several_slots(self):
        scan = _blank()
        i0 = _bearing_index(0.0)
        for off in range(-6, 7):  # one contiguous wall
            scan[0, (i0 + off) % HBEAMS] = 4.0
        r, i, v = _select(scan, (2.0, 0.0))
        self.assertEqual(int(v[0].sum()), 1, "a single cluster must occupy exactly one slot")

    def test_below_min_speed_ranking_degrades_to_nearest_first(self):
        scan = _blank()
        _put(scan, math.pi, 2.0)   # nearest, but behind
        _put(scan, 0.0, 9.0)
        r, i, v = _select(scan, (0.01, 0.0), max_obstacles=1)
        self.assertEqual(_picked_ranges(r, v), [2.0], "at a standstill, fall back to proximity")

    def test_ranges_are_sorted_nearest_first(self):
        scan = _blank()
        for th, rng in ((0.0, 7.0), (math.radians(50), 3.0), (math.radians(-80), 5.0)):
            _put(scan, th, rng)
        r, i, v = _select(scan, (1.5, 0.5))
        valid = r[0][v[0]]
        self.assertTrue(torch.all(valid[1:] >= valid[:-1] - 1e-6))

    def test_batch_rows_are_independent(self):
        scan = torch.full((2, HBEAMS), MAX_RANGE)
        i0 = _bearing_index(0.0)
        scan[0, i0] = 4.0
        vel = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        r, i, v = _P.select_ttc_obstacles(
            scan, BEARINGS, vel, max_range=MAX_RANGE, max_obstacles=8, cluster_gap_m=0.45
        )
        self.assertEqual(int(v[0].sum()), 1)
        self.assertEqual(int(v[1].sum()), 0)

    def test_deterministic(self):
        scan = _blank()
        _put(scan, 0.3, 5.0)
        _put(scan, -1.2, 4.0)
        a = _select(scan, (2.0, 0.3))
        b = _select(scan, (2.0, 0.3))
        for x, y in zip(a, b):
            self.assertTrue(bool(torch.equal(x, y)))

    def test_rejects_malformed_inputs(self):
        scan = _blank()
        with self.assertRaises(ValueError):
            _P.select_ttc_obstacles(
                scan, BEARINGS, torch.zeros(1, 3),
                max_range=MAX_RANGE, max_obstacles=8, cluster_gap_m=0.45,
            )
        with self.assertRaises(ValueError):
            _P.select_ttc_obstacles(
                scan, BEARINGS, torch.zeros(1, 2),
                max_range=MAX_RANGE, max_obstacles=0, cluster_gap_m=0.45,
            )
        with self.assertRaises(ValueError):
            _P.select_ttc_obstacles(
                scan, BEARINGS, torch.zeros(1, 2),
                max_range=MAX_RANGE, max_obstacles=8, cluster_gap_m=0.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
