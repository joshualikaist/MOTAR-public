"""Star-convex free-space region (A3). CPU-only, no Isaac.

These pin the two properties the contact forensics says we need: no bearing is exempt, and an
unreturned ray bounds the region instead of opening it.
"""

import math
import pathlib
import unittest

import torch

# Loaded by path, not by package: aerial_gym/__init__.py pulls in isaacgym, which refuses to be
# imported after torch. The module under test is deliberately Isaac-free so this works.
import importlib.util

_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "aerial_gym" / "task" / "navrl_task" / "star_convex.py"
)
_spec = importlib.util.spec_from_file_location("navrl_star_convex", _SRC)
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)
direction_clearance = _sc.direction_clearance
flip_points = _sc.flip_points
point_in_star_region = _sc.point_in_star_region
scan_to_points = _sc.scan_to_points

RNG = 12.0
CONE = math.radians(5.0)


def _scan(fill, v=4, h=72):
    return torch.full((1, v, h), float(fill))


def _geom(v=4, h=72):
    bearings = torch.linspace(math.pi, -math.pi + 2 * math.pi / h, h)
    elev = torch.linspace(math.radians(20.0), math.radians(-10.0), v)
    return bearings, elev


class BallFlip(unittest.TestCase):
    def test_points_on_the_sphere_are_fixed(self):
        p = torch.tensor([[[RNG, 0.0, 0.0], [0.0, -RNG, 0.0]]])
        self.assertTrue(torch.allclose(flip_points(p, RNG), p, atol=1e-5))

    def test_flip_reverses_radial_order(self):
        """|x_hat| = 2r - |x|, so the nearest return ends up farthest out and lands on the hull."""
        near = torch.tensor([[[1.0, 0.0, 0.0]]])
        far = torch.tensor([[[11.0, 0.0, 0.0]]])
        fn, ff = flip_points(near, RNG).norm(), flip_points(far, RNG).norm()
        self.assertAlmostEqual(float(fn), 2 * RNG - 1.0, places=4)
        self.assertAlmostEqual(float(ff), 2 * RNG - 11.0, places=4)
        self.assertGreater(float(fn), float(ff), "nearer input must flip farther out")

    def test_flip_preserves_bearing(self):
        p = torch.tensor([[[3.0, 4.0, 0.0]]])
        f = flip_points(p, RNG)
        self.assertAlmostEqual(
            float(torch.atan2(p[..., 1], p[..., 0])), float(torch.atan2(f[..., 1], f[..., 0])), places=5
        )


class UnknownBoundsTheRegion(unittest.TestCase):
    """The defect this whole module exists to fix."""

    def test_no_return_ray_becomes_a_wall_at_max_range(self):
        bearings, elev = _geom()
        pts, valid = scan_to_points(_scan(float("inf")), bearings, elev, RNG)
        self.assertFalse(bool(valid.any()), "an all-infinite scan must have no valid returns")
        self.assertTrue(
            torch.allclose(pts.norm(dim=-1), torch.full_like(pts.norm(dim=-1), RNG), atol=1e-4),
            "unreturned rays must sit on the sensing horizon, not vanish",
        )

    def test_unknown_space_is_bounded_not_free(self):
        bearings, elev = _geom()
        pts, _ = scan_to_points(_scan(float("inf")), bearings, elev, RNG)
        d = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(float(direction_clearance(pts, d, RNG, CONE)), RNG, places=3)
        far = torch.tensor([[RNG + 1.0, 0.0, 0.0]])
        self.assertFalse(bool(point_in_star_region(pts, far, RNG, CONE)))


class EveryBearingParticipates(unittest.TestCase):
    """The 57-58% LATERAL finding: a lateral obstacle must bound its own bearing."""

    def test_lateral_obstacle_bounds_its_own_bearing_not_the_forward_one(self):
        bearings, elev = _geom()
        ranges = _scan(RNG)
        lateral_bin = int(torch.argmin((bearings - (math.pi / 2)).abs()))
        ranges[0, :, lateral_bin] = 2.0
        pts, _ = scan_to_points(ranges, bearings, elev, RNG)
        fwd = direction_clearance(pts, torch.tensor([[1.0, 0.0]]), RNG, CONE)
        lat = direction_clearance(pts, torch.tensor([[0.0, 1.0]]), RNG, CONE)
        self.assertAlmostEqual(float(fwd), RNG, places=2)
        self.assertLess(float(lat), 2.6, "the lateral bearing must see its own obstacle")

    def test_widening_the_cone_degenerates_to_omnidirectional(self):
        """A4's `omni` baseline falls out of the same operator."""
        bearings, elev = _geom()
        ranges = _scan(RNG)
        ranges[0, :, int(torch.argmin((bearings - (math.pi / 2)).abs()))] = 2.0
        pts, _ = scan_to_points(ranges, bearings, elev, RNG)
        d = torch.tensor([[1.0, 0.0]])
        narrow = float(direction_clearance(pts, d, RNG, CONE))
        wide = float(direction_clearance(pts, d, RNG, math.pi))
        self.assertAlmostEqual(narrow, RNG, places=2)
        self.assertLess(wide, 2.6, "a full-sphere cone is the omnidirectional nearest return")


class Batching(unittest.TestCase):
    def test_batch_independence(self):
        bearings, elev = _geom()
        a = _scan(RNG); a[0, :, 0] = 1.0
        b = _scan(RNG)
        both = torch.cat([a, b], dim=0)
        pts, _ = scan_to_points(both, bearings, elev, RNG)
        d = torch.tensor([[math.cos(math.pi), math.sin(math.pi)], [math.cos(math.pi), math.sin(math.pi)]])
        c = direction_clearance(pts, d, RNG, CONE)
        self.assertLess(float(c[0]), 1.6)
        self.assertAlmostEqual(float(c[1]), RNG, places=2)

    def test_inputs_are_not_mutated(self):
        bearings, elev = _geom()
        ranges = _scan(5.0)
        before = ranges.clone()
        scan_to_points(ranges, bearings, elev, RNG)
        self.assertTrue(torch.equal(ranges, before))


if __name__ == "__main__":
    unittest.main()
