"""Unit tests for the appearance domain-shift knobs (검증 2, WORKLOG 2026-08-12).

CPU-only: loads navrl_detector by file path with `warp` and the aerial_gym math import stubbed,
so the pure-torch appearance helpers and the source-level invariants are testable without a GPU.
The end-to-end render (warp kernels) is exercised separately by the GPU smoke.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_detector_appearance.py
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import torch

# ---- stub warp: @wp.kernel decorators and annotation types evaluate at import time ----
_warp = types.ModuleType("warp")


def _passthrough(fn=None, **_kwargs):
    if fn is None:
        return lambda g: g
    return fn


def _any_callable(*_args, **_kwargs):
    return None


_warp.kernel = _passthrough
class _WarpStub(types.ModuleType):
    def __getattr__(self, name):
        if name == "kernel":
            return _passthrough
        return _any_callable


_warp.__class__ = _WarpStub
sys.modules.setdefault("warp", _warp)

# ---- stub the aerial_gym math import (only quat_rotate/quat_mul are pulled) ----
for _pkg in ("aerial_gym", "aerial_gym.utils"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_math_stub = types.ModuleType("aerial_gym.utils.math")
_math_stub.quat_rotate = _any_callable
_math_stub.quat_mul = _any_callable
sys.modules.setdefault("aerial_gym.utils.math", _math_stub)

_PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_detector.py"
_SPEC = importlib.util.spec_from_file_location("navrl_detector_standalone", _PATH)
_DET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DET)

# The stub above was needed only while navrl_detector was being executed: its @wp.kernel
# decorators and annotations evaluate at import. Leaving it in sys.modules made every LATER test
# module in a full-suite run import a fake warp whose Mesh returns None, which is how the
# dynamic-mesh raycast gates came to error with AttributeError on a None mesh. Put the name back.
if sys.modules.get("warp") is _warp:
    del sys.modules["warp"]
SOURCE = _PATH.read_text(encoding="utf-8")


class HueRotation(unittest.TestCase):
    RED = torch.tensor([[0.88, 0.08, 0.045]])

    def test_zero_angle_is_identity(self):
        out = _DET._rotate_hue_rgb(self.RED, torch.zeros(1))
        self.assertTrue(torch.allclose(out, self.RED, atol=1e-6))

    def test_120_degrees_permutes_channels(self):
        """Rotating hue by 120° about the grey axis cycles R->G->B for any colour."""
        out = _DET._rotate_hue_rgb(self.RED, torch.full((1,), math.radians(120.0)))
        expected = self.RED[:, [2, 0, 1]]  # value moves to the next channel
        self.assertTrue(torch.allclose(out, expected, atol=1e-5), (out, expected))

    def test_luminance_neutral_in_gamut(self):
        """Hue rotation preserves brightness wherever the result stays inside the sRGB cube.

        The saturated nominal red leaves the cube under large rotations and the clamp then
        (correctly) clips it, so neutrality is asserted on an in-gamut colour instead.
        """
        angles = torch.tensor([0.4, -1.1, 2.0])
        rgb = torch.tensor([[0.55, 0.40, 0.35]]).expand(3, 3)
        out = _DET._rotate_hue_rgb(rgb, angles)
        self.assertTrue(bool((out > 0).all() and (out < 1).all()))  # clamp was a no-op
        self.assertTrue(torch.allclose(out.sum(dim=1), rgb.sum(dim=1), atol=1e-5))

    def test_per_row_angles_are_independent(self):
        angles = torch.tensor([0.0, math.radians(120.0)])
        out = _DET._rotate_hue_rgb(self.RED.expand(2, 3), angles)
        self.assertTrue(torch.allclose(out[0], self.RED[0], atol=1e-6))
        self.assertFalse(torch.allclose(out[1], self.RED[0], atol=1e-3))


class SmallRandomQuat(unittest.TestCase):
    def test_unit_norm_and_angle_bound(self):
        torch.manual_seed(0)
        max_angle = math.radians(5.0)
        q = _DET._small_random_quat(max_angle, 512, "cpu")
        self.assertTrue(torch.allclose(q.norm(dim=1), torch.ones(512), atol=1e-5))
        angles = 2.0 * torch.acos(q[:, 3].clamp(-1.0, 1.0).abs())
        self.assertLessEqual(float(angles.max()), max_angle + 1e-5)

    def test_zero_angle_is_identity(self):
        q = _DET._small_random_quat(0.0, 8, "cpu")
        self.assertTrue(torch.allclose(q[:, 3].abs(), torch.ones(8), atol=1e-6))


class SourceInvariants(unittest.TestCase):
    """Source-level guards for what the GPU smoke cannot cheaply prove."""

    def test_nominal_literals_live_in_the_buffers(self):
        # Zero knobs must reproduce the historical paint exactly; that only holds if the
        # nominal literals moved into the buffers rather than being re-derived.
        for token in ('[0.88, 0.08, 0.045]', '[0.92, 1.00, 1.05]', '0.08', '0.42'):
            self.assertIn(token, SOURCE)
        # The paint path must consume buffers, not literals.
        paint = SOURCE[SOURCE.index("def render_raw_rgbd"):]
        self.assertIn("self.albedo_base + self.albedo_gain * proximity", paint)
        self.assertIn("self.target_color.view(-1, 3, 1, 1)", paint)
        self.assertNotIn("torch.tensor(\n            [0.88", paint)

    def test_light_gain_applies_after_the_target_paint(self):
        paint = SOURCE[SOURCE.index("def render_raw_rgbd"):]
        self.assertLess(
            paint.index("visible_target_pixels.unsqueeze(1)"),
            paint.index("rgb = rgb * self.light_gain"),
        )

    def test_reset_resamples_and_invalidates_blur(self):
        reset = SOURCE[SOURCE.index("def reset_idx"):SOURCE.index("def _render(self")]
        self.assertIn("self._resample_appearance(env_ids)", reset)
        sampler = SOURCE[SOURCE.index("def _resample_appearance"):SOURCE.index("def reset_idx")]
        self.assertIn("self._blur_valid[env_ids] = False", sampler)

    def test_depth_is_never_blurred_or_tinted(self):
        paint = SOURCE[SOURCE.index("def render_raw_rgbd"):]
        depth_lines = [l for l in paint.splitlines() if "depth" in l and "=" in l]
        for line in depth_lines:
            self.assertNotIn("blur", line.lower())
            self.assertNotIn("light_gain", line)
            self.assertNotIn("tint", line)

    def test_fov_error_perturbs_only_the_ray_table(self):
        init = SOURCE[SOURCE.index("def __init__"):SOURCE.index("def _resample_appearance")]
        # rays are built from the perturbed model...
        self.assertIn("-(uu - self.cx) / render_fx", init)
        # ...while the nominal fx/fy consumers keep the unperturbed values.
        self.assertIn("self.fx = self.width / (2.0 * math.tan(self.half_hfov))", init)
        self.assertNotIn("self.fx = render_fx", init)

    def test_mount_error_is_renderer_only(self):
        render = SOURCE[SOURCE.index("def _render(self"):SOURCE.index("def render_raw_rgbd")]
        self.assertIn("offset = offset + self.mount_trans", render)
        self.assertIn("quat_mul(vehicle_quat, self.mount_quat)", render)
        # perception's copy must stay untouched by this module
        perception = (Path(__file__).parents[1] /
                      "aerial_gym/task/navrl_task/navrl_perception.py").read_text()
        self.assertNotIn("mount_quat", perception)
        self.assertNotIn("mount_trans", perception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
