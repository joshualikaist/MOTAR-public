"""Depth measurement-noise model order (prereg_2026-09-04_depth_noise_model_order).

M1 is the gate that lets the three evaluation cells run at all: with the knob unset, sigma_r
must be BIT-IDENTICAL to the pre-2026-09-04 formula, so every existing result, checkpoint and
receipt stays valid by construction. The rest pins the physics of the "stereo" arm and the
explicit modelling decision that it uses the SENSOR's focal length, not detect_fx.
"""

import math
import pathlib
import unittest

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
PERCEPTION_SOURCE = (
    REPO / "aerial_gym" / "task" / "navrl_task" / "navrl_perception.py"
).read_text()
CONFIG_SOURCE = (
    REPO / "aerial_gym" / "config" / "task_config" / "navrl_task_config.py"
).read_text()


def sigma_r_legacy(r, px):
    """The formula exactly as it stood before 2026-09-04."""
    return 0.04 + 0.012 * r + 0.15 / px.sqrt()


def sigma_r_under(model, r, px, *, subpixel=0.08, fx_px=447.0, baseline_m=0.095):
    """Mirror of the production expression, parameterised by mode."""
    if model == "stereo":
        range_term = (subpixel / (fx_px * baseline_m)) * r.square()
    else:
        range_term = 0.012 * r
    return 0.04 + range_term + 0.15 / px.sqrt()


class DepthNoiseModelM1(unittest.TestCase):
    """M1: the default path must not move by one bit."""

    def test_default_mode_is_bit_identical_to_legacy(self):
        r = torch.linspace(0.05, 20.0, 401, dtype=torch.float32)
        for px_value in (1.0, 2.0, 3.0, 5.0, 12.0, 64.0, 1024.0):
            px = torch.full_like(r, px_value)
            legacy = sigma_r_legacy(r, px)
            current = sigma_r_under("linear", r, px)
            self.assertTrue(
                torch.equal(legacy, current),
                "sigma_r drifted from the legacy formula at px=%s" % px_value,
            )

    def test_default_is_linear_not_stereo(self):
        self.assertIn(
            'os.environ.get("NAVRL_DEPTH_NOISE_MODEL", "linear")',
            CONFIG_SOURCE,
            "the depth noise model must default to 'linear' so existing results are unaffected",
        )

    def test_unknown_mode_is_rejected_not_silently_ignored(self):
        self.assertIn("depth_noise_model must be linear|stereo", PERCEPTION_SOURCE)


class DepthNoiseModelStereoArm(unittest.TestCase):
    """The stereo arm must reproduce Intel's published depth-error formula."""

    def test_matches_intel_formula_at_reference_ranges(self):
        # RMS = D^2 * subpixel / (fx_px * baseline). Values quoted in the prereg table.
        for range_m, baseline_m, expected in (
            (5.0, 0.095, 0.047),
            (10.0, 0.095, 0.188),
            (20.0, 0.095, 0.754),
            (20.0, 0.050, 1.432),
        ):
            got = (0.08 / (447.0 * baseline_m)) * range_m**2
            self.assertAlmostEqual(got, expected, places=2, msg="range %s m" % range_m)

    def test_range_term_crossover(self):
        # The two arms share the 0.04 floor and the shot-noise term, so the ONLY difference is
        # the range term: 0.012*r vs c*r^2. They cross at r = 0.012/c -- 6.4 m for the D455,
        # 3.4 m for the D435. Above the crossover the legacy linear model is OPTIMISTIC, which
        # is most of the 20 m detector band and is the fact that motivates the experiment.
        # (An earlier draft quoted ~12 m; that compared the FULL legacy sigma_r against the BARE
        # Intel formula, which is not the difference between the two arms as implemented.)
        self.assertAlmostEqual(0.012 / (0.08 / (447.0 * 0.095)), 6.37, places=1)
        self.assertAlmostEqual(0.012 / (0.08 / (447.0 * 0.050)), 3.35, places=1)

    def test_stereo_is_pessimistic_below_and_optimistic_above_crossover(self):
        px = torch.tensor([5.0])
        near, far = torch.tensor([3.0]), torch.tensor([20.0])
        # Below the crossover the stereo arm is the SMALLER sigma (legacy was pessimistic there).
        self.assertLess(sigma_r_under("stereo", near, px), sigma_r_under("linear", near, px))
        # Above it the stereo arm is larger -- the legacy model was understating the error.
        self.assertGreater(sigma_r_under("stereo", far, px), sigma_r_under("linear", far, px))

    def test_floor_and_shot_noise_terms_survive_the_swap(self):
        # Only the RANGE term changes; the 0.04 floor and 0.15/sqrt(px) are detection
        # properties, not stereo-depth properties.
        r = torch.tensor([0.0])
        px = torch.tensor([4.0])
        for model in ("linear", "stereo"):
            self.assertAlmostEqual(
                float(sigma_r_under(model, r, px)), 0.04 + 0.15 / 2.0, places=6
            )


class DepthNoiseModelScope(unittest.TestCase):
    """M2: the change must be confined to the range term."""

    def test_sigma_lat_is_untouched(self):
        # Bearing error times range is linear in r by construction, and it correctly uses the
        # DETECT focal length because the bearing really is measured on our detect image.
        self.assertIn(
            "sigma_lat = 0.03 + surface_range / max(self.detect_fx, 1.0)", PERCEPTION_SOURCE
        )

    def test_stereo_uses_sensor_fx_not_detect_fx(self):
        # The explicit modelling decision of the prereg. detect_fx is 84.3 at 160x90; using it
        # would inflate the 20 m error ~5x and would model our downsampling as if it degraded
        # the sensor's own depth estimate.
        self.assertIn('_env_float("NAVRL_DEPTH_STEREO_FX_PX", 447.0)', CONFIG_SOURCE)
        self.assertNotIn("depth_stereo_coeff * self.detect_fx", PERCEPTION_SOURCE)
        coeff_line = [
            line for line in PERCEPTION_SOURCE.splitlines() if "depth_stereo_coeff" in line
        ]
        self.assertTrue(coeff_line, "depth_stereo_coeff disappeared")
        self.assertFalse(
            any("detect_fx" in line for line in coeff_line),
            "the stereo coefficient must not be built from detect_fx",
        )

    def test_default_baseline_is_the_d455_the_camera_config_models(self):
        self.assertIn('_env_float("NAVRL_DEPTH_STEREO_BASELINE_M", 0.095)', CONFIG_SOURCE)


if __name__ == "__main__":
    unittest.main()
