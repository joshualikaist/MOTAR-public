"""Unit tests for the 4-1 synthetic detector-noise injection (WORKLOG 2026-08-13).

Preregistration: docs/prereg_2026-08-13_detector_coupling.md

CPU-only: exercises the pure-torch helpers and the source-level invariants without a GPU or Isaac
Gym, the same way tests/test_navrl_detector_appearance.py does.

The properties that matter, and why:

  * OFF is a bit-exact no-op. Every clean number in the archive was produced without these knobs;
    if enabling the code path perturbs anything at zero magnitude, the arms are not comparable.
  * The noise draws from a DEDICATED generator. Verification 3 lost a campaign because pose noise
    consumed the global torch RNG and silently changed obstacle placement between arms.
  * Dropout is Markov, not iid, and its stationary rate and run-length match the closed form. The
    whole point of the experiment is reproducing v7's temporal miss structure; an iid chain with
    the right marginal would quietly fail that and make a null result uninterpretable.
  * Bearing noise is injected through the pixel centroid, not by writing `bearing` afterwards --
    otherwise the Kalman filter sees a clean position while only the map carve-out sees the
    perturbed angle.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_detector_noise.py
"""

import math
from pathlib import Path
import re
import sys
import types
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = (REPO / "aerial_gym/task/navrl_task/navrl_perception.py").read_text(encoding="utf-8")


class _Stub:
    """Minimal stand-in exposing only what the two helpers touch."""

    def __init__(self, num_envs=4096, device="cpu", seed=9409,
                 bearing_std=0.0, range_std=0.0, p01=0.0, p10=1.0):
        self.num_envs = num_envs
        self.device = device
        self.detector_noise_bearing_std_rad = bearing_std
        self.detector_noise_range_std_m = range_std
        self.detector_noise_dropout_p01 = p01
        self.detector_noise_dropout_p10 = p10
        self._detector_noise_generator = torch.Generator(device=device)
        self._detector_noise_generator.manual_seed(seed)
        self._detector_noise_missing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # bound from the real class below
    _detector_noise_visibility = None


def _bind_visibility():
    """Extract the real _detector_noise_visibility and bind it to the stub, so the test exercises
    shipped code rather than a paraphrase of it."""
    start = SOURCE.index("    def _detector_noise_visibility(self, visible):")
    end = SOURCE.index("    def _dump_detector_profile(self)")
    src = SOURCE[start:end]
    ns = {"torch": torch}
    exec("class _Host:\n" + src, ns)  # noqa: S102 - executing our own repo source, by design
    return ns["_Host"]._detector_noise_visibility


VISIBILITY = _bind_visibility()
_Stub._detector_noise_visibility = VISIBILITY


class TestMarkovDropout(unittest.TestCase):

    def test_zero_p01_never_hides_anything(self):
        s = _Stub(p01=0.0, p10=1.0)
        visible = torch.ones(s.num_envs, dtype=torch.bool)
        for _ in range(50):
            out = s._detector_noise_visibility(visible)
            self.assertTrue(bool(out.all()), "p01=0 must be a no-op")

    def test_stationary_miss_rate_matches_closed_form(self):
        """Stationary miss rate of a two-state chain is p01 / (p01 + p10)."""
        p01, p10 = 0.10, 0.40
        s = _Stub(num_envs=20000, p01=p01, p10=p10)
        visible = torch.ones(s.num_envs, dtype=torch.bool)
        misses = 0.0
        steps = 400
        for _ in range(steps):  # burn-in included; the chain mixes in a few steps at these rates
            out = s._detector_noise_visibility(visible)
            misses += float((~out).float().mean())
        observed = misses / steps
        expected = p01 / (p01 + p10)
        self.assertAlmostEqual(observed, expected, delta=0.01,
                               msg=f"observed {observed:.4f} vs closed form {expected:.4f}")

    def test_mean_run_length_matches_closed_form(self):
        """Mean miss run-length is 1 / p10 -- the property iid Bernoulli gets wrong."""
        p01, p10 = 0.08, 0.25
        s = _Stub(num_envs=4000, p01=p01, p10=p10)
        visible = torch.ones(s.num_envs, dtype=torch.bool)
        trace = torch.stack([~s._detector_noise_visibility(visible) for _ in range(600)])
        runs = []
        for env in range(0, s.num_envs, 40):        # subsample envs; 100 traces is plenty
            col = trace[:, env].tolist()
            run = 0
            for miss in col:
                if miss:
                    run += 1
                elif run:
                    runs.append(run)
                    run = 0
        self.assertGreater(len(runs), 200, "not enough completed runs to estimate the mean")
        observed = sum(runs) / len(runs)
        self.assertAlmostEqual(observed, 1.0 / p10, delta=0.35,
                               msg=f"mean run {observed:.3f} vs closed form {1.0 / p10:.3f}")

    def test_iid_bernoulli_would_have_a_different_run_length(self):
        """Guard the premise: with the same marginal, iid draws give mean run ~1/(1-rate), which
        is materially shorter. If this ever stops holding the Markov model buys nothing."""
        p01, p10 = 0.08, 0.25
        rate = p01 / (p01 + p10)
        markov_mean = 1.0 / p10
        iid_mean = 1.0 / (1.0 - rate)
        self.assertGreater(markov_mean, 1.5 * iid_mean)

    def test_cannot_resurrect_a_detection_that_was_already_missing(self):
        """The chain may only remove detections, never add them."""
        s = _Stub(num_envs=2000, p01=0.5, p10=0.5)
        visible = torch.rand(s.num_envs) > 0.5
        for _ in range(20):
            out = s._detector_noise_visibility(visible)
            self.assertTrue(bool((out <= visible).all()), "noise must not create detections")

    def test_uses_only_its_own_generator(self):
        """The global torch RNG must be untouched -- this is the verification-3 lesson."""
        torch.manual_seed(1234)
        before = torch.rand(3)
        torch.manual_seed(1234)
        s = _Stub(num_envs=512, p01=0.3, p10=0.3)
        visible = torch.ones(s.num_envs, dtype=torch.bool)
        for _ in range(25):
            s._detector_noise_visibility(visible)
        after = torch.rand(3)
        self.assertTrue(torch.equal(before, after), "global RNG stream advanced")

    def test_repeatable_for_a_fixed_seed(self):
        visible = torch.ones(1024, dtype=torch.bool)
        runs = []
        for _ in range(2):
            s = _Stub(num_envs=1024, seed=4242, p01=0.2, p10=0.3)
            runs.append(torch.stack([s._detector_noise_visibility(visible) for _ in range(30)]))
        self.assertTrue(torch.equal(runs[0], runs[1]))


class TestSourceInvariants(unittest.TestCase):
    """Properties that live in _detect_rgbd, checked at the source level so they cannot regress
    without someone noticing."""

    def setUp(self):
        # Whole method, not a fixed-width window: _detect_rgbd grew when the detect-resolution
        # branch landed (WORKLOG 2026-08-22) and a 4000-char slice silently stopped covering the
        # noise-injection block it is supposed to guard -- a test that passes by truncation.
        start = SOURCE.index("    def _detect_rgbd(self")
        end = SOURCE.index("\n    def ", start + 1) + 1
        self.detect = SOURCE[start:end]

    def test_bearing_noise_is_injected_through_the_centroid(self):
        """Perturbing `bearing` after it is derived would desync it from measurement_vehicle: the
        tracker would get the clean position and only the obstacle-map carve-out the perturbed
        angle. The injection must move `u`."""
        # detect_fx, not fx: the centroid `u` is measured on the DETECT-resolution image, whose
        # pixels-per-radian is detect_width/(2 tan(hfov/2)). The two are the same number whenever
        # the resolutions are equal, so this is unchanged for every historical configuration.
        self.assertIn("u = u - self.detect_fx * d_bearing", self.detect)
        noise_block = self.detect[self.detect.index("_detector_noise_active"):]
        self.assertNotIn("bearing = bearing +", noise_block)
        self.assertNotIn("bearing +=", noise_block)

    def test_noise_is_applied_before_measurement_vehicle_is_built(self):
        i_noise = self.detect.index("d_bearing")
        i_meas = self.detect.index("measurement_vehicle = self.camera_offset")
        self.assertLess(i_noise, i_meas)

    def test_every_noise_draw_names_the_dedicated_generator(self):
        """Any bare torch.rand/randn inside the injection block would leak into the global stream."""
        block = self.detect[self.detect.index("if self._detector_noise_active:"):]
        for call in re.finditer(r"torch\.(rand|randn)\(", block):
            tail = block[call.start():call.start() + 400]
            self.assertIn("generator=self._detector_noise_generator", tail,
                          f"undedicated RNG draw at offset {call.start()}")

    def test_profiling_does_not_feed_the_live_path(self):
        """The profile head must observe only; if its outputs reached the tracker or the map the
        profiled arm would no longer be the arm we evaluate."""
        start = SOURCE.index("    def _record_detector_profile(self")
        end = SOURCE.index("    def _detect_rgbd(self")
        body = SOURCE[start:end]
        for forbidden in ("self.tracker", "self.target_history", "self.last_visible",
                          "self.last_target_like"):
            self.assertNotIn(forbidden, body, f"profiling touches {forbidden}")
        # The only writes are to the record list.
        writes = set(re.findall(r"^\s{8}(self\.[A-Za-z_]+)\s*=", body, re.M))
        self.assertEqual(writes, set(), f"profiling assigns to module state: {writes}")

    def test_profiling_result_is_discarded_at_the_call_site(self):
        """Called as a bare statement -- nothing downstream may consume the profile head."""
        call = [ln.strip() for ln in self.detect.splitlines()
                if "_record_detector_profile(" in ln]
        self.assertTrue(call, "call site not found")
        self.assertTrue(call[0].startswith("self._record_detector_profile("),
                        f"return value is used: {call[0]}")

    def test_scale_does_not_touch_p10(self):
        """The dose ladder must change magnitude, not the run-length shape -- otherwise the three
        rungs are three different noise families and the monotonicity check is meaningless."""
        init = SOURCE[SOURCE.index("_dn_scale = float("):SOURCE.index("self.detector_noise_seed =")]
        self.assertIn("detector_noise_dropout_p01", init)
        self.assertIn("* _dn_scale", init)
        p10_line = [ln for ln in init.splitlines() if "dropout_p10" in ln]
        self.assertTrue(p10_line, "p10 assignment not found")
        self.assertNotIn("_dn_scale", p10_line[0])


class TestConfigDefaults(unittest.TestCase):

    def test_all_knobs_default_to_off(self):
        cfg = (REPO / "aerial_gym/config/task_config/navrl_task_config.py").read_text(
            encoding="utf-8")
        for name, default in (
            ("NAVRL_DETNOISE_BEARING_STD_RAD", "0.0"),
            ("NAVRL_DETNOISE_RANGE_STD_M", "0.0"),
            ("NAVRL_DETNOISE_DROPOUT_P01", "0.0"),
        ):
            self.assertIn(f'_env_float("{name}", {default})', cfg)
        self.assertIn('os.environ.get("NAVRL_DETPROFILE_CHECKPOINT", "")', cfg)
        self.assertIn('"NAVRL_DETNOISE_RANGE_BIAS_PROFILE", ""', cfg)

    def test_activation_requires_a_nonzero_magnitude(self):
        gate = SOURCE[SOURCE.index("self._detector_noise_active = ("):]
        gate = gate[:gate.index(")") + 1]
        for knob in ("bearing_std_rad", "range_std_m", "dropout_p01"):
            self.assertIn(knob, gate)


class TestRangeBiasProfile(unittest.TestCase):
    """Source-level guards for the gate-passing bin-wise systematic-bias model."""

    def test_profile_is_selected_from_clean_surface_range(self):
        body = SOURCE[SOURCE.index("    def _detector_noise_range(self, surface_range):"):
                      SOURCE.index("    def _detector_noise_visibility(self, visible):")]
        self.assertIn("torch.bucketize(surface_range, self._detector_noise_bias_edges)", body)
        self.assertIn("bias = self._detector_noise_bias_values[bias_idx]", body)

    def test_bias_values_follow_dose_scale(self):
        init = SOURCE[SOURCE.index("bias_edges, bias_values = [], []"):
                      SOURCE.index("self.detector_noise_seed =")]
        self.assertIn("float(value) * _dn_scale", init)


if __name__ == "__main__":
    unittest.main(verbosity=2)
