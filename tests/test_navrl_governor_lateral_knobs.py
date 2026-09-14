"""Plan I3: the three additive governor knobs (L2 width, L3 lateral channel, L5 yaw cap).

Invariants: (1) all off at defaults -> every earlier arm is bit-identical; (2) each knob is a
magnitude-only change (xy direction and yaw sign never move); (3) the lateral channel has a
floor so it cannot deadlock; (4) the shell and the Python side agree on the six knobs.
"""

import importlib.util
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "speed_governor_knobs_test", ROOT / "aerial_gym/task/navrl_task/speed_governor.py")
SG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SG)
SHEET = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
KNOBS = ("NAVRL_SPEED_GOVERNOR_WIDTH_PER_MPS", "NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M",
         "NAVRL_SPEED_GOVERNOR_WIDTH_OPEN_REF_M", "NAVRL_SPEED_GOVERNOR_WIDTH_MAX_M", "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M",
         "NAVRL_SPEED_GOVERNOR_LATERAL_SPAN_M", "NAVRL_SPEED_GOVERNOR_LATERAL_FLOOR_MPS",
         "NAVRL_SPEED_GOVERNOR_YAW_CAP_RADPS", "NAVRL_SPEED_GOVERNOR_YAW_CAP_MARGIN_M")


def bearings(h=72):
    return torch.linspace(math.pi, -math.pi + 2 * math.pi / h, h)


class DefaultsAreOff(unittest.TestCase):
    def test_defaults_reproduce_the_old_cap(self):
        cfg = SG.SpeedGovernorConfig.from_environ({"NAVRL_SPEED_GOVERNOR": "stopcap"})
        self.assertFalse(cfg.lateral_channel_enabled)
        self.assertFalse(cfg.yaw_cap_enabled)
        self.assertEqual(SG.speed_dependent_half_width(cfg, torch.tensor([3.0])), 0.45)
        cmd = torch.tensor([[2.0, 1.0], [0.5, -2.0]])
        clear = torch.tensor([1.5, 6.0])
        g0, t0 = SG.apply_speed_governor(cmd, clear, cfg)
        g1, t1 = SG.apply_speed_governor(cmd, clear, cfg, lateral_clearance_m=torch.tensor([0.1, 0.1]))
        self.assertTrue(torch.equal(g0, g1))
        self.assertTrue(torch.isinf(t0["lateral_cap_mps"]).all())

    def test_missing_lateral_input_is_refused_when_enabled(self):
        cfg = SG.SpeedGovernorConfig.from_environ(
            {"NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M": "0.6"})
        with self.assertRaises(ValueError):
            SG.apply_speed_governor(torch.ones(1, 2), torch.ones(1), cfg)


class L2Width(unittest.TestCase):
    def test_width_grows_with_requested_speed_and_widens_the_corridor(self):
        cfg = SG.SpeedGovernorConfig.from_environ(
            {"NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_WIDTH_PER_MPS": "0.2"})
        w = SG.speed_dependent_half_width(cfg, torch.tensor([0.0, 2.5]))
        self.assertEqual([round(float(x), 6) for x in w], [0.45, 0.95])
        # A bar 0.7 m beside the command line at 4 m ahead: outside a 0.45 corridor, inside 0.95.
        b = bearings()
        scan = torch.full((2, 4, 72), 12.0)
        off = int(torch.argmin((b - math.atan2(0.7, 4.0)).abs()))
        scan[:, :, off] = math.hypot(4.0, 0.7)
        cmd = torch.tensor([[0.0, 0.0], [2.5, 0.0]])
        cmd[0] = torch.tensor([2.5, 0.0])
        clear = SG.directional_lidar_clearance(scan, b, cmd, max_range_m=12.0, path_half_width_m=w)
        self.assertGreater(float(clear[1]), 3.5)  # seen by the wide corridor (~4 m ahead)
        clear_narrow = SG.directional_lidar_clearance(scan, b, cmd, max_range_m=12.0, path_half_width_m=0.45)
        self.assertAlmostEqual(float(clear_narrow[1]), 12.0, places=2)
        # tensor width at k=0 equals the scalar path bit for bit
        same = SG.directional_lidar_clearance(scan, b, cmd, max_range_m=12.0, path_half_width_m=torch.tensor([0.45, 0.45]))
        self.assertTrue(torch.equal(same, clear_narrow))


class L3Lateral(unittest.TestCase):
    def test_lateral_clearance_sees_beside_not_ahead_or_behind(self):
        b = bearings()
        scan = torch.full((1, 4, 72), 12.0)
        for deg, rng in ((0.0, 2.0), (90.0, 3.0), (-120.0, 1.5), (175.0, 0.8)):
            scan[0, :, int(torch.argmin((b - math.radians(deg)).abs()))] = rng
        lat = SG.lateral_clearance(scan, b, torch.tensor([[2.5, 0.0]]), max_range_m=12.0)
        # ahead (0 deg) and behind (175 deg) are excluded; -120 deg at 1.5 m projected wins.
        self.assertAlmostEqual(float(lat), 1.5 * math.cos(math.radians(20.0)), places=3)

    def test_cap_has_a_floor_and_releases_to_free(self):
        cfg = SG.SpeedGovernorConfig.from_environ({
            "NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M": "0.6",
            "NAVRL_SPEED_GOVERNOR_LATERAL_SPAN_M": "2.0", "NAVRL_SPEED_GOVERNOR_LATERAL_FLOOR_MPS": "1.0"})
        cap = SG.lateral_cap(torch.tensor([0.0, 0.6, 1.6, 2.6, 12.0]), cfg)
        free = math.sqrt(2.0) * 2.5
        self.assertEqual([round(float(x), 4) for x in cap],
                         [1.0, 1.0, round(1.0 + 0.5 * (free - 1.0), 4), round(free, 4), round(free, 4)])
        # direction preserved: governed command is a non-negative scalar multiple
        cmd = torch.tensor([[2.0, -1.5]])
        governed, tel = SG.apply_speed_governor(cmd, torch.tensor([12.0]), cfg, lateral_clearance_m=torch.tensor([0.3]))
        self.assertAlmostEqual(float(governed.norm()), 1.0, places=5)
        self.assertGreater(float((governed * cmd).sum()), 0.0)
        self.assertAlmostEqual(float(tel["lateral_cap_mps"]), 1.0, places=6)

    def test_floor_above_free_is_refused(self):
        with self.assertRaises(ValueError):
            SG.SpeedGovernorConfig.from_environ({
                "NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M": "0.6",
                "NAVRL_SPEED_GOVERNOR_LATERAL_FLOOR_MPS": "9.0"})


class L5Yaw(unittest.TestCase):
    def test_scale_only_inside_margin_and_never_flips(self):
        cfg = SG.SpeedGovernorConfig.from_environ({
            "NAVRL_SPEED_GOVERNOR": "riskcap", "NAVRL_SPEED_GOVERNOR_YAW_CAP_RADPS": "1.5",
            "NAVRL_SPEED_GOVERNOR_YAW_CAP_MARGIN_M": "1.0"})
        s = SG.yaw_scale(torch.tensor([0.5, 0.99, 1.0, 5.0]), 3.0, cfg)
        self.assertEqual([round(float(x), 4) for x in s], [0.5, 0.5, 1.0, 1.0])
        self.assertTrue((s > 0).all())


class L8OpenWidth(unittest.TestCase):
    """Width from the vehicle's own clutter read. Density is not observable; nearest sensed
    surface is, and L1 showed it separates the densities whose optimum widths differ."""

    def cfg(self, **extra):
        env = {"NAVRL_SPEED_GOVERNOR": "dwa_arc"}
        env.update(extra)
        return SG.SpeedGovernorConfig.from_environ(env)

    def test_off_by_default_and_identical_to_the_fixed_width(self):
        cfg = self.cfg()
        self.assertFalse(cfg.open_width_enabled)
        self.assertEqual(SG.speed_dependent_half_width(cfg, torch.tensor([2.5])), 0.45)
        # supplying a clutter read changes nothing while the gain is zero
        self.assertEqual(SG.speed_dependent_half_width(cfg, torch.tensor([2.5]), open_m=torch.tensor([9.0])), 0.45)

    def test_width_grows_with_open_space_and_is_clamped(self):
        cfg = self.cfg(NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M="0.5",
                       NAVRL_SPEED_GOVERNOR_WIDTH_OPEN_REF_M="1.2",
                       NAVRL_SPEED_GOVERNOR_WIDTH_MAX_M="2.0")
        # at the reference clutter the width is w0; more open -> wider; tighter -> narrower
        open_m = torch.tensor([1.2, 2.2, 0.7, 12.0])
        w = SG.speed_dependent_half_width(cfg, torch.zeros(4), open_m=open_m)
        self.assertAlmostEqual(float(w[0]), 0.45, places=6)
        self.assertAlmostEqual(float(w[1]), 0.95, places=6)
        self.assertAlmostEqual(float(w[2]), 0.45, places=6)   # never narrower than w0
        self.assertAlmostEqual(float(w[3]), 2.00, places=6)   # clamped by width_max_m
        self.assertTrue((w >= 0.45).all())

    def test_all_width_knobs_are_recorded_in_both_condition_dicts(self):
        source = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        for key in ("width_per_open_m", "width_open_ref_m", "width_max_m", "width_per_mps",
                    "lateral_margin_m", "yaw_cap_radps"):
            self.assertIn(f'"cfg_speed_governor_{key}"', source, key)
            self.assertIn(f'"speed_governor_{key}"', source, key)

    def test_missing_clutter_read_is_refused(self):
        cfg = self.cfg(NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M="0.5")
        with self.assertRaises(ValueError):
            SG.speed_dependent_half_width(cfg, torch.zeros(2))

    def test_combines_with_the_speed_term(self):
        cfg = self.cfg(NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M="0.5",
                       NAVRL_SPEED_GOVERNOR_WIDTH_PER_MPS="0.1")
        w = SG.speed_dependent_half_width(cfg, torch.tensor([2.0]), open_m=torch.tensor([2.2]))
        self.assertAlmostEqual(float(w), 0.45 + 0.2 + 0.5, places=6)

    def test_task_reads_clutter_from_the_omni_clearance(self):
        source = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        block = source.split("open_m = None", 1)[1].split("half_width = SG.speed_dependent_half_width", 1)[0]
        self.assertIn("omnidirectional_clearance(", block)
        self.assertIn("target_return_mask=target_return", block)


class ShellAgrees(unittest.TestCase):
    def test_shell_parses_and_exports_all_six(self):
        text = SHEET.read_text()
        for knob in KNOBS:
            self.assertGreaterEqual(text.count(knob), 3, knob)  # spec, read, export
        script = text.split('GOVERNOR_VALUES="$(${PYTHON} - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        env = {k: v for k, v in os.environ.items() if not k.startswith("NAVRL_")}
        env.update({"NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M": "0.6"})
        out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        values = out.stdout.split()
        # Resolve positions from the shell's own spec rather than hard-coding indices: inserting a
        # knob used to break this test instead of the thing it guards.
        names = re.findall(r'\("(NAVRL_SPEED_GOVERNOR_[A-Z0-9_]+)"', script)
        self.assertEqual(len(values), len(names))
        for knob in KNOBS:
            self.assertIn(knob, names, knob)
        self.assertEqual(float(values[names.index("NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M")]), 0.6)
        self.assertEqual(float(values[names.index("NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M")]), 0.0)


if __name__ == "__main__":
    unittest.main()
