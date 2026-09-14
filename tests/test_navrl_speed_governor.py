import math
import importlib.util
from pathlib import Path
import os
import subprocess
import sys
import unittest

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym/task/navrl_task/speed_governor.py"
)
_SPEC = importlib.util.spec_from_file_location("navrl_speed_governor_for_test", _MODULE_PATH)
_GOVERNOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOVERNOR)
SpeedGovernorConfig = _GOVERNOR.SpeedGovernorConfig
apply_speed_governor = _GOVERNOR.apply_speed_governor
directional_lidar_clearance = _GOVERNOR.directional_lidar_clearance


class SpeedGovernorTests(unittest.TestCase):
    def test_off_is_byte_equal(self):
        command = torch.tensor([[2.5, 2.5], [0.0, 0.0]])
        clearance = torch.tensor([0.5, 0.2])
        governed, diag = apply_speed_governor(
            command, clearance, _GOVERNOR.SpeedGovernorConfig(mode="off")
        )
        self.assertTrue(torch.equal(governed, command))
        self.assertTrue(torch.equal(diag["scale"], torch.ones(2)))

    def test_fixed_cap_preserves_direction_and_norm(self):
        command = torch.tensor([[3.0, 4.0]])
        governed, _ = apply_speed_governor(
            command,
            torch.tensor([12.0]),
            _GOVERNOR.SpeedGovernorConfig(mode="fixed", fixed_cap_mps=2.0),
        )
        self.assertTrue(torch.allclose(governed, torch.tensor([[1.2, 1.6]])))
        self.assertAlmostEqual(float(governed.norm()), 2.0, places=6)

    def test_clearance_mode_slows_only_inside_slow_zone(self):
        config = _GOVERNOR.SpeedGovernorConfig(
            mode="clearance", hard_margin_m=0.5, slow_distance_m=3.0,
            free_speed_cap_mps=2.5,
        )
        command = torch.tensor([[2.5, 0.0], [2.5, 0.0], [1.0, 0.0]])
        governed, diag = apply_speed_governor(
            command, torch.tensor([0.5, 1.5, 4.0]), config
        )
        self.assertAlmostEqual(float(governed[0].norm()), 0.0, places=6)
        self.assertAlmostEqual(float(governed[1].norm()), 1.0, places=6)
        self.assertAlmostEqual(float(governed[2].norm()), 1.0, places=6)
        self.assertLess(float(diag["stopping_margin_requested_m"][1]), 0.0)

    def test_ttc_mode_enforces_requested_time_headway(self):
        config = _GOVERNOR.SpeedGovernorConfig(mode="ttc", hard_margin_m=0.5, ttc_s=1.0)
        command = torch.tensor([[3.0, 0.0]])
        governed, diag = apply_speed_governor(command, torch.tensor([1.5]), config)
        self.assertAlmostEqual(float(governed.norm()), 1.0, places=6)
        self.assertAlmostEqual(float(diag["scale"]), 1.0 / 3.0, places=6)

    def test_riskcap_never_forces_a_stop_and_releases_in_open_space(self):
        config = _GOVERNOR.SpeedGovernorConfig(
            mode="riskcap",
            fixed_cap_mps=2.0,
            free_speed_cap_mps=3.5,
            slow_distance_m=3.0,
            release_distance_m=5.0,
        )
        command = torch.tensor([[3.5, 0.0], [3.5, 0.0], [3.5, 0.0], [1.0, 0.0]])
        governed, diag = apply_speed_governor(
            command, torch.tensor([0.45, 3.0, 4.0, 0.45]), config
        )
        self.assertTrue(
            torch.allclose(
                governed.norm(dim=1), torch.tensor([2.0, 2.0, 2.75, 1.0]), atol=1e-6
            )
        )
        self.assertTrue(torch.all(diag["executed_speed_mps"] > 0.0))

    def test_stopcap_allows_full_stop_and_matches_closed_form(self):
        config = _GOVERNOR.SpeedGovernorConfig(
            mode="stopcap",
            hard_margin_m=0.45,
            brake_mps2=2.9608856678,
            reaction_s=0.1,
            free_speed_cap_mps=3.53553390593,
        )
        command = torch.tensor([[3.5, 0.0]] * 4)
        clearance = torch.tensor([0.45, 1.0, 1.5, 12.0])
        governed, diag = apply_speed_governor(command, clearance, config)
        # usable=0 -> cap=0: unlike riskcap the mode may force a stop at contact range.
        self.assertAlmostEqual(float(governed[0].norm()), 0.0, places=6)
        # Closed form: cap = sqrt((a*t)^2 + 2*a*usable) - a*t
        a, t = 2.9608856678, 0.1
        for i, c in enumerate([0.45, 1.0, 1.5, 12.0]):
            usable = max(c - 0.45, 0.0)
            expected = min(math.sqrt((a * t) ** 2 + 2 * a * usable) - a * t, 3.53553390593)
            self.assertAlmostEqual(float(diag["speed_cap_mps"][i]), expected, places=5)
        # Monotone non-decreasing in clearance, never above the free cap.
        caps = diag["speed_cap_mps"]
        self.assertTrue(torch.all(caps[1:] >= caps[:-1]))
        self.assertTrue(torch.all(caps <= 3.53553390593 + 1e-6))

    def test_stopcap_executed_stopping_margin_is_nonnegative_by_construction(self):
        config = _GOVERNOR.SpeedGovernorConfig(
            mode="stopcap", hard_margin_m=0.45, brake_mps2=2.9608856678, reaction_s=0.1
        )
        torch.manual_seed(0)
        command = torch.randn(256, 2) * 3.0
        clearance = torch.rand(256) * 12.0
        _, diag = apply_speed_governor(command, clearance, config)
        self.assertTrue(
            torch.all(diag["stopping_margin_executed_m"] >= -1e-5),
            msg=f"min margin {float(diag['stopping_margin_executed_m'].min())}",
        )

    def test_stopcap_environ_contract_requires_positive_brake(self):
        with self.assertRaises(ValueError):
            SpeedGovernorConfig.from_environ(
                {
                    "NAVRL_SPEED_GOVERNOR": "stopcap",
                    "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2": "0.0",
                }
            )
        config = SpeedGovernorConfig.from_environ(
            {"NAVRL_SPEED_GOVERNOR": "stopcap"}
        )
        self.assertEqual(config.mode, "stopcap")

    def test_directional_clearance_uses_command_frame(self):
        bearings = torch.tensor([math.pi / 2.0, 0.0, -math.pi / 2.0])
        scan = torch.full((2, 1, 3), 12.0)
        scan[0, 0, 1] = 2.0  # forward obstacle
        scan[1, 0, 0] = 3.0  # left obstacle
        command = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        clearance = _GOVERNOR.directional_lidar_clearance(
            scan,
            bearings,
            command,
            max_range_m=12.0,
            path_half_width_m=0.45,
            vertical_fov_deg=(0.0, 0.0),
        )
        self.assertTrue(torch.allclose(clearance, torch.tensor([2.0, 3.0]), atol=1e-5))

    def test_target_return_is_not_a_collision_obstacle(self):
        bearings = torch.tensor([0.0])
        scan = torch.tensor([[[1.0]]])
        target = torch.ones_like(scan, dtype=torch.bool)
        clearance = _GOVERNOR.directional_lidar_clearance(
            scan,
            bearings,
            torch.tensor([[1.0, 0.0]]),
            max_range_m=12.0,
            path_half_width_m=0.45,
            target_return_mask=target,
            vertical_fov_deg=(0.0, 0.0),
        )
        self.assertAlmostEqual(float(clearance), 12.0, places=6)

    def test_vertical_slant_range_is_projected(self):
        bearings = torch.tensor([0.0])
        scan = torch.tensor([[[12.0], [2.0]]])
        clearance = _GOVERNOR.directional_lidar_clearance(
            scan,
            bearings,
            torch.tensor([[1.0, 0.0]]),
            max_range_m=12.0,
            path_half_width_m=0.45,
            vertical_fov_deg=(0.0, 60.0),
        )
        self.assertAlmostEqual(float(clearance), 1.0, places=5)

    def test_default_vertical_angles_follow_warp_tensor_row_order(self):
        bearings = torch.tensor([0.0])
        scan = torch.tensor([[[2.0], [12.0], [12.0], [12.0]]])
        clearance = _GOVERNOR.directional_lidar_clearance(
            scan,
            bearings,
            torch.tensor([[1.0, 0.0]]),
            max_range_m=12.0,
            path_half_width_m=0.45,
        )
        self.assertAlmostEqual(
            float(clearance), 2.0 * math.cos(math.radians(20.0)), places=5
        )

    def test_environment_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            SpeedGovernorConfig.from_environ({"NAVRL_SPEED_GOVERNOR": "magic"})
        with self.assertRaises(ValueError):
            SpeedGovernorConfig.from_environ(
                {
                    "NAVRL_SPEED_GOVERNOR": "clearance",
                    "NAVRL_SPEED_GOVERNOR_MARGIN_M": "3",
                    "NAVRL_SPEED_GOVERNOR_SLOW_M": "2",
                }
            )
        with self.assertRaises(ValueError):
            SpeedGovernorConfig.from_environ(
                {
                    "NAVRL_SPEED_GOVERNOR": "riskcap",
                    "NAVRL_SPEED_GOVERNOR_SLOW_M": "3",
                    "NAVRL_SPEED_GOVERNOR_RELEASE_M": "3",
                }
            )


class A4BaselineGeometries(unittest.TestCase):
    """omni and dwa_arc change ONLY the geometry of the clearance measurement.

    Both reuse the stopcap stopping law, so any difference in outcome is attributable to where
    the filter looks -- which is the question the contact forensics raised (57-58% of contacts
    were lateral to the 0.45 m corridor).
    """

    RNG = 12.0
    H = 72

    def _bearings(self):
        return torch.linspace(math.pi, -math.pi + 2 * math.pi / self.H, self.H)

    def _scan(self, fill=None):
        return torch.full((1, 4, self.H), self.RNG if fill is None else float(fill))

    def test_omni_sees_a_purely_lateral_obstacle_that_the_corridor_misses(self):
        bearings = self._bearings()
        scan = self._scan()
        lateral_bin = int(torch.argmin((bearings - (math.pi / 2)).abs()))
        scan[0, :, lateral_bin] = 1.5
        cmd = torch.tensor([[2.5, 0.0]])

        corridor = _GOVERNOR.directional_lidar_clearance(
            scan, bearings, cmd, max_range_m=self.RNG, path_half_width_m=0.45
        )
        omni = _GOVERNOR.omnidirectional_clearance(scan, max_range_m=self.RNG)
        # The corridor never sees it; omni does. The expected value is the HORIZONTAL projection
        # of the 1.5 m slant range at the +20 deg row -- the same projection the corridor applies,
        # which is the point: only where we look changed, not how range is reduced.
        self.assertAlmostEqual(float(corridor), self.RNG, places=2)
        self.assertAlmostEqual(float(omni), 1.5 * math.cos(math.radians(20.0)), places=3)

    def test_arc_degenerates_to_the_straight_corridor_at_zero_yaw_rate(self):
        bearings = self._bearings()
        scan = self._scan()
        fwd = int(torch.argmin(bearings.abs()))
        scan[0, :, fwd] = 4.0
        cmd = torch.tensor([[2.5, 0.0]])

        straight = _GOVERNOR.directional_lidar_clearance(
            scan, bearings, cmd, max_range_m=self.RNG, path_half_width_m=0.45
        )
        arc = _GOVERNOR.arc_clearance(
            scan, bearings, cmd, torch.zeros(1),
            max_range_m=self.RNG, path_half_width_m=0.45,
        )
        self.assertAlmostEqual(float(arc), float(straight), places=3)

    def test_arc_and_line_disagree_while_turning(self):
        """The DWA objection, made measurable: turning changes what is in the way."""
        bearings = self._bearings()
        scan = self._scan()
        # An obstacle off to one side: not in the straight tube, but on a curving path.
        off = int(torch.argmin((bearings - math.radians(35.0)).abs()))
        scan[0, :, off] = 3.0
        cmd = torch.tensor([[2.5, 0.0]])

        straight = float(
            _GOVERNOR.directional_lidar_clearance(
                scan, bearings, cmd, max_range_m=self.RNG, path_half_width_m=0.45
            )
        )
        turning = float(
            _GOVERNOR.arc_clearance(
                scan, bearings, cmd, torch.tensor([1.2]),
                max_range_m=self.RNG, path_half_width_m=0.45,
            )
        )
        self.assertAlmostEqual(straight, self.RNG, places=2)
        self.assertLess(turning, self.RNG, "a turn must bring the off-axis obstacle into play")

    def test_both_modes_use_the_stopcap_law(self):
        cmd = torch.tensor([[3.0, 0.0]])
        clear = torch.tensor([2.0])
        caps = {}
        for mode in ("stopcap", "omni", "dwa_arc"):
            cfg = _GOVERNOR.SpeedGovernorConfig(mode=mode)
            _, tel = apply_speed_governor(cmd, clear, cfg)
            caps[mode] = float(tel["speed_cap_mps"])
        self.assertAlmostEqual(caps["omni"], caps["stopcap"], places=6)
        self.assertAlmostEqual(caps["dwa_arc"], caps["stopcap"], places=6)

    def test_modes_are_accepted_by_the_config(self):
        for mode in ("omni", "dwa_arc"):
            cfg = SpeedGovernorConfig.from_environ({"NAVRL_SPEED_GOVERNOR": mode})
            self.assertEqual(cfg.mode, mode)


class WhitelistsAgree(unittest.TestCase):
    """The evaluator shell keeps its own mode whitelist, independent of the Python one.

    The first A4 attempt died after four arms because `omni` was added to
    VALID_SPEED_GOVERNOR_MODES but not to eval_navrl_v2_density_sweep.sh. This pins them
    together so the next mode cannot repeat it.
    """

    def test_shell_whitelist_matches_the_python_one(self):
        sweep = (
            Path(__file__).resolve().parents[1]
            / "aerial_gym" / "rl_training" / "rl_games" / "eval_navrl_v2_density_sweep.sh"
        ).read_text()
        line = next(
            l for l in sweep.splitlines() if l.strip().startswith("off|") and l.strip().endswith(") ;;")
        )
        shell_modes = set(line.strip().rstrip(") ;;").split("|"))
        self.assertEqual(
            shell_modes,
            set(_GOVERNOR.VALID_SPEED_GOVERNOR_MODES),
            "the evaluator shell and speed_governor.py must accept the same modes",
        )

    def test_shell_applies_riskcap_validation_to_both_geometries(self):
        sweep = (
            Path(__file__).resolve().parents[1]
            / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
        ).read_text()
        validation = next(
            line for line in sweep.splitlines()
            if line.startswith('if os.environ.get("NAVRL_SPEED_GOVERNOR",')
        )
        self.assertIn('in ("riskcap", "riskcap_arc")', validation)
        # Execute only the shell's numeric-validation Python, never its simulator path.
        script = sweep.split('GOVERNOR_VALUES="$(${PYTHON} - <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        for mode in ("riskcap", "riskcap_arc"):
            for bad in (
                {"NAVRL_SPEED_GOVERNOR_RELEASE_M": "3"},
                {"NAVRL_SPEED_GOVERNOR_FREE_MPS": "1.9"},
            ):
                with self.subTest(mode=mode, invalid=bad):
                    env = {k: v for k, v in os.environ.items() if not k.startswith("NAVRL_")}
                    env.update({"NAVRL_SPEED_GOVERNOR": mode, **bad})
                    result = subprocess.run(
                        [sys.executable, "-c", script], env=env,
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("[eval_v2] riskcap", result.stderr)


class A7RiskcapArc(unittest.TestCase):
    def test_m1_caps_and_commands_equal_at_every_clearance(self):
        generator = torch.Generator().manual_seed(509)
        command = torch.randn(256, 2, generator=generator) * 3.0
        clearance = torch.cat((torch.linspace(0.0, 12.0, 252), torch.tensor([0.45, 3.0, 4.0, 5.0])))
        line, line_diag = apply_speed_governor(command, clearance, SpeedGovernorConfig(mode="riskcap"))
        arc, arc_diag = apply_speed_governor(command, clearance, SpeedGovernorConfig(mode="riskcap_arc"))
        self.assertTrue(torch.allclose(line, arc, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(
            line_diag["speed_cap_mps"], arc_diag["speed_cap_mps"], atol=1e-6, rtol=0.0,
        ))

    def test_modes_share_environment_validation(self):
        for mode in ("riskcap", "riskcap_arc"):
            self.assertEqual(SpeedGovernorConfig.from_environ({"NAVRL_SPEED_GOVERNOR": mode}).mode, mode)
            for invalid in (
                {"NAVRL_SPEED_GOVERNOR_RELEASE_M": "3"},
                {"NAVRL_SPEED_GOVERNOR_RELEASE_M": "2.9"},
                {"NAVRL_SPEED_GOVERNOR_FREE_MPS": "1.9"},
            ):
                with self.subTest(mode=mode, invalid=invalid), self.assertRaises(ValueError):
                    SpeedGovernorConfig.from_environ({"NAVRL_SPEED_GOVERNOR": mode, **invalid})

    def test_m2_zero_yaw_clearance_and_cap_match_at_corridor_boundary(self):
        # Horizontal returns on either side of the 0.45 m boundary. A huge-radius
        # approximation loses this distinction in float32 through cancellation.
        for lateral in (0.44, 0.45, 0.46, -0.44, -0.45, -0.46):
            with self.subTest(lateral=lateral):
                bearings = torch.tensor([math.asin(lateral / 4.0)])
                scan = torch.tensor([[[4.0 / math.cos(math.radians(20.0))]]])
                command = torch.tensor([[3.5, 0.0]])
                kwargs = {"max_range_m": 12.0, "path_half_width_m": 0.45}
                line = directional_lidar_clearance(scan, bearings, command, **kwargs)
                arc = _GOVERNOR.arc_clearance(scan, bearings, command, torch.zeros(1), **kwargs)
                self.assertTrue(torch.equal(line, arc), msg=f"line={line}, arc={arc}")
                _, line_diag = apply_speed_governor(command, line, SpeedGovernorConfig(mode="riskcap"))
                _, arc_diag = apply_speed_governor(command, arc, SpeedGovernorConfig(mode="riskcap_arc"))
                self.assertTrue(torch.allclose(
                    line_diag["speed_cap_mps"], arc_diag["speed_cap_mps"], atol=1e-6, rtol=0.0,
                ))

    def test_m2_zero_yaw_matches_for_batched_scans_masks_and_directions(self):
        generator = torch.Generator().manual_seed(491)
        scan = torch.rand(128, 4, 72, generator=generator) * 12.0
        scan[0] = 12.0
        scan[1, 0, 0] = float("nan")
        scan[1, 0, 1] = float("inf")
        command = torch.randn(128, 2, generator=generator) * 2.0
        command[2] = 0.0
        bearings = torch.linspace(math.pi, -math.pi + 2 * math.pi / 72, 72)
        kwargs = {
            "max_range_m": 12.0, "path_half_width_m": 0.45,
            "target_return_mask": torch.rand(128, 4, 72, generator=generator) < 0.2,
        }
        line = directional_lidar_clearance(scan, bearings, command, **kwargs)
        arc = _GOVERNOR.arc_clearance(scan, bearings, command, torch.zeros(128), **kwargs)
        self.assertTrue(torch.equal(line, arc), msg=f"max delta={float((line - arc).abs().max())}")

    def test_task_routes_both_arc_modes_to_the_same_clearance_function(self):
        source = (
            Path(__file__).resolve().parents[1] / "aerial_gym/task/navrl_task/navrl_task.py"
        ).read_text()
        branch = source.split('elif mode in ("dwa_arc", "riskcap_arc"):', 1)[1].split('\n        else:', 1)[0]
        self.assertIn("clearance = arc_clearance(", branch)
        self.assertIn('self.obs_dict["robot_body_angvel"][:, 2].detach()', branch)


if __name__ == "__main__":
    unittest.main()
