"""Unit tests for latency compensation P0/P1 (WORKLOG 2026-08-05 checklist item 4).

CPU-only: loads navrl_perception by file path (no aerial_gym package import, so no Isaac Gym),
drives the real BatchedConstantVelocityTracker with tau-delayed measurements of a synthetic
constant-velocity target, and asserts the forward-predicted position beats the raw delayed
estimate on both position and bearing error.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_latency_compensate.py
"""

import importlib.util
import math
import re
from pathlib import Path
import sys
import types
import unittest

import torch

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

_SPEC = importlib.util.spec_from_file_location(
    "navrl_perception_standalone", _TASK_DIR / "navrl_perception.py"
)
_PERCEPTION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PERCEPTION)
BatchedConstantVelocityTracker = _PERCEPTION.BatchedConstantVelocityTracker

DT = 0.1
TAU = 0.1  # NAVRL_DETECTION_LATENCY_S of the R3 catastrophic arm
STEPS = 60


def _run_delayed_cv_tracking(tau_steps, speed_mps=1.5, heading_rad=0.7):
    """Feed the KF measurements that lag the true CV target by tau_steps; return final errors."""
    tracker = BatchedConstantVelocityTracker(1, "cpu", DT, memory_s=3.0)
    vel = torch.tensor([[math.cos(heading_rad), math.sin(heading_rad), 0.0]]) * speed_mps
    pos0 = torch.tensor([[4.0, -2.0, 1.0]])
    meas_var = torch.full((1, 3), 0.05**2)

    truth_now = None
    for k in range(STEPS):
        t_now = k * DT
        t_meas = max(0.0, t_now - tau_steps * DT)
        truth_now = pos0 + vel * t_now
        measurement = pos0 + vel * t_meas
        tracker.step(measurement, torch.ones(1, dtype=torch.bool), meas_var)

    raw_pos = tracker.state[:, :3]
    # P0: exactly the expression _target_features uses when latency_compensate is on.
    pred_pos = tracker.state[:, :3] + tracker.state[:, 3:] * (tau_steps * DT)

    def bearing_err(est):
        # Bearing from a drone at the origin, as the policy's vehicle frame would see it.
        b_est = math.atan2(float(est[0, 1]), float(est[0, 0]))
        b_true = math.atan2(float(truth_now[0, 1]), float(truth_now[0, 0]))
        return abs(math.atan2(math.sin(b_est - b_true), math.cos(b_est - b_true)))

    return {
        "raw_pos_err": float((raw_pos - truth_now).norm()),
        "pred_pos_err": float((pred_pos - truth_now).norm()),
        "raw_bearing_err": bearing_err(raw_pos),
        "pred_bearing_err": bearing_err(pred_pos),
        "vel_err": float((tracker.state[:, 3:] - vel).norm()),
    }


class LatencyCompensationMath(unittest.TestCase):
    def test_forward_predict_beats_raw_delayed(self):
        """Checklist 4: CV target + tau=0.1 -> predicted bearing error < raw delayed error."""
        r = _run_delayed_cv_tracking(tau_steps=1)
        self.assertLess(r["pred_bearing_err"], r["raw_bearing_err"])
        self.assertLess(r["pred_pos_err"], r["raw_pos_err"])
        # The KF's velocity estimate is unbiased for a CV target even from delayed positions,
        # which is the whole reason output-side extrapolation works.
        self.assertLess(r["vel_err"], 0.15)
        # The raw estimate must actually exhibit the ~v*tau lag being compensated
        # (0.15 m at 1.5 m/s * 0.1 s); the prediction should remove most of it.
        self.assertGreater(r["raw_pos_err"], 0.10)
        self.assertLess(r["pred_pos_err"], 0.5 * r["raw_pos_err"])

    def test_two_step_latency_also_recovered(self):
        r = _run_delayed_cv_tracking(tau_steps=2)
        self.assertLess(r["pred_pos_err"], 0.5 * r["raw_pos_err"])
        self.assertLess(r["pred_bearing_err"], r["raw_bearing_err"])

    def test_zero_latency_predict_is_identity(self):
        """With tau=0 the compensation expression is a no-op by construction."""
        r = _run_delayed_cv_tracking(tau_steps=0)
        self.assertAlmostEqual(r["pred_pos_err"], r["raw_pos_err"], places=6)

    def test_static_target_unaffected(self):
        """A hovering target has ~zero estimated velocity: predict must not invent motion."""
        r = _run_delayed_cv_tracking(tau_steps=1, speed_mps=0.0)
        self.assertLess(abs(r["pred_pos_err"] - r["raw_pos_err"]), 0.02)


def _method_body(source, name):
    """The whole text of a method: from its `def` to the next definition at the same indent.

    A fixed-size character window silently stops covering the code it was written for as the
    method grows. `observe` passed 12 kB in 2026-09, which pushed the LiDAR-backup gate to
    offset 4,220 and out of the old 4 kB window, failing this check while the gate was intact."""
    match = re.search(rf"^([ \t]*)def {re.escape(name)}\b", source, re.M)
    if match is None:
        raise AssertionError(f"method not found: {name}")
    start = match.start()
    following = re.search(rf"^{match.group(1)}(?:def |@)", source[start + 1:], re.M)
    return source[start:start + 1 + following.start()] if following else source[start:]


class LatencyCompensationPlumbing(unittest.TestCase):
    """Guard the cfg wiring by source inspection (the module needs no simulator to check this)."""

    SOURCE = (_TASK_DIR / "navrl_perception.py").read_text(encoding="utf-8")

    def test_p0_is_output_side_only(self):
        # P0 must live in _target_features (policy-facing output), not inside the tracker.
        # Anchored on the method itself, so reordering methods cannot silently empty the slice.
        body = _method_body(self.SOURCE, "_target_features")
        self.assertIn("latency_compensate", body)
        self.assertIn("state[:, 3:] * self.detection_latency_s", body)
        tracker_src = self.SOURCE[
            self.SOURCE.index("class BatchedConstantVelocityTracker") :
            self.SOURCE.index("class NavRLPerceptionModule")
        ]
        self.assertNotIn("latency_compensate", tracker_src)

    def test_p1_gate_reads_backup_flag(self):
        body = _method_body(self.SOURCE, "observe")
        self.assertIn("latency_lidar_backup", body)
        self.assertIn("lidar_camera_gate", body)

    def test_knobs_read_from_cfg_defaults_off(self):
        self.assertIn('getattr(cfg, "latency_compensate", False)', self.SOURCE)
        self.assertIn('getattr(cfg, "latency_lidar_backup", False)', self.SOURCE)


def _yaw_quat(yaw):
    return torch.tensor([[0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)]])


def _module(latency_s, obstacle_fix, num_envs=1, ego_motion_fix=False):
    camera = types.SimpleNamespace(
        detector_max_range=20.0,
        detector_hfov_deg=87.0,
        detector_vfov_deg=58.0,
        camera_width=80,
        camera_height=45,
        camera_translation=[0.1, 0.0, 0.03],
        camera_target_radius=0.15,
        tracker_memory_s=5.0,
    )
    perception = types.SimpleNamespace(
        lidar_max_range=4.0,
        min_target_pixels=2,
        pixel_threshold=0.55,
        detection_dropout_prob=0.0,
        detection_latency_s=latency_s,
        range_error_m=0.0,
        latency_compensate=False,
        latency_lidar_backup=False,
        latency_obstacle_fix=obstacle_fix,
        latency_ego_motion_fix=ego_motion_fix,
        pose_noise_seed=9163,
        rgb_noise_std=0.0,
        depth_noise_std=0.0,
        history_interval_s=0.5,
        detector_checkpoint="",
    )
    return _PERCEPTION.NavRLPerceptionModule(num_envs, "cpu", perception, DT, camera)


def _frame(center_col, target_range):
    """RGB-D frame with the orange target blob at a given column, everything else far."""
    rgb = torch.full((1, 3, 45, 80), 0.15)
    depth = torch.full((1, 45, 80), 10.0)
    cols = slice(center_col - 2, center_col + 3)
    rgb[0, 0, 20:24, cols] = 0.88
    rgb[0, 1, 20:24, cols] = 0.08
    rgb[0, 2, 20:24, cols] = 0.045
    depth[0, 20:24, cols] = target_range
    return rgb, depth


def _sweep(module, lidar, columns, target_range=3.0):
    """Drive the module over a target sweeping across the image; capture the map-path inputs."""
    captured = {}
    original = module._fuse_static_and_extract_obstacles

    def spy(lidar_m, raw_depth, target_pixels, surface, bearing, visible, **kwargs):
        captured["pixels"] = target_pixels.clone()
        captured["surface"] = surface.clone()
        captured["bearing"] = bearing.clone()
        captured["visible"] = visible.clone()
        return original(lidar_m, raw_depth, target_pixels, surface, bearing, visible, **kwargs)

    module._fuse_static_and_extract_obstacles = spy
    pos, vel = torch.zeros(1, 3), torch.zeros(1, 3)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    for col in columns:
        rgb, depth = _frame(col, target_range)
        module.observe(
            rgb, depth, lidar.clone(), pos, vel, quat,
            torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
        )
    return captured


class LatencyObstacleMapFix(unittest.TestCase):
    """P2: the obstacle map must not be edited at a STALE target bearing (WORKLOG 2026-08-05)."""

    COLUMNS = [30, 33, 36, 39, 42, 45]  # target sweeping right across the image

    def _lidar(self):
        # A bar at 3.0 m in every bearing bin: whichever bin the carve-out touches gets erased,
        # which is exactly the failure being measured.
        return torch.full((1, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), 3.0)

    def test_predict_relocates_carve_out_to_the_tracked_target(self):
        stale = _sweep(_module(TAU, "off"), self._lidar(), self.COLUMNS)
        fixed_module = _module(TAU, "predict")
        fixed = _sweep(fixed_module, self._lidar(), self.COLUMNS)

        state = fixed_module.tracker.state
        predicted = state[:, :3] + state[:, 3:] * TAU
        expected_bearing = math.atan2(float(predicted[0, 1]), float(predicted[0, 0]))
        self.assertAlmostEqual(float(fixed["bearing"][0]), expected_bearing, places=5)
        # The whole point: the map is no longer edited where the target USED to be.
        self.assertGreater(
            abs(float(fixed["bearing"][0]) - float(stale["bearing"][0])), math.radians(2.0)
        )

    def test_predict_rebuilds_the_pixel_mask_at_the_predicted_bearing(self):
        fixed_module = _module(TAU, "predict")
        fixed = _sweep(fixed_module, self._lidar(), self.COLUMNS)
        self.assertTrue(bool(fixed["visible"][0]))
        columns = fixed["pixels"][0].any(dim=0).nonzero().flatten()
        self.assertGreater(columns.numel(), 0)
        angles = fixed_module._pixel_angles[columns]
        self.assertLess(
            float((angles - float(fixed["bearing"][0])).abs().max()),
            _PERCEPTION.TARGET_LIKE_ANGLE_RAD,
        )

    def test_skip_never_deletes_a_return_from_the_map(self):
        """Conservative arm: the fused map may only get MORE occupied, never more free."""
        off_module = _module(TAU, "off")
        _sweep(off_module, self._lidar(), self.COLUMNS)
        skip_module = _module(TAU, "skip")
        _sweep(skip_module, self._lidar(), self.COLUMNS)
        self.assertFalse(bool(skip_module.last_target_like.any()))
        self.assertTrue(bool(off_module.last_target_like.any()))
        self.assertTrue(
            bool((skip_module.last_scan_nearest <= off_module.last_scan_nearest + 1e-6).all())
        )
        self.assertTrue(
            bool((skip_module.last_scan_nearest < off_module.last_scan_nearest - 1e-6).any())
        )

    def test_no_latency_makes_every_mode_bit_identical(self):
        """Clean runs must be untouched, so the fix can ship enabled-by-arm without a re-baseline."""
        lidar = self._lidar()
        baseline = _sweep(_module(0.0, "off"), lidar, self.COLUMNS)
        for mode in ("predict", "skip"):
            arm = _sweep(_module(0.0, mode), lidar, self.COLUMNS)
            for key in ("pixels", "surface", "bearing", "visible"):
                self.assertTrue(
                    bool(torch.equal(baseline[key], arm[key])),
                    f"mode={mode} changed {key} at zero latency",
                )

    def test_pixel_angles_invert_the_lidar_column_map(self):
        """Guards the reconstruction: _pixel_angles must be the inverse of the camera_u map."""
        module = _module(TAU, "predict")
        u = (
            (module.hfov * 0.5 - module._pixel_angles) / module.hfov * (module.width - 1)
        ).round().long()
        self.assertTrue(bool(torch.equal(u, torch.arange(module.width))))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _module(TAU, "compensate")


class TargetMaskBackfill(unittest.TestCase):
    """The two halves of the obstacle map must agree about where the target is.

    The LiDAR target_like carve-out is gated on fused visibility (camera OR LiDAR) while the
    depth blanking is gated on the camera-only pixel mask. On a frame the camera missed but the
    LiDAR track survived -- 30% of frames under detection dropout -- the LiDAR half deleted the
    target and the camera half put it straight back as a phantom obstacle dead ahead
    (WORKLOG 2026-08-07).
    """

    TARGET_RANGE = 3.0
    FREE_RANGE = 4.0

    def _scene(self, module):
        n, h, w = 1, module.height, module.width
        lidar = torch.full((n, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), self.FREE_RANGE)
        zero_bin = int(torch.argmin(module._lidar_angles.abs()))
        lidar.view(n, _PERCEPTION.VBEAMS, _PERCEPTION.HBEAMS)[:, :, zero_bin] = self.TARGET_RANGE
        depth = torch.full((n, h, w), 10.0)
        centre = w // 2
        depth[:, 18:26, centre - 3 : centre + 4] = self.TARGET_RANGE
        pixels = torch.zeros(n, h, w, dtype=torch.bool)
        pixels[:, 18:26, centre - 3 : centre + 4] = True
        return lidar, depth, pixels, zero_bin

    def _map_range_ahead(self, module, camera_mask_present):
        lidar, depth, pixels, zero_bin = self._scene(module)
        mask = pixels if camera_mask_present else torch.zeros_like(pixels)
        if module.target_mask_backfill and not camera_mask_present:
            mask = module._reconstruct_target_pixels(
                depth, torch.zeros(1), torch.full((1,), self.TARGET_RANGE),
                torch.ones(1, dtype=torch.bool),
            )
        module._fuse_static_and_extract_obstacles(
            lidar, depth, mask, torch.full((1,), self.TARGET_RANGE), torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            drone_vel_w=torch.zeros(1, 3), vehicle_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        return float(module.last_scan_nearest[0, zero_bin])

    def test_missing_camera_mask_reinstates_the_target_as_an_obstacle(self):
        """The defect itself: same visibility, but a dropped camera frame fills the map."""
        module = _module(0.0, "off")
        self.assertFalse(module.target_mask_backfill)
        self.assertAlmostEqual(self._map_range_ahead(module, True), self.FREE_RANGE, places=3)
        self.assertAlmostEqual(self._map_range_ahead(module, False), self.TARGET_RANGE, places=3)

    def test_backfill_makes_both_halves_agree(self):
        module = _module(0.0, "off")
        module.target_mask_backfill = True
        self.assertAlmostEqual(self._map_range_ahead(module, True), self.FREE_RANGE, places=3)
        self.assertAlmostEqual(self._map_range_ahead(module, False), self.FREE_RANGE, places=3)

    def test_backfill_leaves_a_present_camera_mask_alone(self):
        """It must only fill a gap, never override what the detector actually reported."""
        plain, filled = _module(0.0, "off"), _module(0.0, "off")
        filled.target_mask_backfill = True
        self.assertAlmostEqual(
            self._map_range_ahead(plain, True), self._map_range_ahead(filled, True), places=6
        )


class LidarRangeOnlyUpdate(unittest.TestCase):
    """H3: a range-along-a-predicted-bearing must not shrink the covariance it never observed."""

    LIDAR_VAR = torch.tensor([[0.08**2, 0.15**2, 0.20**2]])
    TRUTH = torch.tensor([[6.0, 0.0, 1.0]])

    def _coast(self, steps, range_only):
        """Lock on, then lose the camera while the LiDAR keeps 'correcting' every step."""
        tracker = _PERCEPTION.BatchedConstantVelocityTracker(1, "cpu", DT, memory_s=5.0)
        cam_var = torch.full((1, 3), 0.05**2)
        seen, unseen = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
        for _ in range(2):
            tracker.step(self.TRUTH, seen, cam_var)
        for _ in range(steps):
            tracker.step(torch.zeros(1, 3), unseen, cam_var)
            rel = tracker.state[:, :3]
            bearing = torch.atan2(rel[:, 1], rel[:, 0])
            center_range = rel.norm(dim=1)
            # Exactly what _associate_lidar_target builds: bearing and z from the PREDICTION.
            meas = torch.stack(
                [center_range * torch.cos(bearing), center_range * torch.sin(bearing), rel[:, 2]],
                dim=1,
            )
            cov = None
            if range_only:
                unit = meas / meas.norm(dim=1, keepdim=True).clamp(min=1e-6)
                outer = unit.unsqueeze(2) * unit.unsqueeze(1)
                eye = torch.eye(3).unsqueeze(0)
                cov = (self.LIDAR_VAR[:, 0].view(-1, 1, 1) * outer
                       + _PERCEPTION.LIDAR_UNOBSERVED_SIGMA_M**2 * (eye - outer))
            tracker.correct(meas, seen, self.LIDAR_VAR, measurement_cov=cov)
        sigma = torch.diagonal(tracker.cov[:, :3, :3], dim1=1, dim2=2)[0].sqrt()
        return float(sigma[1]), float(sigma[2])

    def test_full_3d_update_freezes_the_unobserved_covariance(self):
        """The defect: lateral sigma barely moves over 20 blind steps."""
        near_lat, _ = self._coast(2, range_only=False)
        far_lat, _ = self._coast(20, range_only=False)
        self.assertLess(far_lat, near_lat * 2.0)

    def test_range_only_lets_the_unobserved_covariance_grow(self):
        near_lat, near_vert = self._coast(2, range_only=True)
        far_lat, far_vert = self._coast(20, range_only=True)
        self.assertGreater(far_lat, near_lat * 3.0)
        self.assertGreater(far_vert, near_vert * 3.0)

    def test_range_only_reports_more_uncertainty_than_the_full_update(self):
        blind_lat, blind_vert = self._coast(20, range_only=True)
        full_lat, full_vert = self._coast(20, range_only=False)
        self.assertGreater(blind_lat, full_lat)
        self.assertGreater(blind_vert, full_vert)

    def test_default_keeps_the_current_behaviour(self):
        module = _module(0.0, "off")
        self.assertFalse(module.lidar_range_only_update)
        self.assertEqual(module.lidar_assoc_gate_m, 0.0)


class LidarSilentCorrect(unittest.TestCase):
    """H4: the range correction stays; the 'visible, just seen' flags must go."""

    COLUMNS = [30, 33, 36, 39, 42, 45]

    def _run(self, silent):
        module = _module(0.0, "off")
        module.lidar_silent_correct = silent
        lidar = torch.full((1, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), 4.0)
        zero_bin = int(torch.argmin(module._lidar_angles.abs()))
        lidar.view(1, _PERCEPTION.VBEAMS, _PERCEPTION.HBEAMS)[:, :, zero_bin] = 3.0
        pos, vel = torch.zeros(1, 3), torch.zeros(1, 3)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        diag = None
        for step, col in enumerate(self.COLUMNS):
            rgb, depth = _frame(col, 3.0)
            if step >= 3:
                rgb = torch.full_like(rgb, 0.15)  # camera drops the target
            _, diag = module.observe(
                rgb, depth, lidar.clone(), pos, vel, quat,
                torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
            )
        return module, diag

    def test_flags_are_silenced_but_the_state_is_still_corrected(self):
        loud, loud_diag = self._run(False)
        quiet, quiet_diag = self._run(True)
        # Flags: the loud path claims sight through the association; the quiet one does not.
        self.assertTrue(bool(loud_diag["visible"][0]))
        self.assertFalse(bool(quiet_diag["visible"][0]))
        self.assertGreater(float(loud_diag["track_age"][0]), -1)  # present in diagnostics
        self.assertGreater(
            float(quiet_diag["track_age"][0]), float(loud_diag["track_age"][0])
        )
        # State: both trackers received the SAME range corrections, so they must agree --
        # this is exactly what separates H4 from H2, whose tracker coasts uncorrected.
        self.assertTrue(bool(torch.allclose(loud.tracker.state, quiet.tracker.state)))

    def test_differs_from_no_assoc_in_state_only(self):
        quiet, _ = self._run(True)
        coast = _module(0.0, "off")
        coast.lidar_target_assoc = False
        lidar = torch.full((1, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), 4.0)
        zero_bin = int(torch.argmin(coast._lidar_angles.abs()))
        lidar.view(1, _PERCEPTION.VBEAMS, _PERCEPTION.HBEAMS)[:, :, zero_bin] = 3.0
        pos, vel = torch.zeros(1, 3), torch.zeros(1, 3)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        for step, col in enumerate(self.COLUMNS):
            rgb, depth = _frame(col, 3.0)
            if step >= 3:
                rgb = torch.full_like(rgb, 0.15)
            coast.observe(
                rgb, depth, lidar.clone(), pos, vel, quat,
                torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
            )
        self.assertFalse(bool(torch.allclose(quiet.tracker.state, coast.tracker.state)))

    def test_default_off(self):
        self.assertFalse(_module(0.0, "off").lidar_silent_correct)


class LidarAssociationGate(unittest.TestCase):
    """A constant gate must decouple the association window from the track covariance."""

    SOURCE = (_TASK_DIR / "navrl_perception.py").read_text(encoding="utf-8")

    def test_gate_is_overridden_by_the_constant(self):
        body = self.SOURCE[
            self.SOURCE.index("def _associate_lidar_target") :
            self.SOURCE.index("def _target_features")
        ]
        self.assertIn("gate = (0.35 + 2.0 * pos_sigma).clamp(max=1.0)", body)
        self.assertIn("if self.lidar_assoc_gate_m > 0.0:", body)
        self.assertIn("gate = torch.full_like(gate, self.lidar_assoc_gate_m)", body)
        # The override must come after the covariance-scaled default, or it would be discarded.
        self.assertLess(
            body.index("gate = (0.35 + 2.0 * pos_sigma)"),
            body.index("gate = torch.full_like(gate, self.lidar_assoc_gate_m)"),
        )

    def test_covariance_scaled_gate_saturates_once_the_covariance_is_honest(self):
        """Why the constant is needed: the honest covariance pushes the gate to its cap."""
        probe = LidarRangeOnlyUpdate("test_default_keeps_the_current_behaviour")

        def gate_after(steps, range_only):
            lat, vert = probe._coast(steps, range_only=range_only)
            return lat, vert

        frozen_lat, _ = gate_after(20, False)
        honest_lat, _ = gate_after(20, True)
        # Gate uses the trace, but the lateral term alone already tells the story.
        self.assertLess(0.35 + 2.0 * frozen_lat, 1.0)
        self.assertGreater(0.35 + 2.0 * honest_lat, 1.0)


class LidarTargetAssociationSwitch(unittest.TestCase):
    """H2 probe: the LiDAR fallback must be switchable off without disturbing anything else."""

    COLUMNS = [30, 33, 36, 39, 42, 45]

    def _run(self, assoc, camera_blind=False):
        module = _module(0.0, "off")
        module.lidar_target_assoc = assoc
        lidar = torch.full((1, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), 4.0)
        zero_bin = int(torch.argmin(module._lidar_angles.abs()))
        lidar.view(1, _PERCEPTION.VBEAMS, _PERCEPTION.HBEAMS)[:, :, zero_bin] = 3.0
        pos, vel = torch.zeros(1, 3), torch.zeros(1, 3)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        diag = None
        for step, col in enumerate(self.COLUMNS):
            rgb, depth = _frame(col, 3.0)
            if camera_blind and step >= 3:
                # Simulate a dropped detection: nothing orange left in the frame.
                rgb = torch.full_like(rgb, 0.15)
            _, diag = module.observe(
                rgb, depth, lidar.clone(), pos, vel, quat,
                torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
            )
        return module, diag

    def test_disabling_removes_lidar_visibility(self):
        _, on = self._run(True, camera_blind=True)
        _, off = self._run(False, camera_blind=True)
        self.assertTrue(bool(on["lidar_visible"][0]))
        self.assertFalse(bool(off["lidar_visible"][0]))
        # With the camera blind and the fallback off, nothing reports the target at all.
        self.assertFalse(bool(off["visible"][0]))
        self.assertTrue(bool(on["visible"][0]))

    def test_default_is_current_behaviour(self):
        module = _module(0.0, "off")
        self.assertTrue(module.lidar_target_assoc)

    def test_switch_is_inert_while_the_camera_sees_the_target(self):
        """It must only change camera-missed frames, so a fully visible run is untouched."""
        on, diag_on = self._run(True)
        off, diag_off = self._run(False)
        self.assertTrue(bool(diag_on["camera_visible"][0]))
        self.assertTrue(bool(torch.equal(diag_on["camera_visible"], diag_off["camera_visible"])))
        self.assertTrue(bool(torch.allclose(on.tracker.state, off.tracker.state)))


class PosePremiseSensitivity(unittest.TestCase):
    """검증 3: clock offset / interpolation / odometry noise on the P3 capture pose."""

    def _drive(self, module, steps=6, speed=2.0):
        poses = []
        for k in range(steps):
            pos = torch.tensor([[speed * k * DT, 0.0, 1.0]])
            quat = _yaw_quat(0.3 * k * DT)
            poses.append((pos, quat))
            module._apply_detection_latency(
                torch.zeros(1, 3), torch.zeros(1), torch.zeros(1),
                torch.ones(1, dtype=torch.bool), torch.ones(1),
                torch.zeros(1, module.height, module.width, dtype=torch.bool),
                drone_pos_w=pos, vehicle_quat=quat,
            )
        return poses

    def _module_with(self, **kw):
        module = _module(TAU, "off", ego_motion_fix=True)
        for key, value in kw.items():
            setattr(module, key, value)
        module._pose_premise_active = (
            module.pose_clock_offset_s != 0.0
            or module.pose_noise_pos_m > 0.0
            or module.pose_noise_yaw_deg > 0.0
        )
        return module

    def test_zero_knobs_publish_the_exact_capture_pose(self):
        exact = _module(TAU, "off", ego_motion_fix=True)
        poses = self._drive(exact)
        pos, quat = exact._latency_delayed_pose
        self.assertTrue(bool(torch.equal(pos, poses[-2][0])))
        self.assertTrue(bool(torch.equal(quat, poses[-2][1])))

    def test_offset_plus_tau_reproduces_the_current_pose(self):
        """The built-in anchor: pose stamped tau late == the naive pre-P3 transform."""
        module = self._module_with(pose_clock_offset_s=TAU)
        poses = self._drive(module)
        pos, quat = module._latency_delayed_pose
        self.assertTrue(bool(torch.allclose(pos, poses[-1][0], atol=1e-6)))
        self.assertTrue(bool(torch.allclose(quat, poses[-1][1], atol=1e-6)))

    def test_fractional_offset_interpolates_between_samples(self):
        module = self._module_with(pose_clock_offset_s=TAU * 0.5)
        poses = self._drive(module)
        pos, _ = module._latency_delayed_pose
        midpoint = 0.5 * (poses[-2][0] + poses[-1][0])
        self.assertTrue(bool(torch.allclose(pos, midpoint, atol=1e-6)))

    def test_negative_offset_reads_an_older_pose(self):
        module = self._module_with(pose_clock_offset_s=-TAU)
        poses = self._drive(module)
        pos, _ = module._latency_delayed_pose
        self.assertTrue(bool(torch.allclose(pos, poses[-3][0], atol=1e-6)))

    def test_yaw_noise_keeps_unit_norm_and_zero_noise_is_silent(self):
        module = self._module_with(pose_noise_yaw_deg=5.0)
        self._drive(module)
        _, quat = module._latency_delayed_pose
        self.assertAlmostEqual(float(quat.norm(dim=1)), 1.0, places=5)
        silent = self._module_with(pose_noise_pos_m=0.0, pose_noise_yaw_deg=0.0)
        self.assertFalse(silent._pose_premise_active)

    def test_pose_noise_does_not_advance_global_rng(self):
        """A pose-noise arm must not change simulator randomness in the same process."""
        module = self._module_with(pose_noise_pos_m=0.03, pose_noise_yaw_deg=2.0)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        self._drive(module)
        after = torch.random.get_rng_state()
        self.assertTrue(bool(torch.equal(before, after)))

    def test_pose_noise_stream_is_repeatable_and_seeded(self):
        a = self._module_with(pose_noise_pos_m=0.03, pose_noise_yaw_deg=2.0)
        b = self._module_with(pose_noise_pos_m=0.03, pose_noise_yaw_deg=2.0)
        self._drive(a)
        self._drive(b)
        for left, right in zip(a._latency_delayed_pose, b._latency_delayed_pose):
            self.assertTrue(bool(torch.equal(left, right)))


class LatencyEgoMotionFix(unittest.TestCase):
    """P3: a delayed measurement must be lifted to world with the pose it was TAKEN at."""

    TRUE_WORLD = torch.tensor([[5.0, 1.0, 1.0]])
    STEPS = 12

    def _drive(self, tau_steps, speed=2.33, yaw_rate=0.8):
        """Observer translating and yawing while a world-STATIC target is measured."""
        module = _module(tau_steps * DT, "off", ego_motion_fix=True)
        naive_err = corrected_err = None
        for k in range(self.STEPS):
            drone_pos = torch.tensor([[speed * k * DT, 0.0, 1.0]])
            quat = _yaw_quat(yaw_rate * k * DT)
            # What the camera would report right now, in the CURRENT vehicle frame.
            meas_vehicle = _PERCEPTION._quat_rotate_inverse_xyzw(
                quat, self.TRUE_WORLD - drone_pos
            )
            delayed = module._apply_detection_latency(
                meas_vehicle,
                torch.zeros(1),
                torch.zeros(1),
                torch.ones(1, dtype=torch.bool),
                torch.ones(1),
                torch.zeros(1, module.height, module.width, dtype=torch.bool),
                drone_pos_w=drone_pos,
                vehicle_quat=quat,
            )
            delayed_meas = delayed[0]
            if module._latency_delayed_pose is None:
                continue
            past_pos, past_quat = module._latency_delayed_pose
            naive = drone_pos + _PERCEPTION._quat_rotate_xyzw(quat, delayed_meas)
            corrected = past_pos + _PERCEPTION._quat_rotate_xyzw(past_quat, delayed_meas)
            naive_err = float((naive - self.TRUE_WORLD).norm())
            corrected_err = float((corrected - self.TRUE_WORLD).norm())
        return naive_err, corrected_err

    def test_capture_time_pose_makes_a_static_target_exact(self):
        naive_err, corrected_err = self._drive(tau_steps=1)
        # A world-static target has no motion lag left to explain: whatever error survives is
        # purely the observer's own motion, and P3 must remove all of it.
        self.assertLess(corrected_err, 1e-5)
        # And that error is large -- bigger than the <=0.15 m target lag P0 was aimed at.
        self.assertGreater(naive_err, 0.2)

    def test_two_step_latency_also_exact(self):
        naive_err, corrected_err = self._drive(tau_steps=2)
        self.assertLess(corrected_err, 1e-5)
        self.assertGreater(naive_err, 0.4)

    def test_pose_buffer_returns_the_pose_from_tau_steps_ago(self):
        module = _module(TAU, "off", ego_motion_fix=True)
        poses = []
        for k in range(6):
            drone_pos = torch.tensor([[float(k), 0.0, 1.0]])
            quat = _yaw_quat(0.1 * k)
            poses.append((drone_pos, quat))
            delayed = module._apply_detection_latency(
                torch.zeros(1, 3),
                torch.zeros(1),
                torch.zeros(1),
                torch.ones(1, dtype=torch.bool),
                torch.ones(1),
                torch.zeros(1, module.height, module.width, dtype=torch.bool),
                drone_pos_w=drone_pos,
                vehicle_quat=quat,
            )
            past_pos, past_quat = module._latency_delayed_pose
            if k == 0:
                # The buffer has not filled yet, so the slot holds its init pose -- harmless
                # only because the detection it belongs to is reported as NOT visible.
                self.assertFalse(bool(delayed[3][0]))
                continue
            self.assertTrue(bool(delayed[3][0]))
            expected_pos, expected_quat = poses[k - 1]
            self.assertTrue(bool(torch.allclose(past_pos, expected_pos)))
            self.assertTrue(bool(torch.allclose(past_quat, expected_quat)))

    def test_zero_latency_publishes_no_pose(self):
        """No latency, no correction: observe() must fall back to the current pose."""
        module = _module(0.0, "off", ego_motion_fix=True)
        module._apply_detection_latency(
            torch.zeros(1, 3),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
            torch.zeros(1, module.height, module.width, dtype=torch.bool),
            drone_pos_w=torch.zeros(1, 3),
            vehicle_quat=_yaw_quat(0.0),
        )
        self.assertIsNone(module._latency_delayed_pose)

    def test_observe_uses_the_buffered_pose_only_when_enabled(self):
        lidar = torch.full((1, _PERCEPTION.VBEAMS * _PERCEPTION.HBEAMS), 3.0)
        columns = [30, 33, 36, 39, 42, 45]
        off = _module(TAU, "off", ego_motion_fix=False)
        on = _module(TAU, "off", ego_motion_fix=True)
        pos, vel = torch.zeros(1, 3), torch.zeros(1, 3)
        for step, col in enumerate(columns):
            rgb, depth = _frame(col, 3.0)
            # Moving, yawing observer: the two modes must diverge.
            drone_pos = torch.tensor([[2.33 * step * DT, 0.0, 1.0]])
            quat = _yaw_quat(0.8 * step * DT)
            for module in (off, on):
                module.observe(
                    rgb, depth, lidar.clone(), drone_pos, vel, quat,
                    torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
                )
        self.assertFalse(bool(torch.allclose(off.tracker.state, on.tracker.state)))
        # A stationary observer leaves nothing for P3 to compensate.
        still_off = _module(TAU, "off", ego_motion_fix=False)
        still_on = _module(TAU, "off", ego_motion_fix=True)
        for col in columns:
            rgb, depth = _frame(col, 3.0)
            for module in (still_off, still_on):
                module.observe(
                    rgb, depth, lidar.clone(), pos, vel, _yaw_quat(0.0),
                    torch.zeros(1), torch.zeros(1, 4), 2.0, 1.0, training=False,
                )
        self.assertTrue(bool(torch.allclose(still_off.tracker.state, still_on.tracker.state)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
