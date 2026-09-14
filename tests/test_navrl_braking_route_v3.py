"""CPU contracts for the fresh braking-aware route v3 lineage."""

import ast
import inspect
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


os.environ.pop("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", None)

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEO = _load("braking_v3_geometry", "aerial_gym/task/navrl_task/target_route_geometry.py")
ROUTE = _load("braking_v3_route", "aerial_gym/task/navrl_task/target_route_planner.py")
MOTION = _load("braking_v3_motion", "aerial_gym/task/navrl_task/target_motion.py")
GATE = _load("braking_v3_gate", "tools/verify_navrl_braking_route_v3_gate.py")

SPEC = GEO.SoftEnvelopeSpec(wall_margin_m=0.5, boundary_reserve_m=0.75, tracking_margin_m=0.45)
BRAKE_SPEEDS = (0.3, 0.6, 0.9, 1.2, 1.5)
BRAKE_DISTS = (0.05, 0.12, 0.22, 0.35, 0.50)
FORBIDDEN_SYNC = (".cpu(", ".item(", ".numpy(", "bool(")


def _source_has_no_sync(fn):
    source = inspect.getsource(fn)
    for token in FORBIDDEN_SYNC:
        if token in source:
            return False, token, source
    return True, "", source


class BrakingRouteV3GeometryTest(unittest.TestCase):
    def test_astar_and_local_wall_reserve_match(self):
        arena_lo = np.array([0.0, 0.0])
        arena_hi = np.array([40.0, 40.0])
        support = np.array([0.2068816087, 0.2068816087])
        half = np.array([[0.3, 0.3], [0.5, 0.4]])
        lo, hi, infl = GEO.numpy_soft_envelope(arena_lo, arena_hi, half, support, SPEC)
        np.testing.assert_allclose(lo, arena_lo + 1.25 + support)
        np.testing.assert_allclose(hi, arena_hi - 1.25 - support)
        np.testing.assert_allclose(infl, half + support[None, :] + 0.45)
        planner = ROUTE.DeterministicAStarRoutePlanner(
            ROUTE.RoutePlannerConfig(boundary_margin_m=1.25, tracking_margin_m=0.45)
        )
        result = planner.plan(
            np.array([5.0, 5.0]), np.array([8.0, 5.0]), np.array([[20.0, 20.0]]),
            np.array([[0.3, 0.3]]), arena_lo, arena_hi, support,
        )
        self.assertTrue(result.valid, result.status)
        for a, b in zip(result.waypoints_xy[:-1], result.waypoints_xy[1:]):
            self.assertTrue(GEO.numpy_segments_soft_safe(a, b, lo, hi, np.array([[20.0, 20.0]]), infl[:1]))

    def test_rounded_corner_passes_but_exact_aabb_rejects(self):
        # Same corner-gap fixture as recovery-v2: the point sits outside the raw AABB but inside
        # the axis-aligned tracking inflation.  Rounded Euclidean clearance still exceeds 0.45 m.
        p0 = np.array([6.6, 6.6])
        p1 = np.array([7.0, 6.6])
        lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
        bars = np.array([[5.0, 5.0]])
        raw_half = np.array([[1.2, 1.2]])
        inflated = raw_half + 0.45
        rounded = np.linalg.norm(np.maximum(np.abs(p0 - bars[0]) - raw_half[0], 0.0))
        self.assertGreater(rounded, 0.45)
        self.assertFalse(GEO.numpy_segments_soft_safe(p0, p1, lo, hi, bars, inflated))
        rounded_ok = MOTION.bounded_drone_target_step(
            old_xy=torch.tensor([[6.6, 6.6]]),
            current_velocity=torch.zeros((1, 2)),
            desired_velocity=torch.tensor([[1.0, 0.0]]),
            speed_limit=torch.tensor([0.0]), dt=0.1,
            bars_xy=torch.tensor([[[5.0, 5.0]]]),
            lo=torch.tensor([[0.0, 0.0]]), hi=torch.tensor([[10.0, 10.0]]),
            clearance=0.45, turn_sign=torch.ones(1),
            max_accel=torch.tensor([4.0]), max_turn_rate=torch.tensor([2.6]),
            lookahead_s=0.1, bars_half_extents_xy=torch.tensor([[[1.2, 1.2]]]),
        )[3]
        self.assertTrue(bool(rounded_ok.item()))
        _pos, _vel, accepted, _pre, cert = MOTION.braking_aware_route_step(
            old_xy=torch.tensor([[6.6, 6.6]]),
            current_velocity=torch.zeros((1, 2)),
            desired_velocity=torch.tensor([[1.0, 0.0]]),
            speed_limit=torch.tensor([0.0]), dt=0.1,
            bars_xy=torch.tensor([[[5.0, 5.0]]]),
            admissible_lo=torch.tensor([[0.0, 0.0]]),
            admissible_hi=torch.tensor([[10.0, 10.0]]),
            inflated_bar_half_xy=torch.tensor([[[1.65, 1.65]]]),
            turn_sign=torch.ones(1), max_accel=torch.tensor([4.0]),
            max_turn_rate=torch.tensor([2.6]), lookahead_s=0.1,
            brake_speed_knots=BRAKE_SPEEDS, brake_distance_knots=BRAKE_DISTS,
            lateral_tube_m=0.02,
        )
        self.assertFalse(bool(accepted[0] and cert["progress"][0]))

    def test_numpy_torch_random_geometry_parity(self):
        rng = np.random.default_rng(829)
        lo = np.array([-8.0, -8.0])
        hi = np.array([8.0, 8.0])
        for _ in range(40):
            p0 = rng.uniform(-6.0, 6.0, size=2)
            p1 = p0 + rng.uniform(-1.5, 1.5, size=2)
            bars = rng.uniform(-5.0, 5.0, size=(5, 2))
            half = rng.uniform(0.05, 0.6, size=(5, 2))
            numpy_safe = GEO.numpy_segments_soft_safe(p0, p1, lo, hi, bars, half)
            torch_safe = GEO.torch_segments_soft_safe(
                torch.tensor(p0, dtype=torch.float64).view(1, 1, 2),
                torch.tensor(p1, dtype=torch.float64).view(1, 1, 2),
                torch.tensor(lo, dtype=torch.float64).view(1, 2),
                torch.tensor(hi, dtype=torch.float64).view(1, 2),
                torch.tensor(bars, dtype=torch.float64).view(1, 5, 2),
                torch.tensor(half, dtype=torch.float64).view(1, 5, 2),
            )
            self.assertEqual(bool(numpy_safe), bool(torch_safe[0, 0].item()))

    def test_nan_and_malformed_geometry_fail_closed(self):
        lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
        self.assertFalse(
            GEO.numpy_segments_soft_safe(
                [0.0, 0.0], [1.0, 0.0], lo, hi, [[float("nan"), 1.0]], [[0.2, 0.2]]
            )
        )
        with self.assertRaises(ValueError):
            GEO.numpy_soft_envelope(lo, hi, [[float("nan"), 0.2]], [0.2, 0.2], SPEC)
        bad = GEO.torch_segments_soft_safe(
            torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]]),
            torch.tensor([[0.0, 0.0]]), torch.tensor([[10.0, 10.0]]),
            torch.tensor([[[float("nan"), 1.0]]]), torch.tensor([[[0.2, 0.2]]]),
        )
        self.assertFalse(bool(bad[0, 0]))

    def test_ceiling_lookup_is_not_interpolation(self):
        distance, valid = GEO.numpy_ceiling_stop_distance(
            np.array([0.0, 0.3, 0.31, 1.5, 1.51]), BRAKE_SPEEDS, BRAKE_DISTS
        )
        self.assertTrue(valid[0] and valid[1] and valid[2] and valid[3])
        self.assertFalse(bool(valid[4]))
        self.assertAlmostEqual(float(distance[1]), 0.05)
        self.assertAlmostEqual(float(distance[2]), 0.12)
        gpu_dist, gpu_valid = GEO.ceiling_stop_distance(
            torch.tensor([0.31, 1.51], dtype=torch.float32), BRAKE_SPEEDS, BRAKE_DISTS
        )
        self.assertAlmostEqual(float(gpu_dist[0]), 0.12, places=5)
        self.assertFalse(bool(gpu_valid[1]))

    def test_receipt_missing_tamper_and_overrange_fail_closed(self):
        with self.assertRaises(ValueError):
            GEO.validate_brake_lookup((), ())
        with self.assertRaises(ValueError):
            GEO.validate_brake_lookup((0.3, 0.2), (0.1, 0.2))
        with self.assertRaises(ValueError):
            GEO.validate_brake_lookup((0.3, 0.6), (0.2, 0.1))
        with self.assertRaises(ValueError):
            GEO.validate_brake_lookup((0.3, float("nan")), (0.1, 0.2))
        _, valid = GEO.numpy_ceiling_stop_distance(2.0, BRAKE_SPEEDS, BRAKE_DISTS)
        self.assertFalse(bool(valid))


class BrakingAwareRouteStepTest(unittest.TestCase):
    def _step(self, **kwargs):
        defaults = dict(
            old_xy=torch.tensor([[0.0, 0.0]]),
            current_velocity=torch.zeros(1, 2),
            desired_velocity=torch.tensor([[1.0, 0.0]]),
            speed_limit=torch.tensor([1.0]),
            dt=0.1,
            bars_xy=torch.empty(1, 0, 2),
            admissible_lo=torch.tensor([[-10.0, -10.0]]),
            admissible_hi=torch.tensor([[10.0, 10.0]]),
            inflated_bar_half_xy=torch.empty(1, 0, 2),
            turn_sign=torch.ones(1),
            max_accel=torch.tensor([8.0]),
            max_turn_rate=torch.tensor([2.5]),
            lookahead_s=1.0,
            brake_speed_knots=BRAKE_SPEEDS,
            brake_distance_knots=BRAKE_DISTS,
            lateral_tube_m=0.02,
        )
        defaults.update(kwargs)
        return MOTION.braking_aware_route_step(**defaults)

    def test_safe_prefix_only_candidate_is_rejected(self):
        # Narrow x-corridor: turning is immediately wall-unsafe, so the only remaining progress
        # candidates run into a bar after a safe first sample.  v3 must not take that prefix.
        bars = torch.tensor([[[0.20, 0.0]]])
        half = torch.tensor([[[0.05, 1.0]]])
        _pos, _vel, accepted, prebrake, cert = self._step(
            current_velocity=torch.tensor([[1.0, 0.0]]),
            bars_xy=bars, inflated_bar_half_xy=half, lookahead_s=1.0,
            admissible_lo=torch.tensor([[-10.0, -0.02]]),
            admissible_hi=torch.tensor([[10.0, 0.02]]),
        )
        self.assertFalse(bool(cert["progress"][0]))
        self.assertFalse(bool(accepted[0] and cert["progress"][0]))
        self.assertFalse(bool(cert["progress"][0] and not cert["full_horizon_safe"][0]))
        v1_pos, _v1_vel, _steered, v1_feasible = MOTION.bounded_drone_target_step(
            old_xy=torch.tensor([[0.0, 0.0]]),
            current_velocity=torch.tensor([[1.0, 0.0]]),
            desired_velocity=torch.tensor([[1.0, 0.0]]),
            speed_limit=torch.tensor([1.0]), dt=0.1,
            bars_xy=bars,
            lo=torch.tensor([[-10.0, -0.02]]), hi=torch.tensor([[10.0, 0.02]]),
            clearance=0.0, turn_sign=torch.ones(1),
            max_accel=torch.tensor([8.0]), max_turn_rate=torch.tensor([2.5]),
            lookahead_s=1.0, bars_half_extents_xy=half, exact_aabb_clearance=True,
        )
        self.assertGreater(float(v1_pos[0, 0]), 0.0)
        self.assertIn("longest safe prefix", inspect.getsource(MOTION.bounded_drone_target_step))

    def test_unsafe_terminal_stop_tail_is_rejected(self):
        # One-step horizon stays inside hi=0.15, but the canonical stop from residual speed
        # overshoots.  No progress command may be accepted.
        _pos, _vel, accepted, _pre, cert = self._step(
            old_xy=torch.tensor([[0.0, 0.0]]),
            current_velocity=torch.tensor([[1.2, 0.0]]),
            speed_limit=torch.tensor([1.2]),
            lookahead_s=0.1,
            admissible_hi=torch.tensor([[0.15, 5.0]]),
            admissible_lo=torch.tensor([[-5.0, -5.0]]),
        )
        self.assertFalse(bool(accepted[0] and cert["progress"][0]))
        position = torch.tensor([[[0.0, 0.0]]])
        velocity = torch.tensor([[[1.2, 0.0]]])
        safe, _stop, distance = GEO.terminal_stop_certificate(
            position, velocity,
            torch.tensor([[-5.0, -5.0]]), torch.tensor([[0.15, 5.0]]),
            torch.empty(1, 0, 2), torch.empty(1, 0, 2),
            BRAKE_SPEEDS, BRAKE_DISTS, 0.0,
        )
        self.assertGreater(float(distance[0, 0]), 0.15)
        self.assertFalse(bool(safe[0, 0]))

    def test_safe_terminal_stop_tail_is_accepted(self):
        _pos, vel, accepted, _pre, cert = self._step(
            current_velocity=torch.tensor([[0.6, 0.0]]),
            speed_limit=torch.tensor([0.6]),
        )
        self.assertTrue(bool(accepted[0]))
        self.assertTrue(bool(cert["terminal_stop_safe"][0]))
        self.assertGreater(float(vel[0, 0]), 0.0)

    def test_certified_first_step_preserves_recursive_certificate_in_ideal_model(self):
        """A certified receding-horizon step must remain certifiable under its own model.

        This is deliberately an obstacle-rich randomized property fixture.  It does not claim
        PhysX tracking equivalence; it separates a controller-model bug from the physical
        tracking loss measured by the GPU gate.
        """
        torch.manual_seed(839)
        n, bars_n = 512, 8
        old_xy = torch.rand(n, 2) * 12.0 - 6.0
        current_velocity = torch.rand(n, 2) * 2.0 - 1.0
        current_velocity = (
            current_velocity
            / current_velocity.norm(dim=1, keepdim=True).clamp(min=1e-6)
            * (torch.rand(n, 1) * 1.25)
        )
        desired_velocity = torch.rand(n, 2) * 2.0 - 1.0
        desired_velocity = (
            desired_velocity
            / desired_velocity.norm(dim=1, keepdim=True).clamp(min=1e-6)
            * 1.25
        )
        bars = torch.rand(n, bars_n, 2) * 14.0 - 7.0
        inflated_half = torch.rand(n, bars_n, 2) * 0.35 + 0.35
        lo = torch.full((n, 2), -8.0)
        hi = torch.full((n, 2), 8.0)
        turn_sign = torch.where(torch.rand(n) < 0.5, -torch.ones(n), torch.ones(n))
        kwargs = dict(
            speed_limit=torch.full((n,), 1.25), dt=0.1,
            bars_xy=bars, admissible_lo=lo, admissible_hi=hi,
            inflated_bar_half_xy=inflated_half, turn_sign=turn_sign,
            max_accel=torch.full((n,), 4.0), max_turn_rate=torch.full((n,), 2.6),
            lookahead_s=1.0, brake_speed_knots=BRAKE_SPEEDS,
            brake_distance_knots=BRAKE_DISTS, lateral_tube_m=0.02,
        )
        current_safe = GEO.torch_segments_soft_safe(
            old_xy[:, None, :], old_xy[:, None, :], lo, hi, bars, inflated_half
        )[:, 0]
        next_xy, next_velocity, accepted, _prebrake, certificate = (
            MOTION.braking_aware_route_step(
                old_xy, current_velocity, desired_velocity, **kwargs
            )
        )
        _next2, _velocity2, accepted_next, _prebrake2, _certificate2 = (
            MOTION.braking_aware_route_step(
                next_xy, next_velocity, desired_velocity, **kwargs
            )
        )
        certified_rows = current_safe & accepted
        self.assertGreater(int(certified_rows.sum()), 450)
        self.assertFalse(bool((certified_rows & ~accepted_next).any()))
        self.assertFalse(bool((
            certified_rows
            & ~(certificate["full_horizon_safe"] & certificate["terminal_stop_safe"])
        ).any()))

    def test_gpu_hot_path_has_no_cpu_sync(self):
        for fn in (
            MOTION.braking_aware_route_step,
            GEO.ceiling_stop_distance,
            GEO.terminal_stop_certificate,
            GEO.torch_segments_soft_safe,
            ROUTE.BatchedTargetRouteManager.braking_v3_follow_reference,
        ):
            ok, token, _ = _source_has_no_sync(fn)
            self.assertTrue(ok, "%s contains %s" % (fn.__qualname__, token))
        step_source = inspect.getsource(MOTION.braking_aware_route_step)
        self.assertNotIn("longest safe prefix", step_source)
        self.assertNotIn("trapped_score", step_source)


class BrakingV3ManagerTest(unittest.TestCase):
    def _manager(self, **kwargs):
        config = ROUTE.RoutePlannerConfig(
            resolution_m=0.25, tracking_margin_m=0.45, boundary_margin_m=1.25,
            max_waypoints=16, replan_cooldown_steps=1,
        )
        return ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), config, braking_v3_enabled=True, **kwargs
        )

    def test_v1_diagnostics_shape_unchanged(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=False
        )
        self.assertEqual(manager.diagnostics()["mode"], "global_astar_v1")
        self.assertNotIn("v3_diagnostics", manager.diagnostics())
        self.assertEqual(ROUTE.TARGET_ROUTE_MODES, ("off", "global_astar_v1"))
        self.assertNotIn(ROUTE.TARGET_ROUTE_MODE_BRAKING_V3, ROUTE.TARGET_ROUTE_MODES)

    def test_unsafe_start_does_not_call_astar(self):
        manager = self._manager()
        calls = []
        original = manager.planner.plan

        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        manager.planner.plan = wrapped
        env_ids = torch.tensor([0])
        start = torch.tensor([[5.0, 5.0]])
        goal = torch.tensor([[8.0, 5.0]])
        bars = torch.tensor([[[5.0, 5.0]]])
        half = torch.tensor([[[1.0, 1.0]]])
        lo = torch.tensor([[0.0, 0.0]])
        hi = torch.tensor([[10.0, 10.0]])
        support = torch.tensor([[0.2, 0.2]])
        manager.plan_idx(
            env_ids, start, goal, bars, half, lo, hi, support, 0,
            is_replan=True, plan_cause="runtime_replan",
        )
        self.assertEqual(calls, [])
        self.assertEqual(int(manager.status_code[0]), manager.STATUS_CODES["unsafe_start"])
        self.assertEqual(int(manager.v3_runtime_replan_unsafe_start_count.item()), 1)
        v1 = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), braking_v3_enabled=False
        )
        v1_calls = []
        v1_orig = v1.planner.plan

        def v1_wrap(*args, **kwargs):
            v1_calls.append(1)
            return v1_orig(*args, **kwargs)

        v1.planner.plan = v1_wrap
        v1.plan_idx(env_ids, start, goal, bars, half, lo, hi, support, 0, is_replan=True)
        self.assertEqual(v1_calls, [1])

    def test_nan_start_skips_astar_fail_closed(self):
        manager = self._manager()
        calls = []
        original = manager.planner.plan

        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        manager.planner.plan = wrapped
        manager.plan_idx(
            torch.tensor([0]),
            torch.tensor([[float("nan"), 5.0]]),
            torch.tensor([[8.0, 5.0]]),
            torch.zeros(1, 0, 2),
            torch.zeros(1, 0, 2),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[10.0, 10.0]]),
            torch.tensor([[0.2, 0.2]]),
            0, is_replan=True, plan_cause="runtime_replan",
        )
        self.assertEqual(calls, [])
        self.assertEqual(int(manager.status_code[0]), manager.STATUS_CODES["unsafe_start"])

    def test_recovery_and_v1_contracts_are_not_rewritten(self):
        planner = (ROOT / "aerial_gym/task/navrl_task/target_route_planner.py").read_text()
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("two-envelope recovery requires a positive target-specific zero-command", task)
        self.assertIn("recovery mode requires the common braking-probe receipt validator", task)
        self.assertIn("scalar p05/p95 values cannot arm recovery", task)
        self.assertIn("TARGET_ROUTE_MODES = (TARGET_ROUTE_MODE_OFF, TARGET_ROUTE_MODE_GLOBAL_ASTAR)", planner)
        self.assertIn("def velocity_reference(self, position_xy, speed, reach_m: float):", planner)

    def test_stop_turn_go_and_corner_prebrake(self):
        manager = self._manager()
        manager.valid[0] = True
        manager.length[0] = 2
        manager.cursor[0] = 0
        manager.segment_start[0] = torch.tensor([0.0, 0.0])
        manager.waypoints[0, 0] = torch.tensor([1.0, 0.0])
        manager.waypoints[0, 1] = torch.tensor([1.0, 1.0])
        manager.handoff_clearance[0, 0] = 10.0
        manager.handoff_clearance[0, 1] = 10.0
        lo = torch.tensor([[-5.0, -5.0]])
        hi = torch.tensor([[5.0, 5.0]])
        # Block the corner fillet so the follower must stop-turn-go rather than skip.
        bars = torch.tensor([[[1.0, 0.5]]])
        half = torch.tensor([[[0.15, 0.15]]])
        _vel, speed_limit, _active, _complete, stop_turn, _corner, _fillet = (
            manager.braking_v3_follow_reference(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.2, 0.0]]),
                torch.tensor([1.2]),
                0.05,
                lo, hi, bars, half,
                torch.tensor([1.0]),
                0.10,
                BRAKE_SPEEDS, BRAKE_DISTS, 0.1,
            )
        )
        self.assertTrue(bool(stop_turn[0]))
        self.assertEqual(float(speed_limit[0]), 0.0)
        _vel2, speed2, _a, _c, stop2, _corner2, _f = manager.braking_v3_follow_reference(
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([1.2]),
            0.05,
            lo, hi, bars, half,
            torch.tensor([1.0]),
            0.10,
            BRAKE_SPEEDS, BRAKE_DISTS, 0.1,
        )
        self.assertFalse(bool(stop2[0]))
        self.assertGreater(float(speed2[0]), 0.0)

    def test_pre_corner_deceleration_uses_ceiling_stop_distance(self):
        manager = self._manager()
        manager.valid[0] = True
        manager.length[0] = 2
        manager.cursor[0] = 0
        manager.segment_start[0] = torch.tensor([0.0, 0.0])
        manager.waypoints[0, 0] = torch.tensor([2.0, 0.0])
        manager.waypoints[0, 1] = torch.tensor([2.0, 2.0])
        manager.handoff_clearance[0, 0] = 10.0
        manager.handoff_clearance[0, 1] = 10.0
        _vel, speed_limit, _active, _complete, stop_turn, corner, _fillet = (
            manager.braking_v3_follow_reference(
                torch.tensor([[1.70, 0.0]]),
                torch.tensor([[1.2, 0.0]]),
                torch.tensor([1.2]),
                0.05,
                torch.tensor([[-5.0, -5.0]]),
                torch.tensor([[5.0, 5.0]]),
                torch.empty(1, 0, 2),
                torch.empty(1, 0, 2),
                torch.tensor([1.0]),
                0.10,
                BRAKE_SPEEDS, BRAKE_DISTS, 0.1,
            )
        )
        self.assertFalse(bool(stop_turn[0]))
        self.assertTrue(bool(corner[0]))
        self.assertEqual(float(speed_limit[0]), 0.0)

    def test_v3_gate_fields_are_present(self):
        diagnostics = self._manager().v3_gate_diagnostics()
        for name in GATE.V3_DIAGNOSTIC_FIELDS:
            self.assertIn(name, diagnostics)


class BrakingV3GateVerifierTest(unittest.TestCase):
    def test_frozen_grid_and_prereg_sha(self):
        self.assertEqual(GATE.STAGES["pilot"]["seed"], 829)
        self.assertEqual(GATE.STAGES["confirmatory"]["seed"], 839)
        self.assertEqual(list(GATE.STAGES["pilot"]["densities"]), [70])
        self.assertEqual(list(GATE.STAGES["confirmatory"]["densities"]), [70, 115, 160, 205])
        self.assertEqual(GATE.PREREG_SHA256, GATE.sha256_file(GATE.PREREG))
        self.assertEqual(GATE.ROUTE_ARMS, ("off", "global_astar_braking_v3"))
        self.assertNotIn(300, GATE.STAGES["confirmatory"]["densities"])


def _load_task_method(name):
    tree = ast.parse(
        (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(encoding="utf-8")
    )
    task_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
    )
    fn_node = next(
        node for node in task_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[fn_node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), "navrl_task.py", "exec"), namespace)
    return namespace[name]


class BrakingV3MatchedSpawnTest(unittest.TestCase):
    def test_samplers_do_not_branch_on_v3(self):
        tree = ast.parse(
            (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(encoding="utf-8")
        )
        task_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
        )
        forbidden = {
            "_target_route_braking_v3_enabled",
            "torch_soft_envelope",
            "torch_segments_soft_safe",
        }
        for name in ("_sample_general_target", "_sample_waypoints"):
            fn_node = next(
                node for node in task_class.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            used = {node.id for node in ast.walk(fn_node) if isinstance(node, ast.Name)}
            self.assertTrue(forbidden.isdisjoint(used), used & forbidden)

    def test_waypoint_samples_ignore_v3_flag(self):
        sample = _load_task_method("_sample_waypoints")
        b_min = torch.zeros((4, 3))
        b_max = torch.tensor([40.0, 40.0, 3.0]).repeat(4, 1)
        env_ids = torch.arange(4)

        def draw(flag):
            torch.manual_seed(829)
            stub = types.SimpleNamespace(
                device="cpu",
                _physical_target=True,
                _target_route_braking_v3_enabled=flag,
                cur=types.SimpleNamespace(wall_margin=0.5),
                tm=types.SimpleNamespace(physical_boundary_margin=0.75),
                obs_dict={"env_bounds_min": b_min, "env_bounds_max": b_max},
            )
            return sample(stub, env_ids)

        self.assertTrue(torch.equal(draw(False), draw(True)))

    def test_general_target_samples_ignore_v3_flag(self):
        sample = _load_task_method("_sample_general_target")
        env_ids = torch.arange(8)
        start = torch.full((8, 3), 20.0)
        b_min = torch.zeros((8, 3))
        b_max = torch.tensor([40.0, 40.0, 3.0]).repeat(8, 1)
        bars = torch.tensor([[[10.0, 10.0]]]).expand(8, 1, 2).clone()
        half = torch.tensor([[[0.3, 0.3]]]).expand(8, 1, 2).clone()

        def draw(flag):
            torch.manual_seed(829)
            stub = types.SimpleNamespace(
                device="cpu",
                _physical_target=True,
                _target_route_braking_v3_enabled=flag,
                _target_dynamics="physical",
                _target_speed_max=lambda: 1.25,
                _general_goal_distance_bounds=lambda: (2.0, 15.0, None),
                _target_spawn_center_clearance=lambda: 1.0,
                n_bars_active=1,
                cur=types.SimpleNamespace(wall_margin=0.5),
                tm=types.SimpleNamespace(
                    physical_boundary_margin=0.75,
                    physical_tracking_margin=0.45,
                ),
                task_config=types.SimpleNamespace(flight_altitude=1.5),
            )
            return sample(stub, env_ids, start, b_min, b_max, bars, half)

        off = draw(False)
        routed = draw(True)
        self.assertTrue(torch.equal(off, routed))
        self.assertTrue(torch.isfinite(off).all())

    def test_spawn_identity_is_stable_across_preregistered_and_adversarial_seeds(self):
        general = _load_task_method("_sample_general_target")
        waypoints = _load_task_method("_sample_waypoints")
        env_ids = torch.arange(32)
        b_min = torch.zeros((32, 3))
        b_max = torch.tensor([40.0, 40.0, 3.0]).repeat(32, 1)
        start = torch.full((32, 3), 20.0)
        bars = torch.tensor([[[10.0, 10.0], [30.0, 30.0]]]).expand(32, 2, 2).clone()
        half = torch.tensor([[[0.3, 0.3], [0.4, 0.5]]]).expand(32, 2, 2).clone()

        def stub(flag):
            return types.SimpleNamespace(
                device="cpu", _physical_target=True,
                _target_route_braking_v3_enabled=flag, _target_dynamics="physical",
                _target_speed_max=lambda: 1.25,
                _general_goal_distance_bounds=lambda: (2.0, 15.0, None),
                _target_spawn_center_clearance=lambda: 1.0, n_bars_active=2,
                cur=types.SimpleNamespace(wall_margin=0.5),
                tm=types.SimpleNamespace(
                    physical_boundary_margin=0.75, physical_tracking_margin=0.45,
                ),
                task_config=types.SimpleNamespace(flight_altitude=1.5),
                obs_dict={"env_bounds_min": b_min, "env_bounds_max": b_max},
            )

        for seed in (1, 59, 367, 827, 829, 839, 65521):
            torch.manual_seed(seed)
            off_general = general(stub(False), env_ids, start, b_min, b_max, bars, half)
            torch.manual_seed(seed)
            v3_general = general(stub(True), env_ids, start, b_min, b_max, bars, half)
            self.assertTrue(torch.equal(off_general, v3_general), seed)
            torch.manual_seed(seed)
            off_waypoints = waypoints(stub(False), env_ids)
            torch.manual_seed(seed)
            v3_waypoints = waypoints(stub(True), env_ids)
            self.assertTrue(torch.equal(off_waypoints, v3_waypoints), seed)


if __name__ == "__main__":
    unittest.main()
