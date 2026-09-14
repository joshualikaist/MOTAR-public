"""CPU-only contracts for the fresh two-envelope route recovery lineage."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROUTE = _load("route_recovery_route", "aerial_gym/task/navrl_task/target_route_planner.py")
MOTION = _load("route_recovery_motion", "aerial_gym/task/navrl_task/target_motion.py")
# Keep this CPU fixture independent of Isaac Gym's package initializer.
_aerial_stub = types.ModuleType("aerial_gym")
_utils_stub = types.ModuleType("aerial_gym.utils")
_math_stub = types.ModuleType("aerial_gym.utils.math")
_math_stub.get_euler_xyz_tensor = lambda q: torch.zeros(q.shape[0], 3, device=q.device)
_math_stub.quat_rotate_inverse = lambda q, v: v
_math_stub.quat_to_rotation_matrix = lambda q: torch.eye(3, device=q.device).expand(q.shape[0], 3, 3)
sys.modules["aerial_gym"] = _aerial_stub
sys.modules["aerial_gym.utils"] = _utils_stub
sys.modules["aerial_gym.utils.math"] = _math_stub
PHYS = _load("route_recovery_physical", "aerial_gym/task/navrl_task/physical_target.py")
for _name in ("aerial_gym.utils.math", "aerial_gym.utils", "aerial_gym"):
    sys.modules.pop(_name, None)


class TwoEnvelopeRecoveryTest(unittest.TestCase):
    def test_support_and_soft_inflation_are_frozen(self):
        support = ROUTE.conservative_xy_support_from_box([0.28, 0.28, 0.12])
        self.assertAlmostEqual(float(support[0]), 0.2068816087, places=9)
        self.assertAlmostEqual(float(support[0] + 0.45), 0.6568816087, places=9)

    def test_anchor_is_nearest_deterministic_and_hard_segment_safe(self):
        anchor = ROUTE.nearest_soft_free_anchor(
            [4.0, 5.0],
            np.asarray([[5.0, 5.0]], dtype=np.float64),
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            [0.0, 0.0], [10.0, 10.0], [0.2, 0.2],
            0.5, 1.25, resolution_m=0.25, radius_cells=3, tracking_margin_m=0.45,
        )
        self.assertTrue(anchor["exists"])
        self.assertTrue(anchor["hard_connector_safe"])
        self.assertEqual(anchor["cell_ij"], [13, 19])
        # The candidate itself carries the release hysteresis, not merely soft-free > 0.
        self.assertGreaterEqual(anchor["xy_m"][0], 0.5 + 0.2 + 0.45 + 0.25)

    def test_anchor_fails_closed_for_nan_and_blocked_neighbourhood(self):
        bad = ROUTE.nearest_soft_free_anchor(
            [float("nan"), 0.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0],
            [0.2, 0.2], 0.5, 1.25,
        )
        self.assertIsNone(bad["exists"])
        bars = np.asarray([[5.0, y] for y in np.linspace(0.0, 10.0, 20)])
        blocked = ROUTE.nearest_soft_free_anchor(
            [4.0, 5.0], bars, np.full((20, 2), 1.0), [0.0, 0.0], [10.0, 10.0],
            [0.2, 0.2], 0.5, 1.25,
        )
        self.assertFalse(blocked["exists"])

    def test_exact_aabb_rollout_rejects_corner_that_rounded_test_accepts(self):
        kwargs = dict(
            old_xy=torch.tensor([[6.6, 6.6]]),
            current_velocity=torch.zeros((1, 2)),
            desired_velocity=torch.tensor([[1.0, 0.0]]),
            speed_limit=torch.tensor([0.0]), dt=0.1,
            bars_xy=torch.tensor([[[5.0, 5.0]]]),
            lo=torch.tensor([[0.0, 0.0]]), hi=torch.tensor([[10.0, 10.0]]),
            clearance=0.45, turn_sign=torch.ones(1),
            max_accel=torch.tensor([4.0]), max_turn_rate=torch.tensor([2.6179939]),
            lookahead_s=0.1,
            bars_half_extents_xy=torch.tensor([[[1.2, 1.2]]]),
        )
        _, _, _, rounded = MOTION.bounded_drone_target_step(**kwargs)
        soft_kwargs = dict(kwargs)
        soft_kwargs["bars_half_extents_xy"] = torch.tensor([[[1.65, 1.65]]])
        soft_kwargs["clearance"] = 0.0
        _, _, _, exact = MOTION.bounded_drone_target_step(
            **soft_kwargs, exact_aabb_clearance=True
        )
        self.assertTrue(bool(rounded.item()))
        self.assertFalse(bool(exact.item()))

    def test_no_teleport_or_reset_in_recovery_sources(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("RECOVERY_NO_CONNECTOR", task)
        self.assertIn("recovery_anchor_idx", task)
        self.assertNotIn("self.target_position[recovery", task)
        self.assertIn("cfg_target_route_recovery_schema", task)

    def test_local_infeasible_soft_free_is_latched_separate_reason(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=True
        )
        mask = torch.tensor([True])
        manager.mark_local_infeasible_soft_free(mask)
        self.assertEqual(int(manager.recovery_state[0]), ROUTE.RECOVERY_NO_CONNECTOR)
        self.assertEqual(
            int(manager.status_code[0]),
            manager.STATUS_CODES["recovery_local_infeasible_soft_free"],
        )
        manager.recovery_state[0] = ROUTE.RECOVERY_ROUTE
        manager.enter_recovery(mask, 10)
        self.assertEqual(int(manager.recovery_state[0]), ROUTE.RECOVERY_BRAKE)

    def test_watchdog_source_has_continuous_segment_certificate(self):
        source = (ROOT / "aerial_gym/task/navrl_task/physical_target.py").read_text()
        self.assertIn("watchdog_prev_xy", source)
        self.assertIn("segment_hits_bar", source)
        self.assertIn("t_enter <= t_exit", source)
        # Both endpoints miss the closed obstacle, while the diagonal crosses its corner.
        self.assertFalse(
            ROUTE.segment_is_safe(
                [-1.0, -1.0], [1.0, 1.0], [-2.0, -2.0], [2.0, 2.0],
                np.asarray([[0.0, 0.0]]), np.asarray([[0.5, 0.5]]),
            )
        )

    def test_recovery_mode_is_not_v1_alias(self):
        self.assertEqual(ROUTE.TARGET_ROUTE_MODE_GLOBAL_ASTAR, "global_astar_v1")
        self.assertEqual(ROUTE.TARGET_ROUTE_MODE_RECOVERY, "global_astar_recovery_v2")
        self.assertNotEqual(ROUTE.TARGET_ROUTE_MODEL, ROUTE.TARGET_ROUTE_RECOVERY_MODEL)
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("if self._target_route_enabled and (", task)

    def test_v1_diagnostics_shape_is_unchanged(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=False
        )
        self.assertEqual(
            set(manager.diagnostics()),
            {
                "mode", "plan_attempts", "plan_successes", "replan_attempts",
                "connected_goal_replans", "same_goal_reselection_count", "no_path_count",
                "invalid_count", "local_step_invalidations", "fallback_intervals",
                "goal_completions", "invalidation_counts", "planning_batches", "planning_envs",
                "total_planning_wall_s", "max_batch_wall_s", "max_batch_size",
                "planning_wall_ms_per_env", "expanded_nodes", "raw_waypoints",
                "smoothed_waypoints", "currently_valid", "status_counts",
            },
        )

    def test_recovery_geometry_malformed_rows_fail_closed(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("geometry_valid", task)
        self.assertIn("torch.isfinite(bars_xy)", task)
        self.assertIn('float("-inf")', task)

    def test_timeout_and_tube_contract_are_explicit(self):
        source = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("recovery_brake_age_steps", source)
        self.assertIn("recovery_connect_age_steps", source)
        self.assertIn("max_anchor_distance", source)
        self.assertIn("TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M", source)

    def test_min_clearance_bars_without_half_extents_is_fixed_and_deterministic(self):
        torch.manual_seed(123)
        kwargs = dict(
            old_xy=torch.randn(4, 2),
            current_velocity=torch.randn(4, 2) * 0.3,
            desired_velocity=torch.randn(4, 2),
            speed_limit=torch.tensor([1.2, 0.8, 1.5, 1.0]), dt=0.1,
            bars_xy=torch.tensor([[[0.4, 0.2], [-0.3, 0.6]], [[0.1, -0.2], [0.8, 0.8]],
                                  [[0.0, 0.0], [0.5, -0.5]], [[0.2, 0.3], [-0.7, 0.2]]]),
            lo=torch.full((4, 2), -2.0), hi=torch.full((4, 2), 2.0), clearance=0.25,
            turn_sign=torch.ones(4), max_accel=torch.full((4,), 4.0),
            max_turn_rate=torch.full((4,), 2.6), lookahead_s=0.2,
            bars_half_extents_xy=None,
        )
        first = MOTION.bounded_drone_target_step(**kwargs)
        second = MOTION.bounded_drone_target_step(**kwargs)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.allclose(first[0], torch.tensor(
            [[-0.0689, 0.0943], [-0.3561, -0.2915], [-1.1884, 0.2692], [-0.9628, -0.6859]]
        ), atol=1e-4))

    def test_recovery_only_candidate_certificate_preserves_default_api(self):
        kwargs = dict(
            old_xy=torch.zeros(1, 2), current_velocity=torch.zeros(1, 2),
            desired_velocity=torch.tensor([[1.0, 0.0]]), speed_limit=torch.ones(1), dt=0.1,
            bars_xy=torch.zeros(1, 0, 2), lo=torch.full((1, 2), -2.0),
            hi=torch.full((1, 2), 2.0), clearance=0.0, turn_sign=torch.ones(1),
            max_accel=torch.full((1,), 4.0), max_turn_rate=torch.full((1,), 2.6),
            lookahead_s=1.0, bars_half_extents_xy=torch.zeros(1, 0, 2),
            exact_aabb_clearance=True, hard_epsilon_m=0.0124,
        )
        legacy = MOTION.bounded_drone_target_step(**kwargs)
        certified = MOTION.bounded_drone_target_step(**kwargs, return_certificate=True)
        self.assertEqual(len(legacy), 4)
        self.assertEqual(len(certified), 5)
        for left, right in zip(legacy, certified[:4]):
            self.assertTrue(torch.equal(left, right))
        certificate = certified[4]
        self.assertEqual(certificate["candidate_count"], 73)
        self.assertEqual(certificate["horizon_steps"], 10)
        self.assertTrue(bool(certificate["full_horizon_safe"].all()))
        self.assertTrue(bool((certificate["safe_prefix_steps"] == 10).all()))
        self.assertEqual(tuple(certificate["selected_final_pos"].shape), (1, 2))
        self.assertEqual(tuple(certificate["selected_final_position_xy"].shape), (1, 2))
        self.assertTrue(torch.equal(certificate["row_ids"], torch.tensor([0])))

    def test_connect_consumes_full_horizon_not_immediate_only(self):
        source = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn('connect_certificate["full_horizon_safe"]', source)
        self.assertIn('connect_certificate["safe_prefix_steps"]', source)
        self.assertIn('connect_certificate["selected_final_position_xy"]', source)

    def test_watchdog_install_point_breach_latches_after_interval_begin(self):
        tensors = {
            "dt": 0.01,
            "obstacle_position": torch.tensor([[[2.0, 0.0, 0.0]]]),
            "obstacle_orientation": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
            "obstacle_linvel": torch.zeros(1, 1, 3),
            "obstacle_angvel": torch.zeros(1, 1, 3),
            "obstacle_force_tensor": torch.zeros(1, 1, 3),
            "obstacle_torque_tensor": torch.zeros(1, 1, 3),
            "obstacle_contact_force_tensor": torch.zeros(1, 1, 3),
            "gravity": torch.tensor([[0.0, 0.0, -9.81]]),
        }
        cfg = SimpleNamespace(
            physical_mass=1.2, physical_max_motor_thrust=9.6, physical_motor_tau=0.04,
            physical_max_tilt_deg=45.0, physical_velocity_kp=2.5, physical_altitude_kp=4.0,
            physical_attitude_kp=[0.08, 0.08, 0.04], physical_rate_kp=[0.04, 0.04, 0.03],
            physical_yaw_torque_ratio=0.01, physical_motor_arm_xy=0.0777817,
        )
        controller = PHYS.PhysicalTargetController(tensors, 0, cfg, torch.device("cpu"))
        controller.begin_control_interval()
        controller.set_hard_watchdog(
            torch.empty(1, 0, 2), torch.empty(1, 0, 2),
            torch.tensor([[-1.0, -1.0]]), torch.tensor([[1.0, 1.0]]),
            active=torch.tensor([True]),
        )
        self.assertTrue(bool(controller.watchdog_breach[0]))
        controller.begin_control_interval()
        self.assertFalse(bool(controller.watchdog_breach[0]))

    def test_watchdog_and_soft_segment_geometry_match_on_randomized_batch(self):
        """The PhysX substep watchdog and v3 certificate use the same closed geometry."""
        torch.manual_seed(829)
        n, bars_n = 256, 4
        p0 = torch.rand(n, 2) * 12.0 - 6.0
        p1 = p0 + torch.rand(n, 2) * 3.0 - 1.5
        bars = torch.rand(n, bars_n, 2) * 12.0 - 6.0
        half = torch.rand(n, bars_n, 2) * 0.5 + 0.2
        lo = torch.full((n, 2), -7.0)
        hi = torch.full((n, 2), 7.0)
        tensors = {
            "dt": 0.01,
            "obstacle_position": torch.cat(
                (p0, torch.zeros(n, 1)), dim=1
            ).view(n, 1, 3),
            "obstacle_orientation": torch.tensor(
                [0.0, 0.0, 0.0, 1.0]
            ).repeat(n, 1).view(n, 1, 4),
            "obstacle_linvel": torch.zeros(n, 1, 3),
            "obstacle_angvel": torch.zeros(n, 1, 3),
            "obstacle_force_tensor": torch.zeros(n, 1, 3),
            "obstacle_torque_tensor": torch.zeros(n, 1, 3),
            "obstacle_contact_force_tensor": torch.zeros(n, 1, 3),
            "gravity": torch.tensor([0.0, 0.0, -9.81]).repeat(n, 1),
        }
        cfg = SimpleNamespace(
            physical_mass=1.2, physical_max_motor_thrust=9.6, physical_motor_tau=0.04,
            physical_max_tilt_deg=45.0, physical_velocity_kp=2.5,
            physical_altitude_kp=4.0, physical_attitude_kp=[0.08, 0.08, 0.04],
            physical_rate_kp=[0.04, 0.04, 0.03], physical_yaw_torque_ratio=0.01,
            physical_motor_arm_xy=0.0777817,
        )
        controller = PHYS.PhysicalTargetController(tensors, 0, cfg, torch.device("cpu"))
        controller.begin_control_interval()
        controller.set_hard_watchdog(bars, half, lo, hi, active=torch.ones(n, dtype=torch.bool))
        controller.position[:, :2] = p1
        controller.post_physics_step()
        certificate_safe = MOTION.torch_segments_soft_safe(
            p0[:, None, :], p1[:, None, :], lo, hi, bars, half
        )[:, 0]
        self.assertTrue(torch.equal(controller.watchdog_breach, ~certificate_safe))

    def test_recovery_checkpoint_requires_fixed_runtime_and_source_fields(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        for key in (
            "cfg_target_max_accel_mps2", "cfg_target_max_turn_rate_degps",
            "cfg_target_lookahead_s", "cfg_physics_dt_s", "cfg_physics_substeps",
            "cfg_physics_steps_per_rl_step", "cfg_rl_step_dt_s",
            "cfg_target_physical_tracking_margin_m", "cfg_target_physical_boundary_margin_m",
            "cfg_target_route_support_xy_m", "cfg_training_source_manifest",
            "cfg_training_source_manifest_sha256", "cfg_target_recovery_probe_receipt_sha256",
        ):
            self.assertIn(key, task)
        self.assertIn("missing provenance", task)

    def test_connect_budget_uses_cell_projection_half_cell(self):
        self.assertAlmostEqual((2.0 ** 0.5) * 3.5 * 0.25, 1.237436867, places=7)
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("(3.0 + 0.5)", task)

    def test_timeout_reasons_and_stop_distance_lookup_contract(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=True
        )
        mask = torch.tensor([True])
        manager.mark_no_connector(mask, timeout_kind="brake")
        self.assertEqual(int(manager.status_code[0]), manager.STATUS_CODES["recovery_brake_timeout"])
        self.assertEqual(int(manager.recovery_brake_timeout_count.item()), 1)
        manager.reset_idx(torch.tensor([0]))
        manager.mark_no_connector(mask, timeout_kind="connect")
        self.assertEqual(int(manager.status_code[0]), manager.STATUS_CODES["recovery_connect_timeout"])
        self.assertEqual(int(manager.recovery_connect_timeout_count.item()), 1)
        self.assertIn("brake_speed_samples_mps", (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text())

    def test_probe_receipt_requires_receipt_and_probe_schema(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn('"navrl_target_recovery_braking_receipt_v1"', task)
        self.assertIn('"navrl_target_recovery_braking_probe_v1"', task)

    def test_brake_certificate_uses_ceiling_lookup_and_lateral_tube(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=True
        )
        result = manager.brake_connector_idx(
            torch.tensor([0]), torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.8]]),
            torch.empty(1, 0, 2), torch.empty(1, 0, 2),
            torch.tensor([[-2.0, -2.0]]), torch.tensor([[2.0, 2.0]]),
            torch.zeros(1, 2), 0.1, 4.0,
            brake_speed_samples_mps=(0.6, 1.0),
            brake_stop_distance_samples_m=(0.2, 0.5),
            certified_lateral_tube_m=0.1,
        )
        self.assertTrue(bool(result[0]))
        planner = (ROOT / "aerial_gym/task/navrl_task/target_route_planner.py").read_text()
        self.assertIn("np.searchsorted(sample_speeds, speed, side=\"left\")", planner)
        self.assertIn("certified_lateral_tube_m", planner)

    def test_recovery_geometry_projects_xyz_arena_bounds_to_xy(self):
        manager = ROUTE.BatchedTargetRouteManager(
            1, torch.device("cpu"), ROUTE.RoutePlannerConfig(), recovery_enabled=True
        )
        env_ids = torch.tensor([0])
        position = torch.tensor([[0.0, 0.0]])
        bars = torch.empty(1, 0, 2)
        half = torch.empty(1, 0, 2)
        lo_xyz = torch.tensor([[-2.0, -2.0, 0.0]])
        hi_xyz = torch.tensor([[2.0, 2.0, 3.0]])
        support = torch.zeros(1, 2)
        brake = manager.brake_connector_idx(
            env_ids, position, torch.tensor([[0.0, 0.8]]), bars, half,
            lo_xyz, hi_xyz, support, 0.1, 4.0,
            brake_speed_samples_mps=(0.6, 1.0),
            brake_stop_distance_samples_m=(0.2, 0.5),
            certified_lateral_tube_m=0.1,
        )
        anchor = manager.recovery_anchor_idx(
            env_ids, position, bars, half, lo_xyz, hi_xyz, support, 0.1, 0.2
        )
        self.assertTrue(bool(brake[0]))
        self.assertTrue(bool(anchor[0]))

    def test_validated_flag_cannot_bypass_canonical_receipt_handoff(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        for field in (
            "validator.verify_receipt", "result.get(\"summary\")", "result.get(\"core_integration\")",
            "certified_monotone_speed_to_p95_lookup",
            "certified_lateral_tube_p95_m", "source_manifest_sha256",
            "recovery environment braking lookup differs from canonical receipt",
        ):
            self.assertIn(field, task)
        self.assertIn('"selected_final_pos"', (ROOT / "aerial_gym/task/navrl_task/target_motion.py").read_text())
        self.assertIn('"selected_final_position_xy"', (ROOT / "aerial_gym/task/navrl_task/target_motion.py").read_text())

    def test_raw_mutation_cannot_be_hidden_by_receipt_or_validated_flag(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text()
        self.assertIn("validator.verify_receipt(receipt_file.parent", task)
        self.assertIn("RECOVERY_RECEIPT_VALIDATOR_SHA256", task)
        self.assertIn("RECOVERY_PROBE_VALIDATOR_SHA256", task)
        # The task must consume verifier-returned data; producer receipt fields are deliberately
        # absent from the recovery arm path, so forging receipt.json/manifest/VALIDATED cannot
        # bypass raw-cell recomputation.
        self.assertNotIn("probe.get(\"core_integration\")", task)


if __name__ == "__main__":
    unittest.main()
