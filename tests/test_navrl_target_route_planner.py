"""CPU contracts for the opt-in physical-target global route planner."""

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"
SPEC = importlib.util.spec_from_file_location("navrl_target_route_planner_standalone", PATH)
ROUTE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ROUTE
SPEC.loader.exec_module(ROUTE)


def planner(resolution=0.20, tracking=0.10, boundary=0.0):
    return ROUTE.DeterministicAStarRoutePlanner(
        ROUTE.RoutePlannerConfig(
            resolution_m=resolution,
            tracking_margin_m=tracking,
            boundary_margin_m=boundary,
            max_expansions=100000,
            max_waypoints=128,
        )
    )


def plan(p, bars, half, start=(1.0, 5.0), goal=(9.0, 5.0), support=(0.2, 0.2)):
    return p.plan(
        np.asarray(start), np.asarray(goal), np.asarray(bars).reshape((-1, 2)),
        np.asarray(half).reshape((-1, 2)), np.array([0.0, 0.0]), np.array([10.0, 10.0]),
        np.asarray(support),
    )


class TargetRoutePlannerTest(unittest.TestCase):
    def test_ordinary_gpu_follower_has_no_cpu_materialization_or_sync(self):
        source = inspect.getsource(ROUTE.BatchedTargetRouteManager.velocity_reference)
        for forbidden in (".cpu(", ".item(", ".numpy(", "bool("):
            self.assertNotIn(forbidden, source)

    def test_support_envelope_is_full_orientation_half_diagonal(self):
        support = ROUTE.conservative_xy_support_from_box((0.28, 0.28, 0.12))
        expected = 0.5 * np.linalg.norm([0.28, 0.28, 0.12])
        np.testing.assert_allclose(support, [expected, expected])

    def test_blocked_straight_line_routes_around_exact_aabb(self):
        p = planner()
        result = plan(p, [(5.0, 5.0)], [(0.8, 2.4)])
        self.assertTrue(result.valid, result.status)
        self.assertGreater(result.smoothed_nodes, 2)
        inflated = np.array([[1.1, 2.7]])
        for a, b in zip(result.waypoints_xy[:-1], result.waypoints_xy[1:]):
            self.assertTrue(
                ROUTE.segment_is_safe(
                    a, b, np.array([0.2, 0.2]), np.array([9.8, 9.8]),
                    np.array([[5.0, 5.0]]), inflated,
                )
            )

    def test_full_height_wall_fails_closed(self):
        result = plan(planner(), [(5.0, 5.0)], [(0.3, 5.0)])
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "no_path")
        self.assertEqual(result.waypoints_xy.shape, (0, 2))

    def test_narrow_corridor_passes_only_above_inflated_width(self):
        # Two long horizontal AABBs. With support+tracking=0.30 m per wall, a 1.2 m raw opening
        # leaves 0.6 m for the target centre; a 0.5 m opening closes completely.
        centers = [(5.0, 3.4), (5.0, 6.6)]
        wide = plan(planner(), centers, [(5.0, 1.0), (5.0, 1.0)])
        self.assertTrue(wide.valid, wide.status)
        narrow_centers = [(5.0, 4.25), (5.0, 5.75)]
        narrow = plan(planner(), narrow_centers, [(5.0, 0.5), (5.0, 0.5)])
        self.assertFalse(narrow.valid)
        self.assertIn(narrow.status, ("unsafe_start", "unsafe_start_cell", "no_path"))

    def test_reflection_and_repeat_are_deterministic(self):
        p = planner(resolution=0.25)
        bars = np.array([[4.0, 3.0], [6.0, 3.0]])
        half = np.array([[0.8, 3.0], [0.8, 3.0]])
        first = plan(p, bars, half, start=(1.0, 5.0), goal=(9.0, 5.0))
        second = plan(p, bars, half, start=(1.0, 5.0), goal=(9.0, 5.0))
        self.assertTrue(first.valid, first.status)
        np.testing.assert_array_equal(first.waypoints_xy, second.waypoints_xy)

        reflected_bars = bars.copy()
        reflected_bars[:, 1] = 10.0 - reflected_bars[:, 1]
        reflected = plan(
            p, reflected_bars, half, start=(1.0, 5.0), goal=(9.0, 5.0)
        )
        self.assertTrue(reflected.valid, reflected.status)
        expected = first.waypoints_xy.copy()
        expected[:, 1] = 10.0 - expected[:, 1]
        np.testing.assert_allclose(reflected.waypoints_xy, expected, atol=0.26)

    def test_line_of_sight_rejects_aabb_corner_cut(self):
        lo, hi = np.array([0.0, 0.0]), np.array([3.0, 3.0])
        bars = np.array([[1.0, 1.0]])
        half = np.array([[0.2, 0.2]])
        self.assertFalse(
            ROUTE.segment_is_safe(
                np.array([0.2, 0.2]), np.array([1.8, 1.8]), lo, hi, bars, half
            )
        )
        self.assertTrue(
            ROUTE.segment_is_safe(
                np.array([0.2, 0.2]), np.array([1.8, 0.5]), lo, hi, bars, half
            )
        )

    def test_nonfinite_geometry_is_invalid_not_optimistically_empty(self):
        result = plan(planner(), [(float("nan"), 5.0)], [(0.5, 0.5)])
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid_input")

    def test_connected_goal_sampling_returns_proved_distant_route(self):
        p = planner()
        result = p.plan_to_connected_goal(
            np.array([1.0, 5.0]), np.array([[5.0, 5.0]]), np.array([[0.8, 2.4]]),
            np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.array([0.2, 0.2]),
            min_goal_distance_m=4.0, selector=0.37,
        )
        self.assertTrue(result.valid, result.status)
        self.assertGreaterEqual(np.linalg.norm(result.waypoints_xy[-1] - result.waypoints_xy[0]), 4.0)

    def test_waypoint_overshoot_advances_without_exact_radius_hit(self):
        config = ROUTE.RoutePlannerConfig(
            resolution_m=0.25, tracking_margin_m=0.0, boundary_margin_m=0.0,
            max_waypoints=8,
        )
        manager = ROUTE.BatchedTargetRouteManager(1, torch.device("cpu"), config)
        manager.valid[0] = True
        manager.length[0] = 2
        manager.cursor[0] = 0
        manager.segment_start[0] = torch.tensor([0.0, 0.0])
        manager.handoff_clearance[0, 0] = 10.0
        manager.waypoints[0, 0] = torch.tensor([1.0, 0.0])
        manager.waypoints[0, 1] = torch.tensor([2.0, 0.0])
        velocity, active, complete = manager.velocity_reference(
            torch.tensor([[1.2, 0.0]]), torch.tensor([1.0]), reach_m=0.05
        )
        self.assertEqual(int(manager.cursor[0]), 1)
        self.assertTrue(bool(active[0]))
        self.assertFalse(bool(complete[0]))
        self.assertGreater(float(velocity[0, 0]), 0.0)

        # Crossing the waypoint plane far away from the segment is not progress along the route.
        manager.cursor[0] = 0
        manager.segment_start[0] = torch.tensor([0.0, 0.0])
        manager.velocity_reference(
            torch.tensor([[1.2, 2.0]]), torch.tensor([1.0]), reach_m=0.5
        )
        self.assertEqual(int(manager.cursor[0]), 0)

        manager.cursor[0] = 1
        manager.segment_start[0] = torch.tensor([1.0, 0.0])
        _, _, complete = manager.velocity_reference(
            torch.tensor([[2.2, 0.0]]), torch.tensor([1.0]), reach_m=0.5
        )
        self.assertTrue(bool(complete[0]))
        manager.velocity_reference(
            torch.tensor([[2.2, 0.0]]), torch.tensor([1.0]), reach_m=0.5
        )
        self.assertEqual(manager.diagnostics()["goal_completions"], 1)

    def test_local_failure_is_distinct_and_requires_connected_goal_replacement(self):
        config = ROUTE.RoutePlannerConfig(replan_cooldown_steps=3)
        manager = ROUTE.BatchedTargetRouteManager(1, torch.device("cpu"), config)
        ids = torch.tensor([0])
        start = torch.tensor([[5.0, 5.0]])
        arbitrary_goal = torch.tensor([[8.0, 5.0]])
        bars = torch.empty((1, 0, 2))
        bounds_lo = torch.tensor([[0.0, 0.0]])
        bounds_hi = torch.tensor([[10.0, 10.0]])
        support = torch.tensor([[0.1, 0.1]])
        manager.plan_idx(
            ids, start, arbitrary_goal, bars, bars, bounds_lo, bounds_hi, support,
            current_step=0, connected_goal_selector=torch.tensor([0.1]),
            min_goal_distance_m=2.0,
        )
        first_goal = manager.goal.clone()
        manager.invalidate(
            torch.tensor([True]), "local_step_infeasible", current_step=10
        )
        self.assertEqual(
            int(manager.status_code[0]), manager.STATUS_CODES["local_step_infeasible"]
        )
        self.assertFalse(bool(manager.needs_replan(manager.goal, manager.planned_support, 12)[0]))
        self.assertTrue(bool(manager.needs_replan(manager.goal, manager.planned_support, 13)[0]))
        manager.plan_idx(
            ids, start, arbitrary_goal, bars, bars, bounds_lo, bounds_hi, support,
            current_step=13, is_replan=True,
            connected_goal_selector=torch.tensor([0.1]), min_goal_distance_m=2.0,
            excluded_goal_xy=first_goal,
            goal_exclusion_radius_m=config.goal_exclusion_radius_m,
        )
        self.assertFalse(torch.equal(first_goal, manager.goal))
        self.assertGreater(
            float((first_goal - manager.goal).norm()), config.goal_exclusion_radius_m
        )
        self.assertEqual(manager.diagnostics()["local_step_invalidations"], 1)
        self.assertEqual(manager.diagnostics()["connected_goal_replans"], 1)
        diagnostics = manager.diagnostics()
        self.assertEqual(diagnostics["invalidation_counts"]["local_step_infeasible"], 1)
        self.assertGreaterEqual(diagnostics["planning_batches"], 2)
        self.assertGreaterEqual(diagnostics["planning_envs"], 2)
        self.assertGreaterEqual(diagnostics["total_planning_wall_s"], 0.0)
        self.assertGreaterEqual(diagnostics["max_batch_wall_s"], 0.0)
        self.assertGreaterEqual(diagnostics["planning_wall_ms_per_env"], 0.0)
        self.assertEqual(diagnostics["same_goal_reselection_count"], 0)

    def test_corner_handoff_requires_exact_clearance_certificate(self):
        p = planner()
        # Match runtime float32 geometry exactly; raster occupancy is deliberately sensitive to
        # closed-AABB boundaries, so constructing a separate float64 fixture can choose a
        # neighbouring (also safe) grid centre.
        bars = np.array([[5.0, 5.0]], dtype=np.float32)
        half = np.array([[0.8, 2.4]], dtype=np.float32)
        support = np.array([0.2, 0.2], dtype=np.float32)
        result = p.plan(
            np.array([1.0, 5.0]), np.array([9.0, 5.0]), bars, half,
            np.array([0.0, 0.0]), np.array([10.0, 10.0]), support,
        )
        self.assertTrue(result.valid, result.status)
        inflated = half + support[None, :] + p.config.tracking_margin_m
        certificates = ROUTE.route_handoff_clearance_certificates(
            result.waypoints_xy,
            np.array([0.0, 0.0]) + support,
            np.array([10.0, 10.0]) - support,
            bars,
            inflated,
        )
        corner, next_waypoint = result.waypoints_xy[1], result.waypoints_xy[2]
        self.assertLess(certificates[1], 0.5)
        unsafe_early = np.array([3.65, 2.30])
        self.assertLess(np.linalg.norm(unsafe_early - corner), 0.5)
        self.assertGreater(np.linalg.norm(unsafe_early - corner), certificates[1])
        self.assertFalse(
            ROUTE.segment_is_safe(
                unsafe_early, next_waypoint, np.array([0.2, 0.2]),
                np.array([9.8, 9.8]), bars, inflated,
            )
        )

        manager = ROUTE.BatchedTargetRouteManager(1, torch.device("cpu"), p.config)
        manager.plan_idx(
            torch.tensor([0]),
            torch.tensor([[1.0, 5.0]]),
            torch.tensor([[9.0, 5.0]]),
            torch.as_tensor(bars[None, :, :], dtype=torch.float32),
            torch.as_tensor(half[None, :, :], dtype=torch.float32),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[10.0, 10.0]]),
            torch.as_tensor(support[None, :], dtype=torch.float32),
            current_step=0,
        )
        self.assertTrue(bool(manager.valid[0]))
        self.assertTrue(
            np.allclose(manager.waypoints[0, :2].numpy(), result.waypoints_xy[1:3])
        )
        self.assertAlmostEqual(
            float(manager.handoff_clearance[0, 0]), float(certificates[1]), places=6
        )
        manager.velocity_reference(
            torch.as_tensor(unsafe_early[None, :], dtype=torch.float32),
            torch.tensor([1.0]), reach_m=0.5,
        )
        self.assertEqual(int(manager.cursor[0]), 0, "unsafe corner shortcut was accepted")

        safe_near = corner + np.array([0.25 * certificates[1], 0.0])
        self.assertTrue(
            ROUTE.segment_is_safe(
                safe_near, next_waypoint, np.array([0.2, 0.2]),
                np.array([9.8, 9.8]), bars, inflated,
            )
        )
        manager.velocity_reference(
            torch.as_tensor(safe_near[None, :], dtype=torch.float32),
            torch.tensor([1.0]), reach_m=0.5,
        )
        self.assertEqual(int(manager.cursor[0]), 1)

    def test_connected_replan_excludes_previous_goal_and_counts_regression(self):
        config = ROUTE.RoutePlannerConfig(
            boundary_margin_m=0.0, goal_exclusion_radius_m=1.0
        )
        manager = ROUTE.BatchedTargetRouteManager(1, torch.device("cpu"), config)
        ids = torch.tensor([0])
        start = torch.tensor([[5.0, 5.0]])
        requested_goal = torch.tensor([[8.0, 5.0]])
        bars = torch.empty((1, 0, 2))
        bounds_lo = torch.tensor([[0.0, 0.0]])
        bounds_hi = torch.tensor([[10.0, 10.0]])
        support = torch.tensor([[0.1, 0.1]])
        selector = torch.tensor([0.37])
        manager.plan_idx(
            ids, start, requested_goal, bars, bars, bounds_lo, bounds_hi, support,
            current_step=0, connected_goal_selector=selector, min_goal_distance_m=2.0,
        )
        previous_goal = manager.goal.clone()
        manager.plan_idx(
            ids, start, requested_goal, bars, bars, bounds_lo, bounds_hi, support,
            current_step=1, is_replan=True, connected_goal_selector=selector,
            min_goal_distance_m=2.0, excluded_goal_xy=previous_goal,
            goal_exclusion_radius_m=1.0,
        )
        self.assertTrue(bool(manager.valid[0]))
        self.assertGreater(float((manager.goal - previous_goal).norm()), 1.0)
        self.assertEqual(manager.diagnostics()["same_goal_reselection_count"], 0)

        original = manager.planner.plan_to_connected_goal
        manager.planner.plan_to_connected_goal = lambda *args, **kwargs: ROUTE.RoutePlan(
            "ok", previous_goal.numpy().copy(), 0, 2, 2, 0.0
        )
        manager.plan_idx(
            ids, start, requested_goal, bars, bars, bounds_lo, bounds_hi, support,
            current_step=2, is_replan=True, connected_goal_selector=selector,
            min_goal_distance_m=2.0, excluded_goal_xy=previous_goal,
            goal_exclusion_radius_m=1.0,
        )
        manager.planner.plan_to_connected_goal = original
        self.assertFalse(bool(manager.valid[0]))
        self.assertEqual(
            int(manager.status_code[0]), manager.STATUS_CODES["same_goal_reselected"]
        )
        self.assertEqual(manager.diagnostics()["same_goal_reselection_count"], 1)


if __name__ == "__main__":
    unittest.main()
