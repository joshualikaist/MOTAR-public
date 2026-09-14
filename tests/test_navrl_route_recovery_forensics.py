"""CPU contracts for the evaluation-only routed recovery observer."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/diagnose_navrl_physical_target_route_recovery.py"
SPEC = importlib.util.spec_from_file_location("route_recovery_forensics", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class RouteRecoveryForensicsTest(unittest.TestCase):
    def test_frozen_cells_keep_attempt2_density_knots_and_speed_endpoints(self):
        self.assertEqual(MOD.DENSITIES, (70, 150, 205, 300))
        self.assertEqual(MOD.SPEEDS, (0.6, 1.5))
        self.assertEqual(MOD.TRACKING_MARGIN_M, 0.45)
        self.assertEqual(MOD.GRID_RESOLUTION_M, 0.25)
        self.assertEqual(MOD.ANCHOR_RADIUS_CELLS, 3)
        contract = MOD.frozen_contract()
        self.assertEqual(contract["runtime_wall_margin_m"], 0.50)
        self.assertEqual(contract["route_boundary_margin_m"], 1.25)
        self.assertEqual(contract["boundary_soft_minus_hard_m"], 0.75)
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"NAVRL_NUM_BARS": "70"', source)
        self.assertIn('"NAVRL_MAX_BARS": "300"', source)
        self.assertIn('"AERIAL_GYM_SIM_NAME": "base_sim"', source)
        self.assertIn('"NAVRL_PLACEMENT_TOUCH_M": "0.4"', source)
        self.assertIn('"NAVRL_PLACEMENT_GAP_M": "1.6"', source)

    def test_hard_and_soft_bar_clearance_are_exact_aabb_distances(self):
        row = MOD.geometry_metrics(
            [1.0, 5.0], [[5.0, 5.0]], [[1.0, 1.0]], [0.0, 0.0], [10.0, 10.0],
            [0.2, 0.2], 0.0,
        )
        self.assertAlmostEqual(row["hard_bar_clearance_m"], 2.8)
        self.assertAlmostEqual(row["soft_bar_clearance_m"], 2.35)
        self.assertEqual(row["unsafe_start_reason"], "none")

    def test_runtime_rounded_and_route_square_soft_models_disagree_at_corner(self):
        row = MOD.geometry_metrics(
            [6.6, 6.6], [[5.0, 5.0]], [[1.0, 1.0]], [0.0, 0.0], [10.0, 10.0],
            [0.2, 0.2], 0.0,
        )
        self.assertTrue(row["local_soft_free_rounded"])
        self.assertFalse(row["route_soft_free_aabb"])
        self.assertTrue(row["rounded_vs_aabb_soft_disagreement"])

    def test_unsafe_start_reason_separates_boundary_and_obstacle(self):
        boundary = MOD.geometry_metrics(
            [0.5, 5.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
            soft_boundary_margin=0.75,
        )
        obstacle = MOD.geometry_metrics(
            [4.0, 5.0], [[5.0, 5.0]], [[0.5, 0.5]], [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
        )
        self.assertEqual(boundary["unsafe_start_reason"], "boundary")
        self.assertAlmostEqual(boundary["hard_boundary_clearance_m"], 0.3)
        self.assertAlmostEqual(boundary["soft_boundary_clearance_m"], -0.45)
        self.assertEqual(obstacle["unsafe_start_reason"], "obstacle")

    def test_route_cross_track_and_polyline_error_are_distinct_fields(self):
        row = MOD.geometry_metrics(
            [2.0, 1.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
            route=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]],
            active_segment=([0.0, 0.0], [4.0, 0.0]),
        )
        self.assertAlmostEqual(row["route_cross_track_error_m"], 1.0)
        self.assertAlmostEqual(row["route_polyline_error_m"], 1.0)

    def test_synthetic_anchor_probe_has_bounded_cpu_work(self):
        bars = np.stack((np.linspace(2.0, 8.0, 32), np.full(32, 5.0)), axis=1)
        half = np.full((32, 2), 0.05)
        started = time.perf_counter()
        for _ in range(64):
            MOD.nearest_soft_free_anchor(
                [1.0, 5.0], bars, half, [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.5,
                soft_boundary_margin=1.25,
            )
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_anchor_requires_soft_free_point_and_exact_hard_connector(self):
        anchor = MOD.nearest_soft_free_anchor(
            [0.5, 5.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
        )
        self.assertTrue(anchor["exists"])
        self.assertTrue(anchor["hard_connector_safe"])
        self.assertGreater(anchor["distance_m"], 0.0)

    def test_bad_geometry_fails_closed(self):
        row = MOD.geometry_metrics(
            [float("nan"), 0.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
        )
        self.assertIsNone(row["hard_clearance_m"])
        self.assertIsNone(row["unsafe_start_reason"])
        anchor = MOD.nearest_soft_free_anchor(
            [float("nan"), 0.0], [], np.empty((0, 2)), [0.0, 0.0], [10.0, 10.0], [0.2, 0.2], 0.0,
        )
        self.assertIsNone(anchor["exists"])

    def test_diagnostic_does_not_modify_original_evaluator_or_attempt2_path(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn("attempt2_artifacts_read_only", source)
        self.assertIn("original_evaluator_unchanged", source)
        self.assertIn("ATTEMPT2_SUMMARY", source)
        self.assertNotEqual(MOD.OUTPUT_ROOT, ROOT / "results/navrl_physical_target_routed_gate_seed827_attempt2")
        self.assertIn("authorization_token", source)
        self.assertIn(".COMPLETE.json", source)
        self.assertIn("OUTPUT_ROOT.name + \".partial-\"", source)
        self.assertIn("_build_summary", source)
        self.assertIn("connector_total = unsafe_count", source)
        self.assertIn("local_fallback_intervals", source)
        self.assertIn("cell_key + \"#\"", source)

    def test_post_commit_verify_binds_recorded_commit_and_runtime_bytes(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('["git", "cat-file", "-e"', source)
        self.assertNotIn('payload.get("git_head") != current_head', source)
        self.assertIn("runtime_source_manifest", source)
        self.assertIn("software_provenance", source)

    def test_unique_origins_are_cell_scoped_and_local_only(self):
        cells = []
        for density, speed in ((d, s) for d in MOD.DENSITIES for s in MOD.SPEEDS):
            events = []
            fallback = []
            if density == 70 and speed == 0.6:
                events = [
                    {"event": "invalidation", "event_id": 0, "reason": "local_step_infeasible"},
                    {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                     "hard_free_exact": True, "soft_free": False,
                     "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}},
                    {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                     "hard_free_exact": True, "soft_free": False,
                     "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}},
                    {"event": "invalidation", "event_id": 1, "reason": "goal_changed"},
                    {"event": "replan", "origin_invalidation_id": 1, "plan_status": "unsafe_start"},
                ]
                fallback = [{"invalidation_event_id": 0, "reason": "local_step_infeasible",
                             "intervals": 11, "max_age_steps": 10},
                            {"invalidation_event_id": 1, "reason": "goal_changed",
                             "intervals": 99, "max_age_steps": 10}]
            elif density == 70 and speed == 1.5:
                events = [
                    {"event": "invalidation", "event_id": 0, "reason": "local_step_infeasible"},
                    {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                     "hard_free_exact": True, "soft_free": False,
                     "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}},
                ]
                fallback = [{"invalidation_event_id": 0, "reason": "local_step_infeasible",
                             "intervals": 7, "max_age_steps": 6}]
            name = "cell_%s_%s.json" % (density, speed)
            cells.append({"density": density, "speed_mps": speed, "path": name})
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "receipt.json").write_text("{}\n", encoding="utf-8")
            for entry in cells:
                events, fallback = [], []
                if entry["density"] == 70 and entry["speed_mps"] == 0.6:
                    events = [
                        {"event": "invalidation", "event_id": 0, "reason": "local_step_infeasible"},
                        {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                         "hard_free_exact": True, "soft_free": False,
                         "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}},
                        {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                         "hard_free_exact": True, "soft_free": False,
                         "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}},
                        {"event": "invalidation", "event_id": 1, "reason": "goal_changed"},
                        {"event": "replan", "origin_invalidation_id": 1, "plan_status": "unsafe_start"},
                    ]
                    fallback = [{"invalidation_event_id": 0, "reason": "local_step_infeasible", "intervals": 11,
                                 "max_age_steps": 10}, {"invalidation_event_id": 1, "reason": "goal_changed",
                                 "intervals": 99, "max_age_steps": 10}]
                elif entry["density"] == 70 and entry["speed_mps"] == 1.5:
                    events = [{"event": "invalidation", "event_id": 0, "reason": "local_step_infeasible"},
                              {"event": "replan", "origin_invalidation_id": 0, "plan_status": "unsafe_start",
                               "hard_free_exact": True, "soft_free": False,
                               "nearest_soft_free_anchor": {"exists": True, "hard_connector_safe": True}}]
                    fallback = [{"invalidation_event_id": 0, "reason": "local_step_infeasible", "intervals": 7,
                                 "max_age_steps": 6}]
                cell = {"events": events, "fallback_by_origin": fallback}
                (directory / entry["path"]).write_text(json.dumps(cell), encoding="utf-8")
            summary = MOD._build_summary(directory, {"cells": cells})
        decision = summary["decision_rule"]
        self.assertEqual(decision["unsafe_start_replan_n"], 2)
        self.assertEqual(decision["unsafe_start_replan_repeats"], 1)
        self.assertEqual(decision["unsafe_start_replan_other_origin"], 1)
        self.assertEqual(summary["local_fallback_intervals"], 18)
        self.assertEqual(len(summary["per_cell"][0]["origins"]), 2)
        self.assertEqual(summary["per_cell"][0]["origins"][0]["repeats"], 1)
        self.assertEqual(summary["per_cell"][0]["origins"][1]["fallback_intervals"], 99)
        self.assertEqual(summary["per_cell"][0]["origins"][1]["max_fallback_age_steps"], 10)

    def test_completed_summary_is_immutable(self):
        with self.assertRaises(RuntimeError):
            MOD.summarize(MOD.OUTPUT_ROOT)

    def test_cached_route_clearance_skips_bar_by_segment_recomputation(self):
        bars = np.stack((np.linspace(0.0, 30.0, 300), np.full(300, 5.0)), axis=1)
        half = np.full((300, 2), 0.05)
        route = np.stack((np.linspace(0.0, 30.0, 128), np.full(128, 1.0)), axis=1)
        cached = {"route_hard_min_segment_clearance_m": 0.7,
                  "route_soft_min_segment_clearance_m": 0.2}
        with mock.patch.object(MOD, "_segment_aabb_distance", side_effect=AssertionError("cache miss")):
            row = MOD.geometry_metrics([1.0, 1.0], bars, half, [0.0, 0.0], [40.0, 40.0],
                                        [0.2, 0.2], 0.5, route=route,
                                        active_segment=([0.0, 1.0], [1.0, 1.0]),
                                        route_clearance=cached)
        self.assertEqual(row["route_hard_min_segment_clearance_m"], 0.7)
        self.assertEqual(row["route_soft_min_segment_clearance_m"], 0.2)

    def test_float32_materialization_tolerance_is_bounded(self):
        support = np.asarray([0.2068816086567407, 0.2068816086567407], dtype=np.float32)
        MOD._require_float32_close(support, [0.2068816086567407, 0.2068816086567407], "support")
        gains = np.asarray([0.08, 0.08, 0.04], dtype=np.float32)
        MOD._require_float32_close(gains, [0.08, 0.08, 0.04], "gains")
        with self.assertRaises(RuntimeError):
            MOD._require_float32_close([0.08, 0.08, 0.040002], [0.08, 0.08, 0.04], "gains")
        with self.assertRaises(RuntimeError):
            MOD._require_float32_close([0.206883], [0.2068816086567407], "support")

    def test_plan_event_uses_old_route_cache_before_installing_new_route_cache(self):
        class Manager:
            STATUS_CODES = {"ok": 0}
            status_code = np.array([0])
            valid = np.array([True])
            length = np.array([2])
            waypoints = np.array([[[4.0, 1.0], [5.0, 1.0]]])

        geometry = {
            "position": np.array([[1.0, 1.0]]), "bars": np.empty((1, 0, 2)),
            "bar_half": np.empty((1, 0, 2)), "bounds_lo": np.array([[0.0, 0.0]]),
            "bounds_hi": np.array([[10.0, 10.0]]), "support": np.array([[0.2, 0.2]]),
            "hard_boundary_margin": 0.5, "soft_boundary_margin": 1.25,
        }
        recorder = MOD.RouteForensicsRecorder(Manager(), lambda: geometry, lambda: 0)
        recorder.routes[0] = np.array([[1.0, 1.0], [2.0, 1.0]])
        recorder.route_clearances[0] = {
            "route_hard_min_segment_clearance_m": 91.0,
            "route_soft_min_segment_clearance_m": 92.0,
        }
        recorder.plan(np.array([0]), np.array([[1.0, 1.0]]), geometry["bars"], geometry["bar_half"],
                      geometry["bounds_lo"], geometry["bounds_hi"], geometry["support"],
                      is_replan=True, before_routes=dict(recorder.routes))
        self.assertEqual(recorder.events[-1]["route_hard_min_segment_clearance_m"], 91.0)
        self.assertNotEqual(recorder.route_clearances[0]["route_hard_min_segment_clearance_m"], 91.0)


if __name__ == "__main__":
    unittest.main()
