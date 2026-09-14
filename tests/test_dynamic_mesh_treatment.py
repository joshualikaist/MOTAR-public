"""CPU contracts for the preregistered D8 mesh-derived target observation."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types
import unittest

from history_helpers import require_research_history

import numpy as np
import torch  # Load before the Warp stub; torch inspection expects real module attributes.


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "aerial_gym/task/navrl_task/navrl_dynamic_mesh_treatment.py"
DETECTOR = ROOT / "aerial_gym/task/navrl_task/navrl_detector.py"
TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
PREREG = ROOT / "docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md"
RUNNER = ROOT / "tools/probe_dynamic_mesh_treatment.py"
V3 = ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf"
RESULT = ROOT / "results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/receipt.json"
RESULT_SHA = "92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee"
V3_SHA = "c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0"
_ABSENT = object()


def load_with_warp_stub():
    class Stub(types.ModuleType):
        def __getattr__(self, _name):
            return passthrough

    def passthrough(fn=None, **_kwargs):
        return fn if callable(fn) else passthrough

    saved = sys.modules.get("warp", _ABSENT)
    stub = Stub("warp")
    stub.kernel = passthrough
    sys.modules["warp"] = stub
    try:
        spec = importlib.util.spec_from_file_location("d8_treatment_cpu_test", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is _ABSENT:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = saved


class ModeAndMaterialContract(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get("NAVRL_DYNAMIC_MESH_TREATMENT")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("NAVRL_DYNAMIC_MESH_TREATMENT", None)
        else:
            os.environ["NAVRL_DYNAMIC_MESH_TREATMENT"] = self.saved

    def test_mode_is_off_by_default_and_unknown_values_fail_closed(self):
        module = load_with_warp_stub()
        os.environ.pop(module.FLAG, None)
        self.assertEqual(module.treatment_mode(), module.OFF)
        for value in ("", "0", "false", "no", "off"):
            os.environ[module.FLAG] = value
            self.assertEqual(module.treatment_mode(), module.OFF)
        for value in (module.MESH_FLAT, module.MESH_SHADED):
            os.environ[module.FLAG] = value
            self.assertEqual(module.treatment_mode(), value)
        for value in ("1", "true", "mesh", "shaded", "analytic_flat"):
            os.environ[module.FLAG] = value
            with self.assertRaises(ValueError, msg=value):
                module.treatment_mode()

    def test_material_gain_is_bounded_deterministic_and_not_a_hue_replacement(self):
        module = load_with_warp_stub()
        rgba = np.asarray([
            [0.18, 0.18, 0.20, 1.0],
            [0.15, 0.15, 0.16, 1.0],
            [0.45, 0.45, 0.47, 1.0],
            [0.72, 0.72, 0.75, 1.0],
        ])
        first = module.material_gains(rgba)
        second = module.material_gains(rgba.copy())
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(first.min()), 0.55, places=6)
        self.assertAlmostEqual(float(first.max()), 1.0, places=6)
        self.assertEqual(first.shape, (4,))
        np.testing.assert_array_equal(
            module.material_gains(np.ones((2, 4), dtype=np.float32) * 0.5),
            np.ones(2, dtype=np.float32),
        )

    def test_invalid_material_tables_are_refused(self):
        module = load_with_warp_stub()
        for value in (
            np.zeros((0, 4)),
            np.zeros((2, 3)),
            np.asarray([[np.nan, 0, 0, 1]]),
            np.asarray([[1.1, 0, 0, 1]]),
        ):
            with self.assertRaises(ValueError):
                module.material_gains(value)


class SourceBoundaryContract(unittest.TestCase):
    def test_preregistration_and_asset_are_pinned(self):
        text = PREREG.read_text()
        self.assertIn(V3_SHA, text)
        self.assertEqual(hashlib.sha256(V3.read_bytes()).hexdigest(), V3_SHA)
        for phrase in (
            "analytic_flat",
            "mesh_flat",
            "mesh_shaded",
            "TECHNICAL_GO",
            "TECHNICAL_NO_GO",
            "No learning is authorized",
        ):
            self.assertIn(phrase, text)

    def test_detector_import_is_lazy_and_default_allocates_nothing(self):
        source = DETECTOR.read_text()
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(
            any("dynamic_mesh_treatment" in (getattr(node, "module", "") or "") for node in imports)
        )
        self.assertIn("self._dynamic_mesh_treatment = None", source)
        self.assertIn('self.target_render_mode = "analytic_flat"', source)

    def test_attach_is_fail_closed_and_decoupled_mode_is_refused(self):
        source = DETECTOR.read_text()
        for phrase in (
            "refusing to attach a D8 observation treatment",
            "does not implement detect-resolution decoupling",
            "D8 treatment and D7 shadow cannot be attached together",
            "D7 shadow and D8 treatment cannot be attached together",
            "a D8 dynamic-mesh treatment is already attached",
        ):
            self.assertIn(phrase, source)

    def test_kernel_has_separate_dynamic_and_static_queries_and_no_randomness(self):
        source = MODULE.read_text()
        tree = ast.parse(source)
        kernel = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "render_dynamic_mesh_target_kernel"
        )
        calls = [
            node for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mesh_query_ray"
        ]
        self.assertEqual(len(calls), 2)
        kernel_source = ast.get_source_segment(source, kernel)
        self.assertIn("static_mesh_ids[env_id]", kernel_source)
        self.assertIn("target_mesh", kernel_source)
        for forbidden in ("random", "rand(", "randn", "seed("):
            self.assertNotIn(forbidden, source)

    def test_debug_geometry_cannot_enter_the_task_observation(self):
        task_source = TASK.read_text()
        for name in (
            "target_normal",
            "target_face",
            "target_material",
            "target_shade",
            "raw_mesh_hit",
            "scene_occluded",
        ):
            self.assertNotIn(name, task_source)
        detector = DETECTOR.read_text()
        self.assertNotIn('obs_dict["target_normal"]', detector)
        self.assertNotIn('obs_dict["target_face"]', detector)

    def test_mesh_modes_share_mask_and_depth_and_only_shaded_mode_modulates_rgb(self):
        source = MODULE.read_text()
        tree = ast.parse(source)
        method = next(
            child for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DynamicMeshTargetTreatment"
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "target_rgb"
        )
        body = ast.get_source_segment(source, method)
        self.assertIn("if self.mode == MESH_FLAT", body)
        self.assertIn("self.target_shade.unsqueeze(1)", body)
        run_method = next(
            child for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DynamicMeshTargetTreatment"
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "run"
        )
        self.assertNotIn("self.mode", ast.get_source_segment(source, run_method))


def load_runner():
    spec = importlib.util.spec_from_file_location("d8_treatment_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_pose_run(module, geometry_effect=True, shaded_variance=1.0e-3):
    poses = []
    for distance in (4.5, 5.5, 6.5):
        for _ in range(7):
            poses.append({
                "distance_m": distance,
                "occlusion": False,
                "analytic_pixels": 20,
                "mesh_pixels": 15 if geometry_effect else 20,
                "mesh_flat_red_variance": 0.0,
                "mesh_shaded_red_variance": shaded_variance,
            })
    poses.append({
        "distance_m": 4.0,
        "occlusion": True,
        "analytic_pixels": 0,
        "mesh_pixels": 0,
        "mesh_flat_red_variance": None,
        "mesh_shaded_red_variance": None,
    })
    valid = {
        "occluded_survivors": 0,
        "invalid_depth": 0,
        "invalid_face": 0,
        "invalid_material": 0,
        "invalid_normal": 0,
        "scene_occluded_hits": 4,
    }
    return {
        "hashes": {arm: {"mask": arm, "depth": arm} for arm in module.ARMS},
        "mesh_flat_shaded_mask_equal": True,
        "mesh_flat_shaded_depth_equal": True,
        "poses": poses,
        "treatment_diagnostics": {"mesh_flat": dict(valid), "mesh_shaded": dict(valid)},
        "runtime": {"python": "fixed"},
        "launcher": {"python": "fixed", "ninja": "fixed"},
    }


def fake_step_run(mode, median_ms=10.0, reserved_mib=500.0):
    return {
        "hashes": {
            "target_mask": mode,
            "target_depth": mode,
            "raw_rgb": mode,
            "raw_depth": mode,
            "structured_observation": mode,
            "robot_position": "fixed-robot-trajectory",
            "target_position": "fixed-target-trajectory",
        },
        "runtime": {"python": "fixed"},
        "launcher": {"python": "fixed", "ninja": "fixed"},
        "timing": {"median_ms": median_ms, "steps_per_second": 1000.0 / median_ms},
        "torch_peak_reserved_mib": reserved_mib,
        "device_memory_after_mib": 800.0,
    }


class RunnerDecisionContract(unittest.TestCase):
    def setUp(self):
        self.module = load_runner()

    def reports(self, mesh_ms=10.5, reserved_delta=20.0):
        pose = fake_pose_run(self.module)
        pose_runs = [pose, {**pose, "poses": [dict(row) for row in pose["poses"]]}]
        step_runs = {}
        for mode in self.module.ARMS:
            milliseconds = 10.0 if mode == "analytic_flat" else mesh_ms
            reserved = 500.0 if mode == "analytic_flat" else 500.0 + reserved_delta
            row = fake_step_run(mode, milliseconds, reserved)
            step_runs[mode] = [row, {**row, "hashes": dict(row["hashes"])}]
        return pose_runs, step_runs

    def test_all_preregistered_gates_produce_technical_go(self):
        pose_runs, step_runs = self.reports()
        result = self.module.evaluate(pose_runs, step_runs)
        self.assertEqual(result["verdict"], "TECHNICAL_GO")
        self.assertTrue(result["integrity_pass"])
        self.assertTrue(result["go_cost"])
        self.assertFalse(result["no_go_cost"])

    def test_correctness_failure_is_technical_no_go_even_when_cost_is_small(self):
        pose_runs, step_runs = self.reports()
        pose_runs[0]["poses"][0]["mesh_shaded_red_variance"] = 0.0
        for row in pose_runs[0]["poses"]:
            if not row["occlusion"]:
                row["mesh_shaded_red_variance"] = 0.0
        pose_runs[1] = {**pose_runs[0], "poses": [dict(row) for row in pose_runs[0]["poses"]]}
        result = self.module.evaluate(pose_runs, step_runs)
        self.assertEqual(result["verdict"], "TECHNICAL_NO_GO")
        self.assertFalse(result["checks"]["shaded_target_variance_nonzero"])

    def test_cost_no_go_threshold_is_not_reclassified_as_inconclusive(self):
        pose_runs, step_runs = self.reports(mesh_ms=14.0)
        result = self.module.evaluate(pose_runs, step_runs)
        self.assertEqual(result["verdict"], "TECHNICAL_NO_GO")
        self.assertTrue(result["no_go_cost"])

    def test_intermediate_cost_is_inconclusive(self):
        pose_runs, step_runs = self.reports(mesh_ms=11.5)
        result = self.module.evaluate(pose_runs, step_runs)
        self.assertEqual(result["verdict"], "TECHNICAL_INCONCLUSIVE")
        self.assertTrue(result["integrity_pass"])
        self.assertFalse(result["go_cost"])
        self.assertFalse(result["no_go_cost"])

    def test_runner_refuses_dirty_tree_and_records_no_policy(self):
        source = RUNNER.read_text()
        for phrase in (
            "D8 execution requires a clean tracked/untracked working tree",
            '"merge-base", "--is-ancestor"',
            '"fixed_actions": True',
            '"policy_loaded": False',
            '"TECHNICAL_NO_GO_UNFINISHED"',
            'child_env["PATH"]',
            'child_env["NAVRL_NINJA"]',
            '"ninja_sha256"',
        ):
            self.assertIn(phrase, source)


class EvidenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = RESULT.read_bytes()
        cls.receipt = json.loads(cls.raw)

    def test_completed_receipt_is_the_pinned_technical_go(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), RESULT_SHA)
        self.assertEqual(self.receipt["schema"], "dynamic_mesh_detector_d8a_v1")
        self.assertEqual(self.receipt["decision"]["verdict"], "TECHNICAL_GO")
        self.assertTrue(self.receipt["decision"]["integrity_pass"])
        self.assertTrue(all(self.receipt["decision"]["checks"].values()))
        self.assertFalse(self.receipt["protocol"]["policy_loaded"])
        self.assertTrue(self.receipt["protocol"]["fixed_actions"])

    def test_receipt_sources_match_the_recorded_clean_commit_not_the_later_tree(self):

        require_research_history(ROOT)
        commit = self.receipt["provenance"]["head"]
        self.assertEqual(commit, "c50a26d86f820611da879a84e3a252c787bd3dd9")
        for path, expected in self.receipt["source_sha256"].items():
            content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected, path)

    def test_cost_contrasts_recompute_from_raw_runs(self):
        runs = self.receipt["step_runs"]
        aggregate = {
            mode: {
                "milliseconds": statistics.median(row["timing"]["median_ms"] for row in rows),
                "steps_per_second": statistics.median(
                    row["timing"]["steps_per_second"] for row in rows
                ),
                "reserved": max(row["torch_peak_reserved_mib"] for row in rows),
            }
            for mode, rows in runs.items()
        }
        base = aggregate["analytic_flat"]
        for mode in ("mesh_flat", "mesh_shaded"):
            actual = self.receipt["decision"]["contrasts"][mode]
            delta = aggregate[mode]["milliseconds"] - base["milliseconds"]
            throughput = 100.0 * (
                base["steps_per_second"] - aggregate[mode]["steps_per_second"]
            ) / base["steps_per_second"]
            self.assertAlmostEqual(actual["absolute_median_increase_ms"], delta, places=12)
            self.assertAlmostEqual(
                actual["relative_median_increase_percent"],
                100.0 * delta / base["milliseconds"],
                places=12,
            )
            self.assertAlmostEqual(actual["throughput_loss_percent"], throughput, places=12)
            self.assertAlmostEqual(
                actual["torch_reserved_increase_mib"],
                aggregate[mode]["reserved"] - base["reserved"],
                places=12,
            )

    def test_pose_counts_and_debug_validity_recompute(self):
        pose = self.receipt["pose_runs"][0]
        visible = [row for row in pose["poses"] if not row["occlusion"]]
        self.assertEqual(len(visible), 21)
        self.assertEqual(len(pose["poses"]) - len(visible), 1)
        self.assertTrue(any(row["analytic_pixels"] != row["mesh_pixels"] for row in visible))
        self.assertEqual(
            statistics.median(row["area_ratio"] for row in visible),
            0.5555555555555556,
        )
        for diagnostics in pose["treatment_diagnostics"].values():
            self.assertEqual(diagnostics["visible_hits"], 152)
            self.assertEqual(diagnostics["scene_occluded_hits"], 12)
            for key in (
                "occluded_survivors", "invalid_depth", "invalid_face",
                "invalid_material", "invalid_normal",
            ):
                self.assertEqual(diagnostics[key], 0, key)


if __name__ == "__main__":
    unittest.main()
