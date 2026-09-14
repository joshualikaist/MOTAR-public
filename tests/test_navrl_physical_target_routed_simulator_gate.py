"""CPU-only integrity and verdict tests for the routed physical-target simulator gate."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/verify_navrl_physical_target_routed_simulator_gate.py"
SPEC = importlib.util.spec_from_file_location("routed_simulator_gate", TOOL)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)
ROUTE_PATH = ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"
ROUTE_SPEC = importlib.util.spec_from_file_location("routed_gate_route_module", ROUTE_PATH)
ROUTE = importlib.util.module_from_spec(ROUTE_SPEC)
sys.modules[ROUTE_SPEC.name] = ROUTE
ROUTE_SPEC.loader.exec_module(ROUTE)


def cell(route, speed, bars, *, passing=True):
    good = {
        "mean_speed_ratio": 0.9,
        "tracking_rmse_mps": 0.2,
        "contact_step_fraction": 0.0,
        "motor_saturation_fraction": 0.1,
        "max_tilt_deg": 40.0,
        "invalid_state_fraction": 0.0,
    }
    if not passing:
        good["tracking_rmse_mps"] = 0.5
    counter = {key: 0 for key in MOD.ROUTE_COUNTER_KEYS}
    counter.update({
        "plan_attempts": 32, "plan_successes": 32, "fallback_intervals": 0,
        "goal_completions": 16 if (bars, speed) == (70, 0.6) else 2,
        "same_goal_reselection_count": 0,
    })
    row = {
        "record_id": MOD.record_id(route, speed, bars),
        "seed": MOD.SEED, "route_mode": route, "speed_mps": speed, "bars": bars,
        "envs": MOD.ENVS, "steps": MOD.STEPS, "warmup_steps": MOD.WARMUP_STEPS,
        "route_goal_exclusion_m": 1.0,
        "bar_offset": 1,
        "active_bar_aabb_count": bars,
        "initial_layout_sha256": (f"{bars:04x}" * 16)[:64],
        "initial_robot_pose_sha256": (f"{bars + 1:04x}" * 16)[:64],
        "initial_target_pose_sha256": (f"{bars + 2:04x}" * 16)[:64],
        "initial_task_waypoint_sha256": (
            (f"{bars + (3 if route == 'off' else 4):04x}" * 16)[:64]
        ),
        "initial_route_goal_sha256": (
            (f"{bars + 5:04x}" * 16)[:64] if route == "global_astar_v1" else None
        ),
        "commanded_env_intervals": MOD.ENVS * MOD.STEPS,
        "tracking_measurement_env_intervals": MOD.ENVS * (MOD.STEPS - MOD.WARMUP_STEPS),
        "safety_measurement_env_intervals": MOD.ENVS * MOD.STEPS,
        "position_measurement_env_intervals": MOD.ENVS * MOD.STEPS,
        "failed_reset_monitoring_env_intervals": MOD.ENVS * MOD.STEPS,
        "task_clock": {
            "start_num_task_steps": 0, "end_num_task_steps": MOD.STEPS,
            "increments": MOD.STEPS,
        },
        "neutral_pursuer_command_contract": {
            "policy_action": "all_zero_[N,4]",
            "mapping": "NavRLTask.transform_action_to_command",
            "mapping_order": "after_target_advance_before_sim_step",
            "mapping_calls": MOD.STEPS,
            "command_shape": [MOD.ENVS, 4],
            "all_commands_finite": True,
        },
        "route": {
            "mode": route,
            "counter_delta": counter if route == "global_astar_v1" else {},
            "goal_completions_per_env": counter["goal_completions"] / MOD.ENVS
            if route == "global_astar_v1" else None,
            "initial_reset_included": True,
        },
        "mean_speed_mps": 0.8,
        "reset_wall": {"batches": 1, "total_s": 0.1},
        "throughput": {"rollout_wall_s": 1.0},
        "import_origin": {"enforced": True, "sha256": "a", "manifest_sha256": "a"},
        **good,
    }
    row["off_bounded_local_step_infeasible_fraction"] = 0.0 if route == "off" else None
    row["routed_local_step_invalidation_fraction"] = (
        0.0 if route == "global_astar_v1" else None
    )
    row["record_contract_sha256"] = MOD.record_contract_sha256(row)
    row["gates"] = MOD.physical_gate_metrics(row)
    row["pass"] = all(row["gates"].values())
    return row


def full_grid():
    return [
        cell(route, speed, bars)
        for route in MOD.ROUTE_ARMS for speed in MOD.SPEEDS for bars in MOD.DENSITIES
    ]


class RoutedSimulatorGateContractTest(unittest.TestCase):
    def test_frozen_grid_and_existing_gates(self):
        self.assertEqual(MOD.SEED, 827)
        self.assertEqual(MOD.ROUTE_ARMS, ("off", "global_astar_v1"))
        self.assertEqual(MOD.SPEEDS, (0.6, 0.9, 1.2, 1.5))
        self.assertEqual(MOD.DENSITIES, (70, 150, 205, 300))
        self.assertEqual((MOD.ENVS, MOD.STEPS, MOD.WARMUP_STEPS), (32, 300, 20))
        self.assertEqual(MOD.GATES["tracking_rmse_mps_max"], 0.35)
        self.assertEqual(MOD.GATES["invalid_state_fraction_max"], 0.0)
        self.assertEqual(MOD.frozen_environment("off", 0.6)["AERIAL_GYM_SIM_NAME"], "base_sim")

    def test_counter_delta_preserves_nested_future_counters_and_excludes_gauges(self):
        before = {
            "mode": "global_astar_v1", "plan_attempts": 2,
            "invalidation_counts": {"local": 1, "connector_reject": 3},
            "currently_valid": 20, "max_batch_wall_s": 0.2,
        }
        after = {
            "mode": "global_astar_v1", "plan_attempts": 7,
            "invalidation_counts": {"local": 2, "connector_reject": 7},
            "currently_valid": 31, "max_batch_wall_s": 0.3,
        }
        delta = MOD.recursive_counter_delta(before, after)
        self.assertEqual(delta["plan_attempts"], 5)
        self.assertEqual(delta["invalidation_counts"]["connector_reject"], 4)
        self.assertNotIn("currently_valid", delta)
        with self.assertRaises(MOD.IntegrityError):
            MOD.recursive_counter_delta({"count": 5}, {"count": 4})

    def test_complete_grid_yields_all_three_pass_verdicts(self):
        records = full_grid()
        MOD.validate_grid_records(records)
        verdicts = MOD.derive_verdicts(records, True)
        self.assertEqual(verdicts["route_mechanism"], "PASS_ROUTE_MECHANISM")
        self.assertEqual(verdicts["full_1p5_contract"], "PASS_FULL_1P5_CONTRACT")
        self.assertEqual(
            verdicts["density_conditioned_envelope"], "PASS_DENSITY_CONDITIONED_ENVELOPE"
        )
        self.assertFalse(verdicts["long_training_authority"])
        self.assertEqual(len(MOD.matched_deltas(records)), 16)
        self.assertNotIn(
            "routed_local_step_invalidation_fraction",
            MOD.matched_deltas(records)[0]["deltas"],
        )

    def test_low_level_clock_mirrors_task_step_and_releases_ten_step_cooldown(self):
        class FakeTask:
            num_task_steps = 0

        task = FakeTask()
        manager = ROUTE.BatchedTargetRouteManager(
            1, "cpu", ROUTE.RoutePlannerConfig(replan_cooldown_steps=10)
        )
        manager.valid[:] = True
        manager.goal[:] = 0.0
        manager.planned_support[:] = 0.2
        manager.invalidate(torch.tensor([True]), "local_step_infeasible", current_step=0)
        goal = torch.zeros((1, 2))
        support = torch.full((1, 2), 0.2)
        observed_steps = []
        for _ in range(10):
            start = MOD.begin_low_level_evaluation_interval(task)
            observed_steps.append(start)
            self.assertFalse(bool(manager.needs_replan(goal, support, start).item()))
            MOD.finish_low_level_evaluation_interval(task, start)
        self.assertEqual(observed_steps, list(range(10)))
        self.assertEqual(task.num_task_steps, 10)
        self.assertTrue(bool(manager.needs_replan(goal, support, task.num_task_steps).item()))
        with self.assertRaises(MOD.IntegrityError):
            MOD.finish_low_level_evaluation_interval(task, 9)

    def test_neutral_pursuer_uses_canonical_mapping_after_target_advance(self):
        events = []

        class Controller:
            def begin_control_interval(self):
                events.append("target_begin")

        class FakeTask:
            num_task_steps = 4
            _target_controller = Controller()

            def _advance_target(self):
                events.append("target_advance")

            def transform_action_to_command(self, action):
                events.append("canonical_action_map")
                return action + 0.25

        action = torch.zeros((2, 4))
        start, command = MOD.prepare_neutral_pursuer_interval(FakeTask(), action)
        self.assertEqual(start, 4)
        self.assertEqual(events, ["target_begin", "target_advance", "canonical_action_map"])
        torch.testing.assert_close(command, torch.full((2, 4), 0.25))

    def test_child_environment_repairs_hostile_path_and_requires_conda_ninja(self):
        hostile = {
            "PATH": "/hostile/only", "NAVRL_STALE": "1",
            "AERIAL_GYM_SIM_NAME": "base_sim_4gb", "KEEP_ME": "yes",
        }
        child = MOD.build_child_environment(hostile, sys.executable)
        expected_bin = str(Path(sys.executable).absolute().parent)
        self.assertEqual(child["PATH"].split(":"), [expected_bin, "/hostile/only"])
        self.assertNotIn("NAVRL_STALE", child)
        self.assertNotIn("AERIAL_GYM_SIM_NAME", child)
        self.assertEqual(child["KEEP_ME"], "yes")
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "python"
            fake_python.touch()
            with self.assertRaises(MOD.IntegrityError):
                MOD.build_child_environment(hostile, str(fake_python))

    def test_mechanism_or_missing_density_fails_closed(self):
        records = full_grid()
        routed70 = next(
            row for row in records
            if row["route_mode"] == "global_astar_v1" and row["bars"] == 70
        )
        routed70["route"]["counter_delta"]["same_goal_reselection_count"] = 1
        verdicts = MOD.derive_verdicts(records, True)
        self.assertEqual(verdicts["route_mechanism"], "FAIL_ROUTE_MECHANISM")
        self.assertEqual(verdicts["physical_training"], "BLOCKED_PHYSICAL_TRAINING")
        with self.assertRaises(MOD.IntegrityError):
            MOD.validate_grid_records(records[:-1])
        void = MOD.derive_verdicts([], False)
        self.assertEqual(void["execution_integrity"], "VOID_EXECUTION")
        self.assertEqual(void["physical_training"], "BLOCKED_PHYSICAL_TRAINING")

    def test_safety_denominator_and_record_identity_fail_closed(self):
        records = full_grid()
        records[0]["safety_measurement_env_intervals"] -= MOD.ENVS * MOD.WARMUP_STEPS
        with self.assertRaises(MOD.IntegrityError):
            MOD.validate_grid_records(records)
        records = full_grid()
        records[0]["record_id"] = MOD.record_id("off", 0.9, 70)
        with self.assertRaises(MOD.IntegrityError):
            MOD.validate_grid_records(records)

    def test_density_conditioned_pass_does_not_unblock_full_speed_launcher(self):
        records = full_grid()
        for row in records:
            if (
                row["route_mode"] == "global_astar_v1"
                and row["speed_mps"] == 1.5 and row["bars"] == 300
            ):
                row["tracking_rmse_mps"] = 0.5
                row["gates"] = MOD.physical_gate_metrics(row)
                row["pass"] = all(row["gates"].values())
        verdicts = MOD.derive_verdicts(records, True)
        self.assertEqual(
            verdicts["density_conditioned_envelope"], "PASS_DENSITY_CONDITIONED_ENVELOPE"
        )
        self.assertEqual(verdicts["full_1p5_contract"], "FAIL_FULL_1P5_CONTRACT")
        self.assertEqual(verdicts["physical_training"], "BLOCKED_PHYSICAL_TRAINING")
        self.assertEqual(
            verdicts["density_conditioned_training"],
            "ELIGIBLE_FOR_SEPARATE_DENSITY_CONDITIONED_PREREGISTRATION",
        )

    def test_source_manifest_root_and_bytes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_manifest.json"
            entry = {
                "path": str(TOOL.relative_to(ROOT)),
                "sha256": MOD.sha256_file(TOOL),
                "size": TOOL.stat().st_size,
            }
            payload = {
                "schema": MOD.SOURCE_SCHEMA, "repository_root": str(ROOT),
                "runtime_file_count": 1, "runtime_files": [entry],
            }
            MOD.atomic_json(path, payload)
            digest = MOD.sha256_file(path)
            self.assertEqual(MOD.verify_source_manifest(path, digest)["schema"], MOD.SOURCE_SCHEMA)
            payload["repository_root"] = "/tmp/wrong-root"
            MOD.atomic_json(path, payload)
            with self.assertRaises(MOD.IntegrityError):
                MOD.verify_source_manifest(path, MOD.sha256_file(path))

    def test_verify_mode_recomputes_cells_verdicts_and_bound_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            bound_paths = (
                TOOL,
                ROOT / "aerial_gym/__init__.py",
                ROOT / "aerial_gym/task/navrl_task/target_route_planner.py",
                ROOT / "aerial_gym/task/navrl_task/navrl_task.py",
                ROOT / "aerial_gym/config/task_config/navrl_task_config.py",
                ROOT / "aerial_gym/config/sim_config/base_sim_config.py",
                ROOT / "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
                ROOT / "resources/robots/quad/quad_navrl_ref5in.urdf",
            )
            entries = [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": MOD.sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in bound_paths
            ]
            source = {
                "schema": MOD.SOURCE_SCHEMA, "repository_root": str(ROOT),
                "runtime_file_count": len(entries), "runtime_files": entries,
            }
            source_path = directory / "source_manifest.json"
            MOD.atomic_json(source_path, source)
            source_sha = MOD.sha256_file(source_path)
            hashes = MOD.source_hash_map(source)
            origin_sha = MOD.sha256_file(ROOT / "aerial_gym/__init__.py")
            records = full_grid()
            for row in records:
                row["import_origin"] = {
                    "enforced": True, "sha256": origin_sha,
                    "manifest_sha256": origin_sha,
                }
            child_entries = []
            instantiated_contracts = []
            software_provenance = []
            for route in MOD.ROUTE_ARMS:
                for speed in MOD.SPEEDS:
                    child_path = directory / f"child_{route}_{speed}.json"
                    cells = [
                        row for row in records
                        if row["route_mode"] == route and row["speed_mps"] == speed
                    ]
                    instantiated = {
                        "physics": {
                            "physics_dt_s": 0.01, "physics_steps_per_rl_step": 10,
                            "rl_step_dt_s": 0.1,
                        },
                        "sim": {
                            "name": "base_sim", "config_class": "BaseSimConfig",
                            "config_path": "aerial_gym/config/sim_config/base_sim_config.py",
                            "config_sha256": hashes[
                                "aerial_gym/config/sim_config/base_sim_config.py"
                            ],
                        },
                        "robot": {
                            "robot_name": "navrl_ref5in_quad",
                            "robot_config_path": (
                                "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py"
                            ),
                            "robot_config_sha256": hashes[
                                "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py"
                            ],
                            "robot_asset_path": "resources/robots/quad/quad_navrl_ref5in.urdf",
                            "robot_asset_sha256": hashes[
                                "resources/robots/quad/quad_navrl_ref5in.urdf"
                            ],
                        },
                        "physical_target_box_xyz_m": [0.28, 0.28, 0.12],
                        "declared_conservative_support_xy_m": [
                            0.2068816086567407, 0.2068816086567407
                        ],
                        "active_route_support_xy_m": (
                            [0.2068816086567407, 0.2068816086567407]
                            if route == "global_astar_v1" else None
                        ),
                        "bar_offset": 1,
                    }
                    external_path = str(TOOL.resolve())
                    external_sha = MOD.sha256_file(TOOL)
                    ninja_path = MOD.require_conda_ninja(sys.executable)
                    ninja_version = subprocess.run(
                        [str(ninja_path), "--version"], text=True,
                        stdout=subprocess.PIPE, check=True,
                    ).stdout.strip()
                    provenance = {
                        "python": {
                            "executable": external_path, "executable_sha256": external_sha,
                            "version": "3.8-test", "implementation": "CPython",
                        },
                        "torch": {
                            "version": "test", "origin": external_path,
                            "origin_sha256": external_sha, "compiled_cuda_version": "test",
                        },
                        "isaac_gym": {"origin": external_path, "origin_sha256": external_sha},
                        "ninja": {
                            "path": str(ninja_path), "sha256": MOD.sha256_file(ninja_path),
                            "version": ninja_version,
                        },
                        "cuda": {
                            "available": True, "device_count": 1, "current_device": 0,
                            "gpu_names": ["test-gpu"], "driver_versions": ["test-driver"],
                        },
                        "repo_modules": {
                            "navrl_task": {
                                "relative_path": "aerial_gym/task/navrl_task/navrl_task.py",
                                "sha256": hashes["aerial_gym/task/navrl_task/navrl_task.py"],
                                "manifest_sha256": hashes[
                                    "aerial_gym/task/navrl_task/navrl_task.py"
                                ],
                            },
                            "target_route_planner": {
                                "relative_path": (
                                    "aerial_gym/task/navrl_task/target_route_planner.py"
                                ),
                                "sha256": hashes[
                                    "aerial_gym/task/navrl_task/target_route_planner.py"
                                ],
                                "manifest_sha256": hashes[
                                    "aerial_gym/task/navrl_task/target_route_planner.py"
                                ],
                            },
                        },
                    }
                    child = {
                        "schema": MOD.CHILD_SCHEMA, "route_mode": route, "speed_mps": speed,
                        "source_manifest_sha256": source_sha,
                        "environment_contract": MOD.frozen_environment(route, speed),
                        "import_origin": {
                            "enforced": True, "sha256": origin_sha,
                            "manifest_sha256": origin_sha,
                        },
                        "instantiated_contract": instantiated,
                        "software_provenance": provenance,
                        "cells": cells,
                    }
                    MOD.atomic_json(child_path, child)
                    child_entries.append({
                        "route_mode": route, "speed_mps": speed,
                        "summary": child_path.name,
                        "summary_sha256": MOD.sha256_file(child_path),
                    })
                    instantiated_contracts.append({
                        "route_mode": route, "speed_mps": speed, "contract": instantiated,
                    })
                    software_provenance.append({
                        "route_mode": route, "speed_mps": speed, "provenance": provenance,
                    })
            off_payload = json.loads(
                (directory / "child_off_0.6.json").read_text(encoding="utf-8")
            )
            routed_payload = json.loads(
                (directory / "child_global_astar_v1_0.6.json").read_text(encoding="utf-8")
            )
            swapped_arm = json.loads(json.dumps(off_payload))
            swapped_arm["cells"][0] = routed_payload["cells"][0]
            swapped_arm_path = directory / "bad_swapped_arm.json"
            MOD.atomic_json(swapped_arm_path, swapped_arm)
            with self.assertRaises(MOD.IntegrityError):
                MOD.validate_child(swapped_arm_path, "off", 0.6, source_sha, hashes)
            swapped_density = json.loads(json.dumps(off_payload))
            swapped_density["cells"][0], swapped_density["cells"][1] = (
                swapped_density["cells"][1], swapped_density["cells"][0]
            )
            swapped_density_path = directory / "bad_swapped_density.json"
            MOD.atomic_json(swapped_density_path, swapped_density)
            with self.assertRaises(MOD.IntegrityError):
                MOD.validate_child(swapped_density_path, "off", 0.6, source_sha, hashes)
            execution = {
                "schema": MOD.EXECUTION_SCHEMA, "integrity_ok": True,
                "children": child_entries,
            }
            execution_path = directory / "execution_manifest.json"
            MOD.atomic_json(execution_path, execution)
            verdicts = MOD.derive_verdicts(records, True)
            summary = {
                "schema": MOD.SCHEMA, "source_manifest": source_path.name,
                "source_manifest_sha256": source_sha,
                "execution_manifest": execution_path.name,
                "execution_manifest_sha256": MOD.sha256_file(execution_path),
                "instantiated_contracts": instantiated_contracts,
                "software_provenance": software_provenance,
                "matched_route_on_minus_off": MOD.matched_deltas(records),
                "cells": records, "verdicts": verdicts,
            }
            summary_path = directory / "summary.json"
            MOD.atomic_json(summary_path, summary)
            runtime_keys = (
                "aerial_gym/task/navrl_task/target_route_planner.py",
                "aerial_gym/task/navrl_task/navrl_task.py",
                "aerial_gym/config/task_config/navrl_task_config.py",
                "aerial_gym/config/sim_config/base_sim_config.py",
                "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
                "resources/robots/quad/quad_navrl_ref5in.urdf",
            )
            receipt = {
                "schema": "navrl_physical_target_routed_gate_receipt_v2",
                "summary": summary_path.name,
                "summary_sha256": MOD.sha256_file(summary_path),
                "execution_manifest_sha256": MOD.sha256_file(execution_path),
                "source_manifest_sha256": source_sha,
                "evaluator_source_sha256": MOD.sha256_file(TOOL),
                "bound_runtime_sha256": {key: hashes[key] for key in runtime_keys},
                "record_count": 32,
                "record_ids": [row["record_id"] for row in records],
                "verdicts": verdicts,
            }
            MOD.atomic_json(directory / "receipt.json", receipt)
            self.assertEqual(MOD.verify_result(summary_path, "full_1p5"), 0)
            summary["matched_route_on_minus_off"][0]["deltas"]["mean_speed_ratio"] += 0.1
            MOD.atomic_json(summary_path, summary)
            receipt["summary_sha256"] = MOD.sha256_file(summary_path)
            MOD.atomic_json(directory / "receipt.json", receipt)
            with self.assertRaises(MOD.IntegrityError):
                MOD.verify_result(summary_path, "full_1p5")


if __name__ == "__main__":
    unittest.main()
