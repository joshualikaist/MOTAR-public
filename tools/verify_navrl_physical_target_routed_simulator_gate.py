#!/usr/bin/env python3
"""Run the frozen routed physical-target simulator engineering gate.

The parent process creates a source manifest and launches one fresh Isaac Gym process for every
route-mode x speed arm.  Each child evaluates all four densities, emitting per-cell counter deltas.
No pursuer policy or checkpoint is loaded.  This file is safe to import for CPU-only contract tests;
Isaac Gym and torch are imported only inside the hidden child entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs/preregistration_physical_target_routed_simulator_gate_2026-08-25.md"
SCHEMA = "navrl_physical_target_routed_simulator_gate_v2"
CHILD_SCHEMA = "navrl_physical_target_routed_simulator_child_v2"
SOURCE_SCHEMA = "navrl_physical_target_routed_source_manifest_v1"
EXECUTION_SCHEMA = "navrl_physical_target_routed_execution_manifest_v1"

SEED = 827
ROUTE_ARMS = ("off", "global_astar_v1")
SPEEDS = (0.6, 0.9, 1.2, 1.5)
DENSITIES = (70, 150, 205, 300)
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20
DEFAULT_OUTPUT = ROOT / "results/navrl_physical_target_routed_gate_seed827/summary.json"

GATES = {
    "tracking_rmse_mps_max": 0.35,
    "mean_speed_ratio_min": 0.80,
    "contact_step_fraction_max": 0.01,
    "off_bounded_local_step_infeasible_fraction_max": 0.01,
    "routed_local_step_invalidation_fraction_max": 0.01,
    "motor_saturation_fraction_max": 0.15,
    "max_tilt_deg_max": 60.0,
    "invalid_state_fraction_max": 0.0,
}

ROUTE_COUNTER_KEYS = (
    "plan_attempts",
    "plan_successes",
    "replan_attempts",
    "connected_goal_replans",
    "no_path_count",
    "invalid_count",
    "fallback_intervals",
    "goal_completions",
    "same_goal_reselection_count",
    "expanded_nodes",
    "raw_waypoints",
    "smoothed_waypoints",
    "planning_batches",
    "planning_envs",
    "total_planning_wall_s",
)
ROUTE_GAUGE_KEYS = (
    "currently_valid",
    "status_counts",
    "max_batch_wall_s",
    "max_batch_size",
    "planning_wall_ms_per_env",
)


class IntegrityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    require(completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def runtime_source_paths() -> List[Path]:
    """Return the committed runtime surface; results and unrelated docs are excluded."""
    tracked = _git(
        "ls-files", "aerial_gym", "resources/robots/quad/quad_navrl_ref5in.urdf",
        "resources/models/environment_assets/bars",
        str(PREREG.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
    ).splitlines()
    paths = [ROOT / name for name in tracked if name]
    require(Path(__file__).resolve() in paths, "evaluator is not tracked in the source manifest")
    require(PREREG in paths, "preregistration is not tracked in the source manifest")
    require(all(path.is_file() for path in paths), "runtime source manifest contains a missing file")
    return sorted(set(paths))


def build_source_manifest() -> Dict[str, object]:
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    require(not dirty, f"tracked runtime is dirty; commit before evaluation: {dirty}")
    entries = []
    for path in runtime_source_paths():
        entries.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        })
    return {
        "schema": SOURCE_SCHEMA,
        "repository_root": str(ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "preregistration": str(PREREG.relative_to(ROOT)),
        "runtime_file_count": len(entries),
        "runtime_files": entries,
    }


def verify_source_manifest(path: Path, expected_sha: str) -> Dict[str, object]:
    require(path.is_file(), f"source manifest missing: {path}")
    require(sha256_file(path) == expected_sha, "source manifest digest changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == SOURCE_SCHEMA, "wrong source manifest schema")
    require(Path(str(payload.get("repository_root", ""))).resolve() == ROOT, "source root drift")
    entries = payload.get("runtime_files")
    require(isinstance(entries, list) and len(entries) == payload.get("runtime_file_count"),
            "source manifest file count mismatch")
    seen = set()
    for entry in entries:
        relative = Path(str(entry.get("path", "")))
        require(relative and not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe source path: {relative}")
        require(str(relative) not in seen, f"duplicate source path: {relative}")
        seen.add(str(relative))
        source = ROOT / relative
        require(source.is_file(), f"runtime source disappeared: {relative}")
        require(source.stat().st_size == int(entry.get("size", -1)), f"source size drift: {relative}")
        require(sha256_file(source) == entry.get("sha256"), f"source hash drift: {relative}")
    return payload


def source_hash_map(manifest: Mapping) -> Dict[str, str]:
    return {str(entry["path"]): str(entry["sha256"]) for entry in manifest["runtime_files"]}


def attest_repo_module(module_name: str, expected_relative: str, source: Mapping) -> Dict[str, object]:
    module = importlib.import_module(module_name)
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    expected = (ROOT / expected_relative).resolve()
    require(path == expected, f"{module_name} import origin drift: {path}")
    hashes = source_hash_map(source)
    digest = sha256_file(path)
    require(digest == hashes.get(expected_relative), f"{module_name} imported bytes drift")
    return {
        "module": module_name, "path": str(path), "relative_path": expected_relative,
        "sha256": digest, "manifest_sha256": hashes[expected_relative],
    }


def runtime_software_provenance(torch, source: Mapping) -> Dict[str, object]:
    require(torch.cuda.is_available(), "CUDA is unavailable in routed simulator GPU gate")
    python_path = Path(sys.executable).resolve()
    torch_path = Path(str(torch.__file__)).resolve()
    isaacgym = importlib.import_module("isaacgym")
    isaac_path = Path(str(getattr(isaacgym, "__file__", ""))).resolve()
    require(python_path.is_file(), "Python executable is not a file")
    require(torch_path.is_file(), "torch module origin is not a file")
    require(isaac_path.is_file(), "Isaac Gym module origin is not a file")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    require(driver.returncode == 0, f"nvidia-smi driver query failed: {driver.stderr.strip()}")
    driver_versions = sorted(set(line.strip() for line in driver.stdout.splitlines() if line.strip()))
    require(driver_versions, "GPU driver version is empty")
    ninja_path = require_conda_ninja(sys.executable)
    ninja_version = subprocess.run(
        [str(ninja_path), "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    require(ninja_version.returncode == 0 and ninja_version.stdout.strip(),
            f"ninja version query failed: {ninja_version.stderr.strip()}")
    device_count = int(torch.cuda.device_count())
    require(device_count >= 1, "CUDA reports zero devices")
    return {
        "python": {
            "executable": str(python_path), "executable_sha256": sha256_file(python_path),
            "version": sys.version, "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": str(torch.__version__), "origin": str(torch_path),
            "origin_sha256": sha256_file(torch_path), "compiled_cuda_version": str(torch.version.cuda),
        },
        "isaac_gym": {"origin": str(isaac_path), "origin_sha256": sha256_file(isaac_path)},
        "ninja": {
            "path": str(ninja_path), "sha256": sha256_file(ninja_path),
            "version": ninja_version.stdout.strip(),
        },
        "cuda": {
            "available": True, "device_count": device_count,
            "current_device": int(torch.cuda.current_device()),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(device_count)],
            "driver_versions": driver_versions,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "repo_modules": {
            "navrl_task": attest_repo_module(
                "aerial_gym.task.navrl_task.navrl_task",
                "aerial_gym/task/navrl_task/navrl_task.py", source,
            ),
            "target_route_planner": attest_repo_module(
                "aerial_gym.task.navrl_task.target_route_planner",
                "aerial_gym/task/navrl_task/target_route_planner.py", source,
            ),
        },
    }


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def recursive_counter_delta(before: Mapping, after: Mapping) -> Dict[str, object]:
    """Subtract cumulative numeric diagnostics, preserving future reason/connector counters."""
    result: Dict[str, object] = {}
    for key, final in after.items():
        if key in ROUTE_GAUGE_KEYS or key == "mode":
            continue
        initial = before.get(key)
        if isinstance(final, Mapping):
            require(isinstance(initial, Mapping), f"route diagnostic shape drift at {key}")
            result[key] = recursive_counter_delta(initial, final)
        elif _is_number(final):
            require(_is_number(initial), f"route counter missing/nonfinite at {key}")
            delta = final - initial
            require(delta >= -1e-9, f"route counter decreased at {key}: {initial}->{final}")
            result[key] = delta
    return result


def normalized_route_diagnostics(task) -> Dict[str, object]:
    if task._target_route_manager is None:
        return {"mode": "off"}
    diagnostics = task._target_route_manager.diagnostics()
    require(diagnostics.get("mode") == "global_astar_v1", "routed manager reports wrong mode")
    for key in ROUTE_COUNTER_KEYS:
        require(_is_number(diagnostics.get(key)), f"required route counter missing: {key}")
    require(isinstance(diagnostics.get("invalidation_counts"), Mapping),
            "route invalidation counters missing")
    return diagnostics


def attest_instantiated_contract(task, route_mode: str, source: Mapping) -> Dict[str, object]:
    physics = task._runtime_physics_contract()
    require(abs(float(physics["physics_dt_s"]) - 0.01) < 1e-12, "runtime physics dt drift")
    require(int(physics["physics_steps_per_rl_step"]) == 10, "runtime physics-step count drift")
    require(abs(float(physics["rl_step_dt_s"]) - 0.1) < 1e-12, "runtime RL dt drift")
    require(task.task_config.sim_name == "base_sim", "runtime sim name is not base_sim")
    sim_config = task.sim_env.sim_config
    sim_config_path = Path(inspect.getsourcefile(sim_config) or "").resolve()
    expected_sim_config_path = (ROOT / "aerial_gym/config/sim_config/base_sim_config.py").resolve()
    require(sim_config_path == expected_sim_config_path, "runtime sim config source drift")
    hashes = source_hash_map(source)
    sim_relative = str(sim_config_path.relative_to(ROOT))
    require(sha256_file(sim_config_path) == hashes.get(sim_relative),
            "runtime sim config bytes differ from source manifest")
    robot = task._runtime_robot_provenance()
    require(robot["robot_name"] == "navrl_ref5in_quad", "runtime robot is not ref5in")
    require(robot["robot_config_sha256"] == hashes.get(robot["robot_config_path"]),
            "instantiated robot config differs from source manifest")
    require(robot["robot_asset_sha256"] == hashes.get(robot["robot_asset_path"]),
            "instantiated robot URDF differs from source manifest")
    box = [float(value) for value in task.tm.physical_box_xyz]
    require(box == [0.28, 0.28, 0.12], "physical target collision box drift")
    support = 0.5 * math.sqrt(sum(value * value for value in box))
    require(abs(support - 0.2068816086567407) < 1e-12, "conservative target support drift")
    active_support = None
    if route_mode == "global_astar_v1":
        active_support = [float(value) for value in task._target_route_support_xy[0].tolist()]
        require(all(abs(value - support) < 1e-6 for value in active_support),
                "active routed support differs from physical box envelope")
    require(int(task._bar_offset) == 1, "physical target bar offset must be one")
    return {
        "physics": physics,
        "sim": {
            "name": str(task.task_config.sim_name),
            "config_class": getattr(sim_config, "__name__", type(sim_config).__name__),
            "config_path": sim_relative,
            "config_sha256": sha256_file(sim_config_path),
        },
        "robot": robot,
        "physical_target_box_xyz_m": box,
        "declared_conservative_support_xy_m": [support, support],
        "active_route_support_xy_m": active_support,
        "bar_offset": int(task._bar_offset),
    }


def route_cell_delta(before: Mapping, after: Mapping, commanded_intervals: int) -> Dict[str, object]:
    mode = str(after.get("mode", ""))
    require(mode == before.get("mode"), "route mode changed inside a child process")
    if mode == "off":
        return {
            "mode": "off", "counter_delta": {}, "plan_success_fraction": None,
            "fallback_interval_fraction": None, "goal_completions_per_env": None,
            "planning_wall_s": 0.0, "planning_wall_ms_per_planned_env": None,
            "end_gauges": {}, "initial_reset_included": True,
        }
    delta = recursive_counter_delta(before, after)
    attempts = float(delta["plan_attempts"])
    successes = float(delta["plan_successes"])
    fallback = float(delta["fallback_intervals"])
    completions = float(delta["goal_completions"])
    planning_envs = float(delta["planning_envs"])
    planning_wall = float(delta["total_planning_wall_s"])
    return {
        "mode": mode,
        "counter_delta": delta,
        "plan_success_fraction": successes / attempts if attempts else 0.0,
        "fallback_interval_fraction": fallback / max(1, commanded_intervals),
        "goal_completions_per_env": completions / ENVS,
        "planning_wall_s": planning_wall,
        "planning_wall_ms_per_planned_env": (
            1000.0 * planning_wall / planning_envs if planning_envs else 0.0
        ),
        "end_gauges": {key: after.get(key) for key in ROUTE_GAUGE_KEYS if key in after},
        "initial_reset_included": True,
    }


def physical_gate_metrics(row: Mapping) -> Dict[str, bool]:
    if row["route_mode"] == "off":
        local_feasibility = (
            row["off_bounded_local_step_infeasible_fraction"]
            <= GATES["off_bounded_local_step_infeasible_fraction_max"]
        )
    else:
        local_feasibility = (
            row["routed_local_step_invalidation_fraction"]
            <= GATES["routed_local_step_invalidation_fraction_max"]
        )
    return {
        "tracking": row["tracking_rmse_mps"] <= GATES["tracking_rmse_mps_max"],
        "speed": row["mean_speed_ratio"] >= GATES["mean_speed_ratio_min"],
        "contact": row["contact_step_fraction"] <= GATES["contact_step_fraction_max"],
        "arm_specific_local_feasibility": local_feasibility,
        "motors": row["motor_saturation_fraction"] <= GATES["motor_saturation_fraction_max"],
        "tilt": row["max_tilt_deg"] <= GATES["max_tilt_deg_max"],
        "state": row["invalid_state_fraction"] <= GATES["invalid_state_fraction_max"],
    }


def record_id(route_mode: str, speed: float, density: int) -> str:
    return f"route_{route_mode}__speed_{speed:.1f}__bars_{density}"


def record_contract_sha256(row: Mapping) -> str:
    payload = {
        "record_id": row["record_id"], "route_mode": row["route_mode"],
        "speed_mps": row["speed_mps"], "bars": row["bars"], "seed": row["seed"],
        "initial_layout_sha256": row["initial_layout_sha256"],
        "initial_robot_pose_sha256": row["initial_robot_pose_sha256"],
        "initial_target_pose_sha256": row["initial_target_pose_sha256"],
        "initial_task_waypoint_sha256": row["initial_task_waypoint_sha256"],
        "initial_route_goal_sha256": row["initial_route_goal_sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_grid_records(records: Sequence[Mapping]) -> None:
    expected = {
        record_id(route, speed, density)
        for route in ROUTE_ARMS for speed in SPEEDS for density in DENSITIES
    }
    actual = [str(row.get("record_id", "")) for row in records]
    require(len(actual) == 32, f"expected 32 cell records, got {len(actual)}")
    require(len(set(actual)) == len(actual), "duplicate cell record")
    require(set(actual) == expected, f"grid mismatch missing={sorted(expected-set(actual))} extra={sorted(set(actual)-expected)}")
    for row in records:
        require(
            row.get("record_id")
            == record_id(str(row.get("route_mode")), float(row.get("speed_mps")), int(row.get("bars"))),
            "record_id does not match its route/speed/density fields",
        )
        require(row.get("seed") == SEED and row.get("envs") == ENVS, "seed/env contract drift")
        require(row.get("steps") == STEPS and row.get("warmup_steps") == WARMUP_STEPS,
                "step contract drift")
        require(row.get("route_goal_exclusion_m") == 1.0, "route goal exclusion drift")
        require(row.get("bar_offset") == 1, "cell physical-target bar offset drift")
        require(row.get("active_bar_aabb_count") == row.get("bars"),
                "cell active AABB count differs from density")
        for digest_name in (
            "initial_layout_sha256", "initial_robot_pose_sha256",
            "initial_target_pose_sha256", "initial_task_waypoint_sha256",
            "record_contract_sha256",
        ):
            require(isinstance(row.get(digest_name), str) and len(row[digest_name]) == 64,
                    f"initial/record digest missing: {digest_name}")
        route_goal_digest = row.get("initial_route_goal_sha256")
        require(
            (row["route_mode"] == "off" and route_goal_digest is None)
            or (row["route_mode"] == "global_astar_v1" and isinstance(route_goal_digest, str)
                and len(route_goal_digest) == 64),
            "route-goal digest arm contract drift",
        )
        require(row["record_contract_sha256"] == record_contract_sha256(row),
                "record contract digest mismatch")
        require(row.get("route_mode") in ROUTE_ARMS, "cell route mode invalid")
        route = row.get("route")
        require(isinstance(route, Mapping) and route.get("mode") == row["route_mode"],
                "cell route diagnostics mismatch")
        require(route.get("initial_reset_included") is True, "cell route delta excludes initial reset")
        metrics = (
            "mean_speed_mps", "mean_speed_ratio", "tracking_rmse_mps",
            "contact_step_fraction", "invalid_state_fraction", "motor_saturation_fraction",
            "max_tilt_deg",
        )
        local_key = (
            "off_bounded_local_step_infeasible_fraction" if row["route_mode"] == "off"
            else "routed_local_step_invalidation_fraction"
        )
        absent_local_key = (
            "routed_local_step_invalidation_fraction" if row["route_mode"] == "off"
            else "off_bounded_local_step_infeasible_fraction"
        )
        require(row.get(absent_local_key) is None,
                "inapplicable arm-specific local metric must be null")
        metrics += (local_key,)
        for key in metrics:
            require(_is_number(row.get(key)) and row[key] >= 0.0, f"invalid cell metric: {key}")
        for key in (
            "contact_step_fraction", local_key, "invalid_state_fraction",
            "motor_saturation_fraction",
        ):
            require(row[key] <= 1.0, f"fraction outside [0,1]: {key}")
        reset_wall = row.get("reset_wall")
        throughput = row.get("throughput")
        require(isinstance(reset_wall, Mapping) and reset_wall.get("batches", 0) >= 1,
                "reset wall telemetry missing")
        require(isinstance(throughput, Mapping) and throughput.get("rollout_wall_s", 0) > 0,
                "throughput telemetry missing")
        origin = row.get("import_origin")
        require(
            isinstance(origin, Mapping) and origin.get("enforced") is True
            and origin.get("sha256") == origin.get("manifest_sha256"),
            "cell import-origin evidence missing",
        )
        if row["route_mode"] == "global_astar_v1":
            counters = route.get("counter_delta")
            require(isinstance(counters, Mapping), "routed counter delta missing")
            for key in ROUTE_COUNTER_KEYS:
                require(_is_number(counters.get(key)) and counters[key] >= 0,
                        f"routed counter missing: {key}")
        require(row.get("gates") == physical_gate_metrics(row), "cell gate recomputation mismatch")
        require(row.get("pass") == all(row["gates"].values()), "cell conjunctive gate mismatch")
        require(row.get("tracking_measurement_env_intervals") == ENVS * (STEPS - WARMUP_STEPS),
                "tracking warmup denominator drift")
        require(row.get("safety_measurement_env_intervals") == ENVS * STEPS,
                "safety denominator does not cover all intervals")
        require(row.get("position_measurement_env_intervals") == ENVS * STEPS,
                "position envelope does not cover all intervals")
        require(row.get("failed_reset_monitoring_env_intervals") == ENVS * STEPS,
                "failed reset monitoring does not cover all intervals")
        clock = row.get("task_clock", {})
        require(
            clock.get("increments") == STEPS
            and clock.get("end_num_task_steps") - clock.get("start_num_task_steps") == STEPS,
            "task clock did not advance once per evaluation interval",
        )
        neutral = row.get("neutral_pursuer_command_contract", {})
        require(
            neutral.get("policy_action") == "all_zero_[N,4]"
            and neutral.get("mapping") == "NavRLTask.transform_action_to_command"
            and neutral.get("mapping_order") == "after_target_advance_before_sim_step"
            and neutral.get("mapping_calls") == STEPS
            and neutral.get("command_shape") == [ENVS, 4]
            and neutral.get("all_commands_finite") is True,
            "neutral pursuer canonical action-mapping contract drift",
        )
    for density in DENSITIES:
        digests = {row["initial_layout_sha256"] for row in records if row["bars"] == density}
        require(len(digests) == 1, f"matched-arm initial layout drift at {density} bars")
        for digest_name in ("initial_robot_pose_sha256", "initial_target_pose_sha256"):
            digests = {row[digest_name] for row in records if row["bars"] == density}
            require(len(digests) == 1, f"matched-arm {digest_name} drift at {density} bars")
        for route_mode in ROUTE_ARMS:
            waypoint_digests = {
                row["initial_task_waypoint_sha256"] for row in records
                if row["bars"] == density and row["route_mode"] == route_mode
            }
            require(len(waypoint_digests) == 1,
                    f"speed-arm initial waypoint drift at {route_mode}/{density} bars")
            if route_mode == "global_astar_v1":
                goal_digests = {
                    row["initial_route_goal_sha256"] for row in records
                    if row["bars"] == density and row["route_mode"] == route_mode
                }
                require(len(goal_digests) == 1,
                        f"speed-arm routed goal drift at {density} bars")


def matched_deltas(records: Sequence[Mapping]) -> List[Dict[str, object]]:
    index = {(row["speed_mps"], row["bars"], row["route_mode"]): row for row in records}
    metrics = (
        "mean_speed_ratio", "tracking_rmse_mps", "contact_step_fraction",
        "invalid_state_fraction",
    )
    rows = []
    for speed in SPEEDS:
        for density in DENSITIES:
            off = index[(speed, density, "off")]
            routed = index[(speed, density, "global_astar_v1")]
            rows.append({
                "speed_mps": speed, "bars": density,
                "delta_definition": "global_astar_v1_minus_off",
                "deltas": {key: routed[key] - off[key] for key in metrics},
            })
    return rows


def derive_verdicts(records: Sequence[Mapping], integrity_ok: bool) -> Dict[str, object]:
    if not integrity_ok:
        return {
            "execution_integrity": "VOID_EXECUTION",
            "route_mechanism": "NOT_INTERPRETED",
            "full_1p5_contract": "NOT_INTERPRETED",
            "density_conditioned_envelope": "NOT_INTERPRETED",
            "physical_training": "BLOCKED_PHYSICAL_TRAINING",
            "density_conditioned_training": "BLOCKED_PHYSICAL_TRAINING",
            "highest_passing_speed_mps_by_density": {str(v): None for v in DENSITIES},
        }
    validate_grid_records(records)
    routed70 = [r for r in records if r["route_mode"] == "global_astar_v1" and r["bars"] == 70]
    attempts = sum(r["route"]["counter_delta"]["plan_attempts"] for r in routed70)
    successes = sum(r["route"]["counter_delta"]["plan_successes"] for r in routed70)
    fallback = sum(r["route"]["counter_delta"]["fallback_intervals"] for r in routed70)
    reselections70 = sum(
        r["route"]["counter_delta"]["same_goal_reselection_count"] for r in routed70
    )
    commanded = sum(r["commanded_env_intervals"] for r in routed70)
    low = next(r for r in routed70 if r["speed_mps"] == 0.6)
    route_mechanism_pass = (
        attempts > 0 and successes / attempts >= 0.99
        and fallback / max(1, commanded) <= 0.01
        and reselections70 == 0
        and low["route"]["goal_completions_per_env"] >= 0.5
    )
    routed = [r for r in records if r["route_mode"] == "global_astar_v1"]
    reselections_all = sum(
        r["route"]["counter_delta"]["same_goal_reselection_count"] for r in routed
    )
    route_mechanism_pass = route_mechanism_pass and reselections_all == 0
    full_1p5 = route_mechanism_pass and all(
        r["pass"] for r in routed if r["speed_mps"] == 1.5
    )
    highest = {}
    for density in DENSITIES:
        passing = [r["speed_mps"] for r in routed if r["bars"] == density and r["pass"]]
        highest[str(density)] = max(passing) if passing else None
    envelope = route_mechanism_pass and all(value is not None for value in highest.values())
    return {
        "execution_integrity": "PASS_32_CELL_INTEGRITY",
        "route_mechanism": "PASS_ROUTE_MECHANISM" if route_mechanism_pass else "FAIL_ROUTE_MECHANISM",
        "full_1p5_contract": "PASS_FULL_1P5_CONTRACT" if full_1p5 else "FAIL_FULL_1P5_CONTRACT",
        "density_conditioned_envelope": (
            "PASS_DENSITY_CONDITIONED_ENVELOPE" if envelope
            else "FAIL_DENSITY_CONDITIONED_ENVELOPE"
        ),
        "physical_training": (
            "NOT_BLOCKED_FOR_SEPARATELY_PREREGISTERED_SHORT_SMOKE"
            if full_1p5 else "BLOCKED_PHYSICAL_TRAINING"
        ),
        "density_conditioned_training": (
            "ELIGIBLE_FOR_SEPARATE_DENSITY_CONDITIONED_PREREGISTRATION"
            if envelope else "BLOCKED_PHYSICAL_TRAINING"
        ),
        "highest_passing_speed_mps_by_density": highest,
        "route_mechanism_inputs": {
            "plan_success_fraction_70": successes / attempts if attempts else 0.0,
            "fallback_interval_fraction_70": fallback / max(1, commanded),
            "same_goal_reselection_count_70": reselections70,
            "same_goal_reselection_count_all_routed_cells": reselections_all,
            "goal_completions_per_env_70_speed_0p6": low["route"]["goal_completions_per_env"],
        },
        "long_training_authority": False,
    }


def frozen_environment(route_mode: str, speed: float) -> Dict[str, str]:
    return {
        "AERIAL_GYM_SIM_NAME": "base_sim",
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_ROUTE_MODE": route_mode,
        "NAVRL_TARGET_PATTERN": "waypoint",
        "NAVRL_TARGET_SPEED": str(speed),
        "NAVRL_TARGET_SPEED_FINAL": str(speed),
        "NAVRL_TARGET_SPEED_MIN": str(speed),
        "NAVRL_TARGET_SPEED_RAMP_EPOCHS": "1",
        "NAVRL_TARGET_MAX_ACCEL": "4.0",
        "NAVRL_TARGET_MAX_TURN_RATE_DEG": "150.0",
        "NAVRL_TARGET_LOOKAHEAD_S": "1.0",
        "NAVRL_TARGET_OBSTACLE_CLEARANCE": "0.77",
        "NAVRL_TARGET_MASS_KG": "1.20",
        "NAVRL_TARGET_MOTOR_ARM_XY_M": "0.0777817",
        "NAVRL_TARGET_MAX_MOTOR_THRUST_N": "9.60",
        "NAVRL_TARGET_MOTOR_TAU_S": "0.04",
        "NAVRL_TARGET_YAW_TORQUE_RATIO_M": "0.01",
        "NAVRL_TARGET_MAX_TILT_DEG": "45.0",
        "NAVRL_TARGET_VEL_KP": "2.5",
        "NAVRL_TARGET_ALT_KP": "4.0",
        "NAVRL_TARGET_TRACKING_MARGIN_M": "0.45",
        "NAVRL_TARGET_BOUNDARY_MARGIN_M": "0.75",
        "NAVRL_TARGET_ROUTE_RESOLUTION_M": "0.25",
        "NAVRL_TARGET_ROUTE_MAX_EXPANSIONS": "50000",
        "NAVRL_TARGET_ROUTE_MAX_WAYPOINTS": "128",
        "NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS": "10",
        "NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M": "0.05",
        "NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M": "6.0",
        "NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M": "1.0",
        "NAVRL_NUM_BARS": str(DENSITIES[0]),
        "NAVRL_MAX_BARS": str(max(DENSITIES)),
        "NAVRL_DENSITY_CURRICULUM": "0",
        "NAVRL_VISION": "0",
        "NAVRL_PERCEPTION": "0",
        "NAVRL_GENERAL_TRAIN": "1",
        "NAVRL_ARENA_XY": "40",
        "NAVRL_ARENA_Z": "3",
        "NAVRL_BAR_POOL": "bars_h3",
        "NAVRL_BAR_X_MIN": "0",
        "NAVRL_BAR_X_MAX": "1",
        "NAVRL_PLACEMENT_MODE": "navrl_band",
        "NAVRL_PLACEMENT_TOUCH_M": "0.4",
        "NAVRL_PLACEMENT_GAP_M": "1.6",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(ROOT),
    }


def configure_child(route_mode: str, speed: float) -> Dict[str, str]:
    values = frozen_environment(route_mode, speed)
    os.environ.update(values)
    return values


def require_conda_ninja(executable: str) -> Path:
    executable_path = Path(executable).absolute()
    ninja = executable_path.parent / "ninja"
    require(ninja.is_file(), f"ninja missing next to Python executable: {ninja}")
    require(os.access(str(ninja), os.X_OK), f"ninja is not executable: {ninja}")
    return ninja


def build_child_environment(parent_environment=None, executable: Optional[str] = None) -> Dict[str, str]:
    """Create a hermetic child env with the selected Python's build tools first on PATH."""
    source = dict(os.environ if parent_environment is None else parent_environment)
    python_executable = sys.executable if executable is None else executable
    ninja = require_conda_ninja(python_executable)
    python_bin = str(ninja.parent)
    old_path = str(source.get("PATH", ""))
    path_parts = [part for part in old_path.split(os.pathsep) if part and part != python_bin]
    source["PATH"] = os.pathsep.join([python_bin] + path_parts)
    source = {
        key: value for key, value in source.items()
        if not key.startswith("NAVRL_") and key != "AERIAL_GYM_SIM_NAME"
    }
    source.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT)})
    require(source["PATH"].split(os.pathsep)[0] == python_bin,
            "selected Python bin is not first on child PATH")
    return source


def _sync_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_reset(task, torch, env_ids=None) -> Dict[str, float]:
    _sync_cuda(torch)
    started = time.perf_counter()
    if env_ids is None:
        task.reset()
        count = ENVS
    else:
        task.sim_env.reset_idx(env_ids)
        task.reset_idx(env_ids)
        count = len(env_ids)
    _sync_cuda(torch)
    return {"wall_s": time.perf_counter() - started, "envs": count}


def initial_layout_sha256(task, density: int) -> str:
    centers = task.obs_dict["obstacle_position"][
        :, task._bar_offset : task._bar_offset + density, :3
    ].detach().cpu().contiguous()
    half = task.obs_dict["asset_collision_half_extents"][
        :, task._bar_offset : task._bar_offset + density, :3
    ].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(centers.shape)).encode("ascii"))
    digest.update(centers.numpy().tobytes())
    digest.update(str(tuple(half.shape)).encode("ascii"))
    digest.update(half.numpy().tobytes())
    return digest.hexdigest()


def tensor_digest_sha256(*tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def begin_low_level_evaluation_interval(task) -> int:
    """Capture the task clock used by _advance_target in canonical NavRLTask.step."""
    value = int(task.num_task_steps)
    require(value >= 0, "task clock is negative")
    return value


def finish_low_level_evaluation_interval(task, interval_start_step: int) -> None:
    """Mirror NavRLTask.step's single end-of-interval num_task_steps increment."""
    require(int(task.num_task_steps) == interval_start_step,
            "task clock changed outside the canonical evaluation increment")
    task.num_task_steps += 1
    require(int(task.num_task_steps) == interval_start_step + 1,
            "task clock did not advance exactly once")


def prepare_neutral_pursuer_interval(task, zero_policy_action):
    """Mirror NavRLTask.step's target-advance then policy-action mapping order."""
    interval_start_step = begin_low_level_evaluation_interval(task)
    task._target_controller.begin_control_interval()
    task._advance_target()
    command = task.transform_action_to_command(zero_policy_action)
    require(tuple(command.shape) == tuple(zero_policy_action.shape),
            "canonical neutral pursuer command shape drift")
    require(bool(command.isfinite().all()), "canonical neutral pursuer command is nonfinite")
    return interval_start_step, command


def run_cell(task, torch, route_mode: str, speed: float, density: int, import_origin: Mapping) -> Dict:
    # Re-establish the frozen seed before every density so route-specific RNG consumption in the
    # preceding density cannot silently change the next cell's initial bar layout.
    task.seed(SEED)
    route_before = normalized_route_diagnostics(task)
    _sync_cuda(torch)
    initial_started = time.perf_counter()
    task._set_active_bars(density)
    task.reset()
    _sync_cuda(torch)
    initial_reset_s = time.perf_counter() - initial_started
    layout_sha = initial_layout_sha256(task, density)
    initial_robot_pose_sha = tensor_digest_sha256(
        task.obs_dict["robot_position"], task.obs_dict["robot_orientation"]
    )
    initial_target_pose_sha = tensor_digest_sha256(task.target_position, task.target_orientation)
    initial_task_waypoint_sha = tensor_digest_sha256(task._tm_waypoint)
    initial_route_goal_sha = (
        tensor_digest_sha256(task._target_route_manager.goal)
        if route_mode == "global_astar_v1" else None
    )
    active_centers = task.obs_dict["obstacle_position"][
        :, task._bar_offset : task._bar_offset + task.n_bars_active, :2
    ]
    active_half = task.obs_dict["asset_collision_half_extents"][
        :, task._bar_offset : task._bar_offset + task.n_bars_active, :2
    ]
    require(int(task.n_bars_active) == density, "active bar count differs from density cell")
    require(active_centers.shape[1] == density and active_half.shape[1] == density,
            "active bar AABB tensor count mismatch")
    require(bool(torch.isfinite(active_centers).all()) and bool(torch.isfinite(active_half).all()),
            "active bar AABB tensor is nonfinite")
    require(bool((active_half > 0).all()), "active bar AABB has nonpositive extent")
    reset_batches = 1
    reset_envs = ENVS
    reset_total_s = initial_reset_s
    reset_max_s = initial_reset_s

    ctrl = task._target_controller
    zero_policy_action = torch.zeros((ENVS, 4), device=task.device)
    speed_sum = err_sq_sum = 0.0
    tracking_samples = 0
    safety_samples = contact_samples = off_infeasible_samples = invalid_samples = 0
    invalid_axis_samples = [0, 0, 0]
    position_min = torch.full((3,), float("inf"), device=task.device)
    position_max = torch.full((3,), -float("inf"), device=task.device)
    saturation_substeps = torch.zeros((), dtype=torch.long, device=task.device)
    controller_substeps = torch.zeros((), dtype=torch.long, device=task.device)
    controller_counter_decrease = torch.zeros((), dtype=torch.bool, device=task.device)
    max_tilt_seen_rad = torch.zeros((), dtype=task.target_position.dtype, device=task.device)
    _sync_cuda(torch)
    rollout_started = time.perf_counter()
    cell_clock_start = int(task.num_task_steps)
    canonical_mapping_calls = 0
    for step in range(STEPS):
        interval_start_step, pursuer_command = prepare_neutral_pursuer_interval(
            task, zero_policy_action
        )
        canonical_mapping_calls += 1
        saturation_before = ctrl.saturation_substeps.clone()
        substeps_before = ctrl.substeps.clone()
        task.sim_env.step(actions=pursuer_command)
        saturation_delta = ctrl.saturation_substeps - saturation_before
        substep_delta = ctrl.substeps - substeps_before
        controller_counter_decrease |= (saturation_delta < 0).any() | (substep_delta < 0).any()
        saturation_substeps += saturation_delta.clamp(min=0).sum()
        controller_substeps += substep_delta.clamp(min=0).sum()
        max_tilt_seen_rad = torch.maximum(max_tilt_seen_rad, ctrl.max_tilt_seen_rad.max())
        if step >= WARMUP_STEPS:
            actual = task.target_vel_w[:, :2]
            desired = ctrl.velocity_command[:, :2]
            speed_sum += float(actual.norm(dim=1).sum().item())
            err_sq_sum += float(((actual - desired) ** 2).sum(dim=1).sum().item())
            tracking_samples += ENVS
        safety_samples += ENVS
        contact = ctrl.contact_seen.clone()
        contact_samples += int(contact.sum().item())
        if route_mode == "off":
            off_infeasible_samples += int((~task._tm_last_step_feasible).sum().item())
        bmin = task.obs_dict["env_bounds_min"]
        bmax = task.obs_dict["env_bounds_max"]
        support = task._physical_target_support_xyz()
        invalid = (
            (task.target_position[:, :2] - support[:, :2] < bmin[:, :2])
            | (task.target_position[:, :2] + support[:, :2] > bmax[:, :2])
        ).any(dim=1)
        invalid |= (
            (task.target_position[:, 2] - support[:, 2] < bmin[:, 2])
            | (task.target_position[:, 2] + support[:, 2] > bmax[:, 2])
            | ~torch.isfinite(task.target_position).all(dim=1)
        )
        invalid_only = invalid & ~contact
        invalid_samples += int(invalid_only.sum().item())
        for axis in range(3):
            axis_invalid = (
                (task.target_position[:, axis] - support[:, axis] < bmin[:, axis])
                | (task.target_position[:, axis] + support[:, axis] > bmax[:, axis])
            )
            invalid_axis_samples[axis] += int((axis_invalid & ~contact).sum().item())
        position_min = torch.minimum(position_min, task.target_position.amin(dim=0))
        position_max = torch.maximum(position_max, task.target_position.amax(dim=0))
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            timing = _timed_reset(task, torch, failed_ids)
            reset_batches += 1
            reset_envs += timing["envs"]
            reset_total_s += timing["wall_s"]
            reset_max_s = max(reset_max_s, timing["wall_s"])
        finish_low_level_evaluation_interval(task, interval_start_step)
    _sync_cuda(torch)
    rollout_wall_s = time.perf_counter() - rollout_started
    route_after = normalized_route_diagnostics(task)
    routed_local_invalidations = None
    if route_mode == "global_astar_v1":
        full_route_delta = recursive_counter_delta(route_before, route_after)
        invalidations = full_route_delta.get("invalidation_counts", {})
        require("local_step_infeasible" in invalidations,
                "local-step invalidation counter missing from routed diagnostics")
        routed_local_invalidations = int(invalidations["local_step_infeasible"])
    route = route_cell_delta(route_before, route_after, ENVS * STEPS)
    require(not bool(controller_counter_decrease.item()), "controller counters decreased before reset")
    require(tracking_samples == ENVS * (STEPS - WARMUP_STEPS),
            "tracking sample count drift")
    require(safety_samples == ENVS * STEPS, "safety sample count drift")
    mean_speed = speed_sum / max(1, tracking_samples)
    row = {
        "record_id": record_id(route_mode, speed, density),
        "seed": SEED, "route_mode": route_mode, "speed_mps": speed, "bars": density,
        "envs": ENVS, "steps": STEPS, "warmup_steps": WARMUP_STEPS,
        "route_goal_exclusion_m": float(task.tm.route_goal_exclusion_radius_m),
        "initial_layout_sha256": layout_sha,
        "initial_robot_pose_sha256": initial_robot_pose_sha,
        "initial_target_pose_sha256": initial_target_pose_sha,
        "initial_task_waypoint_sha256": initial_task_waypoint_sha,
        "initial_route_goal_sha256": initial_route_goal_sha,
        "bar_offset": int(task._bar_offset),
        "active_bar_aabb_count": int(active_half.shape[1]),
        "tracking_measurement_env_intervals": tracking_samples,
        "safety_measurement_env_intervals": safety_samples,
        "position_measurement_env_intervals": safety_samples,
        "failed_reset_monitoring_env_intervals": safety_samples,
        "commanded_env_intervals": ENVS * STEPS,
        "task_clock": {
            "start_num_task_steps": cell_clock_start,
            "end_num_task_steps": int(task.num_task_steps),
            "increments": int(task.num_task_steps) - cell_clock_start,
            "increment_order": "after physics and any failed-env reset, once per interval",
        },
        "neutral_pursuer_command_contract": {
            "policy_action": "all_zero_[N,4]",
            "mapping": "NavRLTask.transform_action_to_command",
            "mapping_order": "after_target_advance_before_sim_step",
            "mapping_calls": canonical_mapping_calls,
            "command_shape": [ENVS, 4],
            "all_commands_finite": True,
        },
        "mean_speed_mps": mean_speed,
        "mean_speed_ratio": mean_speed / speed,
        "tracking_rmse_mps": math.sqrt(err_sq_sum / max(1, tracking_samples)),
        "contact_step_fraction": contact_samples / max(1, safety_samples),
        "off_bounded_local_step_infeasible_fraction": (
            off_infeasible_samples / max(1, safety_samples) if route_mode == "off" else None
        ),
        "routed_local_step_invalidation_fraction": (
            routed_local_invalidations / max(1, safety_samples)
            if route_mode == "global_astar_v1" else None
        ),
        "invalid_state_fraction": invalid_samples / max(1, safety_samples),
        "invalid_axis_fraction_xyz": [v / max(1, safety_samples) for v in invalid_axis_samples],
        "position_min_xyz": [float(v) for v in position_min.tolist()],
        "position_max_xyz": [float(v) for v in position_max.tolist()],
        "motor_saturation_fraction": float(
            saturation_substeps.float().div(controller_substeps.clamp(min=1)).item()
        ),
        "max_tilt_deg": math.degrees(float(max_tilt_seen_rad.item())),
        "controller_counter_contract": {
            "saturated_physics_substeps": int(saturation_substeps.item()),
            "physics_substeps": int(controller_substeps.item()),
            "method": "sum nonnegative before/after deltas prior to any failed-env reset",
        },
        "route": route,
        "reset_wall": {
            "batches": reset_batches, "reset_envs": reset_envs,
            "initial_batch_s": initial_reset_s, "total_s": reset_total_s,
            "max_batch_s": reset_max_s,
            "ms_per_reset_env": 1000.0 * reset_total_s / max(1, reset_envs),
        },
        "throughput": {
            "rollout_wall_s": rollout_wall_s,
            "rl_steps_per_s": STEPS / max(rollout_wall_s, 1e-9),
            "env_intervals_per_s": ENVS * STEPS / max(rollout_wall_s, 1e-9),
        },
        "import_origin": dict(import_origin),
    }
    row["record_contract_sha256"] = record_contract_sha256(row)
    row["gates"] = physical_gate_metrics(row)
    row["pass"] = all(row["gates"].values())
    return row


def child_main(args) -> int:
    require(args.route_mode in ROUTE_ARMS and args.speed in SPEEDS, "child arm outside frozen grid")
    manifest_path = Path(args.source_manifest).resolve()
    source = verify_source_manifest(manifest_path, args.source_manifest_sha256)
    environment_contract = configure_child(args.route_mode, args.speed)
    os.chdir(ROOT)
    sys.path[:] = [str(ROOT)] + [value for value in sys.path if Path(value or ".").resolve() != ROOT]
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry  # Isaac Gym before torch
    import aerial_gym
    import torch

    origin = Path(aerial_gym.__file__).resolve()
    expected_origin = (ROOT / "aerial_gym/__init__.py").resolve()
    require(origin == expected_origin, f"aerial_gym import origin drift: {origin}")
    origin_entry = next(
        (entry for entry in source["runtime_files"] if entry["path"] == "aerial_gym/__init__.py"),
        None,
    )
    require(origin_entry is not None, "aerial_gym import origin absent from source manifest")
    require(sha256_file(origin) == origin_entry["sha256"], "imported aerial_gym bytes mismatch")
    import_origin = {
        "enforced": True, "path": str(origin), "sha256": sha256_file(origin),
        "manifest_sha256": origin_entry["sha256"],
    }
    task = task_registry.make_task(
        "navrl_task", seed=SEED, num_envs=ENVS, headless=True, use_warp=True
    )
    require(task._target_dynamics == "physical", "task did not instantiate physical target")
    require(task._target_route_mode == args.route_mode, "task route mode differs from child arm")
    require(str(task.tm.pattern) == "waypoint", "task pattern is not waypoint")
    require(abs(float(task.tm.speed_fixed) - args.speed) < 1e-9, "task fixed speed drift")
    require(abs(float(task.tm.route_goal_exclusion_radius_m) - 1.0) < 1e-9,
            "task route goal exclusion drift")
    instantiated_contract = attest_instantiated_contract(task, args.route_mode, source)
    software_provenance = runtime_software_provenance(torch, source)
    rows = [run_cell(task, torch, args.route_mode, args.speed, density, import_origin)
            for density in DENSITIES]
    verify_source_manifest(manifest_path, args.source_manifest_sha256)
    payload = {
        "schema": CHILD_SCHEMA, "seed": SEED, "route_mode": args.route_mode,
        "speed_mps": args.speed, "densities": list(DENSITIES), "envs": ENVS,
        "steps": STEPS, "warmup_steps": WARMUP_STEPS,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": args.source_manifest_sha256,
        "import_origin": import_origin, "environment_contract": environment_contract,
        "instantiated_contract": instantiated_contract,
        "software_provenance": software_provenance,
        "cells": rows,
    }
    atomic_json(Path(args.child_output), payload)
    return 0


def validate_child(
    path: Path, route_mode: str, speed: float, manifest_sha: str,
    expected_source_hashes: Optional[Mapping[str, str]] = None,
) -> Dict:
    require(path.is_file(), f"child summary missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == CHILD_SCHEMA, f"wrong child schema: {path}")
    require(payload.get("route_mode") == route_mode and payload.get("speed_mps") == speed,
            f"child arm mismatch: {path}")
    require(payload.get("source_manifest_sha256") == manifest_sha, f"child source drift: {path}")
    require(payload.get("environment_contract") == frozen_environment(route_mode, speed),
            f"child environment contract drift: {path}")
    contract = payload.get("instantiated_contract")
    require(isinstance(contract, Mapping), f"child instantiated contract missing: {path}")
    require(contract.get("bar_offset") == 1, f"child bar offset drift: {path}")
    physics = contract.get("physics", {})
    require(
        physics.get("physics_dt_s") == 0.01
        and physics.get("physics_steps_per_rl_step") == 10
        and physics.get("rl_step_dt_s") == 0.1,
        f"child runtime timing drift: {path}",
    )
    require(contract.get("physical_target_box_xyz_m") == [0.28, 0.28, 0.12],
            f"child target box drift: {path}")
    support = contract.get("declared_conservative_support_xy_m", [])
    require(len(support) == 2 and all(abs(value - 0.2068816086567407) < 1e-12 for value in support),
            f"child conservative support drift: {path}")
    require(contract.get("robot", {}).get("robot_name") == "navrl_ref5in_quad",
            f"child robot identity drift: {path}")
    sim_contract = contract.get("sim", {})
    require(
        sim_contract.get("name") == "base_sim"
        and sim_contract.get("config_class") == "BaseSimConfig",
        f"child simulator identity drift: {path}",
    )
    if expected_source_hashes is not None:
        robot = contract["robot"]
        require(
            robot.get("robot_config_sha256")
            == expected_source_hashes.get(str(robot.get("robot_config_path", "")))
            and robot.get("robot_asset_sha256")
            == expected_source_hashes.get(str(robot.get("robot_asset_path", ""))),
            f"child instantiated robot bytes differ from source manifest: {path}",
        )
        require(
            sim_contract.get("config_sha256")
            == expected_source_hashes.get(str(sim_contract.get("config_path", ""))),
            f"child instantiated sim config differs from source manifest: {path}",
        )
    provenance = payload.get("software_provenance")
    require(isinstance(provenance, Mapping), f"child software provenance missing: {path}")
    python = provenance.get("python", {})
    torch_info = provenance.get("torch", {})
    isaac = provenance.get("isaac_gym", {})
    ninja = provenance.get("ninja", {})
    cuda = provenance.get("cuda", {})
    require(bool(python.get("version")) and bool(python.get("executable")),
            f"child Python identity missing: {path}")
    require(bool(torch_info.get("version")) and bool(torch_info.get("compiled_cuda_version")),
            f"child torch/CUDA identity missing: {path}")
    require(
        cuda.get("available") is True and cuda.get("device_count", 0) >= 1
        and cuda.get("gpu_names") and cuda.get("driver_versions"),
        f"child CUDA/GPU/driver identity missing: {path}",
    )
    for label, entry, path_key, hash_key in (
        ("Python", python, "executable", "executable_sha256"),
        ("torch", torch_info, "origin", "origin_sha256"),
        ("Isaac Gym", isaac, "origin", "origin_sha256"),
        ("ninja", ninja, "path", "sha256"),
    ):
        external = Path(str(entry.get(path_key, ""))).resolve()
        require(external.is_file() and sha256_file(external) == entry.get(hash_key),
                f"child {label} origin bytes drift: {path}")
        if label == "ninja":
            require(os.access(str(external), os.X_OK), f"child ninja is not executable: {path}")
    ninja_version = subprocess.run(
        [str(ninja.get("path", "")), "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    require(
        ninja_version.returncode == 0
        and ninja_version.stdout.strip() == ninja.get("version"),
        f"child ninja version drift: {path}",
    )
    repo_modules = provenance.get("repo_modules", {})
    if expected_source_hashes is not None:
        for name, relative in (
            ("navrl_task", "aerial_gym/task/navrl_task/navrl_task.py"),
            ("target_route_planner", "aerial_gym/task/navrl_task/target_route_planner.py"),
        ):
            module = repo_modules.get(name, {})
            require(
                module.get("relative_path") == relative
                and module.get("sha256") == expected_source_hashes.get(relative)
                and module.get("manifest_sha256") == expected_source_hashes.get(relative),
                f"child {name} module origin/hash drift: {path}",
            )
    origin = payload.get("import_origin")
    require(isinstance(origin, Mapping) and origin.get("enforced") is True,
            f"child import origin missing: {path}")
    if expected_source_hashes is not None:
        expected_origin_sha = expected_source_hashes["aerial_gym/__init__.py"]
        require(
            origin.get("sha256") == expected_origin_sha
            and origin.get("manifest_sha256") == expected_origin_sha,
            f"child import origin does not match source manifest: {path}",
        )
    cells = payload.get("cells")
    require(isinstance(cells, list) and len(cells) == 4, f"child must contain four cells: {path}")
    require([row.get("bars") for row in cells] == list(DENSITIES), f"child density order drift: {path}")
    for row, density in zip(cells, DENSITIES):
        require(
            row.get("route_mode") == route_mode
            and row.get("speed_mps") == speed
            and row.get("bars") == density,
            f"child header/cell arm mismatch at {density} bars: {path}",
        )
        require(
            row.get("record_id") == record_id(route_mode, speed, density),
            f"child cell record_id mismatch at {density} bars: {path}",
        )
    return payload


def parent_main(args) -> int:
    output = Path(args.output).resolve()
    require(output == DEFAULT_OUTPUT.resolve() or output.parent != ROOT,
            "custom output must name a result directory, not repository root")
    require(not output.exists(), f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path = output.parent / "source_manifest.json"
    source_manifest = build_source_manifest()
    atomic_json(source_manifest_path, source_manifest)
    source_sha = sha256_file(source_manifest_path)
    children_dir = output.parent / "children"
    logs_dir = output.parent / "logs"
    children_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)
    children = []
    records = []
    instantiated_contracts = []
    software_provenance = []
    errors = []
    # Do not let an interactive shell alter NavRL or hide the selected conda build tools.
    base_env = build_child_environment()
    for route_mode in ROUTE_ARMS:
        for speed in SPEEDS:
            label = f"route_{route_mode}__speed_{speed:.1f}".replace(".", "p")
            child_path = children_dir / f"{label}.json"
            log_path = logs_dir / f"{label}.log"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--_child",
                "--route-mode", route_mode, "--speed", str(speed),
                "--source-manifest", str(source_manifest_path),
                "--source-manifest-sha256", source_sha,
                "--child-output", str(child_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=ROOT, env=base_env, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
            entry = {
                "route_mode": route_mode, "speed_mps": speed,
                "command": command, "returncode": completed.returncode,
                "summary": str(child_path.relative_to(output.parent)),
                "log": str(log_path.relative_to(output.parent)),
            }
            if completed.returncode != 0:
                errors.append(f"{label}: child exit {completed.returncode}")
                children.append(entry)
                break
            try:
                payload = validate_child(
                    child_path, route_mode, speed, source_sha,
                    source_hash_map(source_manifest),
                )
                entry["summary_sha256"] = sha256_file(child_path)
                records.extend(payload["cells"])
                instantiated_contracts.append({
                    "route_mode": route_mode, "speed_mps": speed,
                    "contract": payload["instantiated_contract"],
                })
                software_provenance.append({
                    "route_mode": route_mode, "speed_mps": speed,
                    "provenance": payload["software_provenance"],
                })
            except Exception as exc:  # retained in VOID manifest; never interpreted as a result
                errors.append(f"{label}: {exc}")
            children.append(entry)
        if errors:
            break
    try:
        verify_source_manifest(source_manifest_path, source_sha)
        if not errors:
            validate_grid_records(records)
    except Exception as exc:
        errors.append(str(exc))
    integrity_ok = not errors
    execution_manifest = {
        "schema": EXECUTION_SCHEMA, "source_manifest": "source_manifest.json",
        "source_manifest_sha256": source_sha, "children": children,
        "record_ids": [row.get("record_id") for row in records],
        "record_count": len(records), "integrity_ok": integrity_ok, "errors": errors,
    }
    execution_manifest_path = output.parent / "execution_manifest.json"
    atomic_json(execution_manifest_path, execution_manifest)
    verdicts = derive_verdicts(records, integrity_ok)
    summary = {
        "schema": SCHEMA, "preregistration": str(PREREG.relative_to(ROOT)),
        "seed": SEED, "route_arms": list(ROUTE_ARMS), "speeds_mps": list(SPEEDS),
        "densities": list(DENSITIES), "envs": ENVS, "steps": STEPS,
        "warmup_steps": WARMUP_STEPS, "gates_preregistered": GATES,
        "source_manifest": "source_manifest.json", "source_manifest_sha256": source_sha,
        "execution_manifest": "execution_manifest.json",
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "instantiated_contracts": instantiated_contracts if integrity_ok else [],
        "software_provenance": software_provenance if integrity_ok else [],
        "cells": records if integrity_ok else [],
        "matched_route_on_minus_off": matched_deltas(records) if integrity_ok else [],
        "verdicts": verdicts,
        "claim_boundary": {
            "ppo_policy_loaded": False, "hardware_validation": False,
            "arena_wide_connectivity_300_claim": False,
            "long_training_authorized": False,
            "matched_initial_bar_robot_target_pose": True,
            "route_goal_intentionally_differs_between_arms": True,
        },
    }
    atomic_json(output, summary)
    receipt = {
        "schema": "navrl_physical_target_routed_gate_receipt_v2",
        "summary": output.name, "summary_sha256": sha256_file(output),
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "source_manifest_sha256": source_sha,
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "bound_runtime_sha256": {
            key: source_hash_map(source_manifest)[key]
            for key in (
                "aerial_gym/task/navrl_task/target_route_planner.py",
                "aerial_gym/task/navrl_task/navrl_task.py",
                "aerial_gym/config/task_config/navrl_task_config.py",
                "aerial_gym/config/sim_config/base_sim_config.py",
                "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
                "resources/robots/quad/quad_navrl_ref5in.urdf",
            )
        },
        "record_count": len(records),
        "record_ids": [row["record_id"] for row in records] if integrity_ok else [],
        "verdicts": verdicts,
    }
    atomic_json(output.parent / "receipt.json", receipt)
    print(f"saved {output} integrity={integrity_ok} verdict={verdicts['physical_training']}")
    return 0 if integrity_ok else 3


def verify_result(summary_path: Path, required_contract: str) -> int:
    summary_path = summary_path.resolve()
    require(summary_path.is_file(), f"summary missing: {summary_path}")
    directory = summary_path.parent
    receipt_path = directory / "receipt.json"
    require(receipt_path.is_file(), "receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "navrl_physical_target_routed_gate_receipt_v2",
            "wrong receipt schema")
    require(receipt.get("summary") == summary_path.name, "receipt summary path mismatch")
    require(receipt.get("summary_sha256") == sha256_file(summary_path), "summary bytes drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == SCHEMA, "wrong summary schema")
    source_path = directory / str(summary.get("source_manifest", ""))
    source_sha = str(summary.get("source_manifest_sha256", ""))
    require(receipt.get("source_manifest_sha256") == source_sha, "receipt source digest mismatch")
    source = verify_source_manifest(source_path, source_sha)
    hashes = source_hash_map(source)
    evaluator_relative = str(Path(__file__).resolve().relative_to(ROOT))
    require(receipt.get("evaluator_source_sha256") == hashes.get(evaluator_relative),
            "receipt evaluator digest mismatch")
    required_bindings = {
        key: hashes.get(key)
        for key in (
            "aerial_gym/task/navrl_task/target_route_planner.py",
            "aerial_gym/task/navrl_task/navrl_task.py",
            "aerial_gym/config/task_config/navrl_task_config.py",
            "aerial_gym/config/sim_config/base_sim_config.py",
            "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
            "resources/robots/quad/quad_navrl_ref5in.urdf",
        )
    }
    require(receipt.get("bound_runtime_sha256") == required_bindings,
            "receipt runtime binding mismatch")
    execution_path = directory / str(summary.get("execution_manifest", ""))
    require(execution_path.is_file(), "execution manifest missing")
    execution_sha = sha256_file(execution_path)
    require(execution_sha == summary.get("execution_manifest_sha256"),
            "summary execution-manifest digest mismatch")
    require(execution_sha == receipt.get("execution_manifest_sha256"),
            "receipt execution-manifest digest mismatch")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    require(execution.get("schema") == EXECUTION_SCHEMA and execution.get("integrity_ok") is True,
            "execution integrity is not true")
    children = execution.get("children")
    require(isinstance(children, list) and len(children) == 8, "exactly eight child summaries required")
    child_records = []
    child_contracts = []
    child_provenance = []
    for child in children:
        child_path = directory / str(child.get("summary", ""))
        require(child_path.is_file(), f"child summary missing during verify: {child_path}")
        require(child.get("summary_sha256") == sha256_file(child_path),
                f"child summary bytes drift: {child_path}")
        payload = validate_child(
            child_path, str(child["route_mode"]), float(child["speed_mps"]), source_sha,
            hashes,
        )
        child_records.extend(payload["cells"])
        child_contracts.append({
            "route_mode": child["route_mode"], "speed_mps": child["speed_mps"],
            "contract": payload["instantiated_contract"],
        })
        child_provenance.append({
            "route_mode": child["route_mode"], "speed_mps": child["speed_mps"],
            "provenance": payload["software_provenance"],
        })
    validate_grid_records(child_records)
    require(child_records == summary.get("cells"), "summary cells differ from child summaries")
    require(child_contracts == summary.get("instantiated_contracts"),
            "summary instantiated contracts differ from child summaries")
    require(child_provenance == summary.get("software_provenance"),
            "summary software provenance differs from child summaries")
    require(matched_deltas(child_records) == summary.get("matched_route_on_minus_off"),
            "summary matched route deltas differ from child recomputation")
    require(receipt.get("record_count") == 32 and len(receipt.get("record_ids", [])) == 32,
            "receipt does not bind exact 32-cell accounting")
    require(receipt.get("record_ids") == [row["record_id"] for row in child_records],
            "receipt record IDs mismatch")
    recomputed = derive_verdicts(child_records, True)
    require(recomputed == summary.get("verdicts"), "summary verdict recomputation mismatch")
    require(recomputed == receipt.get("verdicts"), "receipt verdict mismatch")
    require(recomputed.get("route_mechanism") == "PASS_ROUTE_MECHANISM",
            "route mechanism did not pass")
    contract_field, expected = {
        "full_1p5": ("full_1p5_contract", "PASS_FULL_1P5_CONTRACT"),
        "density_conditioned": (
            "density_conditioned_envelope", "PASS_DENSITY_CONDITIONED_ENVELOPE"
        ),
    }[required_contract]
    require(recomputed.get(contract_field) == expected,
            f"required training contract absent: {expected}")
    print(f"VERIFIED {summary_path} PASS_ROUTE_MECHANISM {expected}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verify")
    parser.add_argument(
        "--require-contract", choices=("full_1p5", "density_conditioned"),
        default="density_conditioned",
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--route-mode", choices=ROUTE_ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--speed", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--source-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--source-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.verify:
            require(not args._child, "--verify cannot be combined with child mode")
            return verify_result(Path(args.verify), args.require_contract)
        if args._child:
            require(args.route_mode is not None and args.speed is not None, "incomplete child arm")
            require(args.source_manifest and args.source_manifest_sha256 and args.child_output,
                    "child provenance arguments missing")
            return child_main(args)
        return parent_main(args)
    except IntegrityError as exc:
        print(f"VOID_EXECUTION: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
