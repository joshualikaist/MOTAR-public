#!/usr/bin/env python3
"""Fresh-process 32-cell gate for ``global_astar_recovery_v2``.

This is intentionally a new evaluator.  The immutable v1 routed evaluator is imported only for
small, already-tested provenance/physics helpers and is never edited or used as a v2 alias.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "tools/verify_navrl_physical_target_routed_simulator_gate.py"
BASE_SPEC = importlib.util.spec_from_file_location("navrl_routed_gate_v1_immutable", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)

# The packed observer imports torch.  Isaac Gym requires its extension loader to run first in a
# simulator child; the parent/verify paths stay CPU-safe and do not import Isaac Gym.  Detecting
# the private child token here preserves that mandatory import order before loading the observer.
if "--_child" in sys.argv:
    import isaacgym  # noqa: F401  # must precede torch; imported for side effects

PACKED_PATH = ROOT / "tools/navrl_recovery_packed_telemetry.py"
PACKED_SPEC = importlib.util.spec_from_file_location("navrl_recovery_packed_runtime", PACKED_PATH)
PACKED = importlib.util.module_from_spec(PACKED_SPEC)
sys.modules[PACKED_SPEC.name] = PACKED
PACKED_SPEC.loader.exec_module(PACKED)

BRAKE_VERIFY_PATH = ROOT / "tools/verify_navrl_physical_target_braking.py"
BRAKE_PROBE_PATH = ROOT / "tools/probe_navrl_physical_target_braking.py"
BRAKE_PREREG_PATH = ROOT / "docs/preregistration_navrl_physical_target_braking_2026-08-25.md"
SOURCE_BUNDLE_PATH = ROOT / "tools/create_navrl_source_bundle.py"
BRAKE_PROBE_SPEC = importlib.util.spec_from_file_location(
    "probe_navrl_physical_target_braking", BRAKE_PROBE_PATH
)
BRAKE_PROBE = importlib.util.module_from_spec(BRAKE_PROBE_SPEC)
sys.modules[BRAKE_PROBE_SPEC.name] = BRAKE_PROBE
BRAKE_PROBE_SPEC.loader.exec_module(BRAKE_PROBE)
BRAKE_SPEC = importlib.util.spec_from_file_location(
    "navrl_physical_target_braking_verifier", BRAKE_VERIFY_PATH
)
BRAKE_VERIFY = importlib.util.module_from_spec(BRAKE_SPEC)
sys.modules[BRAKE_SPEC.name] = BRAKE_VERIFY
BRAKE_SPEC.loader.exec_module(BRAKE_VERIFY)
SOURCE_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "navrl_recovery_training_source_bundle", SOURCE_BUNDLE_PATH
)
SOURCE_BUNDLE = importlib.util.module_from_spec(SOURCE_BUNDLE_SPEC)
sys.modules[SOURCE_BUNDLE_SPEC.name] = SOURCE_BUNDLE
SOURCE_BUNDLE_SPEC.loader.exec_module(SOURCE_BUNDLE)

PREREG = ROOT / (
    "docs/preregistration_physical_target_recovery_v2_lower1p25_gate_2026-08-26.md"
    if BRAKE_PROBE.CONTRACT_VARIANT == "baseline_1p25"
    else "docs/preregistration_physical_target_recovery_v2_gate_2026-08-25.md"
)
RECOVERY_PREREG = ROOT / "docs/preregistration_physical_target_two_envelope_recovery_2026-08-25.md"
SCHEMA = "navrl_physical_target_recovery_v2_gate_v1"
CHILD_SCHEMA = "navrl_physical_target_recovery_v2_child_v1"
SOURCE_SCHEMA = "navrl_physical_target_recovery_v2_source_manifest_v1"
EXECUTION_SCHEMA = "navrl_physical_target_recovery_v2_execution_manifest_v1"
RECEIPT_SCHEMA = "navrl_physical_target_recovery_v2_receipt_v1"

SEED = 827
ROUTE_ARMS = ("off", "global_astar_recovery_v2")
CONTRACT_VARIANT = BRAKE_PROBE.CONTRACT_VARIANT
SPEEDS = tuple(BRAKE_PROBE.REGISTERED_SPEEDS)
DENSITIES = (70, 150, 205, 300)
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20
PHYSICS_SUBSTEPS = 10
DEFAULT_DIR = ROOT / (
    "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827"
    if CONTRACT_VARIANT == "baseline_1p25"
    else "results/navrl_physical_target_recovery_v2_gate_seed827"
)

GATES = {
    "tracking_rmse_mps_max": 0.35,
    "mean_speed_ratio_min": 0.80,
    "contact_step_fraction_max": 0.01,
    "off_rounded_local_infeasible_fraction_max": 0.01,
    "recovery_normal_route_rounded_invalidation_fraction_max": 0.01,
    "motor_saturation_fraction_max": 0.15,
    "max_tilt_deg_max": 60.0,
    "invalid_state_fraction_max": 0.0,
    "watchdog_breach_substeps_max": 0,
    "direct_position_writes_max": 0,
    "reset_calls_during_advance_max": 0,
    "connect_actual_regressions_max": 0,
    # Operational gate only: more than 2x matched wall-time overhead makes the fixed 32-cell gate
    # impractical and is reported as an implementation regression, not a navigation claim.
    "matched_recovery_vs_off_throughput_ratio_min": 0.50,
}


class IntegrityError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def atomic_json(path: Path, payload: Mapping) -> None:
    require(not path.exists(), "refusing to overwrite %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    require(not temporary.exists(), "stale temporary exists: %s" % temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    require(completed.returncode == 0, "git %s failed: %s" % (" ".join(args), completed.stderr))
    return completed.stdout.strip()


def runtime_source_paths() -> List[Path]:
    requested = [
        "aerial_gym", "resources/robots/quad/quad_navrl_ref5in.urdf",
        "resources/models/environment_assets/bars", str(PREREG.relative_to(ROOT)),
        str(RECOVERY_PREREG.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
        str(PACKED_PATH.relative_to(ROOT)), str(BASE_PATH.relative_to(ROOT)),
        str(BRAKE_VERIFY_PATH.relative_to(ROOT)),
        str(BRAKE_PROBE_PATH.relative_to(ROOT)), str(BRAKE_PREREG_PATH.relative_to(ROOT)),
        str(SOURCE_BUNDLE_PATH.relative_to(ROOT)),
    ]
    if CONTRACT_VARIANT == "baseline_1p25":
        requested.append("tools/run_navrl_physical_target_recovery_v2_lower1p25_gate.sh")
    tracked = git("ls-files", *requested).splitlines()
    paths = sorted({ROOT / name for name in tracked if name})
    required = {
        PREREG, RECOVERY_PREREG, Path(__file__).resolve(), PACKED_PATH, BASE_PATH,
        BRAKE_VERIFY_PATH, BRAKE_PROBE_PATH, BRAKE_PREREG_PATH, SOURCE_BUNDLE_PATH,
    }
    require(required.issubset(set(paths)), "evaluator/prereg source is not fully tracked")
    require(all(path.is_file() for path in paths), "source manifest contains missing path")
    return paths


def build_source_manifest() -> Dict[str, object]:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    require(not dirty, "tracked repository is dirty; commit before GPU evaluation: %s" % dirty)
    entries = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
         "size": path.stat().st_size}
        for path in runtime_source_paths()
    ]
    return {
        "schema": SOURCE_SCHEMA, "repository_root": str(ROOT),
        "git_commit": git("rev-parse", "HEAD"),
        "preregistration": str(PREREG.relative_to(ROOT)),
        "recovery_preregistration": str(RECOVERY_PREREG.relative_to(ROOT)),
        "runtime_file_count": len(entries), "runtime_files": entries,
    }


def verify_source_manifest(path: Path, expected_sha: str) -> Dict[str, object]:
    require(path.is_file() and sha256_file(path) == expected_sha, "source manifest SHA drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == SOURCE_SCHEMA, "wrong source schema")
    require(Path(payload.get("repository_root", "")).resolve() == ROOT, "source root drift")
    entries = payload.get("runtime_files")
    require(isinstance(entries, list) and len(entries) == payload.get("runtime_file_count"),
            "source file count drift")
    seen = set()
    for entry in entries:
        relative = Path(str(entry.get("path", "")))
        require(relative and not relative.is_absolute() and ".." not in relative.parts,
                "unsafe source path")
        require(str(relative) not in seen, "duplicate source path")
        seen.add(str(relative))
        source = ROOT / relative
        require(source.is_file() and source.stat().st_size == int(entry.get("size", -1)),
                "source disappeared/size drift: %s" % relative)
        require(sha256_file(source) == entry.get("sha256"), "source hash drift: %s" % relative)
    for required in (
        PREREG, RECOVERY_PREREG, Path(__file__).resolve(), PACKED_PATH, BASE_PATH,
        BRAKE_VERIFY_PATH, BRAKE_PROBE_PATH, BRAKE_PREREG_PATH, SOURCE_BUNDLE_PATH,
    ):
        require(str(required.relative_to(ROOT)) in seen, "required source not bound: %s" % required)
    return payload


def source_hashes(manifest: Mapping) -> Dict[str, str]:
    return {str(entry["path"]): str(entry["sha256"]) for entry in manifest["runtime_files"]}


def nvidia_smi_provenance() -> Dict[str, str]:
    executable = shutil.which("nvidia-smi")
    require(executable, "nvidia-smi executable is not on PATH")
    path = Path(executable).resolve()
    completed = subprocess.run(
        [str(path), "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    require(completed.returncode == 0 and completed.stdout.strip(),
            "nvidia-smi identity query failed: %s" % completed.stderr.strip())
    return {"path": str(path), "sha256": sha256_file(path),
            "identity": completed.stdout.strip()}


def validate_braking_probe_values(values: Mapping[str, str]) -> Dict[str, str]:
    names = (
        "NAVRL_TARGET_RECOVERY_BRAKE_P05",
        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S",
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT",
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256",
    )
    values = {name: str(values.get(name, "")).strip() for name in names}
    require(all(values.values()), "target-specific braking receipt contract is incomplete")
    decel = float(values[names[0]])
    stop = float(values[names[1]])
    require(math.isfinite(decel) and decel > 0 and math.isfinite(stop) and stop > 0,
            "braking quantiles must be finite and positive")
    receipt = Path(values[names[2]]).expanduser().resolve()
    digest = values[names[3]].lower()
    require(receipt.is_file() and len(digest) == 64 and sha256_file(receipt) == digest,
            "braking receipt file/SHA mismatch")
    try:
        verified = BRAKE_VERIFY.verify_receipt(receipt.parent, ROOT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IntegrityError("raw braking receipt verification failed: %s" % exc)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    require(payload.get("schema") == BRAKE_PROBE.RECEIPT_SCHEMA
            and payload.get("probe_schema") == BRAKE_PROBE.SCHEMA,
            "wrong braking receipt/probe schema pair")
    require(abs(float(payload.get("decel_p05_mps2")) - decel) <= 1e-9,
            "braking p05 differs from receipt")
    require(abs(float(payload.get("stop_time_p95_s")) - stop) <= 1e-9,
            "stop-time p95 differs from receipt")
    summary = verified["summary"]
    lookup = summary.get("measured_speed_to_p95_lookup")
    certified = summary.get("certified_monotone_speed_to_p95_lookup")
    require(isinstance(lookup, Mapping) and sorted(float(key) for key in lookup) == list(SPEEDS),
            "receipt does not contain the exact registered speed lookup")
    require(isinstance(certified, Mapping) and sorted(float(key) for key in certified) == list(SPEEDS),
            "receipt does not contain the exact certified speed lookup")
    for speed_key, cell in lookup.items():
        measured = float(cell.get("p95_stop_distance_m", float("nan")))
        require(math.isfinite(measured) and measured >= 0.0,
                "speed cell p95 stop distance missing")
    ordered_distances = []
    for speed in SPEEDS:
        key = BRAKE_VERIFY.speed_key(speed)
        measured = float(lookup[key]["p95_stop_distance_m"])
        certified_distance = float(certified[key]["p95_stop_distance_m"])
        require(math.isfinite(certified_distance) and certified_distance + 1e-12 >= measured,
                "certified stop distance is nonfinite or below the measured p95")
        ordered_distances.append(certified_distance)
    require(all(right + 1e-12 >= left for left, right in zip(
        ordered_distances, ordered_distances[1:]
    )), "certified speed/stop-distance lookup is not monotone")
    core = payload.get("core_integration")
    lateral = core.get("certified_lateral_tube_p95_m") if isinstance(core, Mapping) else None
    require(core == BRAKE_VERIFY.core_integration_object(summary),
            "receipt core integration is not the standalone verifier recomputation")
    require(isinstance(lateral, (int, float)) and math.isfinite(float(lateral))
            and float(lateral) >= 0.0, "certified braking lateral tube is missing")
    values["__MEASURED_LOOKUP_JSON"] = json.dumps(lookup, sort_keys=True, separators=(",", ":"))
    values["__CERTIFIED_LOOKUP_JSON"] = json.dumps(
        certified, sort_keys=True, separators=(",", ":")
    )
    values["__LATERAL_TUBE_P95_M"] = repr(float(lateral))
    values[names[2]] = str(receipt)
    return values


def braking_probe_contract() -> Dict[str, str]:
    names = (
        "NAVRL_TARGET_RECOVERY_BRAKE_P05",
        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S",
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT",
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256",
    )
    return validate_braking_probe_values({name: os.environ.get(name, "") for name in names})


def build_child_environment() -> Dict[str, str]:
    """Build the hermetic child environment while preserving the frozen speed lineage.

    The v1 helper intentionally strips every ``NAVRL_*`` variable.  Recovery-v2 imports its
    speed grid at module load, before the per-child environment is installed, so the exact
    contract variant must be reintroduced for the fresh interpreter itself.
    """
    values = BASE.build_child_environment()
    values["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"] = CONTRACT_VARIANT
    return values


def snapshot_braking_probe(probe: Mapping[str, str], stage: Path) -> Dict[str, str]:
    source = Path(probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT"]).resolve()
    target_dir = stage / "inputs/braking_probe"
    target_dir.parent.mkdir(parents=True, exist_ok=False)
    require(all(not path.is_symlink() for path in source.parent.rglob("*")),
            "braking receipt bundle contains a symlink")
    shutil.copytree(str(source.parent), str(target_dir), copy_function=shutil.copy2)
    target = target_dir / "receipt.json"
    copied = dict(probe)
    copied["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT"] = str(target)
    return validate_braking_probe_values(copied)


def frozen_environment(route_mode: str, speed: float, probe: Mapping[str, str]) -> Dict[str, str]:
    values = BASE.frozen_environment("off", speed)
    values.update({
        "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT": CONTRACT_VARIANT,
        "NAVRL_TARGET_ROUTE_MODE": route_mode,
        "NAVRL_NUM_BARS": "70", "NAVRL_MAX_BARS": "300",
        "NAVRL_TARGET_RECOVERY_EVAL_TELEMETRY": "1",
    })
    for name in (
        "NAVRL_TRAINING_SOURCE_MANIFEST", "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256",
        "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT", "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE",
    ):
        require(probe.get(name), "training source contract is incomplete: %s" % name)
        values[name] = str(probe[name])
    if route_mode == "global_astar_recovery_v2":
        values.update({key: value for key, value in probe.items() if not key.startswith("__")})
        measured_lookup = json.loads(probe["__MEASURED_LOOKUP_JSON"])
        certified_lookup = json.loads(probe["__CERTIFIED_LOOKUP_JSON"])
        values["NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS"] = ",".join(
            BRAKE_VERIFY.speed_key(value) for value in SPEEDS
        )
        values["NAVRL_TARGET_RECOVERY_BRAKE_STOP_DISTANCES_M"] = ",".join(
            str(certified_lookup[BRAKE_VERIFY.speed_key(value)]["p95_stop_distance_m"])
            for value in SPEEDS
        )
        values["NAVRL_TARGET_RECOVERY_STOP_DISTANCE_P95_M"] = str(
            measured_lookup[BRAKE_VERIFY.speed_key(speed)]["p95_stop_distance_m"]
        )
        values["NAVRL_TARGET_RECOVERY_BRAKE_LATERAL_TUBE_P95_M"] = (
            probe["__LATERAL_TUBE_P95_M"]
        )
        # This evaluator is the common fail-closed receipt validator for this run.  The flag is
        # emitted only after the hash, subject, per-speed cells, safety gates and provenance above
        # have all passed; a caller cannot provide it directly.
        values["NAVRL_TARGET_RECOVERY_PROBE_VALIDATED"] = "1"
    return values


RECOVERY_COUNTERS = (
    "recovery_entries", "recovery_brake_intervals", "recovery_connect_intervals",
    "recovery_no_connector_count", "recovery_hard_breach_count", "recovery_route_resumes",
    "recovery_brake_timeout_count", "recovery_connect_timeout_count",
)


def normalized_route_diagnostics(task) -> Dict[str, object]:
    manager = task._target_route_manager
    if manager is None:
        return {"mode": "off"}
    diagnostics = manager.diagnostics()
    require(diagnostics.get("mode") == "global_astar_recovery_v2", "manager mode drift")
    names = BASE.ROUTE_COUNTER_KEYS + (
        "local_step_invalidations", "connected_goal_replans", "same_goal_reselection_count",
    ) + RECOVERY_COUNTERS
    result = {"mode": diagnostics["mode"]}
    for name in names:
        value = diagnostics.get(name)
        require(BASE._is_number(value), "route counter missing/nonfinite: %s" % name)
        result[name] = value
    result["invalidation_counts"] = dict(diagnostics["invalidation_counts"])
    return result


def route_delta(before: Mapping, after: Mapping, commanded: int) -> Dict[str, object]:
    if after.get("mode") == "off":
        return {"mode": "off", "counter_delta": {}, "plan_success_fraction": None,
                "fallback_interval_fraction": None, "goal_completions_per_env": None,
                "planning_wall_s": 0.0, "initial_reset_included": True}
    delta = BASE.recursive_counter_delta(before, after)
    attempts = float(delta["plan_attempts"])
    return {
        "mode": after["mode"], "counter_delta": delta,
        "plan_success_fraction": float(delta["plan_successes"]) / attempts if attempts else 0.0,
        "fallback_interval_fraction": float(delta["fallback_intervals"]) / max(1, commanded),
        "goal_completions_per_env": float(delta["goal_completions"]) / ENVS,
        "planning_wall_s": float(delta["total_planning_wall_s"]),
        "initial_reset_included": True,
    }


def record_id(route: str, speed: float, density: int) -> str:
    return "route_%s__speed_%s__bars_%d" % (
        route, BRAKE_VERIFY.speed_key(speed), density
    )


def row_gates(row: Mapping) -> Dict[str, bool]:
    local = row["off_rounded_local_infeasible_fraction"] if row["route_mode"] == "off" else (
        row["recovery_normal_route_rounded_invalidation_fraction"]
    )
    local_max = GATES["off_rounded_local_infeasible_fraction_max"] if row["route_mode"] == "off" else (
        GATES["recovery_normal_route_rounded_invalidation_fraction_max"]
    )
    telemetry = row["telemetry_summary"]
    return {
        "tracking": row["tracking_rmse_mps"] <= GATES["tracking_rmse_mps_max"],
        "speed": row["mean_speed_ratio"] >= GATES["mean_speed_ratio_min"],
        "contact": row["contact_step_fraction"] <= GATES["contact_step_fraction_max"],
        "local_feasibility": local <= local_max,
        "motor_saturation": row["motor_saturation_fraction"] <= GATES["motor_saturation_fraction_max"],
        "tilt": row["max_tilt_deg"] <= GATES["max_tilt_deg_max"],
        "finite_state": row["invalid_state_fraction"] <= GATES["invalid_state_fraction_max"],
        "watchdog": telemetry["watchdog_breach_substeps"] <= GATES["watchdog_breach_substeps_max"],
        "no_position_write": telemetry["direct_position_writes"] <= GATES["direct_position_writes_max"],
        "no_reset_in_advance": telemetry["reset_calls_during_advance"] <= GATES["reset_calls_during_advance_max"],
        "connect_actual_progress": (
            int(telemetry["connect_actual_regressions"]) <= GATES["connect_actual_regressions_max"]
        ),
    }


def run_cell(task, torch, route_mode: str, speed: float, density: int,
             import_origin: Mapping, raw_path: Path) -> Dict[str, object]:
    task.seed(SEED)
    route_before = normalized_route_diagnostics(task)
    BASE._sync_cuda(torch)
    started = time.perf_counter()
    task._set_active_bars(density)
    task.reset()
    BASE._sync_cuda(torch)
    initial_reset_s = time.perf_counter() - started
    require(int(task.n_bars_active) == density, "active bar count drift")
    centers = task.obs_dict["obstacle_position"][
        :, task._bar_offset:task._bar_offset + density, :2
    ]
    half = task.obs_dict["asset_collision_half_extents"][
        :, task._bar_offset:task._bar_offset + density, :2
    ]
    require(bool(torch.isfinite(centers).all()) and bool(torch.isfinite(half).all())
            and bool((half > 0).all()), "active AABB geometry invalid")
    layout_sha = BASE.initial_layout_sha256(task, density)
    robot_sha = BASE.tensor_digest_sha256(
        task.obs_dict["robot_position"], task.obs_dict["robot_orientation"]
    )
    target_sha = BASE.tensor_digest_sha256(task.target_position, task.target_orientation)
    waypoint_sha = BASE.tensor_digest_sha256(task._tm_waypoint)
    route_goal_sha = BASE.tensor_digest_sha256(task._target_route_manager.goal) if (
        route_mode != "off"
    ) else None

    observer = PACKED.RecoveryPackedObserver(task, STEPS, PHYSICS_SUBSTEPS)
    ctrl = task._target_controller
    zero = torch.zeros((ENVS, 4), device=task.device)
    speed_sum = err_sq_sum = 0.0
    tracking_samples = safety_samples = contact_samples = invalid_samples = 0
    off_infeasible = 0
    saturation = torch.zeros((), dtype=torch.long, device=task.device)
    substeps = torch.zeros((), dtype=torch.long, device=task.device)
    max_tilt = torch.zeros((), dtype=task.target_position.dtype, device=task.device)
    reset_batches, reset_envs, reset_total = 1, ENVS, initial_reset_s
    clock_start = int(task.num_task_steps)
    BASE._sync_cuda(torch)
    rollout_started = time.perf_counter()
    try:
        for step in range(STEPS):
            interval_clock = BASE.begin_low_level_evaluation_interval(task)
            observer.begin_interval(step)
            observer.advance_target()
            command = task.transform_action_to_command(zero)
            require(bool(command.isfinite().all()), "neutral pursuer command is nonfinite")
            sat_before = ctrl.saturation_substeps.clone()
            sub_before = ctrl.substeps.clone()
            task.sim_env.step(actions=command)
            observer.finish_interval()
            sat_delta = ctrl.saturation_substeps - sat_before
            sub_delta = ctrl.substeps - sub_before
            require(bool((sat_delta >= 0).all()) and bool((sub_delta >= 0).all()),
                    "controller counter decreased")
            saturation += sat_delta.sum()
            substeps += sub_delta.sum()
            max_tilt = torch.maximum(max_tilt, ctrl.max_tilt_seen_rad.max())
            if step >= WARMUP_STEPS:
                actual = task.target_vel_w[:, :2]
                desired = ctrl.velocity_command[:, :2]
                speed_sum += float(actual.norm(dim=1).sum().item())
                err_sq_sum += float(((actual - desired) ** 2).sum().item())
                tracking_samples += ENVS
            safety_samples += ENVS
            contact = ctrl.contact_seen.clone()
            contact_samples += int(contact.sum().item())
            if route_mode == "off":
                off_infeasible += int((~task._tm_last_step_feasible).sum().item())
            bounds_lo, bounds_hi = task.obs_dict["env_bounds_min"], task.obs_dict["env_bounds_max"]
            support = task._physical_target_support_xyz()
            invalid = (
                (task.target_position[:, :2] - support[:, :2] < bounds_lo[:, :2])
                | (task.target_position[:, :2] + support[:, :2] > bounds_hi[:, :2])
            ).any(dim=1)
            invalid |= (
                (task.target_position[:, 2] - support[:, 2] < bounds_lo[:, 2])
                | (task.target_position[:, 2] + support[:, 2] > bounds_hi[:, 2])
                | ~torch.isfinite(task.target_position).all(dim=1)
            )
            invalid_samples += int((invalid & ~contact).sum().item())
            failed = contact | invalid
            if bool(failed.any()):
                ids = failed.nonzero(as_tuple=False).squeeze(-1)
                observer.mark_runner_reset(ids)
                timing = BASE._timed_reset(task, torch, ids)
                reset_batches += 1
                reset_envs += timing["envs"]
                reset_total += timing["wall_s"]
            BASE.finish_low_level_evaluation_interval(task, interval_clock)
    finally:
        observer.close()
    BASE._sync_cuda(torch)
    rollout_wall = time.perf_counter() - rollout_started
    route_after = normalized_route_diagnostics(task)
    route = route_delta(route_before, route_after, ENVS * STEPS)
    local_invalid = None
    if route_mode != "off":
        local_invalid = int(route["counter_delta"]["invalidation_counts"]["local_step_infeasible"])
    telemetry_artifact = observer.write(raw_path, {
        "record_id": record_id(route_mode, speed, density), "route_mode": route_mode,
        "speed_mps": speed, "bars": density, "seed": SEED,
        "brake_timeout_steps": int(math.ceil(
            (float(getattr(task.tm, "recovery_brake_stop_time_p95", 0.0)) + 0.20) / task.step_dt
        )) if route_mode != "off" else 0,
    })
    telemetry_summary = PACKED.load_and_verify(raw_path, telemetry_artifact["sha256"])
    observer_device_s = telemetry_summary["candidate_observer_device_ms"] / 1000.0
    observer_connector_s = telemetry_summary["connector_observer_cpu_ms"] / 1000.0
    adjusted_wall = max(rollout_wall - observer_device_s - observer_connector_s, 1e-9)
    require(tracking_samples == ENVS * (STEPS - WARMUP_STEPS), "tracking denominator drift")
    require(safety_samples == ENVS * STEPS, "safety denominator drift")
    mean_speed = speed_sum / max(1, tracking_samples)
    row = {
        "record_id": record_id(route_mode, speed, density), "seed": SEED,
        "route_mode": route_mode, "speed_mps": speed, "bars": density,
        "envs": ENVS, "steps": STEPS, "warmup_steps": WARMUP_STEPS,
        "initial_layout_sha256": layout_sha, "initial_robot_pose_sha256": robot_sha,
        "initial_target_pose_sha256": target_sha, "initial_task_waypoint_sha256": waypoint_sha,
        "initial_route_goal_sha256": route_goal_sha,
        "mean_speed_mps": mean_speed, "mean_speed_ratio": mean_speed / speed,
        "tracking_rmse_mps": math.sqrt(err_sq_sum / max(1, tracking_samples)),
        "contact_step_fraction": contact_samples / max(1, safety_samples),
        "off_rounded_local_infeasible_fraction": (
            off_infeasible / max(1, safety_samples) if route_mode == "off" else None
        ),
        "recovery_normal_route_rounded_invalidation_fraction": (
            local_invalid / max(1, safety_samples) if route_mode != "off" else None
        ),
        "invalid_state_fraction": invalid_samples / max(1, safety_samples),
        "motor_saturation_fraction": float(saturation.float().div(substeps.clamp(min=1)).item()),
        "max_tilt_deg": math.degrees(float(max_tilt.item())),
        "measurement_denominators": {
            "tracking_env_intervals": tracking_samples, "safety_env_intervals": safety_samples,
            "controller_substeps": int(substeps.item()),
        },
        "local_metric_contract": {
            "numerator": off_infeasible if route_mode == "off" else local_invalid,
            "denominator": safety_samples,
            "geometry": (
                "bounded rounded Euclidean surface clearance"
                if route_mode == "off"
                else "normal-route rounded invalidation only; exact CONNECT failures are in recovery reasons"
            ),
        },
        "route": route,
        "telemetry": {"path": raw_path.name, "sha256": telemetry_artifact["sha256"],
                      "schema": PACKED.SCHEMA},
        "telemetry_summary": telemetry_summary,
        "reset_wall": {"batches": reset_batches, "reset_envs": reset_envs,
                       "total_s": reset_total},
        "throughput": {
            "rollout_wall_s_instrumented": rollout_wall,
            "candidate_observer_device_s": observer_device_s,
            "connector_observer_cpu_s": observer_connector_s,
            "rollout_wall_s_observer_adjusted": adjusted_wall,
            # The preregistered operational gate conservatively uses instrumented wall time.
            "env_intervals_per_s": ENVS * STEPS / max(rollout_wall, 1e-9),
            "env_intervals_per_s_observer_adjusted": ENVS * STEPS / adjusted_wall,
        },
        "task_clock": {"start": clock_start, "end": int(task.num_task_steps),
                       "increments": int(task.num_task_steps) - clock_start},
        "import_origin": dict(import_origin),
    }
    row["gates"] = row_gates(row)
    row["pass"] = all(row["gates"].values())
    row["record_sha256"] = hashlib.sha256(
        json.dumps({key: row[key] for key in (
            "record_id", "seed", "route_mode", "speed_mps", "bars",
            "initial_layout_sha256", "initial_robot_pose_sha256", "initial_target_pose_sha256",
            "initial_task_waypoint_sha256", "initial_route_goal_sha256",
        )}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return row


def configure_child(route_mode: str, speed: float, probe: Mapping[str, str]):
    values = frozen_environment(route_mode, speed, probe)
    os.environ.update(values)
    return values


def child_main(args) -> int:
    require(args.route_mode in ROUTE_ARMS and args.speed in SPEEDS, "child arm outside grid")
    manifest_path = Path(args.source_manifest).resolve()
    source = verify_source_manifest(manifest_path, args.source_manifest_sha256)
    probe = json.loads(args.probe_json)
    environment = configure_child(args.route_mode, args.speed, probe)
    os.chdir(str(ROOT))
    sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != ROOT]
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry
    import aerial_gym
    import torch

    origin = Path(aerial_gym.__file__).resolve()
    expected = (ROOT / "aerial_gym/__init__.py").resolve()
    hashes = source_hashes(source)
    require(origin == expected and sha256_file(origin) == hashes["aerial_gym/__init__.py"],
            "aerial_gym import origin/hash drift")
    import_origin = {"enforced": True, "path": str(origin), "sha256": sha256_file(origin),
                     "manifest_sha256": hashes["aerial_gym/__init__.py"]}
    task = task_registry.make_task("navrl_task", seed=SEED, num_envs=ENVS,
                                   headless=True, use_warp=True)
    require(task._target_dynamics == "physical" and task._target_route_mode == args.route_mode,
            "instantiated target contract drift")
    require(str(task.tm.pattern) == "waypoint" and abs(float(task.tm.speed_fixed) - args.speed) < 1e-9,
            "target pattern/speed drift")
    # Reuse only the v1-independent physics/robot checks.  Passing a forged v1 mode here would
    # make the receipt claim that a different state machine was instantiated.
    contract = BASE.attest_instantiated_contract(task, "off", source)
    contract["actual_route_mode"] = args.route_mode
    contract["training_source_manifest_sha256"] = str(
        task._training_source_provenance.get("manifest_sha256", "")
    )
    require(contract["training_source_manifest_sha256"]
            == probe["NAVRL_TRAINING_SOURCE_MANIFEST_SHA256"],
            "instantiated training source receipt drift")
    if args.route_mode != "off":
        support = [float(value) for value in task._target_route_support_xy[0].tolist()]
        declared = contract["declared_conservative_support_xy_m"]
        require(all(abs(value - expected) < 1e-6 for value, expected in zip(support, declared)),
                "recovery route support differs from declared physical envelope")
        contract["active_route_support_xy_m"] = support
    contract["recovery_probe_receipt_sha256"] = (
        probe.get("NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256")
        if args.route_mode != "off" else None
    )
    software = BASE.runtime_software_provenance(torch, source)
    software["nvidia_smi"] = nvidia_smi_provenance()
    raw_dir = Path(args.raw_dir).resolve()
    rows = []
    for density in DENSITIES:
        raw = raw_dir / (record_id(args.route_mode, args.speed, density).replace(".", "p") + ".npz")
        rows.append(run_cell(task, torch, args.route_mode, args.speed, density, import_origin, raw))
    verify_source_manifest(manifest_path, args.source_manifest_sha256)
    recorded_environment = dict(environment)
    recorded_environment["NAVRL_TRAINING_SOURCE_MANIFEST"] = (
        "inputs/training_source/source_manifest.json"
    )
    if args.route_mode != "off":
        recorded_environment["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT"] = (
            "inputs/braking_probe/receipt.json"
        )
    payload = {
        "schema": CHILD_SCHEMA, "seed": SEED, "route_mode": args.route_mode,
        "speed_mps": args.speed, "densities": list(DENSITIES), "envs": ENVS,
        "steps": STEPS, "physics_substeps": PHYSICS_SUBSTEPS,
        "process_identity": {"pid": os.getpid(), "ppid": os.getppid(),
                             "nonce": os.urandom(16).hex()},
        "source_manifest_sha256": args.source_manifest_sha256,
        "environment_contract": recorded_environment, "import_origin": import_origin,
        "instantiated_contract": contract, "software_provenance": software, "cells": rows,
    }
    atomic_json(Path(args.child_output), payload)
    return 0


def validate_grid(records: Sequence[Mapping]) -> None:
    expected = {
        record_id(route, speed, density)
        for route in ROUTE_ARMS for speed in SPEEDS for density in DENSITIES
    }
    ids = [str(row.get("record_id")) for row in records]
    require(len(ids) == 32 and len(set(ids)) == 32 and set(ids) == expected,
            "32-cell identity/accounting mismatch")
    index = {(row["speed_mps"], row["bars"], row["route_mode"]): row for row in records}
    for speed in SPEEDS:
        for density in DENSITIES:
            off = index[(speed, density, "off")]
            recovery = index[(speed, density, "global_astar_recovery_v2")]
            for digest in (
                "initial_layout_sha256", "initial_robot_pose_sha256",
                "initial_target_pose_sha256",
            ):
                require(off[digest] == recovery[digest], "matched arm initial-state drift: %s" % digest)
            ratio = recovery["throughput"]["env_intervals_per_s"] / max(
                off["throughput"]["env_intervals_per_s"], 1e-9
            )
            require(ratio >= GATES["matched_recovery_vs_off_throughput_ratio_min"],
                    "matched recovery/off throughput ratio below preregistered bound")
    for density in DENSITIES:
        for route in ROUTE_ARMS:
            subset = [
                row for row in records
                if row["bars"] == density and row["route_mode"] == route
            ]
            waypoints = {row["initial_task_waypoint_sha256"] for row in subset}
            require(len(waypoints) == 1,
                    "speed-arm initial waypoint drift at %s/%s bars" % (route, density))
            if route == "off":
                require(all(row.get("initial_route_goal_sha256") is None for row in subset),
                        "off arm must not record a route goal digest")
            else:
                goals = {row.get("initial_route_goal_sha256") for row in subset}
                require(len(goals) == 1 and None not in goals,
                        "speed-arm routed goal drift at %s bars" % density)
                require(all(
                    row["initial_task_waypoint_sha256"] == row["initial_route_goal_sha256"]
                    for row in subset
                ), "recovery waypoint digest must equal the owned route goal")
    for row in records:
        require(row.get("seed") == SEED and row.get("envs") == ENVS
                and row.get("steps") == STEPS and row.get("warmup_steps") == WARMUP_STEPS,
                "cell seed/env/step contract drift")
        require(row.get("task_clock", {}).get("increments") == STEPS,
                "task clock must increment exactly once per interval")
        require(row["gates"] == row_gates(row), "row gate recomputation mismatch")
        require(row["pass"] == all(row["gates"].values()), "row conjunctive gate mismatch")
        require(row["measurement_denominators"]["tracking_env_intervals"] == ENVS * (STEPS-WARMUP_STEPS),
                "tracking denominator mismatch")
        require(row["measurement_denominators"]["safety_env_intervals"] == ENVS * STEPS,
                "safety denominator mismatch")
        local_contract = row.get("local_metric_contract", {})
        require(local_contract.get("denominator") == ENVS * STEPS,
                "local metric denominator mismatch")
        local_value = (
            row["off_rounded_local_infeasible_fraction"]
            if row["route_mode"] == "off"
            else row["recovery_normal_route_rounded_invalidation_fraction"]
        )
        require(abs(float(local_value) - float(local_contract.get("numerator", -1))
                    / float(ENVS * STEPS)) <= 1e-15,
                "local metric numerator/fraction identity mismatch")
        expected_geometry = (
            "bounded rounded Euclidean surface clearance"
            if row["route_mode"] == "off"
            else "normal-route rounded invalidation only; exact CONNECT failures are in recovery reasons"
        )
        require(local_contract.get("geometry") == expected_geometry,
                "local metric geometry semantics drift")


def matched_deltas(records: Sequence[Mapping]) -> List[Dict[str, object]]:
    index = {(row["speed_mps"], row["bars"], row["route_mode"]): row for row in records}
    result = []
    for speed in SPEEDS:
        for density in DENSITIES:
            off = index[(speed, density, "off")]
            rec = index[(speed, density, "global_astar_recovery_v2")]
            result.append({
                "speed_mps": speed, "bars": density,
                "throughput_ratio_recovery_over_off": rec["throughput"]["env_intervals_per_s"]
                / max(off["throughput"]["env_intervals_per_s"], 1e-9),
                "contact_fraction_delta": rec["contact_step_fraction"] - off["contact_step_fraction"],
                "invalid_fraction_delta": rec["invalid_state_fraction"] - off["invalid_state_fraction"],
            })
    return result


def derive_verdict(records: Sequence[Mapping]) -> Dict[str, object]:
    validate_grid(records)
    recovery = [row for row in records if row["route_mode"] == "global_astar_recovery_v2"]
    routed70 = [row for row in recovery if row["bars"] == 70]
    attempts = sum(row["route"]["counter_delta"]["plan_attempts"] for row in routed70)
    successes = sum(row["route"]["counter_delta"]["plan_successes"] for row in routed70)
    fallback = sum(row["route"]["counter_delta"]["fallback_intervals"] for row in routed70)
    commanded = ENVS * STEPS * len(routed70)
    low = next(row for row in routed70 if row["speed_mps"] == 0.6)
    route_mechanism = (
        attempts > 0 and successes / attempts >= 0.99
        and fallback / max(1, commanded) <= 0.01
        and low["route"]["goal_completions_per_env"] >= 0.5
    )
    return {
        "execution_integrity": "PASS_32_CELL_INTEGRITY",
        "all_cell_controller_and_recovery_gates": all(row["pass"] for row in records),
        "route_mechanism": "PASS_ROUTE_MECHANISM" if route_mechanism else "FAIL_ROUTE_MECHANISM",
        "long_training_authorized": False,
    }


def validate_child(path: Path, route: str, speed: float, source_sha: str,
                   result_root: Path) -> Dict[str, object]:
    require(path.is_file(), "child missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == CHILD_SCHEMA and payload.get("route_mode") == route
            and payload.get("speed_mps") == speed, "child identity drift")
    require(payload.get("source_manifest_sha256") == source_sha, "child source drift")
    origin = payload.get("import_origin")
    require(isinstance(origin, Mapping) and origin.get("enforced") is True,
            "child import origin is not fail-closed")
    require(payload.get("environment_contract", {}).get("NAVRL_NUM_BARS") == "70"
            and payload.get("environment_contract", {}).get("NAVRL_MAX_BARS") == "300",
            "child density pool contract drift")
    training_sha = payload.get("environment_contract", {}).get(
        "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256"
    )
    require(len(str(training_sha or "")) == 64
            and payload.get("instantiated_contract", {}).get(
                "training_source_manifest_sha256"
            ) == training_sha, "child training source receipt drift")
    nvidia = payload.get("software_provenance", {}).get("nvidia_smi", {})
    require(Path(str(nvidia.get("path", ""))).is_absolute()
            and len(str(nvidia.get("sha256", ""))) == 64 and nvidia.get("identity"),
            "child nvidia-smi provenance is incomplete")
    cells = payload.get("cells")
    require(isinstance(cells, list) and len(cells) == 4, "child cell count drift")
    identity = payload.get("process_identity", {})
    require(int(identity.get("pid", 0)) > 0 and int(identity.get("ppid", 0)) > 0
            and len(str(identity.get("nonce", ""))) == 32,
            "fresh-child process identity is incomplete")
    for row, density in zip(cells, DENSITIES):
        require(row["record_id"] == record_id(route, speed, density), "child cell order drift")
        require(row.get("import_origin") == origin, "cell/child import origin drift")
        raw = result_root / "raw" / row["telemetry"]["path"]
        summary = PACKED.load_and_verify(raw, row["telemetry"]["sha256"])
        require(summary == row["telemetry_summary"], "packed telemetry summary drift")
    return payload


def file_entry(path: Path, root: Path) -> Dict[str, object]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path),
            "size": path.stat().st_size}


def parent_main(args) -> int:
    final_dir = Path(args.output_dir).resolve()
    require(not final_dir.exists(), "refusing to overwrite final result directory")
    stage = final_dir.with_name(".%s.incomplete.%d" % (final_dir.name, os.getpid()))
    require(not stage.exists(), "stale incomplete result directory exists")
    stage.mkdir(parents=True)
    (stage / "children").mkdir()
    (stage / "logs").mkdir()
    (stage / "raw").mkdir()
    renamed = False
    try:
        source = build_source_manifest()
        source_path = stage / "source_manifest.json"
        atomic_json(source_path, source)
        source_sha = sha256_file(source_path)
        probe = snapshot_braking_probe(braking_probe_contract(), stage)
        probe_input = Path(probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT"])
        probe_entry = file_entry(probe_input, stage)
        probe_bundle = [
            file_entry(path, stage)
            for path in sorted(probe_input.parent.rglob("*")) if path.is_file()
        ]
        require(probe_bundle, "snapshotted braking receipt bundle is empty")
        training_dir = stage / "inputs/training_source"
        training_contract = SOURCE_BUNDLE.create(training_dir, require_clean=True)
        SOURCE_BUNDLE.verify(
            Path(training_contract["manifest"]), training_contract["manifest_sha256"]
        )
        training_bundle = [
            file_entry(path, stage) for path in sorted(training_dir.rglob("*")) if path.is_file()
        ]
        require(training_bundle, "training source snapshot is empty")
        probe.update({
            "NAVRL_TRAINING_SOURCE_MANIFEST": training_contract["manifest"],
            "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256": training_contract["manifest_sha256"],
            "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT": "1",
            "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE": "1",
        })
        base_env = build_child_environment()
        records, child_entries, contracts, software = [], [], [], []
        for route in ROUTE_ARMS:
            for speed in SPEEDS:
                label = ("route_%s__speed_%s" % (
                    route, BRAKE_VERIFY.speed_key(speed)
                )).replace(".", "p")
                child = stage / "children" / (label + ".json")
                log = stage / "logs" / (label + ".log")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--_child",
                    "--route-mode", route, "--speed", str(speed),
                    "--source-manifest", str(source_path),
                    "--source-manifest-sha256", source_sha,
                    "--probe-json", json.dumps(probe, sort_keys=True),
                    "--child-output", str(child), "--raw-dir", str(stage / "raw"),
                ]
                with log.open("x", encoding="utf-8") as stream:
                    completed = subprocess.run(command, cwd=str(ROOT), env=base_env,
                                               stdout=stream, stderr=subprocess.STDOUT, check=False)
                require(completed.returncode == 0, "child failed: %s" % label)
                payload = validate_child(child, route, speed, source_sha, stage)
                records.extend(payload["cells"])
                contracts.append({"route_mode": route, "speed_mps": speed,
                                  "contract": payload["instantiated_contract"]})
                software.append({"route_mode": route, "speed_mps": speed,
                                 "provenance": payload["software_provenance"]})
                child_entries.append({
                    "route_mode": route, "speed_mps": speed,
                    "child": file_entry(child, stage), "log": file_entry(log, stage),
                })
        validate_grid(records)
        verify_source_manifest(source_path, source_sha)
        raw_entries = [file_entry(path, stage) for path in sorted((stage / "raw").glob("*.npz"))]
        require(len(raw_entries) == 32, "raw artifact count is not 32")
        execution = {
            "schema": EXECUTION_SCHEMA, "source_manifest_sha256": source_sha,
            "probe_receipt_sha256": probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"],
            "probe_receipt": probe_entry,
            "probe_bundle": probe_bundle,
            "training_source_bundle": training_bundle,
            "training_source_manifest_sha256": training_contract["manifest_sha256"],
            "children": child_entries, "raw_artifacts": raw_entries,
            "record_ids": [row["record_id"] for row in records], "record_count": len(records),
        }
        execution_path = stage / "execution_manifest.json"
        atomic_json(execution_path, execution)
        summary = {
            "schema": SCHEMA, "preregistration": str(PREREG.relative_to(ROOT)),
            "seed": SEED, "contract_variant": CONTRACT_VARIANT,
            "route_arms": list(ROUTE_ARMS), "speeds_mps": list(SPEEDS),
            "densities": list(DENSITIES), "envs": ENVS, "steps": STEPS,
            "physics_substeps": PHYSICS_SUBSTEPS, "gates_preregistered": GATES,
            "source_manifest_sha256": source_sha,
            "execution_manifest_sha256": sha256_file(execution_path),
            "probe_receipt_sha256": probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"],
            "probe_contract": {
                "decel_p05_mps2": float(probe["NAVRL_TARGET_RECOVERY_BRAKE_P05"]),
                "stop_time_p95_s": float(probe["NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S"]),
                "receipt": probe_entry,
                "bundle": probe_bundle,
            },
            "instantiated_contracts": contracts, "software_provenance": software,
            "training_source_contract": {
                "manifest_sha256": training_contract["manifest_sha256"],
                "bundle": training_bundle,
            },
            "cells": records, "matched_recovery_minus_off": matched_deltas(records),
            "verdict": derive_verdict(records),
            "claim_boundary": {"ppo_policy_loaded": False, "hardware_validation": False,
                               "long_training_authorized": False},
        }
        summary_path = stage / "summary.json"
        atomic_json(summary_path, summary)
        receipt = {
            "schema": RECEIPT_SCHEMA, "summary": file_entry(summary_path, stage),
            "execution_manifest": file_entry(execution_path, stage),
            "source_manifest": file_entry(source_path, stage),
            "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
            "packed_telemetry_source_sha256": sha256_file(PACKED_PATH),
            "probe_receipt_sha256": probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"],
            "probe_receipt": probe_entry,
            "probe_bundle": probe_bundle,
            "training_source_bundle": training_bundle,
            "training_source_manifest_sha256": training_contract["manifest_sha256"],
            "record_count": 32, "record_ids": [row["record_id"] for row in records],
            "children": child_entries, "raw_artifacts": raw_entries,
            "verdict": summary["verdict"],
        }
        receipt_path = stage / "receipt.json"
        atomic_json(receipt_path, receipt)
        verify_result(stage)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(stage), str(final_dir))
        renamed = True
        verify_result(final_dir)
        print("VERIFIED %s PASS_32_CELL_INTEGRITY" % final_dir)
        return 0
    except Exception as exc:
        void = {"schema": "navrl_recovery_v2_void_v1", "error": str(exc),
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
        failed_root = final_dir if renamed and final_dir.exists() else stage
        if failed_root.exists() and not (failed_root / "VOID.json").exists():
            atomic_json(failed_root / "VOID.json", void)
        void_dir = final_dir.with_name(".%s.VOID.%d" % (final_dir.name, os.getpid()))
        if failed_root.exists() and not void_dir.exists():
            os.replace(str(failed_root), str(void_dir))
        raise


def verify_entry(root: Path, entry: Mapping) -> Path:
    relative = Path(str(entry.get("path", "")))
    require(relative and not relative.is_absolute() and ".." not in relative.parts,
            "unsafe receipt path")
    path = root / relative
    require(path.is_file() and path.stat().st_size == int(entry.get("size", -1))
            and sha256_file(path) == entry.get("sha256"), "receipt-bound file drift: %s" % relative)
    return path


def verify_result(root: Path) -> Dict[str, object]:
    root = root.resolve()
    receipt_path = root / "receipt.json"
    require(receipt_path.is_file(), "final receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == RECEIPT_SCHEMA, "wrong final receipt schema")
    source_path = verify_entry(root, receipt["source_manifest"])
    source = verify_source_manifest(source_path, receipt["source_manifest"]["sha256"])
    summary_path = verify_entry(root, receipt["summary"])
    execution_path = verify_entry(root, receipt["execution_manifest"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == SCHEMA and execution.get("schema") == EXECUTION_SCHEMA,
            "summary/execution schema mismatch")
    require(summary.get("source_manifest_sha256") == receipt["source_manifest"]["sha256"],
            "summary/source receipt mismatch")
    require(summary.get("execution_manifest_sha256") == receipt["execution_manifest"]["sha256"],
            "summary/execution receipt mismatch")
    require(receipt.get("evaluator_source_sha256") == sha256_file(Path(__file__).resolve()),
            "evaluator source drift")
    require(receipt.get("packed_telemetry_source_sha256") == sha256_file(PACKED_PATH),
            "packed telemetry source drift")
    probe_path = verify_entry(root, receipt["probe_receipt"])
    for entry in receipt.get("probe_bundle", []):
        verify_entry(root, entry)
    probe_contract = summary.get("probe_contract", {})
    validate_braking_probe_values({
        "NAVRL_TARGET_RECOVERY_BRAKE_P05": str(probe_contract.get("decel_p05_mps2", "")),
        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S": str(probe_contract.get("stop_time_p95_s", "")),
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT": str(probe_path),
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256": str(
            receipt.get("probe_receipt_sha256", "")
        ),
    })
    require(execution.get("probe_receipt") == receipt.get("probe_receipt")
            == probe_contract.get("receipt"), "probe receipt binding drift")
    require(execution.get("probe_bundle") == receipt.get("probe_bundle")
            == probe_contract.get("bundle") and len(receipt.get("probe_bundle", [])) >= 8,
            "probe bundle binding drift")
    for entry in receipt.get("training_source_bundle", []):
        verify_entry(root, entry)
    training_contract = summary.get("training_source_contract", {})
    require(execution.get("training_source_bundle") == receipt.get("training_source_bundle")
            == training_contract.get("bundle") and receipt.get("training_source_bundle"),
            "training source bundle binding drift")
    require(execution.get("training_source_manifest_sha256")
            == receipt.get("training_source_manifest_sha256")
            == training_contract.get("manifest_sha256"),
            "training source manifest binding drift")
    SOURCE_BUNDLE.verify(
        root / "inputs/training_source/source_manifest.json",
        receipt["training_source_manifest_sha256"],
    )
    for entry in receipt.get("raw_artifacts", []):
        path = verify_entry(root, entry)
        PACKED.load_and_verify(path, entry["sha256"])
    require(len(receipt.get("raw_artifacts", [])) == 32, "receipt raw count drift")
    child_records = []
    child_identities = []
    process_nonces = []
    for item in receipt.get("children", []):
        child_path = verify_entry(root, item["child"])
        verify_entry(root, item["log"])
        child = validate_child(
            child_path, str(item["route_mode"]), float(item["speed_mps"]),
            receipt["source_manifest"]["sha256"], root,
        )
        child_records.extend(child["cells"])
        child_identities.append((child["route_mode"], child["speed_mps"]))
        process_nonces.append(child["process_identity"]["nonce"])
    require(len(child_identities) == 8 and len(set(child_identities)) == 8,
            "fresh-child route/speed identity accounting drift")
    require(len(set(process_nonces)) == 8, "fresh-child process nonce reuse")
    records = summary.get("cells")
    require(isinstance(records, list), "summary cells missing")
    require(child_records == records, "child records differ from summary cells")
    validate_grid(records)
    require(derive_verdict(records) == summary.get("verdict") == receipt.get("verdict"),
            "final verdict recomputation mismatch")
    require(execution.get("record_count") == receipt.get("record_count") == 32,
            "final record count drift")
    require(execution.get("record_ids") == receipt.get("record_ids"),
            "final record IDs drift")
    require(execution.get("children") == receipt.get("children")
            and execution.get("raw_artifacts") == receipt.get("raw_artifacts"),
            "execution/receipt artifact binding drift")
    verify_source_manifest(source_path, receipt["source_manifest"]["sha256"])
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--verify")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--route-mode", choices=ROUTE_ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--speed", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--source-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--source-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--probe-json", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    parser.add_argument("--raw-dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.preflight:
        require(not args.verify and not args._child, "--preflight is standalone")
        source = build_source_manifest()
        probe = braking_probe_contract()
        print(json.dumps({
            "schema": SCHEMA, "gpu_started": False, "git_commit": source["git_commit"],
            "runtime_file_count": source["runtime_file_count"], "cell_count": 32,
            "contract_variant": CONTRACT_VARIANT,
            "route_arms": ROUTE_ARMS, "speeds_mps": SPEEDS, "densities": DENSITIES,
            "envs": ENVS, "steps": STEPS,
            "probe_receipt_sha256": probe["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"],
        }, sort_keys=True))
        return 0
    if args.verify:
        require(not args._child, "--verify cannot be child mode")
        verify_result(Path(args.verify))
        print("VERIFIED %s PASS_32_CELL_INTEGRITY" % Path(args.verify).resolve())
        return 0
    if args._child:
        require(all((args.route_mode, args.speed is not None, args.source_manifest,
                     args.source_manifest_sha256, args.probe_json, args.child_output, args.raw_dir)),
                "child arguments incomplete")
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrityError as exc:
        print("[recovery-v2-gate] REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
