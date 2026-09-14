#!/usr/bin/env python3
"""Braking-route-v3 simulator cell adapter.

The v3 gate launcher (``tools/run_navrl_braking_route_v3_gate.py``) starts one fresh process of
this executable per cell and passes the frozen cell contract through ``NAVRL_V3_*`` environment
variables.  This adapter reuses the corrected-r2 evaluator's frozen environment and canonical
neutral-pursuer interval helpers byte-for-byte (loaded from the tracked r2 module), runs exactly
one route-arm x speed x density cell, and atomically writes the cell JSON expected by
``tools/verify_navrl_braking_route_v3_gate.py``.

Importing this module is CPU-safe.  Isaac Gym and torch are imported only inside ``main``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Mapping


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GATE = _load("navrl_v3_cell_gate_contract", "tools/verify_navrl_braking_route_v3_gate.py")
R2 = _load("navrl_v3_cell_r2_helpers", "tools/verify_navrl_corrected_nonoverlap_route_gate_r2.py")

# Environment keys the gate launcher must have provided.  The braking lookup values are injected
# by the gate only after the raw canonical receipt re-verified; this adapter never invents them.
REQUIRED_GATE_KEYS = (
    "NAVRL_V3_CELL_OUTPUT",
    "NAVRL_V3_RECORD_ID",
    "NAVRL_V3_STAGE",
    "NAVRL_V3_SEED",
    "NAVRL_V3_ROUTE_MODE",
    "NAVRL_V3_SPEED_MPS",
    "NAVRL_V3_BARS",
    "NAVRL_V3_ENVS",
    "NAVRL_V3_STEPS",
    "NAVRL_V3_WARMUP_STEPS",
    "NAVRL_V3_IDENTITY_JSON",
    "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT",
    "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256",
    "NAVRL_TARGET_RECOVERY_PROBE_VALIDATED",
    "NAVRL_TARGET_RECOVERY_BRAKE_P05",
    "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S",
    "NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS",
    "NAVRL_TARGET_RECOVERY_BRAKE_STOP_DISTANCES_M",
    "NAVRL_TARGET_RECOVERY_BRAKE_LATERAL_TUBE_P95_M",
)
# The gate strips every inherited NAVRL_* variable, so the training-source receipt is forwarded
# under a passthrough prefix and remapped here before aerial_gym imports read the environment.
TRAINING_SOURCE_PASSTHROUGH = (
    "MOTAR_V3_TRAINING_SOURCE_MANIFEST",
    "MOTAR_V3_TRAINING_SOURCE_MANIFEST_SHA256",
)


class CellContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CellContractError(message)


def read_cell_contract(environ: Mapping[str, str]) -> Dict[str, object]:
    """Parse and fail-closed-validate the frozen cell contract from the environment."""
    for key in REQUIRED_GATE_KEYS + TRAINING_SOURCE_PASSTHROUGH:
        require(bool(str(environ.get(key, "")).strip()), f"missing cell contract variable: {key}")
    stage = environ["NAVRL_V3_STAGE"].strip()
    require(stage in GATE.STAGES, f"unknown stage: {stage}")
    config = GATE.STAGES[stage]
    seed = int(environ["NAVRL_V3_SEED"])
    require(seed == config["seed"], "cell seed differs from frozen stage seed")
    route = environ["NAVRL_V3_ROUTE_MODE"].strip()
    require(route in GATE.ROUTE_ARMS, f"route arm outside frozen grid: {route}")
    speed = float(environ["NAVRL_V3_SPEED_MPS"])
    require(speed in GATE.SPEEDS, f"speed outside frozen grid: {speed}")
    bars = int(environ["NAVRL_V3_BARS"])
    require(bars in config["densities"], f"density outside frozen grid: {bars}")
    envs = int(environ["NAVRL_V3_ENVS"])
    steps = int(environ["NAVRL_V3_STEPS"])
    warmup = int(environ["NAVRL_V3_WARMUP_STEPS"])
    require(envs == GATE.ENVS and steps == GATE.STEPS and warmup == GATE.WARMUP_STEPS,
            "env/step/warmup contract drift")
    record = environ["NAVRL_V3_RECORD_ID"].strip()
    require(record == GATE.record_id(route, speed, bars), "record id/grid mismatch")
    output = Path(environ["NAVRL_V3_CELL_OUTPUT"]).resolve()
    require(not output.exists(), f"cell output already exists: {output}")
    identity = json.loads(environ["NAVRL_V3_IDENTITY_JSON"])
    require(isinstance(identity, dict), "identity contract is not an object")
    require(identity.get("preregistration_sha256") == GATE.PREREG_SHA256,
            "identity preregistration SHA drift")
    require(
        identity.get("braking_receipt_sha256")
        == environ["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"].strip().lower(),
        "identity braking-receipt hash differs from injected receipt hash",
    )
    require(identity.get("cell_runner_sha256") == GATE.sha256_file(Path(__file__).resolve()),
            "identity cell-runner hash differs from these bytes")
    manifest = Path(environ["MOTAR_V3_TRAINING_SOURCE_MANIFEST"]).resolve()
    require(manifest.is_file(), f"training source manifest missing: {manifest}")
    manifest_sha = environ["MOTAR_V3_TRAINING_SOURCE_MANIFEST_SHA256"].strip().lower()
    require(GATE.sha256_file(manifest) == manifest_sha,
            "training source manifest SHA-256 mismatch")
    return {
        "stage": stage, "seed": seed, "route_mode": route, "speed_mps": speed, "bars": bars,
        "envs": envs, "steps": steps, "warmup_steps": warmup, "record_id": record,
        "output": output, "identity": identity,
        "training_source_manifest": str(manifest),
        "training_source_manifest_sha256": manifest_sha,
    }


def frozen_cell_environment(contract: Mapping[str, object]) -> Dict[str, str]:
    """Exact corrected-r2 frozen environment for this arm, plus the training-source receipt.

    ``R2.frozen_environment`` is reused so the physical target, arena, placement, and planner
    parameters cannot drift from the corrected r2 gate.  Only the route mode differs; the braking
    receipt variables injected by the gate launcher are deliberately not touched here.
    """
    values = R2.frozen_environment(str(contract["route_mode"]), float(contract["speed_mps"]))
    values["NAVRL_TRAINING_SOURCE_MANIFEST"] = str(contract["training_source_manifest"])
    values["NAVRL_TRAINING_SOURCE_MANIFEST_SHA256"] = str(
        contract["training_source_manifest_sha256"]
    )
    return values


def _v3_counters(manager) -> Dict[str, int]:
    counters = {key: int(value) for key, value in manager.v3_gate_diagnostics().items()}
    counters["_local_step_invalidations"] = int(manager.local_step_invalidations.item())
    return counters


def run_cell(task, torch, contract: Mapping[str, object]) -> Dict[str, object]:
    route = str(contract["route_mode"])
    speed = float(contract["speed_mps"])
    bars = int(contract["bars"])
    envs = int(contract["envs"])
    steps = int(contract["steps"])
    warmup_steps = int(contract["warmup_steps"])
    routed = route == GATE.ROUTE_MODE
    manager = task._target_route_manager
    v3_before = _v3_counters(manager) if routed else None

    # Same cell entry as corrected r2: re-seed, set the density, then a full reset whose planning
    # cost belongs to this cell.
    task.seed(int(contract["seed"]))
    R2._sync_cuda(torch)
    initial_started = time.perf_counter()
    task._set_active_bars(bars)
    task.reset()
    R2._sync_cuda(torch)
    initial_reset_s = time.perf_counter() - initial_started
    require(int(task.n_bars_active) == bars, "active bar count differs from density cell")
    active_centers = task.obs_dict["obstacle_position"][
        :, task._bar_offset: task._bar_offset + bars, :2
    ]
    active_half = task.obs_dict["asset_collision_half_extents"][
        :, task._bar_offset: task._bar_offset + bars, :2
    ]
    require(bool(torch.isfinite(active_centers).all()) and bool(torch.isfinite(active_half).all()),
            "active bar AABB tensor is nonfinite")
    require(bool((active_half > 0).all()), "active bar AABB has nonpositive extent")
    layout_sha = R2.initial_layout_sha256(task, bars)
    robot_pose_sha = R2.tensor_digest_sha256(
        task.obs_dict["robot_position"], task.obs_dict["robot_orientation"]
    )
    target_pose_sha = R2.tensor_digest_sha256(task.target_position, task.target_orientation)

    ctrl = task._target_controller
    zero_policy_action = torch.zeros((envs, 4), device=task.device)
    speed_sum = err_sq_sum = 0.0
    tracking_samples = safety_samples = contact_samples = invalid_samples = 0
    off_infeasible_samples = 0
    saturation_substeps = torch.zeros((), dtype=torch.long, device=task.device)
    controller_substeps = torch.zeros((), dtype=torch.long, device=task.device)
    counter_decrease = torch.zeros((), dtype=torch.bool, device=task.device)
    max_tilt_seen_rad = torch.zeros((), dtype=task.target_position.dtype, device=task.device)
    reset_batches, reset_envs, reset_total_s = 1, envs, initial_reset_s
    cell_clock_start = int(task.num_task_steps)
    R2._sync_cuda(torch)
    rollout_started = time.perf_counter()
    for step in range(steps):
        interval_start_step, pursuer_command = R2.prepare_neutral_pursuer_interval(
            task, zero_policy_action
        )
        saturation_before = ctrl.saturation_substeps.clone()
        substeps_before = ctrl.substeps.clone()
        task.sim_env.step(actions=pursuer_command)
        saturation_delta = ctrl.saturation_substeps - saturation_before
        substep_delta = ctrl.substeps - substeps_before
        counter_decrease |= (saturation_delta < 0).any() | (substep_delta < 0).any()
        saturation_substeps += saturation_delta.clamp(min=0).sum()
        controller_substeps += substep_delta.clamp(min=0).sum()
        max_tilt_seen_rad = torch.maximum(max_tilt_seen_rad, ctrl.max_tilt_seen_rad.max())
        if step >= warmup_steps:
            actual = task.target_vel_w[:, :2]
            desired = ctrl.velocity_command[:, :2]
            speed_sum += float(actual.norm(dim=1).sum().item())
            err_sq_sum += float(((actual - desired) ** 2).sum(dim=1).sum().item())
            tracking_samples += envs
        safety_samples += envs
        contact = ctrl.contact_seen.clone()
        contact_samples += int(contact.sum().item())
        if route == "off":
            off_infeasible_samples += int((~task._tm_last_step_feasible).sum().item())
        bounds_min = task.obs_dict["env_bounds_min"]
        bounds_max = task.obs_dict["env_bounds_max"]
        support = task._physical_target_support_xyz()
        invalid = (
            (task.target_position[:, :2] - support[:, :2] < bounds_min[:, :2])
            | (task.target_position[:, :2] + support[:, :2] > bounds_max[:, :2])
        ).any(dim=1)
        invalid |= (
            (task.target_position[:, 2] - support[:, 2] < bounds_min[:, 2])
            | (task.target_position[:, 2] + support[:, 2] > bounds_max[:, 2])
            | ~torch.isfinite(task.target_position).all(dim=1)
        )
        invalid_samples += int((invalid & ~contact).sum().item())
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            timing = R2._timed_reset(task, torch, failed_ids)
            reset_batches += 1
            reset_envs += timing["envs"]
            reset_total_s += timing["wall_s"]
        R2.finish_low_level_evaluation_interval(task, interval_start_step)
    R2._sync_cuda(torch)
    rollout_wall_s = time.perf_counter() - rollout_started
    require(not bool(counter_decrease.item()), "controller counters decreased before reset")
    require(tracking_samples == envs * (steps - warmup_steps), "tracking sample count drift")
    require(safety_samples == envs * steps, "safety sample count drift")

    v3_diagnostics = None
    routed_local_fraction = None
    if routed:
        v3_after = _v3_counters(manager)
        delta = {key: v3_after[key] - v3_before[key] for key in v3_after}
        require(all(value >= 0 for value in delta.values()), "v3 counter decreased inside cell")
        routed_local_fraction = delta.pop("_local_step_invalidations") / max(1, safety_samples)
        v3_diagnostics = delta

    mean_speed = speed_sum / max(1, tracking_samples)
    row = {
        "record_id": str(contract["record_id"]),
        "seed": int(contract["seed"]), "route_mode": route, "speed_mps": speed, "bars": bars,
        "envs": envs, "steps": steps, "warmup_steps": warmup_steps,
        "identity": dict(contract["identity"]),
        "initial_layout_sha256": layout_sha,
        "initial_robot_pose_sha256": robot_pose_sha,
        "initial_target_pose_sha256": target_pose_sha,
        "mean_speed_mps": mean_speed,
        "mean_speed_ratio": mean_speed / speed,
        "tracking_rmse_mps": math.sqrt(err_sq_sum / max(1, tracking_samples)),
        "contact_step_fraction": contact_samples / max(1, safety_samples),
        "off_bounded_local_step_infeasible_fraction": (
            off_infeasible_samples / max(1, safety_samples) if route == "off" else None
        ),
        "routed_local_step_invalidation_fraction": routed_local_fraction,
        "invalid_state_fraction": invalid_samples / max(1, safety_samples),
        "motor_saturation_fraction": float(
            saturation_substeps.float().div(controller_substeps.clamp(min=1)).item()
        ),
        "max_tilt_deg": math.degrees(float(max_tilt_seen_rad.item())),
        "v3_diagnostics": v3_diagnostics,
        "measurement_denominators": {
            "tracking_env_intervals": tracking_samples,
            "safety_env_intervals": safety_samples,
            "controller_substeps": int(controller_substeps.item()),
        },
        "reset_wall": {
            "batches": reset_batches, "reset_envs": reset_envs,
            "initial_batch_s": initial_reset_s, "total_s": reset_total_s,
        },
        "throughput": {
            "rollout_wall_s": rollout_wall_s,
            "env_intervals_per_s": envs * steps / max(rollout_wall_s, 1e-9),
        },
        "task_clock": {
            "start_num_task_steps": cell_clock_start,
            "end_num_task_steps": int(task.num_task_steps),
            "increments": int(task.num_task_steps) - cell_clock_start,
        },
    }
    row["physical_gates"] = GATE.physical_gate_metrics(row)
    row["physical_pass"] = all(row["physical_gates"].values())
    return row


def atomic_write(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main() -> int:
    contract = read_cell_contract(os.environ)
    os.environ.update(frozen_cell_environment(contract))
    os.chdir(ROOT)
    sys.path[:] = [str(ROOT)] + [
        value for value in sys.path if Path(value or ".").resolve() != ROOT
    ]
    sys.argv[:] = [sys.argv[0]]

    from aerial_gym.registry.task_registry import task_registry  # Isaac Gym before torch
    import aerial_gym
    import torch

    origin = Path(aerial_gym.__file__).resolve()
    require(origin == (ROOT / "aerial_gym/__init__.py").resolve(),
            f"aerial_gym import origin drift: {origin}")
    identity = contract["identity"]
    require(GATE.sha256_file(origin) == identity["import_origin_sha256"],
            "imported aerial_gym bytes differ from gate identity")

    task = task_registry.make_task(
        "navrl_task", seed=int(contract["seed"]), num_envs=int(contract["envs"]),
        headless=True, use_warp=True,
    )
    route = str(contract["route_mode"])
    require(task._target_dynamics == "physical", "task did not instantiate a physical target")
    require(task._target_route_mode == route, "task route mode differs from cell arm")
    require(str(task.tm.pattern) == "waypoint", "task pattern is not waypoint")
    require(abs(float(task.tm.speed_fixed) - float(contract["speed_mps"])) < 1e-9,
            "task fixed speed drift")
    if route == GATE.ROUTE_MODE:
        manager = task._target_route_manager
        require(manager is not None and bool(manager.braking_v3_enabled),
                "route manager is not in braking-v3 mode")
        require(not bool(manager.recovery_enabled), "braking-v3 cell must not enable recovery")
    else:
        require(task._target_route_manager is None, "off arm must not construct a route manager")

    row = run_cell(task, torch, contract)
    # Fail closed before writing: the exact validator the gate will run on this payload.
    GATE.validate_cell(row, str(contract["stage"]), identity)
    atomic_write(Path(str(contract["output"])), row)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CellContractError, R2.IntegrityError, GATE.IntegrityError, OSError, ValueError,
            KeyError, TypeError) as exc:
        print(f"CELL_VOID: {exc}", file=sys.stderr)
        raise SystemExit(3)
