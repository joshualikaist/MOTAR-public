#!/usr/bin/env python3
"""Evaluation-only recovery-v2 NO_CONNECTOR geometry forensics.

CPU contract is frozen. ``--run`` launches one fresh Isaac child per 70-bar speed.
The observer wraps existing recovery calls and never changes target commands, planner
decisions, the 32-cell evaluator, gain, ``0.45 m``, or env count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 827
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20
DENSITIES = (70,)
SPEEDS = (0.6, 0.9, 1.2, 1.25)
ROUTE_MODE = "global_astar_recovery_v2"
CONTRACT_VARIANT = "baseline_1p25"
GRID_RESOLUTION_M = 0.25
TRACKING_MARGIN_M = 0.45
ANCHOR_RADIUS_CELLS = 3
SOFT_HYSTERESIS_M = 0.25
HARD_EPSILON_M = 1e-4
REACHABLE_TUBE_MARGIN_M = 0.0123
RECOVERY_HARD_EPSILON_M = HARD_EPSILON_M + REACHABLE_TUBE_MARGIN_M
RUNTIME_WALL_MARGIN_M = 0.50
ROUTE_BOUNDARY_MARGIN_M = 1.25
VEL_KP = 2.5
GATE_SOURCE_COMMIT = "2b151d9a4c4fe078ecc027152e5642fa857a2e2f"
GATE_SUMMARY = ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json"
GATE_RECEIPT = ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/receipt.json"
GATE_SUMMARY_SHA256 = "a85e95764061b7b20cacaa622efc44e2d7e31e054e398f57cfdf48ec98e6c04f"
GATE_RECEIPT_SHA256 = "707636fcbcfe0c855267b39e307af7ac133a0feabbf25d2e7feba726465f1f96"
BRAKING_RECEIPT = ROOT / "results/navrl_physical_target_braking_lower1p25_headingrest_seed827/receipt.json"
BRAKING_RECEIPT_SHA256 = "4e87eb9ddf5dd9cea1fc0354d272a5d18ec6a05427e0f41e672749a57df9047a"
OUTPUT_ROOT = ROOT / "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827"
PREREGISTRATION = "docs/preregistration_physical_target_recovery_v2_no_connector_forensics_2026-08-26.md"
SCHEMA = "navrl_physical_target_recovery_v2_no_connector_forensics_v1"
PRIMARY_PACKED_CLASSES = (
    "brake_no_anchor_likely",
    "same_interval_brake_no_anchor_likely",
)
WILSON_N_MIN = 20
PLANNER_PATH = ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"
PACKED_DIAG_PATH = ROOT / "tools/diagnose_navrl_physical_target_recovery_v2_packed.py"
GATE_DIR = ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827"
GATE_SOURCE_MANIFEST = GATE_DIR / "source_manifest.json"
TRAINING_SOURCE_MANIFEST = GATE_DIR / "inputs/training_source/source_manifest.json"
TRAINING_SOURCE_MANIFEST_SHA256 = "98343363aa938c5e7959b8f5c95662383bb5b5016dd7b57071b07ae52fe374f6"

FROZEN_CHILD_ENV = {
    "AERIAL_GYM_SIM_NAME": "base_sim",
    "NAVRL_ROBOT": "navrl_ref5in_quad",
    "NAVRL_TARGET_DYNAMICS": "physical",
    "NAVRL_TARGET_ROUTE_MODE": ROUTE_MODE,
    "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT": CONTRACT_VARIANT,
    "NAVRL_TARGET_PATTERN": "waypoint",
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
    "NAVRL_NUM_BARS": "70",
    "NAVRL_MAX_BARS": "300",
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
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PLANNER = None


def _load_planner():
    global _PLANNER
    if _PLANNER is not None:
        return _PLANNER
    spec = importlib.util.spec_from_file_location(
        "navrl_target_route_planner_no_connector_forensics", PLANNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _PLANNER = module
    return module


def probe_contract() -> dict[str, Any]:
    """Constants frozen before GPU. Does not read local result files."""
    return {
        "schema": SCHEMA,
        "seed": SEED,
        "envs": ENVS,
        "steps": STEPS,
        "warmup_steps": WARMUP_STEPS,
        "route_mode": ROUTE_MODE,
        "contract_variant": CONTRACT_VARIANT,
        "target_dynamics": "physical",
        "pattern": "waypoint",
        "densities": list(DENSITIES),
        "speeds_mps": list(SPEEDS),
        "grid_resolution_m": GRID_RESOLUTION_M,
        "tracking_margin_m": TRACKING_MARGIN_M,
        "anchor_radius_cells": ANCHOR_RADIUS_CELLS,
        "soft_hysteresis_m": SOFT_HYSTERESIS_M,
        "hard_epsilon_m": HARD_EPSILON_M,
        "reachable_tube_margin_m": REACHABLE_TUBE_MARGIN_M,
        "recovery_hard_epsilon_m": RECOVERY_HARD_EPSILON_M,
        "runtime_wall_margin_m": RUNTIME_WALL_MARGIN_M,
        "route_boundary_margin_m": ROUTE_BOUNDARY_MARGIN_M,
        "boundary_soft_minus_hard_m": ROUTE_BOUNDARY_MARGIN_M - RUNTIME_WALL_MARGIN_M,
        "vel_kp": VEL_KP,
        "gate_source_commit": GATE_SOURCE_COMMIT,
        "gate_summary_sha256": GATE_SUMMARY_SHA256,
        "gate_receipt_sha256": GATE_RECEIPT_SHA256,
        "braking_receipt_sha256": BRAKING_RECEIPT_SHA256,
        "primary_packed_classes": list(PRIMARY_PACKED_CLASSES),
        "wilson_n_min": WILSON_N_MIN,
        "preregistration": PREREGISTRATION,
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "gate_artifacts_read_only": True,
        "original_evaluator_unchanged": True,
    }


def frozen_contract() -> dict[str, Any]:
    """Fail-closed provenance check used by --check-contract and the future GPU child."""
    if not GATE_SUMMARY.is_file() or sha256(GATE_SUMMARY) != GATE_SUMMARY_SHA256:
        raise RuntimeError("lower-1.25 gate summary provenance mismatch; refusing diagnostic")
    if not GATE_RECEIPT.is_file() or sha256(GATE_RECEIPT) != GATE_RECEIPT_SHA256:
        raise RuntimeError("lower-1.25 gate receipt provenance mismatch; refusing diagnostic")
    if not BRAKING_RECEIPT.is_file() or sha256(BRAKING_RECEIPT) != BRAKING_RECEIPT_SHA256:
        raise RuntimeError("heading-rest braking receipt provenance mismatch; refusing diagnostic")
    contract = probe_contract()
    contract["gate_summary_path"] = str(GATE_SUMMARY)
    contract["gate_receipt_path"] = str(GATE_RECEIPT)
    contract["braking_receipt_path"] = str(BRAKING_RECEIPT)
    return contract


def wilson_interval(success: int, total: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    if total < 1:
        return None
    p = float(success) / float(total)
    denominator = 1.0 + z * z / float(total)
    center = p + z * z / (2.0 * float(total))
    spread = z * math.sqrt(p * (1.0 - p) / float(total) + z * z / (4.0 * float(total) * float(total)))
    return float((center - spread) / denominator), float((center + spread) / denominator)


def descriptive_verdict(anchor_present: int, primary_n: int) -> dict[str, Any]:
    """Frozen label. Cannot pass the 32-cell mechanism gate."""
    interval = wilson_interval(int(anchor_present), int(primary_n)) if int(primary_n) else None
    absent_interval = (
        wilson_interval(int(primary_n) - int(anchor_present), int(primary_n))
        if int(primary_n) else None
    )
    if int(primary_n) < WILSON_N_MIN or interval is None or absent_interval is None:
        label = "INCONCLUSIVE"
    elif interval[0] > 0.5:
        label = "ANCHOR_PRESENT_LATCH"
    elif absent_interval[0] > 0.5:
        label = "ANCHOR_ABSENT_AT_LATCH"
    else:
        label = "INCONCLUSIVE"
    return {
        "label": label,
        "primary_n": int(primary_n),
        "anchor_present": int(anchor_present),
        "wilson_present": None if interval is None else {"lower": interval[0], "upper": interval[1]},
        "wilson_absent": (
            None if absent_interval is None
            else {"lower": absent_interval[0], "upper": absent_interval[1]}
        ),
        "passes_32_cell_mechanism": False,
        "authorizes_retune_or_ppo": False,
    }


def recovery_v2_anchor_kwargs() -> dict[str, Any]:
    return {
        "resolution_m": GRID_RESOLUTION_M,
        "radius_cells": ANCHOR_RADIUS_CELLS,
        "tracking_margin_m": TRACKING_MARGIN_M,
        "soft_hysteresis_m": SOFT_HYSTERESIS_M,
        "hard_epsilon_m": RECOVERY_HARD_EPSILON_M,
    }


def v1_forensic_anchor_kwargs() -> dict[str, Any]:
    """The 2026-08-25 search. Using these on recovery-v2 latches is a contract failure."""
    return {
        "resolution_m": GRID_RESOLUTION_M,
        "radius_cells": ANCHOR_RADIUS_CELLS,
        "tracking_margin_m": TRACKING_MARGIN_M,
        "soft_hysteresis_m": 0.0,
        "hard_epsilon_m": HARD_EPSILON_M,
    }


def recovery_anchor_query(
    point,
    bars,
    bar_half,
    bounds_lo,
    bounds_hi,
    support,
    *,
    variant: str = "recovery_v2",
) -> dict[str, Any]:
    planner = _load_planner()
    kwargs = recovery_v2_anchor_kwargs() if variant == "recovery_v2" else v1_forensic_anchor_kwargs()
    return planner.nearest_soft_free_anchor(
        point, bars, bar_half, bounds_lo, bounds_hi, support,
        RUNTIME_WALL_MARGIN_M, ROUTE_BOUNDARY_MARGIN_M, **kwargs,
    )


_PACKED_DIAG = None


def _load_packed_diag():
    global _PACKED_DIAG
    if _PACKED_DIAG is not None:
        return _PACKED_DIAG
    spec = importlib.util.spec_from_file_location(
        "navrl_recovery_v2_packed_forensics_shared", PACKED_DIAG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _PACKED_DIAG = module
    return module


def classify_no_connector_entry(state_before, status_after, soft_margin_before_m) -> str:
    return _load_packed_diag().classify_no_connector_entry(
        state_before, status_after, soft_margin_before_m
    )


def authorization_token(partial_directory: Path) -> str:
    return sha256(Path(__file__).resolve()) + ":" + partial_directory.name


def runtime_source_manifest() -> tuple[list[dict[str, Any]], str]:
    paths = [
        path for base in (ROOT / "aerial_gym", ROOT / "resources")
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.append(Path(__file__).resolve())
    entries = [
        {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path),
         "size": path.stat().st_size}
        for path in sorted(set(paths))
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return entries, hashlib.sha256(encoded).hexdigest()


def require_gate_runtime_bytes() -> None:
    if not GATE_SOURCE_MANIFEST.is_file():
        raise RuntimeError("32-cell gate source manifest is missing; refusing diagnostic")
    manifest = json.loads(GATE_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("runtime_files") or []
    for entry in entries:
        relative = str(entry.get("path", ""))
        if not (relative.startswith("aerial_gym/") or relative.startswith("resources/")):
            continue
        path = ROOT / relative
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise RuntimeError("gate CORE_PATHS drift: %s" % relative)


def _finite_xy(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(2)
    if not np.isfinite(result).all():
        raise ValueError("non-finite XY")
    return result


def signed_aabb_margin(point, bars, half, lo, hi) -> float:
    point = _finite_xy(point)
    lo, hi = _finite_xy(lo), _finite_xy(hi)
    wall = float(np.minimum(point - lo, hi - point).min())
    bars = np.asarray(bars, dtype=np.float64).reshape((-1, 2))
    half = np.asarray(half, dtype=np.float64).reshape((-1, 2))
    if bars.shape[0] == 0:
        return wall
    delta = np.abs(point - bars) - half
    inside = (delta <= 0.0).all(axis=1)
    outside = np.linalg.norm(np.clip(delta, 0.0, None), axis=1)
    penetration = delta.max(axis=1)
    per_bar = np.where(inside, penetration, outside)
    return float(min(wall, float(per_bar.min())))


def latch_clearances(point, bars, half, lo, hi, support) -> dict[str, Any]:
    try:
        point = _finite_xy(point)
        support = _finite_xy(support)
        bars = np.asarray(bars, dtype=np.float64).reshape((-1, 2))
        half = np.asarray(half, dtype=np.float64).reshape((-1, 2))
        lo, hi = _finite_xy(lo), _finite_xy(hi)
        if bars.shape != half.shape or np.any(half < 0.0) or np.any(support < 0.0):
            raise ValueError("invalid geometry")
        hard = signed_aabb_margin(
            point, bars, half + support, lo + RUNTIME_WALL_MARGIN_M + support,
            hi - RUNTIME_WALL_MARGIN_M - support,
        )
        soft = signed_aabb_margin(
            point, bars, half + support + TRACKING_MARGIN_M,
            lo + ROUTE_BOUNDARY_MARGIN_M + support,
            hi - ROUTE_BOUNDARY_MARGIN_M - support,
        )
        return {
            "hard_clearance_m": hard, "soft_clearance_m": soft,
            "hard_free": bool(hard > 0.0), "soft_free": bool(soft > 0.0),
        }
    except (TypeError, ValueError, FloatingPointError):
        return {"hard_clearance_m": None, "soft_clearance_m": None,
                "hard_free": None, "soft_free": None}


def replica_present(replica: Mapping[str, Any]) -> bool:
    return bool(replica.get("exists")) and bool(replica.get("hard_connector_safe"))


def runtime_replica_agree(runtime_ok: Optional[bool], replica: Mapping[str, Any]) -> bool:
    if runtime_ok is None:
        return False
    return bool(runtime_ok) == replica_present(replica)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def latch_event(
    *,
    env: int,
    step: int,
    state_before: int,
    status_after: int,
    realized_speed_mps: Optional[float],
    clearances: Mapping[str, Any],
    runtime_anchor_ok: Optional[bool],
    runtime_brake_ok: Optional[bool],
    resume_replan_status: Optional[str],
    replica: Mapping[str, Any],
) -> dict[str, Any]:
    soft = clearances.get("soft_clearance_m")
    packed = classify_no_connector_entry(
        state_before, status_after, float("nan") if soft is None else soft
    )
    agree = runtime_replica_agree(runtime_anchor_ok, replica)
    return json_safe({
        "event": "no_connector",
        "env": int(env),
        "step": int(step),
        "state_before": int(state_before),
        "status_after": int(status_after),
        "realized_speed_mps": realized_speed_mps,
        "packed_class": packed,
        "hard_clearance_m": clearances.get("hard_clearance_m"),
        "soft_clearance_m": clearances.get("soft_clearance_m"),
        "hard_free": clearances.get("hard_free"),
        "soft_free": clearances.get("soft_free"),
        "runtime_anchor_ok": runtime_anchor_ok,
        "runtime_brake_ok": runtime_brake_ok,
        "resume_replan_status": resume_replan_status,
        "nearest_soft_free_anchor": replica,
        "runtime_replica_agree": agree,
        "primary": packed in PRIMARY_PACKED_CLASSES,
    })


def analyze_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    primary = []
    resume_status: Counter[str] = Counter()
    identity_void = False
    for event in events:
        if event.get("event") != "no_connector":
            continue
        packed = str(event.get("packed_class"))
        class_counts[packed] += 1
        if packed in PRIMARY_PACKED_CLASSES:
            primary.append(event)
            replica = event.get("nearest_soft_free_anchor") or {}
            if event.get("runtime_replica_agree") is not True:
                identity_void = True
            if replica.get("exists") not in (True, False):
                identity_void = True
            if event.get("runtime_anchor_ok") not in (True, False):
                identity_void = True
        if packed == "connect_failed_resume_likely":
            resume_status[str(event.get("resume_replan_status") or "missing")] += 1
    present = sum(1 for event in primary if replica_present(event.get("nearest_soft_free_anchor") or {}))
    hard_free_soft_unsafe = sum(
        1 for event in primary
        if event.get("hard_free") is True and event.get("soft_free") is False
    )
    verdict = descriptive_verdict(present, len(primary))
    if identity_void:
        verdict = dict(verdict)
        verdict["label"] = "VOID_OBSERVER_IDENTITY"
        verdict["identity_void"] = True
    else:
        verdict["identity_void"] = False
    return {
        "class_counts": dict(class_counts),
        "primary_n": len(primary),
        "anchor_present": present,
        "hard_free_soft_unsafe": hard_free_soft_unsafe,
        "resume_replan_status_counts": dict(resume_status),
        "identity_void": identity_void,
        "decision_rule": verdict,
    }


class NoConnectorRecorder:
    def __init__(self, manager, geometry_provider, step_provider, num_envs: int):
        self.manager = manager
        self.geometry_provider = geometry_provider
        self.step_provider = step_provider
        self.num_envs = int(num_envs)
        self.events: list[dict[str, Any]] = []
        self.observer_wall_s = 0.0
        self.last_anchor_ok: list[Optional[bool]] = [None] * self.num_envs
        self.last_brake_ok: list[Optional[bool]] = [None] * self.num_envs
        self.last_replan_status: list[Optional[str]] = [None] * self.num_envs
        self._status_reverse = {int(v): k for k, v in manager.STATUS_CODES.items()}

    def begin_interval(self) -> None:
        self.last_anchor_ok = [None] * self.num_envs
        self.last_brake_ok = [None] * self.num_envs
        self.last_replan_status = [None] * self.num_envs

    @staticmethod
    def _array(value):
        if value is None:
            return None
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def _geometry(self):
        source = self.geometry_provider()
        return {
            key: self._array(value) if key not in ("hard_boundary_margin", "soft_boundary_margin")
            else value
            for key, value in source.items()
        }

    def record_mask(self, mask, state_before) -> None:
        started = time.perf_counter()
        flags = np.asarray(self._array(mask), dtype=bool).reshape(-1)
        if flags.size != self.num_envs:
            raise RuntimeError("recovery mask rank/length drift")
        if not bool(flags.any()):
            self.observer_wall_s += time.perf_counter() - started
            return
        g = self._geometry()
        states = np.asarray(state_before, dtype=np.int16).reshape(-1)
        statuses = np.asarray(self._array(self.manager.status_code)).reshape(-1)
        realized = g.get("realized_velocity")
        for env in np.flatnonzero(flags):
            env = int(env)
            point = g["position"][env, :2]
            bars, half = g["bars"][env], g["bar_half"][env]
            lo, hi = g["bounds_lo"][env], g["bounds_hi"][env]
            support = g["support"][env]
            clearances = latch_clearances(point, bars, half, lo, hi, support)
            replica = recovery_anchor_query(point, bars, half, lo, hi, support)
            speed = None if realized is None else float(np.linalg.norm(realized[env, :2]))
            self.events.append(latch_event(
                env=env,
                step=int(self.step_provider()),
                state_before=int(states[env]),
                status_after=int(statuses[env]),
                realized_speed_mps=speed,
                clearances=clearances,
                runtime_anchor_ok=self.last_anchor_ok[env],
                runtime_brake_ok=self.last_brake_ok[env],
                resume_replan_status=self.last_replan_status[env],
                replica=replica,
            ))
        self.observer_wall_s += time.perf_counter() - started


def attach_observer(task) -> NoConnectorRecorder:
    manager = task._target_route_manager
    if manager is None:
        raise RuntimeError("route manager is disabled; diagnostic requires global_astar_recovery_v2")
    if str(task._target_route_mode) != ROUTE_MODE:
        raise RuntimeError("diagnostic requires %s" % ROUTE_MODE)

    def geometry_provider():
        return {
            "position": task.target_position,
            "bars": task.obs_dict["obstacle_position"][:, task._bar_offset:task._bar_offset + task.n_bars_active, :2],
            "bar_half": task.obs_dict["asset_collision_half_extents"][:, task._bar_offset:task._bar_offset + task.n_bars_active, :2],
            "bounds_lo": task.obs_dict["env_bounds_min"][:, :2],
            "bounds_hi": task.obs_dict["env_bounds_max"][:, :2],
            "support": task._target_route_support_xy,
            "realized_velocity": task.target_vel_w,
        }

    recorder = NoConnectorRecorder(
        manager, geometry_provider, lambda: int(task.num_task_steps), int(task.num_envs)
    )
    original_mark = manager.mark_no_connector
    original_soft_free = manager.mark_local_infeasible_soft_free
    original_anchor = manager.recovery_anchor_idx
    original_brake = manager.brake_connector_idx
    original_plan = manager.plan_idx

    def mark_no_connector(mask, hard_breach: bool = False, timeout_kind=None):
        before = recorder._array(manager.recovery_state).copy()
        result = original_mark(mask, hard_breach=hard_breach, timeout_kind=timeout_kind)
        recorder.record_mask(mask, before)
        return result

    def mark_local_infeasible_soft_free(mask):
        before = recorder._array(manager.recovery_state).copy()
        result = original_soft_free(mask)
        recorder.record_mask(mask, before)
        return result

    def recovery_anchor_idx(env_ids, *args, **kwargs):
        result = original_anchor(env_ids, *args, **kwargs)
        ids = np.atleast_1d(recorder._array(env_ids)).astype(int)
        flags = recorder._array(result)
        for env in ids.tolist():
            recorder.last_anchor_ok[int(env)] = bool(flags[int(env)])
        return result

    def brake_connector_idx(env_ids, *args, **kwargs):
        result = original_brake(env_ids, *args, **kwargs)
        ids = np.atleast_1d(recorder._array(env_ids)).astype(int)
        flags = recorder._array(result)
        for env in ids.tolist():
            recorder.last_brake_ok[int(env)] = bool(flags[int(env)])
        return result

    def plan_idx(env_ids, *args, **kwargs):
        result = original_plan(env_ids, *args, **kwargs)
        if kwargs.get("is_replan"):
            ids = np.atleast_1d(recorder._array(env_ids)).astype(int)
            codes = recorder._array(manager.status_code)
            for env in ids.tolist():
                recorder.last_replan_status[int(env)] = recorder._status_reverse.get(
                    int(codes[int(env)]), "unknown"
                )
        return result

    manager.mark_no_connector = mark_no_connector
    manager.mark_local_infeasible_soft_free = mark_local_infeasible_soft_free
    manager.recovery_anchor_idx = recovery_anchor_idx
    manager.brake_connector_idx = brake_connector_idx
    manager.plan_idx = plan_idx
    return recorder


def _file_provenance(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("runtime provenance file is missing: %s" % path)
    return {"path": str(path), "sha256": sha256(path)}


def _require_float32_close(actual, expected, label: str, tolerance: float = 1e-6) -> None:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    max_error = (
        float(np.max(np.abs(actual_values - expected_values)))
        if actual_values.shape == expected_values.shape and actual_values.size else 0.0
    )
    if (actual_values.shape != expected_values.shape
            or not np.isfinite(actual_values).all()
            or max_error > tolerance):
        raise RuntimeError("instantiated float32 contract drift: %s" % label)


def speed_key(value: float) -> str:
    """Canonical decimal key; must not collapse 1.25 to 1.2."""
    return format(float(value), ".12g")


def braking_child_env(speed: float) -> dict[str, str]:
    if not BRAKING_RECEIPT.is_file() or sha256(BRAKING_RECEIPT) != BRAKING_RECEIPT_SHA256:
        raise RuntimeError("heading-rest braking receipt provenance mismatch; refusing diagnostic")
    payload = json.loads(BRAKING_RECEIPT.read_text(encoding="utf-8"))
    certified = payload["certified_monotone_speed_to_p95_lookup"]
    measured = payload["measured_speed_to_p95_lookup"]
    core = payload["core_integration"]
    key = speed_key(speed)
    if key not in certified or key not in measured:
        raise RuntimeError("braking receipt is missing frozen speed %s" % key)
    return {
        "NAVRL_TARGET_RECOVERY_BRAKE_P05": str(payload["decel_p05_mps2"]),
        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S": str(payload["stop_time_p95_s"]),
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT": str(BRAKING_RECEIPT.resolve()),
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256": BRAKING_RECEIPT_SHA256,
        "NAVRL_TARGET_RECOVERY_PROBE_VALIDATED": "1",
        "NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS": ",".join(speed_key(value) for value in SPEEDS),
        "NAVRL_TARGET_RECOVERY_BRAKE_STOP_DISTANCES_M": ",".join(
            str(certified[speed_key(value)]["p95_stop_distance_m"]) for value in SPEEDS
        ),
        "NAVRL_TARGET_RECOVERY_STOP_DISTANCE_P95_M": str(
            measured[key]["p95_stop_distance_m"]
        ),
        "NAVRL_TARGET_RECOVERY_BRAKE_LATERAL_TUBE_P95_M": str(
            core["certified_lateral_tube_p95_m"]
        ),
        "NAVRL_TRAINING_SOURCE_MANIFEST": str(TRAINING_SOURCE_MANIFEST.resolve()),
        "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256": TRAINING_SOURCE_MANIFEST_SHA256,
        "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT": "1",
        "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE": "1",
    }


def build_child_environment(speed: float) -> dict[str, str]:
    python_bin = Path(sys.executable).resolve().parent
    ninja = python_bin / "ninja"
    if not ninja.is_file() or not os.access(str(ninja), os.X_OK):
        raise RuntimeError("selected Python environment has no executable ninja")
    child = {
        key: value for key, value in os.environ.items()
        if not key.startswith("NAVRL_") and key != "AERIAL_GYM_SIM_NAME"
    }
    old_path = child.get("PATH", "")
    child["PATH"] = os.pathsep.join(
        [str(python_bin)] + [part for part in old_path.split(os.pathsep) if part and part != str(python_bin)]
    )
    child["PYTHONNOUSERSITE"] = "1"
    child["PYTHONPATH"] = str(ROOT)
    child.update(FROZEN_CHILD_ENV)
    child.update(braking_child_env(speed))
    child["NAVRL_TARGET_SPEED"] = str(speed)
    child["NAVRL_TARGET_SPEED_FINAL"] = str(speed)
    child["NAVRL_TARGET_SPEED_MIN"] = str(speed)
    child["NAVRL_TARGET_SPEED_RAMP_EPOCHS"] = "1"
    if any(key.startswith("NAVRL_TARGET_RECOVERY_EVAL") for key in child):
        raise RuntimeError("packed 32-cell telemetry observer is forbidden in this diagnostic")
    if abs(float(speed) - 1.5) < 1e-12 or "1.5" in child.get("NAVRL_TARGET_SPEED", ""):
        raise RuntimeError("canonical 1.5 m/s is out of scope for this diagnostic")
    return child


def _configure(density: int, speed: float) -> None:
    if int(density) != 70 or float(speed) not in SPEEDS:
        raise RuntimeError("forensic cell is outside the frozen 70-bar lower-1.25 grid")
    os.environ.update(build_child_environment(speed))
    os.environ["NAVRL_NUM_BARS"] = "70"
    os.environ["NAVRL_MAX_BARS"] = "300"


def runtime_software_provenance(torch_module, import_origin: Path,
                                task_origin: Path, planner_origin: Path) -> dict[str, Any]:
    python_path = Path(sys.executable).resolve()
    ninja_path = (python_path.parent / "ninja").resolve()
    if not ninja_path.is_file() or not os.access(str(ninja_path), os.X_OK):
        raise RuntimeError("selected Python environment has no executable ninja")
    ninja_version = subprocess.run(
        [str(ninja_path), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    isaac = sys.modules.get("isaacgym")
    isaac_path = Path(getattr(isaac, "__file__", "")).resolve() if isaac is not None else None
    if isaac_path is None or not isaac_path.is_file():
        raise RuntimeError("Isaac Gym import origin is unavailable")
    if not bool(torch_module.cuda.is_available()) or int(torch_module.cuda.device_count()) < 1:
        raise RuntimeError("CUDA/GPU identity is unavailable; refusing forensic cell")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi is unavailable; refusing forensic cell")
    nvidia_query = subprocess.run(
        [nvidia_smi, "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not nvidia_query:
        raise RuntimeError("nvidia-smi returned no GPU identity")
    return {
        "python": {
            "executable": str(python_path), "executable_sha256": sha256(python_path),
            "version": sys.version, "implementation": sys.implementation.name,
        },
        "torch": {
            "version": str(torch_module.__version__),
            "origin": str(Path(torch_module.__file__).resolve()),
            "origin_sha256": sha256(Path(torch_module.__file__).resolve()),
            "compiled_cuda_version": str(torch_module.version.cuda),
        },
        "isaac_gym": _file_provenance(isaac_path),
        "ninja": {"path": str(ninja_path), "sha256": sha256(ninja_path), "version": ninja_version},
        "cuda": {
            "available": True,
            "device_count": int(torch_module.cuda.device_count()),
            "current_device": int(torch_module.cuda.current_device()),
            "gpu_names": [str(torch_module.cuda.get_device_name(0))],
            "nvidia_smi": {
                "path": str(Path(nvidia_smi).resolve()),
                "sha256": sha256(Path(nvidia_smi).resolve()),
                "query": nvidia_query,
            },
        },
        "repo_modules": {
            "aerial_gym": _file_provenance(import_origin),
            "navrl_task": _file_provenance(task_origin),
            "target_route_planner": _file_provenance(planner_origin),
            "target_motion": _file_provenance(ROOT / "aerial_gym/task/navrl_task/target_motion.py"),
            "physical_target_controller": _file_provenance(
                ROOT / "aerial_gym/task/navrl_task/physical_target.py"
            ),
        },
    }


def runtime_contract_attestation(task, density: int, speed: float) -> dict[str, Any]:
    physics = task._runtime_physics_contract()
    arena = task._arena_contract()
    bounds = task.obs_dict["env_bounds_max"][0] - task.obs_dict["env_bounds_min"][0]
    bounds_xyz = [float(value) for value in bounds.detach().cpu().tolist()]
    tm = task.tm
    route = task._target_route_manager
    controller = task._target_controller
    robot = task._robot_provenance
    contract = {
        "density": int(task.n_bars_active),
        "requested_density": int(density),
        "speed_mps": float(speed),
        "route_mode": str(task._target_route_mode),
        "target_dynamics": str(task._target_dynamics),
        "target_pattern": str(tm.pattern),
        "contract_variant": str(tm.recovery_braking_contract_variant),
        "target": {
            "max_accel_mps2": float(tm.max_accel),
            "max_turn_rate_degps": float(tm.max_turn_rate_deg),
            "lookahead_s": float(tm.avoidance_lookahead_s),
            "velocity_kp": float(controller.velocity_kp),
            "tracking_margin_m": float(tm.physical_tracking_margin),
            "boundary_margin_m": float(tm.physical_boundary_margin),
            "box_xyz_m": [float(v) for v in tm.physical_box_xyz],
        },
        "route": {
            "resolution_m": float(route.config.resolution_m),
            "support_xy_m": [float(v) for v in task._target_route_support_xy[0].detach().cpu().tolist()],
            "replan_cooldown_steps": int(route.config.replan_cooldown_steps),
            "goal_tolerance_m": float(route.config.goal_tolerance_m),
            "min_goal_distance_m": float(tm.route_min_goal_distance_m),
            "goal_exclusion_radius_m": float(route.config.goal_exclusion_radius_m),
            "max_expansions": int(route.config.max_expansions),
            "max_waypoints": int(route.config.max_waypoints),
        },
        "recovery": {
            "hysteresis_m": float(route.config.resolution_m),
            "hard_epsilon_m": RECOVERY_HARD_EPSILON_M,
            "reachable_tube_margin_m": REACHABLE_TUBE_MARGIN_M,
        },
        "arena": {"bounds_xyz_m": bounds_xyz, **arena},
        "bar_capacity": int(task.obs_dict["obstacle_position"].shape[1] - task._bar_offset),
        "physics": physics,
        "robot": {key: robot[key] for key in (
            "robot_name", "robot_config_path", "robot_config_sha256",
            "robot_asset_path", "robot_asset_sha256",
        ) if key in robot},
    }
    if contract["route_mode"] != ROUTE_MODE or contract["contract_variant"] != CONTRACT_VARIANT:
        raise RuntimeError("instantiated recovery-v2/lower-1.25 contract drift")
    if contract["target_dynamics"] != "physical" or contract["target_pattern"] != "waypoint":
        raise RuntimeError("instantiated target dynamics/pattern drift")
    if abs(float(contract["target"]["velocity_kp"]) - VEL_KP) > 1e-9:
        raise RuntimeError("instantiated velocity gain drift")
    if abs(float(contract["target"]["tracking_margin_m"]) - TRACKING_MARGIN_M) > 1e-9:
        raise RuntimeError("instantiated tracking margin drift")
    if abs(float(contract["target"]["boundary_margin_m"]) - 0.75) > 1e-9:
        raise RuntimeError("instantiated physical boundary margin drift")
    if int(contract["density"]) != 70 or abs(float(contract["speed_mps"]) - float(speed)) > 1e-12:
        raise RuntimeError("instantiated density/speed drift")
    if abs(float(tm.speed_fixed) - float(speed)) > 1e-9:
        raise RuntimeError("instantiated target speed_fixed drift")
    if abs(float(tm.speed_min) - float(speed)) > 1e-9 or abs(float(tm.speed_final) - float(speed)) > 1e-9:
        raise RuntimeError("instantiated speed min/final drift")
    if bounds_xyz != [40.0, 40.0, 3.0]:
        raise RuntimeError("instantiated arena bounds drift")
    if abs(float(contract["recovery"]["hysteresis_m"]) - SOFT_HYSTERESIS_M) > 1e-9:
        raise RuntimeError("instantiated recovery hysteresis drift")
    if abs(float(contract["recovery"]["hard_epsilon_m"]) - RECOVERY_HARD_EPSILON_M) > 1e-9:
        raise RuntimeError("instantiated recovery hard-epsilon drift")
    exact = {
        "target.max_accel_mps2": (contract["target"]["max_accel_mps2"], 4.0),
        "target.max_turn_rate_degps": (contract["target"]["max_turn_rate_degps"], 150.0),
        "target.lookahead_s": (contract["target"]["lookahead_s"], 1.0),
        "route.resolution_m": (contract["route"]["resolution_m"], 0.25),
        "route.replan_cooldown_steps": (contract["route"]["replan_cooldown_steps"], 10),
        "route.min_goal_distance_m": (contract["route"]["min_goal_distance_m"], 6.0),
        "route.goal_tolerance_m": (contract["route"]["goal_tolerance_m"], 0.05),
        "route.goal_exclusion_radius_m": (contract["route"]["goal_exclusion_radius_m"], 1.0),
        "route.max_expansions": (contract["route"]["max_expansions"], 50000),
        "route.max_waypoints": (contract["route"]["max_waypoints"], 128),
        "physics.physics_dt_s": (contract["physics"]["physics_dt_s"], 0.01),
        "physics.physics_substeps": (contract["physics"]["physics_substeps"], 1),
        "physics.physics_steps_per_rl_step": (contract["physics"]["physics_steps_per_rl_step"], 10),
        "physics.rl_step_dt_s": (contract["physics"]["rl_step_dt_s"], 0.10),
    }
    for name, (actual, expected_value) in exact.items():
        if abs(float(actual) - float(expected_value)) > 1e-9:
            raise RuntimeError("instantiated runtime contract drift: %s=%r" % (name, actual))
    if contract["bar_capacity"] != 300 or contract["target"]["box_xyz_m"] != [0.28, 0.28, 0.12]:
        raise RuntimeError("instantiated bar capacity/target box contract drift")
    if (contract["arena"].get("cfg_bar_pool") != "bars_h3"
            or contract["arena"].get("cfg_placement_mode") != "navrl_band"
            or contract["arena"].get("cfg_placement_gap_m") != 1.6
            or contract["arena"].get("cfg_placement_touch_m") != 0.4
            or contract["arena"].get("cfg_bar_x_min") != 0.0
            or contract["arena"].get("cfg_bar_x_max") != 1.0
            or contract["arena"].get("cfg_arena_xy") != 40.0
            or contract["arena"].get("cfg_arena_z") != 3.0):
        raise RuntimeError("instantiated bar placement contract drift")
    if contract["robot"].get("robot_name") != "navrl_ref5in_quad":
        raise RuntimeError("instantiated robot provenance contract drift")
    _require_float32_close(
        contract["route"]["support_xy_m"],
        [0.2068816086567407, 0.2068816086567407],
        "support_xy_m",
    )
    return contract


def run_cell(density: int, speed: float, output: Path) -> None:
    import isaacgym  # noqa: F401  # must precede torch
    _configure(density, speed)
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry
    import aerial_gym
    import aerial_gym.task.navrl_task.navrl_task as navrl_task_module
    import aerial_gym.task.navrl_task.target_route_planner as route_planner_module
    import torch
    import_origin = Path(aerial_gym.__file__).resolve()
    task_origin = Path(navrl_task_module.__file__).resolve()
    planner_origin = Path(route_planner_module.__file__).resolve()
    if (
        import_origin != ROOT / "aerial_gym/__init__.py"
        or task_origin != ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
        or planner_origin != ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"
    ):
        raise RuntimeError("aerial_gym import escaped or drifted from the worktree")
    require_gate_runtime_bytes()
    _, runtime_manifest_sha = runtime_source_manifest()
    task = task_registry.make_task("navrl_task", seed=SEED, num_envs=ENVS, headless=True, use_warp=True)
    task.seed(SEED)
    task._set_active_bars(int(density))
    if abs(float(task.cur.wall_margin) - RUNTIME_WALL_MARGIN_M) > 1e-9:
        raise RuntimeError("runtime wall margin drift")
    if abs(float(task._target_route_manager.config.boundary_margin_m) - ROUTE_BOUNDARY_MARGIN_M) > 1e-9:
        raise RuntimeError("route boundary margin drift")
    if abs(float(task.tm.physical_boundary_margin) - 0.75) > 1e-9:
        raise RuntimeError("physical boundary margin drift")
    if abs(float(task.tm.physical_tracking_margin) - TRACKING_MARGIN_M) > 1e-9:
        raise RuntimeError("tracking margin drift")
    recorder = attach_observer(task)
    task.reset()
    runtime_contract = runtime_contract_attestation(task, density, speed)
    software_provenance = runtime_software_provenance(
        torch, import_origin, task_origin, planner_origin
    )
    zero_policy_action = torch.zeros((ENVS, 4), device=task.device)
    rollout_started = time.perf_counter()
    for _ in range(STEPS):
        interval_start_step = int(task.num_task_steps)
        recorder.begin_interval()
        task._target_controller.begin_control_interval()
        task._advance_target()
        command = task.transform_action_to_command(zero_policy_action)
        if tuple(command.shape) != tuple(zero_policy_action.shape) or not bool(command.isfinite().all()):
            raise RuntimeError("canonical neutral pursuer command contract drift")
        task.sim_env.step(actions=command)
        contact = task._target_controller.contact_seen.clone()
        bmin, bmax = task.obs_dict["env_bounds_min"], task.obs_dict["env_bounds_max"]
        support = task._physical_target_support_xyz()
        invalid = ((task.target_position[:, :2] - support[:, :2] < bmin[:, :2])
                   | (task.target_position[:, :2] + support[:, :2] > bmax[:, :2])).any(dim=1)
        invalid |= ((task.target_position[:, 2] - support[:, 2] < bmin[:, 2])
                    | (task.target_position[:, 2] + support[:, 2] > bmax[:, 2])
                    | ~torch.isfinite(task.target_position).all(dim=1))
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            task.sim_env.reset_idx(failed_ids)
            task.reset_idx(failed_ids)
        if int(task.num_task_steps) != interval_start_step:
            raise RuntimeError("task clock changed before canonical forensic increment")
        task.num_task_steps += 1
        if int(task.num_task_steps) != interval_start_step + 1:
            raise RuntimeError("task clock did not advance exactly once")
    rollout_wall_s = time.perf_counter() - rollout_started
    analysis = analyze_events(recorder.events)
    if analysis["identity_void"]:
        raise RuntimeError("observer-identity defect on a primary NO_CONNECTOR event; VOID")
    payload = {
        "schema": SCHEMA + "_cell",
        "contract": frozen_contract(),
        "density": int(density),
        "speed_mps": float(speed),
        "runtime_contract": runtime_contract,
        "software_provenance": software_provenance,
        "source": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "tool_sha256": sha256(Path(__file__).resolve()),
            "import_origin": str(import_origin),
            "import_origin_sha256": sha256(import_origin),
            "planner_path": str(planner_origin),
            "planner_sha256": sha256(planner_origin),
            "task_path": str(task_origin),
            "task_sha256": sha256(task_origin),
            "runtime_manifest_sha256": runtime_manifest_sha,
        },
        "observer": {
            "event_count": len(recorder.events),
            "wall_s": recorder.observer_wall_s,
            "rollout_wall_s": rollout_wall_s,
        },
        "analysis": analysis,
        "events": recorder.events,
        "gate_artifacts_read_only": True,
        "original_evaluator_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _build_summary(directory: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    pooled_events: list[dict[str, Any]] = []
    per_cell = []
    for entry in receipt["cells"]:
        cell = json.loads((directory / entry["path"]).read_text(encoding="utf-8"))
        events = list(cell.get("events") or [])
        analysis = analyze_events(events)
        stored = cell.get("analysis")
        if stored != analysis:
            raise RuntimeError("forensic cell analysis drift")
        per_cell.append({
            "density": entry["density"],
            "speed_mps": entry["speed_mps"],
            "path": entry["path"],
            "analysis": analysis,
        })
        pooled_events.extend(events)
    pooled = analyze_events(pooled_events)
    return {
        "schema": SCHEMA + "_summary",
        "contract": frozen_contract(),
        "cells": per_cell,
        "pooled": pooled,
        "decision_rule": pooled["decision_rule"],
        "interpretation": "descriptive_only_no_gate_or_tuning_authority",
        "passes_32_cell_mechanism": False,
    }


def verify_receipt(directory: Path) -> int:
    directory = directory.resolve()
    if directory == GATE_DIR.resolve() or GATE_DIR.name in directory.parts:
        raise RuntimeError("forensic verify refuses the 32-cell gate directory")
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("forensic receipt is missing")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA + "_receipt":
        raise RuntimeError("forensic receipt schema mismatch")
    if payload.get("contract") != frozen_contract():
        raise RuntimeError("forensic receipt frozen contract mismatch")
    if payload.get("tool_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("forensic tool hash mismatch")
    if payload.get("gate_artifacts_read_only") is not True or payload.get("original_evaluator_unchanged") is not True:
        raise RuntimeError("forensic receipt claim-boundary markers missing")
    recorded_head = payload.get("git_head")
    if not isinstance(recorded_head, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", recorded_head):
        raise RuntimeError("forensic receipt has no valid recorded git commit")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", recorded_head + "^{commit}"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if commit_check.returncode != 0:
        raise RuntimeError("forensic recorded git commit is missing")
    if payload.get("worktree_clean") is not True:
        raise RuntimeError("forensic receipt execution was not clean")
    runtime_manifest, runtime_manifest_sha = runtime_source_manifest()
    if payload.get("runtime_manifest_sha256") != runtime_manifest_sha:
        raise RuntimeError("forensic runtime source manifest drift")
    expected = {(70, speed) for speed in SPEEDS}
    observed = set()
    pooled_events: list[dict[str, Any]] = []
    for entry in payload.get("cells", []):
        density, speed = int(entry["density"]), float(entry["speed_mps"])
        if (density, speed) in observed or (density, speed) not in expected:
            raise RuntimeError("forensic receipt cell grid mismatch")
        observed.add((density, speed))
        path = (directory / entry["path"]).resolve()
        if directory not in path.parents or GATE_DIR.name in path.parts:
            raise RuntimeError("forensic cell escapes its output directory")
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise RuntimeError("forensic cell hash mismatch")
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("schema") != SCHEMA + "_cell" or cell.get("contract") != frozen_contract():
            raise RuntimeError("forensic cell contract mismatch")
        if cell.get("source", {}).get("tool_sha256") != payload.get("tool_sha256"):
            raise RuntimeError("forensic cell tool hash mismatch")
        if analyze_events(cell.get("events") or [])["identity_void"]:
            raise RuntimeError("forensic cell has observer-identity VOID")
        pooled_events.extend(cell.get("events") or [])
    if observed != expected:
        raise RuntimeError("forensic receipt is missing a frozen cell")
    if analyze_events(pooled_events)["identity_void"]:
        raise RuntimeError("forensic pooled events have observer-identity VOID")
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        if stored != _build_summary(directory, payload):
            raise RuntimeError("forensic summary semantic/hash mismatch")
    if directory == OUTPUT_ROOT.resolve():
        marker = directory / ".COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError("forensic completion marker is missing")
    print(json.dumps({"verified": True, "cells": len(observed), "receipt": str(receipt_path)}))
    return 0


def summarize(directory: Path, *, internal_partial: bool = False) -> int:
    directory = directory.resolve()
    if (not internal_partial or directory == OUTPUT_ROOT.resolve()
            or directory.parent != OUTPUT_ROOT.parent
            or not directory.name.startswith(OUTPUT_ROOT.name + ".partial-")):
        raise RuntimeError("completed forensic artifacts are immutable; summarize only an authorized partial run")
    verify_receipt(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    summary = _build_summary(directory, receipt)
    if summary["pooled"]["identity_void"]:
        raise RuntimeError("observer-identity defect; refusing to publish a scientific summary")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def preflight() -> int:
    contract = frozen_contract()
    if tuple(contract["densities"]) != DENSITIES or tuple(contract["speeds_mps"]) != SPEEDS:
        raise RuntimeError("frozen cell contract drift")
    print(json.dumps({"schema": SCHEMA + "_preflight", "contract": contract}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-contract", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--_cell-density", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_cell-speed", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--_auth-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(frozen_contract(), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        return preflight()
    if args.verify:
        return verify_receipt(Path(args.output).resolve())
    if args.summarize:
        raise RuntimeError("--summarize is internal-only; completed forensic artifacts are immutable")
    out = Path(args.output).resolve()
    if args._cell_density is not None:
        partial = out.parent
        if (args._auth_token != authorization_token(partial)
                or partial.parent != OUTPUT_ROOT.parent
                or not partial.name.startswith(OUTPUT_ROOT.name + ".partial-")
                or out.exists()
                or GATE_DIR.name in out.parts):
            raise RuntimeError("unauthorized or unsafe forensic child output")
        run_cell(int(args._cell_density), float(args._cell_speed), out)
        return 0
    if not args.run:
        print(json.dumps(probe_contract(), indent=2, sort_keys=True))
        return 0
    if out == GATE_DIR.resolve() or GATE_DIR.name in out.parts:
        raise RuntimeError("forensic run refuses the 32-cell gate directory")
    if out != OUTPUT_ROOT or out.exists():
        raise RuntimeError("forensic run requires a fresh canonical OUTPUT_ROOT")
    if list(OUTPUT_ROOT.parent.glob(OUTPUT_ROOT.name + ".partial-*")):
        raise RuntimeError("forensic partial output exists; refusing rerun or mixed cells")
    require_gate_runtime_bytes()
    frozen_contract()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("forensic run requires a clean committed worktree")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    partial = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".partial-" + str(os.getpid()))
    if partial.exists():
        raise RuntimeError("forensic partial output already exists; refusing rerun")
    partial.mkdir(parents=True)
    token = authorization_token(partial)
    runtime_manifest, runtime_manifest_sha = runtime_source_manifest()
    cells = []
    runtime_contracts = []
    software_provenance = None
    for density in DENSITIES:
        for speed in SPEEDS:
            cell = partial / ("recovery_v2__speed_%s__bars_%s.json" % (
                str(speed).replace(".", "p"), density
            ))
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--_cell-density", str(density), "--_cell-speed", str(speed),
                "--output", str(cell), "--_auth-token", token,
            ]
            completed = subprocess.run(command, cwd=ROOT, env=build_child_environment(speed), check=False)
            if completed.returncode != 0:
                raise RuntimeError("forensic child failed: density=%s speed=%s" % (density, speed))
            cell_payload = json.loads(cell.read_text(encoding="utf-8"))
            cell_software = cell_payload.get("software_provenance")
            if software_provenance is None:
                software_provenance = cell_software
            elif cell_software != software_provenance:
                raise RuntimeError("forensic child software provenance differs across cells")
            runtime_contracts.append({
                "density": density, "speed_mps": speed,
                "contract": cell_payload.get("runtime_contract"),
            })
            cells.append({
                "density": density, "speed_mps": speed,
                "path": str(cell.relative_to(partial)), "sha256": sha256(cell),
            })
    receipt = {
        "schema": SCHEMA + "_receipt",
        "contract": frozen_contract(),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "git_head": git_head,
        "worktree_clean": True,
        "runtime_manifest_sha256": runtime_manifest_sha,
        "runtime_source_manifest": runtime_manifest,
        "software_provenance": software_provenance,
        "runtime_contracts": runtime_contracts,
        "cells": cells,
        "gate_artifacts_read_only": True,
        "original_evaluator_unchanged": True,
    }
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summarize(partial, internal_partial=True)
    verify_receipt(partial)
    marker = partial / ".COMPLETE.json"
    marker.write_text(json.dumps({
        "schema": SCHEMA + "_complete",
        "receipt_sha256": sha256(partial / "receipt.json"),
        "summary_sha256": sha256(partial / "summary.json"),
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, OUTPUT_ROOT)
    verify_receipt(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

