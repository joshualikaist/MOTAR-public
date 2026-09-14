#!/usr/bin/env python3
"""Synchronize the archived machine-readable status snapshot with local NavRL evidence.

The public research page is intentionally a small, hand-reviewed static summary. This tool updates
``docs/status/status.json`` for audits and downstream analysis; it does not rewrite presentation
HTML or create a JavaScript mirror.
"""

from __future__ import annotations

import csv
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
RUNS_ROOT = RL_ROOT / "runs"
STATUS_PATH = ROOT / "docs/status/status.json"
CORRECTED_CURVE_PATH = ROOT / "results/corrected_chirality_density_curve.csv"
RECOVERY_ATTESTATION_VERIFIER_PATH = ROOT / "tools/navrl_v2_recovery_attestation.py"
LIMIT_AUDIT_PATH = ROOT / "results/navrl_v2_ep24000_limit_audit.json"
CAUSAL_1TO3_PATH = ROOT / "results/navrl_v2_ep24000_causal_1to3/summary.json"
# Obstacle-placement areas, the density denominators. Named because the v1 figure is also the
# divisor baked into the archived density x speed map's labels.
V1_PLACEMENT_AREA_M2 = 478.0
V2_PLACEMENT_AREA_M2 = 1600.0
DETECTOR_ROBUSTNESS_PATH = ROOT / "results/navrl_v2_detector_robustness/summary.json"
LATENCY_EGO_MOTION_PATH = ROOT / "results/navrl_v2_latency_ego_motion/summary.json"
LATENCY_BUDGET_PATH = ROOT / "results/navrl_v2_latency_budget/summary.json"
FIXED_SPEED_PATH = ROOT / "results/navrl_v2_ep24000_fixed_speed/summary.json"
FORGETTING_PATH = ROOT / "results/navrl_v2_ep19100_vs_ep24000_forgetting/summary.json"
SPEED_GOVERNOR_SCREEN_PATH = ROOT / "results/navrl_v2_ep24000_speed_governor_screen/summary.json"
RISKCAP_SCREEN_PATH = ROOT / "results/navrl_v2_ep24000_riskcap_seed44_screen/summary.json"
RISKCAP_POST_PATH = ROOT / "results/navrl_v2_riskcap_postadapt/summary.json"
TTC_1650_PATH = ROOT / "results/v2_ttc_ab_1650ti.csv"
MAIN_TTC_RESULT_ROOT = ROOT / "results"
SIM2REAL_PREFLIGHT_PATH = ROOT / "results/navrl_sim2real_software_preflight_2026-08-24/summary.json"
PHYSICAL_SPEED_ENVELOPE_PATH = ROOT / "results/navrl_physical_target_speed_envelope_post_wall_brake_seed509/summary.json"
ROUTED_PHYSICAL_GATE_PATH = ROOT / "results/navrl_physical_target_routed_gate_seed827_attempt2/summary.json"
ROUTE_RECOVERY_FORENSICS_PATH = (
    ROOT / "results/navrl_physical_target_route_recovery_forensics_seed827/summary.json"
)
RECOVERY_V2_LOWER1P25_GATE_PATH = (
    ROOT
    / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json"
)
RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH = (
    ROOT
    / "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json"
)
RECOVERY_V2_PACKED_DIAGNOSTIC_PATH = (
    ROOT / "tools/diagnose_navrl_physical_target_recovery_v2_packed.py"
)
RECOVERY_V2_GATE_VERIFIER_PATH = (
    ROOT / "tools/verify_navrl_physical_target_recovery_v2_gate.py"
)
RECOVERY_V2_NO_CONNECTOR_VERIFIER_PATH = (
    ROOT / "tools/diagnose_navrl_physical_target_recovery_v2_no_connector.py"
)
MODE_PROBE_PATH = ROOT / "results/navrl_ref5in_symmetric_corridor_mode_probe_seed431/summary.json"
REPOSITORY_ID = "joshualikaist/MOTAR"

_RECOVERY_V2_ROUTE_MODES = ("off", "global_astar_recovery_v2")
_RECOVERY_V2_SPEEDS = (0.6, 0.9, 1.2, 1.25)
_RECOVERY_V2_DENSITIES = (70, 150, 205, 300)
_RECOVERY_V2_PRIMARY_CLASSES = {
    "brake_no_anchor_likely",
    "same_interval_brake_no_anchor_likely",
}
_RECOVERY_V2_COUNTER_INTEGER_KEYS = {
    "connected_goal_replans",
    "expanded_nodes",
    "fallback_intervals",
    "goal_completions",
    "invalid_count",
    "local_step_invalidations",
    "no_path_count",
    "plan_attempts",
    "plan_successes",
    "planning_batches",
    "planning_envs",
    "raw_waypoints",
    "recovery_brake_intervals",
    "recovery_brake_timeout_count",
    "recovery_connect_intervals",
    "recovery_connect_timeout_count",
    "recovery_entries",
    "recovery_hard_breach_count",
    "recovery_no_connector_count",
    "recovery_route_resumes",
    "replan_attempts",
    "same_goal_reselection_count",
    "smoothed_waypoints",
}
_RECOVERY_V2_COUNTER_KEYS = _RECOVERY_V2_COUNTER_INTEGER_KEYS | {
    "invalidation_counts",
    "total_planning_wall_s",
}
_RECOVERY_V2_INVALIDATION_KEYS = {
    "goal_changed",
    "local_step_infeasible",
    "support_contract_changed",
}
_RECOVERY_V2_TELEMETRY_INTEGER_KEYS = {
    "candidate_certificate_calls",
    "connect_actual_regressions",
    "connect_intervals",
    "contact_substeps",
    "direct_position_writes",
    "envs",
    "hard_breach_events",
    "interval_denominator",
    "no_connector_events",
    "open_recoveries_at_cell_end",
    "physics_substeps",
    "recovery_entries",
    "reset_calls_during_advance",
    "route_resumes",
    "runner_reset_during_recovery",
    "runner_reset_envs",
    "steps",
    "substep_denominator",
    "timeout_env_intervals",
    "watchdog_breach_substeps",
}
_RECOVERY_V2_TELEMETRY_FLOAT_KEYS = {
    "candidate_observer_device_ms",
    "connect_actual_max_increase_m",
    "connector_observer_cpu_ms",
}
_RECOVERY_V2_TELEMETRY_KEYS = (
    _RECOVERY_V2_TELEMETRY_INTEGER_KEYS
    | _RECOVERY_V2_TELEMETRY_FLOAT_KEYS
    | {"no_connector_reason_counts", "schema", "sha256"}
)
_NO_CONNECTOR_MARKER_SCHEMA = (
    "navrl_physical_target_recovery_v2_no_connector_forensics_v1_complete"
)

_MAIN_TTC_SOURCE_SHA256 = (
    "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
)
_MAIN_TTC_BASELINE_SHA256 = (
    "169ddcddb83c9d74df5c79252274660bc9c52e32d7d5144d325698e32b1d9b08"
)
_RISKCAP_TRAINED_SHA256 = (
    "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
)

_RECOVERY_SOURCE_SHA256 = (
    "3a0c167cbf4bc966426488f562da2b6788bd00ca62e3a31f226f5fbe1967578f"
)
_RECOVERY_CHECKPOINT_CONTRACT = {
    "num_task_steps": 307200,
    "cfg_ppo_horizon": 32,
    "k_max_cur": 28.0,
    "k_min_cur": 20.0,
    "cfg_training_seed": 1,
    "cfg_training_num_envs": 128,
    "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
    "cfg_training_task": "navrl_task",
    "cfg_training_sim": "base_sim",
    "cfg_training_profile": "main",
    "cfg_runtime_sim_config_class": "BaseSimConfig",
    "cfg_physics_dt_s": 0.01,
    "cfg_physics_substeps": 1,
    "cfg_physics_steps_per_rl_step": 10,
    "cfg_rl_step_dt_s": 0.1,
    "cfg_arena_xy": 40.0,
    "cfg_arena_z": 3.0,
    "cfg_bar_pool": "bars_h3",
    "cfg_placement_mode": "navrl_band",
    "cfg_placement_gap_m": 1.6,
    "cfg_placement_touch_m": 0.4,
    "cfg_bar_x_min": 0.0,
    "cfg_bar_x_max": 1.0,
    "cfg_episode_len_steps": 600,
    "cfg_general_goal_dist_min": 6.0,
    "cfg_general_goal_dist_max": 28.0,
    "cfg_lidar_max_range": 12.0,
    "cfg_lidar_hbeams": 72,
    "cfg_lidar_vbeams": 4,
    "cfg_max_obstacles": 8,
    "cfg_token_fov_deg": 240.0,
    "cfg_obstacle_suppress_deg": 10.0,
    "cfg_obstacle_selector": "cluster_sector",
    "cfg_obstacle_cluster_gap_m": 0.45,
    "cfg_obstacle_sectors": 8,
    "cfg_obstacle_ttc_idle_s": 30.0,
    "cfg_obstacle_ttc_min_speed": 0.15,
    "cfg_corridor_tokens": 0,
    "cfg_corridor_horizon_m": 6.0,
    "cfg_corridor_min_width_m": 0.55,
    "cfg_fov_curriculum_epochs": 3000,
    "cfg_detector_min_pixels": 2,
    "cfg_detector_threshold": 0.55,
    "cfg_detector_checkpoint_name": "",
    "cfg_detector_checkpoint_sha256": "",
    "cfg_perception_perturb": False,
    "cfg_detection_dropout": 0.3,
    "cfg_rgb_noise_std": 0.015,
    "cfg_depth_noise_std": 0.02,
    "cfg_max_velocity": 2.5,
    "cfg_alt_hold_vmax": 2.5,
    "cfg_yaw_rate_max": 3.0,
    "cfg_max_tilt_deg": 45.0,
    "cfg_tilt_comp": True,
    "cfg_target_motion_model": "symmetric_local_steer_v2_heading_continuity90",
    "cfg_target_pattern": "mixed",
    "cfg_target_speed_min": 0.3,
    "cfg_target_speed_final": 1.5,
    "cfg_target_speed_fixed": -1.0,
    "cfg_target_speed_ramp_epochs": 300,
    "cfg_target_speed_ramp_start_epochs": 0,
    "cfg_general_train": True,
    "cfg_oob_margin": 1.0,
    "cfg_action_policy": "squashed_gaussian",
    "cfg_action_std": "0.35,0.35,0.05,0.08",
    "cfg_action_mu_scale": "1.0,0.4,1.0,1.0",
    "cfg_action_entropy_coef": 0.0,
    "cfg_action_learning_rate": 5e-6,
    "current_action_learning_rate": 5e-6,
    "cfg_ppo_log_ratio_clamp": 10.0,
    "cfg_ppo_kl_stop": 0.04,
    "cfg_ppo_epoch_rollback": True,
    "cfg_ppo_rollback_lr_factor": 0.5,
    "cfg_ppo_rollback_min_lr": 1e-6,
    "cfg_ppo_rollback_patience": 5,
    "cfg_density_guard_window_epochs": 50,
    "cfg_density_guard_min_epochs": 100,
    "cfg_density_guard_min_peak": 0.5,
    "cfg_density_guard_drop": 0.25,
    "cfg_density_guard_patience": 25,
    "cfg_lateral_latent_margin_y": 1.25,
    "cfg_latent_margin": "2.0,1.25,2.0,2.0",
    "cfg_lateral_latent_margin_coef": 0.01,
    "cfg_episode_limit_comparator": "gte",
    "cfg_rlgames_timeout_info_key": "time_outs",
    "cfg_time_limit_bootstrap_signal": True,
}
_RECOVERY_RESULT_CONTRACT = {
    "schema_version": 2,
    "episode_limit_steps": 600,
    "episode_limit_comparator": "gte",
    "timeout_observed_at_step": 600,
    "pursuer_speed_limit_semantics": "per_axis_xy",
    "pursuer_per_axis_speed_limit_mps": 2.5,
    "pursuer_max_horizontal_request_norm_mps": math.sqrt(2.0) * 2.5,
    "policy_output_dim": 4,
    "policy_z_output_overwritten_by_altitude_pi": True,
    "policy_z_persisted_in_prev_action_observation": True,
    "runtime_sim": "base_sim",
    "runtime_profile": "main",
    "runtime_num_envs": 128,
    "sim_physics_contract": "base_sim_dt0.01",
    "runtime_sim_config_class": "BaseSimConfig",
    "physics_dt_s": 0.01,
    "physics_substeps": 1,
    "physics_steps_per_rl_step": 10,
    "rl_step_dt_s": 0.1,
    "arena_xy_m": 40.0,
    "goal_dist_min_m": 6.0,
    "goal_dist_max_m": 28.0,
    "full_goal_distribution": True,
    "fov_curriculum_saturated": True,
    "target_speed_distribution": "uniform",
    "target_speed_min_mps": 0.3,
    "target_speed_max_mps": 1.5,
    "target_pattern": "mixed",
    "lidar_beams": [4, 72],
    "lidar_range_m": 12.0,
    "obstacle_tokens": 8,
    "obstacle_fov_deg": 240.0,
    "obstacle_selector": "cluster_sector",
    "obstacle_ttc_idle_s": 30.0,
    "obstacle_ttc_min_speed": 0.15,
    "fov_curriculum_epochs": 3000,
    "detector_checkpoint_sha256": "",
    "detector_min_pixels": 2,
    "detector_threshold": 0.55,
    "perception_perturb": False,
    "detection_dropout": 0.3,
    "detection_dropout_active": 0.0,
    "rgb_noise_std": 0.015,
    "depth_noise_std": 0.02,
    "max_tilt_deg": 45.0,
    "tilt_comp": True,
    "oob_margin_m": 1.0,
    "seed": 42,
}
_RECOVERY_ATTESTATION_THRESHOLDS = {
    "min_episodes": 2049,
    "min_capture_rate": 0.65,
    "max_crash_rate": 0.35,
    "max_timeout_rate": 0.10,
    "max_training_kl": 0.04,
    "max_task_input_oob_rate": 1e-9,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object without allowing absent or corrupt evidence to crash the snapshot."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _unavailable_evidence(path: Path) -> Dict[str, Any]:
    try:
        source = str(path.relative_to(ROOT))
    except ValueError:
        source = str(path)
    return {
        "source": source,
        "status": "RESULT_UNAVAILABLE_OR_MALFORMED",
        "authority": "NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN",
        "physical_ppo": "BLOCKED",
        "hardware_claim": False,
    }


def _load_status_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"module loader unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_recovery_v2_gate_summary() -> Optional[Dict[str, Any]]:
    """Verify the immutable gate receipt before reading its summary."""

    env_key = "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"
    previous_variant = os.environ.get(env_key)
    try:
        os.environ[env_key] = "baseline_1p25"
        module = _load_status_module(
            RECOVERY_V2_GATE_VERIFIER_PATH,
            "_recovery_v2_gate_verifier_for_status",
        )
        verifier = getattr(module, "verify_result", None)
        if not callable(verifier):
            raise AttributeError("verify_result is unavailable")
        with contextlib.redirect_stdout(io.StringIO()):
            payload = verifier(RECOVERY_V2_LOWER1P25_GATE_PATH.parent)
        if not isinstance(payload, dict):
            raise TypeError("canonical verifier returned a non-object")
        return payload
    except Exception:
        return None
    finally:
        if previous_variant is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_variant


def _canonical_no_connector_summary() -> Optional[Dict[str, Any]]:
    """Verify the forensic receipt and completion marker before reading its summary."""

    output_dir = RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH.parent
    receipt_path = output_dir / "receipt.json"
    summary_path = output_dir / "summary.json"
    marker_path = output_dir / ".COMPLETE.json"
    try:
        module = _load_status_module(
            RECOVERY_V2_NO_CONNECTOR_VERIFIER_PATH,
            "_recovery_v2_no_connector_verifier_for_status",
        )
        verifier = getattr(module, "verify_receipt", None)
        if not callable(verifier):
            raise AttributeError("verify_receipt is unavailable")
        with contextlib.redirect_stdout(io.StringIO()):
            result = verifier(output_dir)
        if result != 0:
            raise RuntimeError(f"forensic verifier returned {result}")

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            not isinstance(marker, dict)
            or set(marker) != {"schema", "receipt_sha256", "summary_sha256"}
            or marker["schema"] != _NO_CONNECTOR_MARKER_SCHEMA
            or marker["receipt_sha256"] != _sha256(receipt_path)
            or marker["summary_sha256"] != _sha256(summary_path)
        ):
            raise ValueError("forensic completion marker mismatch")
        payload = _read_json_object(summary_path)
        if payload is None:
            raise ValueError("forensic summary is unavailable or malformed")
        return payload
    except Exception:
        return None


def _require_real_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_recovery_v2_telemetry(telemetry: Any, label: str) -> None:
    if not isinstance(telemetry, dict) or set(telemetry) != _RECOVERY_V2_TELEMETRY_KEYS:
        raise ValueError(f"{label} schema mismatch")
    if telemetry["schema"] != "navrl_target_recovery_packed_telemetry_v1":
        raise ValueError(f"{label}.schema mismatch")
    if (
        not isinstance(telemetry["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", telemetry["sha256"])
    ):
        raise ValueError(f"{label}.sha256 mismatch")
    for key in _RECOVERY_V2_TELEMETRY_INTEGER_KEYS:
        _require_nonnegative_int(telemetry[key], f"{label}.{key}")
    for key in _RECOVERY_V2_TELEMETRY_FLOAT_KEYS:
        _require_finite_number(
            telemetry[key],
            f"{label}.{key}",
            nonnegative=key != "connect_actual_max_increase_m",
        )
    reason_counts = telemetry["no_connector_reason_counts"]
    if not isinstance(reason_counts, dict):
        raise ValueError(f"{label}.no_connector_reason_counts must be an object")
    for key, value in reason_counts.items():
        if not isinstance(key, str):
            raise ValueError(f"{label}.no_connector_reason_counts key mismatch")
        _require_nonnegative_int(value, f"{label}.no_connector_reason_counts[{key}]")
    if sum(reason_counts.values()) != telemetry["no_connector_events"]:
        raise ValueError(f"{label}.no_connector_reason_counts arithmetic mismatch")
    if telemetry["interval_denominator"] != telemetry["envs"] * telemetry["steps"]:
        raise ValueError(f"{label}.interval_denominator mismatch")
    if telemetry["no_connector_events"] > telemetry["interval_denominator"]:
        raise ValueError(f"{label}.no_connector_events exceeds denominator")
    if telemetry["hard_breach_events"] > telemetry["no_connector_events"]:
        raise ValueError(f"{label}.hard_breach_events exceeds entries")


def _validate_recovery_v2_cell(cell: Any, expected: tuple[str, float, int]) -> None:
    if not isinstance(cell, dict):
        raise ValueError("recovery-v2 cell is not an object")
    route_mode, speed, bars = expected
    if (
        cell.get("route_mode") != route_mode
        or cell.get("bars") != bars
        or cell.get("speed_mps") != speed
    ):
        raise ValueError("recovery-v2 cell identity mismatch")
    if not isinstance(cell.get("record_id"), str):
        raise ValueError("recovery-v2 record ID missing")
    if cell.get("seed") != 827 or cell.get("envs") != 32:
        raise ValueError("recovery-v2 seed/env contract mismatch")
    if cell.get("steps") != 300 or cell.get("warmup_steps") != 20:
        raise ValueError("recovery-v2 interval contract mismatch")
    _require_real_bool(cell.get("pass"), "cell.pass")
    for key in (
        "mean_speed_mps",
        "mean_speed_ratio",
        "tracking_rmse_mps",
        "contact_step_fraction",
        "invalid_state_fraction",
        "motor_saturation_fraction",
        "max_tilt_deg",
    ):
        _require_finite_number(cell[key], f"cell.{key}", nonnegative=True)
    local_key = (
        "off_rounded_local_infeasible_fraction"
        if route_mode == "off"
        else "recovery_normal_route_rounded_invalidation_fraction"
    )
    _require_finite_number(cell[local_key], f"cell.{local_key}", nonnegative=True)
    telemetry = cell.get("telemetry_summary")
    _validate_recovery_v2_telemetry(telemetry, "cell.telemetry_summary")
    if telemetry["envs"] != cell["envs"] or telemetry["steps"] != cell["steps"]:
        raise ValueError("cell telemetry contract mismatch")

    route = cell.get("route")
    if not isinstance(route, dict) or route.get("mode") != route_mode:
        raise ValueError("cell.route schema mismatch")
    counter = route.get("counter_delta")
    if not isinstance(counter, dict):
        raise ValueError("cell.route.counter_delta schema mismatch")
    if route_mode == "off":
        if counter:
            raise ValueError("off cell has route counters")
        if any(route.get(key) is not None for key in (
            "plan_success_fraction",
            "fallback_interval_fraction",
            "goal_completions_per_env",
        )):
            raise ValueError("off cell has route metrics")
        _require_finite_number(route.get("planning_wall_s"), "route.planning_wall_s", nonnegative=True)
        return
    if set(counter) != _RECOVERY_V2_COUNTER_KEYS:
        raise ValueError("recovery cell counter schema mismatch")
    for key in _RECOVERY_V2_COUNTER_INTEGER_KEYS:
        _require_nonnegative_int(counter[key], f"cell.route.counter_delta.{key}")
    _require_finite_number(
        counter["total_planning_wall_s"],
        "cell.route.counter_delta.total_planning_wall_s",
        nonnegative=True,
    )
    invalidation_counts = counter["invalidation_counts"]
    if not isinstance(invalidation_counts, dict) or set(invalidation_counts) != _RECOVERY_V2_INVALIDATION_KEYS:
        raise ValueError("cell invalidation counter schema mismatch")
    for key in _RECOVERY_V2_INVALIDATION_KEYS:
        _require_nonnegative_int(invalidation_counts[key], f"cell.invalidation_counts.{key}")
    if counter["plan_successes"] > counter["plan_attempts"]:
        raise ValueError("plan successes exceed attempts")
    if counter["fallback_intervals"] > telemetry["interval_denominator"]:
        raise ValueError("fallback intervals exceed denominator")
    for key in (
        "plan_success_fraction",
        "fallback_interval_fraction",
        "goal_completions_per_env",
    ):
        value = _require_finite_number(route[key], f"cell.route.{key}", nonnegative=True)
        if key != "goal_completions_per_env" and value > 1.0:
            raise ValueError(f"cell.route.{key} exceeds one")
    _require_finite_number(route["planning_wall_s"], "route.planning_wall_s", nonnegative=True)


def _recovery_v2_packed_report() -> Optional[Dict[str, Any]]:
    """Recompute occupancy from finalized packed telemetry without launching a simulator."""

    try:
        spec = importlib.util.spec_from_file_location(
            "_recovery_v2_packed_for_status", RECOVERY_V2_PACKED_DIAGNOSTIC_PATH
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.diagnose_gate(RECOVERY_V2_LOWER1P25_GATE_PATH.parent)
    except Exception:
        return None
    return report if isinstance(report, dict) else None


def _recovery_v2_lower1p25_gate() -> Dict[str, Any]:
    """Build the current Track-B gate block from the finalized 32-cell summary."""

    payload = _canonical_recovery_v2_gate_summary()
    if payload is None:
        return _unavailable_evidence(RECOVERY_V2_LOWER1P25_GATE_PATH)
    try:
        if payload["schema"] != "navrl_physical_target_recovery_v2_gate_v1":
            raise ValueError("unexpected schema")
        if payload["contract_variant"] != "baseline_1p25":
            raise ValueError("unexpected contract variant")
        verdict = payload["verdict"]
        cells = payload["cells"]
        if not isinstance(verdict, dict) or not isinstance(cells, list) or len(cells) != 32:
            raise ValueError("invalid verdict or cell grid")

        expected_cells = {
            (route, speed, bars)
            for route in _RECOVERY_V2_ROUTE_MODES
            for speed in _RECOVERY_V2_SPEEDS
            for bars in _RECOVERY_V2_DENSITIES
        }
        seen_cells = set()
        for cell in cells:
            if not isinstance(cell, dict):
                raise ValueError("recovery-v2 cell is not an object")
            identity = (cell.get("route_mode"), cell.get("speed_mps"), cell.get("bars"))
            if identity in seen_cells or identity not in expected_cells:
                raise ValueError("invalid or duplicate recovery-v2 cell identity")
            seen_cells.add(identity)
            _validate_recovery_v2_cell(cell, identity)
        if seen_cells != expected_cells:
            raise ValueError("recovery-v2 grid is incomplete")

        route_off = [cell for cell in cells if cell["route_mode"] == "off"]
        recovery = [cell for cell in cells if cell["route_mode"] != "off"]
        route_off_pass = sum(cell["pass"] is True for cell in route_off)
        recovery_pass = sum(cell["pass"] is True for cell in recovery)
        cells_pass = route_off_pass + recovery_pass

        pool70 = [cell for cell in recovery if int(cell["bars"]) == 70]
        if len(pool70) != 4:
            raise ValueError("invalid 70-bar pool")
        plan_attempts = sum(
            int(cell["route"]["counter_delta"]["plan_attempts"]) for cell in pool70
        )
        plan_successes = sum(
            int(cell["route"]["counter_delta"]["plan_successes"]) for cell in pool70
        )
        fallback_intervals_70 = sum(
            int(cell["route"]["counter_delta"]["fallback_intervals"])
            for cell in pool70
        )
        interval_denominator_70 = sum(
            int(cell["telemetry_summary"]["interval_denominator"]) for cell in pool70
        )
        low_speed = [
            cell for cell in pool70 if math.isclose(float(cell["speed_mps"]), 0.6)
        ]
        if len(low_speed) != 1:
            raise ValueError("missing 70-bar 0.6 m/s cell")
        low_speed_cell = low_speed[0]
        goals = int(low_speed_cell["route"]["counter_delta"]["goal_completions"])
        goal_envs = int(low_speed_cell["envs"])

        packed = _recovery_v2_packed_report()
        if packed is None:
            raise ValueError("packed telemetry unavailable")
        occupancy = packed["pooled_occupancy"]
        if not isinstance(occupancy, dict):
            raise ValueError("packed occupancy is not an object")
        no_connector_intervals = _require_nonnegative_int(
            occupancy["no_connector"], "packed occupancy.no_connector"
        )
        recovery_intervals = _require_nonnegative_int(
            occupancy["intervals"], "packed occupancy.intervals"
        )
        hard_breaches = sum(
            int(cell["telemetry_summary"]["hard_breach_events"]) for cell in recovery
        )
        no_connector_entries = sum(
            int(cell["telemetry_summary"]["no_connector_events"]) for cell in recovery
        )
        if plan_attempts <= 0 or interval_denominator_70 <= 0 or goal_envs <= 0:
            raise ValueError("invalid denominator")
        if plan_successes > plan_attempts:
            raise ValueError("pooled plan successes exceed attempts")
        if fallback_intervals_70 > interval_denominator_70:
            raise ValueError("pooled fallback exceeds denominator")
        if recovery_intervals <= 0 or no_connector_entries <= 0:
            raise ValueError("invalid recovery denominator")
        if no_connector_intervals > recovery_intervals:
            raise ValueError("pooled occupancy exceeds denominator")
        if hard_breaches > no_connector_entries:
            raise ValueError("pooled hard breaches exceed entries")
        if recovery_intervals != sum(
            cell["telemetry_summary"]["interval_denominator"] for cell in recovery
        ):
            raise ValueError("pooled occupancy denominator mismatch")
        if (
            verdict["execution_integrity"] != "PASS_32_CELL_INTEGRITY"
            or verdict["route_mechanism"] != "FAIL_ROUTE_MECHANISM"
            or _require_real_bool(
                verdict["long_training_authorized"], "verdict.long_training_authorized"
            )
            or _require_real_bool(
                verdict["all_cell_controller_and_recovery_gates"],
                "verdict.all_cell_controller_and_recovery_gates",
            )
            != (cells_pass == len(cells))
            or packed["execution_integrity"] != verdict["execution_integrity"]
            or packed["route_mechanism"] != verdict["route_mechanism"]
            or _require_nonnegative_int(
                packed["pooled_no_connector_classes"]["entries"],
                "packed pooled_no_connector_classes.entries",
            )
            != no_connector_entries
        ):
            raise ValueError("unexpected gate verdict")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _unavailable_evidence(RECOVERY_V2_LOWER1P25_GATE_PATH)

    return {
        "source": str(RECOVERY_V2_LOWER1P25_GATE_PATH.relative_to(ROOT)),
        "receipt": (
            "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/"
            "receipt.json"
        ),
        "result_doc": "docs/physical_target_recovery_v2_lower1p25_result_2026-08-26.md",
        "preregistration": (
            "docs/preregistration_physical_target_recovery_v2_lower1p25_gate_"
            "2026-08-26.md"
        ),
        "tool": "tools/verify_navrl_physical_target_recovery_v2_gate.py",
        "status": "VERIFIED_FAIL",
        "contract_variant": "baseline_1p25",
        "canonical_1p5_contract": "SEPARATE_UNCHANGED_NOT_PASSED",
        "integrity": verdict["execution_integrity"],
        "route_mechanism": verdict["route_mechanism"],
        "cells": {
            "passed": cells_pass,
            "total": len(cells),
            "route_off_passed": route_off_pass,
            "route_off_total": len(route_off),
            "recovery_passed": recovery_pass,
            "recovery_total": len(recovery),
            "passing_lineage": "route_off_only",
        },
        "plan_success_70bar_4speed": {
            "numerator": plan_successes,
            "denominator": plan_attempts,
            "fraction": plan_successes / plan_attempts,
            "pct": round(100.0 * plan_successes / plan_attempts, 2),
            "gate_pct": 99.0,
        },
        "fallback_70bar_4speed": {
            "numerator": fallback_intervals_70,
            "denominator": interval_denominator_70,
            "fraction": fallback_intervals_70 / interval_denominator_70,
            "pct": round(100.0 * fallback_intervals_70 / interval_denominator_70, 2),
            "gate_pct": 1.0,
        },
        "goals_per_env_70bar_0_6mps": {
            "numerator": goals,
            "denominator": goal_envs,
            "value": goals / goal_envs,
            "gate": 0.5,
        },
        "no_connector_occupancy": {
            "numerator": no_connector_intervals,
            "denominator": recovery_intervals,
            "fraction": no_connector_intervals / recovery_intervals,
            "pct": round(100.0 * no_connector_intervals / recovery_intervals, 2),
        },
        "hard_breach_no_connector_entries": {
            "numerator": hard_breaches,
            "denominator": no_connector_entries,
        },
        "physical_ppo": "BLOCKED",
        "hardware_claim": False,
        "authority": "NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN",
    }


def _historical_route_recovery_forensics() -> Dict[str, Any]:
    """Rebuild the v1 diagnostic lineage from its finalized summary."""

    payload = _read_json_object(ROUTE_RECOVERY_FORENSICS_PATH)
    if payload is None:
        unavailable = _unavailable_evidence(ROUTE_RECOVERY_FORENSICS_PATH)
        unavailable["lineage_status"] = "HISTORICAL_V1_DIAGNOSTIC"
        return unavailable
    try:
        if payload["schema"] != "navrl_physical_target_route_recovery_forensics_v1_summary":
            raise ValueError("unexpected schema")
        rule = payload["decision_rule"]
        event_counts = payload["event_counts"]
        plans = payload["plan_status_counts"]
        ages = payload["metric_stats"]["fallback_age_steps"]
        per_cell = payload["per_cell"]
        if rule["verdict"] != "RECOVERY_DOMINANT" or len(per_cell) != 8:
            raise ValueError("unexpected historical verdict")
    except (KeyError, TypeError, ValueError):
        unavailable = _unavailable_evidence(ROUTE_RECOVERY_FORENSICS_PATH)
        unavailable["lineage_status"] = "HISTORICAL_V1_DIAGNOSTIC"
        return unavailable
    return {
        "source": str(ROUTE_RECOVERY_FORENSICS_PATH.relative_to(ROOT)),
        "summary_markdown": (
            "results/navrl_physical_target_route_recovery_forensics_seed827/"
            "summary.md"
        ),
        "receipt": (
            "results/navrl_physical_target_route_recovery_forensics_seed827/"
            "receipt.json"
        ),
        "preregistration": (
            "docs/preregistration_physical_target_route_recovery_forensics_"
            "2026-08-25.md"
        ),
        "cells_verified": f"{len(per_cell)}/{len(per_cell)}",
        "diagnostic_verdict": rule["verdict"],
        "claim_scope": "evaluation-only; no PPO/fix/tuning authority",
        "local_invalidations": int(event_counts["invalidation"]),
        "local_fallback_intervals": int(payload["local_fallback_intervals"]),
        "fallback_amplification": round(
            float(rule["attributed_fallback_to_local_invalidation"]), 4
        ),
        "fallback_age_steps": {
            "median": int(ages["median"]),
            "p90": int(ages["p90"]),
            "max": int(ages["max"]),
        },
        "unique_local_origins": int(rule["unsafe_start_replan_n"]),
        "hard_free_soft_unsafe_pct": round(
            100.0 * float(rule["hard_free_soft_unsafe_ratio"]), 2
        ),
        "hard_free_soft_unsafe_wilson_lower_pct": round(
            100.0 * float(rule["hard_free_soft_unsafe_wilson95_lower"]), 2
        ),
        "exact_hard_connector_pct": round(
            100.0 * float(rule["exact_hard_connector_ratio"]), 2
        ),
        "exact_hard_connector_wilson_lower_pct": round(
            100.0 * float(rule["exact_hard_connector_wilson95_lower"]), 2
        ),
        "replan_status": dict(plans["replan"]),
        "initial_plan_status": dict(plans["initial_plan"]),
        "rounded_vs_square_disagreements": int(
            payload["rounded_vs_aabb_soft_disagreements"]
        ),
        "margin_tuning_allowed": False,
        "physical_ppo": "BLOCKED",
        "lineage_status": "HISTORICAL_V1_DIAGNOSTIC",
    }


def _validate_no_connector_class_counts(
    classes: Any, *, allow_empty: bool = False
) -> Dict[str, int]:
    if not isinstance(classes, dict) or (not classes and not allow_empty):
        raise ValueError("forensic class counts are missing")
    result: Dict[str, int] = {}
    for key, value in classes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("forensic class name is invalid")
        result[key] = _require_nonnegative_int(value, f"forensic class {key}")
    return result


def _validate_no_connector_wilson(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"lower", "upper"}:
        raise ValueError(f"{label} schema mismatch")
    lower = _require_finite_number(value["lower"], f"{label}.lower")
    upper = _require_finite_number(value["upper"], f"{label}.upper")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"{label} bounds mismatch")


def _validate_no_connector_decision(
    decision: Any,
    primary_n: int,
    anchor_present: int,
    identity_void: bool,
    label: str,
) -> None:
    if not isinstance(decision, dict):
        raise ValueError(f"{label}.decision_rule is not an object")
    if decision["label"] != "INCONCLUSIVE":
        raise ValueError(f"{label}.decision_rule label mismatch")
    if (
        _require_nonnegative_int(decision["primary_n"], f"{label}.decision_rule.primary_n")
        != primary_n
        or _require_nonnegative_int(
            decision["anchor_present"], f"{label}.decision_rule.anchor_present"
        )
        != anchor_present
        or _require_real_bool(
            decision["identity_void"], f"{label}.decision_rule.identity_void"
        )
        != identity_void
    ):
        raise ValueError(f"{label}.decision_rule count mismatch")
    if (
        _require_real_bool(
            decision["passes_32_cell_mechanism"],
            f"{label}.decision_rule.passes_32_cell_mechanism",
        )
        or _require_real_bool(
            decision["authorizes_retune_or_ppo"],
            f"{label}.decision_rule.authorizes_retune_or_ppo",
        )
    ):
        raise ValueError(f"{label}.decision_rule authority mismatch")
    _validate_no_connector_wilson(
        decision["wilson_present"], f"{label}.decision_rule.wilson_present"
    )
    _validate_no_connector_wilson(
        decision["wilson_absent"], f"{label}.decision_rule.wilson_absent"
    )


def _recovery_v2_no_connector_forensics() -> Dict[str, Any]:
    """Build the current descriptive forensics block from its decision rule."""

    payload = _canonical_no_connector_summary()
    if payload is None:
        return _unavailable_evidence(RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH)
    try:
        if (
            payload["schema"]
            != "navrl_physical_target_recovery_v2_no_connector_forensics_v1_summary"
        ):
            raise ValueError("unexpected schema")
        if payload.get("interpretation") != "descriptive_only_no_gate_or_tuning_authority":
            raise ValueError("unexpected forensic interpretation")
        if _require_real_bool(
            payload["passes_32_cell_mechanism"], "forensics.passes_32_cell_mechanism"
        ):
            raise ValueError("forensic result cannot pass the 32-cell mechanism")

        expected_cells = {(70, speed) for speed in _RECOVERY_V2_SPEEDS}
        seen_cells = set()
        aggregate_classes: Dict[str, int] = {}
        aggregate_primary = 0
        aggregate_anchor = 0
        aggregate_hard_free = 0
        aggregate_identity_void = False
        cells = payload["cells"]
        if not isinstance(cells, list):
            raise ValueError("forensic cells are not a list")
        for cell in cells:
            if not isinstance(cell, dict):
                raise ValueError("forensic cell is not an object")
            density = _require_nonnegative_int(cell["density"], "forensic cell.density")
            speed = _require_finite_number(cell["speed_mps"], "forensic cell.speed_mps")
            identity = (density, speed)
            if identity in seen_cells or identity not in expected_cells:
                raise ValueError("invalid or duplicate forensic cell identity")
            seen_cells.add(identity)
            analysis = cell["analysis"]
            if not isinstance(analysis, dict):
                raise ValueError("forensic cell analysis is not an object")
            cell_classes = _validate_no_connector_class_counts(
                analysis["class_counts"], allow_empty=True
            )
            for key, value in cell_classes.items():
                aggregate_classes[key] = aggregate_classes.get(key, 0) + value
            cell_primary = _require_nonnegative_int(
                analysis["primary_n"], "forensic cell.primary_n"
            )
            cell_anchor = _require_nonnegative_int(
                analysis["anchor_present"], "forensic cell.anchor_present"
            )
            cell_hard_free = _require_nonnegative_int(
                analysis["hard_free_soft_unsafe"],
                "forensic cell.hard_free_soft_unsafe",
            )
            cell_identity_void = _require_real_bool(
                analysis["identity_void"], "forensic cell.identity_void"
            )
            if (
                cell_primary
                != sum(
                    cell_classes.get(name, 0)
                    for name in _RECOVERY_V2_PRIMARY_CLASSES
                )
                or cell_anchor > cell_primary
                or cell_hard_free > cell_primary
            ):
                raise ValueError("forensic cell count arithmetic mismatch")
            _validate_no_connector_decision(
                analysis["decision_rule"],
                cell_primary,
                cell_anchor,
                cell_identity_void,
                f"forensic cell {density}/{speed}",
            )
            aggregate_primary += cell_primary
            aggregate_anchor += cell_anchor
            aggregate_hard_free += cell_hard_free
            aggregate_identity_void = aggregate_identity_void or cell_identity_void
        if seen_cells != expected_cells:
            raise ValueError("forensic cell grid is incomplete")

        pooled = payload["pooled"]
        if not isinstance(pooled, dict):
            raise ValueError("forensic pooled data is not an object")
        class_counts = _validate_no_connector_class_counts(
            pooled["class_counts"]
        )
        total_entries = sum(class_counts.values())
        pooled_primary = _require_nonnegative_int(
            pooled["primary_n"], "forensics.pooled.primary_n"
        )
        pooled_anchor = _require_nonnegative_int(
            pooled["anchor_present"], "forensics.pooled.anchor_present"
        )
        pooled_hard_free = _require_nonnegative_int(
            pooled["hard_free_soft_unsafe"],
            "forensics.pooled.hard_free_soft_unsafe",
        )
        pooled_identity_void = _require_real_bool(
            pooled["identity_void"], "forensics.pooled.identity_void"
        )
        if (
            total_entries <= 0
            or class_counts != aggregate_classes
            or pooled_primary != aggregate_primary
            or pooled_anchor != aggregate_anchor
            or pooled_hard_free != aggregate_hard_free
            or pooled_identity_void != aggregate_identity_void
            or pooled_primary > total_entries
            or pooled_anchor > pooled_primary
            or pooled_hard_free > pooled_primary
            or pooled_primary
            != sum(class_counts.get(name, 0) for name in _RECOVERY_V2_PRIMARY_CLASSES)
        ):
            raise ValueError("pooled forensic count arithmetic mismatch")
        decision = payload["decision_rule"]
        _validate_no_connector_decision(
            decision,
            pooled_primary,
            pooled_anchor,
            pooled_identity_void,
            "forensics",
        )
        if decision != pooled["decision_rule"]:
            raise ValueError("pooled/top-level forensic decision mismatch")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _unavailable_evidence(RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH)

    return {
        "source": str(RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH.relative_to(ROOT)),
        "receipt": (
            "results/navrl_physical_target_recovery_v2_no_connector_forensics_"
            "seed827/receipt.json"
        ),
        "result_doc": (
            "docs/physical_target_recovery_v2_no_connector_forensics_result_"
            "2026-08-26.md"
        ),
        "preregistration": (
            "docs/preregistration_physical_target_recovery_v2_no_connector_"
            "forensics_2026-08-26.md"
        ),
        "tool": (
            "tools/diagnose_navrl_physical_target_recovery_v2_no_connector.py"
        ),
        "status": "DESCRIPTIVE_ONLY",
        "decision_rule": {
            "label": decision["label"],
            "primary_n": int(decision["primary_n"]),
            "anchor_present": int(decision["anchor_present"]),
            "hard_free_soft_unsafe": pooled_hard_free,
            "identity_void": bool(decision["identity_void"]),
            "passes_32_cell_mechanism": bool(
                decision["passes_32_cell_mechanism"]
            ),
            "authorizes_retune_or_ppo": bool(
                decision["authorizes_retune_or_ppo"]
            ),
        },
        "no_connector_classes": {
            "total": total_entries,
            **class_counts,
        },
        "claim_scope": "DESCRIPTIVE_ONLY_NO_GATE_TUNING_RERUN_PPO_OR_HARDWARE_AUTHORITY",
        "physical_ppo": "BLOCKED",
        "hardware_claim": False,
        "canonical_1p5_claim": False,
        "authority": "NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN",
    }


def _contract_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        number = _finite_number(actual)
        return number is not None and math.isclose(
            number, float(expected), rel_tol=0.0, abs_tol=1e-9
        )
    return actual == expected


def _check_contract(
    actual: Any, expected: Dict[str, Any], prefix: str, errors: List[str]
) -> None:
    if not isinstance(actual, dict):
        errors.append(f"{prefix}: not an object")
        return
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{prefix}.{key}: missing")
        elif not _contract_value_matches(actual[key], expected_value):
            errors.append(f"{prefix}.{key}: contract mismatch")


def _run_canonical_recovery_verifier(
    checkpoint_path: Path, attestation_path: Path, errors: List[str]
) -> None:
    """Recompute the attestation through its canonical checkpoint/result/TB verifier.

    The dashboard must fail closed if the verifier cannot be imported or executed.  In
    particular, self-reported KL/OOB/rollback values are not evidence: the canonical verifier
    rereads the exact 100-epoch TensorBoard window and rebuilds the complete payload.
    """

    try:
        spec = importlib.util.spec_from_file_location(
            "_navrl_v2_recovery_attestation_for_status",
            RECOVERY_ATTESTATION_VERIFIER_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError("module spec has no loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verifier = getattr(module, "verify_existing_attestation", None)
        if not callable(verifier):
            raise AttributeError("verify_existing_attestation is unavailable")
        verifier(checkpoint_path, attestation_path)
    except Exception as exc:
        errors.append(f"canonical verifier: {exc}")


def _normal_completion_marker_epoch(run_dir: Path) -> Optional[int]:
    """Return the epoch only for the exact marker emitted by a normal trainer exit.

    A metrics row at epoch 9600 proves that an epoch was logged, not that the runner completed its
    save/flush/finish path.  Treat extra or malformed marker content as invalid too, so a stale or
    hand-edited file cannot turn an interrupted smoke into a completed one on the dashboard.
    """

    try:
        marker_text = (run_dir / ".aerial_training_finished").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.fullmatch(r"epoch=(\d+)\n?", marker_text)
    return int(match.group(1)) if match else None


def _validate_recovery_attestation(run_dir: Path) -> List[str]:
    """Validate every artifact behind the dashboard's recovery PASS claim.

    The attestation file is deliberately not treated as a signature.  It is accepted only when its
    checkpoint and held-out paths still exist, both byte digests match, the checkpoint carries the
    complete safe-recovery contract, and the held-out artifact independently reproduces every
    seed/outcome/OOB/rollback field quoted by the attestation.
    """
    run_dir = run_dir.resolve()
    errors: List[str] = []
    attestation_path = run_dir / ".navrl_v2_recovery_eval_pass.json"
    if not attestation_path.is_file():
        return ["attestation: missing"]
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"attestation: unreadable ({exc})"]
    if not isinstance(attestation, dict):
        return ["attestation: not an object"]

    def expect_equal(label: str, actual: Any, expected: Any) -> None:
        if not _contract_value_matches(actual, expected):
            errors.append(f"{label}: mismatch")

    expect_equal("attestation.schema_version", attestation.get("schema_version"), 1)
    expect_equal("attestation.verdict", attestation.get("verdict"), "PASS")
    expect_equal("attestation.checkpoint_epoch", attestation.get("checkpoint_epoch"), 9600)
    expect_equal("attestation.checkpoint_frame", attestation.get("checkpoint_frame"), 39321600)
    expect_equal(
        "attestation.source_checkpoint_sha256",
        attestation.get("source_checkpoint_sha256"),
        _RECOVERY_SOURCE_SHA256,
    )
    expect_equal("attestation.source_epoch", attestation.get("source_epoch"), 9500)
    expect_equal("attestation.smoke_epochs", attestation.get("smoke_epochs"), 100)
    expect_equal("attestation.bars", attestation.get("bars"), 130)
    expect_equal("attestation.seed", attestation.get("seed"), 42)
    _check_contract(
        attestation.get("thresholds"),
        _RECOVERY_ATTESTATION_THRESHOLDS,
        "attestation.thresholds",
        errors,
    )
    _check_contract(
        attestation.get("evaluation_contract"),
        _RECOVERY_RESULT_CONTRACT,
        "attestation.evaluation_contract",
        errors,
    )
    created_at = str(attestation.get("created_at_utc", ""))
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError:
        errors.append("attestation.created_at_utc: invalid")

    if _normal_completion_marker_epoch(run_dir) != 9600:
        errors.append("training marker: missing, malformed, or not exact epoch 9600")

    checkpoint_path = Path(str(attestation.get("checkpoint", ""))).expanduser().resolve()
    expected_nn_dir = (run_dir / "nn").resolve()
    if checkpoint_path.parent != expected_nn_dir:
        errors.append("checkpoint: outside recovery run nn directory")
    if not checkpoint_path.is_file():
        errors.append("checkpoint: missing")
        checkpoint_digest = None
        checkpoint = None
    else:
        try:
            checkpoint_digest = _sha256(checkpoint_path)
        except OSError as exc:
            errors.append(f"checkpoint: unreadable ({exc})")
            checkpoint_digest = None
        if checkpoint_digest != attestation.get("checkpoint_sha256"):
            errors.append("checkpoint: SHA-256 mismatch")
        try:
            import torch

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            errors.append(f"checkpoint: cannot load ({exc})")
            checkpoint = None

    if not isinstance(checkpoint, dict):
        if checkpoint is not None:
            errors.append("checkpoint: not an object")
    else:
        expect_equal("checkpoint.epoch", checkpoint.get("epoch"), 9600)
        expect_equal("checkpoint.frame", checkpoint.get("frame"), 39321600)
        state = checkpoint.get("env_state")
        _check_contract(
            state, _RECOVERY_CHECKPOINT_CONTRACT, "checkpoint.env_state", errors
        )
        if isinstance(state, dict):
            lineage = {
                "cfg_recovery_stage": "smoke",
                "cfg_recovery_source_sha256": _RECOVERY_SOURCE_SHA256,
                "cfg_recovery_source_epoch": 9500,
                "cfg_recovery_smoke_required_epochs": 100,
                "cfg_recovery_smoke_bars": 130,
                "n_bars_active": 130,
            }
            _check_contract(state, lineage, "checkpoint.env_state", errors)

    # This is the authoritative recomputation of checkpoint/result/TensorBoard evidence. Keep the
    # local checks below as an independent, dashboard-specific fail-closed layer, but never unlock
    # merely because a hand-written JSON repeats plausible PASS values.
    if checkpoint_path.is_file():
        _run_canonical_recovery_verifier(checkpoint_path, attestation_path, errors)

    result_path = Path(
        str(attestation.get("heldout_result_json", ""))
    ).expanduser().resolve()
    if not result_path.is_file():
        errors.append("held-out result: missing")
        result = None
    else:
        try:
            result_digest = _sha256(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"held-out result: unreadable ({exc})")
            result = None
        else:
            if result_digest != attestation.get("heldout_result_sha256"):
                errors.append("held-out result: SHA-256 mismatch")

    if not isinstance(result, dict):
        if result is not None:
            errors.append("held-out result: not an object")
        return errors

    expect_equal("held-out.schema_version", result.get("schema_version"), 1)
    try:
        result_checkpoint = Path(str(result.get("checkpoint", ""))).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        result_checkpoint = Path("/")
    if result_checkpoint != checkpoint_path:
        errors.append("held-out.checkpoint: mismatch")

    requested = _finite_number(result.get("requested_episodes"))
    actual = _finite_number(result.get("actual_episodes"))
    if requested is None or not requested.is_integer() or requested < 2049:
        errors.append("held-out.requested_episodes: invalid")
    if (
        actual is None
        or not actual.is_integer()
        or actual < 2049
        or (requested is not None and actual < requested)
    ):
        errors.append("held-out.actual_episodes: invalid")

    condition = result.get("condition")
    expected_condition = {
        "seed": 42,
        "bars": 130,
        "num_envs": 128,
        "target_pattern": "mixed",
        "target_speed_mode": "uniform",
        "target_speed_min_mps": 0.3,
        "target_speed_max_mps": 1.5,
        "pursuer_max_speed_mps": 2.5,
        "oob_margin_m": 1.0,
        "episode_len_steps": 600,
        "goal_dist_min_m": 6.0,
        "goal_dist_max_m": 28.0,
        "full_goal_distribution": True,
        "fov_curriculum_saturated": True,
        "runtime_sim_config_class": "BaseSimConfig",
        "physics_dt_s": 0.01,
        "physics_substeps": 1,
        "physics_steps_per_rl_step": 10,
        "rl_step_dt_s": 0.1,
    }
    _check_contract(condition, expected_condition, "held-out.condition", errors)
    _check_contract(
        result.get("v2_evaluation_contract"),
        _RECOVERY_RESULT_CONTRACT,
        "held-out.v2_evaluation_contract",
        errors,
    )

    outcome = result.get("outcome")
    rates: Dict[str, Optional[float]] = {}
    count_values: Dict[str, Optional[float]] = {}
    if not isinstance(outcome, dict):
        errors.append("held-out.outcome: not an object")
    else:
        counts = []
        for name in ("captured", "crash", "timeout"):
            value = _finite_number(outcome.get(name))
            if value is None or not value.is_integer() or value < 0:
                errors.append(f"held-out.outcome.{name}: invalid")
            counts.append(value)
            count_values[name] = value
        for name in ("capture_rate", "crash_rate", "timeout_rate"):
            value = _finite_number(outcome.get(name))
            rates[name] = value
            if value is None or not 0.0 <= value <= 1.0:
                errors.append(f"held-out.outcome.{name}: invalid")
        if actual is not None and all(value is not None for value in counts):
            if not math.isclose(sum(counts), actual, rel_tol=0.0, abs_tol=0.0):
                errors.append("held-out.outcome: counts do not equal actual episodes")
            elif actual > 0:
                for count, rate_name in zip(
                    counts, ("capture_rate", "crash_rate", "timeout_rate")
                ):
                    rate = rates.get(rate_name)
                    if rate is not None and not math.isclose(
                        rate, count / actual, rel_tol=0.0, abs_tol=1e-12
                    ):
                        errors.append(f"held-out.outcome.{rate_name}: count mismatch")

    capture_rate = rates.get("capture_rate")
    crash_rate = rates.get("crash_rate")
    timeout_rate = rates.get("timeout_rate")
    if capture_rate is not None and capture_rate < 0.65:
        errors.append("held-out.outcome.capture_rate: below gate")
    if crash_rate is not None and crash_rate > 0.35:
        errors.append("held-out.outcome.crash_rate: above gate")
    if timeout_rate is not None and timeout_rate > 0.10:
        errors.append("held-out.outcome.timeout_rate: above gate")

    action = result.get("action")
    eval_oob: List[float] = []
    if not isinstance(action, dict):
        errors.append("held-out.action: not an object")
    else:
        expect_equal("held-out.action.policy", action.get("policy"), "squashed_gaussian")
        samples = _finite_number(action.get("samples"))
        if samples is None or not samples.is_integer() or samples <= 0:
            errors.append("held-out.action.samples: invalid")
        raw_oob = action.get("task_input_oob_rate")
        if not isinstance(raw_oob, list) or len(raw_oob) != 4:
            errors.append("held-out.action.task_input_oob_rate: expected four axes")
        else:
            for value in raw_oob:
                number = _finite_number(value)
                if number is None or not 0.0 <= number <= 1e-9:
                    errors.append("held-out.action.task_input_oob_rate: invalid")
                else:
                    eval_oob.append(number)

    mirror_fields = {
        "checkpoint_epoch": 9600,
        "bars": 130,
        "seed": 42,
        "requested_episodes": requested,
        "episodes": actual,
        "captured": count_values.get("captured"),
        "crash": count_values.get("crash"),
        "timeout": count_values.get("timeout"),
        "capture_rate": capture_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
    }
    for name, expected in mirror_fields.items():
        if expected is None or not _contract_value_matches(attestation.get(name), expected):
            errors.append(f"attestation.{name}: held-out mismatch")

    training_kl = _finite_number(attestation.get("training_max_kl"))
    training_oob = _finite_number(
        attestation.get("training_max_task_input_oob_rate")
    )
    attested_eval_oob = _finite_number(
        attestation.get("evaluation_max_task_input_oob_rate")
    )
    combined_oob = _finite_number(attestation.get("max_task_input_oob_rate"))
    if training_kl is None or not -1e-6 <= training_kl <= 0.04:
        errors.append("attestation.training_max_kl: invalid")
    for name, value in (
        ("training_max_task_input_oob_rate", training_oob),
        ("evaluation_max_task_input_oob_rate", attested_eval_oob),
        ("max_task_input_oob_rate", combined_oob),
    ):
        if value is None or not 0.0 <= value <= 1e-9:
            errors.append(f"attestation.{name}: invalid")
    if eval_oob and attested_eval_oob is not None and not math.isclose(
        attested_eval_oob, max(eval_oob), rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("attestation.evaluation_max_task_input_oob_rate: held-out mismatch")
    if (
        training_oob is not None
        and attested_eval_oob is not None
        and combined_oob is not None
        and not math.isclose(
            combined_oob,
            max(training_oob, attested_eval_oob),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        errors.append("attestation.max_task_input_oob_rate: component mismatch")
    for name in (
        "max_rollback_streak",
        "final_rollback_streak",
        "max_epoch_rollback",
        "max_rollback_total",
        "final_rollback_total",
    ):
        value = _finite_number(attestation.get(name))
        if value is None or value != 0.0:
            errors.append(f"attestation.{name}: expected zero")
    return errors


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(row: Dict[str, str], key: str) -> Optional[int]:
    value = _float(row, key)
    return int(value) if value is not None else None


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _mean(rows: Iterable[Dict[str, str]], key: str) -> Optional[float]:
    values = [value for row in rows if (value := _float(row, key)) is not None]
    return fmean(values) if values else None


def _training_process_exists() -> bool:
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "runner.py" in command and "--task navrl_task" in command and "--train" in command:
            return True
    return False


def _is_smoke_run(run_name: str) -> bool:
    """Keep short wiring checks in history without presenting them as research results."""
    lowered = run_name.lower()
    # The 100-epoch v2 recovery smoke is a real gated recovery stage and must become the latest
    # dashboard record when complete. Only one/few-epoch wiring checks stay out of Latest.
    if "v2-recover-smoke" in lowered:
        return False
    return any(
        marker in lowered
        for marker in ("smoke", "integration", "forced", "preflight")
    )


def _live_training_max_epochs(default: int = 12000) -> int:
    """Read the active NavRL runner's explicit epoch ceiling when available."""
    proc = Path("/proc")
    if not proc.is_dir():
        return default
    pattern = re.compile(r"(?:^|\s)--max_epochs\s+(\d+)(?:\s|$)")
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "runner.py" not in command or "--task navrl_task" not in command or "--train" not in command:
            continue
        if match := pattern.search(command):
            return int(match.group(1))
    return default


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize_run(csv_path: Path, *, is_active: bool) -> Dict[str, Any]:
    run_dir = csv_path.parents[1]
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"empty metrics CSV: {csv_path}")
    last = rows[-1]
    peak_capture = max(rows, key=lambda row: _float(row, "captured_rate") or -math.inf)
    peak_reward = max(rows, key=lambda row: _float(row, "mean_reward") or -math.inf)
    summary_path = run_dir / "aerial_run/run_summary.json"
    saved = {}
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))

    finalized_at = None if is_active else saved.get("finalized_at", _iso_mtime(csv_path))
    summary = {
        "run": run_dir.name,
        "finalized_at": finalized_at,
        "exit_reason": "running" if is_active else saved.get("exit_reason", "interrupted"),
        "epochs_logged": len(rows),
        "first_epoch": _int(rows[0], "epoch"),
        "last_epoch": _int(last, "epoch"),
        "is_navrl": True,
        "last_mean_reward": _float(last, "mean_reward"),
        "last_mean_episode_length": _float(last, "mean_episode_length"),
        "last_captured_rate": _float(last, "captured_rate"),
        "last_crash_rate": _float(last, "crash_rate"),
        "last_timeout_rate": _float(last, "timeout_rate"),
        "last_closest_approach_m": _float(last, "closest_approach_m"),
        "last_closest_min_m": _float(last, "closest_min_m"),
        "last_curriculum_max_m": _float(last, "curriculum_max_m"),
        "last_n_bars_active": _int(last, "n_bars_active"),
        "peak_captured_rate": _float(peak_capture, "captured_rate"),
        "peak_captured_epoch": _int(peak_capture, "epoch"),
        "peak_mean_reward": _float(peak_reward, "mean_reward"),
        "peak_epoch": _int(peak_reward, "epoch"),
        "reward_collapse": bool(saved.get("reward_collapse", False)),
    }
    # The generic peak-relative guard was disabled for density curricula, but this run is a
    # measured PPO collapse rather than a normal promotion drop.
    if run_dir.name.startswith("ppo_260730_1154_"):
        summary["reward_collapse"] = True
        summary["collapse_detail"] = (
            "KL crossed 0.04 at epoch 10276; latent means exploded and tail500 capture fell to 1.0%."
        )
    if run_dir.name.startswith("ppo_260731_2012_"):
        summary["reward_collapse"] = True
        summary["collapse_detail"] = (
            "Actor-update collapse at epoch 10836: KL peaked above 2.6, transformed entropy "
            "fell below -100, and capture reached 0%. The same-density guard fail-stopped it."
        )
    return summary


def _latest_barprobe() -> Dict[str, Optional[float]]:
    live_link = RL_ROOT / "train_session_logs/current_training.log"
    if not live_link.exists():
        return {"unique": None, "duplicate": None}
    pattern = re.compile(r"unique=([0-9.]+) duplicate=([0-9.]+)")
    matches = pattern.findall(live_link.read_text(encoding="utf-8", errors="ignore"))
    if not matches:
        return {"unique": None, "duplicate": None}
    unique, duplicate = matches[-1]
    return {"unique": float(unique), "duplicate": float(duplicate)}


def _corrected_density_curve() -> Dict[str, Any]:
    rows = []
    with CORRECTED_CURVE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return {
        "notes": [
            "Corrected bearing/chirality + bounded squashed-Gaussian backbone.",
            "Deterministic held-out evaluation of the 85-bar training policy; ~2049 episodes/cell.",
            "85-bar training plateau is 0.676 +/- 0.001; the 0.689 table cell is held-out evaluation.",
            "65-bar predecessor plateau was 0.678: equal competence at +31% training density.",
            "This curve used greedy +/-10 deg suppression. The active cluster-sector selector is a same-shape ablation and is not in this curve.",
        ],
        "trained_max_bars": 85,
        "rows": rows,
        "mtime": _iso_mtime(CORRECTED_CURVE_PATH),
    }


def _active_record(
    summary: Dict[str, Any], csv_path: Path, *, max_epochs: int
) -> Dict[str, Any]:
    rows = _load_rows(csv_path)
    tail = rows[-50:]
    age_min = max(0.0, (datetime.now(timezone.utc).timestamp() - csv_path.stat().st_mtime) / 60.0)
    return {
        "run": summary["run"],
        "is_live": True,
        "metrics_age_min": age_min,
        "epoch": summary["last_epoch"],
        "max_epochs": max_epochs,
        "epochs_logged": summary["epochs_logged"],
        "tail_epochs": len(tail),
        "captured_rate": _mean(tail, "captured_rate"),
        "crash_rate": _mean(tail, "crash_rate"),
        "timeout_rate": _mean(tail, "timeout_rate"),
        "mean_reward": _mean(tail, "mean_reward"),
        "closest_approach_m": _mean(tail, "closest_approach_m"),
        "n_bars_active": _mean(tail, "n_bars_active"),
        "curriculum_max_m": _mean(tail, "curriculum_max_m"),
    }


def _live_density_promotions() -> List[Dict[str, Any]]:
    live_link = RL_ROOT / "train_session_logs/current_training.log"
    if not live_link.exists():
        return []
    pattern = re.compile(
        r"density curriculum promoted \| bars (\d+) -> (\d+) "
        r"after (\d+) eps, capture=([0-9.]+)"
    )
    promotions = []
    for source, target, episodes, capture in pattern.findall(
        live_link.read_text(encoding="utf-8", errors="ignore")
    ):
        promotions.append(
            {
                "source": int(source),
                "target": int(target),
                "episodes": int(episodes),
                "capture": float(capture),
            }
        )
    return promotions


def _main_ttc_result(profile: str, arm: str) -> Optional[Dict[str, Any]]:
    """Load one hash/contract-checked held-out cell for the ep24000 TTC A/B."""

    path = (
        MAIN_TTC_RESULT_ROOT
        / f"navrl_v2_ep24000_ttc_{profile}_{arm}"
        / "205bars.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    condition = payload.get("condition", {})
    contract = payload.get("v2_evaluation_contract", {})
    outcome = payload.get("outcome", {})
    expected_selector = "cluster_sector" if arm == "baseline" else "ttc_sector"
    expected_profile = "main" if profile == "main" else "4gb"
    actual = int(payload.get("actual_episodes", -1))
    expected_sha = _MAIN_TTC_BASELINE_SHA256 if (profile, arm) == ("main", "baseline") else None
    problems = []
    if payload.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if int(payload.get("requested_episodes", -1)) != 2049 or actual < 2049:
        problems.append("requires requested/actual episodes >= 2049")
    if int(condition.get("bars", -1)) != 205 or int(condition.get("seed", -1)) != 42:
        problems.append("requires bars=205 and seed=42")
    if condition.get("action_selection") != "deterministic":
        problems.append("requires deterministic actions")
    if condition.get("reflection_mode") != "original":
        problems.append("requires original reflection mode")
    if condition.get("target_speed_mode") != "uniform":
        problems.append("requires uniform target speed")
    if contract.get("obstacle_selector") != expected_selector:
        problems.append(f"requires selector={expected_selector}")
    if contract.get("runtime_profile") != expected_profile:
        problems.append(f"requires runtime_profile={expected_profile}")
    counts = sum(int(outcome.get(key, -actual - 1)) for key in ("captured", "crash", "timeout"))
    if counts != actual:
        problems.append("outcome counts do not sum to actual episodes")
    if expected_sha and payload.get("checkpoint_sha256") != expected_sha:
        problems.append("baseline checkpoint SHA-256 mismatch")
    if payload.get("evaluated_checkpoint_snapshot_sha256") != payload.get("checkpoint_sha256"):
        problems.append("evaluated snapshot SHA-256 mismatch")
    if problems:
        raise RuntimeError(f"invalid ep24000 TTC A/B result {path}: " + "; ".join(problems))
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "episodes": actual,
        "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]),
        "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": float(payload.get("crash_causes", {}).get("bar_contact", 0)) / actual,
    }


def _riskcap_screen_result() -> Optional[Dict[str, Any]]:
    if not RISKCAP_SCREEN_PATH.is_file():
        return None
    payload = json.loads(RISKCAP_SCREEN_PATH.read_text(encoding="utf-8"))
    rows = {row.get("tag"): row for row in payload.get("rows", [])}
    candidate = rows.get("riskcap") or {}
    baseline = rows.get("off") or {}
    checks = {
        "schema": payload.get("schema_version") == 1,
        "experiment": payload.get("experiment") == "ep24000_seed44_riskcap_screen",
        "seed": int(payload.get("heldout_seed", -1)) == 44,
        "sha": payload.get("source_checkpoint_sha256") == _MAIN_TTC_SOURCE_SHA256,
        "go": payload.get("adaptive_go") is True and payload.get("selected_tag") == "riskcap",
        "candidate": candidate.get("screen_pass") is True and int(candidate.get("episodes", 0)) >= 2049,
        "baseline": int(baseline.get("episodes", 0)) >= 2049,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid riskcap screen {RISKCAP_SCREEN_PATH}: {', '.join(failed)}")
    return {"payload": payload, "baseline": baseline, "candidate": candidate}


def _speed_governor_screen_result() -> Optional[Dict[str, Any]]:
    if not SPEED_GOVERNOR_SCREEN_PATH.is_file():
        return None
    payload = json.loads(SPEED_GOVERNOR_SCREEN_PATH.read_text(encoding="utf-8"))
    rows = {row.get("tag"): row for row in payload.get("rows", [])}
    checks = {
        "schema": payload.get("schema_version") == 1,
        "experiment": payload.get("experiment") == "ep24000_speed_governor_screen",
        "sha": payload.get("source_checkpoint_sha256") == _MAIN_TTC_SOURCE_SHA256,
        "decision": payload.get("adaptive_go") is False and payload.get("selected_tag") is None,
        "cells": set(rows) == {"off", "fixed2p0", "fixed1p5", "clearance", "ttc"},
        "episodes": all(int(row.get("episodes", 0)) >= 2049 for row in rows.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"invalid speed-governor screen {SPEED_GOVERNOR_SCREEN_PATH}: {', '.join(failed)}"
        )
    return {"payload": payload, "rows": rows}


def _riskcap_post_result() -> Optional[Dict[str, Any]]:
    if not RISKCAP_POST_PATH.is_file():
        return None
    payload = json.loads(RISKCAP_POST_PATH.read_text(encoding="utf-8"))
    uniform = {row.get("tag"): row for row in payload.get("uniform_rows", [])}
    fixed = payload.get("fixed_speed_rows", [])
    checks = {
        "schema": payload.get("schema_version") == 1,
        "source_sha": payload.get("source_checkpoint_sha256") == _MAIN_TTC_SOURCE_SHA256,
        "trained_sha": payload.get("trained_checkpoint_sha256") == _RISKCAP_TRAINED_SHA256,
        "winner": payload.get("winner_kind") == "trained"
        and payload.get("winner_checkpoint_sha256") == _RISKCAP_TRAINED_SHA256,
        "gates": payload.get("mechanism_replication_pass") is True
        and payload.get("adaptation_pass") is True
        and payload.get("generalization_pass") is True,
        "uniform_cells": set(uniform)
        == {"uniform_off", "uniform_source_riskcap", "uniform_trained_riskcap"}
        and all(int(row.get("episodes", 0)) >= 2049 for row in uniform.values()),
        "fixed_cells": [float(row.get("target_speed_mps", -1)) for row in fixed] == [0.3, 0.9, 1.5]
        and all(row.get("direction_pass") is True for row in fixed),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid riskcap post-adaptation result: {', '.join(failed)}")
    return {"payload": payload, "uniform": uniform, "fixed": fixed}


def _v2_search_update(record: Dict[str, Any], *, is_live: bool) -> Dict[str, Any]:
    run_name = str(record.get("run", ""))
    if "v2-speedgov-ep24000-205bars-main-riskcap-s1" in run_name:
        complete_stop = _speed_governor_screen_result()
        if complete_stop is None:
            raise RuntimeError("riskcap training exists without the corrected five-cell screen")
        screen = _riskcap_screen_result()
        if screen is None:
            raise RuntimeError("riskcap training exists without its held-out authorization")
        baseline, candidate = screen["baseline"], screen["candidate"]
        epoch = int(record.get("epoch") if is_live else record.get("last_epoch") or 0)
        completed = not is_live and epoch >= 25000
        post_result = _riskcap_post_result()
        post = post_result["payload"] if post_result else None
        phase = (
            "TRAINING" if is_live else "FINAL PASS" if post and post.get("generalization_pass")
            else "FINAL FAIL" if post else "EVAL PENDING" if completed else "INCOMPLETE"
        )
        comparison = []
        if post:
            for tag, label in (
                ("uniform_off", "seed45 · canonical off"),
                ("uniform_source_riskcap", "seed45 · source + riskcap"),
                ("uniform_trained_riskcap", "seed45 · trained + riskcap"),
            ):
                row = post_result["uniform"][tag]
                comparison.append(
                    {
                        "label": label,
                        "bars": 205,
                        "capture": row["capture_rate"],
                        "unique": None,
                        "verdict": f"crash {row['crash_rate'] * 100:.2f}%; timeout {row['timeout_rate'] * 100:.2f}%",
                    }
                )
            for row in post_result["fixed"]:
                winner = row["winner"]
                comparison.append(
                    {
                        "label": f"final winner · fixed {row['target_speed_mps']:.1f} m/s",
                        "bars": 205,
                        "capture": winner["capture_rate"],
                        "unique": None,
                        "verdict": f"crash {winner['crash_rate'] * 100:.2f}%; {'PASS' if row['direction_pass'] else 'FAIL'} vs off",
                    }
                )
        else:
            comparison.extend(
                [
                    {
                        "label": "seed44 · canonical off",
                        "bars": 205,
                        "capture": baseline["capture_rate"],
                        "unique": None,
                        "verdict": f"crash {baseline['crash_rate'] * 100:.2f}%; timeout {baseline['timeout_rate'] * 100:.2f}%",
                    },
                    {
                        "label": "seed44 · minimum-intervention riskcap",
                        "bars": 205,
                        "capture": candidate["capture_rate"],
                        "unique": None,
                        "verdict": f"crash {candidate['crash_rate'] * 100:.2f}%; timeout {candidate['timeout_rate'] * 100:.2f}%",
                    },
                ]
            )
        adaptation_capture = post["adaptation_deltas"]["capture_rate"][0] if post else None
        mechanism_capture = post["mechanism_deltas"]["capture_rate"][0] if post else None
        mechanism_crash = post["mechanism_deltas"]["crash_rate"][0] if post else None
        return {
            "subtitle": f"2026-08-05 · R2b minimum-intervention riskcap · {phase.lower()}",
            "headline": (
                "Riskcap passed on an unseen seed and its matched adaptation is running."
                if is_live
                else "Riskcap training finished; unseen-seed post-adaptation evaluation is required."
                if completed and not post
                else f"Riskcap final generalization {'passed' if post and post.get('generalization_pass') else 'failed'}."
                if post
                else "Riskcap adaptation stopped before its matched budget."
            ),
            "summary": (
                f"On unseen seed45, sensor-only riskcap improved capture by {mechanism_capture * 100:+.2f} pp "
                f"and crash by {mechanism_crash * 100:+.2f} pp versus the frozen off policy. Exactly "
                f"1,000 adaptation epochs added {adaptation_capture * 100:+.2f} pp capture; the trained "
                "winner passed every seed46 fixed-speed direction check. The layer caps horizontal norm "
                "at 2.0 m/s inside 3 m, releases by 5 m, and never forces a stop."
                if post
                else f"The frozen ep24000 policy improved from {baseline['capture_rate'] * 100:.2f}% to "
                f"{candidate['capture_rate'] * 100:.2f}% capture on seed44 while crash fell "
                f"{(candidate['crash_rate'] - baseline['crash_rate']) * 100:+.2f} pp. The layer caps "
                "horizontal norm at 2.0 m/s inside 3 m and releases it to 3.535 m/s by 5 m; it never "
                "forces a stop. Training-tail values remain diagnostic until seed45/46 evaluation."
            ),
            "active_experiment": {
                **record,
                "is_live": is_live,
                "stage_status": phase,
                "epoch": epoch,
                "max_epochs": 25000,
                "warm_start_epoch": 24000,
                "adaptation_samples": max(0, epoch - 24000) * 32 * 128,
                "bars": 205,
                "density_curriculum": False,
                "selector": "cluster_sector",
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "speed_governor_mode": "riskcap",
                "speed_governor_fixed_mps": 2.0,
                "speed_governor_slow_m": 3.0,
                "speed_governor_release_m": 5.0,
                "screen_pass": True,
                "post_evaluation_complete": post is not None,
                "generalization_pass": post.get("generalization_pass") if post else None,
                "winner_checkpoint_sha256": post.get("winner_checkpoint_sha256") if post else None,
                "heldout_capture": post_result["uniform"]["uniform_trained_riskcap"]["capture_rate"] if post else None,
                "heldout_crash": post_result["uniform"]["uniform_trained_riskcap"]["crash_rate"] if post else None,
                "heldout_timeout": post_result["uniform"]["uniform_trained_riskcap"]["timeout_rate"] if post else None,
                "heldout_episodes": post_result["uniform"]["uniform_trained_riskcap"]["episodes"] if post else None,
            },
            "milestones": [
                {"label": "SEED44 SCREEN", "value": "PASS", "detail": f"capture {candidate['capture_rate'] * 100:.2f}% · crash {candidate['crash_rate'] * 100:.2f}%", "state": "pass"},
                {"label": "NO DEADLOCK", "value": f"{candidate['near_stop_rate'] * 100:.2f}%", "detail": f"intervention {candidate['intervention_rate'] * 100:.1f}% · timeout {candidate['timeout_rate'] * 100:.2f}%", "state": "pass"},
                {"label": "ADAPTATION", "value": f"capture {adaptation_capture * 100:+.2f} pp" if post else f"{max(0, epoch - 24000)} / 1000", "detail": "trained winner · 4.096M samples" if post else "128 envs · 4.096M samples · LR 5e-6", "state": "pass" if post or completed else "active" if is_live else "warn"},
                {"label": "FINAL HELD-OUT", "value": "PASS · 3/3 speeds" if post else "PENDING", "detail": "seed45 uniform + seed46 fixed-speed" if not post else "winner=trained; 0.3/0.9/1.5 m/s all improved", "state": "pass" if post and post.get("generalization_pass") else "warn" if post else "active"},
            ],
            "comparison": comparison,
            "gates": [
                {"label": "R2 complete-stop governors", "value": "REJECTED · 16.59–23.57% timeout"},
                {"label": "R2b seed44", "value": f"PASS · capture {(candidate['capture_rate'] - baseline['capture_rate']) * 100:+.2f} pp; crash {(candidate['crash_rate'] - baseline['crash_rate']) * 100:+.2f} pp"},
                {"label": "PPO adaptation", "value": "RUNNING" if is_live else "COMPLETE" if completed else "INCOMPLETE"},
                {"label": "unseen seed45/46", "value": "PASS" if post and post.get("generalization_pass") else "FAIL" if post else "PENDING"},
            ],
            "decision": (
                "Finish exactly 1,000 epochs, then compare off/source-riskcap/trained-riskcap on seed45."
                if is_live
                else "Run eval_navrl_v2_riskcap_postadapt.sh on the final ep25000 checkpoint."
                if completed and not post
                else "Freeze ep25000 + riskcap as the navigation/control candidate. Do not extend fixed-density PPO. First re-measure matched A/B/C cells under the corrected 600-step/source-manifest contract, then gate a better detector offline before navigation replay."
            ),
            "speed_governor_screen": complete_stop["payload"],
            "riskcap_screen": screen["payload"],
            "riskcap_postadapt": post,
        }
    if "v2-ep24000-205bars-" in run_name:
        match = re.search(r"v2-ep24000-205bars-(main|4gb)-(baseline|ttc)-s1", run_name)
        if not match:
            raise RuntimeError(f"unrecognized ep24000 TTC A/B run name: {run_name}")
        profile, arm = match.groups()
        selector = "cluster_sector" if arm == "baseline" else "ttc_sector"
        budget_epochs = 1000 if profile == "main" else 2000
        envs = 128 if profile == "main" else 64
        final_epoch = 24000 + budget_epochs
        epoch = int(record.get("epoch") if is_live else record.get("last_epoch") or 0)
        completed = not is_live and epoch >= final_epoch
        baseline_result = _main_ttc_result(profile, "baseline")
        ttc_result = _main_ttc_result(profile, "ttc")
        current_result = baseline_result if arm == "baseline" else ttc_result
        gate_complete = baseline_result is not None and ttc_result is not None
        capture_delta = (
            ttc_result["capture_rate"] - baseline_result["capture_rate"]
            if gate_complete
            else None
        )
        crash_delta = (
            ttc_result["crash_rate"] - baseline_result["crash_rate"]
            if gate_complete
            else None
        )
        gate_pass = bool(
            gate_complete and capture_delta >= 0.020 and crash_delta <= -0.020
        )

        if is_live:
            phase_value = f"{arm.upper()} TRAINING"
            headline = f"The fixed-205 {arm} arm is running; held-out comparison remains locked."
        elif not completed:
            phase_value = f"{arm.upper()} INCOMPLETE"
            headline = f"The fixed-205 {arm} arm stopped before its matched sample budget."
        elif current_result is None:
            phase_value = f"{arm.upper()} EVAL PENDING"
            headline = (
                f"The fixed-205 {arm} adaptation finished normally; its independent held-out score is the next gate."
            )
        elif arm == "baseline" and ttc_result is None:
            phase_value = "TTC ARM READY"
            headline = "The fixed-205 baseline score is frozen; the sample-matched TTC arm may now start."
        elif gate_complete:
            phase_value = "TTC PASS" if gate_pass else "TTC REJECT"
            headline = (
                "The main fixed-205 TTC selector passed both preregistered gates."
                if gate_pass
                else (
                    "The current fixed-205 TTC bundle failed both preregistered gates and remains "
                    "below the canonical ep24000 policy."
                )
            )
        else:
            phase_value = "TTC EVAL PENDING"
            headline = "The fixed-205 TTC adaptation finished; matched held-out evaluation is now required."

        comparison = [
            {
                "label": "frozen source ep24000 · held-out",
                "bars": 205,
                "capture": 0.724390243902439,
                "unique": None,
                "verdict": "n=2,050; context anchor only, not the adapted A/B baseline",
            }
        ]
        for result_arm, result in (("baseline", baseline_result), ("ttc", ttc_result)):
            if result is not None:
                comparison.append(
                    {
                        "label": f"main A/B · {result_arm} held-out",
                        "bars": 205,
                        "capture": result["capture_rate"],
                        "unique": None,
                        "verdict": (
                            f"n={result['episodes']:,}; crash {result['crash_rate'] * 100:.2f}%; "
                            f"bar contact {result['bar_contact_rate'] * 100:.2f}%"
                        ),
                    }
                )
        if current_result is None:
            comparison.append(
                {
                    "label": f"{arm} training final epoch · diagnostic only",
                    "bars": 205,
                    "capture": (
                        record.get("captured_rate") if is_live else record.get("last_captured_rate")
                    ),
                    "unique": None,
                    "verdict": "small on-policy termination sample; never use for the A/B decision",
                }
            )

        return {
            "subtitle": (
                f"{'2026-08-05' if ttc_result else '2026-08-04' if baseline_result else '2026-08-03'} · "
                f"fixed-205 main selector A/B · {phase_value.lower()}"
            ),
            "headline": headline,
            "summary": (
                f"Both arms start from the SHA-256-pinned ep24000 policy, keep 205 bars, seed 1, "
                f"LR 5e-6, action noise and task dynamics fixed, and receive {budget_epochs:,} epochs × "
                f"32 × {envs} = 4,096,000 adaptation samples. The selector mode is the only launcher "
                "knob, but its current semantics bundle ranking with candidate FOV: cluster_sector uses "
                "240° while ttc_sector uses 360°. This therefore tests the deployable representation "
                "bundle, not a pure ranking-only intervention. "
                "Training-tail capture is diagnostic; adoption uses seed-42 deterministic/original "
                "held-out evaluation with at least 2,049 episodes per arm."
            ),
            "active_experiment": {
                **record,
                "is_live": is_live,
                "ab_experiment": True,
                "ab_arm": arm,
                "ab_phase": phase_value,
                "ab_gate_complete": gate_complete,
                "ab_gate_pass": gate_pass if gate_complete else None,
                "representation_bundle": True,
                "baseline_effective_fov_deg": 240,
                "ttc_effective_fov_deg": 360,
                "pure_ranking_isolated": False,
                "epoch": epoch,
                "max_epochs": final_epoch,
                "bars": 205,
                "selector": selector,
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "density_curriculum": False,
                "arena_xy_m": 40,
                "arena_z_m": 3,
                "warm_start_epoch": 24000,
                "adaptation_samples": max(0, epoch - 24000) * 32 * envs,
                "source_checkpoint_sha256": _MAIN_TTC_SOURCE_SHA256,
                "final_checkpoint_sha256": (
                    current_result.get("checkpoint_sha256")
                    if completed and current_result
                    else _MAIN_TTC_BASELINE_SHA256
                    if (profile, arm, completed) == ("main", "baseline", True)
                    else None
                ),
                "training_tail_capture": (
                    record.get("captured_rate") if is_live else record.get("last_captured_rate")
                ),
                "training_tail_crash": (
                    record.get("crash_rate") if is_live else record.get("last_crash_rate")
                ),
                "heldout_complete": current_result is not None,
                "heldout_capture": current_result.get("capture_rate") if current_result else None,
                "heldout_crash": current_result.get("crash_rate") if current_result else None,
                "heldout_episodes": current_result.get("episodes") if current_result else None,
                "capture_delta": capture_delta,
                "crash_delta": crash_delta,
            },
            "milestones": [
                {
                    "label": "CURRENT ARM",
                    "value": phase_value,
                    "detail": (
                        f"selector={selector}; effective candidate FOV "
                        f"{'240' if selector == 'cluster_sector' else '360'}°; density fixed at 205"
                    ),
                    "state": (
                        "active"
                        if is_live or current_result is None
                        else "warn"
                        if gate_complete and not gate_pass
                        else "pass"
                        if completed
                        else "warn"
                    ),
                },
                {
                    "label": "TRAIN BUDGET",
                    "value": f"{max(0, epoch - 24000)} / {budget_epochs}",
                    "detail": f"{envs} envs · 4.096M matched samples",
                    "state": "pass" if completed else ("active" if is_live else "warn"),
                },
                {
                    "label": "HELD-OUT",
                    "value": (
                        f"{current_result['capture_rate'] * 100:.2f}%"
                        if current_result
                        else "PENDING"
                    ),
                    "detail": (
                        f"n={current_result['episodes']:,}; crash {current_result['crash_rate'] * 100:.2f}%"
                        if current_result
                        else "205 bars · seed42 · deterministic · original · ≥2,049 episodes"
                    ),
                    "state": "pass" if current_result else "active",
                },
                {
                    "label": "A/B DECISION",
                    "value": (
                        "PASS" if gate_pass else "REJECT"
                        if gate_complete
                        else "LOCKED"
                    ),
                    "detail": (
                        f"capture {capture_delta * 100:+.2f} pp · crash {crash_delta * 100:+.2f} pp"
                        if gate_complete
                        else "requires baseline + TTC held-out; +2pp capture and -2pp crash"
                    ),
                    "state": "pass" if gate_pass else ("warn" if gate_complete else "active"),
                },
            ],
            "comparison": comparison,
            "gates": [
                {
                    "label": "matched training budget",
                    "value": "PASS · 4.096M samples" if completed else "PENDING",
                },
                {
                    "label": "baseline held-out",
                    "value": "PASS · frozen" if baseline_result else "PENDING · TTC arm blocked",
                },
                {
                    "label": "capture delta",
                    "value": (
                        f"{'PASS' if capture_delta >= 0.020 else 'FAIL'} · {capture_delta * 100:+.2f} pp"
                        if gate_complete
                        else "PENDING · TTC - baseline ≥ +2.0pp"
                    ),
                },
                {
                    "label": "crash delta",
                    "value": (
                        f"{'PASS' if crash_delta <= -0.020 else 'FAIL'} · {crash_delta * 100:+.2f} pp"
                        if gate_complete
                        else "PENDING · TTC - baseline ≤ -2.0pp"
                    ),
                },
                {
                    "label": "canonical replacement floor",
                    "value": (
                        "PASS · beats ep24000 on capture and crash"
                        if gate_complete
                        and ttc_result["capture_rate"] >= 0.724390243902439
                        and ttc_result["crash_rate"] <= 0.25073170731707317
                        else "FAIL · keep ep24000 as deployment default"
                        if gate_complete
                        else "PENDING · capture ≥72.44%; crash ≤25.07%"
                    ),
                },
                {
                    "label": "experimental isolation",
                    "value": (
                        "BUNDLE · candidate FOV 240→360; pure ranking not isolated"
                    ),
                },
            ],
            "decision": (
                "Run the canonical held-out evaluation for the completed baseline and freeze its result. "
                "Do not start TTC training until that artifact exists."
                if completed and baseline_result is None and arm == "baseline"
                else "Run the TTC arm with the same launcher and sample budget; do not change any other knob."
                if baseline_result and arm == "baseline" and ttc_result is None
                else "Adopt TTC and move to replication only; do not resume an open-ended density curriculum."
                if gate_pass
                else (
                    "Close the current TTC-360 representation bundle as a negative result and keep ep24000 "
                    "as the canonical policy. Pure TTC ranking was not isolated: either screen a decoupled "
                    "TTC-240 arm first, or move to the preregistered R2 control-risk screen."
                )
                if gate_complete
                else "Finish the current arm, then evaluate its final checkpoint before any new training."
            ),
        }

    if (
        not is_live
        and run_name == "ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1"
        and LIMIT_AUDIT_PATH.is_file()
    ):
        audit = json.loads(LIMIT_AUDIT_PATH.read_text(encoding="utf-8"))
        checkpoint = audit.get("checkpoint", {})
        if (
            audit.get("schema_version") != 1
            or audit.get("status") != "training-stopped-core-audit-complete"
            or checkpoint.get("epoch") != 24000
            or checkpoint.get("sha256")
            != "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
        ):
            raise RuntimeError(f"invalid final v2 limit audit: {LIMIT_AUDIT_PATH}")

        deterministic = audit["action_mode_205"]["deterministic"]
        stochastic = audit["action_mode_205"]["stochastic"]
        difference = audit["action_mode_205"]["difference"]
        training = audit["training"]
        density_rows = audit["density_sweep_deterministic"]
        causal = None
        if CAUSAL_1TO3_PATH.is_file():
            causal = json.loads(CAUSAL_1TO3_PATH.read_text(encoding="utf-8"))
            if (
                causal.get("schema_version") != 1
                or causal.get("training_performed") is not False
                or causal.get("checkpoint_sha256") != checkpoint["sha256"]
                or causal.get("second_seed", {}).get("practical_replication") is not True
            ):
                raise RuntimeError(f"invalid v2 causal 1--3 result: {CAUSAL_1TO3_PATH}")
        mirror = causal.get("mirror", {}) if causal else {}
        seed_check = causal.get("second_seed", {}) if causal else {}
        action_pair = mirror.get("action_pair", {})
        bearing = mirror.get("initial_target_bearing", {})
        causal_done = causal is not None
        fixed_speed = None
        if FIXED_SPEED_PATH.is_file():
            fixed_speed = json.loads(FIXED_SPEED_PATH.read_text(encoding="utf-8"))
            speed_cells = fixed_speed.get("cells", [])
            if (
                fixed_speed.get("schema_version") != 1
                or fixed_speed.get("checkpoint_sha256") != checkpoint["sha256"]
                or [cell.get("target_speed_mps") for cell in speed_cells] != [0.3, 0.9, 1.5]
                or fixed_speed.get("high_minus_low_capture", {}).get(
                    "material_speed_sensitivity"
                )
                is not True
            ):
                raise RuntimeError(f"invalid v2 fixed-speed result: {FIXED_SPEED_PATH}")
        forgetting = None
        if FORGETTING_PATH.is_file():
            forgetting = json.loads(FORGETTING_PATH.read_text(encoding="utf-8"))
            comparisons = forgetting.get("comparisons", {})
            if (
                forgetting.get("schema_version") != 1
                or forgetting.get("checkpoints", {}).get("24000") != checkpoint["sha256"]
                or forgetting.get("material_forgetting_detected") is not False
                or comparisons.get("uniform", {}).get("verdict") != "improvement"
                or comparisons.get("fast1p5", {}).get("verdict") != "improvement"
            ):
                raise RuntimeError(f"invalid v2 forgetting result: {FORGETTING_PATH}")
        ttc_1650_rows = []
        if TTC_1650_PATH.is_file():
            with TTC_1650_PATH.open(encoding="utf-8", newline="") as stream:
                ttc_1650_rows = list(csv.DictReader(stream))
        ttc_1650 = {row.get("arm"): row for row in ttc_1650_rows}
        ttc_1650_pass = (
            set(ttc_1650) == {"baseline", "ttc"}
            and ttc_1650["ttc"].get("gate_pass") == "PASS"
            and float(ttc_1650["ttc"]["capture_pct"])
            - float(ttc_1650["baseline"]["capture_pct"])
            >= 2.0
            and float(ttc_1650["ttc"]["crash_pct"])
            - float(ttc_1650["baseline"]["crash_pct"])
            <= -2.0
        )
        causal_all_done = causal_done and fixed_speed is not None and forgetting is not None
        speed_delta = (
            fixed_speed["high_minus_low_capture"]["delta"] if fixed_speed else None
        )
        speed_bar_delta = (
            fixed_speed["high_minus_low_rates"]["bar_contact"] if fixed_speed else None
        )
        forgetting_uniform = (
            forgetting["comparisons"]["uniform"] if forgetting else {}
        )
        forgetting_fast = forgetting["comparisons"]["fast1p5"] if forgetting else {}
        comparison_rows = [
            {
                "label": f"held-out deterministic · {row['density_per_100m2']:.2f}/100m²",
                "bars": row["bars"],
                "capture": row["capture_rate"],
                "unique": None,
                "verdict": (
                    f"n={row['episodes']:,}; crash {row['crash_rate'] * 100:.2f}%; "
                    f"timeout {row['timeout_rate'] * 100:.2f}%"
                ),
            }
            for row in density_rows
        ]
        if causal_all_done:
            comparison_rows.extend(
                {
                    "label": f"ep24000 · fixed target {cell['target_speed_mps']:.1f} m/s",
                    "bars": 205,
                    "capture": cell["capture_rate"],
                    "unique": None,
                    "verdict": (
                        f"crash {cell['crash_rate'] * 100:.2f}%; "
                        f"bar contact {cell['bar_contact_rate'] * 100:.2f}%"
                    ),
                }
                for cell in fixed_speed["cells"]
            )
        return {
            "subtitle": (
                "2026-08-03 · v2 causal audit complete · fixed-205 TTC A/B ready"
                if causal_all_done and ttc_1650_pass
                else "2026-08-02 · v2 ep24000 causal checks 1–3 complete · fixed-speed checks pending"
                if causal_done
                else "2026-08-02 · v2 ep24000 core limit audit · causal checks pending"
            ),
            "headline": (
                "The 205-bar stage improved rather than forgot; high-speed bar contact is the next controlled bottleneck."
                if causal_all_done
                else "The 205-bar score replicates across seeds, but the policy has a strong learned turn-direction habit."
                if causal_done
                else "The unchanged 205-bar curriculum is closed; mirror and multi-condition evaluations come next."
            ),
            "summary": (
                f"After {training['epochs_at_205']:,} epochs and "
                f"{training['held_windows_at_205']} complete holds at 205 bars, the last seven "
                f"16,384-episode gates averaged {training['last_seven_gate_mean'] * 100:.2f}%. "
                f"The frozen ep24000 policy captures {deterministic['capture_rate'] * 100:.2f}% "
                f"with deterministic deployment actions versus {stochastic['capture_rate'] * 100:.2f}% "
                "with stochastic training actions. PPO divergence and disconnected geometry were rejected; "
                "bar contact on long, fast trajectories is the remaining failure. "
                + (
                    f"Seed 43 reproduced capture within {seed_check['capture_seed43_minus_seed42'] * 100:+.2f} pp. "
                    f"Aggregate mirror performance differed by only {mirror['differences']['capture_conjugate_minus_original'] * 100:+.2f} pp, "
                    f"but exact reflected observations disagreed in lateral action sign {action_pair['lateral_sign_mismatch_rate'] * 100:.1f}% of comparable samples."
                    if causal_done
                    else ""
                )
                + (
                    f" Fixed-speed capture falls {speed_delta * 100:.2f} pp from 0.3 to 1.5 m/s while bar contact rises {speed_bar_delta * 100:.2f} pp. "
                    f"There is no 205-stage forgetting: ep24000 improves over ep19100 by {forgetting_uniform['ep24000_minus_ep19100_capture'] * 100:+.2f} pp on U[0.3,1.5] and {forgetting_fast['ep24000_minus_ep19100_capture'] * 100:+.2f} pp at fixed 1.5 m/s."
                    if causal_all_done
                    else ""
                )
            ),
            "active_experiment": {
                **record,
                "is_live": False,
                "core_audit_complete": True,
                "causal_checks_pending": not causal_all_done,
                "causal_checks_1to3_complete": causal_done,
                "causal_checks_complete": causal_all_done,
                "fixed_speed_complete": fixed_speed is not None,
                "forgetting_complete": forgetting is not None,
                "ttc_1650_gate_pass": ttc_1650_pass,
                "next_training_authorized": causal_all_done and ttc_1650_pass,
                "stage_status": "closed",
                "epoch": 24000,
                "max_epochs": 24000,
                "bars": 205,
                "selector": "cluster_sector",
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "arena_xy_m": 40,
                "arena_z_m": 3,
                "density_curriculum": False,
                "frozen_checkpoint_sha256": checkpoint["sha256"],
                "deterministic_capture": deterministic["capture_rate"],
                "deterministic_crash": deterministic["crash_rate"],
                "deterministic_timeout": deterministic["timeout_rate"],
                "deterministic_episodes": deterministic["episodes"],
                "stochastic_capture": stochastic["capture_rate"],
                "stochastic_crash": stochastic["crash_rate"],
                "stochastic_episodes": stochastic["episodes"],
            },
            "milestones": [
                {
                    "label": "FINAL ARTIFACT",
                    "value": "ep24000 FROZEN",
                    "detail": f"SHA-256 {checkpoint['sha256'][:12]}…; no training active",
                    "state": "pass",
                },
                {
                    "label": "SPEED SENSITIVITY" if causal_all_done else "SEED REPLICATION" if causal_done else "205 · DEPLOY",
                    "value": (
                        f"{speed_delta * 100:+.2f} pp"
                        if causal_all_done
                        else f"Δ {seed_check['capture_seed43_minus_seed42'] * 100:+.2f} pp"
                        if causal_done
                        else f"{deterministic['capture_rate'] * 100:.2f}%"
                    ),
                    "detail": (
                        f"0.3→1.5 m/s; bar contact {speed_bar_delta * 100:+.2f} pp · material"
                        if causal_all_done
                        else f"seed42 {seed_check['seed42']['capture_rate'] * 100:.2f}% · "
                        f"seed43 {seed_check['seed43']['capture_rate'] * 100:.2f}% · PASS"
                        if causal_done
                        else f"n={deterministic['episodes']:,}; crash "
                        f"{deterministic['crash_rate'] * 100:.2f}%"
                    ),
                    "state": "pass",
                },
                {
                    "label": "205-STAGE FORGETTING" if causal_all_done else "MIRROR OUTCOME" if causal_done else "205 · TRAIN POLICY",
                    "value": (
                        "NO · IMPROVED"
                        if causal_all_done
                        else f"{mirror['differences']['capture_conjugate_minus_original'] * 100:+.2f} pp"
                        if causal_done
                        else f"{stochastic['capture_rate'] * 100:.2f}%"
                    ),
                    "detail": (
                        f"uniform {forgetting_uniform['ep24000_minus_ep19100_capture'] * 100:+.2f} pp · fast1.5 {forgetting_fast['ep24000_minus_ep19100_capture'] * 100:+.2f} pp"
                        if causal_all_done
                        else f"4,096/arm; initial-bearing Δ {bearing['positive_minus_negative_capture'] * 100:+.2f} pp; no outcome gap"
                        if causal_done
                        else f"n={stochastic['episodes']:,}; exploration costs "
                        f"{difference['capture_rate_difference_deterministic_minus_stochastic'] * 100:.2f} pp"
                    ),
                    "state": "pass" if causal_done else "warn",
                },
                {
                    "label": "1650 Ti TTC GATE" if causal_all_done else "ACTION EQUIVARIANCE" if causal_done else "NEXT",
                    "value": (
                        "PASS" if ttc_1650_pass else "MISSING"
                        if causal_all_done
                        else f"{action_pair['lateral_sign_mismatch_rate'] * 100:.1f}% MISMATCH"
                        if causal_done
                        else "MIRROR EVAL"
                    ),
                    "detail": (
                        "70 bars · capture +9.86 pp · crash -8.06 pp"
                        if causal_all_done and ttc_1650_pass
                        else "paired 4GB evidence not verified"
                        if causal_all_done
                        else f"lateral MAE {action_pair['mean_abs_error'][1]:.3f}; learned chirality, outcome-neutral in current symmetric arena"
                        if causal_done
                        else "evaluation only; frozen weights; no training"
                    ),
                    "state": "pass" if causal_all_done and ttc_1650_pass else "warn" if causal_done else "active",
                },
                {
                    "label": "NEXT CONTROLLED STEP",
                    "value": "MAIN 205 TTC A/B" if causal_all_done and ttc_1650_pass else "BLOCKED",
                    "detail": "run baseline first; then sample-matched ttc_sector arm"
                    if causal_all_done and ttc_1650_pass
                    else "complete causal and 4GB gates before training",
                    "state": "active" if causal_all_done and ttc_1650_pass else "warn",
                },
            ],
            "comparison": comparison_rows,
            "gates": [
                {"label": "curriculum continuation", "value": "CLOSED · more unchanged epochs rejected"},
                {"label": "PPO divergence", "value": "REJECTED · behavior KL < 0.04; no rollback/OOB"},
                {"label": "geometry disconnection", "value": "REJECTED · 99.83% random-pair connectivity @0.2m"},
                {
                    "label": "mirror + second seed",
                    "value": (
                        "COMPLETE · outcome symmetric, action non-equivariant"
                        if causal_done
                        else "PENDING · frozen checkpoint, no learning"
                    ),
                },
                {
                    "label": "fixed-speed + forgetting",
                    "value": "COMPLETE · speed-sensitive; no forgetting"
                    if causal_all_done
                    else "PENDING · evaluation only",
                },
                {
                    "label": "1650 Ti TTC transfer",
                    "value": "PASS · capture +9.86 pp; crash -8.06 pp"
                    if ttc_1650_pass
                    else "PENDING",
                },
                {
                    "label": "next training",
                    "value": "READY · fixed-205 main baseline → TTC"
                    if causal_all_done and ttc_1650_pass
                    else "BLOCKED · causal checks and 1650 Ti gate pending",
                },
            ],
            "decision": (
                "Do not resume the open-ended 205-bar curriculum. Preserve ep24000 as the canonical "
                "artifact. "
                + (
                    "All frozen-policy causal checks are complete. The 205 stage improved both uniform and fixed-1.5 performance, so replay for catastrophic forgetting is not the next intervention. "
                    "Speed sensitivity is material and paid almost entirely in bar contact; the 1650 Ti TTC selector transfer gate also passed. "
                    "Authorize the sample-matched fixed-205 main baseline arm first, then the TTC arm. Keep reflection/RMS regularization and learned-detector work as separate later experiments. "
                    if causal_all_done and ttc_1650_pass
                    else "Mirror and second-seed evaluation are complete: success is seed-stable and outcome-symmetric, "
                    "but the controller is not action-equivariant. Finish the fixed-speed and forgetting checks before authorizing main training. "
                    if causal_done
                    else "Evaluate original/mirrored layouts without updating the policy, then finish "
                    "the second-seed and fixed-speed checks. "
                )
            ),
            "limit_audit": audit,
            "causal_1to3": causal,
            "fixed_speed": fixed_speed,
            "forgetting": forgetting,
            "ttc_1650": ttc_1650_rows,
        }

    if "v2-ttc-" in run_name:
        arm = "ttc" if "v2-ttc-ttc-" in run_name else "baseline"
        selector = "ttc_sector" if arm == "ttc" else "cluster_sector"
        epoch = int(record.get("epoch") if is_live else record.get("last_epoch") or 0)
        bars = int(
            round(record.get("n_bars_active") or record.get("last_n_bars_active") or 70)
        )
        completed = not is_live and epoch >= 5250
        capture_tail = (
            record.get("captured_rate") if is_live else record.get("last_captured_rate")
        )
        crash_tail = record.get("crash_rate") if is_live else record.get("last_crash_rate")
        return {
            "subtitle": f"2026-08-01 · fixed-70 selector A/B · {arm} arm",
            "headline": (
                f"The {arm} arm is running at a fixed 70 bars with {selector}."
                if is_live
                else (
                    f"The {arm} arm finished its matched 4.1M-step adaptation budget."
                    if completed
                    else f"The {arm} arm stopped before its matched adaptation budget completed."
                )
            ),
            "summary": (
                "This is not a density curriculum. Both arms start from the SHA-256-pinned ep3250 "
                "70-bar checkpoint, keep every task/action setting fixed, and differ only in how "
                "eight obstacle tokens are ranked. Training-tail rates are diagnostics; the "
                "preregistered decision uses matched held-out evaluations of both final checkpoints."
            ),
            "active_experiment": {
                **record,
                "is_live": is_live,
                "epoch": epoch,
                "max_epochs": 5250,
                "bars": bars,
                "selector": selector,
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "ab_arm": arm,
                "density_curriculum": False,
                "arena_xy_m": 40,
                "arena_z_m": 3,
                "warm_start_epoch": 3250,
                "adaptation_steps": max(0, epoch - 3250) * 2048,
            },
            "milestones": [
                {
                    "label": "A/B ARM",
                    "value": arm.upper(),
                    "detail": f"selector={selector}; all other variables fixed",
                    "state": "active" if is_live else ("pass" if completed else "warn"),
                },
                {
                    "label": "DENSITY",
                    "value": "70 FIXED",
                    "detail": "promotion disabled for the entire comparison",
                    "state": "pass",
                },
                {
                    "label": "BUDGET",
                    "value": f"{max(0, epoch - 3250)} / 2000 epochs",
                    "detail": "64 envs · 4.1M adaptation samples",
                    "state": "pass" if completed else "active",
                },
                {
                    "label": "DECISION",
                    "value": "HELD-OUT A/B",
                    "detail": "capture +2pp and crash -2pp required together",
                    "state": "active",
                },
            ],
            "comparison": [
                {
                    "label": f"{arm} training tail",
                    "bars": bars,
                    "capture": capture_tail,
                    "unique": None,
                    "verdict": (
                        f"crash={crash_tail:.3f}; diagnostic only"
                        if isinstance(crash_tail, (int, float))
                        else "diagnostic only; held-out result pending"
                    ),
                }
            ],
            "gates": [
                {"label": "single experimental variable", "value": f"PASS · {selector}"},
                {"label": "density curriculum", "value": "OFF · 70 bars fixed"},
                {"label": "capture delta", "value": "PENDING · TTC - baseline ≥ +2.0pp"},
                {"label": "crash delta", "value": "PENDING · TTC - baseline ≤ -2.0pp"},
            ],
            "decision": (
                "Finish both arms, then evaluate their final checkpoints at the same 70-bar "
                "condition. Adopt TTC ranking only if both preregistered gates pass."
            ),
        }

    if "v2-recover-smoke" in run_name:
        epoch = int(record.get("epoch") if is_live else record.get("last_epoch") or 0)
        completed_epochs = max(0, epoch - 9500)
        recovery_run_dir = RUNS_ROOT / run_name
        normal_marker_epoch = _normal_completion_marker_epoch(recovery_run_dir)
        complete = not is_live and epoch == 9600 and normal_marker_epoch == 9600
        completion_anomaly = not is_live and epoch >= 9600 and not complete
        attestation_path = recovery_run_dir / ".navrl_v2_recovery_eval_pass.json"
        attested = False
        attestation_errors: List[str] = []
        if complete:
            attestation_errors = _validate_recovery_attestation(recovery_run_dir)
            attested = not attestation_errors
        attestation_present = attestation_path.is_file()
        bars = int(
            round(record.get("n_bars_active") or record.get("last_n_bars_active") or 130)
        )
        return {
            "subtitle": "2026-08-01 · transactional PPO recovery · fixed-130 safety smoke",
            "headline": (
                f"Recovery smoke is running: {completed_epochs}/100 epochs at {bars} bars."
                if is_live
                else (
                    "Recovery smoke and its hash-bound held-out gate passed this snapshot's verification."
                    if attested
                    else "Recovery reached its epoch budget without the exact normal-completion marker; curriculum remains blocked."
                    if completion_anomaly
                    else "The recovery attestation failed artifact or contract verification; curriculum remains blocked."
                    if complete and attestation_present
                    else "The 100-epoch recovery smoke completed; held-out evaluation is the next gate."
                    if complete
                    else f"Recovery smoke stopped early after {completed_epochs}/100 epochs."
                )
            ),
            "summary": (
                "This stage starts from the SHA-256-pinned ep9500 last-known-good policy, freezes "
                "density at 130 bars, preserves Adam moments, uses LR 5e-6, and atomically rolls "
                "back actor and central critic together whenever immutable-behavior KL exceeds "
                "0.04 or a model, RMS, optimizer, scaler, loss, or output becomes non-finite. "
                "Promotion evidence is recomputed from the real 100-epoch TensorBoard window and "
                "an evaluator receipt bound to an immutable checkpoint snapshot, nonce, log, and "
                "measured physics under main/base_sim/128 envs, full 6–28 m goals, and final FOV."
            ),
            "active_experiment": {
                **record,
                "is_live": is_live,
                "epoch": epoch,
                "max_epochs": 9600,
                "bars": bars,
                "selector": "cluster_sector",
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "arena_xy_m": 40,
                "arena_z_m": 3,
                "recovery_checkpoint_epoch": 9500,
                "recovery_lr": 5e-6,
                "rollback_kl_gate": 0.04,
                "recovery_attestation_valid": attested,
                "recovery_attestation_errors": attestation_errors,
                "recovery_normal_completion_marker_valid": complete,
            },
            "milestones": [
                {
                    "label": "RECOVERY SMOKE",
                    "value": f"{completed_epochs} / 100",
                    "detail": (
                        "fixed 130 bars; exact normal marker epoch=9600"
                        if complete
                        else "fixed 130 bars; normal marker epoch=9600 required"
                    ),
                    "state": "pass" if complete else ("active" if is_live else "warn"),
                },
                {
                    "label": "SOURCE",
                    "value": "ep9500",
                    "detail": "audited LKG; SHA-256 pinned",
                    "state": "pass",
                },
                {
                    "label": "PPO COMMIT",
                    "value": "ATOMIC",
                    "detail": "actor+central model/RMS/Adam/scaler/output/loss + KL audit",
                    "state": "pass",
                },
                {
                    "label": "NEXT",
                    "value": (
                        "CURRICULUM"
                        if attested
                        else "DIAGNOSE EXIT"
                        if completion_anomaly
                        else "RE-EVALUATE"
                        if complete and attestation_present
                        else "HELD-OUT EVAL"
                        if complete
                        else "FINISH SMOKE"
                    ),
                    "detail": "curriculum requires the checkpoint-bound evaluation PASS",
                    "state": "pass" if attested else "active",
                },
            ],
            "comparison": [],
            "gates": [
                {
                    "label": "100 recovery epochs + normal exit",
                    "value": (
                        "PASS · marker epoch=9600"
                        if complete
                        else "INVALID · marker missing/mismatched"
                        if completion_anomaly
                        else "PENDING"
                    ),
                },
                {"label": "post-update transaction", "value": "ENFORCED · KL 0.04 + finite state"},
                {
                    "label": "held-out 130-bar attestation",
                    "value": (
                        "PASS"
                        if attested
                        else "INVALID · artifact/contract mismatch"
                        if complete and attestation_present
                        else "PENDING · 2,049 episodes"
                    ),
                },
                {
                    "label": "curriculum resume",
                    "value": (
                        "VERIFIED IN THIS SNAPSHOT · LAUNCHER RECHECKS"
                        if attested
                        else "BLOCKED UNTIL HASH-BOUND PASS"
                    ),
                },
            ],
            "decision": (
                "The ep9600 evidence is verified in this static snapshot. Resume only through the safe launcher, which re-verifies the live artifacts before training."
                if attested
                else "Do not evaluate or resume: the run reached epoch 9600 without the exact normal-completion marker. Diagnose the exit and rerun the smoke."
                if completion_anomaly
                else "Do not resume: the saved attestation, checkpoint, or held-out artifact failed strict verification. Re-run the canonical recovery evaluation."
                if complete and attestation_present
                else "Evaluate the final ep9600 checkpoint at 130 bars on the pinned U[0.3,1.5] target "
                "distribution. Resume the curriculum only if KL, saturation, capture, and crash remain healthy."
                if complete
                else "Let this fixed-density stage finish; do not use an intermediate checkpoint to resume the curriculum."
            ),
        }

    if (
        not is_live
        and run_name.startswith("ppo_260731_2012_")
    ):
        return {
            "subtitle": "2026-08-01 · v2 PPO/FOV/provenance fixes audited · recovery smoke pending",
            "headline": "The 145-bar run did not hit a proven density ceiling; its PPO actor update collapsed.",
            "summary": (
                "The run fail-stopped at epoch 10,836 after KL rose to 2.69 and capture fell to 0%. "
                "The old gate skipped later minibatches but could not undo an already committed update, "
                "then rebased its reference onto that rejected policy. PPO epochs are now transactional: "
                "actor and asymmetric central-critic model, RMS, Adam and AMP state are restored against "
                "immutable rollout-policy KL. A forced rollback restored every actor/critic tensor, value "
                "statistic and Adam moment/step exactly; a normal update audited at KL 0.0063 under the 0.04 gate."
                " The recovery gate now independently rebuilds its result from the exact checkpoint, "
                "held-out JSON, evaluator receipt, and TensorBoard window; it binds an immutable "
                "checkpoint snapshot, nonce, log and measured physics. The previously inert general-spawn "
                "FOV curriculum now aligns only initial yaw while keeping target directions unbiased. "
                "Rollback counters/LR are durable across fail-stop resumes."
            ),
            "active_experiment": {
                **record,
                "is_live": False,
                "epoch": record.get("last_epoch"),
                "max_epochs": 30000,
                "bars": 145,
                "selector": "cluster_sector",
                "cluster_gap_m": 0.45,
                "sectors": 8,
                "arena_xy_m": 40,
                "arena_z_m": 3,
                "recovery_checkpoint_epoch": 9500,
                "recovery_bars": 130,
                "recovery_lr": 5e-6,
                "rollback_kl_gate": 0.04,
                "heldout_capture_130": 0.7470817120622568,
                "heldout_crash_130": 0.2529182879377432,
                "heldout_episodes": 257,
            },
            "milestones": [
                {
                    "label": "ROOT CAUSE",
                    "value": "ACTOR UPDATE",
                    "detail": "moving KL reference + no model/Adam/RMS rollback",
                    "state": "pass",
                },
                {
                    "label": "FORCED ROLLBACK",
                    "value": "EXACT",
                    "detail": "actor+central model/RMS/Adam exact; actor LR backoff only",
                    "state": "pass",
                },
                {
                    "label": "130-BAR HEALTH",
                    "value": "74.71%",
                    "detail": "257 held-out episodes; crash 25.29%; OOB 0",
                    "state": "pass",
                },
                {
                    "label": "NEXT",
                    "value": "100 EPOCH",
                    "detail": "fixed-130 recovery smoke before curriculum resume",
                    "state": "active",
                },
            ],
            "comparison": [
                {
                    "label": "ep9500 LKG + one safe update · held-out",
                    "bars": 130,
                    "capture": 0.7470817120622568,
                    "unique": None,
                    "verdict": "257 episodes; crash 25.29%; timeout 0%",
                },
                {
                    "label": "ep10824+ collapsed actor · training tail",
                    "bars": 145,
                    "capture": 0.0,
                    "unique": None,
                    "verdict": "invalid for density-ceiling inference; crash 100%",
                },
            ],
            "gates": [
                {"label": "epoch transaction", "value": "PASS · actor+central/RMS/Adam/AMP atomic restore"},
                {"label": "post-update KL", "value": "PASS · 0.0063 < 0.04"},
                {"label": "evaluation contract", "value": "PASS · receipt/snapshot + measured sim/goal/FOV"},
                {"label": "145-bar ceiling", "value": "UNRESOLVED · contaminated by actor drift"},
            ],
            "decision": (
                "Resume from last_gen_ppo_ep_9500, not ep10800. Run the fixed-130 100-epoch safety "
                "smoke at 5e-6 first; only its final checkpoint with a 2,049-episode hash-bound "
                "held-out PASS attestation may re-enter the density curriculum. "
                "Re-measure 145 bars before claiming a representation or geometric ceiling."
            ),
        }

    promotions = _live_density_promotions()
    bars = int(
        round(record.get("n_bars_active") or record.get("last_n_bars_active") or 70)
    )
    capture_tail = (
        record.get("captured_rate")
        if is_live
        else record.get("last_captured_rate")
    )
    epoch = record.get("epoch") if is_live else record.get("last_epoch")
    run_state = "running" if is_live else "paused snapshot"
    comparison = [
        {
            "label": f"{item['source']} → {item['target']} promotion",
            "bars": item["target"],
            "capture": item["capture"],
            "unique": None,
            "verdict": f"PASS over {item['episodes']:,} episodes",
        }
        for item in promotions
    ]
    comparison.append(
        {
            "label": "current stage · rolling tail" if is_live else "latest stopped epoch",
            "bars": bars,
            "capture": capture_tail,
            "unique": None,
            "verdict": (
                "live diagnostic only; not the 16,384-episode promotion gate"
                if is_live
                else "stopped-run diagnostic; not a held-out result"
            ),
        }
    )
    promotion_text = " → ".join(
        [str(promotions[0]["source"])] + [str(item["target"]) for item in promotions]
    ) if promotions else str(bars)
    gate_captures = ", ".join(f"{item['capture'] * 100:.1f}%" for item in promotions)
    tail_text = f"{capture_tail * 100:.1f}%" if capture_tail is not None else "pending"
    return {
        "subtitle": f"2026-07-31 · v2 search-arena density curriculum · {run_state}",
        "headline": (
            f"Task-v2 training is live at {bars} bars after {len(promotions)} promotions."
            if is_live
            else f"Task-v2 training is paused at epoch {epoch}, {bars} bars."
        ),
        "summary": (
            f"The 40 m search-arena run has advanced {promotion_text}; completed promotion-window "
            f"capture values are {gate_captures or 'not yet available'}. The displayed capture "
            f"is {tail_text}, which is diagnostic only and must not be mistaken for the 16,384-episode "
            "promotion gate or a held-out result."
        ),
        "active_experiment": {
            **record,
            "is_live": is_live,
            "epoch": epoch,
            "max_epochs": record.get("max_epochs") or 30000,
            "bars": bars,
            "selector": "cluster_sector",
            "cluster_gap_m": 0.45,
            "sectors": 8,
            "arena_xy_m": 40,
            "arena_z_m": 3,
            "density_final": 300,
            "density_step": 15,
            "density_threshold": 0.70,
            "density_window_episodes": 16384,
        },
        "milestones": [
            {
                "label": "DENSITY",
                "value": f"{bars} / 300",
                "detail": f"self-paced +15 bars; chain {promotion_text}",
                "state": "active",
            },
            {
                "label": "PROMOTIONS",
                "value": str(len(promotions)),
                "detail": f"window capture {gate_captures or 'pending'}",
                "state": "pass" if promotions else "active",
            },
            {
                "label": "PROVENANCE",
                "value": "9 / 9",
                "detail": "arena, placement, episode and full-width bar-band contract saved",
                "state": "pass",
            },
            {
                "label": "1650 Ti · 4GB",
                "value": "CONDITIONAL",
                "detail": "64-env path fits the batch; real-card 8-epoch smoke still required",
                "state": "warn",
            },
        ],
        "comparison": comparison,
        "gates": [
            {
                "label": "density curriculum",
                "value": f"{'RUNNING' if is_live else 'PAUSED'} · {promotion_text}",
            },
            {"label": "collapse safety", "value": "PASS · atomic PPO rollback + NaN/Inf fail-fast"},
            {"label": "evaluation contract", "value": "PASS · z/gap/touch/bar-band enforced"},
            {"label": "1650 Ti", "value": "SMOKE REQUIRED · recommend free VRAM ≥3.6–3.7 GiB"},
        ],
        "decision": (
            "Keep the current process running, but do not interpret the rolling tail as a promotion "
            "decision." if is_live else
            "The run is paused. Validate the revised speed ramp and density dwell state in a short "
            "smoke before starting the next full training run."
        ) + " Treat the 4GB launcher as provisional until it passes an actual 1650 Ti smoke run.",
    }


def _ref5in_p2_update() -> Optional[Dict[str, Any]]:
    """Use the current ref5in fail-closed chain when its attestation exists.

    Older dashboard branches describe useful historical experiments, but selecting them merely
    because the newest full-budget run predates P1/P2 made the public page look a week stale.
    """
    attestation_path = ROOT / "results/navrl_ref5in_p2_seed313/attestation.json"
    if not attestation_path.is_file():
        return None
    try:
        proof = json.loads(attestation_path.read_text(encoding="utf-8"))
        cell = proof["decision_cell"]
        outcome = cell["outcome"]
        condition = cell["condition"]
        checks = proof["checks"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if proof.get("scope") != "ref5in_p2_heldout_70bar_decision":
        return None
    capture = float(outcome["capture_rate"])
    crash = float(outcome["crash_rate"])
    timeout = float(outcome["timeout_rate"])
    episodes = int(outcome["captured"]) + int(outcome["crash"]) + int(outcome["timeout"])
    diagnostic_path = ROOT / "results/navrl_ref5in_outcome_diagnostic_v2_seed317/summary.json"
    diagnostic = None
    if diagnostic_path.is_file():
        try:
            candidate = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if (
                candidate.get("scope") == "post_p2_descriptive_outcome_strata_v2_seed317"
                and candidate.get("decision_authority") == "none"
                and candidate.get("p3_unlocked") is False
                and candidate.get("behavioral_parity_with_v1") is True
            ):
                diagnostic = candidate
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            diagnostic = None
    experiment = {
        "ref5in_p2": True,
        "core_audit_complete": True,
        "is_live": False,
        "run": "ref5in P2 · seed313",
        "epoch": 900,
        "bars": int(condition["bars"]),
        "deterministic_capture": capture,
        "deterministic_crash": crash,
        "deterministic_timeout": timeout,
        "deterministic_episodes": episodes,
        "heldout_capture": capture,
        "heldout_crash": crash,
        "heldout_timeout": timeout,
        "heldout_episodes": episodes,
        "p1c_verdict": "PASS",
        "p2_verdict": str(proof.get("verdict", "UNKNOWN")),
        "p3_unlocked": proof.get("unlocks") == "manual_p3_seed211_only",
        "robot_name": condition.get("robot_name"),
        "checkpoint_sha256": proof.get("p1c", {}).get("checkpoint_sha256"),
        "runtime_map_sha256": proof.get("evaluation_runtime", {}).get("runtime_map_sha256"),
        "diagnostic_complete": diagnostic is not None,
    }
    diagnostic_milestone = {
        "label": "POST-P2 DIAGNOSTIC",
        "value": "PENDING",
        "detail": "outcome-aware distance/pattern measurement not finalized",
        "state": "warn",
    }
    diagnostic_comparison = None
    if diagnostic is not None:
        distance = diagnostic["strata"]["distance"]
        joint = diagnostic["strata"]["distance_by_pattern"]
        diagnostic_milestone = {
            "label": "POST-P2 DIAGNOSTIC",
            "value": "COMPLETE · descriptive",
            "detail": (
                f"q3 timeout {distance['q3']['timeout_rate']*100:.2f}% · "
                f"CV {joint['q3']['cv']['timeout_rate']*100:.2f}% vs "
                f"waypoint {joint['q3']['waypoint']['timeout_rate']*100:.2f}%"
            ),
            "state": "pass",
        }
        diagnostic_comparison = {
            "label": "diagnostic q3 · 22.5–28 m",
            "bars": 70,
            "capture": distance["q3"]["capture_rate"],
            "unique": None,
            "verdict": (
                f"descriptive only · crash {distance['q3']['crash_rate']*100:.2f}% · "
                f"timeout {distance['q3']['timeout_rate']*100:.2f}%"
            ),
        }
    headline = "The ref5in policy learned, but missed the held-out timeout ceiling by 12 episodes."
    summary = (
        f"P1c passed every engineering gate. In the preregistered seed-313 held-out cell, "
        f"capture/crash/timeout were {capture*100:.2f}/{crash*100:.2f}/{timeout*100:.2f}% "
        f"over {episodes:,} episodes. Capture and crash passed, but 114 timeouts exceeded the "
        "5% ceiling (maximum 102), so P3 remains blocked."
    )
    decision = (
        "Do not start P3 or relax the timeout gate after seeing the result. Add outcome-aware "
        "distance strata, then run a separately labelled diagnostic evaluation to distinguish "
        "long-range timeout from contact and arena-boundary failure."
    )
    if diagnostic is not None:
        headline = "Long-range CV pursuit, not target speed alone, is the clearest ref5in bottleneck."
        summary += (
            " A separate seed-317 diagnostic preserved the P2 decision but localized the failure: "
            "capture fell 78.05→55.94% from the nearest to farthest distance bin, while timeout "
            "rose 0.06→14.97%. In the farthest bin, CV timeout was 22.16% versus 7.96% for waypoint."
        )
        decision = (
            "P2 remains a strict FAIL and P3 remains blocked. Treat longer episode time as a "
            "measurement ablation, not a fix: first test whether additional saturated-distance "
            "CV exposure removes the q3 timeout without increasing contact or OOB crashes."
        )
    return {
        "subtitle": "2026-08-13 · ref5in P1c PASS → held-out P2 strict FAIL",
        "headline": headline,
        "summary": summary,
        "active_experiment": experiment,
        "milestones": [
            {"label": "PLATFORM P0", "value": "PASS · 26/26 + 21/21", "detail": "repository consistency + same-controller simulator gate", "state": "pass"},
            {"label": "P1c ENGINEERING", "value": "PASS", "detail": "72.77/23.94/3.30% · applied [6,28] m · rollback 0", "state": "pass"},
            {"label": "P2 HELD-OUT", "value": "STRICT FAIL", "detail": "timeout 114/2,049 = 5.56% > 5%", "state": "warn"},
            diagnostic_milestone,
            {"label": "P3 FULL BUDGET", "value": "BLOCKED", "detail": "seed 211 was not started", "state": "warn"},
        ],
        "comparison": [
            {"label": "P1c · on-policy last 100", "bars": 70, "capture": 0.7276812463, "unique": None, "verdict": "engineering-only · 3,338 episodes"},
            {"label": "P2 · held-out seed313", "bars": 70, "capture": capture, "unique": None, "verdict": f"strict FAIL · crash {crash*100:.2f}% · timeout {timeout*100:.2f}%"},
        ] + ([diagnostic_comparison] if diagnostic_comparison is not None else []),
        "gates": [
            {"label": "capture", "value": f"PASS · {capture*100:.2f}% ≥ 65%"},
            {"label": "crash", "value": f"PASS · {crash*100:.2f}% ≤ 33%"},
            {"label": "timeout", "value": f"FAIL · {timeout*100:.2f}% > 5%"},
            {"label": "provenance", "value": "PASS · P1/current/eval runtime maps identical"},
        ],
        "decision": decision,
    }


DETECTION_RANGE_STAGE1_PATH = (
    ROOT / "results/navrl_ref5in_detection_range_stage1_s457/summary.json"
)


def _detection_range_stage1_result() -> Optional[Dict[str, Any]]:
    """Load the completed two-arm screen, failing closed on its decision contract."""

    if not DETECTION_RANGE_STAGE1_PATH.is_file():
        return None
    payload = json.loads(DETECTION_RANGE_STAGE1_PATH.read_text(encoding="utf-8"))
    arms = payload.get("arms") or {}
    problems = []
    if payload.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if payload.get("verdict") != "RANGE_INCONCLUSIVE_AT_THIS_BUDGET":
        problems.append("unexpected verdict")
    if payload.get("stage2_authorised") is not False:
        problems.append("stage2 must remain unauthorised")
    if set(arms) != {"clip20", "clip28"}:
        problems.append("requires exactly clip20 and clip28")
    gates = payload.get("quality_gates") or {}
    if len(gates) != 17 or any(gate.get("passed") is not True for gate in gates.values()):
        problems.append("requires all 17 quality gates")

    rows = {}
    for name, expected_range in (("clip20", 20.0), ("clip28", 28.0)):
        arm = arms.get(name) or {}
        condition = arm.get("condition") or {}
        measurements = arm.get("measurements") or {}
        outcome = measurements.get("outcome_raw") or {}
        training = arm.get("training") or {}
        episodes = int(measurements.get("episodes", -1))
        if episodes != 2049 or int(condition.get("actual_episodes", -1)) != episodes:
            problems.append(f"{name}: requires 2049 episodes")
        if not math.isclose(
            float(condition.get("detector_max_range_m", -1.0)), expected_range,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            problems.append(f"{name}: detector range mismatch")
        if int(condition.get("detect_width", -1)) != 1920 or int(
            condition.get("detect_height", -1)
        ) != 1200 or int(condition.get("detector_min_pixels", -1)) != 50:
            problems.append(f"{name}: honest-detection condition mismatch")
        if int(training.get("adaptation_epochs", -1)) != 1000:
            problems.append(f"{name}: adaptation budget mismatch")
        counts = sum(int(outcome.get(key, -episodes - 1)) for key in ("capture", "crash", "timeout"))
        if counts != episodes:
            problems.append(f"{name}: outcome counts do not sum to episodes")
        rows[name] = {
            "episodes": episodes,
            "never_acquired": float(measurements.get("never_acquired_rate")),
            "capture": float(outcome.get("capture_rate")),
            "crash": float(outcome.get("crash_rate")),
            "timeout": float(outcome.get("timeout_rate")),
            "epoch": int(training.get("terminal_epoch", -1)),
            "frame": int(training.get("terminal_frame", -1)),
            "rollback_total": int(training.get("ppo_rollback_total", -1)),
        }
    if problems:
        raise RuntimeError(
            f"invalid detection-range Stage 1 result {DETECTION_RANGE_STAGE1_PATH}: "
            + "; ".join(problems)
        )
    delta = {
        key: rows["clip28"][key] - rows["clip20"][key]
        for key in ("never_acquired", "capture", "crash", "timeout")
    }
    threshold = float((payload.get("threshold_pp") or {}).get("range_helps_at_or_below"))
    if not math.isclose(threshold, -15.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("detection-range Stage 1 threshold drift")
    if delta["never_acquired"] * 100.0 <= threshold:
        raise RuntimeError("Stage 1 values contradict the frozen inconclusive verdict")
    return {
        "payload": payload,
        "rows": rows,
        "delta": delta,
        "threshold_pp": threshold,
        "quality_gate_count": len(gates),
    }


def _governor_geometry_result() -> Optional[Dict[str, Any]]:
    """Read the speed-governor geometry grids: the arc tube vs the two straight-corridor laws.

    Returns None until the seed replication exists, so a clone without those result roots simply
    keeps showing the previous experiment instead of inventing one.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("navrl_stats", ROOT / "tools/navrl_stats.py")
    if spec is None or not (ROOT / "tools/navrl_stats.py").is_file():
        return None
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)

    roots = {
        523: [ROOT / "results/navrl_grid_d1p_ep25000_seed523", ROOT / "results/navrl_grid_l1_ep25000_seed523"],
        527: [ROOT / "results/navrl_grid_r1_seedrep_ep25000_s527"],
        531: [ROOT / "results/navrl_grid_r1_seedrep_ep25000_s531"],
    }
    cells: Dict[Any, Dict[str, Any]] = {}
    for seed, paths in roots.items():
        for root in paths:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/*bars.json")):
                if path.name.endswith(("receipt.json", "manifest.json")):
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                condition = data["condition"]
                key = (seed, int(condition["bars"]), condition["speed_governor_mode"],
                       round(float(condition["speed_governor_half_width_m"]), 2))
                cells.setdefault(key, data)
    seeds = sorted({key[0] for key in cells})
    if len(seeds) < 3:
        return None

    densities = (70, 100, 130, 160, 205)

    def counts(cell, field="crash_rate"):
        return stats.outcome_count(cell["outcome"], field), int(cell["actual_episodes"])

    def contrast(arm_a, arm_b, field="crash_rate", bars=None):
        estimates = []
        for seed in seeds:
            for density in densities:
                if bars is not None and density != bars:
                    continue
                a, b = cells.get((seed, density) + arm_a), cells.get((seed, density) + arm_b)
                if a and b:
                    estimates.append(stats.wald_diff(*counts(a, field), *counts(b, field)))
        if not estimates:
            return None
        delta, se = stats.pool_fixed(estimates)
        _, _, i_squared = stats.cochran_q(estimates) if len(estimates) > 1 else (0.0, 0, 0.0)
        return {"delta_pp": delta, "se_pp": se, "ci_pp": list(stats.ci(delta, se)),
                "cells": len(estimates), "i_squared": i_squared}

    arc, riskcap, stopcap = ("dwa_arc", 0.45), ("riskcap", 0.45), ("stopcap", 0.45)
    arc_wide, straight_wide = ("dwa_arc", 1.2), ("stopcap", 1.2)
    lowest = 0
    judged = 0
    for seed in seeds:
        for density in densities:
            arms = {name: cells.get((seed, density) + arm)
                    for name, arm in (("arc", arc), ("riskcap", riskcap), ("stopcap", stopcap))}
            if not all(arms.values()):
                continue
            judged += 1
            rates = {name: counts(cell)[0] / counts(cell)[1] for name, cell in arms.items()}
            if min(rates, key=rates.get) == "arc":
                lowest += 1

    wide_capture = []
    for seed in seeds:
        cell = cells.get((seed, 205) + straight_wide)
        if cell:
            captured, total = counts(cell, "capture_rate")
            wide_capture.append(100.0 * captured / total)

    return {
        "seeds": seeds,
        "cells": len(cells),
        "arc_lowest": lowest,
        "judged": judged,
        "arc_vs_riskcap": contrast(arc, riskcap),
        "arc_vs_stopcap": contrast(arc, stopcap),
        "stopcap_vs_riskcap": contrast(stopcap, riskcap),
        "stopcap_vs_riskcap_70": contrast(stopcap, riskcap, bars=70),
        "width_70": contrast(arc_wide, arc, bars=70),
        "width_205": contrast(arc_wide, arc, bars=205),
        "width_capture_205": contrast(arc_wide, arc, field="capture_rate", bars=205),
        "straight_wide_capture_205": sum(wide_capture) / len(wide_capture) if wide_capture else None,
    }


def _governor_geometry_update() -> Optional[Dict[str, Any]]:
    result = _governor_geometry_result()
    if result is None:
        return None
    arc_risk, arc_stop = result["arc_vs_riskcap"], result["arc_vs_stopcap"]
    width70, width205 = result["width_70"], result["width_205"]
    capture205 = result["width_capture_205"]
    stop_risk, stop_risk70 = result["stopcap_vs_riskcap"], result["stopcap_vs_riskcap_70"]
    seeds = ", ".join(str(seed) for seed in result["seeds"])
    return {
        "subtitle": "2026-09-07 · speed-governor geometry · seed replication complete",
        "headline": "Watching an arc instead of a straight corridor lowers crashes at every density.",
        "summary": (
            "The filter only ever scales the commanded speed; it never steers. Three evaluation "
            f"seeds ({seeds}) x five bar densities were re-run with the corridor replaced by the "
            "arc the vehicle will actually fly. The arc tube had the lowest crash rate in "
            f"{result['arc_lowest']} of {result['judged']} seed-density cells, pooling to "
            f"{arc_risk['delta_pp']:+.2f} pp [{arc_risk['ci_pp'][0]:+.2f}, {arc_risk['ci_pp'][1]:+.2f}] "
            f"against the risk cap with no seed heterogeneity (I2 {arc_risk['i_squared'] * 100:.0f}%). "
            "Widening that tube to 1.2 m helps twice over, while widening the straight corridor "
            "destroys the mission."
        ),
        "experiment_id": "2026-09-07-ep25000-governor-geometry-seed-replication",
        "active_experiment": {
            "is_live": False,
            "ab_experiment": True,
            "ab_gate_complete": True,
            "ab_gate_pass": True,
            "run": "ep25000 · governor geometry · R-B seed replication",
            "bars": 205,
            "seeds": result["seeds"],
            "cells": result["cells"],
            "pooled_arc_vs_riskcap_pp": arc_risk["delta_pp"],
            "pooled_arc_vs_stopcap_pp": arc_stop["delta_pp"],
        },
        "milestones": [
            {
                "label": "GEOMETRY",
                "value": f"{arc_risk['delta_pp']:+.2f} pp",
                "detail": f"arc vs risk cap, pooled over {arc_risk['cells']} cells; CI excludes 0",
                "state": "pass",
            },
            {
                "label": "REPLICATION",
                "value": f"{result['arc_lowest']}/{result['judged']}",
                "detail": "seed-density cells where the arc had the lowest crash rate",
                "state": "pass",
            },
            {
                "label": "WIDTH",
                "value": f"{width205['delta_pp']:+.2f} pp",
                "detail": (f"arc 0.45 m to 1.2 m at 205 bars; capture "
                           f"{capture205['delta_pp']:+.2f} pp, so safety and mission move together"),
                "state": "pass",
            },
            {
                "label": "LAW",
                "value": f"{stop_risk['delta_pp']:+.2f} pp",
                "detail": ("stopping law vs risk cap, pooled; the interval includes 0 once the "
                           "policy was trained with a filter in the loop"),
                "state": "warn",
            },
        ],
        "comparison": [
            {
                "label": "arc tube · 0.45 m",
                "bars": 205,
                "capture": None,
                "unique": None,
                "verdict": (f"lowest crash in {result['arc_lowest']}/{result['judged']} cells · "
                            f"pooled {arc_risk['delta_pp']:+.2f} pp vs risk cap"),
            },
            {
                "label": "arc tube · 1.2 m",
                "bars": 205,
                "capture": None,
                "unique": None,
                "verdict": (f"crash {width205['delta_pp']:+.2f} pp · capture "
                            f"{capture205['delta_pp']:+.2f} pp against the 0.45 m arc"),
            },
            {
                "label": "straight corridor · 1.2 m",
                "bars": 205,
                "capture": (result["straight_wide_capture_205"] or 0.0) / 100.0,
                "unique": None,
                "verdict": "mission collapses: most episodes time out instead of reaching the goal",
            },
        ],
        "gates": [
            {"label": "P1 pooled geometry",
             "value": (f"PASS · {arc_risk['delta_pp']:+.2f} pp "
                       f"[{arc_risk['ci_pp'][0]:+.2f}, {arc_risk['ci_pp'][1]:+.2f}]")},
            {"label": "P2 per-seed sign", "value": "PASS · 5/5 densities in both new seeds"},
            {"label": "P3 width effect", "value": f"5 of 6 · one seed short of the −2 pp threshold at 70 bars"},
            {"label": "P5 law equivalence",
             "value": (f"REJECTED · {stop_risk70['delta_pp']:+.2f} pp at 70 bars, "
                       "vanishing above 100 bars")},
        ],
        "decision": (
            "Train without the filter, then deploy the arc tube widened to about 1.2 m. The "
            "geometry result replicates across evaluation seeds and is not a property of one run. "
            "Training-seed replication is the remaining check before the co-adaptation claim is "
            "written as general."
        ),
    }


def _detection_range_stage1_update() -> Optional[Dict[str, Any]]:
    result = _detection_range_stage1_result()
    if result is None:
        return None
    rows, delta = result["rows"], result["delta"]
    clip20, clip28 = rows["clip20"], rows["clip28"]
    return {
        "subtitle": "2026-08-23 · detection-range Stage 1 complete",
        "headline": "Longer range helped, but the preregistered gate did not open.",
        "summary": (
            "Under 1920×1200 / 50-pixel detection, 20 m and 28 m arms each adapted for "
            "1,000 epochs and evaluated 2,049 episodes. Pooled never-acquired fell "
            f"{clip20['never_acquired']*100:.3f}%→{clip28['never_acquired']*100:.3f}% "
            f"({delta['never_acquired']*100:+.3f} pp), short of the frozen −15 pp gate. "
            "Capture improved descriptively, but was excluded from the Stage 1 verdict."
        ),
        "experiment_id": "2026-08-23-ref5in-detection-range-stage1",
        "active_experiment": {
            "is_live": False,
            "ab_experiment": True,
            "ab_gate_complete": True,
            "ab_gate_pass": False,
            "run": "ref5in · detection-range Stage 1",
            "epoch": clip28["epoch"],
            "max_epochs": clip28["epoch"],
            "bars": 70,
            "heldout_capture": clip28["capture"],
            "heldout_crash": clip28["crash"],
            "heldout_timeout": clip28["timeout"],
            "heldout_episodes": clip28["episodes"],
            "stage2_authorised": False,
        },
        "milestones": [
            {
                "label": "QUALITY",
                "value": "PASS · 17/17",
                "detail": "source, checkpoint, receipt, import origin and arm identity",
                "state": "pass",
            },
            {
                "label": "PRIMARY",
                "value": f"{delta['never_acquired']*100:+.3f} pp",
                "detail": "never-acquired; required ≤ −15 pp",
                "state": "warn",
            },
            {
                "label": "CAPTURE",
                "value": f"{delta['capture']*100:+.3f} pp",
                "detail": "descriptive only; excluded from the Stage 1 verdict",
                "state": "pass",
            },
            {
                "label": "SIM-TO-REAL",
                "value": "NOT CLAIMED",
                "detail": "far range remained analytic and exact in both arms",
                "state": "warn",
            },
        ],
        "comparison": [
            {
                "label": "clip20 · control",
                "bars": 70,
                "capture": clip20["capture"],
                "unique": None,
                "verdict": (
                    f"never-acq {clip20['never_acquired']*100:.3f}% · "
                    f"crash {clip20['crash']*100:.3f}% · timeout {clip20['timeout']*100:.3f}%"
                ),
            },
            {
                "label": "clip28 · treatment",
                "bars": 70,
                "capture": clip28["capture"],
                "unique": None,
                "verdict": (
                    f"never-acq {clip28['never_acquired']*100:.3f}% · "
                    f"crash {clip28['crash']*100:.3f}% · timeout {clip28['timeout']*100:.3f}%"
                ),
            },
        ],
        "gates": [
            {"label": "primary", "value": "CLOSED · −5.271 pp > −15 pp"},
            {"label": "Stage 2", "value": "BLOCKED · stage2_authorised=false"},
            {"label": "training integrity", "value": "PASS · normal exit, rollback 0 both arms"},
            {"label": "hardware claim", "value": "NO-GO · real sensor profile not measured"},
        ],
        "decision": (
            "Do not run the 10k Stage 2 or relax its threshold. For the next 72 hours, freeze the "
            "exact BOM/calibration/time contract, measure real bearing/range/latency/dropout by "
            "independent trial, then validate a far-bearing/near-range tracker before fresh PPO."
        ),
    }


def _sim2real_72h() -> Dict[str, Any]:
    result = _detection_range_stage1_result()
    historical_recovery = _historical_route_recovery_forensics()
    recovery_v2_gate = _recovery_v2_lower1p25_gate()
    recovery_v2_forensics = _recovery_v2_no_connector_forensics()
    evidence = None
    if result is not None:
        rows, delta = result["rows"], result["delta"]
        evidence = {
            "verdict": "RANGE_INCONCLUSIVE_AT_THIS_BUDGET",
            "stage2_authorised": False,
            "episodes_per_arm": rows["clip20"]["episodes"],
            "quality_gates": result["quality_gate_count"],
            "never_acquired_20": rows["clip20"]["never_acquired"],
            "never_acquired_28": rows["clip28"]["never_acquired"],
            "never_acquired_delta_pp": delta["never_acquired"] * 100.0,
            "capture_delta_pp": delta["capture"] * 100.0,
        }
    simulation_verification: Dict[str, Any] = {
        "status": "COMPLETE_WITH_LIMITS",
        "preflight_claim_status": "SYNTHETIC_ONLY",
        "preflight_path": "results/navrl_sim2real_software_preflight_2026-08-24/summary.json",
        "physical_gate": "BLOCKED",
        "fresh_ppo": "BLOCKED",
    }
    if SIM2REAL_PREFLIGHT_PATH.exists():
        try:
            preflight = json.loads(SIM2REAL_PREFLIGHT_PATH.read_text(encoding="utf-8"))
            simulation_verification["preflight_steps"] = preflight.get("steps", {})
            simulation_verification["preflight_claim_status"] = preflight.get(
                "claim_status", "SYNTHETIC_ONLY"
            )
        except (OSError, json.JSONDecodeError):
            simulation_verification["status"] = "PRE_FLIGHT_READ_ERROR"
    if PHYSICAL_SPEED_ENVELOPE_PATH.exists():
        try:
            physical = json.loads(PHYSICAL_SPEED_ENVELOPE_PATH.read_text(encoding="utf-8"))
            simulation_verification["historical_post_wall_brake_speed_envelope"] = {
                "source": str(PHYSICAL_SPEED_ENVELOPE_PATH.relative_to(ROOT)),
                "route_mode": "off_historical_lineage",
                "all_cells_pass": physical.get("all_cells_pass"),
                "highest_passing_speed_mps_by_density": physical.get(
                    "highest_passing_speed_mps_by_density"
                ),
            }
        except (OSError, json.JSONDecodeError):
            simulation_verification["physical_gate"] = "RESULT_READ_ERROR"
    if ROUTED_PHYSICAL_GATE_PATH.exists():
        try:
            routed = json.loads(ROUTED_PHYSICAL_GATE_PATH.read_text(encoding="utf-8"))
            verdicts = routed["verdicts"]
            inputs = verdicts["route_mechanism_inputs"]
            routed_gate = {
                "source": str(ROUTED_PHYSICAL_GATE_PATH.relative_to(ROOT)),
                "attempt": 2,
                "all_cells_pass": False,
                "integrity": verdicts["execution_integrity"],
                "route_mechanism": verdicts["route_mechanism"],
                "physical_ppo": "BLOCKED",
                "highest_passing_speed_mps_by_density": verdicts[
                    "highest_passing_speed_mps_by_density"
                ],
                "plan_success_70bar_4speed_pool_pct": round(
                    100.0 * inputs["plan_success_fraction_70"], 4
                ),
                "plan_success_gate_pct": 99.0,
                "fallback_70bar_4speed_pool_pct": round(
                    100.0 * inputs["fallback_interval_fraction_70"], 4
                ),
                "fallback_gate_pct": 1.0,
                "goals_per_env_70bar_0_6mps": inputs[
                    "goal_completions_per_env_70_speed_0p6"
                ],
                "goals_per_env_gate": 0.5,
                "authority": "NO_PPO_PERMISSION",
                "supported_diagnosis": (
                    "unsafe_start recovery deadlock in fail-closed zero-command fallback"
                ),
                "non_failing_gates": [
                    "tracking", "local_feasibility", "motor_saturation", "tilt", "contact",
                    "state",
                ],
            }
            routed_gate["lineage_status"] = "HISTORICAL_ATTEMPT2"
            simulation_verification["routed_physical_target_gate_attempt2"] = routed_gate
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            simulation_verification["physical_gate"] = "ROUTED_RESULT_READ_ERROR"
    if MODE_PROBE_PATH.exists():
        try:
            mode_probe = json.loads(MODE_PROBE_PATH.read_text(encoding="utf-8"))
            simulation_verification["mode_probe_verdict"] = mode_probe.get("interpretation")
        except (OSError, json.JSONDecodeError):
            simulation_verification["mode_probe_verdict"] = "RESULT_READ_ERROR"
    preflight_steps = simulation_verification.setdefault("preflight_steps", {})
    preflight_steps["route_recovery_forensics"] = historical_recovery
    preflight_steps["physical_target_gate"] = recovery_v2_gate
    simulation_verification["route_recovery_forensics"] = historical_recovery
    simulation_verification["recovery_v2_lower1p25_gate"] = recovery_v2_gate
    simulation_verification[
        "recovery_v2_no_connector_forensics"
    ] = recovery_v2_forensics
    simulation_verification[
        "track_b_authority"
    ] = "CLOSED_NO_FURTHER_GPU_PPO_RETUNE_RERUN"
    track_b_ready = (
        isinstance(recovery_v2_gate, dict)
        and recovery_v2_gate.get("status") == "VERIFIED_FAIL"
        and isinstance(recovery_v2_forensics, dict)
        and recovery_v2_forensics.get("status") == "DESCRIPTIVE_ONLY"
    )
    track_b_status = (
        "TRACK B RECOVERY-V2 ROUTE MECHANISM FAILED · NO FURTHER TRACK B AUTHORITY"
        if track_b_ready
        else "TRACK B EVIDENCE UNAVAILABLE/MALFORMED · NO TRACK B AUTHORITY"
    )
    return {
        "as_of": "2026-08-26",
        "status": (
            f"TRACK A STAGE 1 RANGE INCONCLUSIVE · {track_b_status} · HARDWARE NEXT"
        ),
        "plan_path": "docs/SIM2REAL_3DAY_EXECUTION_PLAN.md",
        "evidence": evidence,
        "simulation_verification": simulation_verification,
        "days": [
            {
                "day": "DAY 1",
                "title": "hardware + raw data",
                "detail": (
                    "Next authority: execute the hardware SIM2REAL_3DAY plan; exact "
                    "BOM/AUW/CG, calibration and time-sync, then 210 independent sensor trials."
                ),
            },
            {
                "day": "DAY 2",
                "title": "measured sensor profile",
                "detail": "Trial-level bearing/range/latency/dropout statistics and a data-chosen near/far boundary.",
            },
            {
                "day": "DAY 3",
                "title": "tracker replay + preregistration",
                "detail": "Held-out real-log replay, observation-contract smoke, then one-axis fresh-PPO preregistration.",
            },
        ],
        "software_readiness": {
            "status": "READY_FOR_REAL_LOG_PIPELINE",
            "tool": "tools/navrl_sim2real_ingest.py + tools/navrl_sim2real_telemetry.py + tools/navrl_sensor_profile.py + tools/navrl_two_zone_replay.py",
            "tests": "16/16 CPU contract tests + software preflight passed",
            "synthetic_verdict": "PASS",
            "claim_status": "SYNTHETIC_ONLY",
            "next": "Convert real rosbag/CSV, validate telemetry, build trial profile, then replay the measured two-zone contract.",
        },
        "training_blockers": [
            "exact BOM / measured AUW and CG",
            "intrinsics, extrinsics and timestamp semantics",
            "trial-level bearing/range/latency/dropout profile",
            "range_valid / uncertainty-aware target token contract",
        ],
    }


def _research_update(
    active: Optional[Dict[str, Any]], latest: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    governor_geometry = _governor_geometry_update()
    if governor_geometry is not None:
        return governor_geometry
    detection_range = _detection_range_stage1_update()
    if detection_range is not None:
        return detection_range
    ref5in = _ref5in_p2_update()
    if ref5in is not None:
        return ref5in
    record = active or latest
    if _is_v2(record):
        return _v2_search_update(record, is_live=bool(active))
    record = record or {}
    experiment = {
        "is_live": bool(active),
        "max_epochs": 13800,
        "bars": int(record.get("n_bars_active") or record.get("last_n_bars_active") or 100),
        "run": record.get(
            "run", "ppo_260731_0343_navrl_corridor6-fixed100-s1"
        ),
        "epoch": record.get("epoch") if active else record.get("last_epoch", 13800),
        "capture_tail": active["captured_rate"] if active else None,
        "crash_tail": active["crash_rate"] if active else None,
        "selector": "cluster_sector",
        "corridor_tokens": 6,
        "cluster_gap_m": 0.45,
        "sectors": 8,
        "unique_bars": 4.9,
        "duplicate_tokens": 0.2,
    }
    return {
        "subtitle": "2026-07-31 · corridor-token fixed-100 A/B complete",
        "headline": "Corridor tokens reduced bar contact, but missed the advancement gate.",
        "summary": (
            "The K=6 free-gap representation passed its physical geometry and checkpoint-schema "
            "tests, then completed an 800-epoch fixed-100 adaptation. On 4,003 held-out episodes, "
            "ep13800 reached 66.10% capture versus 64.53% for ep12500 (+1.57 percentage points; "
            "95% CI [-0.51,+3.66]) while bar contact fell from 33.18% to 32.25%. The collision "
            "signal is encouraging, but the preregistered 68% and +3pp gates were not met."
        ),
        "active_experiment": experiment,
        "milestones": [
            {
                "label": "GEOMETRY",
                "value": "PASS",
                "detail": "center clear 100%; bound-on-bar 97.8%; width sanity 98.8%",
                "state": "pass",
            },
            {
                "label": "OBS SCHEMA",
                "value": "898 → 946",
                "detail": "explicit six-corridor append and contract-safe checkpoint expansion",
                "state": "pass",
            },
            {
                "label": "HELD-OUT CAPTURE",
                "value": "66.10%",
                "detail": "4,003 episodes at 100 bars; pilot target was 68%",
                "state": "warn",
            },
            {
                "label": "BAR CONTACT",
                "value": "32.25%",
                "detail": "down from the immutable ep12500 baseline of 33.18%",
                "state": "pass",
            },
        ],
        "comparison": [
            {
                "label": "ep12500 · cluster-sector baseline",
                "bars": 100,
                "capture": 0.6453,
                "unique": 4.9,
                "verdict": "immutable baseline; bar contact 33.18%",
            },
            {
                "label": "ep13100 · corridor screen",
                "bars": 100,
                "capture": 0.6420,
                "unique": 4.9,
                "verdict": "no early gain; bar contact 33.75%",
            },
            {
                "label": "ep13450 · corridor screen",
                "bars": 100,
                "capture": 0.5993,
                "unique": 4.9,
                "verdict": "mid-run regression; bar contact 38.15%",
            },
            {
                "label": "ep13800 · corridor confirm",
                "bars": 100,
                "capture": 0.6610,
                "unique": 4.9,
                "verdict": "4,003 episodes; bar contact 32.25%",
            },
        ],
        "gates": [
            {"label": "capture target", "value": "FAIL · 66.10% < 68%"},
            {"label": "minimum gain", "value": "FAIL · +1.57pp < +3pp"},
            {"label": "bar contact", "value": "PASS · 32.25% < 33.18%"},
            {"label": "significance", "value": "INCONCLUSIVE · 95% CI crosses zero"},
        ],
        "decision": (
            "Freeze corridor6 as a partial/negative result and do not buy more epochs with the same "
            "representation. Keep density=100 and goal distance=4..16 fixed for the next diagnostic, "
            "then test whether two depth layers per sector preserve the front/back surfaces that one "
            "corridor token compresses away."
        ),
    }


V2_HELDOUT_DIR = ROOT / "results/navrl_v2_ep24000_heldout"
V2_DENSITY_RISKCAP_DIR = ROOT / "results/navrl_v2_density_curve_riskcap"
V2_DENSITY_SPEED_MAP_PATH = ROOT / "results/navrl_v2_density_speed_map/summary.json"
RISKCAP_POSTADAPT_PATH = ROOT / "results/navrl_v2_riskcap_postadapt/summary.json"

# Curves measured BEFORE the 2026-07-29 chirality fix. The LiDAR bearing table was mirrored
# (13.9% on-bar agreement vs 94.8% for the real convention) and the camera far plane fused into
# the scan as a phantom wall, so these are records of a broken observation, not of policy skill.
# They stay in the snapshot -- refuted evidence is still evidence -- but must never be displayed
# as current performance.
_PRE_CHIRALITY_CURVES = {
    "general_repr_density_curve",
    "vision_density_curve",
    "altitude_pi_speed_axis",
    "baseline_speed_axis_peak986",
    "general_12m_lookahead_speed_axis",
    "general_8m_speed_axis",
    "general_8m_tiltcomp_speed_axis",
    "general_repr_fov240_speed_axis",
    "general_repr_speed_axis",
}
_PRE_CHIRALITY_REASON = (
    "measured before the 2026-07-29 chirality fix: mirrored LiDAR bearing table plus a phantom "
    "wall from the camera far plane. Kept as a record of that condition, not as performance."
)


def _stamp_curve_provenance(status: Dict[str, Any]) -> None:
    """Tell every archived curve which task version and placement area it belongs to.

    The dashboard divides bar counts by a single global placement area, so a v1 curve was being
    reported at the v2 denominator -- 25 bars showed as 1.6/100m2 instead of 5.2, a 3.3x error.
    Stamping the area per curve lets the renderer use the right one for each series.
    """
    for group in ("density_curves", "speed_curves", "other_curves"):
        for name, pack in (status.get(group) or {}).items():
            if not isinstance(pack, dict):
                continue
            pack.setdefault("task_version", "v1")
            pack.setdefault("arena_xy_m", 24.0)
            pack.setdefault("placement_area_m2", V1_PLACEMENT_AREA_M2)
            if name in _PRE_CHIRALITY_CURVES:
                pack["superseded"] = True
                pack["superseded_reason"] = _PRE_CHIRALITY_REASON
            else:
                pack.setdefault("superseded", False)
            # Valid historical evidence is not automatically a current-task headline.  In
            # particular, this 500-epoch v1 Gaussian pilot used to replace the v2 speed curve
            # silently whenever snapshot generation failed.
            if name == "corrected_sensorfix_legacy_speed_axis":
                pack["headline_eligible"] = False
                pack["archive_reason"] = (
                    "post-fix but v1/25-bars/legacy-Gaussian 500-epoch pilot; valid archive, "
                    "not representative of the current v2 policy"
                )
            else:
                pack.setdefault("headline_eligible", pack.get("task_version") == "v2")


def _v2_density_curve() -> Dict[str, Any]:
    """v2 held-out density curve for the current frozen candidate.

    The old ep24000/off row is intentionally not attached: it differs in checkpoint, governor,
    seed and evaluator revision, so placing it beside this curve invited a causal subtraction.
    The raw result remains archived under results/ until a matched A/B/C grid is re-measured.
    """
    source = V2_DENSITY_RISKCAP_DIR if V2_DENSITY_RISKCAP_DIR.is_dir() else V2_HELDOUT_DIR
    riskcap = source is V2_DENSITY_RISKCAP_DIR
    if not source.is_dir():
        return {}
    rows = []
    for path in sorted(source.glob("*bars.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        outcome = payload.get("outcome")
        if not outcome:
            continue
        bars = int(path.stem.replace("bars", ""))
        rows.append({
            "bars": bars,
            "density_per_100m2": round(bars / V2_PLACEMENT_AREA_M2 * 100.0, 2),
            "capture": outcome["capture_rate"],
            "crash": outcome["crash_rate"],
            "timeout": outcome["timeout_rate"],
            "episodes": payload.get("actual_episodes"),
        })
    if not rows:
        return {}
    rows.sort(key=lambda r: r["bars"])
    notes = [
        "v2 search arena held-out density curve, deterministic, ~2050 episodes per cell.",
        "205 bars is the density the curriculum reached; 220 is generalisation.",
        "Legacy evaluator semantics: configured 600-step episodes timed out at action 601. "
        "Re-measure before combining with schema-v2 results.",
    ]
    return {
        "task_version": "v2",
        "arena_xy_m": 40.0,
        "placement_area_m2": V2_PLACEMENT_AREA_M2,
        "superseded": False,
        "headline_eligible": True,
        "evaluation_semantics": "legacy_timeout_at_601",
        "trained_max_bars": 205,
        "policy": ("ep25000 + riskcap (current frozen candidate)" if riskcap
                   else "ep24000 (frozen), speed governor off"),
        "notes": notes,
        "rows": rows,
    }


def _v2_speed_axis() -> Dict[str, Any]:
    """v2 fixed-target-speed axis for the CURRENT frozen candidate (ep25000 + riskcap)."""
    if not RISKCAP_POSTADAPT_PATH.exists():
        return {}
    payload = json.loads(RISKCAP_POSTADAPT_PATH.read_text(encoding="utf-8"))
    rows = []
    for entry in payload.get("fixed_speed_rows") or []:
        winner = entry.get("winner") or {}
        if "target_speed_mps" not in winner:
            continue
        rows.append({
            "target_speed_ms": winner["target_speed_mps"],
            "captured": winner["capture_rate"],
            "crash": winner["crash_rate"],
            "timeout": winner.get("timeout_rate"),
            "bar_contact": winner.get("bar_contact_rate"),
            "episodes": winner.get("episodes"),
        })
    if not rows:
        return {}
    rows.sort(key=lambda r: r["target_speed_ms"])
    return {
        "task_version": "v2",
        "arena_xy_m": 40.0,
        "placement_area_m2": V2_PLACEMENT_AREA_M2,
        "superseded": False,
        "headline_eligible": True,
        "evaluation_semantics": "legacy_timeout_at_601",
        "bars": 205,
        "policy": "ep25000 + riskcap (current frozen candidate)",
        "notes": [
            "v2 fixed-target-speed axis at 205 bars, deterministic, ~2050 episodes per cell.",
            "Same cells as the riskcap adaptation screen; these are the winner (riskcap) arm.",
        ],
        "rows": rows,
    }


def _v2_density_speed_map() -> Dict[str, Any]:
    """v2 density x target-speed map for the frozen candidate -- the headline figure, re-measured.

    The v1 map is kept (labelled) but cannot be quoted for the current task: different arena,
    placement band and target-motion law. This one is measured on the policy the page describes.
    """
    if not V2_DENSITY_SPEED_MAP_PATH.exists():
        return {}
    payload = json.loads(V2_DENSITY_SPEED_MAP_PATH.read_text(encoding="utf-8"))
    rows = [
        {
            "bars": r["bars"],
            "density_per_100m2": r["density_per_100m2"],
            "target_speed_ms": r["target_speed_ms"],
            "capture": r["capture"],
            "crash": r["crash"],
        }
        for r in payload.get("rows", [])
    ]
    if not rows:
        return {}
    return {
        "task_version": "v2",
        "arena_xy_m": 40.0,
        "placement_area_m2": V2_PLACEMENT_AREA_M2,
        "comparable_with_v2": True,
        "trained_max_bars": payload.get("trained_max_bars", 205),
        "policy": payload.get("policy"),
        "density_cost_pp": payload.get("density_cost_pp"),
        "speed_cost_pp": payload.get("speed_cost_pp"),
        "interaction_test": {
            "scope": "trained density support (130/160/190/205 bars)",
            "continuous_likelihood_ratio_p": 0.337,
            "categorical_omnibus_p": 0.817,
            "verdict": "not confirmed",
        },
        "ood_note": "220 bars is OOD and is excluded from the primary interaction test.",
        "evaluation_semantics": "legacy_timeout_at_601",
        "notes": [
            "DENSITY x TARGET-SPEED map on the v2 search arena -- the paper headline figure.",
            f"{payload.get('policy')}; deterministic, ~2050 episodes per cell, seed 47.",
            "Speeds span the trained U[0.3,1.5] support; v1's stationary-target column has no "
            "counterpart because a 0 m/s target is outside that support.",
            "Cells above the trained max measure generalisation, not a method ceiling.",
            "No density×speed interaction was confirmed inside the trained density support "
            "(continuous LR p=0.337; categorical omnibus p=0.817).",
        ],
        "rows": rows,
    }


def _stamp_density_speed_map_v1(status: Dict[str, Any]) -> None:
    """Stamp the archived v1 map wherever it now lives."""
    pack = status.get("density_speed_map_v1")
    if not pack:
        return
    pack["task_version"] = "v1"
    pack["arena_xy_m"] = 24.0
    pack["placement_area_m2"] = V1_PLACEMENT_AREA_M2
    pack["comparable_with_v2"] = False
    pack["superseded_note"] = (
        "v1 task: 24 m arena, 478 m² placement band. Superseded for the current task by the v2 "
        "map; kept as the record of the v1 contract."
    )


def _stamp_density_speed_map(status: Dict[str, Any]) -> None:
    """Make the density x speed map say which task version it belongs to.

    The map is v1 data (24 m arena, bars confined to x in 0.13..0.96 -> 478 m^2), and its
    density_per_100m2 labels were divided by that 478. The page around it now reports a v2
    placement area of 1600 m^2, so an unlabelled 17.8/100m2 at 85 bars invites exactly the
    comparison against v2's 205-bar cells that WORKLOG 2026-07-31 ruled out -- the real densities
    differ by 3.3x. Stamping the denominator into the data makes it self-describing instead of an
    implicit constant living in this file.
    """
    pack = status.get("density_speed_map")
    if not pack:
        return
    if pack.get("task_version"):
        # A map that already declares its own version is not the archived v1 one -- stamping it
        # here would relabel the current v2 map as v1, which is worse than no label at all.
        return
    pack["task_version"] = "v1"
    pack["arena_xy_m"] = 24.0
    pack["placement_area_m2"] = V1_PLACEMENT_AREA_M2
    pack["comparable_with_v2"] = False
    pack["superseded_note"] = (
        "v1 task: 24 m arena, 478 m² placement band. Densities are per that area, so these "
        "cells are NOT comparable with the v2 40 m arena (1600 m²) the rest of this page "
        "reports. A v2 re-measurement is pending."
    )


def _perception_robustness() -> Dict[str, Any]:
    """Held-out perception robustness of the frozen ep25000+riskcap policy.

    Reads the measured cells rather than restating them, so the dashboard cannot drift from
    results/. The latency axis is served from the ego-motion/budget sweeps, NOT from the R3
    screen: R3's latency cells were measured before the timestamp-aware transform and are
    superseded (WORKLOG 2026-08-06).
    """
    if not DETECTOR_ROBUSTNESS_PATH.exists():
        return {}
    r3 = json.loads(DETECTOR_ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    clean = r3["baseline"]
    axes = []

    def add(label, capture, crash, note=""):
        axes.append({
            "label": label,
            "capture_rate": capture,
            "crash_rate": crash,
            "capture_delta_pp": (capture - clean["capture_rate"]) * 100.0,
            "note": note,
        })

    for tag, label, note in (
        ("range_error_0p15m", "range error ±0.15 m", ""),
        ("range_error_0p30m", "range error ±0.30 m", ""),
        ("dropout_0p3", "detection dropout 0.3", ""),
        # Not a perturbation but a detector swap: the analytic segmenter is a bootstrap, so this
        # row is what the sensor-only claim actually rests on.
        (
            "learned_clean",
            "diagnostic 1×1 detector artifact",
            "one under-gated artifact; not a learned-detector family limit",
        ),
    ):
        cell = r3["cells"].get(tag)
        if cell:
            add(label, cell["capture_rate"], cell["crash_rate"], note)

    superseded = None
    if LATENCY_EGO_MOTION_PATH.exists():
        ego = json.loads(LATENCY_EGO_MOTION_PATH.read_text(encoding="utf-8"))["cells"]
        add("detection latency 0.1 s", ego["latency_0p1s_p3"]["capture"],
            ego["latency_0p1s_p3"]["crash"], "timestamp-aware transform")
        superseded = {
            "label": "detection latency 0.1 s, naive transform",
            "capture_rate": ego["latency_0p1s_raw"]["capture"],
            "crash_rate": ego["latency_0p1s_raw"]["crash"],
            "capture_delta_pp": (ego["latency_0p1s_raw"]["capture"] - clean["capture_rate"]) * 100.0,
        }
    if LATENCY_BUDGET_PATH.exists():
        budget = json.loads(LATENCY_BUDGET_PATH.read_text(encoding="utf-8"))["cells"]
        for tag, label in (
            ("latency_0p2s_p3", "detection latency 0.2 s"),
            ("latency_0p3s_p3", "detection latency 0.3 s"),
            ("latency_0p5s_p3", "detection latency 0.5 s"),
        ):
            if tag in budget:
                add(label, budget[tag]["capture"], budget[tag]["crash"],
                    "timestamp-aware transform")

    axes.sort(key=lambda a: a["capture_delta_pp"])
    return {
        "title": "Perception robustness",
        "subtitle": "frozen ep25000 + riskcap · seed47 · 205 bars · ~2050 episodes per cell",
        "clean": {
            "label": "analytic appearance bootstrap",
            "capture_rate": clean["capture_rate"],
            "crash_rate": clean["crash_rate"],
        },
        "axes": axes,
        "superseded": superseded,
        "finding": (
            "Detection latency looked like the dominant failure axis (-42.7pp at 0.1s) until the "
            "cause was isolated: a delayed detection was a vehicle-frame measurement from t-tau "
            "being lifted to world with the pose at t, so the drone's own motion entered every "
            "filter correction -- 0.23m from translation and 0.41m from yaw, against the 0.15m of "
            "target motion. Buffering the pose beside the detection and lifting with it recovers "
            "94% of the 0.1 s loss. Under exact timestamp/pose history the residual at 0.1 s is "
            "-2.5 pp; longer delays remain material (-15.8 pp at 0.5 s), so latency is not "
            "generally benign."
        ),
    }


def _corridor_token_plan() -> Dict[str, Any]:
    return {
        "title": "Corridor token",
        "subtitle": "completed pilot · physical geometry passed, policy gate failed",
        "definition": (
            "A corridor token is a compact description of one locally traversable opening between "
            "obstacles. The current obstacle token says where a bar surface is; a corridor token says "
            "where the drone may fit, how wide the opening is, and how far that opening remains clear."
        ),
        "why_now": (
            "The geometry was real and the actor consumed it safely: K=6 expanded the observation "
            "from 898 to 946 dimensions without changing historical offsets. It reduced bar contact "
            "by 0.93pp, but capture improved only 1.57pp and its confidence interval crossed zero. "
            "That rules out simply extending this same run; the next representation must preserve "
            "more depth structure rather than add more copies of the same local gap summary."
        ),
        "current": {
            "label": "Obstacle token · current",
            "question": "Where are the nearest bars?",
            "fields": ["bearing", "range", "relative geometry", "valid mask"],
            "weakness": "The policy must infer the opening between bars and whether the drone fits.",
        },
        "proposed": {
            "label": "Corridor token · tested",
            "question": "Where can the drone pass?",
            "fields": [
                "gap center bearing",
                "usable width",
                "left/right clearance",
                "clear depth or TTC",
                "valid mask",
            ],
            "weakness": "Local affordance only; it is not a full planner and still needs policy choice.",
        },
        "steps": [
            {
                "id": "P0",
                "title": "Freeze evidence",
                "detail": "Keep ep12500 and ep13000 plus the fixed 100-bar four-speed evaluation as immutable baselines.",
                "state": "done",
            },
            {
                "id": "P1",
                "title": "Geometry-only diagnostic",
                "detail": "Physical probe passed: center clearance 100%, bar-boundary 97.8%, width sanity 98.8%.",
                "state": "done",
            },
            {
                "id": "P2",
                "title": "Contract-safe input",
                "detail": "Expanded 898→946 dimensions, selectively initialized the new projection, and recorded schema provenance.",
                "state": "done",
            },
            {
                "id": "P3",
                "title": "Short fixed-100 A/B",
                "detail": "Completed 800 epochs and 4,003 held-out episodes: 66.10% capture, 32.25% bar contact.",
                "state": "done",
            },
        ],
        "pilot_gate": (
            "FAIL: capture 66.10% < 68% and gain +1.57pp < +3pp; bar-contact reduction passed. "
            "Do not extend corridor6. Advance to a separately gated two-depth-layer diagnostic."
        ),
    }


def _is_v2(record: Optional[Dict[str, Any]]) -> bool:
    """True when the run being described is a task-v2 search-arena run.

    Takes whichever record the dashboard is showing (active run, or the latest finished one when
    nothing is training) -- keying only on `active` made the page fall back to v1 geometry and the
    v1 density denominator whenever training was stopped.
    """
    if not record:
        return False
    run_name = str(record.get("run") or "").lower()
    return any(
        marker in run_name
        for marker in ("v2-search", "v2-recover", "v2-ttc", "navrl_v2-")
    )


def _arena_geometry(active: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Real geometry of the task the dashboard is describing, for the 3D arena panel.

    The panel used to hardcode v1 (24 m, bars in x 0.13..0.96, 2 m tall, relax placement); with
    v2 running that silently drew the wrong arena at the wrong density.
    """
    if _is_v2(active):
        return {
            "arena_xy_m": 40,
            "arena_z_m": 3,
            "bar_x_min_ratio": 0.0,
            "bar_x_max_ratio": 1.0,
            "bar_height_m": 3,
            "placement_mode": "navrl_band",
            "placement_touch_m": 0.4,
            "placement_gap_m": 1.6,
            "bars_min": 70,
            "bars_slider_min": 10,
            "bars_max": 300,
            "episode_len_steps": 600,
            "goal_dist_m": [6, 28],
            "target_speed_m": [0.3, 1.5],
            "label": "40×40×3 m · full-width bar band · 3 m bars (no fly-over)",
        }
    return {
        "arena_xy_m": 24,
        "arena_z_m": 3,
        "bar_x_min_ratio": 0.13,
        "bar_x_max_ratio": 0.96,
        "bar_height_m": 2,
        "placement_mode": "random",
        "placement_touch_m": 0.4,
        "placement_gap_m": 1.6,
        "bars_min": 25,
        "bars_slider_min": 10,
        "bars_max": 150,
        "episode_len_steps": 300,
        "goal_dist_m": [4, 16],
        "target_speed_m": [0.0, 1.5],
        "label": "24×24×3 m · bar band x 0.13–0.96 · 2 m bars",
    }


def _placement_area_m2(active: Optional[Dict[str, Any]]) -> float:
    """Obstacle-placement area used as the density denominator.

    v1: 24 m arena, bars confined to x in 0.13..0.96 -> 0.83*24*24 = 478 m^2.
    v2: 40 m arena with the band widened to full width -> the whole 1600 m^2 footprint.
    """
    return V2_PLACEMENT_AREA_M2 if _is_v2(active) else V1_PLACEMENT_AREA_M2


def _success_criteria(active: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What 'working' means for this project, stated explicitly.

    PPO's own scalars (a_loss, c_loss, entropy, kl, explained_variance) describe whether the
    OPTIMIZER is healthy, not whether the TASK is solved. They are reported as guardrails only.
    Task success is measured exclusively by held-out capture rate at a stated density.
    """
    return {
        "headline": "Success is held-out capture rate at a stated density -- never a PPO loss curve.",
        "primary": {
            "metric": "capture rate",
            "definition": "fraction of held-out episodes ending with the pursuer within 0.5 m of the target",
            "measured_by": "eval_navrl_v2_density_sweep.sh -- a frozen checkpoint replayed on episodes it never trained on",
            "why": (
                "Training-time capture mixes densities while the curriculum ramps, so it cannot be "
                "compared across epochs. A held-out sweep fixes the density per cell."
            ),
        },
        "secondary": [
            {
                "metric": "crash rate",
                "definition": "episodes ending in a bar/wall collision",
                "role": "the dominant failure mode; a capture gain paid for with more crashes is not progress",
            },
            {
                "metric": "timeout rate",
                "definition": "episodes reaching the step limit without capture",
                "role": "separates 'could not find the target' from 'found it and crashed'",
            },
            {
                "metric": "bar contact rate",
                "definition": "episodes touching an obstacle at any point",
                "role": "sensitivity to representation changes even when capture is flat",
            },
        ],
        "not_success_metrics": [
            {
                "metric": "mean reward",
                "why": (
                    "Reward falls when the density curriculum promotes, so a declining curve during "
                    "an active curriculum measures rising task difficulty, not a worsening policy. "
                    "Only compare reward WITHIN one fixed density."
                ),
            },
            {
                "metric": "ppo/a_loss, ppo/c_loss",
                "why": (
                    "PPO's surrogate and value losses are optimizer diagnostics on a moving target "
                    "distribution. They do not decrease monotonically in a healthy run and carry no "
                    "task-performance meaning."
                ),
            },
            {
                "metric": "ppo/entropy",
                "why": (
                    "Entropy is not task success. For this squashed Gaussian, however, a sudden plunge "
                    "far below its normal range is a latent-tanh saturation alarm and must be read with KL."
                ),
            },
            {
                "metric": "ppo/kl, ppo/explained_variance",
                "why": (
                    "Guardrails only. KL flags too-large policy steps (rollback at 0.04); explained "
                    "variance flags a critic that has stopped tracking returns. Both being healthy "
                    "is necessary, never sufficient."
                ),
            },
        ],
        "curriculum_gate": {
            "what": "density promotion gate (a TRAINING control, not a result)",
            "rule": (
                "promote +15 bars when capture over a 16,384-episode window clears the threshold, "
                "using the measured knot schedule 0.82@70, 0.77@85, 0.72@100, 0.70@115+, and only "
                "after at least 1,000 PPO epochs at the current density"
            ),
            "why_ramped": (
                "A flat threshold strands the curriculum forever once the achievable capture ceiling "
                "falls below it -- the failure mode behind the v1 100-bar plateau. Demand mastery "
                "where the task is easy; relax where it is hard."
            ),
            "caution": (
                "The rolling 50-epoch tail shown in Live is a diagnostic, not this gate. Never quote "
                "the tail as a promotion or publication number."
            ),
        },
        "checkpoint_rule": (
            "Evaluate a curriculum run with last_gen_ppo_ep_*.pth. gen_ppo.pth is the best-REWARD "
            "checkpoint, and reward peaks at low density, so it is a sparse-density policy that "
            "scores near 15% when replayed at high density."
        ),
    }


def _research_contract() -> Dict[str, Any]:
    """Expose the exact frozen-policy and reward semantics, including known legacy defects."""
    p2 = ROOT / "results/navrl_ref5in_p2_seed313/attestation.json"
    if p2.is_file():
        return {
            "title": "Current ref5in P2 contract & reward audit",
            "frozen_policy": "ref5in P1c ep900 · deterministic held-out P2 · governor off",
            "checkpoint_sha256": "f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf",
            "task": [
                ["arena", "40×40×3 m · 70 bars · full 1600 m² placement area"],
                ["actor input", "camera/LiDAR external scene + simulator ego-state; no GT target/bar state"],
                ["target", "mixed motion · U[0.3,1.5] m/s · goal distance U[6,28] m"],
                ["robot", "navrl_ref5in_quad · 1.20 kg synthetic design point · hardware unvalidated"],
                ["evaluation", "seed 313 · deterministic · original reflection · 2,049 episodes"],
            ],
            "reward": [
                ["range rate", "+1.0 · relative closing speed (m/s)"],
                ["time", "−0.05 per step"],
                ["static safety", "+1.5 · mean(log(d/range)) ≤ 0"],
                ["smooth / height", "−0.1·Δv − 8.0·height² outside ±0.2 m"],
                ["ego progress", "+1.0·(d(prev drone,target_new) − 0.99·d_new)"],
                ["terminal", "+30 capture; crash reward overwritten to −20"],
            ],
            "audit": {
                "frozen_training": "P1c exact-600/time_outs/source receipt PASS; on-policy result is not held-out performance.",
                "current_source": "P2 runtime byte map and Python environment matched P1c; result/receipt/log/checkpoint hashes are attested.",
                "comparison_rule": "P2 strict FAIL: timeout 5.56% exceeded 5%. P3 and legacy descriptive anchor were not run.",
            },
        }
    return {
        "title": "Frozen contract & reward audit",
        "frozen_policy": "ep25000 + sensor-only riskcap",
        "checkpoint_sha256": _RISKCAP_TRAINED_SHA256,
        "task": [
            ["arena", "40×40×3 m · 3 m bars · full 1600 m² placement area"],
            ["sensor/actor", "LiDAR 4×72 @12 m · 8 cluster-sector tokens · 240° selection"],
            ["target", "mixed motion · U[0.3,1.5] m/s · goal distance U[6,28] m"],
            ["capture", "swept relative segment enters 0.5 m; capture wins same-step contact"],
            ["command", "x/y each limited to ±2.5 m/s (XY norm can reach 3.54 m/s)"],
            [
                "policy output",
                "4-D; z has no direct actuator authority, but raw z persists in next prev_action",
            ],
        ],
        "reward": [
            ["range rate", "+1.0 · relative closing speed (m/s)"],
            ["time", "−0.05 per step"],
            ["static safety", "+1.5 · mean(log(d/range)) ≤ 0"],
            ["smooth / height", "−0.1·Δv − 8.0·height² outside ±0.2 m"],
            ["ego progress", "+1.0·(d(prev drone,target_new) − 0.99·d_new); heuristic for moving targets"],
            ["yaw", "−0.3·crab penalty − 0.02·yaw_command²"],
            ["terminal", "+30 capture; crash reward overwritten to −20"],
        ],
        "audit": {
            "frozen_training": (
                "The checkpoint was trained before the timeout fix: a configured 600-step horizon "
                "ended at action 601 and rl_games did not receive infos['time_outs'], despite "
                "value_bootstrap=True. This is a frozen provenance limitation, not retroactively fixed."
            ),
            "current_source": (
                "Source schema v2 now truncates exactly at action 600, emits both timeouts and "
                "time_outs, snapshots runtime source/environment in every evaluation receipt, and "
                "records reward/action semantics in new checkpoints."
            ),
            "comparison_rule": (
                "Do not combine the displayed legacy 601-step cells with new schema-v2 cells. "
                "Matched A/B/C arms must all be re-measured under one source manifest and seed."
            ),
        },
    }


def build_snapshot() -> Dict[str, Any]:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).isoformat()
    csv_paths = sorted(RUNS_ROOT.glob("*/aerial_run/epoch_metrics.csv"))
    training_live = _training_process_exists()
    active_path = None
    if training_live:
        unfinished = [
            path for path in csv_paths if not (path.parent / "run_summary.json").is_file()
        ]
        if unfinished:
            active_path = max(unfinished, key=lambda path: path.stat().st_mtime)

    summaries = []
    for path in csv_paths:
        summaries.append(_summarize_run(path, is_active=(path == active_path)))

    # TensorBoard runs are intentionally archived off the hot runs/ path.  Rebuilding the static
    # site from only the currently mounted directories used to make those already-published runs
    # disappear.  Preserve the checked-in history and replace records only when fresher local
    # evidence for the same run is available.  status.legacy.json covers pre-v2 archives; the
    # checked-in HEAD copy also protects history after a local status.json was regenerated once.
    historical = list(status.get("runs") or [])
    legacy_path = STATUS_PATH.with_name("status.legacy.json")
    if legacy_path.is_file():
        try:
            historical.extend(
                (json.loads(legacy_path.read_text(encoding="utf-8")).get("runs") or [])
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    try:
        checked_in = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "HEAD:docs/status/status.json"],
            stderr=subprocess.DEVNULL,
        )
        historical.extend((json.loads(checked_in).get("runs") or []))
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError, json.JSONDecodeError):
        pass
    by_run = {
        item["run"]: item
        for item in historical
        if isinstance(item, dict) and isinstance(item.get("run"), str)
    }
    by_run.update({item["run"]: item for item in summaries})
    summaries = sorted(by_run.values(), key=lambda item: item["run"])

    active = None
    if active_path is not None:
        active_summary = next(item for item in summaries if item["run"] == active_path.parents[1].name)
        active = _active_record(
            active_summary,
            active_path,
            max_epochs=_live_training_max_epochs(),
        )

    finalized = [item for item in summaries if item["run"] != (active or {}).get("run")]
    reportable_finalized = [
        item for item in finalized if not _is_smoke_run(str(item.get("run", "")))
    ]
    latest = max(
        reportable_finalized or finalized,
        key=lambda item: (item.get("finalized_at") or "", item["run"]),
        default=None,
    )
    research_update = _research_update(active, latest)
    experiment = research_update.get("active_experiment", {})
    if experiment.get("recovery_attestation_valid") is True:
        # GitHub Pages is a static snapshot, not a live filesystem verifier.  Bind the displayed
        # PASS to this generation time; the training launcher independently rechecks the artifacts.
        experiment["recovery_attestation_verified_at"] = generated_at
    status.update(
        {
            "generated_at": generated_at,
            "repo": REPOSITORY_ID,
            "n_runs": len(summaries),
            "active_run": active,
            "latest_run": latest,
            "runs": summaries,
            "research_update": research_update,
            "sim2real_72h": _sim2real_72h(),
            "corridor_token": _corridor_token_plan(),
            "perception_robustness": _perception_robustness(),
            "success_criteria": _success_criteria(active or latest),
            "research_contract": _research_contract(),
            "placement_area_m2": _placement_area_m2(active or latest),
            "arena_geometry": _arena_geometry(active or latest),
        }
    )
    v2_map = _v2_density_speed_map()
    if v2_map:
        # Keep the v1 map under its own key: it is the record of the v1 contract, and the
        # dashboard labels it as such rather than deleting measured evidence.
        if "density_speed_map" in status:
            status.setdefault("density_speed_map_v1", status["density_speed_map"])
        status["density_speed_map"] = v2_map
    _stamp_density_speed_map(status)
    _stamp_density_speed_map_v1(status)
    v2_density = _v2_density_curve()
    if v2_density:
        status.setdefault("density_curves", {})["v2_heldout_density_curve"] = v2_density
    v2_speed = _v2_speed_axis()
    if v2_speed:
        status.setdefault("speed_curves", {})["v2_riskcap_fixed_speed_axis"] = v2_speed
    status.setdefault("density_curves", {})[
        "corrected_chirality_density_curve"
    ] = _corrected_density_curve()
    # Stamped LAST so it also covers curves rebuilt above; stamping earlier let the rebuilt
    # chirality curve overwrite its own provenance and render at the wrong denominator.
    _stamp_curve_provenance(status)
    return status


def write_snapshot(status: Dict[str, Any]) -> None:
    STATUS_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="write even when no local run evidence is visible (almost never correct)",
    )
    args = parser.parse_args()

    status = build_snapshot()
    # runs/ is gitignored, so a fresh clone -- a Cursor cloud/mobile agent, or any
    # machine that has not trained -- sees zero runs. Writing then silently strips
    # the published run history and latest_run, and publishing that wipes the site.
    if status["n_runs"] == 0 and not args.allow_empty:
        print(
            "[status] refusing to write: no runs found under\n"
            f"  {RUNS_ROOT}\n"
            "runs/ is gitignored, so this is expected on a fresh clone (cloud/mobile\n"
            "agent). Publishing an empty snapshot would erase the dashboard's run\n"
            "history. Run this on the training workstation, or pass --allow-empty if\n"
            "you really intend an empty dashboard.",
            file=sys.stderr,
        )
        return 2

    write_snapshot(status)
    active = status.get("active_run")
    print(
        "[status] synchronized "
        f"{status['n_runs']} runs; active={active['run'] if active else 'none'}; "
        f"generated_at={status['generated_at']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
