#!/usr/bin/env python3
"""Create the held-out PASS attestation required by the safe NavRL-v2 recovery launcher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRUSTED_SOURCE_SHA256 = (
    "3a0c167cbf4bc966426488f562da2b6788bd00ca62e3a31f226f5fbe1967578f"
)
SOURCE_EPOCH = 9500
REQUIRED_EPOCHS = 100
SMOKE_BARS = 130
MIN_EPISODES = 2049
MAX_KL = 0.04
MAX_TASK_INPUT_OOB_RATE = 1e-9
MIN_CAPTURE_RATE = 0.65
MAX_CRASH_RATE = 0.35
MAX_TIMEOUT_RATE = 0.10
ACTION_AXES = ("x", "y", "z", "yaw")
EVALUATOR_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
)
SMOKE_STATE_CONTRACT = {
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
    "cfg_episode_len_steps": 600.0,
    "cfg_general_goal_dist_min": 6.0,
    "cfg_general_goal_dist_max": 28.0,
    "cfg_general_train": True,
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
    "cfg_max_tilt_deg": 45.0,
    "cfg_tilt_comp": True,
    "cfg_max_velocity": 2.5,
    "cfg_yaw_rate_max": 3.0,
    "cfg_alt_hold_vmax": 2.5,
    "cfg_oob_margin": 1.0,
    "cfg_target_motion_model": "symmetric_local_steer_v2_heading_continuity90",
    "cfg_target_pattern": "mixed",
    "cfg_target_speed_min": 0.3,
    "cfg_target_speed_final": 1.5,
    "cfg_target_speed_fixed": -1.0,
    "cfg_target_speed_ramp_epochs": 300,
    "cfg_target_speed_ramp_start_epochs": 0,
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
EVALUATION_CONTRACT = {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {result!r}")
    return result


def _contract_mismatches(actual, expected, prefix: str) -> List[str]:
    if not isinstance(actual, dict):
        return [f"{prefix} is not an object"]
    problems: List[str] = []
    for key, want in expected.items():
        if key not in actual:
            problems.append(f"{prefix}.{key} is missing")
            continue
        got = actual[key]
        if isinstance(want, bool):
            matches = isinstance(got, bool) and got is want
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            try:
                numeric = float(got)
                matches = math.isfinite(numeric) and abs(numeric - float(want)) <= 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = got == want
        if not matches:
            problems.append(f"{prefix}.{key}={got!r}, expected {want!r}")
    return problems


def _is_sha256(value) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_evaluator_receipt(
    checkpoint_path: Path,
    result_path: Path,
    result: dict,
    problems: List[str],
) -> dict:
    """Validate the evaluator-produced, byte-bound receipt for one held-out cell.

    This is a local provenance guard, not a remote signature: a user who can replace both code and
    every artifact can still manufacture files.  It does prevent a plausible hand-written metric
    JSON or a checkpoint swapped after evaluation from satisfying the normal recovery workflow.
    """

    checkpoint_sha = _sha256(checkpoint_path)
    expected_receipt_path = result_path.with_suffix(".receipt.json").resolve()
    receipt_path = Path(str(result.get("evaluation_receipt", ""))).expanduser().resolve()
    if receipt_path != expected_receipt_path:
        problems.append(
            "held-out result does not name its canonical sibling evaluator receipt"
        )
        return {}
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"evaluator receipt cannot be read: {exc}")
        return {}
    if not isinstance(receipt, dict):
        problems.append("evaluator receipt is not an object")
        return {}

    evaluator_path = Path(str(result.get("evaluator_script", ""))).expanduser().resolve()
    snapshot_path = Path(
        str(result.get("evaluated_checkpoint_snapshot", ""))
    ).expanduser().resolve()
    expected_log_path = result_path.with_suffix(".log").resolve()
    log_path = Path(str(receipt.get("log_file", ""))).expanduser().resolve()
    condition = result.get("condition") or {}
    nonce = str(condition.get("evaluation_nonce", ""))

    manifest_path = Path(
        str(result.get("runtime_source_manifest", ""))
    ).expanduser().resolve()
    canonical_manifest = (result_path.parent / "source_manifest.json").resolve()
    manifest = {}
    if manifest_path != canonical_manifest:
        problems.append("held-out result names a non-canonical runtime source manifest")
        manifest_sha = ""
    elif not manifest_path.is_file():
        problems.append(f"runtime source manifest is missing: {manifest_path}")
        manifest_sha = ""
    else:
        manifest_sha = _sha256(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"runtime source manifest cannot be read: {exc}")
            manifest = {}
    if result.get("runtime_source_manifest_sha256") != manifest_sha:
        problems.append("held-out runtime source manifest SHA-256 is invalid")
    if manifest:
        if manifest.get("schema_version") != 2:
            problems.append("runtime source manifest schema_version is not 2")
        files = manifest.get("runtime_files") or []
        if int(manifest.get("runtime_file_count", -1)) != len(files) or not files:
            problems.append("runtime source manifest has invalid/empty file accounting")
        repo = Path(str(manifest.get("repository_root", ""))).expanduser().resolve()
        for entry in files:
            try:
                expected = str(entry["sha256"])
                original = (repo / entry["path"]).resolve()
                snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
            except (KeyError, TypeError):
                problems.append("runtime source manifest contains a malformed file entry")
                continue
            if not original.is_file() or _sha256(original) != expected:
                problems.append(f"runtime source changed after evaluation: {entry.get('path')}")
            if not snapshot.is_file() or _sha256(snapshot) != expected:
                problems.append(f"runtime source snapshot is invalid: {entry.get('snapshot')}")
        environment_path = (
            manifest_path.parent / str(manifest.get("python_environment", ""))
        ).resolve()
        if (
            not environment_path.is_file()
            or _sha256(environment_path) != manifest.get("python_environment_sha256")
        ):
            problems.append("Python environment manifest is missing or changed")

    if evaluator_path != EVALUATOR_SCRIPT.resolve():
        problems.append("held-out result names a non-canonical evaluator script")
    if not EVALUATOR_SCRIPT.is_file():
        problems.append(f"canonical evaluator script is missing: {EVALUATOR_SCRIPT}")
        evaluator_sha = ""
    else:
        evaluator_sha = _sha256(EVALUATOR_SCRIPT)
    if result.get("evaluator_script_sha256") != evaluator_sha:
        problems.append("held-out evaluator script SHA-256 does not match current code")

    canonical_snapshot = (result_path.parent / "checkpoint_snapshot.pth").resolve()
    if snapshot_path != canonical_snapshot:
        problems.append("held-out result names a non-canonical checkpoint snapshot")
        snapshot_sha = ""
    elif not snapshot_path.is_file():
        problems.append(f"evaluated checkpoint snapshot is missing: {snapshot_path}")
        snapshot_sha = ""
    else:
        snapshot_sha = _sha256(snapshot_path)
    if snapshot_sha != checkpoint_sha:
        problems.append("evaluated checkpoint snapshot bytes differ from the source checkpoint")
    if result.get("checkpoint_sha256") != checkpoint_sha:
        problems.append("held-out result checkpoint SHA-256 does not match current checkpoint bytes")
    if result.get("evaluated_checkpoint_snapshot_sha256") != snapshot_sha:
        problems.append("held-out result checkpoint-snapshot SHA-256 is invalid")
    if not _is_sha256(nonce):
        problems.append("held-out evaluation nonce is missing or malformed")

    if log_path != expected_log_path:
        problems.append("evaluator receipt names a non-canonical cell log")
        log_sha = ""
    elif not log_path.is_file():
        problems.append(f"held-out cell log is missing: {log_path}")
        log_sha = ""
    else:
        log_sha = _sha256(log_path)

    expected_receipt = {
        "schema_version": 2,
        "producer": "eval_navrl_v2_density_sweep.sh",
        "evaluation_nonce": nonce,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "evaluated_checkpoint_snapshot": str(snapshot_path),
        "evaluated_checkpoint_snapshot_sha256": snapshot_sha,
        "result_json": str(result_path),
        "result_sha256": _sha256(result_path),
        "log_file": str(expected_log_path),
        "log_sha256": log_sha,
        "evaluator_script": str(EVALUATOR_SCRIPT.resolve()),
        "evaluator_script_sha256": evaluator_sha,
        "runtime_source_manifest": str(canonical_manifest),
        "runtime_source_manifest_sha256": manifest_sha,
        "runtime_source_file_count": int(manifest.get("runtime_file_count", -1)),
        "runtime_git_commit": manifest.get("git_commit"),
        "runtime_git_dirty": manifest.get("git_dirty"),
        "python_environment_manifest": str(
            (canonical_manifest.parent / str(manifest.get("python_environment", ""))).resolve()
        ),
        "python_environment_manifest_sha256": manifest.get(
            "python_environment_sha256"
        ),
        "bars": (result.get("condition") or {}).get("bars"),
        "seed": (result.get("condition") or {}).get("seed"),
        "requested_episodes": result.get("requested_episodes"),
        "actual_episodes": result.get("actual_episodes"),
    }
    problems.extend(
        _contract_mismatches(receipt, expected_receipt, "evaluator_receipt")
    )
    for key in ("started_at_utc", "completed_at_utc"):
        value = str(receipt.get(key, ""))
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError:
            problems.append(f"evaluator_receipt.{key} is invalid")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha,
        "evaluator_sha256": evaluator_sha,
    }


def _scalar_window(
    accumulator: EventAccumulator,
    tag: str,
    expected_steps: Iterable[int],
) -> List[float]:
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if tag not in scalar_tags:
        raise ValueError(f"training TensorBoard scalar is missing: {tag}")
    expected = list(expected_steps)
    expected_set = set(expected)
    by_step: Dict[int, float] = {}
    duplicates = set()
    for event in accumulator.Scalars(tag):
        step = int(event.step)
        if step not in expected_set:
            continue
        value = _finite_float(event.value, f"{tag}@{step}")
        if step in by_step:
            duplicates.add(step)
        else:
            by_step[step] = value
    if duplicates:
        preview = ", ".join(str(step) for step in sorted(duplicates)[:8])
        suffix = "..." if len(duplicates) > 8 else ""
        raise ValueError(
            f"{tag} has duplicate writes in the smoke window: {preview}{suffix}"
        )
    missing = [step for step in expected if step not in by_step]
    if missing:
        preview = ", ".join(str(step) for step in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"{tag} lacks {len(missing)} smoke epochs: {preview}{suffix}")
    return [by_step[step] for step in expected]


def _build_attestation_payload(
    checkpoint_path: Path,
    result_path: Path,
    *,
    created_at_utc: Optional[str] = None,
):
    checkpoint_path = checkpoint_path.resolve()
    result_path = result_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    epoch = int(checkpoint.get("epoch", -1))
    run_root = checkpoint_path.parents[1]
    problems: List[str] = []

    if state.get("cfg_recovery_stage") != "smoke":
        problems.append("checkpoint recovery stage is not smoke")
    if state.get("cfg_recovery_source_sha256") != TRUSTED_SOURCE_SHA256:
        problems.append("checkpoint does not descend from the audited ep9500 source")
    if int(state.get("cfg_recovery_source_epoch", -1)) != SOURCE_EPOCH:
        problems.append("recovery source epoch is not 9500")
    if int(state.get("cfg_recovery_smoke_required_epochs", -1)) != REQUIRED_EPOCHS:
        problems.append("recovery smoke budget is not 100 epochs")
    if int(state.get("cfg_recovery_smoke_bars", -1)) != SMOKE_BARS:
        problems.append("recovery smoke density metadata is not 130 bars")
    if int(state.get("n_bars_active", -1)) != SMOKE_BARS:
        problems.append("checkpoint active density is not 130 bars")
    if epoch != SOURCE_EPOCH + REQUIRED_EPOCHS:
        problems.append(f"checkpoint epoch is {epoch}; expected exactly 9600")
    if int(checkpoint.get("frame", -1)) != 39321600:
        problems.append(
            f"checkpoint frame is {checkpoint.get('frame')!r}; expected exactly 39321600"
        )
    marker_path = run_root / ".aerial_training_finished"
    try:
        marker_lines = marker_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        problems.append("normal-completion marker is missing")
    else:
        if f"epoch={SOURCE_EPOCH + REQUIRED_EPOCHS}" not in {
            line.strip() for line in marker_lines
        }:
            problems.append("normal-completion marker does not certify epoch 9600")
    problems.extend(_contract_mismatches(state, SMOKE_STATE_CONTRACT, "checkpoint.env_state"))

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"held-out JSON cannot be read: {exc}")
        result = {}
    if not isinstance(result, dict):
        problems.append("held-out JSON is not an object")
        result = {}

    receipt_evidence = _validate_evaluator_receipt(
        checkpoint_path, result_path, result, problems
    )

    condition = result.get("condition") or {}
    outcome = result.get("outcome") or {}
    action = result.get("action") or {}
    problems.extend(
        _contract_mismatches(
            result.get("v2_evaluation_contract"), EVALUATION_CONTRACT, "v2_evaluation_contract"
        )
    )
    try:
        requested_episodes = int(result.get("requested_episodes", -1))
        actual_episodes = int(result.get("actual_episodes", -1))
        seed = int(condition.get("seed", -1))
        bars = int(condition.get("bars", -1))
        outcome_counts = [
            int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")
        ]
        capture_rate = _finite_float(outcome.get("capture_rate"), "capture_rate")
        crash_rate = _finite_float(outcome.get("crash_rate"), "crash_rate")
        timeout_rate = _finite_float(outcome.get("timeout_rate"), "timeout_rate")
        raw_eval_oob = action.get("task_input_oob_rate") or []
        if len(raw_eval_oob) != len(ACTION_AXES):
            raise ValueError("action.task_input_oob_rate must contain exactly four action axes")
        eval_oob = [
            _finite_float(value, f"action.task_input_oob_rate[{axis}]")
            for axis, value in zip(ACTION_AXES, raw_eval_oob)
        ]
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid held-out metrics: {exc}")
        requested_episodes = -1
        actual_episodes = -1
        seed = -1
        bars = -1
        outcome_counts = [-1, -1, -1]
        capture_rate = crash_rate = timeout_rate = float("nan")
        eval_oob = [float("inf")]

    if result.get("schema_version") != 1:
        problems.append("held-out JSON schema_version is not 1")
    if Path(str(result.get("checkpoint", ""))).resolve() != checkpoint_path:
        problems.append("held-out JSON names a different checkpoint")
    if requested_episodes < MIN_EPISODES:
        problems.append(f"requested held-out episodes {requested_episodes} < {MIN_EPISODES}")
    if actual_episodes < requested_episodes:
        problems.append(
            f"held-out evaluation is incomplete: {actual_episodes} < requested {requested_episodes}"
        )
    if actual_episodes < MIN_EPISODES:
        problems.append(f"held-out episodes {actual_episodes} < {MIN_EPISODES}")
    if any(value < 0 for value in outcome_counts) or sum(outcome_counts) != actual_episodes:
        problems.append(
            "held-out captured+crash+timeout counts do not equal actual episodes"
        )
    if actual_episodes > 0 and all(value >= 0 for value in outcome_counts):
        for name, reported, count in zip(
            ("capture", "crash", "timeout"),
            (capture_rate, crash_rate, timeout_rate),
            outcome_counts,
        ):
            measured = count / actual_episodes
            if math.isfinite(reported) and abs(reported - measured) > 1e-9:
                problems.append(
                    f"held-out {name} rate {reported:.12g} disagrees with count/actual "
                    f"{measured:.12g}"
                )
    if bars != SMOKE_BARS:
        problems.append(f"held-out density is {bars}, expected {SMOKE_BARS}")
    if seed != 42:
        problems.append(f"held-out seed is {seed}, expected 42")
    if condition.get("target_speed_mode") != "uniform":
        problems.append("target-speed mode is not uniform")
    if condition.get("target_pattern") != "mixed":
        problems.append("target pattern is not mixed")
    try:
        if int(condition.get("num_envs", -1)) != 128:
            problems.append("held-out evaluation did not use 128 environments")
    except (TypeError, ValueError):
        problems.append("held-out num_envs is invalid")
    if condition.get("full_goal_distribution") is not True:
        problems.append("held-out evaluation did not use the full goal-distance distribution")
    if condition.get("fov_curriculum_saturated") is not True:
        problems.append("held-out evaluation did not use the final FOV distribution")
    try:
        if abs(_finite_float(condition.get("target_speed_min_mps"), "target speed min") - 0.3) > 1e-6:
            problems.append("target-speed minimum is not 0.3 m/s")
        if abs(_finite_float(condition.get("target_speed_max_mps"), "target speed max") - 1.5) > 1e-6:
            problems.append("target-speed maximum is not 1.5 m/s")
        if abs(_finite_float(condition.get("oob_margin_m"), "OOB margin") - 1.0) > 1e-6:
            problems.append("OOB margin is not 1.0 m")
        if condition.get("pursuer_speed_limit_semantics") != "per_axis_xy":
            problems.append("pursuer speed limit is not recorded as per-axis XY")
        if abs(
            _finite_float(
                condition.get("pursuer_per_axis_speed_limit_mps"),
                "pursuer per-axis speed limit",
            )
            - 2.5
        ) > 1e-6:
            problems.append("pursuer per-axis speed limit is not 2.5 m/s")
        if abs(
            _finite_float(
                condition.get("pursuer_max_horizontal_request_norm_mps"),
                "pursuer maximum horizontal request norm",
            )
            - math.sqrt(2.0) * 2.5
        ) > 1e-6:
            problems.append("pursuer maximum horizontal request norm is not sqrt(2)*2.5")
        if int(condition.get("policy_output_dim", -1)) != 4:
            problems.append("policy output dimension is not 4")
        if condition.get("policy_z_output_overwritten_by_altitude_pi") is not True:
            problems.append("policy-z compatibility dimension is not marked as overwritten")
        if condition.get("policy_z_persisted_in_prev_action_observation") is not True:
            problems.append("indirect policy-z prev_action channel is not attested")
        if int(condition.get("episode_len_steps", -1)) != 600:
            problems.append("held-out episode length is not 600 steps")
        if abs(_finite_float(condition.get("goal_dist_min_m"), "goal distance min") - 6.0) > 1e-6:
            problems.append("held-out goal-distance minimum is not 6 m")
        if abs(_finite_float(condition.get("goal_dist_max_m"), "goal distance max") - 28.0) > 1e-6:
            problems.append("held-out goal-distance maximum is not 28 m")
    except ValueError as exc:
        problems.append(str(exc))
    if action.get("policy") != "squashed_gaussian":
        problems.append("held-out action policy is not squashed_gaussian")
    try:
        action_samples = int(action.get("samples", -1))
    except (TypeError, ValueError):
        action_samples = -1
    if action_samples <= 0:
        problems.append("held-out action sample count is missing or non-positive")
    for label, rate in (
        ("capture", capture_rate),
        ("crash", crash_rate),
        ("timeout", timeout_rate),
    ):
        if math.isfinite(rate) and not 0.0 <= rate <= 1.0:
            problems.append(f"{label} rate is outside [0,1]: {rate}")
    if any(not 0.0 <= value <= 1.0 for value in eval_oob):
        problems.append("held-out task-input OOB rate is outside [0,1]")
    if math.isfinite(capture_rate) and capture_rate < MIN_CAPTURE_RATE:
        problems.append(f"capture {capture_rate:.6f} < {MIN_CAPTURE_RATE:.2f}")
    if math.isfinite(crash_rate) and crash_rate > MAX_CRASH_RATE:
        problems.append(f"crash {crash_rate:.6f} > {MAX_CRASH_RATE:.2f}")
    if math.isfinite(timeout_rate) and timeout_rate > MAX_TIMEOUT_RATE:
        problems.append(f"timeout {timeout_rate:.6f} > {MAX_TIMEOUT_RATE:.2f}")

    training_max_kl = float("inf")
    training_max_oob = float("inf")
    final_rollback_streak = float("inf")
    max_rollback_streak = float("inf")
    max_epoch_rollback = float("inf")
    final_rollback_total = float("inf")
    max_rollback_total = float("inf")
    try:
        summaries = run_root / "summaries"
        accumulator = EventAccumulator(str(summaries), size_guidance={"scalars": 0})
        accumulator.Reload()
        expected_steps = range(SOURCE_EPOCH + 1, epoch + 1)
        kl_values = _scalar_window(
            accumulator, "ppo/behavior_kl_audit_max", expected_steps
        )
        rollback_streak = _scalar_window(
            accumulator, "ppo/epoch_rollback_streak", expected_steps
        )
        epoch_rollback = _scalar_window(
            accumulator, "ppo/epoch_rollback", expected_steps
        )
        rollback_total = _scalar_window(
            accumulator, "ppo/epoch_rollback_total", expected_steps
        )
        training_oob = []
        for axis in ACTION_AXES:
            training_oob.extend(
                _scalar_window(
                    accumulator, f"policy_action/raw_oob_{axis}", expected_steps
                )
            )
        training_max_kl = max(kl_values)
        training_max_oob = max(training_oob)
        final_rollback_streak = rollback_streak[-1]
        max_rollback_streak = max(rollback_streak)
        max_epoch_rollback = max(epoch_rollback)
        final_rollback_total = rollback_total[-1]
        max_rollback_total = max(rollback_total)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        problems.append(f"training audit window is incomplete: {exc}")

    max_task_input_oob_rate = max(training_max_oob, max(eval_oob))
    if training_max_kl < -1e-6 or training_max_kl > MAX_KL:
        problems.append(
            f"training max KL {training_max_kl:.6f} is outside [-1e-6,{MAX_KL:.2f}]"
        )
    if training_max_oob < 0.0 or max_task_input_oob_rate > MAX_TASK_INPUT_OOB_RATE:
        problems.append(
            f"max task-input OOB rate {max_task_input_oob_rate:.6g} > {MAX_TASK_INPUT_OOB_RATE:.1g}"
        )
    if max_rollback_streak != 0.0 or final_rollback_streak != 0.0:
        problems.append(
            "recovery smoke contains a rollback: "
            f"max streak={max_rollback_streak}, final streak={final_rollback_streak}"
        )
    if (
        max_epoch_rollback != 0.0
        or max_rollback_total != 0.0
        or final_rollback_total != 0.0
    ):
        problems.append(
            "recovery smoke contains a rollback counter/event: "
            f"epoch max={max_epoch_rollback}, total max={max_rollback_total}, "
            f"total final={final_rollback_total}"
        )

    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError("[eval_v2] recovery attestation REFUSED:\n" + detail)

    checkpoint_sha256 = _sha256(checkpoint_path)
    payload = {
        "schema_version": 1,
        "verdict": "PASS",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": epoch,
        "checkpoint_frame": int(checkpoint.get("frame", -1)),
        "source_checkpoint_sha256": TRUSTED_SOURCE_SHA256,
        "source_epoch": SOURCE_EPOCH,
        "smoke_epochs": REQUIRED_EPOCHS,
        "bars": bars,
        "seed": seed,
        "episodes": actual_episodes,
        "capture_rate": capture_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
        "requested_episodes": requested_episodes,
        "captured": outcome_counts[0],
        "crash": outcome_counts[1],
        "timeout": outcome_counts[2],
        "training_max_kl": training_max_kl,
        "training_max_task_input_oob_rate": training_max_oob,
        "evaluation_max_task_input_oob_rate": max(eval_oob),
        "max_task_input_oob_rate": max_task_input_oob_rate,
        "max_rollback_streak": max_rollback_streak,
        "final_rollback_streak": final_rollback_streak,
        "max_epoch_rollback": max_epoch_rollback,
        "max_rollback_total": max_rollback_total,
        "final_rollback_total": final_rollback_total,
        "heldout_result_json": str(result_path),
        "heldout_result_sha256": _sha256(result_path),
        "evaluator_receipt_json": receipt_evidence["path"],
        "evaluator_receipt_sha256": receipt_evidence["sha256"],
        "evaluated_checkpoint_snapshot": receipt_evidence["snapshot_path"],
        "evaluated_checkpoint_snapshot_sha256": receipt_evidence[
            "snapshot_sha256"
        ],
        "evaluator_script_sha256": receipt_evidence["evaluator_sha256"],
        "thresholds": {
            "min_episodes": MIN_EPISODES,
            "min_capture_rate": MIN_CAPTURE_RATE,
            "max_crash_rate": MAX_CRASH_RATE,
            "max_timeout_rate": MAX_TIMEOUT_RATE,
            "max_training_kl": MAX_KL,
            "max_task_input_oob_rate": MAX_TASK_INPUT_OOB_RATE,
        },
        "evaluation_contract": result["v2_evaluation_contract"],
    }
    return run_root, payload


def create_attestation(checkpoint_path: Path, result_path: Path) -> Path:
    run_root, payload = _build_attestation_payload(checkpoint_path, result_path)
    output_path = run_root / ".navrl_v2_recovery_eval_pass.json"
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return output_path


def verify_existing_attestation(checkpoint_path: Path, attestation_path: Path) -> Path:
    """Recompute all checkpoint/result/TensorBoard evidence and compare the stored artifact.

    This deliberately does not trust the attestation's self-reported KL/OOB/rollback or outcome
    values.  The canonical builder is rerun against the byte-bound result and exact 100-epoch
    summary window; every generated field except the already-validated creation time must match.
    """

    checkpoint_path = checkpoint_path.resolve()
    attestation_path = attestation_path.resolve()
    try:
        actual = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"stored recovery attestation cannot be read: {exc}") from exc
    if not isinstance(actual, dict):
        raise RuntimeError("stored recovery attestation is not an object")
    created_at = str(actual.get("created_at_utc", ""))
    try:
        parsed_created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if parsed_created.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError as exc:
        raise RuntimeError("stored recovery attestation has an invalid creation time") from exc
    result_path = Path(str(actual.get("heldout_result_json", ""))).expanduser()
    run_root, expected = _build_attestation_payload(
        checkpoint_path,
        result_path,
        created_at_utc=created_at,
    )
    canonical_path = (run_root / ".navrl_v2_recovery_eval_pass.json").resolve()
    if attestation_path != canonical_path:
        raise RuntimeError(
            f"recovery attestation is outside its canonical run root: {attestation_path}"
        )
    if actual != expected:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "stored recovery attestation disagrees with recomputed evidence: "
            + ", ".join(differing)
        )
    return attestation_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result_json", type=Path, nargs="?")
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        if args.verify_existing is not None:
            if args.result_json is not None:
                parser.error("result_json is not accepted with --verify-existing")
            output = verify_existing_attestation(args.checkpoint, args.verify_existing)
            print(
                f"[eval_v2] recovery attestation VERIFIED | {output} | sha256={_sha256(output)}"
            )
            return 0
        if args.result_json is None:
            parser.error("result_json is required unless --verify-existing is used")
        output = create_attestation(args.checkpoint, args.result_json)
    except RuntimeError as exc:
        message = str(exc)
        if "recovery attestation REFUSED" not in message:
            message = "[eval_v2] recovery attestation REFUSED:\n  - " + message
        raise SystemExit(message) from exc
    print(
        f"[eval_v2] recovery attestation PASS | {output} | sha256={_sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
