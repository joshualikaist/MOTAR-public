#!/usr/bin/env python3
"""Run and attest the preregistered ref5in P2 held-out decision cell.

This is deliberately narrower than the generic density evaluator.  It binds one P1c checkpoint to
one held-out seed/condition, proves that the evaluated simulator byte map is identical to P1c, and
applies integer-count decision gates.  A PASS unlocks only manual review of P3 seed 211.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/fair/miniconda3/envs/aerialgym/bin/python")
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
P1_REPORT = ROOT / "results/navrl_ref5in_smoke_seed197/p1c/summary.json"
P1_REPORT_SHA = "14646d816685e3563a9a908205b4eeb81027699f6c31db90a48b90c6f1ed3fb2"
P1_CHECKPOINT = ROOT / (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197/nn/"
    "last_gen_ppo_ep_900_rew_137.08087.pth"
)
P1_CHECKPOINT_SHA = "f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf"
P1_MANIFEST = ROOT / (
    "aerial_gym/rl_training/rl_games/train_source_receipts/"
    "ref5in_smoke_c_s197_260813_054048_2228846/source_manifest.json"
)
P1_MANIFEST_SHA = "ce4a52b850014eab85ee57315ee1834da2d5d16a92e78dc86d2fa6996efcd1ff"
P1_RUNTIME_MAP_SHA = "bbd3f45447c7e2ba03c51d65149e6028085e4280970703cb9887b2c6eaa8fb3a"
PYTHON_ENV_SHA = "70b26f5afc73f6703d04e36d545a43296aebd299cc04a2626bc72eba6d59003e"
EVALUATOR_SHA = "df46894c790ef43ca7d8fc3042d4fa648d94e250cb1ab69ff1ae0e9f8a23666c"
ROBOT_CONFIG_SHA = "ebb71802f19b630ba6c2ac4c04b113c269d8bbd3e40e094e126913caa8731297"
ROBOT_URDF_SHA = "5c160b0d19caebf9a4a3c38be861a77637ee0fb2b80febf4ac54d8b143db6a32"
LEGACY_CHECKPOINT = ROOT / (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260812_1620_navrl_v2-v5a-semantics-smoke-s197/nn/"
    "last_gen_ppo_ep_500_rew_109.84576.pth"
)
LEGACY_CHECKPOINT_SHA = "3c13173e32b82bb1a6665a9430996c83c231c294728def72d8ce827993be2ff4"
OUTPUT_ROOT = ROOT / "results/navrl_ref5in_p2_seed313"
DECISION_DIR = OUTPUT_ROOT / "ref5in"
ANCHOR_DIR = OUTPUT_ROOT / "legacy_anchor"
SOURCE_BUNDLE = OUTPUT_ROOT / "source_bundle"
ATTESTATION = OUTPUT_ROOT / "attestation.json"
RUNTIME_ROOTS = ("aerial_gym", "resources/robots")
RUNTIME_EXTENSIONS = {".py", ".pyx", ".sh", ".yaml", ".yml", ".toml", ".json", ".csv", ".urdf"}


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    reject_nonfinite(value, str(path))
    return value


def reject_nonfinite(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"non-finite number at {where}")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{where}[{index}]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def parse_time(value: Any, name: str) -> datetime:
    require(isinstance(value, str), f"{name} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"invalid {name}: {value!r}") from exc
    require(parsed.tzinfo is not None, f"{name} is timezone-naive")
    return parsed


def git_paths(*args: str) -> list[Path]:
    raw = subprocess.check_output(["git", "-C", str(ROOT), *args])
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def current_runtime_map() -> dict[str, tuple[str, int]]:
    paths = set(git_paths("ls-files", "-z", "--", *RUNTIME_ROOTS))
    paths.update(git_paths("ls-files", "--others", "--exclude-standard", "-z", "--", *RUNTIME_ROOTS))
    selected = sorted(
        path for path in paths
        if path.suffix.lower() in RUNTIME_EXTENSIONS and "__pycache__" not in path.parts
    )
    result: dict[str, tuple[str, int]] = {}
    for relative in selected:
        source = (ROOT / relative).resolve()
        require(source.is_file() and ROOT in source.parents, f"invalid runtime path: {relative}")
        result[relative.as_posix()] = (sha256_file(source), source.stat().st_size)
    require(bool(result), "current runtime map is empty")
    return result


def map_digest(mapping: dict[str, tuple[str, int]]) -> str:
    rows = [
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, (digest, size) in sorted(mapping.items())
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_map(
    path: Path,
    schema: int,
    roots_only: bool = True,
    *,
    require_original: bool = True,
) -> tuple[dict[str, tuple[str, int]], dict]:
    manifest = load_json(path)
    require(manifest.get("schema_version") == schema, f"wrong manifest schema: {path}")
    entries = manifest.get("runtime_files")
    require(isinstance(entries, list) and entries, f"manifest runtime_files missing: {path}")
    require(len(entries) == int(manifest.get("runtime_file_count", -1)), "manifest file count mismatch")
    mapping: dict[str, tuple[str, int]] = {}
    repository = Path(str(manifest.get("repository_root", ""))).resolve()
    for entry in entries:
        require(isinstance(entry, dict), "manifest entry is not an object")
        name = str(entry.get("path", ""))
        if roots_only and not name.startswith(tuple(root + "/" for root in RUNTIME_ROOTS)):
            continue
        relative = Path(name)
        require(name and not relative.is_absolute() and ".." not in relative.parts, f"unsafe manifest path: {name}")
        require(name not in mapping, f"duplicate manifest path: {name}")
        digest = str(entry.get("sha256", ""))
        size = int(entry.get("size_bytes", -1))
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and size >= 0, f"bad manifest entry: {name}")
        original = (repository / relative).resolve()
        snapshot = (path.parent / str(entry.get("snapshot", ""))).resolve()
        require(repository in original.parents, f"manifest original escapes repository: {name}")
        require(path.parent in snapshot.parents and snapshot.is_file(), f"missing manifest snapshot: {name}")
        if require_original:
            require(original.is_file(), f"missing manifest original: {name}")
            require(sha256_file(original) == digest and original.stat().st_size == size, f"runtime byte drift: {name}")
        require(sha256_file(snapshot) == digest and snapshot.stat().st_size == size, f"snapshot byte drift: {name}")
        mapping[name] = (digest, size)
    require(bool(mapping), "manifest root map is empty")
    environment = (path.parent / str(manifest.get("python_environment", ""))).resolve()
    require(environment.is_file(), "Python environment receipt missing")
    require(sha256_file(environment) == manifest.get("python_environment_sha256"), "Python environment receipt drift")
    return mapping, manifest


def verify_p1() -> dict[str, Any]:
    require(P1_REPORT.is_file() and sha256_file(P1_REPORT) == P1_REPORT_SHA, "P1c report identity mismatch")
    report = load_json(P1_REPORT)
    require(report.get("schema_version") == 1, "P1c report schema mismatch")
    require(report.get("scope") == "ref5in_learning_viability_engineering_smoke", "P1c scope mismatch")
    require(report.get("verdict") == "PASS", "P1c did not PASS")
    require(report.get("performance_claim_allowed") is False, "P1c claim scope changed")
    require(report.get("next_step") == "held_out_70bar_eval_only", "P1c next-step mismatch")
    require(report.get("expected_epochs") == 900, "P1c epoch budget mismatch")
    require(math.isclose(float(report.get("expected_learning_rate", 0)), 1.5e-5, abs_tol=1e-12), "P1c LR mismatch")
    checks = report.get("checks") or {}
    require(checks and all(item.get("pass") is True for item in checks.values()), "not every P1c check passed")
    require(P1_CHECKPOINT.is_file() and sha256_file(P1_CHECKPOINT) == P1_CHECKPOINT_SHA, "P1c checkpoint identity mismatch")
    require(report.get("checkpoint_sha256") == P1_CHECKPOINT_SHA, "P1c report checkpoint SHA mismatch")
    require(Path(str(report.get("checkpoint", ""))).name == P1_CHECKPOINT.name, "P1c report checkpoint path mismatch")
    run = P1_CHECKPOINT.parents[1]
    require((run / ".aerial_training_finished").is_file(), "P1c finish marker missing")
    summary = load_json(run / "aerial_run/run_summary.json")
    require(summary.get("exit_reason") == "max_epochs" and summary.get("last_epoch") == 900, "P1c terminal summary mismatch")

    import torch
    checkpoint = torch.load(str(P1_CHECKPOINT), map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    expected = {
        "cfg_training_seed": 197,
        "cfg_training_num_envs": 128,
        "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
        "cfg_training_task": "navrl_task",
        "cfg_training_sim": "base_sim",
        "cfg_training_profile": "main",
        "cfg_ppo_horizon": 32,
        "cfg_robot_name": "navrl_ref5in_quad",
        "cfg_robot_contract_version": 1,
        "cfg_robot_config_sha256": ROBOT_CONFIG_SHA,
        "cfg_robot_asset_sha256": ROBOT_URDF_SHA,
        "cfg_obstacle_selector": "cluster_sector",
        "cfg_action_policy": "squashed_gaussian",
        "cfg_speed_governor_mode": "off",
        "cfg_perception_perturb": False,
        "n_bars_active": 70,
        "k_min_cur": 20.0,
        "k_max_cur": 28.0,
        "cfg_training_source_manifest_sha256": P1_MANIFEST_SHA,
        "cfg_training_source_runtime_file_count": 312,
        "cfg_training_source_git_dirty": False,
    }
    mismatches = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
    require(not mismatches, f"P1c checkpoint contract mismatch: {mismatches}")
    require(checkpoint.get("epoch") == 900 and checkpoint.get("frame") == 3_686_400, "P1c checkpoint epoch/frame mismatch")
    require(Path(str(state.get("cfg_training_source_manifest", ""))).resolve() == P1_MANIFEST, "P1c manifest path mismatch")
    require(P1_MANIFEST.is_file() and sha256_file(P1_MANIFEST) == P1_MANIFEST_SHA, "P1c manifest identity mismatch")
    training_map, manifest = manifest_map(P1_MANIFEST, 1)
    require(map_digest(training_map) == P1_RUNTIME_MAP_SHA, "P1c runtime-map digest mismatch")
    require(manifest.get("git_dirty") is False, "P1c runtime source was dirty")
    require(manifest.get("python_environment_sha256") == PYTHON_ENV_SHA, "P1c Python environment mismatch")
    require(current_runtime_map() == training_map, "current runtime byte map differs from P1c")
    require(sha256_file(EVALUATOR) == EVALUATOR_SHA, "generic evaluator differs from P1c")
    return {"report": report, "checkpoint": checkpoint, "state": state, "runtime_map": training_map}


def canonical_env(result_dir: Path, preflight: bool = False) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("NAVRL_") and key not in {
            "AERIAL_RUN_TAG", "AERIAL_GYM_SIM_NAME", "GPU4GB", "NUM_ENVS", "FILE", "TASK",
            "PYTHON", "CKPT", "HEADLESS", "PLAY_GAMES_NUM", "PYTHONPATH", "PYTHONHOME",
        }
    }
    env.update({
        "PYTHON": str(PYTHON), "PYTHONNOUSERSITE": "1", "GPU4GB": "0",
        "NAVRL_SEED": "313", "NAVRL_V2_DENSITIES": "70",
        "NAVRL_V2_ACTION_MODE": "deterministic", "NAVRL_EVAL_REFLECTION_MODE": "original",
        "NAVRL_SPEED_GOVERNOR": "off", "NAVRL_SPEED_GOVERNOR_FIXED_MPS": "2",
        "NAVRL_SPEED_GOVERNOR_FREE_MPS": "3.53553390593",
        "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M": "0.45", "NAVRL_SPEED_GOVERNOR_MARGIN_M": "0.45",
        "NAVRL_SPEED_GOVERNOR_SLOW_M": "3", "NAVRL_SPEED_GOVERNOR_RELEASE_M": "5",
        "NAVRL_SPEED_GOVERNOR_TTC_S": "1", "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2": "2",
        "NAVRL_SPEED_GOVERNOR_REACTION_S": "0.1", "NAVRL_PERCEPTION_PERTURB": "0",
        "NAVRL_TILT_COMP": "1", "NAVRL_DETECTOR_MIN_PIXELS": "2",
        "NAVRL_DETECTOR_THRESHOLD": "0.55", "NAVRL_DETECTION_DROPOUT": "0.3",
        "NAVRL_DETECTION_LATENCY_S": "0", "NAVRL_RANGE_ERROR_M": "0",
        "NAVRL_LATENCY_COMPENSATE": "0", "NAVRL_LATENCY_LIDAR_BACKUP": "0",
        "NAVRL_LATENCY_OBSTACLE_FIX": "off", "NAVRL_LATENCY_EGO_MOTION_FIX": "1",
        "NAVRL_POSE_CLOCK_OFFSET_S": "0", "NAVRL_POSE_NOISE_POS_M": "0",
        "NAVRL_POSE_NOISE_YAW_DEG": "0", "NAVRL_POSE_NOISE_SEED": "9163",
        "NAVRL_TARGET_MASK_BACKFILL": "0", "NAVRL_LIDAR_TARGET_ASSOC": "1",
        "NAVRL_LIDAR_RANGE_ONLY_UPDATE": "0", "NAVRL_LIDAR_ASSOC_GATE_M": "0",
        "NAVRL_LIDAR_SILENT_CORRECT": "0", "NAVRL_APP_HUE_DEG": "0",
        "NAVRL_APP_LIGHT_GAIN": "0", "NAVRL_APP_ALBEDO_JITTER": "0",
        "NAVRL_APP_TEXTURE_STD": "0", "NAVRL_APP_MOTION_BLUR": "0",
        "NAVRL_CAM_MOUNT_ROT_DEG": "0", "NAVRL_CAM_MOUNT_TRANS_M": "0",
        "NAVRL_CAM_FOV_SCALE_ERR": "0", "NAVRL_RGB_NOISE_STD": "0.015",
        "NAVRL_DEPTH_NOISE_STD": "0.02", "NAVRL_V2_RESULT_DIR": str(result_dir),
        "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
    })
    if preflight:
        env["NAVRL_PREFLIGHT_ONLY"] = "1"
    return env


def run_evaluator(checkpoint: Path, result_dir: Path, preflight: bool = False) -> None:
    require(PYTHON.is_file(), f"canonical Python missing: {PYTHON}")
    command = ["bash", str(EVALUATOR), str(checkpoint), "2049"]
    subprocess.run(command, cwd=ROOT, env=canonical_env(result_dir, preflight), check=True)


def verify_strata(payload: dict, actual: int, captured: int) -> None:
    strata = payload.get("strata") or {}
    for name in ("distance", "speed", "pattern"):
        cells = strata.get(name) or {}
        require(isinstance(cells, dict) and cells, f"missing {name} strata")
        require(sum(int(cell.get("episodes", -1)) for cell in cells.values()) == actual, f"{name} episode accounting mismatch")
        require(sum(int(cell.get("successes", -1)) for cell in cells.values()) == captured, f"{name} capture accounting mismatch")


def verify_cell(
    result_dir: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    robot: str,
    *,
    require_current_runtime: bool = True,
) -> dict[str, Any]:
    result_path = result_dir / "70bars.json"
    receipt_path = result_dir / "70bars.receipt.json"
    log_path = result_dir / "70bars.log"
    snapshot = result_dir / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, log_path, snapshot):
        require(path.is_file(), f"missing P2 artifact: {path}")
    result = load_json(result_path)
    receipt = load_json(receipt_path)
    require(result.get("schema_version") == 1 and receipt.get("schema_version") == 2, "evaluation schema mismatch")
    require(receipt.get("producer") == "eval_navrl_v2_density_sweep.sh", "unexpected evaluator producer")
    require(sha256_file(checkpoint) == checkpoint_sha, "source checkpoint changed")
    require(sha256_file(snapshot) == checkpoint_sha, "evaluated checkpoint snapshot mismatch")
    require(receipt.get("source_checkpoint_sha256") == checkpoint_sha, "receipt source checkpoint mismatch")
    require(receipt.get("evaluated_checkpoint_snapshot_sha256") == checkpoint_sha, "receipt snapshot mismatch")
    require(Path(str(receipt.get("source_checkpoint", ""))).resolve() == checkpoint, "receipt source path mismatch")
    require(Path(str(receipt.get("evaluated_checkpoint_snapshot", ""))).resolve() == snapshot, "receipt snapshot path mismatch")
    require(Path(str(receipt.get("result_json", ""))).resolve() == result_path, "receipt result path mismatch")
    require(Path(str(receipt.get("log_file", ""))).resolve() == log_path, "receipt log path mismatch")
    require(sha256_file(result_path) == receipt.get("result_sha256"), "result hash mismatch")
    require(sha256_file(log_path) == receipt.get("log_sha256"), "log hash mismatch")
    require(Path(str(result.get("evaluation_receipt", ""))).resolve() == receipt_path, "result receipt path mismatch")
    require(result.get("evaluator_script_sha256") == EVALUATOR_SHA, "result evaluator hash mismatch")
    require(receipt.get("evaluator_script_sha256") == EVALUATOR_SHA, "receipt evaluator hash mismatch")
    nonce = str(receipt.get("evaluation_nonce", ""))
    require(re.fullmatch(r"[0-9a-f]{64}", nonce) is not None, "malformed evaluation nonce")
    require((result.get("condition") or {}).get("evaluation_nonce") == nonce, "evaluation nonce mismatch")
    require(parse_time(receipt.get("started_at_utc"), "started_at") <= parse_time(receipt.get("completed_at_utc"), "completed_at"), "evaluation timestamps reversed")

    manifest_path = Path(str(receipt.get("runtime_source_manifest", ""))).resolve()
    require(manifest_path == SOURCE_BUNDLE / "source_manifest.json", "non-canonical evaluation manifest")
    require(sha256_file(manifest_path) == receipt.get("runtime_source_manifest_sha256"), "evaluation manifest hash mismatch")
    require(result.get("runtime_source_manifest_sha256") == receipt.get("runtime_source_manifest_sha256"), "result/receipt manifest mismatch")
    eval_map, eval_manifest = manifest_map(
        manifest_path, 2, require_original=require_current_runtime
    )
    training_map, _ = manifest_map(
        P1_MANIFEST, 1, require_original=require_current_runtime
    )
    require(eval_map == training_map, "evaluation runtime differs from P1c")
    if require_current_runtime:
        require(current_runtime_map() == training_map, "current runtime differs from P1c/P2")
    require(map_digest(eval_map) == P1_RUNTIME_MAP_SHA, "evaluation runtime-map digest mismatch")
    require(eval_manifest.get("python_environment_sha256") == PYTHON_ENV_SHA, "evaluation Python environment differs from P1c")
    require(receipt.get("python_environment_manifest_sha256") == PYTHON_ENV_SHA, "receipt Python environment mismatch")
    require(receipt.get("source_detector_checkpoint") == "" and receipt.get("evaluated_detector_snapshot") == "" and receipt.get("evaluated_detector_snapshot_sha256") == "", "P2 must use analytic detector")

    actual = int(result.get("actual_episodes", -1))
    requested = int(result.get("requested_episodes", -1))
    outcome = result.get("outcome") or {}
    counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
    require(requested == 2049 and actual >= requested, "episode count contract mismatch")
    require(all(count >= 0 for count in counts) and sum(counts) == actual, "outcome accounting mismatch")
    for name, count in zip(("capture_rate", "crash_rate", "timeout_rate"), counts):
        require(math.isclose(float(outcome.get(name)), count / actual, abs_tol=1e-12), f"{name} disagrees with counts")
    verify_strata(result, actual, counts[0])
    bearing = (result.get("strata") or {}).get("initial_target_bearing") or {}
    require(sum(int(cell.get("episodes", -1)) for cell in bearing.values()) == actual, "bearing episode accounting mismatch")
    for key, expected in zip(("captured", "crash", "timeout"), counts):
        require(sum(int(cell.get(key, -1)) for cell in bearing.values()) == expected, f"bearing {key} accounting mismatch")

    condition = result.get("condition") or {}
    exact_condition = {
        "seed": 313, "bars": 70, "num_envs": 128, "action_selection": "deterministic",
        "reflection_mode": "original", "runtime_sim_config_class": "BaseSimConfig",
        "physics_dt_s": 0.01, "physics_substeps": 1, "physics_steps_per_rl_step": 10,
        "rl_step_dt_s": 0.1, "episode_len_steps": 600, "goal_dist_min_m": 6.0,
        "goal_dist_max_m": 28.0, "full_goal_distribution": True,
        "fov_curriculum_saturated": True, "target_speed_mode": "uniform",
        "target_speed_min_mps": 0.3, "target_speed_max_mps": 1.5,
        "target_pattern": "mixed", "robot_name": robot,
        "policy_output_dim": 4, "policy_z_output_overwritten_by_altitude_pi": True,
        "policy_z_persisted_in_prev_action_observation": True,
    }
    if robot == "navrl_ref5in_quad":
        exact_condition.update({"robot_config_sha256": ROBOT_CONFIG_SHA, "robot_asset_sha256": ROBOT_URDF_SHA})
    mismatches = {key: (condition.get(key), expected) for key, expected in exact_condition.items() if condition.get(key) != expected}
    require(not mismatches, f"held-out condition mismatch: {mismatches}")
    contract = result.get("v2_evaluation_contract") or {}
    exact_contract = {
        "schema_version": 2, "episode_limit_steps": 600, "episode_limit_comparator": "gte",
        "timeout_observed_at_step": 600, "runtime_sim": "base_sim", "runtime_profile": "main",
        "runtime_num_envs": 128, "action_selection": "deterministic", "reflection_mode": "original",
        "speed_governor_mode": "off", "arena_xy_m": 40.0, "goal_dist_min_m": 6.0,
        "goal_dist_max_m": 28.0, "full_goal_distribution": True,
        "fov_curriculum_saturated": True, "target_speed_distribution": "uniform",
        "target_speed_mps": None, "target_speed_min_mps": 0.3, "target_speed_max_mps": 1.5,
        "target_pattern": "mixed", "lidar_beams": [4, 72], "lidar_range_m": 12.0,
        "obstacle_tokens": 8, "obstacle_fov_deg": 240.0, "obstacle_selector": "cluster_sector",
        "detector_checkpoint_sha256": "", "detector_min_pixels": 2, "detector_threshold": 0.55,
        "perception_perturb": False, "detection_dropout": 0.3, "detection_dropout_active": 0.0,
        "detection_latency_s": 0.0, "range_error_m": 0.0, "rgb_noise_std": 0.015,
        "depth_noise_std": 0.02, "max_tilt_deg": 45.0, "tilt_comp": True,
        "sim_physics_contract": "base_sim_dt0.01", "pursuer_speed_limit_semantics": "per_axis_xy",
        "policy_output_dim": 4, "seed": 313,
    }
    mismatches = {key: (contract.get(key), expected) for key, expected in exact_contract.items() if contract.get(key) != expected}
    require(not mismatches, f"v2 evaluation contract mismatch: {mismatches}")
    action = result.get("action") or {}
    require(action.get("policy") == "squashed_gaussian" and int(action.get("samples", 0)) > 0, "action diagnostics invalid")
    require(action.get("task_input_oob_rate") == [0.0, 0.0, 0.0, 0.0], "task input OOB is nonzero")
    timeouts = ((result.get("speed_governor") or {}).get("outcome_steps") or {}).get("timeout") or {}
    if counts[2]:
        require(int(timeouts.get("count", -1)) == counts[2], "timeout diagnostic count mismatch")
        require(all(float(timeouts.get(key, -1)) == 600.0 for key in ("mean", "p10", "p50", "p90")), "timeout not at exact action 600")
    return {
        "result": result, "receipt": receipt, "actual": actual, "counts": counts,
        "result_sha256": sha256_file(result_path), "receipt_sha256": sha256_file(receipt_path),
        "log_sha256": sha256_file(log_path), "manifest_sha256": sha256_file(manifest_path),
        "runtime_git_commit": eval_manifest.get("git_commit"),
    }


def make_attestation(decision: dict[str, Any], anchor: dict[str, Any] | None) -> dict[str, Any]:
    captured, crash, timeout = decision["counts"]
    actual = decision["actual"]
    performance_pass = captured * 100 >= 65 * actual and crash * 100 <= 33 * actual and timeout * 100 <= 5 * actual
    checks = {
        "capture_ge_0p65": {"pass": captured * 100 >= 65 * actual, "detail": f"{captured}/{actual}"},
        "crash_le_0p33": {"pass": crash * 100 <= 33 * actual, "detail": f"{crash}/{actual}"},
        "timeout_le_0p05": {"pass": timeout * 100 <= 5 * actual, "detail": f"{timeout}/{actual}"},
        "descriptive_anchor_complete": {"pass": anchor is not None if performance_pass else True, "detail": "not a decision-gate input"},
    }
    passed = performance_pass and all(item["pass"] for item in checks.values())
    payload = {
        "schema_version": 1, "producer": "tools/attest_navrl_ref5in_p2.py",
        "scope": "ref5in_p2_heldout_70bar_decision", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if passed else "FAIL", "performance_claim_allowed": False,
        "unlocks": "manual_p3_seed211_only" if passed else "none",
        "p1c": {"report": str(P1_REPORT), "report_sha256": P1_REPORT_SHA,
            "checkpoint": str(P1_CHECKPOINT), "checkpoint_sha256": P1_CHECKPOINT_SHA,
            "epoch": 900, "frame": 3_686_400, "training_source_manifest": str(P1_MANIFEST),
            "training_source_manifest_sha256": P1_MANIFEST_SHA,
            "training_runtime_map_sha256": P1_RUNTIME_MAP_SHA, "python_environment_sha256": PYTHON_ENV_SHA},
        "evaluation_runtime": {"manifest": str(SOURCE_BUNDLE / "source_manifest.json"),
            "manifest_sha256": decision["manifest_sha256"], "runtime_map_sha256": P1_RUNTIME_MAP_SHA,
            "matches_p1c_runtime": True, "python_environment_sha256": PYTHON_ENV_SHA,
            "evaluator_script_sha256": EVALUATOR_SHA, "runtime_git_commit": decision["runtime_git_commit"],
            "attestor_script_sha256": sha256_file(Path(__file__)),
            "launcher_sha256": sha256_file(ROOT / "tools/run_navrl_ref5in_p2.sh")},
        "decision_cell": {"result_json": str(DECISION_DIR / "70bars.json"),
            "result_sha256": decision["result_sha256"], "receipt_json": str(DECISION_DIR / "70bars.receipt.json"),
            "receipt_sha256": decision["receipt_sha256"], "log_sha256": decision["log_sha256"],
            "condition": decision["result"]["condition"], "outcome": decision["result"]["outcome"]},
        "descriptive_anchor": None if anchor is None else {
            "checkpoint": str(LEGACY_CHECKPOINT), "checkpoint_sha256": LEGACY_CHECKPOINT_SHA,
            "robot_name": "navrl_quad", "gate_contribution": "none",
            "provenance_limit": "contract-v0/no full training-source receipt",
            "result_json": str(ANCHOR_DIR / "70bars.json"), "result_sha256": anchor["result_sha256"],
            "receipt_sha256": anchor["receipt_sha256"], "condition": anchor["result"]["condition"],
            "outcome": anchor["result"]["outcome"]},
        "checks": checks,
        "limitations": ["one training seed", "one held-out evaluation seed", "70-bar decision cell only",
            "legacy anchor is descriptive and has weaker training provenance", "no hardware validation"],
    }
    return payload


def verify_attestation(path: Path = ATTESTATION) -> dict[str, Any]:
    payload = load_json(path)
    require(payload.get("schema_version") == 1 and payload.get("producer") == "tools/attest_navrl_ref5in_p2.py", "attestation identity mismatch")
    require(payload.get("scope") == "ref5in_p2_heldout_70bar_decision", "attestation scope mismatch")
    # Historical verification proves what was evaluated from the immutable checkpoint, source
    # bundle and receipts.  It must not fail merely because development continued afterwards.
    # The run path still calls verify_cell() with the default current-source check both before and
    # after evaluation, so source drift during a new P2 remains fail-closed.
    decision = verify_cell(
        DECISION_DIR,
        P1_CHECKPOINT,
        P1_CHECKPOINT_SHA,
        "navrl_ref5in_quad",
        require_current_runtime=False,
    )
    anchor_payload = payload.get("descriptive_anchor")
    anchor = None
    if anchor_payload is not None:
        anchor = verify_cell(
            ANCHOR_DIR,
            LEGACY_CHECKPOINT,
            LEGACY_CHECKPOINT_SHA,
            "navrl_quad",
            require_current_runtime=False,
        )
    regenerated = make_attestation(decision, anchor)
    # These identify the producer binaries recorded at evaluation time.  A newer verifier cannot
    # reproduce their own old hashes, so validate their shape and retain the recorded values for
    # the semantic comparison below.
    recorded_runtime = payload.get("evaluation_runtime") or {}
    for field in ("attestor_script_sha256", "launcher_sha256"):
        value = recorded_runtime.get(field)
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                f"malformed recorded {field}")
        regenerated["evaluation_runtime"][field] = value
    for key in ("verdict", "performance_claim_allowed", "unlocks", "p1c", "evaluation_runtime", "decision_cell", "descriptive_anchor", "checks", "limitations"):
        require(payload.get(key) == regenerated.get(key), f"attestation field changed: {key}")
    return payload


def preflight() -> None:
    verify_p1()
    require(not OUTPUT_ROOT.exists(), f"P2 output already exists: {OUTPUT_ROOT}")
    run_evaluator(P1_CHECKPOINT, DECISION_DIR, preflight=True)
    run_evaluator(LEGACY_CHECKPOINT, ANCHOR_DIR, preflight=True)
    require(not OUTPUT_ROOT.exists(), "preflight created output artifacts")
    print("[ref5in-p2] PREFLIGHT PASS")


def run() -> int:
    verify_p1()
    require(not OUTPUT_ROOT.exists(), f"refusing to overwrite P2 output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    run_evaluator(P1_CHECKPOINT, DECISION_DIR)
    decision = verify_cell(DECISION_DIR, P1_CHECKPOINT, P1_CHECKPOINT_SHA, "navrl_ref5in_quad")
    captured, crash, timeout = decision["counts"]
    actual = decision["actual"]
    primary_pass = captured * 100 >= 65 * actual and crash * 100 <= 33 * actual and timeout * 100 <= 5 * actual
    anchor = None
    if primary_pass:
        run_evaluator(LEGACY_CHECKPOINT, ANCHOR_DIR)
        anchor = verify_cell(ANCHOR_DIR, LEGACY_CHECKPOINT, LEGACY_CHECKPOINT_SHA, "navrl_quad")
    payload = make_attestation(decision, anchor)
    ATTESTATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_attestation()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["verdict"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            preflight()
            return 0
        if args.command == "verify":
            print(json.dumps(verify_attestation(), indent=2, sort_keys=True))
            return 0
        return run()
    except (ContractError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"[ref5in-p2] REFUSING: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
