#!/usr/bin/env python3
"""Apply the preregistered engineering gates to one ref5in fresh smoke run.

This script intentionally separates a learning-viability decision from a performance claim.  It
reads the terminal checkpoint, completion marker, TensorBoard safety scalars and the exact outcome
counts printed by the trainer.  A PASS only unlocks a held-out 70-bar evaluation; it does not
justify full training or any hardware claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
LOG_ROOT = RL_ROOT / "train_session_logs"

OUTCOME_RE = re.compile(
    r"captured \(success\)\s*:\s*[0-9.]+%\s*\((\d+)/(\d+)\).*?"
    r"crash\s*:\s*[0-9.]+%\s*\((\d+)/(\d+)\).*?"
    r"timeout \(no capture\)\s*:\s*[0-9.]+%\s*\((\d+)/(\d+)\)",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(run: Path, epoch: int) -> Path:
    candidates = []
    for path in (run / "nn").glob("last_gen_ppo_ep_*.pth"):
        match = re.search(r"last_gen_ppo_ep_(\d+)", path.name)
        if match and int(match.group(1)) == epoch:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no epoch-{epoch} last checkpoint under {run / 'nn'}")
    # The trainer now avoids duplicate terminal saves.  Reject ambiguity rather than picking a
    # favorable spelling when an older implementation produces two endpoint files.
    if len(candidates) != 1:
        names = ", ".join(sorted(path.name for path in candidates))
        raise RuntimeError(f"ambiguous epoch-{epoch} checkpoints: {names}")
    return candidates[0]


def matching_log(run_name: str) -> Path:
    matches = []
    for path in LOG_ROOT.glob("*.log"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if run_name in text:
            matches.append((path.stat().st_mtime_ns, path, text))
    if not matches:
        raise RuntimeError(f"no session log contains run name {run_name!r}")
    _mtime, path, _text = max(matches, key=lambda row: row[0])
    return path


def outcome_window(path: Path, expected_epochs: int, last_n: int = 100) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in OUTCOME_RE.finditer(text):
        capture, cap_n, crash, crash_n, timeout, timeout_n = map(int, match.groups())
        if not (cap_n == crash_n == timeout_n):
            raise RuntimeError("trainer outcome block has inconsistent denominators")
        if capture + crash + timeout != cap_n:
            raise RuntimeError("trainer outcome block does not account for every termination")
        rows.append((capture, crash, timeout, cap_n))
    if len(rows) != expected_epochs:
        raise RuntimeError(
            f"outcome accounting has {len(rows)} blocks; expected exactly {expected_epochs}"
        )
    if len(rows) < last_n:
        raise RuntimeError(f"only {len(rows)} complete outcome blocks; need {last_n}")
    tail = rows[-last_n:]
    captured = sum(row[0] for row in tail)
    crash = sum(row[1] for row in tail)
    timeout = sum(row[2] for row in tail)
    episodes = sum(row[3] for row in tail)
    return {
        "epochs": last_n,
        "episodes": episodes,
        "captured": captured,
        "crash": crash,
        "timeout": timeout,
        "capture_rate": captured / episodes,
        "crash_rate": crash / episodes,
        "timeout_rate": timeout / episodes,
        "all_outcome_blocks": len(rows),
    }


def tensorboard_summary(run: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("tensorboard is unavailable; run with the aerialgym Python") from exc
    acc = EventAccumulator(str(run / "summaries"), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    wanted = (
        "ppo/kl",
        "ppo/behavior_kl_audit_max",
        "ppo/epoch_rollback_total",
        "ppo/kl_skipped_minibatches",
        "policy_action/raw_oob_x",
        "policy_action/raw_oob_y",
        "policy_action/raw_oob_z",
        "policy_action/raw_oob_yaw",
    )
    result = {}
    for tag in wanted:
        if tag not in tags:
            result[tag] = {"count": 0, "max": None, "last": None}
            continue
        values = [float(event.value) for event in acc.Scalars(tag)]
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"TensorBoard tag {tag} is empty or non-finite")
        result[tag] = {"count": len(values), "max": max(values), "last": values[-1]}
    all_scalar_tags = sorted(tags)
    nonfinite_tags = []
    empty_tags = []
    for tag in all_scalar_tags:
        values = [float(event.value) for event in acc.Scalars(tag)]
        if not values:
            empty_tags.append(tag)
        elif not all(math.isfinite(value) for value in values):
            nonfinite_tags.append(tag)
    result["all_scalars"] = {
        "tag_count": len(all_scalar_tags),
        "empty_tags": empty_tags,
        "nonfinite_tags": nonfinite_tags,
    }
    return result


def verify_source_manifest(manifest_path: Path, expected_sha256: str) -> dict:
    """Re-hash the original runtime bytes, immutable snapshot and Python receipt."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"source manifest is missing: {manifest_path}")
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_sha256:
        raise RuntimeError("source manifest SHA-256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported training source manifest schema")
    repository = Path(manifest["repository_root"]).resolve()
    entries = manifest.get("runtime_files") or []
    if not entries or len(entries) != int(manifest.get("runtime_file_count", -1)):
        raise RuntimeError("invalid runtime source accounting")
    for entry in entries:
        original = (repository / entry["path"]).resolve()
        snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
        if repository not in original.parents:
            raise RuntimeError(f"runtime source escapes repository: {entry['path']}")
        expected = entry["sha256"]
        if not original.is_file() or sha256_file(original) != expected:
            raise RuntimeError(f"runtime source changed: {entry['path']}")
        if not snapshot.is_file() or sha256_file(snapshot) != expected:
            raise RuntimeError(f"runtime snapshot changed: {entry['snapshot']}")
    environment = (manifest_path.parent / manifest["python_environment"]).resolve()
    if not environment.is_file() or sha256_file(environment) != manifest.get(
        "python_environment_sha256"
    ):
        raise RuntimeError("Python environment receipt changed")
    return {
        "manifest_sha256": actual_sha,
        "git_commit": manifest.get("git_commit"),
        "git_dirty": bool(manifest.get("git_dirty")),
        "runtime_file_count": len(entries),
        "verified": True,
    }


def all_tensors_finite(value, prefix="checkpoint") -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is unavailable; run with the aerialgym Python") from exc
    failures = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            failures.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            failures.extend(all_tensors_finite(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(all_tensors_finite(item, f"{prefix}[{index}]"))
    return failures


def bool_check(checks: dict, name: str, passed: bool, detail: str) -> None:
    checks[name] = {"pass": bool(passed), "detail": detail}


def analyze(run: Path, expected_epochs: int, expected_learning_rate: float) -> dict:
    import torch

    run = run.expanduser().resolve()
    if not run.is_dir():
        raise RuntimeError(f"run directory is missing: {run}")
    checkpoint_path = latest_checkpoint(run, expected_epochs)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    summary_path = run / "aerial_run/run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = run / ".aerial_training_finished"
    log = matching_log(run.name)
    outcome = outcome_window(log, expected_epochs=expected_epochs)
    tb = tensorboard_summary(run)

    config_path = ROOT / "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py"
    asset_path = ROOT / "resources/robots/quad/quad_navrl_ref5in.urdf"
    checks = {}
    bool_check(
        checks,
        "normal_terminal_completion",
        marker.is_file()
        and summary.get("exit_reason") == "max_epochs"
        and int(summary.get("last_epoch", -1)) == expected_epochs
        and int(checkpoint.get("epoch", -1)) == expected_epochs,
        f"summary={summary.get('exit_reason')} last={summary.get('last_epoch')} "
        f"checkpoint={checkpoint.get('epoch')} marker={marker.is_file()}",
    )
    nonfinite = all_tensors_finite(checkpoint)
    bool_check(checks, "checkpoint_all_finite", not nonfinite, str(nonfinite[:8]))
    all_scalars = tb["all_scalars"]
    bool_check(
        checks,
        "all_tensorboard_scalars_finite",
        not all_scalars["empty_tags"] and not all_scalars["nonfinite_tags"],
        json.dumps(all_scalars, sort_keys=True),
    )

    kl = tb["ppo/kl"]
    behavior_kl = tb["ppo/behavior_kl_audit_max"]
    bool_check(
        checks,
        "ppo_kl_below_0p04",
        kl["count"] == expected_epochs
        and kl["max"] is not None
        and kl["max"] < 0.04,
        json.dumps(kl, sort_keys=True),
    )
    bool_check(
        checks,
        "behavior_kl_below_0p04",
        behavior_kl["count"] == expected_epochs
        and behavior_kl["max"] is not None
        and behavior_kl["max"] < 0.04,
        json.dumps(behavior_kl, sort_keys=True),
    )
    rollback = tb["ppo/epoch_rollback_total"]
    skipped = tb["ppo/kl_skipped_minibatches"]
    bool_check(
        checks,
        "no_rollback_or_skipped_minibatches",
        rollback["count"] == expected_epochs
        and skipped["count"] == expected_epochs
        and rollback["max"] == 0.0
        and skipped["max"] == 0.0,
        f"rollback={rollback} skipped={skipped}",
    )
    raw_oob = {axis: tb[f"policy_action/raw_oob_{axis}"] for axis in ("x", "y", "z", "yaw")}
    bool_check(
        checks,
        "all_axis_raw_oob_zero",
        all(
            item["count"] == expected_epochs and item["max"] == 0.0
            for item in raw_oob.values()
        ),
        json.dumps(raw_oob, sort_keys=True),
    )

    source_manifest_text = str(state.get("cfg_training_source_manifest", ""))
    source_manifest = Path(source_manifest_text) if source_manifest_text else None
    source_sha = str(state.get("cfg_training_source_manifest_sha256", ""))
    source_verification = None
    source_error = ""
    try:
        if not source_manifest or len(source_sha) != 64:
            raise RuntimeError("checkpoint source manifest path/SHA is missing")
        source_verification = verify_source_manifest(source_manifest, source_sha)
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        source_error = str(exc)
    source_ok = bool(
        source_verification
        and source_verification["verified"]
        and source_verification["git_dirty"] is False
        and state.get("cfg_training_source_git_dirty") is False
        and int(state.get("cfg_training_source_runtime_file_count", 0))
        == int(source_verification["runtime_file_count"])
    )
    bool_check(
        checks,
        "clean_training_source_receipt",
        source_ok,
        f"manifest={source_manifest_text!r} sha={source_sha} "
        f"dirty={state.get('cfg_training_source_git_dirty')} "
        f"files={state.get('cfg_training_source_runtime_file_count')} "
        f"verification={source_verification!r} error={source_error!r}",
    )
    robot_ok = (
        int(state.get("cfg_robot_contract_version", 0)) == 1
        and state.get("cfg_robot_name") == "navrl_ref5in_quad"
        and state.get("cfg_robot_config_sha256") == sha256_file(config_path)
        and state.get("cfg_robot_asset_sha256") == sha256_file(asset_path)
    )
    bool_check(
        checks,
        "ref5in_runtime_identity",
        robot_ok,
        f"robot={state.get('cfg_robot_name')} config={state.get('cfg_robot_config_sha256')} "
        f"urdf={state.get('cfg_robot_asset_sha256')}",
    )
    exact_contract = {
        "cfg_training_seed": 197,
        "cfg_training_num_envs": 128,
        "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
        "cfg_training_task": "navrl_task",
        "cfg_training_sim": "base_sim",
        "cfg_training_profile": "main",
        "cfg_ppo_horizon": 32,
        "cfg_obstacle_selector": "cluster_sector",
        "cfg_action_policy": "squashed_gaussian",
        "cfg_speed_governor_mode": "off",
        "cfg_perception_perturb": False,
        "n_bars_active": 70,
        "num_task_steps": expected_epochs * 32,
    }
    contract_mismatches = {
        key: {"actual": state.get(key), "expected": expected}
        for key, expected in exact_contract.items()
        if state.get(key) != expected
    }
    initial_lr = float(state.get("cfg_action_learning_rate", float("nan")))
    current_lr = float(state.get("current_action_learning_rate", float("nan")))
    expected_frame = expected_epochs * 32 * 128
    smoke_contract_ok = (
        not contract_mismatches
        and math.isclose(initial_lr, expected_learning_rate, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(current_lr, expected_learning_rate, rel_tol=0.0, abs_tol=1e-12)
        and int(checkpoint.get("frame", -1)) == expected_frame
    )
    bool_check(
        checks,
        "closed_smoke_contract",
        smoke_contract_ok,
        f"mismatches={contract_mismatches!r} initial_lr={initial_lr} "
        f"current_lr={current_lr} frame={checkpoint.get('frame')}/{expected_frame}",
    )
    bool_check(
        checks,
        "distance_curriculum_state_saturated",
        math.isclose(float(state.get("k_min_cur", -1)), 20.0, abs_tol=1e-9)
        and math.isclose(float(state.get("k_max_cur", -1)), 28.0, abs_tol=1e-9),
        f"state k=[{state.get('k_min_cur')},{state.get('k_max_cur')}]; "
        f"applied general-spawn minimum={state.get('cfg_general_goal_dist_min')}",
    )
    bool_check(
        checks,
        "timeout_path_exercised",
        outcome["timeout"] > 0,
        f"last100 timeout={outcome['timeout']}/{outcome['episodes']}",
    )
    bool_check(
        checks,
        "last100_learning_viability",
        outcome["capture_rate"] >= 0.65
        and outcome["crash_rate"] <= 0.33
        and outcome["timeout_rate"] <= 0.05,
        "capture={:.4f} crash={:.4f} timeout={:.4f} episodes={}".format(
            outcome["capture_rate"],
            outcome["crash_rate"],
            outcome["timeout_rate"],
            outcome["episodes"],
        ),
    )

    passed = all(item["pass"] for item in checks.values())
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ref5in_learning_viability_engineering_smoke",
        "expected_epochs": expected_epochs,
        "expected_learning_rate": expected_learning_rate,
        "verdict": "PASS" if passed else "FAIL",
        "performance_claim_allowed": False,
        "next_step": (
            "held_out_70bar_eval_only" if passed else "stop_and_diagnose_before_any_full_training"
        ),
        "run": str(run.relative_to(ROOT) if ROOT in run.parents else run),
        "checkpoint": str(
            checkpoint_path.relative_to(ROOT) if ROOT in checkpoint_path.parents else checkpoint_path
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "session_log": str(log.relative_to(ROOT) if ROOT in log.parents else log),
        "last100_training_outcomes": outcome,
        "tensorboard": tb,
        "source_verification": source_verification,
        "checks": checks,
        "limitations": [
            "one training seed",
            "on-policy training outcomes rather than held-out navigation performance",
            "whole-platform intervention includes mass, inertia, actuator and tilted collision-envelope changes",
            "same inherited controller gains confound airframe dynamics with controller tuning",
            "no CAD, power, thermal, endurance or flight validation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-epochs", type=int, default=500)
    parser.add_argument("--expected-learning-rate", type=float, default=3e-5)
    args = parser.parse_args()
    if args.expected_epochs < 100:
        parser.error("--expected-epochs must be at least 100")
    if not math.isfinite(args.expected_learning_rate) or args.expected_learning_rate <= 0.0:
        parser.error("--expected-learning-rate must be finite and positive")
    report = analyze(args.run, args.expected_epochs, args.expected_learning_rate)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
