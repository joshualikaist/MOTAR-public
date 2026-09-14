#!/usr/bin/env python3
"""Preregistered frozen ref5in symmetric-corridor mode-averaging diagnostic.

Usage:
  python tools/run_navrl_ref5in_mode_probe.py preflight
  python tools/run_navrl_ref5in_mode_probe.py run
  python tools/run_navrl_ref5in_mode_probe.py finalize
  python tools/run_navrl_ref5in_mode_probe.py verify

The simulator rollout is only a lifecycle host for policy inference.  The three probe actions are
side forwards and are never executed.  This experiment has diagnostic authority only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
CHECKPOINT_REL = Path(
    "aerial_gym/rl_training/rl_games/"
    "runs/ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)


def _resolve_checkpoint():
    local = ROOT / CHECKPOINT_REL
    if local.is_file():
        return local
    # Git worktrees intentionally do not duplicate ignored multi-GB runs/. Resolve the primary
    # worktree through the shared .git directory; identity is still pinned by CHECKPOINT_SHA.
    common = Path(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], text=True
        ).strip()
    ).resolve()
    primary = common.parent / CHECKPOINT_REL
    return primary


CHECKPOINT = _resolve_checkpoint()
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
OUTPUT = ROOT / "results/navrl_ref5in_symmetric_corridor_mode_probe_seed431"
CELL = OUTPUT / "cell"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
PROBE_JSON = CELL / "mode_probe.json"
SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_RECEIPT = OUTPUT / "summary.receipt.json"
SEED = 431
EPISODES = 257
BARS = 70


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P2 = load_module("mode_probe_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_env(*, preflight=False):
    env = P2.canonical_env(CELL, preflight=preflight)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            "NAVRL_V2_RESULT_DIR": str(CELL),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            "NAVRL_MODE_PROBE_JSON": str(PROBE_JSON),
            "NAVRL_MODE_PROBE_OFFSET_DEG": "5.0",
        }
    )
    return env


def verify_prerequisites(*, require_clean):
    require(CHECKPOINT.is_file(), "pinned ref5in checkpoint missing")
    require(sha256(CHECKPOINT) == CHECKPOINT_SHA, "pinned ref5in checkpoint hash mismatch")
    require(EVALUATOR.is_file(), "canonical evaluator missing")
    imported = subprocess.check_output(
        [
            str(P2.PYTHON),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('aerial_gym').origin)",
        ],
        cwd=ROOT,
        env=canonical_env(preflight=True),
        text=True,
    ).strip()
    require(
        Path(imported).resolve() == (ROOT / "aerial_gym/__init__.py").resolve(),
        f"Python would import aerial_gym from the wrong worktree: {imported}",
    )
    require(not SUMMARY_JSON.exists(), "summary already exists; refusing to overwrite")
    require(not SUMMARY_RECEIPT.exists(), "summary receipt already exists; refusing to overwrite")
    if require_clean:
        status = subprocess.check_output(
            [
                "git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no",
                "--", "aerial_gym", "resources/robots", "tools/create_navrl_source_bundle.py",
            ],
            text=True,
        )
        require(not status.strip(), "runtime source is dirty; commit the probe before GPU run")


def run_evaluator(*, preflight):
    subprocess.run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        cwd=ROOT,
        env=canonical_env(preflight=preflight),
        check=True,
    )


def verify_artifacts():
    result_path = CELL / f"{BARS}bars.json"
    receipt_path = CELL / f"{BARS}bars.receipt.json"
    snapshot = CELL / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, snapshot, PROBE_JSON):
        require(path.is_file(), f"missing mode-probe artifact: {path}")
    result, receipt, probe = load_json(result_path), load_json(receipt_path), load_json(PROBE_JSON)
    require(sha256(result_path) == receipt.get("result_sha256"), "result/receipt hash mismatch")
    require(sha256(snapshot) == CHECKPOINT_SHA, "evaluated checkpoint snapshot mismatch")
    condition = result.get("condition") or {}
    expected = {
        "seed": SEED,
        "bars": BARS,
        "robot_name": "navrl_ref5in_quad",
        "action_selection": "deterministic",
        "reflection_mode": "original",
        "speed_governor_mode": "off",
    }
    mismatch = {k: (condition.get(k), v) for k, v in expected.items() if condition.get(k) != v}
    require(not mismatch, f"mode-probe rollout contract mismatch: {mismatch}")
    require(int(result.get("requested_episodes", -1)) == EPISODES, "requested episode mismatch")
    require(int(result.get("actual_episodes", 0)) >= EPISODES, "rollout ended early")
    require(probe.get("schema_version") == 1, "mode-probe schema mismatch")
    require(int(probe.get("samples", 0)) > 0, "mode probe collected no samples")
    require(probe.get("decision_authority") == "diagnostic_only_no_training_or_replacement_authority",
            "mode-probe authority drift")
    fixture = probe.get("fixture_contract") or {}
    pair_errors = fixture.get("reflection_pair_max_abs") or {}
    require(
        set(pair_errors)
        == {"symmetric_lr_to_rl", "left_lr_to_right_rl", "left_rl_to_right_lr"},
        "fixture reflection-pair contract is incomplete",
    )
    require(max(float(value) for value in pair_errors.values()) <= 1e-7,
            "physical fixture pairs are not exact mirrors")
    return result_path, receipt_path, result, receipt, probe


def finalize():
    result_path, receipt_path, result, receipt, probe = verify_artifacts()
    summary = {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_mode_probe.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_ref5in_symmetric_corridor_synthetic_policy_screen",
        "decision_authority": "diagnostic_only",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "condition": {
            "seed": SEED,
            "bars_lifecycle_host_only": BARS,
            "requested_episodes": EPISODES,
            "executed_action_selection": result["condition"]["action_selection"],
            "probe_actions_executed": False,
        },
        "probe": probe,
        "interpretation": probe["verdict"],
        "next_step": (
            "If supported, preregister a real-scene replay or multi-candidate-head ablation. "
            "If chirality-confounded, do not interpret the null; the reflection defect dominates."
        ),
        "sources": {
            "evaluation_result": str(result_path.relative_to(ROOT)),
            "evaluation_receipt": str(receipt_path.relative_to(ROOT)),
            "mode_probe": str(PROBE_JSON.relative_to(ROOT)),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_JSON.with_name(SUMMARY_JSON.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SUMMARY_JSON)
    receipt_payload = {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_mode_probe.py",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "summary_sha256": sha256(SUMMARY_JSON),
        "probe_sha256": sha256(PROBE_JSON),
        "evaluation_result_sha256": sha256(result_path),
        "evaluation_receipt_sha256": sha256(receipt_path),
        "tool_sha256": sha256(Path(__file__)),
    }
    tmp = SUMMARY_RECEIPT.with_name(SUMMARY_RECEIPT.name + ".tmp")
    tmp.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SUMMARY_RECEIPT)
    print(f"[mode-probe] {probe['verdict']} -> {SUMMARY_JSON}")


def verify_summary():
    require(SUMMARY_JSON.is_file() and SUMMARY_RECEIPT.is_file(), "summary bundle missing")
    receipt = load_json(SUMMARY_RECEIPT)
    require(sha256(SUMMARY_JSON) == receipt.get("summary_sha256"), "summary hash mismatch")
    require(sha256(PROBE_JSON) == receipt.get("probe_sha256"), "probe hash mismatch")
    verify_artifacts()
    print(f"[mode-probe] verified -> {SUMMARY_JSON}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "run", "finalize", "verify"}:
        raise SystemExit("usage: run_navrl_ref5in_mode_probe.py {preflight|run|finalize|verify}")
    command = sys.argv[1]
    if command == "preflight":
        verify_prerequisites(require_clean=False)
        run_evaluator(preflight=True)
        print("[mode-probe] preflight PASS")
    elif command == "run":
        verify_prerequisites(require_clean=True)
        run_evaluator(preflight=False)
    elif command == "finalize":
        finalize()
    else:
        verify_summary()


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        print(f"[mode-probe] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(3)
