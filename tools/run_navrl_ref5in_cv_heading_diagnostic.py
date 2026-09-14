#!/usr/bin/env python3
"""Run the preregistered frozen-policy ref5in CV initial-heading diagnostic.

This is a mechanism probe after the failed D1 gate.  It cannot revise P2/D1 or unlock P3.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
CHECKPOINT = RL_ROOT / (
    "runs/ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
D1_SUMMARY = ROOT / "results/navrl_ref5in_d1_eval_seed331/summary.json"
OUTPUT = ROOT / "results/navrl_ref5in_cv_heading_seed337"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
SEED = 337
EPISODES = 2049
MODES = ("toward", "tangent_left", "tangent_right", "away")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P2 = load_module("ref5in_cv_heading_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_runtime_clean_manifest(metadata: dict, mode: str) -> None:
    """Accept repository-wide result dirt while rejecting any snapshotted runtime dirt."""
    roots = tuple(str(root).rstrip("/") + "/" for root in metadata.get("runtime_roots") or ())
    require(roots == ("aerial_gym/", "resources/robots/"), f"{mode} runtime roots changed")
    offending = []
    for row in metadata.get("git_status") or []:
        path_text = str(row)[3:]
        candidates = path_text.split(" -> ")
        if any(candidate.startswith(roots) for candidate in candidates):
            offending.append(row)
    require(not offending, f"{mode} runtime source was dirty: {offending[:8]}")


def cell_dir(mode: str) -> Path:
    return OUTPUT / "cells" / mode


def canonical_env(mode: str, *, preflight: bool, force: bool) -> dict[str, str]:
    env = P2.canonical_env(cell_dir(mode), preflight=preflight)
    env.update(
        {
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(mode)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            "NAVRL_V2_TARGET_PATTERN": "cv",
            "NAVRL_EVAL_CV_INITIAL_HEADING": mode,
        }
    )
    if force:
        env["NAVRL_V2_FORCE"] = "1"
    return env


def verify_prerequisites(*, require_clean: bool) -> None:
    require(CHECKPOINT.is_file(), f"D1 terminal checkpoint missing: {CHECKPOINT}")
    require(P2.sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "D1 checkpoint identity mismatch")
    require(D1_SUMMARY.is_file(), "D1 held-out summary missing")
    d1 = load_json(D1_SUMMARY)
    require(d1.get("verdict") == "FAIL", "D1 verdict is not fail-closed")
    require(d1.get("p3_automatically_unlocked") is False, "D1 unexpectedly unlocked P3")
    require(d1.get("checkpoint_sha256") == CHECKPOINT_SHA, "D1 summary lineage mismatch")
    if require_clean:
        status = subprocess.check_output(
            [
                "git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no",
                "--", "aerial_gym", "resources/robots", "tools/create_navrl_source_bundle.py",
            ],
            text=True,
        )
        require(not status.strip(), "runtime source is dirty; commit the diagnostic before running")


def run_command(mode: str, *, preflight: bool, force: bool, check: bool = True):
    return subprocess.run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        cwd=ROOT,
        env=canonical_env(mode, preflight=preflight, force=force),
        text=True,
        stdout=subprocess.PIPE if not check else None,
        stderr=subprocess.STDOUT if not check else None,
        check=check,
    )


def verify_narrow_override() -> str:
    completed = run_command(MODES[0], preflight=True, force=False, check=False)
    expected = "cfg_target_pattern: checkpoint=mixed expected=cv"
    mismatch_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if "checkpoint=" in line and "expected=" in line
    ]
    require(completed.returncode == 2, "generic evaluator accepted the CV intervention without force")
    require(mismatch_lines == [expected], f"forced mismatch set is not exactly one field: {mismatch_lines}")
    run_command(MODES[0], preflight=True, force=True)
    return expected


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [center - half, center + half]


def difference(a_success: int, a_total: int, b_success: int, b_total: int) -> dict:
    pa, pb = a_success / a_total, b_success / b_total
    delta = pa - pb
    se = math.sqrt(pa * (1.0 - pa) / a_total + pb * (1.0 - pb) / b_total)
    z = 1.959963984540054
    return {"difference": delta, "difference_ci95": [delta - z * se, delta + z * se]}


def verify_cell(mode: str) -> tuple[dict, dict]:
    directory = cell_dir(mode)
    result_path = directory / "70bars.json"
    receipt_path = directory / "70bars.receipt.json"
    snapshot = directory / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, snapshot, directory / "70bars.log"):
        require(path.is_file(), f"missing {mode} artifact: {path}")
    result, receipt = load_json(result_path), load_json(receipt_path)
    require(P2.sha256_file(snapshot) == CHECKPOINT_SHA, f"{mode} checkpoint snapshot mismatch")
    require(P2.sha256_file(result_path) == receipt.get("result_sha256"), f"{mode} result hash mismatch")
    expected_condition = {
        "seed": SEED,
        "bars": 70,
        "robot_name": "navrl_ref5in_quad",
        "action_selection": "deterministic",
        "reflection_mode": "original",
        "speed_governor_mode": "off",
        "goal_dist_min_m": 22.5,
        "goal_dist_max_m": 28.0,
        "target_pattern": "cv",
        "cv_initial_heading": mode,
        "target_speed_mode": "uniform",
        "target_speed_min_mps": 0.3,
        "target_speed_max_mps": 1.5,
        "episode_len_steps": 600,
    }
    condition = result.get("condition") or {}
    mismatches = {
        key: (condition.get(key), value)
        for key, value in expected_condition.items()
        if condition.get(key) != value
    }
    require(not mismatches, f"{mode} condition mismatch: {mismatches}")
    actual = int(result.get("actual_episodes", -1))
    require(int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
            f"{mode} episode contract mismatch")
    outcome = result.get("outcome") or {}
    counts = [int(outcome.get(key, -1)) for key in ("captured", "crash", "timeout")]
    require(min(counts) >= 0 and sum(counts) == actual, f"{mode} outcome accounting mismatch")
    motion = result.get("target_motion") or {}
    expected_radial = {
        "toward": (-1.0, 0.0),
        "tangent_left": (0.0, 1.0),
        "tangent_right": (0.0, -1.0),
        "away": (1.0, 0.0),
    }[mode]
    require(int(motion.get("initial_heading_samples", 0)) >= actual,
            f"{mode} heading audit lacks reset samples")
    require(abs(float(motion.get("initial_heading_mean_radial_cos")) - expected_radial[0]) <= 1e-5,
            f"{mode} radial-cos audit failed")
    require(abs(float(motion.get("initial_heading_mean_radial_sin")) - expected_radial[1]) <= 1e-5,
            f"{mode} radial-sin audit failed")
    require(float(motion.get("initial_heading_max_contract_error", 1.0)) <= 1e-5,
            f"{mode} heading contract error")
    for key, value in {
        "seed": SEED,
        "bars": 70,
        "requested_episodes": EPISODES,
        "source_checkpoint_sha256": CHECKPOINT_SHA,
        "goal_dist_min_m": 22.5,
        "goal_dist_max_m": 28.0,
        "target_pattern": "cv",
        "cv_initial_heading": mode,
    }.items():
        require(receipt.get(key) == value, f"{mode} receipt mismatch: {key}")
    return result, receipt


def cell_summary(result: dict) -> dict:
    outcome = result["outcome"]
    n = int(result["actual_episodes"])
    payload = {"episodes": n}
    for name, count_name in (("capture", "captured"), ("crash", "crash"), ("timeout", "timeout")):
        count = int(outcome[count_name])
        payload[name] = count
        payload[f"{name}_rate"] = count / n
        payload[f"{name}_wilson95"] = wilson(count, n)
    return payload


def build_summary(results: dict[str, dict], mismatch: str) -> dict:
    cells = {mode: cell_summary(result) for mode, result in results.items()}
    toward, away = cells["toward"], cells["away"]
    left, right = cells["tangent_left"], cells["tangent_right"]
    away_toward = {
        name: difference(away[name], away["episodes"], toward[name], toward["episodes"])
        for name in ("capture", "crash", "timeout")
    }
    tangent_lr = {
        name: difference(left[name], left["episodes"], right[name], right["episodes"])
        for name in ("capture", "crash", "timeout")
    }
    tangent_timeouts = [left["timeout_rate"], right["timeout_rate"]]
    path_length = (
        away_toward["timeout"]["difference"] >= 0.08
        and all(toward["timeout_rate"] <= value <= away["timeout_rate"] for value in tangent_timeouts)
    )
    tangent_max_abs = max(abs(item["difference"]) for item in tangent_lr.values())
    chirality = tangent_max_abs >= 0.05
    all_timeout_high = all(cell["timeout_rate"] >= 0.12 for cell in cells.values())
    if path_length:
        interpretation = "supports_radial_heading_channel_path_visibility_wall_coupled"
    elif chirality:
        interpretation = "prioritize_chirality_sensitive_followup"
    elif all_timeout_high:
        interpretation = "reject_initial_heading_alone_prioritize_tracker_progress_telemetry"
    else:
        interpretation = "mixed_or_inconclusive"
    return {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_cv_heading_diagnostic.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_d1_frozen_cv_initial_heading_seed337",
        "decision_authority": "none",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "generic_provenance_override": {
            "used": True,
            "sole_verified_mismatch": mismatch,
            "reason": "controlled CV-only evaluation of a mixed-pattern training checkpoint",
        },
        "condition": {
            "seed": SEED,
            "bars": 70,
            "requested_episodes_per_cell": EPISODES,
            "goal_dist_m": [22.5, 28.0],
            "target_pattern": "cv",
            "target_speed_mps": [0.3, 1.5],
            "action_selection": "deterministic",
            "episode_len_steps": 600,
            "modes": list(MODES),
        },
        "cells": cells,
        "comparisons": {
            "away_minus_toward": away_toward,
            "tangent_left_minus_right": tangent_lr,
        },
        "screen": {
            "radial_heading_channel_support": path_length,
            "chirality_sensitive": chirality,
            "all_cells_timeout_ge_12pct": all_timeout_high,
            "tangent_max_abs_outcome_difference": tangent_max_abs,
            "interpretation": interpretation,
        },
        "limitations": [
            "mechanism diagnostic only; cannot revise P2 or D1 and cannot unlock P3",
            "one frozen training seed and one evaluation seed at 70 bars",
            "same seed is not episode-paired because vector environments reset asynchronously",
            "heading is controlled only at reset; obstacle steering and wall reflection may alter it afterward",
            "radial heading changes path length, target visibility, and time to wall together",
        ],
    }


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in frozen CV initial-heading diagnostic", "",
        "이 결과는 P2/D1을 재판정하거나 P3를 해제하지 않는 원인 진단이다.", "",
        "| initial heading | episodes | capture | crash | timeout |", "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        cell = payload["cells"][mode]
        lines.append(
            f"| {mode} | {cell['episodes']:,} | {cell['capture_rate']*100:.2f}% | "
            f"{cell['crash_rate']*100:.2f}% | {cell['timeout_rate']*100:.2f}% |"
        )
    screen = payload["screen"]
    lines.extend(
        [
            "", f"사전 screen: `{screen['interpretation']}`",
            f"away−toward timeout: {payload['comparisons']['away_minus_toward']['timeout']['difference']*100:+.2f} pp",
            f"tangent L−R 최대 outcome 차이: {screen['tangent_max_abs_outcome_difference']*100:.2f} pp", "",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def verify_all() -> tuple[dict[str, dict], str]:
    mismatch = "cfg_target_pattern: checkpoint=mixed expected=cv"
    results, runtime_maps = {}, []
    for mode in MODES:
        result, receipt = verify_cell(mode)
        results[mode] = result
        manifest = Path(receipt["runtime_source_manifest"]).resolve()
        mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
        verify_runtime_clean_manifest(metadata, mode)
        runtime_maps.append(mapping)
    require(all(mapping == runtime_maps[0] for mapping in runtime_maps[1:]),
            "heading cells used different runtime byte maps")
    # Historical artifacts remain verifiable after later instrumentation-only source changes;
    # each receipt is checked against its immutable snapshot rather than today's working tree.
    return results, mismatch


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(mode in {"preflight", "run", "finalize", "verify"},
            "usage: ... {preflight|run|finalize|verify}")
    verify_prerequisites(require_clean=mode == "run")
    if mode == "preflight":
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        mismatch = verify_narrow_override()
        print(f"[ref5in-cv-heading] PREFLIGHT PASS | {mismatch}")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        mismatch = verify_narrow_override()
        for heading in MODES:
            print(f"[ref5in-cv-heading] RUN {heading}", flush=True)
            run_command(heading, preflight=False, force=True)
        results, _ = verify_all()
        payload = build_summary(results, mismatch)
        write_summary(payload)
        print(f"[ref5in-cv-heading] COMPLETE | {payload['screen']['interpretation']}")
        return 0
    results, mismatch = verify_all()
    expected = build_summary(results, mismatch)
    if mode == "finalize":
        write_summary(expected)
        print("[ref5in-cv-heading] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed",
        "p3_unlocked", "checkpoint_sha256", "generic_provenance_override", "condition",
        "cells", "comparisons", "screen", "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-cv-heading] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, RuntimeError, subprocess.CalledProcessError,
            json.JSONDecodeError) as exc:
        print(f"[ref5in-cv-heading] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
