#!/usr/bin/env python3
"""Run the preregistered one-bar ref5in CV radial-heading diagnostic.

This frozen-policy probe asks whether the 70-bar away-vs-toward timeout penalty requires dense
obstacle occlusion.  It cannot revise P2/D1 or unlock P3.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
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
BASELINE_SUMMARY = ROOT / "results/navrl_ref5in_cv_heading_seed337/summary.json"
OUTPUT = ROOT / "results/navrl_ref5in_cv_heading_near_open_seed347"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
SEED = 347
EPISODES = 2049
BARS = 1
MODES = ("toward", "away")
EXPECTED_MISMATCH = "cfg_target_pattern: checkpoint=mixed expected=cv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    "ref5in_cv_heading_base", ROOT / "tools/run_navrl_ref5in_cv_heading_diagnostic.py"
)
P2 = BASE.P2
ContractError = BASE.ContractError
require = BASE.require
load_json = BASE.load_json


def cell_dir(mode: str) -> Path:
    return OUTPUT / "cells" / mode


def canonical_env(mode: str, *, preflight: bool, force: bool) -> dict[str, str]:
    env = P2.canonical_env(cell_dir(mode), preflight=preflight)
    env.update(
        {
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
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


def verify_prerequisites(*, require_clean: bool) -> dict:
    BASE.verify_prerequisites(require_clean=require_clean)
    require(BASELINE_SUMMARY.is_file(), "70-bar heading summary missing")
    baseline = load_json(BASELINE_SUMMARY)
    require(
        baseline.get("checkpoint_sha256") == CHECKPOINT_SHA,
        "70-bar diagnostic checkpoint lineage mismatch",
    )
    require(
        baseline.get("screen", {}).get("radial_heading_channel_support") is True,
        "70-bar radial-heading screen is not positive",
    )
    require(
        baseline.get("p3_unlocked") is False,
        "70-bar diagnostic unexpectedly unlocked P3",
    )
    return baseline


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
    mismatch_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if "checkpoint=" in line and "expected=" in line
    ]
    require(completed.returncode == 2, "generic evaluator accepted CV intervention without force")
    require(
        mismatch_lines == [EXPECTED_MISMATCH],
        f"forced mismatch set is not exactly one field: {mismatch_lines}",
    )
    run_command(MODES[0], preflight=True, force=True)
    return EXPECTED_MISMATCH


def verify_cell(mode: str) -> tuple[dict, dict]:
    directory = cell_dir(mode)
    result_path = directory / f"{BARS}bars.json"
    receipt_path = directory / f"{BARS}bars.receipt.json"
    snapshot = directory / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, snapshot, directory / f"{BARS}bars.log"):
        require(path.is_file(), f"missing {mode} artifact: {path}")
    result, receipt = load_json(result_path), load_json(receipt_path)
    require(P2.sha256_file(snapshot) == CHECKPOINT_SHA, f"{mode} checkpoint snapshot mismatch")
    require(P2.sha256_file(result_path) == receipt.get("result_sha256"), f"{mode} result hash mismatch")

    expected_condition = {
        "seed": SEED,
        "bars": BARS,
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
    require(
        int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
        f"{mode} episode contract mismatch",
    )
    outcome = result.get("outcome") or {}
    counts = [int(outcome.get(key, -1)) for key in ("captured", "crash", "timeout")]
    require(min(counts) >= 0 and sum(counts) == actual, f"{mode} outcome accounting mismatch")

    motion = result.get("target_motion") or {}
    expected_radial = {"toward": (-1.0, 0.0), "away": (1.0, 0.0)}[mode]
    require(int(motion.get("initial_heading_samples", 0)) >= actual, f"{mode} reset audit missing")
    require(
        abs(float(motion.get("initial_heading_mean_radial_cos")) - expected_radial[0]) <= 1e-5,
        f"{mode} radial-cos audit failed",
    )
    require(
        abs(float(motion.get("initial_heading_mean_radial_sin")) - expected_radial[1]) <= 1e-5,
        f"{mode} radial-sin audit failed",
    )
    require(
        float(motion.get("initial_heading_max_contract_error", 1.0)) <= 1e-5,
        f"{mode} heading contract error",
    )
    for key, value in {
        "seed": SEED,
        "bars": BARS,
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
    payload = BASE.cell_summary(result)
    payload["target_hidden_fraction"] = float(
        result.get("action", {}).get("context", {}).get("target_hidden", {}).get("fraction")
    )
    payload["closest_nocrash_mean_m"] = float(result["outcome"]["closest_nocrash_mean_m"])
    payload["capture_mean_step"] = float(
        result["speed_governor"]["outcome_steps"]["capture"]["mean"]
    )
    return payload


def build_summary(results: dict[str, dict], mismatch: str, baseline: dict) -> dict:
    cells = {mode: cell_summary(result) for mode, result in results.items()}
    toward, away = cells["toward"], cells["away"]
    comparison = {
        name: BASE.difference(away[name], away["episodes"], toward[name], toward["episodes"])
        for name in ("capture", "crash", "timeout")
    }
    timeout_delta = comparison["timeout"]["difference"]
    if timeout_delta <= 0.08:
        interpretation = "supports_dense_obstacle_occlusion_required"
    elif timeout_delta >= 0.15:
        interpretation = "obstacle_occlusion_not_necessary_kinematic_fov_wall_channel_sufficient"
    else:
        interpretation = "mixed_or_inconclusive"
    baseline_delta = baseline["comparisons"]["away_minus_toward"]["timeout"]["difference"]
    return {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_cv_heading_near_open.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_d1_frozen_cv_initial_heading_near_open_seed347",
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
            "bars": BARS,
            "requested_episodes_per_cell": EPISODES,
            "goal_dist_m": [22.5, 28.0],
            "target_pattern": "cv",
            "target_speed_mps": [0.3, 1.5],
            "action_selection": "deterministic",
            "episode_len_steps": 600,
            "modes": list(MODES),
        },
        "cells": cells,
        "comparisons": {"away_minus_toward": comparison},
        "screen": {
            "near_open_timeout_difference": timeout_delta,
            "dense_70bar_timeout_difference": baseline_delta,
            "difference_reduction": baseline_delta - timeout_delta,
            "thresholds": {"occlusion_required_max": 0.08, "occlusion_not_necessary_min": 0.15},
            "interpretation": interpretation,
        },
        "limitations": [
            "mechanism diagnostic only; cannot revise P2 or D1 and cannot unlock P3",
            "one frozen training seed and one evaluation seed",
            "one bar is near-open, not obstacle-free",
            "heading still changes kinematics, FOV visibility, and time to wall together",
            "same seed is not episode-paired because vector environments reset asynchronously",
        ],
    }


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in near-open CV initial-heading diagnostic",
        "",
        "이 결과는 dense obstacle occlusion의 필요성만 분리하는 동결정책 진단이다.",
        "",
        "| initial heading | episodes | capture | crash | timeout | hidden |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        cell = payload["cells"][mode]
        lines.append(
            f"| {mode} | {cell['episodes']:,} | {cell['capture_rate']*100:.2f}% | "
            f"{cell['crash_rate']*100:.2f}% | {cell['timeout_rate']*100:.2f}% | "
            f"{cell['target_hidden_fraction']*100:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"사전 screen: `{payload['screen']['interpretation']}`",
            f"1-bar away−toward timeout: {payload['screen']['near_open_timeout_difference']*100:+.2f} pp",
            f"70-bar away−toward timeout: {payload['screen']['dense_70bar_timeout_difference']*100:+.2f} pp",
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def verify_all() -> dict[str, dict]:
    results, runtime_maps = {}, []
    for mode in MODES:
        result, receipt = verify_cell(mode)
        results[mode] = result
        manifest = Path(receipt["runtime_source_manifest"]).resolve()
        mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
        BASE.verify_runtime_clean_manifest(metadata, mode)
        runtime_maps.append(mapping)
    require(runtime_maps[0] == runtime_maps[1], "near-open cells used different runtime byte maps")
    # Verify the frozen snapshots, not an evolving post-experiment working tree.
    return results


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(mode in {"preflight", "run", "finalize", "verify"}, "usage: ... {preflight|run|finalize|verify}")
    baseline = verify_prerequisites(require_clean=mode == "run")
    if mode == "preflight":
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        mismatch = verify_narrow_override()
        print(f"[ref5in-cv-near-open] PREFLIGHT PASS | bars={BARS} | {mismatch}")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        mismatch = verify_narrow_override()
        for heading in MODES:
            print(f"[ref5in-cv-near-open] RUN {heading}", flush=True)
            run_command(heading, preflight=False, force=True)
        payload = build_summary(verify_all(), mismatch, baseline)
        write_summary(payload)
        print(f"[ref5in-cv-near-open] COMPLETE | {payload['screen']['interpretation']}")
        return 0

    expected = build_summary(verify_all(), EXPECTED_MISMATCH, baseline)
    if mode == "finalize":
        write_summary(expected)
        print("[ref5in-cv-near-open] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed",
        "p3_unlocked", "checkpoint_sha256", "generic_provenance_override", "condition",
        "cells", "comparisons", "screen", "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-cv-near-open] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContractError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[ref5in-cv-near-open] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
