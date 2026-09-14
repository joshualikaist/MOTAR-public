#!/usr/bin/env python3
"""Held-out seed-367 screen for the preregistered active-search geofence A/B.

Usage: tools/run_navrl_ref5in_active_search_geofence_eval.py {status|preflight|run|finalize|verify}
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
sys.path.insert(0, str(ROOT))
TRAIN_ROOT = Path("/home/fair/workspaces/aerial_gym_ws/.codex_worktrees/navrl_oob_seed367")
TRAIN_RL = TRAIN_ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
OUTPUT = ROOT / "results/navrl_ref5in_active_search_geofence_seed367"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
SEED = 367
BARS = 1
EPISODES = 2049
EXPECTED_MISMATCHES = [
    "cfg_general_goal_dist_min: checkpoint=6.0 expected=22.5",
    "cfg_target_pattern: checkpoint=mixed expected=cv",
]
PRIMARY_DELTA_PP = 0.03
NON_OOB_CRASH_RISE_PP = 0.02
MASKED_FRACTION_OF_GAIN = 0.50

ARMS = {
    "control": {"checkpoint_arm": "control", "geofence": False, "masked": False},
    "geofence": {"checkpoint_arm": "geofence", "geofence": True, "masked": False},
    "geofence_masked": {"checkpoint_arm": "geofence", "geofence": True, "masked": True},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P2 = load_module("active_search_p2_base", ROOT / "tools/attest_navrl_ref5in_p2.py")


class ContractError(RuntimeError):
    pass


def require(value, message: str):
    if not value:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(arm: str) -> Path:
    tag = f"v2-ref5in-active-search-{arm}-s197"
    matches = sorted(path for path in (TRAIN_RL / "runs").glob(f"ppo_*_navrl_{tag}") if path.is_dir())
    require(len(matches) == 1, f"expected one completed {arm} run, found {len(matches)}")
    require((matches[0] / ".aerial_training_finished").is_file(), f"{arm} training is not finished")
    return matches[0]


def checkpoint(arm: str) -> Path:
    matches = sorted((run_dir(arm) / "nn").glob("last_gen_ppo_ep_900_rew_*.pth"))
    matches = [path for path in matches if not path.name.endswith("_rlnorm.pth")]
    require(len(matches) == 1, f"expected one raw ep900 {arm} checkpoint, found {len(matches)}")
    return matches[0]


def checkpoints() -> dict[str, Path]:
    return {arm: checkpoint(arm) for arm in ("control", "geofence")}


def cell_dir(arm: str) -> Path:
    return OUTPUT / "cells" / arm


def canonical_env(arm: str, *, preflight: bool, force: bool) -> dict[str, str]:
    cfg = ARMS[arm]
    env = P2.canonical_env(cell_dir(arm), preflight=preflight)
    inherited = os.environ.get("PYTHONPATH", "")
    env.update(
        {
            "PYTHONPATH": str(ROOT) + (os.pathsep + inherited if inherited else ""),
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(arm)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            "NAVRL_V2_TARGET_PATTERN": "cv",
            "NAVRL_EVAL_CV_INITIAL_HEADING": "away",
            "NAVRL_DETECTOR_MAX_RANGE": "20",
            "NAVRL_GEOFENCE_ACTOR": "1" if cfg["geofence"] else "0",
            "NAVRL_GEOFENCE_NOISE_STD_M": "0",
            "NAVRL_GEOFENCE_DROPOUT": "0",
            "NAVRL_GEOFENCE_FORCE_INVALID": "1" if cfg["masked"] else "0",
        }
    )
    if force:
        env["NAVRL_V2_FORCE"] = "1"
    return env


def run_command(arm: str, *, preflight: bool, force: bool, check: bool = True):
    ckpt = checkpoint(ARMS[arm]["checkpoint_arm"])
    return subprocess.run(
        ["bash", str(EVALUATOR), str(ckpt), str(EPISODES)],
        cwd=ROOT,
        env=canonical_env(arm, preflight=preflight, force=force),
        text=True,
        stdout=subprocess.PIPE if not check else None,
        stderr=subprocess.STDOUT if not check else None,
        check=check,
    )


def verify_narrow_overrides():
    for arm in ARMS:
        completed = run_command(arm, preflight=True, force=False, check=False)
        lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("cfg_") and "checkpoint=" in line
        ]
        require(completed.returncode == 2, f"{arm}: evaluator accepted CV intervention without force")
        require(lines == EXPECTED_MISMATCHES, f"{arm}: unexpected mismatch set: {lines}")
        run_command(arm, preflight=True, force=True)


def verify_cell(arm: str) -> tuple[dict, dict]:
    directory = cell_dir(arm)
    result_path = directory / f"{BARS}bars.json"
    receipt_path = directory / f"{BARS}bars.receipt.json"
    snapshot = directory / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, snapshot, directory / f"{BARS}bars.log"):
        require(path.is_file(), f"{arm}: missing artifact {path}")
    result, receipt = load_json(result_path), load_json(receipt_path)
    ckpt = checkpoint(ARMS[arm]["checkpoint_arm"])
    ckpt_sha = sha256_file(ckpt)
    require(sha256_file(snapshot) == ckpt_sha, f"{arm}: checkpoint snapshot mismatch")
    require(sha256_file(result_path) == receipt["result_sha256"], f"{arm}: result receipt mismatch")
    actual = int(result["actual_episodes"])
    require(int(result["requested_episodes"]) == EPISODES and actual >= EPISODES, f"{arm}: episode contract")
    counts = [int(result["outcome"][key]) for key in ("captured", "crash", "timeout")]
    require(sum(counts) == actual, f"{arm}: outcome accounting")
    condition = result["condition"]
    expected_result = {
        "seed": SEED,
        "bars": BARS,
        "target_pattern": "cv",
        "cv_initial_heading": "away",
        "goal_dist_min_m": 22.5,
        "goal_dist_max_m": 28.0,
        "episode_len_steps": 600,
    }
    expected_receipt = {
        "seed": SEED,
        "bars": BARS,
        "target_pattern": "cv",
        "cv_initial_heading": "away",
        "goal_dist_min_m": 22.5,
        "goal_dist_max_m": 28.0,
        "target_camera_max_range_m": 20.0,
        "geofence_actor": ARMS[arm]["geofence"],
        "geofence_noise_std_m": 0.0,
        "geofence_dropout": 0.0,
        "geofence_force_invalid": ARMS[arm]["masked"],
    }
    result_mismatch = {
        key: (condition.get(key), value)
        for key, value in expected_result.items()
        if condition.get(key) != value
    }
    receipt_mismatch = {
        key: (receipt.get(key), value)
        for key, value in expected_receipt.items()
        if receipt.get(key) != value
    }
    require(not result_mismatch, f"{arm}: result condition mismatch {result_mismatch}")
    require(not receipt_mismatch, f"{arm}: receipt condition mismatch {receipt_mismatch}")
    require(
        receipt.get("evaluation_nonce") == condition.get("evaluation_nonce"),
        f"{arm}: result/receipt evaluation nonce mismatch",
    )
    oob = result["target_motion"]["oob_exit_forensics"]
    require(int(oob["exits"]) == int(result["crash_causes"]["out_of_bounds"]), f"{arm}: OOB accounting")
    require(
        int(oob["by_acquisition"]["never_acquired"]["exits"])
        + int(oob["by_acquisition"]["acquired"]["exits"])
        == int(oob["exits"]),
        f"{arm}: acquisition strata accounting",
    )
    return result, receipt


def metrics(result: dict) -> dict:
    n = int(result["actual_episodes"])
    outcome = result["outcome"]
    causes = result["crash_causes"]
    oob = result["target_motion"]["oob_exit_forensics"]
    never_oob = int(oob["never_acquired"])
    non_oob_crash = int(outcome["crash"]) - int(causes["out_of_bounds"])
    return {
        "episodes": n,
        "capture": int(outcome["captured"]) / n,
        "crash": int(outcome["crash"]) / n,
        "timeout": int(outcome["timeout"]) / n,
        "oob": int(causes["out_of_bounds"]) / n,
        "never_acquired_oob": never_oob,
        "never_acquired_oob_rate": never_oob / n,
        "non_oob_crash_rate": non_oob_crash / n,
        "oob_exit_forensics": oob,
    }


def build_summary(results: dict[str, dict]) -> dict:
    cells = {arm: metrics(result) for arm, result in results.items()}
    control = cells["control"]
    geofence = cells["geofence"]
    masked = cells["geofence_masked"]
    primary_gain = control["never_acquired_oob_rate"] - geofence["never_acquired_oob_rate"]
    crash_rise = geofence["non_oob_crash_rate"] - control["non_oob_crash_rate"]
    masked_loss = masked["never_acquired_oob_rate"] - geofence["never_acquired_oob_rate"]
    primary_pass = primary_gain >= PRIMARY_DELTA_PP
    guard_pass = crash_rise <= NON_OOB_CRASH_RISE_PP
    mechanism_pass = primary_gain > 0 and masked_loss >= MASKED_FRACTION_OF_GAIN * primary_gain
    ckpts = checkpoints()
    return {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_active_search_geofence_eval.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fresh_active_search_geofence_ab_seed367",
        "decision_authority": "active_search_geofence_preregistered_screen_only",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        "checkpoints": {
            arm: {"path": str(path), "sha256": sha256_file(path)} for arm, path in ckpts.items()
        },
        "condition": {
            "seed": SEED,
            "bars": BARS,
            "episodes_per_cell": EPISODES,
            "goal_dist_m": [22.5, 28.0],
            "target_pattern": "cv",
            "cv_initial_heading": "away",
            "camera_range_m": 20.0,
            "geofence_noise_std_m": 0.0,
            "geofence_dropout": 0.0,
        },
        "gates": {
            "primary_gain_min_pp": PRIMARY_DELTA_PP,
            "non_oob_crash_rise_max_pp": NON_OOB_CRASH_RISE_PP,
            "masked_loss_fraction_of_gain_min": MASKED_FRACTION_OF_GAIN,
        },
        "cells": cells,
        "screen": {
            "primary_gain": primary_gain,
            "non_oob_crash_rise": crash_rise,
            "masked_loss": masked_loss,
            "primary_pass": primary_pass,
            "guard_pass": guard_pass,
            "mechanism_pass": mechanism_pass,
            "outcome": (
                "PASS_MECHANISM_SUPPORTED"
                if primary_pass and guard_pass and mechanism_pass
                else "PASS_MECHANISM_UNRESOLVED"
                if primary_pass and guard_pass
                else "FAIL"
            ),
        },
        "limitations": [
            "A pass supports mapped-geofence active search, not camera/LiDAR-only exploration.",
            "The geofence assumes deployable VIO/GPS pose and a known flight boundary.",
            "This screen cannot revise P2/D1 or unlock P3.",
        ],
    }


def write_summary(payload: dict):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cells = payload["cells"]
    lines = [
        "# Active-search mapped-geofence A/B (seed 367)",
        "",
        "| arm | capture | crash | timeout | OOB | never-acq OOB/all | non-OOB crash |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = cells[arm]
        lines.append(
            f"| {arm} | {row['capture']:.2%} | {row['crash']:.2%} | {row['timeout']:.2%} | "
            f"{row['oob']:.2%} | {row['never_acquired_oob_rate']:.2%} | {row['non_oob_crash_rate']:.2%} |"
        )
    screen = payload["screen"]
    lines += [
        "",
        f"결론: **{screen['outcome']}**",
        "",
        f"- primary gain: {screen['primary_gain']:.2%} (gate >= {PRIMARY_DELTA_PP:.2%})",
        f"- non-OOB crash rise: {screen['non_oob_crash_rise']:.2%} (guard <= {NON_OOB_CRASH_RISE_PP:.2%})",
        f"- masked loss: {screen['masked_loss']:.2%}; mechanism pass={screen['mechanism_pass']}",
        "",
        "P2 STRICT FAIL, D1 FAIL, P3 BLOCKED는 바뀌지 않는다.",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_all() -> dict[str, dict]:
    results = {}
    maps = []
    for arm in ARMS:
        result, receipt = verify_cell(arm)
        results[arm] = result
        mapping, metadata = P2.manifest_map(Path(receipt["runtime_source_manifest"]), 2, require_original=False)
        require(metadata.get("git_dirty") is False, f"{arm}: dirty runtime source")
        maps.append(mapping)
    require(maps[0] == maps[1] == maps[2], "evaluation arms used different runtime byte maps")
    return results


def status() -> int:
    for arm in ("control", "geofence"):
        try:
            directory = run_dir(arm)
            print(f"{arm}: COMPLETE {directory}")
        except ContractError as exc:
            candidates = sorted((TRAIN_RL / "runs").glob(f"ppo_*_navrl_v2-ref5in-active-search-{arm}-s197"))
            print(f"{arm}: PENDING ({exc}); candidates={len(candidates)}")
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"status", "preflight", "run", "finalize", "verify"}:
        raise SystemExit("usage: ... {status|preflight|run|finalize|verify}")
    mode = sys.argv[1]
    if mode == "status":
        return status()
    package = importlib.util.find_spec("aerial_gym")
    require(package and package.origin and ROOT in Path(package.origin).resolve().parents, "aerial_gym import escaped eval worktree")
    checkpoints()
    if mode == "preflight":
        verify_narrow_overrides()
        print("[active-search-eval] PREFLIGHT PASS")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        verify_narrow_overrides()
        for arm in ARMS:
            print(f"[active-search-eval] RUN {arm}", flush=True)
            run_command(arm, preflight=False, force=True)
        payload = build_summary(verify_all())
        write_summary(payload)
        print(f"[active-search-eval] COMPLETE {payload['screen']['outcome']}")
        return 0
    expected = build_summary(verify_all())
    if mode == "finalize":
        write_summary(expected)
        print("[active-search-eval] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in ("scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed", "p3_unlocked", "checkpoints", "condition", "gates", "cells", "screen", "limitations"):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[active-search-eval] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[active-search-eval] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
