"""A8 readaptation evaluation: ten cells, one root, one source bundle, seed 521.

Prereg: docs/prereg_2026-09-05_a8_filter_readaptation.md section 1. Reuses the distractor
envelope's closed ref5in evaluation environment (cell v7_n0: 70 bars, zero distractors,
22.5..28 m goals) exactly as the contact-geometry runner does, and overrides only the policy
checkpoint, the evaluation-time governor mode and the A7 governor parameters.

Cells are named <policy>_<filter>. Policies: S = frozen D1 ep1900 (the A8 source), T0/T1/T2 =
the A8 arms readapted with governor off / riskcap / dwa_arc. Each arm's terminal checkpoint is
resolved by run tag and MUST be unique.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ENVELOPE = REPO / "tools" / "run_navrl_distractor_envelope.py"
RUNS = REPO / "aerial_gym" / "rl_training" / "rl_games" / "runs"
SEED = int(os.environ.get("NAVRL_A8_SEED", "521"))
RESULT_ROOT = REPO / os.environ.get(
    "NAVRL_A8_RESULT_ROOT", f"results/navrl_a8_readaptation_seed{SEED}"
)
ENVELOPE_CELL = "v7_n0"
TERMINAL_EPOCH = 2900
ARM_MODES = {"T0": "off", "T1": "riskcap", "T2": "dwa_arc"}
# Prereg section 1, in execution order.
CELLS = (
    ("S", "off"), ("S", "riskcap"), ("S", "dwa_arc"),
    ("T0", "off"), ("T0", "riskcap"), ("T0", "dwa_arc"),
    ("T1", "riskcap"), ("T1", "off"),
    ("T2", "dwa_arc"), ("T2", "off"),
)
GOVERNOR_ENV = {
    "NAVRL_SPEED_GOVERNOR_FIXED_MPS": "2.0",
    "NAVRL_SPEED_GOVERNOR_FREE_MPS": "3.53553390593",
    "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M": "0.45",
    "NAVRL_SPEED_GOVERNOR_MARGIN_M": "0.45",
    "NAVRL_SPEED_GOVERNOR_SLOW_M": "3.0",
    "NAVRL_SPEED_GOVERNOR_RELEASE_M": "5.0",
    "NAVRL_SPEED_GOVERNOR_TTC_S": "1.0",
    "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2": "2.0",
    "NAVRL_SPEED_GOVERNOR_REACTION_S": "0.1",
}


def cell_name(policy, filt):
    return f"{policy}_{filt}"


def _load_envelope():
    spec = importlib.util.spec_from_file_location("distractor_envelope_for_a8", ENVELOPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require(cond, msg):
    if not cond:
        raise SystemExit(f"[a8] FAIL: {msg}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_checkpoint(arm, runs=RUNS):
    """The unique terminal checkpoint of one A8 training arm, by run tag."""
    mode = ARM_MODES[arm]
    pattern = f"*navrl_v2-ref5in-a8-readapt-{mode}-s197"
    run_dirs = sorted(p for p in Path(runs).glob(pattern) if p.is_dir())
    _require(len(run_dirs) == 1,
             f"{arm}: expected exactly one run dir matching {pattern}, found {len(run_dirs)}")
    matches = sorted((run_dirs[0] / "nn").glob(f"last_gen_ppo_ep_{TERMINAL_EPOCH}_*.pth"))
    _require(len(matches) == 1,
             f"{arm}: expected exactly one ep{TERMINAL_EPOCH} checkpoint in {run_dirs[0]}, found {len(matches)}")
    return matches[0]


def resolve_checkpoints(mod, policies):
    out = {}
    for policy in policies:
        if policy == "S":
            path = Path(mod.CHECKPOINT)
            _require(_sha256(path) == mod.CHECKPOINT_SHA, "source ep1900 checkpoint SHA drifted")
        else:
            path = arm_checkpoint(policy)
        out[policy] = {"path": str(path), "sha256": _sha256(path)}
    return out


def _require_frozen_source(expected_commit=None):
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all",
         "--", "aerial_gym", "tools", "resources/robots"], text=True,
    ).strip()
    _require(not status, "A8 runtime/launcher sources must be committed before evaluation: " + status)
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    _require(expected_commit is None or commit == expected_commit,
             "A8 git HEAD changed during evaluation; this root is VOID")
    return commit


def cell_env(mod, policy, filt, out_dir):
    env = mod.evaluation_env(ENVELOPE_CELL, preflight=False)
    env["NAVRL_SEED"] = str(SEED)
    env["NAVRL_V2_RESULT_DIR"] = str(out_dir)
    env["NAVRL_V2_SHARED_SOURCE_BUNDLE"] = str(RESULT_ROOT / "source_bundle")
    env["NAVRL_CONTACT_GEOMETRY"] = "1"
    env["NAVRL_SPEED_GOVERNOR_DIAG"] = "1"
    env["NAVRL_STAR_CONVEX_SHADOW"] = "0"
    env.update(GOVERNOR_ENV)
    env["NAVRL_SPEED_GOVERNOR"] = filt
    return env


def run_evaluate(mod, episodes=None):
    _require(not RESULT_ROOT.exists(), f"A8 refuses an existing result root (no partial resume): {RESULT_ROOT}")
    commit = _require_frozen_source()
    checkpoints = resolve_checkpoints(mod, sorted({p for p, _ in CELLS}))
    gate0 = mod.verify_prerequisites()
    _require(mod.gate0_static_passed(gate0), "envelope gate-0 failed: " + mod.gate0_failure_report(gate0))
    n = int(episodes or mod.EPISODES)
    RESULT_ROOT.mkdir(parents=True)
    (RESULT_ROOT / "cells.json").write_text(json.dumps({
        "schema_version": 1, "prereg": "docs/prereg_2026-09-05_a8_filter_readaptation.md",
        "seed": SEED, "episodes": n, "evaluation_commit": commit, "checkpoints": checkpoints,
        "cells": [{"name": cell_name(p, f), "policy": p, "filter": f} for p, f in CELLS],
    }, indent=2, sort_keys=True) + "\n")
    for policy, filt in CELLS:
        _require_frozen_source(commit)
        name = cell_name(policy, filt)
        out_dir = RESULT_ROOT / name
        _require(not out_dir.exists(), f"{name}: cell directory already exists")
        env = cell_env(mod, policy, filt, out_dir)
        print(f"[a8] EVALUATE {name} | policy {policy} ({checkpoints[policy]['sha256'][:12]}) | filter {filt} | seed {SEED} | {n} episodes")
        code = mod.tee_run(["bash", str(mod.EVALUATOR), checkpoints[policy]["path"], str(n)],
                           env, RESULT_ROOT / f"{name}.eval.log.partial")
        _require(code == 0, f"{name}: evaluator exited {code}")
        _require_frozen_source(commit)
        _require(out_dir.is_dir(), f"{name}: no result directory")
        (RESULT_ROOT / f"{name}.eval.log.partial").replace(out_dir / "eval.log")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "resolve":
        mod = _load_envelope()
        print(json.dumps(resolve_checkpoints(mod, sorted({p for p, _ in CELLS})), indent=2))
    elif mode == "evaluate":
        mod = _load_envelope()
        run_evaluate(mod, int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} resolve|evaluate [episodes]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
