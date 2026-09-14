"""Generic cell-grid evaluator for the lateral/density plan (D1, D3, L1-L3, L5, L7).

Prereg-agnostic runner: a JSON spec lists cells, each with a policy, a density and the governor
environment. One root, one shared source bundle, one commit, frozen source for the whole run,
no partial resume. Per-contact records (plan I1/I2) are always on.

Two evaluation contracts exist and they are NOT interchangeable. "ref5in" is the frozen D1/A8
lineage: goals 22.5-28 m and the v7 learned detector, built by the distractor envelope. "v2" is
the ep25000 lineage: goals 6-28 m and the built-in appearance segmenter, mirroring
eval_navrl_v2_ep25000_arc_attribution.sh. The AIRFRAME is not a contract choice -- the evaluator
reads it from the checkpoint (eval_navrl_v2_density_sweep.sh:236) and refuses a mismatch -- but
the goal band and the detector are set by the caller, so they have to be pinned per lineage.
One contract per grid, so every cell in a root stays comparable.

Spec format (JSON):
{
  "contract": "ref5in",
  "seed": 523,
  "episodes": 2049,
  "frame_sample_every": 100,
  "cells": [
    {"name": "T0_d070_stopcap", "policy": "T0", "bars": 70,
     "env": {"NAVRL_SPEED_GOVERNOR": "stopcap"}},
    {"name": "T0_d070_stopcap_w0p8", "policy": "T0", "bars": 70,
     "env": {"NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M": "0.8"}}
  ]
}
Policies: S (frozen D1 ep1900), T0/T1/T2 (A8 arms, resolved by run tag), ep25000 (the 205-bar
navrl_quad lineage), or an absolute .pth path. Results land in <root>/<name>/<bars>bars.json with
the two JSONL sidecars; cells.json holds the resolved checkpoints and the spec.

Usage: python tools/run_navrl_filter_grid.py <spec.json> <result_root> [resolve|evaluate]
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
A8_RUNNER = REPO / "tools" / "run_navrl_a8_readaptation.py"
EP25000 = (REPO / "aerial_gym/rl_training/rl_games/runs/"
           "ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/"
           "last_gen_ppo_ep_25000_rew_39.742134.pth")
ENVELOPE_CELL = "v7_n0"
# A7 governor parameters; a cell's env may override any of them, and the override is the arm.
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
# A cell may vary the governor, and from P10 the perception-error arm. Both are experiment axes
# that a spec is allowed to move; everything else about the condition stays fixed, which is what
# stops a sweep from silently comparing two different tasks. P10 needs NAVRL_P9_* because its
# design is 4 arms over {source, adapted} x {clean, empirical error}
# (docs/plans/perception_p9_p10_execution_2026-09-09.md).
ALLOWED_ENV_PREFIXES = ("NAVRL_SPEED_GOVERNOR", "NAVRL_P9_")
NAME_RE = r"^[A-Za-z0-9_.+-]{1,80}$"
CONTRACTS = ("ref5in", "v2")
# The ep25000 lineage, byte-identical to eval_navrl_v2_ep25000_arc_attribution.sh lines 50-66.
V2_CONTRACT = {
    "GPU4GB": "0",
    "NUM_ENVS": "128",
    "AERIAL_GYM_SIM_NAME": "base_sim",
    "NAVRL_V2_FORCE": "0",
    "NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH": "0",
    "NAVRL_V2_ACTION_MODE": "deterministic",
    "NAVRL_EVAL_REFLECTION_MODE": "original",
    "NAVRL_V2_GOAL_DIST_MIN": "6",
    "NAVRL_V2_GOAL_DIST_MAX": "28",
    "NAVRL_V2_TARGET_PATTERN": "mixed",
    "NAVRL_DISTRACTOR_COUNT": "0",
}


def _require(cond, msg):
    if not cond:
        raise SystemExit(f"[grid] FAIL: {msg}")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def validate_spec(spec):
    import re
    _require(spec.get("contract", "ref5in") in CONTRACTS,
             f"spec.contract must be one of {CONTRACTS}; got {spec.get('contract')!r}")
    _require(isinstance(spec.get("cells"), list) and spec["cells"], "spec.cells must be a non-empty list")
    seen = set()
    for cell in spec["cells"]:
        name = cell.get("name", "")
        _require(re.match(NAME_RE, name or ""), f"bad cell name {name!r}")
        _require(name not in seen, f"duplicate cell name {name}")
        seen.add(name)
        _require(isinstance(cell.get("bars"), int) and 10 <= cell["bars"] <= 300, f"{name}: bars out of range")
        _require(cell.get("policy"), f"{name}: policy missing")
        env = cell.get("env", {})
        _require(isinstance(env, dict) and "NAVRL_SPEED_GOVERNOR" in env, f"{name}: env.NAVRL_SPEED_GOVERNOR required")
        for key, value in env.items():
            _require(key.startswith(ALLOWED_ENV_PREFIXES),
                     f"{name}: {key} is neither a governor knob nor a P9 error-arm knob; only the governor may vary")
            _require(isinstance(value, str), f"{name}: env values must be strings")
    _require(isinstance(spec.get("seed"), int), "spec.seed must be an int")
    return spec


def resolve_policy(policy, envelope, a8):
    if policy == "S":
        path = Path(envelope.CHECKPOINT)
        _require(_sha256(path) == envelope.CHECKPOINT_SHA, "source ep1900 SHA drifted")
    elif policy in a8.ARM_MODES:
        path = a8.arm_checkpoint(policy)
    elif policy == "ep25000":
        path = EP25000
    else:
        path = Path(policy)
        _require(path.is_absolute() and path.is_file(), f"policy {policy!r} is not S/T0/T1/T2/ep25000 or an absolute .pth")
    return {"path": str(path), "sha256": _sha256(path)}


# What must not move during a run is the EVALUATED SOURCE, not the repository as a whole. An
# earlier version compared HEAD, which VOIDed a grid because a paper draft was committed in
# another directory while cells were running -- a commit that cannot reach the simulator. These
# are the only paths whose bytes the evaluator reads.
FROZEN_PATHS = ("aerial_gym", "tools", "resources/robots")


def _source_fingerprint():
    """Tree hashes of the evaluated paths. Changes iff their committed content changes."""
    trees = []
    for path in FROZEN_PATHS:
        trees.append(subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", f"HEAD:{path}"], text=True).strip())
    return ":".join(trees)


def _frozen(expected=None):
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all",
         "--", *FROZEN_PATHS], text=True).strip()
    _require(not status, "runtime/launcher sources must be committed: " + status)
    fingerprint = _source_fingerprint()
    _require(expected is None or fingerprint == expected,
             "evaluated source changed during evaluation; root is VOID")
    return fingerprint


def _v2_env():
    """A CLOSED environment for the ep25000 lineage: no ambient NAVRL_* survives, exactly as the
    A7 P3 launcher's `unset $(compgen -v NAVRL_)` does."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("NAVRL_")
           and k not in {"AERIAL_RUN_TAG", "AERIAL_GYM_SIM_NAME", "GPU4GB", "NUM_ENVS",
                         "FILE", "TASK", "PYTHON", "CKPT", "HEADLESS", "PLAY_GAMES_NUM",
                         "PYTHONPATH", "PYTHONHOME"}}
    env.update(V2_CONTRACT)
    env.update({
        "PYTHON": sys.executable, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(REPO),
        "NAVRL_REQUIRE_SOURCE_ROOT": str(REPO),
    })
    return env


def cell_env(envelope, spec, cell, root, out_dir):
    if spec.get("contract", "ref5in") == "v2":
        env = _v2_env()
    else:
        env = envelope.evaluation_env(ENVELOPE_CELL, preflight=False)
    env["NAVRL_SEED"] = str(spec["seed"])
    env["NAVRL_V2_DENSITIES"] = str(cell["bars"])
    env["NAVRL_V2_RESULT_DIR"] = str(out_dir)
    env["NAVRL_V2_SHARED_SOURCE_BUNDLE"] = str(root / "source_bundle")
    env["NAVRL_CONTACT_GEOMETRY"] = "1"
    env["NAVRL_SPEED_GOVERNOR_DIAG"] = "1"
    env["NAVRL_STAR_CONVEX_SHADOW"] = "0"
    env["NAVRL_CG_FRAME_SAMPLE_EVERY"] = str(int(spec.get("frame_sample_every", 100)))
    env.update(GOVERNOR_ENV)
    env.update(cell["env"])
    return env


def run(spec_path, root, mode):
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    root = Path(root)
    envelope = _load(ENVELOPE, "envelope_for_grid")
    a8 = _load(A8_RUNNER, "a8_for_grid")
    policies = {p: resolve_policy(p, envelope, a8) for p in sorted({c["policy"] for c in spec["cells"]})}
    if mode == "resolve":
        print(json.dumps(policies, indent=2))
        return 0
    _require(not root.exists(), f"refusing an existing result root (no partial resume): {root}")
    fingerprint = _frozen()
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    gate0 = envelope.verify_prerequisites()
    _require(envelope.gate0_static_passed(gate0), "envelope gate-0 failed: " + envelope.gate0_failure_report(gate0))
    n = int(spec.get("episodes", envelope.EPISODES))
    root.mkdir(parents=True)
    (root / "cells.json").write_text(json.dumps({
        "schema_version": 2, "contract": spec.get("contract", "ref5in"),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path), "spec": spec, "evaluation_commit": commit,
        "evaluated_source_fingerprint": fingerprint, "frozen_paths": list(FROZEN_PATHS),
        "checkpoints": policies, "episodes": n, "governor_defaults": GOVERNOR_ENV,
    }, indent=2, sort_keys=True) + "\n")
    for cell in spec["cells"]:
        _frozen(fingerprint)
        out_dir = root / cell["name"]
        _require(not out_dir.exists(), f"{cell['name']}: exists")
        env = cell_env(envelope, spec, cell, root, out_dir)
        print(f"[grid] EVALUATE {cell['name']} | {spec.get('contract', 'ref5in')} | policy {cell['policy']} "
              f"({policies[cell['policy']]['sha256'][:12]}) | bars {cell['bars']} | "
              f"{env['NAVRL_SPEED_GOVERNOR']} | seed {spec['seed']} | {n} ep", flush=True)
        code = envelope.tee_run(["bash", str(envelope.EVALUATOR), policies[cell["policy"]]["path"], str(n)],
                                env, root / f"{cell['name']}.eval.log.partial")
        _require(code == 0, f"{cell['name']}: evaluator exited {code}")
        _frozen(fingerprint)
        _require((out_dir / f"{cell['bars']}bars.json").is_file(), f"{cell['name']}: no result JSON")
        (root / f"{cell['name']}.eval.log.partial").replace(out_dir / "eval.log")
    print(f"[grid] COMPLETE: {root}")
    return 0


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_navrl_filter_grid.py <spec.json> <result_root> [resolve|evaluate]")
    return run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "evaluate")


if __name__ == "__main__":
    raise SystemExit(main())
