#!/usr/bin/env python3
"""Generate the STATIC research task contract the browser preview mirrors.

Why this file exists
--------------------
The browser interception preview needs the research task's termination
semantics: the capture radius and the episode budget. Those are research
values, not browser constants, and they were previously duplicated as literals
in three places (``viewer.js``, ``arena.js``, the validation harness). This tool
reads them from the research source and emits ONE machine-readable contract that
every consumer binds to.

It is deliberately NOT ``docs/status/status.json``. That file is a *dynamic*
dashboard snapshot -- active run, latest run, historical run summaries -- and
regenerating it to pick up three static fields would churn unrelated live data.
The two are separate on purpose:

    status.json                  dynamic dashboard snapshot
    research_task_contract.json  static browser/research task contract

Everything emitted here is read from source text; nothing is instantiated, so
this runs on a machine with no GPU and no Isaac Gym.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/status/research_task_contract.json"

TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
V2_LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"
BARS_ENV = ROOT / "aerial_gym/config/env_config/navrl_bars_env.py"
SIM_CONFIG = ROOT / "aerial_gym/config/sim_config/base_sim_config.py"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"

SOURCES = (TASK_CONFIG, V2_LAUNCHER, BARS_ENV, SIM_CONFIG, NAVRL_TASK)


class ContractError(RuntimeError):
    """A binding could not be read from source. Never fall back to a literal."""


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grab(path: Path, pattern: str, what: str, cast=float):
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise ContractError(
            f"{what}: pattern {pattern!r} no longer matches {path.relative_to(ROOT)}. "
            "The browser mirrors this value, so a silent default would let the page "
            "disagree with the research task."
        )
    return cast(match.group(1))


def build() -> dict:
    # --- termination semantics ------------------------------------------
    success_radius_m = grab(
        TASK_CONFIG, r"^\s*success_radius\s*=\s*([0-9.]+)", "success_radius")
    episode_len_steps = grab(
        V2_LAUNCHER, r"^export NAVRL_EPISODE_LEN_STEPS=(\d+)", "episode_len_steps", int)
    physics_dt_s = grab(SIM_CONFIG, r"^\s*dt\s*=\s*([0-9.]+)", "physics dt")
    physics_steps_per_rl_step = grab(
        BARS_ENV, r"^\s*num_physics_steps_per_env_step_mean\s*=\s*(\d+)",
        "physics steps per RL step", int)
    rl_step_dt_s = round(physics_dt_s * physics_steps_per_rl_step, 12)

    # --- stable arena geometry the browser draws -------------------------
    arena_xy_m = grab(V2_LAUNCHER, r"^export NAVRL_ARENA_XY=([0-9.]+)", "arena_xy_m")
    arena_z_m = grab(V2_LAUNCHER, r"^export NAVRL_ARENA_Z=([0-9.]+)", "arena_z_m")
    touch_m = grab(V2_LAUNCHER, r"^\s*export NAVRL_PLACEMENT_TOUCH_M=([0-9.]+)", "placement touch")
    gap_m = grab(V2_LAUNCHER, r"^\s*export NAVRL_PLACEMENT_GAP_M=([0-9.]+)", "placement gap")
    bar_x_min = grab(V2_LAUNCHER, r"^export NAVRL_BAR_X_MIN=([0-9.]+)", "bar band x min")
    bar_x_max = grab(V2_LAUNCHER, r"^export NAVRL_BAR_X_MAX=([0-9.]+)", "bar band x max")

    # capture must still END the episode, or the browser's mirror claim is void
    if "capture ends the episode" not in NAVRL_TASK.read_text(encoding="utf-8"):
        raise ContractError(
            "navrl_task.py no longer states that capture ends the episode; the browser "
            "interception preview's mirror claim must be re-audited before regenerating."
        )

    return {
        "schema_version": 1,
        "kind": "navrl_static_research_task_contract",
        "task": "navrl",
        "note": (
            "STATIC contract mirrored by the browser interception preview. This is NOT a "
            "dynamic run snapshot: docs/status/status.json owns active/latest run state and "
            "historical run summaries. Values here are read from research source, never "
            "hand-edited."
        ),
        # termination semantics
        "success_radius_m": success_radius_m,
        "episode_len_steps": episode_len_steps,
        "rl_step_dt_s": rl_step_dt_s,
        "episode_timeout_s": round(episode_len_steps * rl_step_dt_s, 12),
        "physics_dt_s": physics_dt_s,
        "physics_steps_per_rl_step": physics_steps_per_rl_step,
        "capture_ends_episode": True,
        # stable arena geometry
        "arena_xy_m": arena_xy_m,
        "arena_z_m": arena_z_m,
        "placement_mode": "navrl_band",
        "placement_touch_m": touch_m,
        "placement_gap_m": gap_m,
        "bar_x_min_ratio": bar_x_min,
        "bar_x_max_ratio": bar_x_max,
        "provenance": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_tree_clean": git("status", "--porcelain") == "",
            "bindings": {
                "success_radius_m": f"{TASK_CONFIG.relative_to(ROOT)}: success_radius",
                "episode_len_steps": f"{V2_LAUNCHER.relative_to(ROOT)}: NAVRL_EPISODE_LEN_STEPS",
                "rl_step_dt_s": (
                    f"{SIM_CONFIG.relative_to(ROOT)}: dt x "
                    f"{BARS_ENV.relative_to(ROOT)}: num_physics_steps_per_env_step_mean"),
                "arena/placement": f"{V2_LAUNCHER.relative_to(ROOT)}: NAVRL_ARENA_* / NAVRL_PLACEMENT_*",
                "capture_ends_episode": f"{NAVRL_TASK.relative_to(ROOT)}: interception semantics comment",
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in SOURCES
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed contract matches source; do not write")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    try:
        contract = build()
    except ContractError as error:
        print(f"CONTRACT BINDING FAILED: {error}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if args.check:
        if not out.is_file():
            print(f"missing {out.relative_to(ROOT)}", file=sys.stderr)
            return 1
        existing = json.loads(out.read_text())
        # provenance carries the commit/hashes, which move with unrelated edits;
        # the CONTRACT itself is what must not drift.
        keys = [k for k in contract if k != "provenance"]
        drift = {k: (existing.get(k), contract[k]) for k in keys if existing.get(k) != contract[k]}
        if drift:
            for key, (was, now) in drift.items():
                print(f"DRIFT {key}: committed={was!r} source={now!r}", file=sys.stderr)
            return 1
        print(f"research task contract matches source ({len(keys)} fields)")
        return 0

    out.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  success_radius_m  = {contract['success_radius_m']}")
    print(f"  episode_len_steps = {contract['episode_len_steps']}")
    print(f"  rl_step_dt_s      = {contract['rl_step_dt_s']}")
    print(f"  episode_timeout_s = {contract['episode_timeout_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
