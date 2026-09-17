#!/usr/bin/env python3
"""Fail closed when an evaluation's environment does not match the checkpoint's.

The simulator already compares a restored checkpoint's ``env_state`` against the
running configuration, but that guard only calls ``logger.warning`` and lets the
run continue -- deliberately, because an evaluation may change a knob on
purpose. That is the right default for the simulator and the wrong default for a
preregistered matched-arm comparison, where an unintended difference is not a
warning but an invalid result.

This gate is the strict counterpart. Given a checkpoint's env_state and the
environment an evaluation is about to run under, it refuses unless every
difference is on an explicitly declared allowlist.

Exit codes:
  0  MATCHED            every key agrees, or differs only by declared intent
  2  CONFOUNDED         an undeclared difference exists; do not run
  3  UNREADABLE         inputs missing or malformed
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Keys whose difference invalidates a frozen-policy comparison outright, because
# they change either what the policy sees or how its output is executed.
CRITICAL = {
    "cfg_max_velocity": "action scale AND observation normalisation denominator",
    "cfg_yaw_rate_max": "yaw action scale",
    "cfg_lidar_hbeams": "observation dimension",
    "cfg_lidar_vbeams": "observation dimension",
    "cfg_lidar_max_range": "observation scale and sensing horizon",
    "cfg_max_obstacles": "observation dimension",
    "cfg_token_fov_deg": "obstacle token selection",
    "cfg_token_effective_fov_deg": "obstacle token selection",
    "cfg_obstacle_selector": "obstacle token selection",
    "cfg_max_tilt_deg": "attitude envelope",
    "cfg_arena_xy": "workspace geometry",
    "cfg_arena_z": "workspace geometry",
    "cfg_episode_len_steps": "episode budget; changes every rate metric's denominator",
    "cfg_physics_dt_s": "integration contract",
    "cfg_physics_steps_per_rl_step": "integration contract",
    "cfg_rl_step_dt_s": "control rate",
    "cfg_placement_mode": "obstacle field distribution",
    "cfg_placement_touch_m": "obstacle field distribution",
    "cfg_placement_gap_m": "obstacle field distribution",
    "cfg_bar_pool": "obstacle asset lineage",
    "cfg_oob_margin": "termination geometry",
    "cfg_speed_governor_mode": "safety filter",
    "cfg_speed_governor_free_mps": "safety filter",
    "cfg_speed_governor_fixed_mps": "safety filter",
    "cfg_speed_governor_ttc_s": "safety filter",
    "cfg_action_policy": "action distribution",
    "cfg_action_mu_scale": "action distribution",
    "cfg_action_std": "action distribution",
}

# The independent variable of the E0/E1/E2 comparison. These MAY differ between
# arms; everything else may not.
DECLARED_INDEPENDENT = {
    "cfg_target_motion_model",
    "cfg_target_pattern",
    "cfg_target_speed_min",
    "cfg_target_speed_final",
    "cfg_target_speed_fixed",
    "cfg_target_behavior_level",
    "cfg_target_dynamics",
}


def compare(checkpoint: dict, runtime: dict, allow: set):
    rows, verdict = [], "MATCHED"
    for key in sorted(set(checkpoint) | set(runtime)):
        if not key.startswith("cfg_"):
            continue
        saved, current = checkpoint.get(key), runtime.get(key)
        if saved is None or current is None:
            rows.append({"key": key, "state": "NOT_ATTESTABLE",
                         "checkpoint": saved, "runtime": current})
            continue
        try:
            same = abs(float(saved) - float(current)) <= 1e-6
        except (TypeError, ValueError):
            same = str(saved).strip() == str(current).strip()
        if same:
            continue
        if key in allow or key in DECLARED_INDEPENDENT:
            rows.append({"key": key, "state": "DECLARED_INDEPENDENT",
                         "checkpoint": saved, "runtime": current})
            continue
        state = "CONFOUNDED_CRITICAL" if key in CRITICAL else "CONFOUNDED"
        rows.append({
            "key": key, "state": state, "checkpoint": saved, "runtime": current,
            "why": CRITICAL.get(key, "undeclared difference between arms"),
        })
        verdict = "CONFOUNDED"
    return verdict, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-env-state", required=True,
                        help="JSON dump of the checkpoint's env_state")
    parser.add_argument("--runtime-env-state", required=True,
                        help="JSON dump of the env_state the evaluation will run under")
    parser.add_argument("--allow", default="",
                        help="comma-separated cfg_ keys that MAY differ, declared before running")
    parser.add_argument("--out")
    args = parser.parse_args()

    try:
        checkpoint = json.loads(Path(args.checkpoint_env_state).read_text())
        runtime = json.loads(Path(args.runtime_env_state).read_text())
    except (OSError, ValueError) as error:
        print(f"UNREADABLE: {error}", file=sys.stderr)
        return 3

    allow = {k.strip() for k in args.allow.split(",") if k.strip()}
    verdict, rows = compare(checkpoint, runtime, allow)
    report = {
        "kind": "navrl_eval_condition_contract_gate",
        "verdict": verdict,
        "declared_allow": sorted(allow),
        "declared_independent": sorted(DECLARED_INDEPENDENT),
        "differences": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    confounds = [r for r in rows if r["state"].startswith("CONFOUNDED")]
    for row in confounds:
        marker = "CRITICAL" if row["state"] == "CONFOUNDED_CRITICAL" else "        "
        print(f"{marker} {row['key']:34s} checkpoint={row['checkpoint']!s:<22s} "
              f"runtime={row['runtime']!s:<22s} {row.get('why','')}")
    declared = [r for r in rows if r["state"] == "DECLARED_INDEPENDENT"]
    if declared:
        print(f"\ndeclared independent variable ({len(declared)} keys):")
        for row in declared:
            print(f"  {row['key']:34s} {row['checkpoint']} -> {row['runtime']}")
    unattestable = [r for r in rows if r["state"] == "NOT_ATTESTABLE"]
    if unattestable:
        print(f"\n{len(unattestable)} keys not attestable on one side "
              "(present in only one env_state)")

    print(f"\nVERDICT: {verdict}")
    if verdict == "CONFOUNDED":
        print("Do not run this evaluation. Either match the environment or declare "
              "the difference with --allow and record why in the preregistration.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
