#!/usr/bin/env python3
"""Generate and check the 48-cell preflight for the frozen-policy E0/E1/E2 comparison.

Dry only. It produces the cell specification and checks each cell's declared
condition against the frozen checkpoint's attested contract through the strict
gate. It runs no episodes, starts no GPU work, and writes no outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/audits"))

CHECKPOINT = (ROOT / "aerial_gym/rl_training/rl_games/runs/"
              "ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/"
              "nn/last_gen_ppo_ep_25000_rew_39.742134.pth")
CHECKPOINT_SHA = "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"

# Amendment 1 of the preregistration. Changing any of these is a protocol change.
ARMS = ("H_historical", "E0_static", "E1_cv", "E2_obstacle_aware")
DENSITIES = (70, 115, 160, 205)
SEEDS = (4101, 4102, 4103)
EPISODES_PER_CELL = 2048

# Per-arm environment: the target-behaviour axis and nothing else.
ARM_ENV = {
    "H_historical": {
        "NAVRL_TARGET_BEHAVIOR_LEVEL": "historical",
        "NAVRL_TARGET_DYNAMICS": "legacy",
        "NAVRL_TARGET_PATTERN": "mixed",
        "NAVRL_TARGET_SPEED_MIN": "0.3",
        "NAVRL_TARGET_SPEED_FINAL": "1.5",
    },
    "E0_static": {
        "NAVRL_TARGET_BEHAVIOR_LEVEL": "e0_static",
        "NAVRL_TARGET_DYNAMICS": "bounded",
        "NAVRL_TARGET_PATTERN": "cv",
        "NAVRL_TARGET_SPEED_MIN": "0.0",
        "NAVRL_TARGET_SPEED_FINAL": "0.0",
    },
    "E1_cv": {
        "NAVRL_TARGET_BEHAVIOR_LEVEL": "e1_cv",
        "NAVRL_TARGET_DYNAMICS": "bounded",
        "NAVRL_TARGET_PATTERN": "cv",
        "NAVRL_TARGET_SPEED_MIN": "0.3",
        "NAVRL_TARGET_SPEED_FINAL": "1.5",
    },
    "E2_obstacle_aware": {
        "NAVRL_TARGET_BEHAVIOR_LEVEL": "e2_obstacle_aware",
        "NAVRL_TARGET_DYNAMICS": "bounded",
        "NAVRL_TARGET_PATTERN": "waypoint",
        "NAVRL_TARGET_SPEED_MIN": "0.3",
        "NAVRL_TARGET_SPEED_FINAL": "1.5",
    },
}

# The training contract, replayed. These MUST be identical in every cell; HEAD
# defaults differ on all of them (see the contract diff artifact).
HELD_FIXED_ENV = {
    "NAVRL_ARENA_XY": "40",
    "NAVRL_EPISODE_LEN_STEPS": "600",
    "NAVRL_LIDAR_HBEAMS": "72",
    "NAVRL_LIDAR_RANGE": "12",
    "NAVRL_MAX_VELOCITY": "2.5",
    "NAVRL_YAW_RATE_MAX": "3.0",
    "NAVRL_OOB_MARGIN": "1.0",
    "NAVRL_OBSTACLE_SELECTOR": "cluster_sector",
    "NAVRL_PLACEMENT_MODE": "navrl_band",
    "NAVRL_TARGET_ROUTE_MODE": "off",
    "NAVRL_SPEED_GOVERNOR": "riskcap",
    "NAVRL_BAR_POOL": "bars_h3",
    "NAVRL_V2_PROFILE": "main",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_condition(arm, density, seed):
    """The env_state a cell is declared to run under."""
    env = dict(HELD_FIXED_ENV)
    env.update(ARM_ENV[arm])
    return {
        "cfg_arena_xy": 40.0, "cfg_arena_z": 3.0,
        "cfg_bar_pool": "bars_h3",
        "cfg_placement_mode": "navrl_band",
        "cfg_placement_touch_m": 0.4, "cfg_placement_gap_m": 1.6,
        "cfg_density_final": float(density),
        "cfg_episode_len_steps": 600.0,
        "cfg_lidar_hbeams": 72, "cfg_lidar_vbeams": 4, "cfg_lidar_max_range": 12.0,
        "cfg_max_obstacles": 8, "cfg_obstacle_selector": "cluster_sector",
        "cfg_token_fov_deg": 240.0, "cfg_token_effective_fov_deg": 240.0,
        "cfg_max_velocity": 2.5, "cfg_yaw_rate_max": 3.0, "cfg_max_tilt_deg": 45.0,
        "cfg_oob_margin": 1.0,
        "cfg_physics_dt_s": 0.01, "cfg_physics_steps_per_rl_step": 10,
        "cfg_rl_step_dt_s": 0.1,
        "cfg_speed_governor_mode": "riskcap",
        "cfg_speed_governor_free_mps": 3.53553390593,
        "cfg_speed_governor_fixed_mps": 2.0, "cfg_speed_governor_ttc_s": 1.2,
        "cfg_action_policy": "squashed_gaussian",
        "cfg_action_mu_scale": "1.0,0.4,1.0,1.0",
        "cfg_action_std": "0.35,0.35,0.05,0.08",
        "cfg_general_goal_dist_min": 6.0, "cfg_general_goal_dist_max": 28.0,
        # the independent variable
        "cfg_target_pattern": env["NAVRL_TARGET_PATTERN"],
        "cfg_target_speed_min": float(env["NAVRL_TARGET_SPEED_MIN"]),
        "cfg_target_speed_final": float(env["NAVRL_TARGET_SPEED_FINAL"]),
        "cfg_target_motion_model": (
            "symmetric_local_steer_v2_heading_continuity90"
            if arm == "H_historical" else "bounded_planar_drone_v1_rollout"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-env-state", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import check_eval_condition_contract as gate

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_env = json.loads(Path(args.checkpoint_env_state).read_text())

    actual_sha = sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None
    checkpoint_ok = actual_sha == CHECKPOINT_SHA

    cells, comparisons, failures = [], [], []
    for arm in ARMS:
        for density in DENSITIES:
            for seed in SEEDS:
                cid = f"{arm}__{density}bars__seed{seed}"
                condition = cell_condition(arm, density, seed)
                # density is a declared design factor, not a confound
                verdict, rows = gate.compare(
                    checkpoint_env, condition, allow={"cfg_density_final"})
                confounds = [r for r in rows if r["state"].startswith("CONFOUNDED")]
                if confounds:
                    failures.append({"cell": cid, "confounds": confounds})
                cells.append({
                    "cell_id": cid, "arm": arm, "density_bars": density,
                    "evaluation_seed": seed,
                    "episodes": EPISODES_PER_CELL,
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "env": {**HELD_FIXED_ENV, **ARM_ENV[arm],
                            "NAVRL_NUM_BARS": str(density),
                            "NAVRL_EVAL_SEED": str(seed)},
                    "gate_verdict": verdict,
                })
                comparisons.append({
                    "cell_id": cid, "verdict": verdict,
                    "declared_independent": [r for r in rows
                                             if r["state"] == "DECLARED_INDEPENDENT"],
                    "confounds": confounds,
                })

    passed = sum(1 for c in cells if c["gate_verdict"] == "MATCHED")
    go = passed == len(cells) and checkpoint_ok

    (out / "cell_manifest.json").write_text(json.dumps({
        "kind": "e0_e2_frozen_policy_cell_manifest",
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree_clean": git("status", "--porcelain") == "",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_present": CHECKPOINT.is_file(),
        "checkpoint_sha_verified": checkpoint_ok,
        "arms": list(ARMS), "densities": list(DENSITIES), "seeds": list(SEEDS),
        "episodes_per_cell": EPISODES_PER_CELL,
        "total_cells": len(cells),
        "total_episodes": len(cells) * EPISODES_PER_CELL,
        "held_fixed_env": HELD_FIXED_ENV,
        "arm_env": ARM_ENV,
        "cells": cells,
    }, indent=1, sort_keys=True) + "\n")

    (out / "contract_comparison.json").write_text(json.dumps({
        "kind": "e0_e2_cell_contract_comparison",
        "cells_checked": len(comparisons),
        "cells_matched": passed,
        "cells_confounded": len(failures),
        "failures": failures,
        "comparisons": comparisons,
    }, indent=1, sort_keys=True) + "\n")

    print(f"cells: {len(cells)}  matched: {passed}  confounded: {len(failures)}")
    print(f"checkpoint present: {CHECKPOINT.is_file()}  sha verified: {checkpoint_ok}")
    for failure in failures[:5]:
        print(f"  FAIL {failure['cell']}: "
              + ", ".join(r["key"] for r in failure["confounds"]))
    print(f"\npreflight contract check: {'PASS' if go else 'FAIL'}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
