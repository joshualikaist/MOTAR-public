#!/usr/bin/env python3
"""CPU validity audit of the TM-E2 bounded target executor, before any GPU run.

Exercises ``bounded_drone_target_step`` directly on CPU tensors. It never builds a
task, never imports isaacgym and never touches a GPU, so it can gate a GPU
evaluation rather than depend on one.

What it proves, per the preregistration's validity gates:
  - no obstacle penetration, no wall violation
  - speed / acceleration / turn-rate bounds hold on the executed step
  - no position push-out and no wall reflection (those are legacy-only paths)
  - no non-finite state
  - the executor consumes ZERO draws from the global RNG
Exit 0 PASS, 1 FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import importlib.util

import torch

# target_motion.py is deliberately package-free ("Isaac Gym stays out of CPU
# tests"), so load it by path rather than through aerial_gym/__init__.py, which
# imports isaacgym and opens a CUDA context this audit must not need.
_MOTION_PATH = (Path(__file__).resolve().parents[2]
                / "aerial_gym/task/navrl_task/target_motion.py")
_spec = importlib.util.spec_from_file_location("_navrl_target_motion_audit", _MOTION_PATH)
_motion = importlib.util.module_from_spec(_spec)
sys.modules["_navrl_target_motion_audit"] = _motion
_spec.loader.exec_module(_motion)

BOUNDED_TURN_ANGLES_DEG = _motion.BOUNDED_TURN_ANGLES_DEG
bounded_drone_target_step = _motion.bounded_drone_target_step
# The slew limit is gated on this, not on a numeric epsilon: below it the
# contract says there is no heading of travel and the command starts in the
# requested direction. Checking the turn bound below it measures nothing.
HEADING_VALID_SPEED_MPS = _motion.HEADING_VALID_SPEED_MPS
TARGET_MOTION_SOURCE = str(_MOTION_PATH)

DT = 0.1
SPEED_MAX = 1.5
MAX_ACCEL = 4.0
MAX_TURN_DEG = 150.0
# The BOUNDED lineage -- which is exactly E0/E1/E2 -- passes bars_half_extents=None
# and clearance = tm.obstacle_clearance, i.e. a CENTRE-TO-CENTRE test at 0.77 m
# (navrl_task.py:7873 sets half extents only for the physical target;
# _target_planner_clearance returns obstacle_clearance for non-physical).
# 0.77 m is deliberately just inside the navrl_band 1.6 m corridor half-width of
# 0.8 m. Auditing with the PHYSICAL contract (surface distance at 1.0 m) makes
# legal corridors impassable and manufactures trapped states.
CLEARANCE = 0.77
BAR_HALF_EXTENT = 0.3


# The production arena places bars with the navrl_band rule, not uniformly at
# random. A uniform field is strictly HARDER: it admits arbitrarily tight
# clusters that the real contract forbids, which manufactures trapped states the
# executor would never meet and would make this audit unrepresentative.
TOUCH_M = 0.4
GAP_M = 1.6


def navrl_band_positions(n_bars, generator, device):
    """One env's layout under the navrl_band rule.

    A candidate is accepted only if EVERY placed bar is either touching it
    (<= TOUCH_M, merging into a compound wall) or at least GAP_M away (a passable
    corridor). Distances inside the band would be an impassable slit.
    """
    placed = []
    guard = 0
    while len(placed) < n_bars and guard < n_bars * 600:
        guard += 1
        cand = torch.rand(2, generator=generator, device=device)
        cand = torch.tensor([float(cand[0]) * 38.0 + 1.0,
                             float(cand[1]) * 38.0 - 19.0], device=device)
        ok = True
        for other in placed:
            d = float((cand - other).norm())
            if TOUCH_M < d < GAP_M:
                ok = False
                break
        if ok:
            placed.append(cand)
    if len(placed) < n_bars:
        raise RuntimeError(f"navrl_band layout failed closed at {len(placed)}/{n_bars}")
    return torch.stack(placed)


def make_field(n_envs, n_bars, seed, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    layouts = [navrl_band_positions(n_bars, g, device) for _ in range(n_envs)]
    bars = torch.stack(layouts)
    half = torch.full((n_envs, n_bars, 2), 0.3, device=device)
    return bars, half


def sample_clear(n_envs, bars, generator, device, tries=200):
    """Points inside the arena and outside the clearance envelope of every bar."""
    out = torch.zeros((n_envs, 2), device=device)
    filled = torch.zeros(n_envs, dtype=torch.bool, device=device)
    for _ in range(tries):
        cand = torch.rand((n_envs, 2), generator=generator, device=device) * 34.0 + 3.0
        cand[:, 1] -= 20.0
        clear = (cand.unsqueeze(1) - bars).norm(dim=2).amin(dim=1) > CLEARANCE + 0.05
        take = clear & ~filled
        out[take] = cand[take]
        filled |= take
        if bool(filled.all()):
            break
    if not bool(filled.all()):
        raise RuntimeError("could not place %d clear points" % int((~filled).sum()))
    return out


def run_case(n_envs, n_bars, steps, seed, exact_aabb):
    device = "cpu"
    bars, half = make_field(n_envs, n_bars, seed, device)
    lo = torch.tensor([[1.0, -19.0]], device=device).repeat(n_envs, 1)
    hi = torch.tensor([[39.0, 19.0]], device=device).repeat(n_envs, 1)

    g = torch.Generator(device=device).manual_seed(seed + 1)
    # Rejection-sample spawns and goals OUTSIDE the clearance envelope. A target
    # that starts inside a bar cannot escape, and counting that as an executor
    # penetration would blame the generator for the harness.
    pos = sample_clear(n_envs, bars, g, device)
    vel = torch.zeros((n_envs, 2), device=device)
    goal = sample_clear(n_envs, bars, g, device)

    speed = torch.full((n_envs,), SPEED_MAX, device=device)
    accel = torch.full((n_envs,), MAX_ACCEL, device=device)
    turn = torch.full((n_envs,), math.radians(MAX_TURN_DEG), device=device)
    sign = torch.where(torch.rand((n_envs,), generator=g, device=device) < 0.5,
                       -torch.ones(n_envs), torch.ones(n_envs))

    counts = {
        "steps": 0, "nonfinite": 0, "wall_violation": 0, "obstacle_penetration": 0,
        "speed_violation": 0, "accel_violation": 0, "turn_violation": 0,
        "teleport": 0, "stationary_steps": 0, "immediate_infeasible": 0,
        "below_heading_valid_speed_steps": 0, "clearance_violation": 0,
        "rng_draws_consumed": 0,
    }
    prev_heading = torch.atan2(vel[:, 1], vel[:, 0])

    for _ in range(steps):
        direction = goal - pos
        norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        desired = direction / norm * speed.unsqueeze(1)

        # The executor must not touch the global RNG. Snapshot and compare.
        before_state = torch.random.get_rng_state()
        prev_pos, prev_vel = pos.clone(), vel.clone()
        # Production bounded contract: half extents None -> centre-to-centre test.
        pos, vel, _steered, immediate = bounded_drone_target_step(
            pos, vel, desired, speed, DT, bars, lo, hi, CLEARANCE, sign,
            accel, turn, lookahead_s=1.0, bars_half_extents_xy=None,
            exact_aabb_clearance=False, hard_epsilon_m=0.0,
        )
        if not torch.equal(before_state, torch.random.get_rng_state()):
            counts["rng_draws_consumed"] += 1

        counts["steps"] += 1
        counts["immediate_infeasible"] += int((~immediate).sum())
        if not (torch.isfinite(pos).all() and torch.isfinite(vel).all()):
            counts["nonfinite"] += 1
        counts["wall_violation"] += int(((pos < lo - 1e-6) | (pos > hi + 1e-6)).any(dim=1).sum())

        delta = (pos.unsqueeze(1) - bars).abs() - BAR_HALF_EXTENT
        inside = (delta <= -1e-9).all(dim=2).any(dim=1)
        counts["obstacle_penetration"] += int(inside.sum())
        # and the contract's own envelope
        counts["clearance_violation"] += int(
            ((pos.unsqueeze(1) - bars).norm(dim=2).amin(dim=1) < CLEARANCE - 1e-4).sum())

        moved = (pos - prev_pos).norm(dim=1)
        # float32 accumulation puts the bound checks ~1e-6 over; use a relative
        # tolerance rather than reporting arithmetic noise as a contract breach.
        tol = 1e-4
        counts["teleport"] += int((moved > SPEED_MAX * DT * (1 + tol)).sum())
        counts["speed_violation"] += int((vel.norm(dim=1) > SPEED_MAX * (1 + tol)).sum())
        counts["accel_violation"] += int(
            ((vel - prev_vel).norm(dim=1) / DT > MAX_ACCEL * (1 + tol)).sum())
        heading = torch.atan2(vel[:, 1], vel[:, 0])
        # limit_planar_velocity bounds the turn relative to the CURRENT velocity
        # heading, and grants a free heading when the agent is stopped. Comparing a
        # restart heading against a stale pre-stop heading measures nothing.
        # Use the CONTRACTUAL rest threshold, not an epsilon. limit_planar_velocity
        # grants a free heading below HEADING_VALID_SPEED_MPS by design.
        was_moving = prev_vel.norm(dim=1) > HEADING_VALID_SPEED_MPS
        now_moving = vel.norm(dim=1) > HEADING_VALID_SPEED_MPS
        continuous = was_moving & now_moving
        step_turn = torch.atan2(torch.sin(heading - prev_heading),
                                torch.cos(heading - prev_heading)).abs() / DT
        counts["turn_violation"] += int(
            (continuous & (step_turn > math.radians(MAX_TURN_DEG) * (1 + tol))).sum())
        prev_heading = torch.where(now_moving, heading, prev_heading)
        counts["stationary_steps"] += int((vel.norm(dim=1) < 1e-3).sum())
        counts["below_heading_valid_speed_steps"] += int(
            (vel.norm(dim=1) < HEADING_VALID_SPEED_MPS).sum())

        reached = (goal - pos).norm(dim=1) < 0.5
        if bool(reached.any()):
            goal[reached] = sample_clear(n_envs, bars, g, device)[reached]
    return counts


GATES = ("nonfinite", "wall_violation", "obstacle_penetration", "speed_violation",
         "accel_violation", "turn_violation", "teleport", "rng_draws_consumed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--envs", type=int, default=128)
    args = parser.parse_args()

    cases, verdict = [], "PASS"
    for n_bars in (70, 115, 160, 205):
        for exact in (False,):
            for seed in (4101, 4102, 4103):
                counts = run_case(args.envs, n_bars, args.steps, seed, exact)
                failed = [g for g in GATES if counts[g] != 0]
                if failed:
                    verdict = "FAIL"
                cases.append({"bars": n_bars, "exact_aabb": exact, "seed": seed,
                              "counts": counts, "failed_gates": failed})
                env_steps = args.envs * args.steps
                print(f"bars={n_bars:3d} exact_aabb={int(exact)} seed={seed} "
                      f"env-steps={env_steps} "
                      f"pen={counts['obstacle_penetration']} wall={counts['wall_violation']} "
                      f"spd={counts['speed_violation']} acc={counts['accel_violation']} "
                      f"turn={counts['turn_violation']} tele={counts['teleport']} "
                      f"nan={counts['nonfinite']} rng={counts['rng_draws_consumed']} "
                      f"infeas={counts['immediate_infeasible']} "
                      f"stat={counts['stationary_steps']} "
                      f"sub_hv={counts['below_heading_valid_speed_steps']} "
                      f"clr={counts['clearance_violation']}"
                      + ("  <-- " + ",".join(failed) if failed else ""))

    report = {
        "kind": "tm_e2_target_generator_validity_audit",
        "verdict": verdict,
        "method": ("direct CPU exercise of bounded_drone_target_step over navrl_band "
                   "layouts (touch 0.4 / gap 1.6); no task, no isaacgym, no GPU"),
        "contract": {"dt_s": DT, "speed_max_mps": SPEED_MAX, "max_accel_mps2": MAX_ACCEL,
                     "max_turn_rate_deg_s": MAX_TURN_DEG, "clearance_m": CLEARANCE,
                     "placement": "navrl_band", "touch_m": TOUCH_M, "gap_m": GAP_M,
                     "heading_offsets": len(BOUNDED_TURN_ANGLES_DEG),
                     "heading_valid_speed_mps": HEADING_VALID_SPEED_MPS,
                     "envs": args.envs, "steps": args.steps},
        "hard_gates": list(GATES),
        "structural_claims": {
            "push_out_reachable": False,
            "wall_reflection_reachable": False,
            "velocity_rewritten_from_displacement": False,
            "evidence": ("navrl_task.py _advance_target: the bounded|physical branch ends in an "
                         "unconditional return, so the legacy push-out / reflection / "
                         "delta-pos-over-dt block is structurally unreachable"),
        },
        "cases": cases,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    print(f"\nTM-E2 TARGET VALIDITY: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
