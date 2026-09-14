#!/usr/bin/env python3
"""Validate the 6-DoF target before any PPO training.

One simulator build is reused across 70/150/205/300-bar cells. The pursuer policy is not run;
only the target planner, physics-substep motor controller and PhysX contacts are measured.
Gates are declared below before results are observed and are intentionally not auto-relaxed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys


def _configure(args):
    values = {
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_SPEED": str(args.speed),
        "NAVRL_TARGET_PATTERN": args.pattern,
        "NAVRL_NUM_BARS": str(args.densities[0]),
        "NAVRL_MAX_BARS": str(max(args.densities)),
        "NAVRL_DENSITY_CURRICULUM": "0",
        "NAVRL_VISION": "0",
        "NAVRL_PERCEPTION": "0",
        "NAVRL_GENERAL_TRAIN": "1",
        "NAVRL_ARENA_XY": "40",
        "NAVRL_ARENA_Z": "3",
        "NAVRL_BAR_POOL": "bars_h3",
        "NAVRL_BAR_X_MIN": "0",
        "NAVRL_BAR_X_MAX": "1",
        "NAVRL_PLACEMENT_MODE": "navrl_band",
    }
    for key, value in values.items():
        os.environ[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", nargs="+", type=int, default=[70, 150, 205, 300])
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--pattern", choices=("cv", "waypoint", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--output", default="results/navrl_physical_target_verification/summary.json")
    args = parser.parse_args()
    _configure(args)
    sys.argv[:] = [sys.argv[0]]

    from aerial_gym.registry.task_registry import task_registry  # Isaac Gym before torch
    import torch

    # Fixed, preregistered engineering gates. Passing proves this simulation contract only.
    gates = {
        "tracking_rmse_mps_max": 0.35,
        "mean_speed_ratio_min": 0.80,
        "contact_step_fraction_max": 0.01,
        "planner_infeasible_fraction_max": 0.01,
        "motor_saturation_fraction_max": 0.15,
        "max_tilt_deg_max": 60.0,
        "invalid_state_fraction_max": 0.0,
    }
    task = task_registry.make_task(
        "navrl_task", seed=args.seed, num_envs=args.envs, headless=True, use_warp=True
    )
    env_ids = torch.arange(args.envs, device=task.device)
    zero_pursuer = torch.zeros((args.envs, 4), device=task.device)
    rows = []

    for density in args.densities:
        task._set_active_bars(density)
        task.reset()
        ctrl = task._target_controller
        speed_sum = 0.0
        err_sq_sum = 0.0
        samples = 0
        contact_samples = 0
        infeasible_samples = 0
        invalid_samples = 0
        invalid_axis_samples = [0, 0, 0]
        position_min = torch.full((3,), float("inf"), device=task.device)
        position_max = torch.full((3,), -float("inf"), device=task.device)
        for step in range(args.steps):
            ctrl.begin_control_interval()
            task._advance_target()
            task.sim_env.step(actions=zero_pursuer)
            actual = task.target_vel_w[:, :2]
            desired = ctrl.velocity_command[:, :2]
            if step >= args.warmup_steps:
                speed_sum += float(actual.norm(dim=1).sum().item())
                err_sq_sum += float(((actual - desired) ** 2).sum(dim=1).sum().item())
                samples += args.envs
                contact = ctrl.contact_seen.clone()
                contact_samples += int(contact.sum().item())
                infeasible_samples += int((~task._tm_last_step_feasible).sum().item())
                bmin = task.obs_dict["env_bounds_min"]
                bmax = task.obs_dict["env_bounds_max"]
                support_xyz = task._physical_target_support_xyz()
                support_xy = support_xyz[:, :2]
                invalid = (
                    (task.target_position[:, :2] - support_xy < bmin[:, :2])
                    | (task.target_position[:, :2] + support_xy > bmax[:, :2])
                ).any(dim=1)
                invalid |= (
                    (task.target_position[:, 2] - support_xyz[:, 2] < bmin[:, 2])
                    | (task.target_position[:, 2] + support_xyz[:, 2] > bmax[:, 2])
                    | ~torch.isfinite(task.target_position).all(dim=1)
                )
                # A bar impact can push the body marginally out of bounds in the same PhysX step.
                # Contact is already the primary failure; do not double-count its consequence as
                # an independent state-integrity defect.
                invalid_only = invalid & ~contact
                invalid_samples += int(invalid_only.sum().item())
                for axis in range(3):
                    if axis < 2:
                        axis_invalid = (
                            (task.target_position[:, axis] - support_xy[:, axis] < bmin[:, axis])
                            | (task.target_position[:, axis] + support_xy[:, axis] > bmax[:, axis])
                        )
                    else:
                        axis_invalid = (
                            (task.target_position[:, axis] - support_xyz[:, axis] < bmin[:, axis])
                            | (task.target_position[:, axis] + support_xyz[:, axis] > bmax[:, axis])
                        )
                    invalid_axis_samples[axis] += int((axis_invalid & ~contact).sum().item())
                position_min = torch.minimum(position_min, task.target_position.amin(dim=0))
                position_max = torch.maximum(position_max, task.target_position.amax(dim=0))
                failed = contact | invalid
                if bool(failed.any()):
                    failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
                    # Match task semantics: physical-target contact/invalid state terminates the
                    # episode. Continuing a crashed actor would contaminate every later tracking,
                    # planner and boundary sample with one initial failure.
                    task.sim_env.reset_idx(failed_ids)
                    task.reset_idx(failed_ids)

        diag = ctrl.diagnostics()
        mean_speed = speed_sum / max(1, samples)
        row = {
            "bars": density,
            "envs": args.envs,
            "measured_steps": args.steps - args.warmup_steps,
            "command_speed_mps": args.speed,
            "mean_speed_mps": mean_speed,
            "mean_speed_ratio": mean_speed / max(args.speed, 1e-9),
            "tracking_rmse_mps": math.sqrt(err_sq_sum / max(1, samples)),
            "contact_step_fraction": contact_samples / max(1, samples),
            "planner_infeasible_fraction": infeasible_samples / max(1, samples),
            "invalid_state_fraction": invalid_samples / max(1, samples),
            "invalid_axis_fraction_xyz": [value / max(1, samples) for value in invalid_axis_samples],
            "position_min_xyz": [float(value) for value in position_min.tolist()],
            "position_max_xyz": [float(value) for value in position_max.tolist()],
            "motor_saturation_fraction": float(diag["motor_saturation_fraction"].mean().item()),
            "max_tilt_deg": float(diag["max_tilt_deg"].max().item()),
        }
        row["gates"] = {
            "tracking": row["tracking_rmse_mps"] <= gates["tracking_rmse_mps_max"],
            "speed": row["mean_speed_ratio"] >= gates["mean_speed_ratio_min"],
            "contact": row["contact_step_fraction"] <= gates["contact_step_fraction_max"],
            "planner": row["planner_infeasible_fraction"] <= gates["planner_infeasible_fraction_max"],
            "motors": row["motor_saturation_fraction"] <= gates["motor_saturation_fraction_max"],
            "tilt": row["max_tilt_deg"] <= gates["max_tilt_deg_max"],
            "state": row["invalid_state_fraction"] <= gates["invalid_state_fraction_max"],
        }
        row["pass"] = all(row["gates"].values())
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    payload = {
        "schema": "navrl_physical_target_verification_v1",
        "seed": args.seed,
        "platform_evidence": "synthetic_ref5in_not_hardware_identified",
        "gates_preregistered": gates,
        "cells": rows,
        "all_pass": all(row["pass"] for row in rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} all_pass={payload['all_pass']}")
    return 0 if payload["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
