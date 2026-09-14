#!/usr/bin/env python3
"""Measure a preregistered feasible-speed envelope for the physical target.

This is an engineering diagnostic, not PPO training and not a replacement for hardware
identification.  It reuses the existing physical-target controller and tests a fixed density x
speed grid.  The highest speed that satisfies the existing tracking/contact/feasibility gates is
reported per density; no speed is selected after looking at an unregistered grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


GATES = {
    "tracking_rmse_mps_max": 0.35,
    "mean_speed_ratio_min": 0.80,
    "contact_step_fraction_max": 0.01,
    "planner_infeasible_fraction_max": 0.01,
    "motor_saturation_fraction_max": 0.15,
    "max_tilt_deg_max": 60.0,
    "invalid_state_fraction_max": 0.0,
}
DEFAULT_SPEEDS = (0.6, 0.9, 1.2, 1.5)
DEFAULT_DENSITIES = (70, 150, 205, 300)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure(densities, speed, pattern):
    values = {
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_SPEED": str(speed),
        "NAVRL_TARGET_PATTERN": pattern,
        "NAVRL_NUM_BARS": str(densities[0]),
        "NAVRL_MAX_BARS": str(max(densities)),
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


def _cell(task, torch, density, speed, args):
    task._set_active_bars(density)
    task.reset()
    ctrl = task._target_controller
    speed_sum = err_sq_sum = 0.0
    samples = contact_samples = infeasible_samples = invalid_samples = 0
    invalid_axis_samples = [0, 0, 0]
    for step in range(args.steps):
        ctrl.begin_control_interval()
        task._advance_target()
        task.sim_env.step(actions=torch.zeros((args.envs, 4), device=task.device))
        actual = task.target_vel_w[:, :2]
        desired = ctrl.velocity_command[:, :2]
        if step < args.warmup_steps:
            continue
        speed_sum += float(actual.norm(dim=1).sum().item())
        err_sq_sum += float(((actual - desired) ** 2).sum(dim=1).sum().item())
        samples += args.envs
        contact = ctrl.contact_seen.clone()
        contact_samples += int(contact.sum().item())
        infeasible_samples += int((~task._tm_last_step_feasible).sum().item())
        bmin = task.obs_dict["env_bounds_min"]
        bmax = task.obs_dict["env_bounds_max"]
        support = task._physical_target_support_xyz()
        invalid = (
            (task.target_position[:, :2] - support[:, :2] < bmin[:, :2])
            | (task.target_position[:, :2] + support[:, :2] > bmax[:, :2])
        ).any(dim=1)
        invalid |= (
            (task.target_position[:, 2] - support[:, 2] < bmin[:, 2])
            | (task.target_position[:, 2] + support[:, 2] > bmax[:, 2])
            | ~torch.isfinite(task.target_position).all(dim=1)
        )
        invalid_only = invalid & ~contact
        invalid_samples += int(invalid_only.sum().item())
        for axis in range(3):
            extent = support[:, axis]
            axis_invalid = (
                (task.target_position[:, axis] - extent < bmin[:, axis])
                | (task.target_position[:, axis] + extent > bmax[:, axis])
            )
            invalid_axis_samples[axis] += int((axis_invalid & ~contact).sum().item())
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            task.sim_env.reset_idx(failed_ids)
            task.reset_idx(failed_ids)
    diag = ctrl.diagnostics()
    mean_speed = speed_sum / max(1, samples)
    row = {
        "bars": density,
        "speed_mps": speed,
        "envs": args.envs,
        "measured_steps": args.steps - args.warmup_steps,
        "mean_speed_mps": mean_speed,
        "mean_speed_ratio": mean_speed / max(speed, 1e-9),
        "tracking_rmse_mps": math.sqrt(err_sq_sum / max(1, samples)),
        "contact_step_fraction": contact_samples / max(1, samples),
        "planner_infeasible_fraction": infeasible_samples / max(1, samples),
        "invalid_state_fraction": invalid_samples / max(1, samples),
        "invalid_axis_fraction_xyz": [v / max(1, samples) for v in invalid_axis_samples],
        "motor_saturation_fraction": float(diag["motor_saturation_fraction"].mean().item()),
        "max_tilt_deg": float(diag["max_tilt_deg"].max().item()),
    }
    row["gates"] = {
        "tracking": row["tracking_rmse_mps"] <= GATES["tracking_rmse_mps_max"],
        "speed": row["mean_speed_ratio"] >= GATES["mean_speed_ratio_min"],
        "contact": row["contact_step_fraction"] <= GATES["contact_step_fraction_max"],
        "planner": row["planner_infeasible_fraction"] <= GATES["planner_infeasible_fraction_max"],
        "motors": row["motor_saturation_fraction"] <= GATES["motor_saturation_fraction_max"],
        "tilt": row["max_tilt_deg"] <= GATES["max_tilt_deg_max"],
        "state": row["invalid_state_fraction"] <= GATES["invalid_state_fraction_max"],
    }
    row["pass"] = all(row["gates"].values())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", nargs="+", type=int, default=list(DEFAULT_DENSITIES))
    parser.add_argument("--speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS))
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--pattern", choices=("cv", "waypoint", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument(
        "--output",
        default="results/navrl_physical_target_speed_envelope_seed509/summary.json",
    )
    parser.add_argument("--_single-speed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sorted(set(args.speeds)) != list(args.speeds) or any(v <= 0 for v in args.speeds):
        parser.error("--speeds must be strictly increasing positive values")
    if sorted(set(args.densities)) != list(args.densities) or any(v <= 0 for v in args.densities):
        parser.error("--densities must be strictly increasing positive values")
    # Isaac Gym cannot safely destroy and recreate multiple simulations in one interpreter. Run
    # each speed arm in its own child process, then aggregate the immutable child summaries. This
    # keeps every arm's controller/RNG state isolated while preserving one final receipt-like JSON.
    if len(args.speeds) > 1 and not args._single_speed:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        child_dir = output.parent / (output.stem + ".cells")
        child_dir.mkdir(parents=True, exist_ok=True)
        for speed in args.speeds:
            child = child_dir / ("speed_" + str(speed).replace(".", "p") + ".json")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--densities",
                *(str(v) for v in args.densities),
                "--speeds",
                str(speed),
                "--envs",
                str(args.envs),
                "--steps",
                str(args.steps),
                "--warmup-steps",
                str(args.warmup_steps),
                "--pattern",
                args.pattern,
                "--seed",
                str(args.seed),
                "--output",
                str(child),
                "--_single-speed",
            ]
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
            # Exit 2 means that this child speed had no passing density; that is a valid measured
            # result for an envelope. Any other non-zero exit is an execution failure.
            if completed.returncode not in (0, 2):
                raise SystemExit(completed.returncode)
            child_payload = json.loads(child.read_text(encoding="utf-8"))
            rows.extend(child_payload["cells"])
        highest_passing = {}
        for density in args.densities:
            passing = [r["speed_mps"] for r in rows if r["bars"] == density and r["pass"]]
            highest_passing[str(density)] = max(passing) if passing else None
        payload = {
            "schema": "navrl_physical_target_speed_envelope_v1",
            "seed": args.seed,
            "platform_evidence": "synthetic_ref5in_not_hardware_identified",
            "pattern": args.pattern,
            "densities": args.densities,
            "speeds_mps": args.speeds,
            "gates_preregistered": GATES,
            "cells": rows,
            "highest_passing_speed_mps_by_density": highest_passing,
            "all_cells_pass": all(row["pass"] for row in rows),
            "arm_process_isolation": True,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        receipt = output.with_name(output.stem + ".receipt.json")
        receipt.write_text(
            json.dumps(
                {
                    "schema": "navrl_physical_target_speed_envelope_receipt_v1",
                    "summary_sha256": _sha256(output),
                    "tool_sha256": _sha256(Path(__file__).resolve()),
                    "cells": [
                        str(p.resolve().relative_to(Path(__file__).resolve().parents[1]))
                        for p in sorted(child_dir.glob("speed_*.json"))
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {output} all_cells_pass={payload['all_cells_pass']}")
        return 0 if any(value is not None for value in highest_passing.values()) else 2

    _configure(args.densities, args.speeds[0], args.pattern)
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry  # Isaac Gym before torch
    import torch

    task = task_registry.make_task(
        "navrl_task", seed=args.seed, num_envs=args.envs, headless=True, use_warp=True
    )
    rows = []
    speed = args.speeds[0]
    os.environ["NAVRL_TARGET_SPEED"] = str(speed)
    for density in args.densities:
        row = _cell(task, torch, density, speed, args)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    highest_passing = {}
    for density in args.densities:
        passing = [r["speed_mps"] for r in rows if r["bars"] == density and r["pass"]]
        highest_passing[str(density)] = max(passing) if passing else None
    payload = {
        "schema": "navrl_physical_target_speed_envelope_v1",
        "seed": args.seed,
        "platform_evidence": "synthetic_ref5in_not_hardware_identified",
        "pattern": args.pattern,
        "densities": args.densities,
        "speeds_mps": args.speeds,
        "gates_preregistered": GATES,
        "cells": rows,
        "highest_passing_speed_mps_by_density": highest_passing,
        "all_cells_pass": all(row["pass"] for row in rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} all_cells_pass={payload['all_cells_pass']}")
    return 0 if any(value is not None for value in highest_passing.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
