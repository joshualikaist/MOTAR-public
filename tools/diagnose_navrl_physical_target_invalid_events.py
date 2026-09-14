#!/usr/bin/env python3
"""Forensic diagnostic for physical-target invalid OBB events.

This does not change a gate, controller, reward, or PPO contract.  It reruns the fixed
physical-target gate conditions at 205 bars and records the exact OBB support, arena margins,
target velocity, command, and task step for every invalid non-contact sample.  The purpose is to
separate a genuine target-boundary bug from a rare reset/finite-state artifact before any
controller or training change is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DENSITY = 205
SPEEDS = (0.6, 0.9, 1.2, 1.5)
SEED = 509
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure(speed: float) -> None:
    import os

    values = {
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_SPEED": str(speed),
        "NAVRL_TARGET_PATTERN": "mixed",
        "NAVRL_NUM_BARS": str(DENSITY),
        "NAVRL_MAX_BARS": str(DENSITY),
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


def _invalid_mask(task, torch):
    bmin = task.obs_dict["env_bounds_min"]
    bmax = task.obs_dict["env_bounds_max"]
    support = task._physical_target_support_xyz()
    pos = task.target_position
    invalid = (
        (pos[:, :2] - support[:, :2] < bmin[:, :2])
        | (pos[:, :2] + support[:, :2] > bmax[:, :2])
    ).any(dim=1)
    invalid |= (
        (pos[:, 2] - support[:, 2] < bmin[:, 2])
        | (pos[:, 2] + support[:, 2] > bmax[:, 2])
        | ~torch.isfinite(pos).all(dim=1)
    )
    return invalid, support, bmin, bmax


def _run_one(speed: float, output: Path) -> int:
    import os

    _configure(speed)
    # Import Isaac Gym only after all contract environment variables are fixed.
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry
    import torch

    task = task_registry.make_task(
        "navrl_task", seed=SEED, num_envs=ENVS, headless=True, use_warp=True
    )
    task._set_active_bars(DENSITY)
    task.reset()
    ctrl = task._target_controller
    events = []
    episode_start = task.target_position.detach().clone()
    measured_samples = invalid_samples = contact_samples = 0
    for step in range(STEPS):
        pre_step_position = task.target_position.detach().clone()
        ctrl.begin_control_interval()
        task._advance_target()
        task.sim_env.step(actions=torch.zeros((ENVS, 4), device=task.device))
        if step < WARMUP_STEPS:
            continue
        invalid, support, bmin, bmax = _invalid_mask(task, torch)
        contact = ctrl.contact_seen.clone()
        invalid_only = invalid & ~contact
        measured_samples += ENVS
        invalid_samples += int(invalid_only.sum().item())
        contact_samples += int(contact.sum().item())
        if bool(invalid_only.any()):
            for env_id in invalid_only.nonzero(as_tuple=False).flatten().tolist():
                position = task.target_position[env_id].detach().cpu().tolist()
                half = support[env_id].detach().cpu().tolist()
                lower = bmin[env_id].detach().cpu().tolist()
                upper = bmax[env_id].detach().cpu().tolist()
                margin = [
                    min(
                        position[axis] - half[axis] - lower[axis],
                        upper[axis] - half[axis] - position[axis],
                    )
                    for axis in range(3)
                ]
                velocity = task.target_vel_w[env_id].detach().cpu().tolist()
                command = ctrl.velocity_command[env_id].detach().cpu().tolist()
                events.append(
                    {
                        "step": step,
                        "env": env_id,
                        "episode_start_position_m": episode_start[env_id].detach().cpu().tolist(),
                        "pre_step_position_m": pre_step_position[env_id].detach().cpu().tolist(),
                        "position_m": position,
                        "velocity_mps": velocity,
                        "command_mps": command,
                        "support_half_extents_m": half,
                        "bounds_min_m": lower,
                        "bounds_max_m": upper,
                        "obb_margin_m_xyz": margin,
                        "contact_seen": bool(contact[env_id].item()),
                        "finite_position": bool(torch.isfinite(task.target_position[env_id]).all().item()),
                    }
                )
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            task.sim_env.reset_idx(failed_ids)
            task.reset_idx(failed_ids)
            episode_start[failed_ids] = task.target_position[failed_ids].detach()
    payload = {
        "schema": "navrl_physical_target_invalid_event_cell_v1",
        "seed": SEED,
        "bars": DENSITY,
        "speed_mps": speed,
        "pattern": "mixed",
        "envs": ENVS,
        "steps": STEPS,
        "warmup_steps": WARMUP_STEPS,
        "measured_samples": measured_samples,
        "invalid_noncontact_samples": invalid_samples,
        "contact_samples": contact_samples,
        "events": events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"speed_mps": speed, "events": len(events), "output": str(output)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/navrl_physical_target_invalid_forensics_seed509/summary.json")
    parser.add_argument("--_single-speed", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    cell_dir = output.parent / (output.stem + ".cells")
    if args._single_speed is not None:
        return _run_one(float(args._single_speed), output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for speed in SPEEDS:
        cell = cell_dir / ("speed_" + str(speed).replace(".", "p") + ".json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(cell),
            "--_single-speed",
            str(speed),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        cells.append(json.loads(cell.read_text(encoding="utf-8")))
    payload = {
        "schema": "navrl_physical_target_invalid_event_forensics_v1",
        "contract": {
            "seed": SEED,
            "bars": DENSITY,
            "speeds_mps": list(SPEEDS),
            "pattern": "mixed",
            "envs": ENVS,
            "steps": STEPS,
            "warmup_steps": WARMUP_STEPS,
        },
        "cells": cells,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    receipt = output.with_name(output.stem + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "schema": "navrl_physical_target_invalid_event_forensics_receipt_v1",
                "summary_sha256": _sha256(output),
                "tool_sha256": _sha256(Path(__file__).resolve()),
                "cells": [
                    str(p.resolve().relative_to(ROOT))
                    for p in sorted(cell_dir.glob("speed_*.json"))
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {output} events={sum(len(cell['events']) for cell in cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
