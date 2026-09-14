"""Measure NavRL vehicle stopping time/distance under the canonical velocity controller.

This is a controller/physics calibration, not a policy evaluation. It accelerates obstacle-free
environments with a forward body-frame command, switches to zero velocity, and records the first
time each vehicle falls below the stop threshold.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/navrl_v2_speed_governor_braking.json")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--accel-steps", type=int, default=25)
    parser.add_argument("--brake-steps", type=int, default=30)
    parser.add_argument("--stop-speed", type=float, default=0.10)
    return parser.parse_args()


ARGS = parse_args()
# Isaac Gym parses process argv during construction; remove probe-only flags first.
sys.argv[:] = [sys.argv[0]]

ENV = {
    "NAVRL_ARENA_XY": "40",
    "NAVRL_ARENA_Z": "3",
    "NAVRL_BAR_POOL": "bars_h3",
    "NAVRL_PLACEMENT_MODE": "navrl_band",
    "NAVRL_PLACEMENT_TOUCH_M": "0.4",
    "NAVRL_PLACEMENT_GAP_M": "1.6",
    "NAVRL_BAR_X_MIN": "0.0",
    "NAVRL_BAR_X_MAX": "1.0",
    "NAVRL_NUM_BARS": "0",
    "NAVRL_MAX_BARS": "10",
    "NAVRL_EPISODE_LEN_STEPS": "600",
    "NAVRL_VISION": "1",
    "NAVRL_PERCEPTION": "1",
    "NAVRL_GENERAL_TRAIN": "0",
    "NAVRL_LIDAR_HBEAMS": "72",
    "NAVRL_LIDAR_VBEAMS": "4",
    "NAVRL_LIDAR_RANGE": "12",
    "NAVRL_MAX_OBSTACLES": "8",
    "NAVRL_OBSTACLE_SELECTOR": "cluster_sector",
    "NAVRL_OBSTACLE_FOV_DEG": "240",
    "NAVRL_OBSTACLE_CLUSTER_GAP_M": "0.45",
    "NAVRL_OBSTACLE_SECTORS": "8",
    "NAVRL_CORRIDOR_TOKENS": "0",
    "NAVRL_MAX_VELOCITY": "2.5",
    "NAVRL_ALT_HOLD_VMAX": "2.5",
    "NAVRL_YAW_RATE_MAX": "3.0",
    "NAVRL_TARGET_SPEED": "0",
    "NAVRL_TARGET_PATTERN": "cv",
    "NAVRL_OOB_MARGIN": "1.0",
    "NAVRL_SPEED_GOVERNOR": "off",
}
for key, value in ENV.items():
    os.environ[key] = value

import isaacgym  # noqa: E402,F401  must precede torch
from aerial_gym.registry.task_registry import task_registry  # noqa: E402
from aerial_gym.utils.math import quat_rotate  # noqa: E402
import torch  # noqa: E402


def quantiles(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    probs = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95], dtype=torch.float64)
    q = torch.quantile(tensor, probs)
    return {
        "mean": float(tensor.mean()),
        "p05": float(q[0]),
        "p10": float(q[1]),
        "p50": float(q[2]),
        "p90": float(q[3]),
        "p95": float(q[4]),
    }


def main():
    if ARGS.envs <= 0 or ARGS.accel_steps <= 0 or ARGS.brake_steps <= 0:
        raise SystemExit("envs/steps must be positive")
    if not math.isfinite(ARGS.stop_speed) or ARGS.stop_speed <= 0.0:
        raise SystemExit("stop-speed must be finite and positive")

    task = task_registry.make_task(
        "navrl_task", seed=17, num_envs=ARGS.envs, headless=True, use_warp=True
    )
    task.reset()
    device = task.device
    # Put the virtual target far along body-forward so capture cannot reset the braking rollout.
    body_forward = torch.zeros((ARGS.envs, 3), device=device)
    body_forward[:, 0] = 1.0
    forward_world = quat_rotate(
        task.obs_dict["robot_vehicle_orientation"], body_forward
    )
    task.target_position[:] = task.obs_dict["robot_position"] + 100.0 * forward_world
    task.target_position[:, 2] = task.task_config.flight_altitude
    task.target_vel_w.zero_()
    task._tm_speed.zero_()
    task.prev_pos[:] = task.obs_dict["robot_position"]
    task.prev_rel[:] = task.prev_pos - task.target_position
    task._sync_target_to_sensor()

    accelerate = torch.zeros((ARGS.envs, 4), device=device)
    accelerate[:, 0] = 1.0
    for _ in range(ARGS.accel_steps):
        _, _, term, trunc, _ = task.step(accelerate)
        if bool(((term > 0) | (trunc > 0)).any()):
            raise RuntimeError("braking probe terminated during acceleration")

    start_pos = task.obs_dict["robot_position"][:, :2].clone()
    start_speed = task.obs_dict["robot_linvel"][:, :2].norm(dim=1).clone()
    zero = torch.zeros_like(accelerate)
    stopped = torch.zeros(ARGS.envs, dtype=torch.bool, device=device)
    stop_step = torch.zeros(ARGS.envs, dtype=torch.long, device=device)
    stop_distance = torch.zeros(ARGS.envs, device=device)
    speed_trace = []
    for step in range(1, ARGS.brake_steps + 1):
        _, _, term, trunc, _ = task.step(zero)
        if bool(((term > 0) | (trunc > 0)).any()):
            raise RuntimeError("braking probe terminated during deceleration")
        speed = task.obs_dict["robot_linvel"][:, :2].norm(dim=1)
        speed_trace.append(float(speed.mean().item()))
        newly = (~stopped) & (speed <= ARGS.stop_speed)
        if bool(newly.any()):
            stopped[newly] = True
            stop_step[newly] = step
            stop_distance[newly] = (
                task.obs_dict["robot_position"][newly, :2] - start_pos[newly]
            ).norm(dim=1)
        if bool(stopped.all()):
            break

    if not bool(stopped.all()):
        raise RuntimeError(
            f"only {int(stopped.sum())}/{ARGS.envs} vehicles stopped within {ARGS.brake_steps} steps"
        )
    stop_time = stop_step.float() * float(task.step_dt)
    effective_decel = start_speed.square() / (2.0 * stop_distance.clamp(min=1e-6))
    time_stats = quantiles(stop_time.cpu())
    distance_stats = quantiles(stop_distance.cpu())
    decel_stats = quantiles(effective_decel.cpu())
    recommended_ttc = max(0.8, time_stats["p95"] + 0.2)
    recommended_brake = decel_stats["p10"]

    payload = {
        "schema_version": 1,
        "probe": "canonical_navrl_zero_velocity_braking",
        "envs": ARGS.envs,
        "physics": {
            "sim": task.task_config.sim_name,
            "rl_step_dt_s": float(task.step_dt),
            "max_velocity_mps": float(task.task_config.max_velocity),
            "accel_steps": ARGS.accel_steps,
            "brake_steps_budget": ARGS.brake_steps,
            "stop_threshold_mps": ARGS.stop_speed,
        },
        "initial_speed_mps": quantiles(start_speed.cpu()),
        "stop_time_s": time_stats,
        "stop_distance_m": distance_stats,
        "effective_deceleration_mps2": decel_stats,
        "mean_speed_trace_mps": speed_trace,
        "recommended": {
            "ttc_s": recommended_ttc,
            "brake_mps2": recommended_brake,
            "reaction_s": float(task.step_dt),
            "rule": "ttc=max(0.8,p95_stop_time+0.2); brake=p10_effective_deceleration",
        },
    }
    output = Path(ARGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["recommended"], sort_keys=True))
    print(f"[braking] saved -> {output}")


if __name__ == "__main__":
    main()
