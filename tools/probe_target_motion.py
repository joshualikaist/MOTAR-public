"""Measure how the moving target ACTUALLY moves vs its commanded speed.

Suspicion: the cv pattern reflects at walls but has no deflection at bars, so a target heading
into a bar is pushed out every step and effectively parks there (realized speed ~ 0) until a wall
reflection eventually changes its heading. This probe quantifies it: realized per-step displacement
of target_position vs commanded speed * dt, per pattern, plus stall/overspeed/clearance rates.

The probe calls ``_advance_target`` directly. Advancing the whole task with zero drone actions
causes drone crashes and environment resets, which can contaminate target displacement with
teleports and newly sampled motion patterns.

Run: python tools/probe_target_motion.py [bars] [speed]
"""

import os
import sys

BARS = sys.argv[1] if len(sys.argv) > 1 else "25"
SPEED = sys.argv[2] if len(sys.argv) > 2 else "1.5"
# Isaac Gym parses process argv during environment construction. Remove probe-only positional
# arguments so they are not reported as unknown simulator flags.
sys.argv[:] = [sys.argv[0]]

for _k, _v in dict(
    NAVRL_VISION="1", NAVRL_PERCEPTION="1", NAVRL_GENERAL_TRAIN="1",
    NAVRL_MAX_OBSTACLES="8", NAVRL_OBSTACLE_FOV_DEG="240", NAVRL_OBSTACLE_SUPPRESS_DEG="10",
    NAVRL_LIDAR_HBEAMS="72", NAVRL_LIDAR_VBEAMS="4", NAVRL_LIDAR_RANGE="12",
    NAVRL_NUM_BARS=BARS, NAVRL_MAX_BARS=str(max(150, int(BARS))),
    NAVRL_MAX_VELOCITY="2.5", NAVRL_ALT_HOLD_VMAX="2.5", NAVRL_YAW_RATE_MAX="3.0",
    NAVRL_TILT_COMP="1", NAVRL_TARGET_SPEED=SPEED, NAVRL_TARGET_PATTERN="mixed",
).items():
    os.environ.setdefault(_k, _v)

import isaacgym  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
import torch

N = int(os.environ.get("NAVRL_TARGET_PROBE_ENVS", "64"))
STEPS = int(os.environ.get("NAVRL_TARGET_PROBE_STEPS", "300"))


def main():
    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=N)
    task.reset()
    dt = task.step_dt
    cmd = float(SPEED)
    measured_clearance = (
        float(task.tm.obstacle_clearance)
        if str(getattr(task, "_target_dynamics", "legacy")) == "bounded"
        else float(getattr(task.task_config, "goal_min_bar_clearance", 1.0))
    )
    per_pattern = {0: [], 1: []}
    stall = {0: 0, 1: 0}
    overspeed = {0: 0, 1: 0}
    clearance_violation = {0: 0, 1: 0}
    infeasible = {0: 0, 1: 0}
    accel_violation = {0: 0, 1: 0}
    turn_violation = {0: 0, 1: 0}
    count = {0: 0, 1: 0}
    prev = task.target_position[:, 0:2].clone()
    prev_vel = task.target_vel_w[:, 0:2].clone()
    initial_bars = task.obs_dict["obstacle_position"][:, : task.n_bars_active, 0:2]
    initial_min_bar = (
        torch.cdist(prev.unsqueeze(1), initial_bars).squeeze(1).amin(dim=1)
        if initial_bars.shape[1] > 0
        else torch.full((N,), float("inf"), device=task.device)
    )
    initial_clearance_violations = int(
        (initial_min_bar < measured_clearance - 1e-4).sum().item()
    )
    for _ in range(STEPS):
        task._advance_target()
        cur = task.target_position[:, 0:2]
        step_d = (cur - prev).norm(dim=1)
        velocity = task.target_vel_w[:, 0:2]
        acceleration = (velocity - prev_vel).norm(dim=1) / dt
        old_heading = torch.atan2(prev_vel[:, 1], prev_vel[:, 0])
        new_heading = torch.atan2(velocity[:, 1], velocity[:, 0])
        heading_delta = torch.atan2(
            torch.sin(new_heading - old_heading), torch.cos(new_heading - old_heading)
        ).abs() / dt
        bars_xy = task.obs_dict["obstacle_position"][
            :, : task.n_bars_active, 0:2
        ]
        min_bar_d = (
            torch.cdist(cur.unsqueeze(1), bars_xy).squeeze(1).amin(dim=1)
            if bars_xy.shape[1] > 0
            else torch.full((N,), float("inf"), device=task.device)
        )
        for pat in (0, 1):
            m = (task._tm_pattern == pat) & (task._tm_speed > 1e-6)
            if bool(m.any()):
                per_pattern[pat].append(step_d[m])
                stall[pat] += int((step_d[m] < 0.2 * cmd * dt).sum().item())
                overspeed[pat] += int((step_d[m] > 1.2 * cmd * dt).sum().item())
                clearance_violation[pat] += int(
                    (min_bar_d[m] < measured_clearance - 1e-4).sum().item()
                )
                feasible = getattr(
                    task,
                    "_tm_last_step_feasible",
                    torch.ones(N, dtype=torch.bool, device=task.device),
                )
                infeasible[pat] += int((~feasible[m]).sum().item())
                accel_limit = float(getattr(task.tm, "max_accel", float("inf")))
                turn_limit = torch.deg2rad(torch.tensor(
                    float(getattr(task.tm, "max_turn_rate_deg", float("inf"))),
                    device=task.device,
                ))
                accel_violation[pat] += int((acceleration[m] > accel_limit + 1e-4).sum().item())
                # Direction of travel is numerically/physically undefined near hover. Only audit
                # heading slew once the target has meaningful translational speed.
                valid_heading = m & (prev_vel.norm(dim=1) > 0.2) & (velocity.norm(dim=1) > 0.2)
                turn_violation[pat] += int((heading_delta[valid_heading] > turn_limit + 1e-4).sum().item())
                count[pat] += int(m.sum().item())
        prev = cur.clone()
        prev_vel = velocity.clone()

    names = {0: "cv", 1: "waypoint"}
    print(f"\n===== target motion probe | bars={BARS} commanded={cmd} m/s (step {cmd*dt:.2f} m)"
          f" initial_clearance_violations={initial_clearance_violations}/{N} =====")
    for pat in (0, 1):
        if not per_pattern[pat]:
            continue
        d = torch.cat(per_pattern[pat])
        q = torch.quantile(d / dt, torch.tensor([0.05, 0.50, 0.95], device=d.device))
        print(
            f"  {names[pat]:<9} realized speed mean={d.mean()/dt:.2f} m/s"
            f"  (={d.mean()/dt/cmd*100:4.0f}% of commanded)"
            f"  p05/50/95={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}"
            f"  stall(<20%)={stall[pat]/max(1,count[pat])*100:4.1f}%"
            f"  overspeed(>120%)={overspeed[pat]/max(1,count[pat])*100:4.1f}%"
            f"  clearance<{measured_clearance:.2f}m="
            f"{clearance_violation[pat]/max(1,count[pat])*100:4.1f}%"
            f"  infeasible={infeasible[pat]/max(1,count[pat])*100:4.1f}%"
            f"  accel_violation={accel_violation[pat]/max(1,count[pat])*100:4.1f}%"
            f"  turn_violation={turn_violation[pat]/max(1,count[pat])*100:4.1f}%"
            f"  n={count[pat]}"
        )


if __name__ == "__main__":
    main()
