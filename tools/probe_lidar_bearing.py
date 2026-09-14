"""Physical adjudicator for the LiDAR bin->bearing convention.

Two candidate tables map scan bin j to a body-frame azimuth:
  increasing (what navrl_perception assumed): theta_inc[j] = -180 + bin + bin*j   [deg]
  decreasing (what warp_lidar.py generates):  theta_dec[j] =  180 - bin*j          [deg]

They are mirror images of each other (theta_inc = bin_deg - theta_dec), so at most one can be
physically correct. This probe reads the RAW scan (pre-fusion, straight from the warp sensor),
back-projects every strong return through BOTH tables into world XY, and measures the distance to
the nearest ground-truth bar center. The correct table lands returns on bars (<= half-diagonal
0.57 m + slack); the mirrored table lands them in empty space on the opposite side.

Run:  python tools/probe_lidar_bearing.py          (aerialgym conda env, PYTHONNOUSERSITE=1)
Zero training impact -- read-only diagnostic. Also usable as a regression guard after any change
to the LiDAR config, the warp sensor, or navrl_perception's angle table.
"""

import os

# Must be set BEFORE any aerial_gym import: several modules read these at import time.
for _k, _v in dict(
    NAVRL_VISION="1",
    NAVRL_PERCEPTION="1",
    NAVRL_GENERAL_TRAIN="1",
    NAVRL_MAX_OBSTACLES="8",
    NAVRL_OBSTACLE_FOV_DEG="240",
    NAVRL_OBSTACLE_SUPPRESS_DEG="10",
    NAVRL_LIDAR_HBEAMS="72",
    NAVRL_LIDAR_VBEAMS="4",
    NAVRL_LIDAR_RANGE="12",
    NAVRL_NUM_BARS="25",
    NAVRL_MAX_BARS="150",
    NAVRL_MAX_VELOCITY="2.5",
    NAVRL_ALT_HOLD_VMAX="2.5",
    NAVRL_YAW_RATE_MAX="3.0",
    NAVRL_TILT_COMP="1",
).items():
    os.environ.setdefault(_k, _v)

import isaacgym  # noqa: F401  (must precede torch)
from aerial_gym.registry.task_registry import task_registry
import math
import torch
from aerial_gym.utils.math import quat_rotate

N_ENVS = 16
STEPS = 4          # a few steps so drones drift off spawn and see bars from varied poses
NEAR_MAX = 11.5    # ignore no-return bins (fill = 12.0)
MATCH_M = 0.77     # bar half-diagonal 0.57 m + 0.2 m slack


def main():
    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=N_ENVS)
    task.reset()
    act = torch.zeros((N_ENVS, 4), device=task.device)
    stats = {}
    hb = int(os.environ["NAVRL_LIDAR_HBEAMS"])
    vb = int(os.environ["NAVRL_LIDAR_VBEAMS"])
    bin_deg = 360.0 / hb
    from aerial_gym.task.navrl_task.navrl_perception import lidar_bin_bearings

    tables = {
        "increasing(옛 perception 가정)": torch.deg2rad(
            torch.linspace(-180.0 + bin_deg, 180.0, hb, device=task.device)
        ),
        "decreasing(warp 센서 생성)": torch.deg2rad(
            torch.linspace(180.0, -180.0 + bin_deg, hb, device=task.device)
        ),
        "perception 현재 테이블": lidar_bin_bearings(task.device),
    }
    agg = {k: {"d": [], "match": []} for k in tables}

    for _ in range(STEPS):
        task.step(act)
        scan = task._lidar_distance_m().view(N_ENVS, vb, hb)
        near = scan.amin(dim=1)                                   # (N, HB) nearest per bearing
        pos = task.obs_dict["robot_position"][:, :3]
        quat = task.obs_dict["robot_vehicle_orientation"]          # yaw-only vehicle frame
        bars = task.obs_dict["obstacle_position"][:, : task.n_bars_active, 0:2]  # (N, B, 2)

        mask = near < NEAR_MAX                                     # (N, HB) real returns only
        if not bool(mask.any()):
            continue
        for name, theta in tables.items():
            pv = torch.stack(
                [near * torch.cos(theta), near * torch.sin(theta), torch.zeros_like(near)],
                dim=2,
            )                                                      # (N, HB, 3) vehicle frame
            q = quat.unsqueeze(1).expand(N_ENVS, hb, 4).reshape(-1, 4)
            pw = quat_rotate(q, pv.reshape(-1, 3)).reshape(N_ENVS, hb, 3)
            world_xy = pos[:, 0:2].unsqueeze(1) + pw[:, :, 0:2]    # (N, HB, 2)
            d = torch.cdist(world_xy, bars).min(dim=2).values      # (N, HB) nearest-bar dist
            agg[name]["d"].append(d[mask])
            agg[name]["match"].append((d[mask] < MATCH_M).float())

    print("\n===== LiDAR bin->bearing physical adjudication =====")
    n = int(torch.cat(agg[next(iter(agg))]["d"]).numel())
    print(f"returns evaluated: {n} (over {STEPS} steps x {N_ENVS} envs)")
    for name in tables:
        d = torch.cat(agg[name]["d"])
        m = torch.cat(agg[name]["match"])
        print(
            f"  {name:<28} nearest-bar dist mean={d.mean():.3f} m  median={d.median():.3f} m  "
            f"on-bar match={m.mean() * 100:.1f}%"
        )
    for name, theta in tables.items():
        stats[name] = torch.cat(agg[name]["match"]).mean().item()
    winner = max(stats, key=stats.get)
    print(f"\n  ==> physically correct table: {winner}")
    print("  (perception._lidar_angles must use this convention; mirror = left/right flipped tokens)")


if __name__ == "__main__":
    main()
