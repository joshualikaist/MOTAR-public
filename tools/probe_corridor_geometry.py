"""P1 corridor-geometry validation probe: extracted corridors vs ground-truth bar layout.

Runs the real perception path (fused LiDAR profile) WITHOUT changing the actor observation,
extracts corridor tokens, and physically adjudicates them against GT bar centers:

  A. center-ray clearance -- sample the corridor center ray out to min(depth, horizon); the
     minimum XY distance to any active GT bar center must be >= 0.20 m (the smallest bar
     half-width). A corridor whose center ray runs through a bar is a false affordance.
  B. bounding-surface realism -- for interior corridors, each bounding endpoint (range at the
     first blocked bin) must lie ON a bar: distance to the nearest GT bar center <= 0.87 m
     (max half-diagonal 0.57 + 0.30 tolerance).
  C. width sanity -- interior corridor chord vs the center distance of the two bounding bars:
     |chord - center_dist| <= 1.2 m (their unknown half-widths bound the gap on both sides).

Usage (GPU free):
  NAVRL_VISION=1 NAVRL_PERCEPTION=1 NAVRL_GENERAL_TRAIN=1 NAVRL_LIDAR_RANGE=12 \
  NAVRL_LIDAR_HBEAMS=72 NAVRL_LIDAR_VBEAMS=4 NAVRL_MAX_OBSTACLES=8 \
  NAVRL_OBSTACLE_SELECTOR=cluster_sector NAVRL_OBSTACLE_FOV_DEG=240 \
  NAVRL_MAX_BARS=150 NAVRL_NUM_BARS=100 PYTHONNOUSERSITE=1 \
    python tools/probe_corridor_geometry.py
"""

import math
import os
import sys

os.environ.setdefault("NAVRL_VISION", "1")
os.environ.setdefault("NAVRL_PERCEPTION", "1")

from aerial_gym.registry.task_registry import task_registry  # noqa: E402  isaacgym before torch
import torch  # noqa: E402

from aerial_gym.task.navrl_task.navrl_corridor import extract_corridor_tokens  # noqa: E402
from aerial_gym.task.navrl_task.navrl_perception import (  # noqa: E402
    OBSTACLE_FOV_DEG,
    lidar_bin_bearings,
    _quat_rotate_xyzw,
)

N = int(os.environ.get("PROBE_ENVS", "64"))
STEPS = int(os.environ.get("PROBE_STEPS", "40"))
K = int(os.environ.get("PROBE_CORRIDORS", "6"))
HORIZON = float(os.environ.get("PROBE_HORIZON_M", "6.0"))
MIN_W = float(os.environ.get("PROBE_MIN_WIDTH_M", "0.55"))

task = task_registry.make_task("navrl_task", seed=11, num_envs=N, headless=True, use_warp=True)
device = task.device
bearings = lidar_bin_bearings(device)
task.reset()

tot = {"corr": 0, "envsteps": 0,
       "A_pass": 0, "A_n": 0, "B_pass": 0, "B_n": 0, "C_pass": 0, "C_n": 0}
ray_clear_all = []
width_err_all = []

for step in range(STEPS):
    act = 0.4 * (2.0 * torch.rand(N, 4, device=device) - 1.0)
    task.step(act)
    nearest = getattr(task.perception, "last_scan_nearest", None)
    if nearest is None:
        continue

    tokens, aux = extract_corridor_tokens(
        nearest, bearings, max_range=float(task.perception.lidar_max_range),
        num_corridors=K, fov_deg=OBSTACLE_FOV_DEG, horizon_m=HORIZON, min_width_m=MIN_W,
    )
    valid = aux["valid"]                        # [N, K]
    if not bool(valid.any()):
        continue

    pos = task.obs_dict["robot_position"]       # [N, 3] world
    quat = task.obs_dict["robot_vehicle_orientation"]
    bars = task.obs_dict["obstacle_position"][:, : task.n_bars_active, 0:2]  # [N, B, 2] world

    center = aux["center"]                      # body-frame bearing
    depth = torch.minimum(aux["depth_m"], torch.full_like(aux["depth_m"], HORIZON))

    # A: center-ray clearance against GT bar centers.
    tsteps = torch.linspace(0.3, 1.0, 16, device=device).view(1, 1, 16)
    ray_dir_body = torch.stack(
        [torch.cos(center), torch.sin(center), torch.zeros_like(center)], dim=2
    )                                            # [N, K, 3]
    ray_dir_world = _quat_rotate_xyzw(
        quat.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
        ray_dir_body.reshape(-1, 3),
    ).view(N, K, 3)[..., 0:2]
    pts = pos[:, None, None, 0:2] + ray_dir_world.unsqueeze(2) * (
        depth.unsqueeze(2) * tsteps
    ).unsqueeze(3)                               # [N, K, T, 2]
    d_bar = torch.cdist(pts.reshape(N, -1, 2), bars).view(N, K, 16, -1)
    ray_clear = d_bar.min(dim=3).values.min(dim=2).values     # [N, K]
    a_pass = (ray_clear >= 0.20) & valid
    tot["A_pass"] += int(a_pass.sum()); tot["A_n"] += int(valid.sum())
    ray_clear_all.append(ray_clear[valid])

    # B/C: interior corridors only (both bounding surfaces exist -> clearance < max range).
    interior = valid & (aux["left_clear_m"] < 11.9) & (aux["right_clear_m"] < 11.9)
    if bool(interior.any()):
        # bounding endpoints in body frame: range at the bearing just outside the run edge.
        half_bin = math.pi / bearings.shape[0]
        bl = center + 0.5 * aux["ang_width"] + half_bin
        br = center - 0.5 * aux["ang_width"] - half_bin
        for side, ang, rng in (("L", bl, aux["left_clear_m"]), ("R", br, aux["right_clear_m"])):
            ep_body = torch.stack(
                [rng * torch.cos(ang), rng * torch.sin(ang), torch.zeros_like(rng)], dim=2
            )
            ep_world = _quat_rotate_xyzw(
                quat.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4), ep_body.reshape(-1, 3)
            ).view(N, K, 3)[..., 0:2] + pos[:, None, 0:2]
            d = torch.cdist(ep_world.reshape(N, K, 2), bars).min(dim=2).values
            b_pass = (d <= 0.87) & interior
            tot["B_pass"] += int(b_pass.sum()); tot["B_n"] += int(interior.sum())
            if side == "L":
                epL = ep_world; dL_idx = torch.cdist(ep_world.reshape(N, K, 2), bars).argmin(dim=2)
            else:
                epR = ep_world; dR_idx = torch.cdist(ep_world.reshape(N, K, 2), bars).argmin(dim=2)
        rowsK = torch.arange(N, device=device).view(N, 1).expand(N, K)
        barL = bars[rowsK.reshape(-1), dL_idx.reshape(-1)].view(N, K, 2)
        barR = bars[rowsK.reshape(-1), dR_idx.reshape(-1)].view(N, K, 2)
        gt_center_dist = (barL - barR).norm(dim=2)
        werr = (aux["width_m"] - gt_center_dist).abs()
        c_pass = (werr <= 1.2) & interior & (dL_idx != dR_idx)
        c_n = interior & (dL_idx != dR_idx)
        tot["C_pass"] += int(c_pass.sum()); tot["C_n"] += int(c_n.sum())
        width_err_all.append(werr[c_n])

    tot["corr"] += int(valid.sum())
    tot["envsteps"] += N

rc = torch.cat(ray_clear_all) if ray_clear_all else torch.zeros(1)
we = torch.cat(width_err_all) if width_err_all else torch.zeros(1)

print("\n================ corridor geometry probe ================")
print(f"envs={N} steps={STEPS} bars_active={int(task.n_bars_active)} "
      f"K={K} horizon={HORIZON} min_width={MIN_W}")
print(f"corridors/env-step        : {tot['corr'] / max(1, tot['envsteps']):.2f}")
aR = 100.0 * tot["A_pass"] / max(1, tot["A_n"])
bR = 100.0 * tot["B_pass"] / max(1, tot["B_n"])
cR = 100.0 * tot["C_pass"] / max(1, tot["C_n"])
print(f"A center-ray clear >=0.2m : {aR:5.1f}%  (n={tot['A_n']}; "
      f"clearance p10/p50 = {rc.quantile(0.1):.2f}/{rc.quantile(0.5):.2f} m)")
print(f"B bound-on-bar <=0.87m    : {bR:5.1f}%  (n={tot['B_n']})")
print(f"C width vs GT <=1.2m      : {cR:5.1f}%  (n={tot['C_n']}; "
      f"err p50/p90 = {we.quantile(0.5):.2f}/{we.quantile(0.9):.2f} m)")
ok = aR >= 95.0 and bR >= 90.0 and cR >= 85.0
print("VERDICT:", "PASS" if ok else "FAIL",
      "(gates: A>=95, B>=90, C>=85)")
sys.exit(0 if ok else 1)
