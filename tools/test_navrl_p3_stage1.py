"""Phase-3 Stage-1 verification: sensor-only actor obs + camera detector + privileged states.

Run (GPU free):
  NAVRL_VISION=1 NAVRL_MAX_BARS=40 NAVRL_NUM_BARS=40 PYTHONNOUSERSITE=1 \
    python tools/test_navrl_p3_stage1.py

Checks:
  A. dims: actor obs 1265, states 1273, internal split 17.
  B. detector geometry, forced scenarios per env:
     - target 3 m directly ahead, clear LOS       -> visible, bearing ~0
     - target 3 m BEHIND (out of the 87-deg FOV)  -> not visible
     - target beyond detector range (25 m)        -> not visible
     - target ahead but occluded by a bar         -> not visible (warp LOS)
  C. no GT leakage: the actor slice contains no monotone function of target position when the
     camera detector is blind (obs must be IDENTICAL for two different target positions that are
     both outside camera and LiDAR range).
  D. tracker: after seeing then losing the target, last-bearing persists and t_since grows.
  E. body-frame action: +x action command equals vehicle-frame velocity command.
  F. states tail carries the GT extras (rpos unit etc).
"""
import os

os.environ.setdefault("NAVRL_VISION", "1")
from aerial_gym.registry.task_registry import task_registry  # isaacgym before torch
import torch

N = 8
fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


task = task_registry.make_task("navrl_task", seed=7, num_envs=N, headless=True, use_warp=True)
tc = task.task_config
vc = tc.vision

# A. dims
check("vision mode on", task.vision_mode)
check("obs dim 1265", tc.observation_space_dim == 1265, f"({tc.observation_space_dim})")
check("states dim 1273", tc.state_space_dim == 1273, f"({tc.state_space_dim})")
check("split 17", tc.internal_state_dim == 17)
obs0 = task.reset()[0]
check("obs buffer shape", tuple(obs0["observations"].shape) == (N, 1265))
check("states buffer shape", tuple(obs0["states"].shape) == (N, 1273))

act = torch.zeros(N, 4, device=task.device)
task.tm.speed_fixed = 0.0
pos = task.obs_dict["robot_position"]
q_veh = task.obs_dict["robot_vehicle_orientation"]

from aerial_gym.utils.math import quat_rotate  # vehicle-frame +x in world coords

fwd_w = quat_rotate(q_veh, torch.tensor([[1.0, 0.0, 0.0]], device=task.device).expand(N, 3))


def place_and_observe(offset_w):
    """Teleport the virtual target to drone + offset (world), render, and rebuild observation."""
    task.target_position[:] = task.obs_dict["robot_position"] + offset_w
    task._sync_target_to_sensor()
    task.sim_env.render(render_components="sensors")
    task.process_obs_for_task()
    return task.task_obs["observations"].clone(), task.task_obs["states"].clone()


# B1. 1.5 m directly ahead (level): inside the bar-free spawn strip (bars start at
# x=3.1 m; drone spawns at x<=1 with +-30 deg yaw) -> LOS guaranteed clear.
obs_a, st_a = place_and_observe(1.5 * fwd_w)
det = obs_a[:, vc.ego_dim : vc.ego_dim + vc.detector_dim]
check("ahead: visible", bool((det[:, 0] > 0.5).all()), f"(vis={det[:, 0].tolist()})")
check("ahead: bearing ~0", bool((det[:, 1].abs() < 0.15).all()), f"(sin={det[:, 1].abs().max():.3f})")
check("ahead: range ~1.5/20", bool(((det[:, 4] - 0.075).abs() < 0.02).all()))
cam_hits = task.obs_dict["target_camera_mask"].sum(dim=(1, 2))
check("ahead: camera target pixels exist", bool((cam_hits > 0).all()),
      f"(pixels={cam_hits.tolist()})")

# B2. 3 m behind -> out of the 87-deg forward FOV
obs_b, _ = place_and_observe(-3.0 * fwd_w)
det_b = obs_b[:, vc.ego_dim : vc.ego_dim + vc.detector_dim]
check("behind: NOT visible", bool((det_b[:, 0] < 0.5).all()))

# B3. beyond range (25 m ahead)
obs_c, _ = place_and_observe(25.0 * fwd_w)
det_c = obs_c[:, vc.ego_dim : vc.ego_dim + vc.detector_dim]
check("far (25m): NOT visible", bool((det_c[:, 0] < 0.5).all()))

# B4. occluded by a bar: put the target on the far side of each env's nearest bar (in front).
bars = task.obs_dict["obstacle_position"][:, :, 0:2]
active = bars[:, :, 0] > -100
dr = task.obs_dict["robot_position"]
d_bar = (bars - dr[:, 0:2].unsqueeze(1)).norm(dim=2)
d_bar = torch.where(active, d_bar, torch.full_like(d_bar, 1e6))
jn = d_bar.argmin(dim=1)
near_bar = bars[torch.arange(N, device=task.device), jn]
to_bar = near_bar - dr[:, 0:2]
to_bar_dist = to_bar.norm(dim=1, keepdim=True)
dir_bar = to_bar / to_bar_dist.clamp(min=1e-6)
off = torch.zeros(N, 3, device=task.device)
off[:, 0:2] = dir_bar * (to_bar_dist + 1.5)  # 1.5 m past the bar center, same direction+height
obs_d, _ = place_and_observe(off)
det_d = obs_d[:, vc.ego_dim : vc.ego_dim + vc.detector_dim]
n_occ = int((det_d[:, 0] < 0.5).sum())
print(f"[stage1] occlusion: {n_occ}/{N} envs report NOT-visible behind their nearest bar "
      f"(bar dist {to_bar_dist.squeeze(1).min():.1f}-{to_bar_dist.squeeze(1).max():.1f} m)")
check("occluded: mostly NOT visible", n_occ >= N - 1, f"({n_occ}/{N})")

# C. no GT leakage while blind: two DIFFERENT undetectable placements -> IDENTICAL actor obs
task.detector.reset_idx(torch.arange(N, device=task.device))  # clear tracker
obs_e1, _ = place_and_observe(-10.0 * fwd_w)
task.detector.reset_idx(torch.arange(N, device=task.device))
obs_e2, _ = place_and_observe(-14.0 * fwd_w)
leak = (obs_e1 - obs_e2).abs().max()
check("blind: actor obs independent of target position", float(leak) < 1e-6, f"(max diff {leak:.2e})")

# D. tracker memory: see it, then lose it -> last bearing persists, t_since grows
place_and_observe(1.5 * fwd_w)          # seen (tracker refreshed, t_since=0)
obs_f, _ = place_and_observe(-3.0 * fwd_w)   # lost (tracker holds)
det_f = obs_f[:, vc.ego_dim : vc.ego_dim + vc.detector_dim]
check("tracker: last bearing cos ~1 kept", bool((det_f[:, 6] > 0.9).all()),
      f"(cos={det_f[:, 6].min():.3f})")
check("tracker: t_since grew from 0", bool((det_f[:, 7] > 0.0).all())
      and bool((det_f[:, 7] < 0.2).all()), f"(t={det_f[:, 7].max():.3f})")

# E. body-frame action mapping: a=[1,0,0,0] -> command == [vmax, 0, 0] in the vehicle frame
a = torch.zeros(N, 4, device=task.device); a[:, 0] = 1.0
cmd = task.transform_action_to_command(a)
check("action: +x maps to vehicle +x at vmax",
      bool((cmd[:, 0] - tc.max_velocity).abs().max() < 1e-5) and bool(cmd[:, 1].abs().max() < 1e-5))

# F. states tail = GT extras (rpos unit toward target)
obs_g, st_g = place_and_observe(1.5 * fwd_w)
extras = st_g[:, 1265:]
check("states: rpos unit +x ~1 (target dead ahead)", bool((extras[:, 0] > 0.95).all()),
      f"(x={extras[:, 0].min():.3f})")
check("states: dist/24 ~ 1.5/24", bool(((extras[:, 3] - 0.0625).abs() < 0.01).all()))

print(f"[stage1] RESULT: {'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
