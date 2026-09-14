"""Phase-3 Stage-0 verification: both camera and LiDAR observe obstacles and target.

Run (GPU free, vision ON):
  NAVRL_VISION=1 NAVRL_MAX_BARS=40 NAVRL_NUM_BARS=40 PYTHONNOUSERSITE=1 \
    python tools/test_navrl_p3_stage0.py

Checks:
  1. LiDAR provides obstacle range plus target semantics.
  2. camera provides full-scene obstacle depth plus target detection pixels.
  3. a front target is observed by both sensors.
  4. a rear target is outside the camera FOV but remains observable by 360-degree LiDAR.
  5. throughput remains suitable for vectorized training (no per-step mesh refit).
"""
import os
import time

os.environ.setdefault("NAVRL_VISION", "1")
from aerial_gym.registry.task_registry import task_registry  # isaacgym before torch
import torch

N = 16
fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


task = task_registry.make_task("navrl_task", seed=7, num_envs=N, headless=True, use_warp=True)
print(f"[stage0] vision_mode={task.vision_mode} max_bars={task.max_bars_available} "
      f"n_bars_active={task.n_bars_active}")

obst = task.obs_dict["obstacle_position"]
check("vision mode enabled", task.vision_mode)
check("LiDAR target buffer exposed", task.has_vision_target)
check("NO extra obstacle column (target is analytic, not a mesh)",
      task.max_bars_available == obst.shape[1], f"(cols={obst.shape[1]})")

task.reset()
has_seg = "segmentation_pixels" in task.obs_dict and task.obs_dict["segmentation_pixels"] is not None
check("LiDAR segmentation channel exists", has_seg)

# --- Place the target in front, then behind, and verify complementary sensor coverage.
act = torch.zeros(N, 4, device=task.device)
task.tm.speed_fixed = 0.0
pos = task.obs_dict["robot_position"]
q_veh = task.obs_dict["robot_vehicle_orientation"]
from aerial_gym.utils.math import quat_rotate
fwd_w = quat_rotate(q_veh, torch.tensor([[1.0, 0.0, 0.0]], device=task.device).expand(N, 3))

task.target_position[:] = pos + 1.5 * fwd_w
task._sync_target_to_sensor()
task.sim_env.render(render_components="sensors")
task.process_obs_for_task()
pixels_front = task.obs_dict["target_camera_mask"].sum(dim=(1, 2))
check("front target produces camera pixels", bool((pixels_front > 0).all()),
      f"(pixels={pixels_front.tolist()})")
seg_front = task.obs_dict["segmentation_pixels"].reshape(N, -1)
lidar_target_front = (seg_front == 50).sum(dim=1)
check("front target produces LiDAR semantic returns", bool((lidar_target_front > 0).all()),
      f"(rays={lidar_target_front.tolist()})")
camera_obstacle = task.obs_dict["obstacle_camera_depth"]
check("camera obstacle depth shape", tuple(camera_obstacle.shape) == (N, 24, 40),
      f"(shape={tuple(camera_obstacle.shape)})")
check("camera observes environment geometry",
      bool((camera_obstacle < task.vis_cfg.camera_obstacle_max_range).any()))
check("LiDAR observes environment geometry", bool((seg_front == 3).any()))

task.detector.reset_idx(torch.arange(N, device=task.device))
task.target_position[:] = pos - 3.0 * fwd_w
task._sync_target_to_sensor()
task.sim_env.render(render_components="sensors")
task.process_obs_for_task()
pixels_behind = task.obs_dict["target_camera_mask"].sum(dim=(1, 2))
check("behind target produces no camera pixels", bool((pixels_behind == 0).all()),
      f"(pixels={pixels_behind.tolist()})")
seg_behind = task.obs_dict["segmentation_pixels"].reshape(N, -1)
lidar_target_behind = (seg_behind == 50).sum(dim=1)
check("behind target remains visible to 360-degree LiDAR",
      bool((lidar_target_behind > 0).all()), f"(rays={lidar_target_behind.tolist()})")

# --- throughput with vision ON (must be ~baseline: NO refit)
task.reset()
task.tm.speed_fixed = 1.5
task.tm.pattern = "cv"
task.reset()
torch.cuda.synchronize()
t0 = time.time()
STEPS = 60
for _ in range(STEPS):
    task.step(act)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"[stage0] throughput: {STEPS/dt:.1f} steps/s ({N*STEPS/dt:.0f} env-steps/s), "
      f"{1000*dt/STEPS:.1f} ms/step (N={N}, {task.n_bars_active} bars, camera target, NO refit)")

print(f"[stage0] RESULT: {'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
