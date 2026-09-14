import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.config.sensor_config.lidar_config.navrl_lidar_config import (
    NavRLLidarConfig,
)


class NavRLQuadCfg(BaseQuadCfg):
    """Base quadrotor equipped with the NavRL-style yaw-only ray-cast LiDAR (no camera).

    Static obstacles are perceived through a 36x4 range scan (see NavRLLidarConfig),
    matching the NavRL navigation environment. Requires use_warp=True; LiDAR rendering
    is not supported by Isaac Gym's native camera path.

    For the Phase-1 bars environment the drone flies in 2D at a fixed altitude: it spawns at
    1 m (z-ratio 1/3 of the [0, 3] m arena) with no initial vertical velocity, and navrl_task
    zeroes the vertical velocity command so it stays at 1 m and tracks the goal in XY only.
    """

    class robot_asset(BaseQuadCfg.robot_asset):
        # Use a navrl-only URDF whose base_link collision is a 0.28 x 0.28 x 0.08 m box bounding the
        # 4 motors (arms/props), instead of the shared quad.urdf's 0.05 m centre sphere. So the drone
        # collides at its full ~0.28 m footprint (can't shave bars with its arms). ~10x the collision
        # cross-section -> more crashes, but a realistic footprint. Scoped here so it does NOT change
        # the intercept / position_setpoint tasks, which rely on the 0.05 m sphere.
        file = "quad_navrl_collide.urdf"

    class init_config(BaseQuadCfg.init_config):
        # [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]
        # z-ratio fixed at 1/3 -> spawn altitude = 1.0 m; initial vz = 0 (index 9).
        # Drone spawns on the LEFT edge (x-ratio ~0 -> x in [0, ~1]) across nearly the full y span,
        # so the goal (placed at x=k on the far side) forces a left->right crossing of the bar field.
        # Yaw ~0 (+/- 30 deg) already faces +x, i.e. toward the goal.
        min_init_state = [
            0.0, 0.02, 0.3333, 0.0, 0.0, -np.pi / 6, 1.0, -0.2, -0.2, 0.0, -0.2, -0.2, -0.2,
        ]
        max_init_state = [
            0.04, 0.98, 0.3333, 0.0, 0.0, np.pi / 6, 1.0, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2,
        ]

    class sensor_config(BaseQuadCfg.sensor_config):
        enable_camera = False  # NavRL static perception uses LiDAR, not a depth camera

        enable_lidar = True
        lidar_config = NavRLLidarConfig

        enable_imu = False
