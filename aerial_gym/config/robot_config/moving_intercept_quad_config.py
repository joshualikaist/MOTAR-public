import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


class MovingInterceptQuadCfg(BaseQuadCfg):
    """Chaser spawn near kinematic target (see intercept_quad_target_params ratios)."""

    class init_config(BaseQuadCfg.init_config):
        # In the ±3 m moving arena, target ratio (0.72, 0.48, 0.52) sits near (1.32, -0.12, 0.12) m.
        # Bootstrap spawn for early success: roughly 1.4–2.3 m initial range.
        min_init_state = [
            0.34,
            0.44,
            0.46,
            0,
            0,
            -np.pi / 6,
            1.0,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
            -0.1,
        ]
        max_init_state = [
            0.48,
            0.56,
            0.58,
            0,
            0,
            np.pi / 6,
            1.0,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]

    class robot_asset(BaseQuadCfg.robot_asset):
        name = "moving_intercept_quadrotor"
        # Same filter bit as white ground → no floor crashes; still collides with target (mask 0).
        collision_mask = 1
