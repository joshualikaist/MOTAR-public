from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class MovingInterceptSimConfig(BaseSimConfig):
    """Brighter viewer lighting for moving-intercept demos (ground plane in env cfg)."""

    class scene_lighting(BaseSimConfig.scene_lighting):
        enabled = True
        directional_intensity = [1.0, 1.0, 1.0]
        ambient = [0.95, 0.95, 0.98]
