from abc import ABC, abstractmethod


class BaseSensorConfig(ABC):
    num_sensors = 1
    randomize_placement = False
    min_translation = [0.07, -0.06, 0.01]
    max_translation = [0.12, 0.03, 0.04]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]

    # If True, the sensor tracks only the robot's yaw (heading); body roll and pitch are
    # ignored so the sensor frame stays level. Mirrors the NavRL / IsaacLab RayCaster
    # `attach_yaw_only` option. Default False keeps the sensor rigidly bolted to the body.
    yaw_only_attach = False
