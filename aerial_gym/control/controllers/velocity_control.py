import math

import torch
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("velocity_controller")


class LeeVelocityController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        self.accel[:] = self.compute_acceleration(
            setpoint_position=self.robot_position,
            setpoint_velocity=command_actions[:, 0:3],
        )
        forces = (self.accel[:] - self.gravity) * self.mass

        max_tilt_deg = float(getattr(self.cfg, "max_tilt_angle_deg", 0.0) or 0.0)
        if max_tilt_deg > 0.0:
            # Tilt limit (PX4 MPC_TILTMAX_AIR / ArduPilot ANGLE_MAX equivalent). The desired body-z
            # axis downstream is just forces/|forces|, so WITHOUT this a large horizontal command
            # with little vertical component asks the drone to pitch past 90 deg -- i.e. to flip. A
            # trained policy rarely commands that, but a from-scratch one does constantly, and once
            # inverted it accelerates itself into the floor and never recovers (measured: floor
            # strikes are 73-87% of all crashes in the first epochs of a fresh run).
            # Altitude-priority form: keep the vertical component and scale the horizontal one down,
            # so limiting the tilt costs lateral acceleration rather than lift. At 45 deg the
            # available lateral accel is still g*tan(45) = 9.8 m/s^2 (turn radius 0.64 m at 2.5 m/s),
            # far more than the weave needs, so this bounds attitude without limiting agility.
            f_z = forces[:, 2].clamp(min=1e-6)  # never command net-downward thrust
            f_xy = forces[:, 0:2]
            f_xy_norm = f_xy.norm(dim=1)
            f_xy_max = f_z * math.tan(math.radians(max_tilt_deg))
            scale = torch.where(
                f_xy_norm > f_xy_max, f_xy_max / f_xy_norm.clamp(min=1e-6), torch.ones_like(f_xy_norm)
            )
            forces = torch.cat([f_xy * scale.unsqueeze(1), f_z.unsqueeze(1)], dim=1)

        # thrust command is transformed by the body orientation's z component
        b3 = quat_to_rotation_matrix(self.robot_orientation)[:, :, 2]
        if getattr(self.cfg, "tilt_thrust_compensation", False):
            # Altitude-priority thrust (PX4-style tilt compensation). The standard Lee projection
            # T = f·b3 delivers a vertical force of (f·b3)*b3_z, which equals the commanded f_z ONLY
            # when the desired force direction and the CURRENT body axis agree. During attitude-lag
            # transients (e.g. a weave reversal: body still tilted one way, f already pointing the
            # other) the achieved vertical force sags below f_z deterministically -- both vectors are
            # known at this line, so no prediction is involved. Choosing T = f_z / b3_z makes the
            # achieved vertical force EXACTLY f_z regardless of the current tilt, decoupling altitude
            # from lateral/yaw transients. Cost: during the mismatch some thrust leaks laterally
            # along the stale body axis (slightly slower reversals) -- acceptable, floor strikes are
            # terminal.
            # Only valid while the drone is actually upright. A bare clamp(min=0.5) would bound the
            # MAGNITUDE but not the SIGN: inverted (b3_z < 0) it returns +0.5 and commands positive
            # thrust along a downward-pointing body axis, driving the drone into the ground and
            # never recovering. The tilt limit above stops us COMMANDING that attitude, but the
            # actual attitude can still overshoot it (collisions, disturbances), so fall back to the
            # standard Lee projection whenever the compensation would leave its valid domain.
            b3z = b3[:, 2]
            upright = b3z > 0.5  # 60 deg from vertical
            self.wrench_command[:, 2] = torch.where(
                upright, forces[:, 2] / b3z.clamp(min=0.5), torch.sum(forces * b3, dim=1)
            )
        else:
            self.wrench_command[:, 2] = torch.sum(forces * b3, dim=1)

        # after calculating forces, we calculate the desired euler angles
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, self.robot_euler_angles[:, 2], self.buffer_tensor
        )

        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )

        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command
