"""Physics-substep controller for the NavRL moving target.

The target is one rigid body in PhysX. Four bounded first-order motors are reduced to their
equivalent body-frame wrench; this is dynamically identical to applying the four thrusts at the
declared arm locations for a rigid frame, while avoiding a second robot-manager instance.
"""

import math

import torch

from aerial_gym.utils.math import get_euler_xyz_tensor, quat_rotate_inverse, quat_to_rotation_matrix


class PhysicalTargetController:
    def __init__(self, tensors, target_index, cfg, device, contact_threshold=0.05):
        self.device = device
        self.index = int(target_index)
        self.dt = float(tensors["dt"])
        self.mass = float(cfg.physical_mass)
        self.max_motor_thrust = float(cfg.physical_max_motor_thrust)
        self.motor_tau = float(cfg.physical_motor_tau)
        self.max_tilt_rad = math.radians(float(cfg.physical_max_tilt_deg))
        self.velocity_kp = float(cfg.physical_velocity_kp)
        self.altitude_kp = float(cfg.physical_altitude_kp)
        self.attitude_kp = torch.tensor(
            cfg.physical_attitude_kp, dtype=torch.float32, device=device
        ).view(1, 3)
        self.rate_kp = torch.tensor(
            cfg.physical_rate_kp, dtype=torch.float32, device=device
        ).view(1, 3)
        self.yaw_torque_ratio = float(cfg.physical_yaw_torque_ratio)
        self.contact_threshold = float(contact_threshold)
        if self.contact_threshold < 0.0:
            raise ValueError("physical target contact threshold must be non-negative")
        arm = float(cfg.physical_motor_arm_xy)
        self.allocation = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-self.yaw_torque_ratio, self.yaw_torque_ratio,
                 -self.yaw_torque_ratio, self.yaw_torque_ratio],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.allocation_pinv = torch.linalg.pinv(self.allocation)

        self.position = tensors["obstacle_position"][:, self.index]
        self.orientation = tensors["obstacle_orientation"][:, self.index]
        self.linvel = tensors["obstacle_linvel"][:, self.index]
        self.angvel_world = tensors["obstacle_angvel"][:, self.index]
        self.force = tensors["obstacle_force_tensor"][:, self.index]
        self.torque = tensors["obstacle_torque_tensor"][:, self.index]
        self.contact_force = tensors["obstacle_contact_force_tensor"][:, self.index]
        self.gravity = tensors["gravity"]
        n = self.position.shape[0]
        self.velocity_command = torch.zeros((n, 3), device=device)
        self.altitude_command = torch.zeros(n, device=device)
        self.yaw_command = torch.zeros(n, device=device)
        self.motor_thrust = torch.zeros((n, 4), device=device)
        self.last_motor_command = torch.zeros_like(self.motor_thrust)
        self.last_velocity_error = torch.zeros((n, 3), device=device)
        self.last_tilt_rad = torch.zeros(n, device=device)
        self.last_saturated = torch.zeros(n, dtype=torch.bool, device=device)
        self.substeps = torch.zeros(n, dtype=torch.long, device=device)
        self.velocity_error_integral = torch.zeros(n, device=device)
        self.saturation_substeps = torch.zeros(n, dtype=torch.long, device=device)
        self.max_tilt_seen_rad = torch.zeros(n, device=device)
        self.contact_seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.watchdog_bars = None
        self.watchdog_half = None
        self.watchdog_lo = None
        self.watchdog_hi = None
        self.watchdog_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.watchdog_breach = torch.zeros(n, dtype=torch.bool, device=device)
        self.watchdog_geometry_invalid = torch.zeros(n, dtype=torch.bool, device=device)
        self.watchdog_prev_xy = self.position[:, :2].clone()

    def set_command(self, velocity_world, altitude, yaw=None):
        self.velocity_command[:] = velocity_world
        self.altitude_command[:] = altitude
        if yaw is None:
            moving = velocity_world[:, :2].norm(dim=1) > 1e-3
            heading = torch.atan2(velocity_world[:, 1], velocity_world[:, 0])
            self.yaw_command[:] = torch.where(moving, heading, self.yaw_command)
        else:
            self.yaw_command[:] = yaw

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        hover = self.mass * abs(float(self.gravity[0, 2])) / 4.0
        self.motor_thrust[env_ids] = hover
        self.last_motor_command[env_ids] = hover
        self.velocity_command[env_ids] = 0.0
        self.altitude_command[env_ids] = self.position[env_ids, 2]
        self.yaw_command[env_ids] = get_euler_xyz_tensor(self.orientation[env_ids])[:, 2]
        self.velocity_error_integral[env_ids] = 0.0
        self.saturation_substeps[env_ids] = 0
        self.max_tilt_seen_rad[env_ids] = 0.0
        self.substeps[env_ids] = 0
        self.contact_seen[env_ids] = False
        self.watchdog_breach[env_ids] = False
        self.watchdog_geometry_invalid[env_ids] = False
        self.watchdog_active[env_ids] = False
        self.watchdog_prev_xy[env_ids] = self.position[env_ids, :2]

    def set_hard_watchdog(self, bars_xy, hard_half_extents_xy, hard_lo, hard_hi, active=None):
        """Install a substep hard-envelope watchdog; it never writes a position."""
        if bars_xy.ndim != 3 or bars_xy.shape[2] != 2:
            raise ValueError("watchdog bars must have shape [N,B,2]")
        if hard_half_extents_xy.shape != bars_xy.shape:
            raise ValueError("watchdog half extents must match bars")
        if hard_lo.shape != (bars_xy.shape[0], 2) or hard_hi.shape != hard_lo.shape:
            raise ValueError("watchdog bounds must have shape [N,2]")
        self.watchdog_bars = bars_xy
        self.watchdog_half = hard_half_extents_xy
        self.watchdog_lo = hard_lo
        self.watchdog_hi = hard_hi
        geometry_valid = (
            torch.isfinite(bars_xy).all(dim=(1, 2))
            & torch.isfinite(hard_half_extents_xy).all(dim=(1, 2))
            & (hard_half_extents_xy >= 0.0).all(dim=(1, 2))
            & torch.isfinite(hard_lo).all(dim=1)
            & torch.isfinite(hard_hi).all(dim=1)
            & (hard_hi > hard_lo).all(dim=1)
        )
        self.watchdog_geometry_invalid[:] = ~geometry_valid
        requested = torch.ones_like(self.watchdog_active) if active is None else active
        if requested.shape != self.watchdog_active.shape:
            raise ValueError("watchdog active mask must have shape [N]")
        self.watchdog_active[:] = requested | ~geometry_valid
        point_inside_bounds = (
            torch.isfinite(self.position[:, :2]).all(dim=1)
            & (self.position[:, :2] > hard_lo).all(dim=1)
            & (self.position[:, :2] < hard_hi).all(dim=1)
        )
        if bars_xy.shape[1] > 0:
            point_delta = (self.position[:, None, :2] - bars_xy).abs() - hard_half_extents_xy
            point_outside_bars = ~(point_delta <= 0.0).all(dim=2).any(dim=1)
        else:
            point_outside_bars = torch.ones_like(point_inside_bounds)
        install_breach = self.watchdog_active & (
            self.watchdog_geometry_invalid | ~(point_inside_bounds & point_outside_bars)
        )
        self.watchdog_breach |= install_breach
        self.velocity_command[install_breach, :2] = 0.0
        # The first substep certificate starts at the actual pose at command installation; no
        # stale cross-interval segment is attributed to this interval.
        self.watchdog_prev_xy[:] = self.position[:, :2]

    def begin_control_interval(self):
        self.contact_seen.zero_()
        self.watchdog_breach.zero_()

    def post_physics_step(self):
        # Called after every PhysX refresh, not merely at the end of the 10-substep RL interval.
        self.contact_seen |= self.contact_force.norm(dim=1) > self.contact_threshold
        if self.watchdog_bars is not None:
            inside_bounds = ((self.position[:, :2] > self.watchdog_lo)
                             & (self.position[:, :2] < self.watchdog_hi)).all(dim=1)
            if self.watchdog_bars.shape[1] > 0:
                delta = (self.position[:, None, :2] - self.watchdog_bars).abs() - self.watchdog_half
                outside_bars = ~(delta <= 0.0).all(dim=2).any(dim=1)
                # Endpoint-only checks can miss a diagonal corner crossing.  Use a closed-AABB
                # slab certificate for the continuous previous-substep -> current segment.
                p0 = self.watchdog_prev_xy[:, None, :]
                p1 = self.position[:, None, :2]
                direction = p1 - p0
                box_lo = self.watchdog_bars - self.watchdog_half
                box_hi = self.watchdog_bars + self.watchdog_half
                parallel = direction.abs() <= 1e-9
                safe_parallel = (~parallel) | ((p0 >= box_lo) & (p0 <= box_hi))
                t0 = torch.where(
                    parallel,
                    torch.full_like(direction, float("-inf")),
                    (box_lo - p0) / direction,
                )
                t1 = torch.where(
                    parallel,
                    torch.full_like(direction, float("inf")),
                    (box_hi - p0) / direction,
                )
                t_enter = torch.maximum(torch.minimum(t0, t1)[:, :, 0], torch.minimum(t0, t1)[:, :, 1])
                t_exit = torch.minimum(torch.maximum(t0, t1)[:, :, 0], torch.maximum(t0, t1)[:, :, 1])
                segment_hits_bar = safe_parallel.all(dim=2) & (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)
                outside_segments = ~segment_hits_bar.any(dim=1)
            else:
                outside_bars = torch.ones_like(inside_bounds)
                outside_segments = outside_bars
            breach = self.watchdog_active & (
                self.watchdog_geometry_invalid | ~(inside_bounds & outside_bars & outside_segments)
            )
            self.watchdog_breach |= breach
            # A breach cannot be repaired by a position write.  Stop requesting planar motion;
            # the physical controller's own dynamics then decelerate the actor.
            self.velocity_command[breach, :2] = 0.0
            self.watchdog_prev_xy[:] = self.position[:, :2]

    def __call__(self):
        # Translational velocity controller with explicit altitude hold. Commands are in world
        # frame because the trajectory planner is world-referenced.
        vel_error = self.velocity_command - self.linvel
        accel = self.velocity_kp * vel_error
        accel[:, 2] += self.altitude_kp * (self.altitude_command - self.position[:, 2])
        force_world = self.mass * (accel - self.gravity)

        # Altitude-priority tilt bound: retain requested vertical force and scale lateral force.
        fz = force_world[:, 2].clamp(min=1e-3)
        fxy = force_world[:, :2]
        fxy_norm = fxy.norm(dim=1)
        fxy_max = fz * math.tan(self.max_tilt_rad)
        scale = torch.minimum(torch.ones_like(fxy_norm), fxy_max / fxy_norm.clamp(min=1e-6))
        force_world = torch.cat((fxy * scale.unsqueeze(1), fz.unsqueeze(1)), dim=1)

        rotation = quat_to_rotation_matrix(self.orientation)
        body_z = rotation[:, :, 2]
        desired_z = force_world / force_world.norm(dim=1, keepdim=True).clamp(min=1e-6)
        heading = torch.stack(
            (torch.cos(self.yaw_command), torch.sin(self.yaw_command), torch.zeros_like(fz)), dim=1
        )
        desired_y = torch.linalg.cross(desired_z, heading, dim=1)
        desired_y = desired_y / desired_y.norm(dim=1, keepdim=True).clamp(min=1e-6)
        desired_x = torch.linalg.cross(desired_y, desired_z, dim=1)
        desired_rotation = torch.stack((desired_x, desired_y, desired_z), dim=2)

        rdtr = torch.bmm(desired_rotation.transpose(1, 2), rotation)
        rtrd = torch.bmm(rotation.transpose(1, 2), desired_rotation)
        skew = rdtr - rtrd
        attitude_error = 0.5 * torch.stack(
            (skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=1
        )
        omega_body = quat_rotate_inverse(self.orientation, self.angvel_world)
        torque_cmd = -self.attitude_kp * attitude_error - self.rate_kp * omega_body
        thrust_cmd = (force_world * body_z).sum(dim=1).clamp(min=0.0)
        wrench_cmd = torch.cat((thrust_cmd.unsqueeze(1), torque_cmd), dim=1)

        motor_cmd_raw = torch.matmul(wrench_cmd, self.allocation_pinv.T)
        motor_cmd = motor_cmd_raw.clamp(0.0, self.max_motor_thrust)
        alpha = 1.0 - math.exp(-self.dt / max(self.motor_tau, 1e-6))
        self.motor_thrust.add_(alpha * (motor_cmd - self.motor_thrust))
        realized = torch.matmul(self.motor_thrust, self.allocation.T)
        self.force.zero_()
        self.torque.zero_()
        self.force[:, 2] = realized[:, 0]
        self.torque[:] = realized[:, 1:4]

        self.last_motor_command[:] = motor_cmd
        self.last_velocity_error[:] = vel_error
        self.last_saturated[:] = (motor_cmd_raw < 0.0).any(dim=1) | (
            motor_cmd_raw > self.max_motor_thrust
        ).any(dim=1)
        self.last_tilt_rad[:] = torch.acos(body_z[:, 2].clamp(-1.0, 1.0))
        self.substeps += 1
        self.velocity_error_integral += vel_error.norm(dim=1) * self.dt
        self.saturation_substeps += self.last_saturated.long()
        self.max_tilt_seen_rad[:] = torch.maximum(self.max_tilt_seen_rad, self.last_tilt_rad)

    def diagnostics(self):
        denom = self.substeps.clamp(min=1).float()
        return {
            "velocity_error_mean_mps": self.velocity_error_integral / (denom * self.dt),
            "max_tilt_deg": torch.rad2deg(self.max_tilt_seen_rad),
            "motor_saturation_fraction": self.saturation_substeps.float() / denom,
            "hard_watchdog_breach": self.watchdog_breach,
            "hard_watchdog_geometry_invalid": self.watchdog_geometry_invalid,
            "motor_thrust": self.motor_thrust,
        }
