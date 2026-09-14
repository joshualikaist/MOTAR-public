from typing import Tuple

import numpy as np
import torch
from gym.spaces import Box, Dict

from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("shooting_moving_target_task")


class ShootingMovingTargetTask(PositionSetpointTask):
    """Intercept a target quad that moves at constant horizontal velocity."""

    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        super().__init__(
            task_config,
            seed=seed,
            num_envs=num_envs,
            headless=headless,
            device=device,
            use_warp=use_warp,
        )
        obs_dim = self.task_config.observation_space_dim
        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)}
        )
        self.task_obs["observations"] = torch.zeros(
            (self.num_envs, obs_dim), device=self.device, requires_grad=False
        )
        self.target_velocity = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._prev_body_dist = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self._milestone_hit = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.bool)
        self._finish_core_hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        vscale = float(getattr(self.task_config, "command_velocity_scale", 1.0))
        logger.info(
            f"Moving intercept target: speed {self.task_config.target_speed_min:.2f}"
            f"–{self.task_config.target_speed_max:.2f} m/s, obs dim {obs_dim},"
            f" chaser cmd scale {vscale:.2f} m/s per action unit"
        )

    def _rp_float(self, key: str, default: float = 0.0) -> float:
        val = self.task_config.reward_parameters.get(key, default)
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device)
        env_ids = env_ids.reshape(-1)
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero(as_tuple=False).reshape(-1)
        return env_ids.to(dtype=torch.long)

    def _sync_prev_body_dist(self, env_ids=None):
        dist = self._body_center_distance_to_target()
        if env_ids is None:
            self._prev_body_dist[:] = dist
            return
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() > 0:
            self._prev_body_dist[env_ids] = dist[env_ids]

    def _scale_velocity_commands(self, actions: torch.Tensor) -> torch.Tensor:
        vscale = float(getattr(self.task_config, "command_velocity_scale", 1.0))
        wscale = float(getattr(self.task_config, "command_yaw_rate_scale", vscale))
        if vscale == 1.0 and wscale == 1.0:
            return actions
        scaled = actions.clone()
        scaled[:, 0:3] = actions[:, 0:3] * vscale
        scaled[:, 3] = actions[:, 3] * wscale
        return scaled

    def _curriculum_alpha(self) -> float:
        if not bool(getattr(self.task_config, "target_speed_curriculum_enable", False)):
            return 1.0
        steps_per_epoch = float(getattr(self.task_config, "curriculum_steps_per_epoch", 64))
        epoch = float(self.counter) / max(steps_per_epoch, 1.0)
        start = float(getattr(self.task_config, "target_speed_curriculum_start_epoch", 0.0))
        end = float(getattr(self.task_config, "target_speed_curriculum_end_epoch", start + 1.0))
        return float(np.clip((epoch - start) / max(end - start, 1.0), 0.0, 1.0))

    def _target_speed_bounds(self) -> Tuple[float, float]:
        alpha = self._curriculum_alpha()
        lo0 = float(getattr(self.task_config, "target_speed_start_min", self.task_config.target_speed_min))
        hi0 = float(getattr(self.task_config, "target_speed_start_max", self.task_config.target_speed_max))
        lo1 = float(self.task_config.target_speed_min)
        hi1 = float(self.task_config.target_speed_max)
        return lo0 + alpha * (lo1 - lo0), hi0 + alpha * (hi1 - hi0)

    def _sample_target_velocity(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        speed_lo, speed_hi = self._target_speed_bounds()
        speed = torch.empty(n, device=self.device).uniform_(speed_lo, speed_hi)
        angle = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * np.pi)
        self.target_velocity[env_ids, 0] = speed * torch.cos(angle)
        self.target_velocity[env_ids, 1] = speed * torch.sin(angle)
        self.target_velocity[env_ids, 2] = 0.0

    def _advance_moving_target(self):
        dt = float(self.obs_dict["dt"])
        asset_state = self.obs_dict["env_asset_state_tensor"]
        asset_state[:, 0, 0:3] += self.target_velocity * dt
        if bool(getattr(self.task_config, "reflect_target_at_env_bounds", False)):
            self._reflect_target_xy_at_env_bounds(asset_state)
        asset_state[:, 0, 7:10] = self.target_velocity

    def _reflect_target_xy_at_env_bounds(self, asset_state: torch.Tensor) -> None:
        bounds_min = self.obs_dict.get("env_bounds_min", None)
        bounds_max = self.obs_dict.get("env_bounds_max", None)
        if bounds_min is None or bounds_max is None:
            return
        pos = asset_state[:, 0, 0:3]
        for axis in (0, 1):
            lo = bounds_min[:, axis]
            hi = bounds_max[:, axis]
            p = pos[:, axis]
            below = p < lo
            above = p > hi
            hit = below | above
            if not bool(hit.any().item()):
                continue
            reflected = torch.where(below, lo + (lo - p), p)
            reflected = torch.where(above, hi - (p - hi), reflected)
            pos[:, axis] = reflected.clamp(min=lo, max=hi)
            self.target_velocity[:, axis] = torch.where(
                hit,
                -self.target_velocity[:, axis],
                self.target_velocity[:, axis],
            )

    def reset(self):
        self._milestone_hit.zero_()
        self._finish_core_hit.zero_()
        self._sample_target_velocity(torch.arange(self.num_envs, device=self.device))
        ret = super().reset()
        self._sync_prev_body_dist()
        return ret

    def reset_idx(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        self._sample_target_velocity(env_ids)
        if env_ids.numel() > 0:
            self._milestone_hit[env_ids] = False
            self._finish_core_hit[env_ids] = False
        super().reset_idx(env_ids)
        self._sync_prev_body_dist(env_ids)

    def step(self, actions):
        self._advance_moving_target()
        ret = super().step(self._scale_velocity_commands(actions))
        just_reset = self.sim_env.sim_steps == 0
        if bool(just_reset.any().item()):
            ids = just_reset.nonzero(as_tuple=False).reshape(-1)
            self._sample_target_velocity(ids)
            self._sync_prev_body_dist(ids)
        return ret

    def process_obs_for_task(self):
        super().process_obs_for_task()
        rel_vel = self.target_velocity - self.obs_dict["robot_linvel"]
        self.task_obs["observations"][:, 13:16] = rel_vel

    def _distance_focus_reward(self, dist: torch.Tensor) -> torch.Tensor:
        far_ref = self._rp_float("dist_focus_far_ref_m")
        far_slope = self._rp_float("dist_focus_far_slope")
        switch = self._rp_float("dist_focus_switch_m")
        near_amp = self._rp_float("dist_focus_near_amp")
        near_exp = self._rp_float("dist_focus_near_exp")
        plateau = self._rp_float("dist_focus_plateau_m")
        if far_ref <= 0.0 or far_slope <= 0.0:
            return torch.zeros_like(dist)

        shaped_dist = dist.clamp(min=plateau) if plateau > 0.0 else dist
        reward = far_slope * (far_ref - shaped_dist).clamp(min=0.0)
        if switch <= 0.0 or near_amp <= 0.0 or near_exp <= 0.0:
            return reward

        x = ((switch - shaped_dist).clamp(min=0.0, max=switch) / switch).clamp(min=0.0, max=1.0)
        denom = float(np.expm1(near_exp))
        if denom <= 1.0e-6:
            return reward
        return reward + near_amp * torch.expm1(x * near_exp) / denom

    def _finish_funnel_reward(self, dist: torch.Tensor, range_rate: torch.Tensor) -> torch.Tensor:
        surface_center = self._surface_contact_center_dist()
        gap = dist - surface_center
        outer = self._rp_float(
            "finish_funnel_outer_gap_m",
            max(self._rp_float("finish_funnel_outer_m", 0.18) - surface_center, 0.0),
        )
        inner = self._rp_float(
            "finish_funnel_inner_gap_m",
            max(
                self._rp_float(
                    "finish_funnel_inner_m",
                    float(getattr(self.task_config, "intercept_success_radius", 0.12)),
                )
                - surface_center,
                0.0,
            ),
        )
        coef = self._rp_float("finish_funnel_closing_coef")
        ceil = self._rp_float("finish_funnel_step_ceil")
        if outer <= 0.0 or coef <= 0.0 or inner < 0.0 or inner >= outer:
            return torch.zeros_like(dist)

        width = max(outer - inner, 1.0e-4)
        zone = ((outer - gap).clamp(min=0.0, max=width) / width).clamp(min=0.0, max=1.0)
        bonus = coef * zone * range_rate.clamp(min=0.0)
        if ceil > 0.0:
            bonus = bonus.clamp(max=ceil)
        return bonus

    def compute_rewards_and_crashes(self, obs_dict):
        rewards, terminations = super().compute_rewards_and_crashes(obs_dict)

        robot_position = obs_dict["robot_position"]
        target_position = self._world_target_positions()
        to_tgt = target_position - robot_position
        dist = torch.norm(to_tgt, dim=1).clamp(min=1e-6)
        dir_t = to_tgt / dist.unsqueeze(1)
        # Signed range rate (m/s): positive only when center distance shrinks.
        range_rate = ((obs_dict["robot_linvel"] - self.target_velocity) * dir_t).sum(dim=1)

        rewards[:] = rewards + self._distance_focus_reward(dist)

        rr_coef = self._rp_float("range_rate_bonus_coef")
        if rr_coef > 0.0:
            rewards[:] = rewards + rr_coef * range_rate.clamp(min=0.0)

        prog_coef = self._rp_float("dist_progress_bonus_coef")
        if prog_coef > 0.0:
            progress_m = (self._prev_body_dist - dist).clamp(min=0.0)
            rewards[:] = rewards + prog_coef * progress_m

        stall_coef = self._rp_float("range_stall_penalty_coef")
        stall_exempt = self._rp_float("range_stall_exempt_dist_m")
        if stall_coef > 0.0:
            stalled = range_rate <= 0.0
            if stall_exempt > 0.0:
                stalled = stalled & (dist > stall_exempt)
            rewards[:] = rewards - stall_coef * dist * stalled.to(dtype=torch.float32)

        open_coef = self._rp_float("range_open_penalty_coef")
        if open_coef > 0.0:
            opening = (-range_rate).clamp(min=0.0)
            rewards[:] = rewards - open_coef * opening

        self._prev_body_dist[:] = dist.detach()

        rewards[:] = self._clamp_dense_rewards(rewards)
        rewards[:] = rewards + self._finish_funnel_reward(dist, range_rate)

        thresholds = (
            self._rp_float("milestone_dist_1_m"),
            self._rp_float("milestone_dist_2_m"),
            self._rp_float("milestone_dist_3_m"),
        )
        bonuses = (
            self._rp_float("milestone_bonus_1"),
            self._rp_float("milestone_bonus_2"),
            self._rp_float("milestone_bonus_3"),
        )
        if thresholds[0] > 0.0 and bonuses[0] > 0.0:
            for i, (thr, bonus) in enumerate(zip(thresholds, bonuses)):
                if thr <= 0.0 or bonus <= 0.0:
                    continue
                crossed = (dist <= thr) & ~self._milestone_hit[:, i]
                rewards[:] = rewards + bonus * crossed.to(dtype=torch.float32)
                self._milestone_hit[crossed, i] = True

        surface_center = self._surface_contact_center_dist()
        core_gap = self._rp_float(
            "finish_core_gap_m",
            max(
                self._rp_float(
                    "finish_core_dist_m",
                    float(getattr(self.task_config, "intercept_success_radius", 0.12)),
                )
                - surface_center,
                0.0,
            ),
        )
        core_bonus = self._rp_float("finish_core_bonus")
        if core_gap >= 0.0 and core_bonus > 0.0:
            crossed_core = ((dist - surface_center) <= core_gap) & ~self._finish_core_hit
            rewards[:] = rewards + core_bonus * crossed_core.to(dtype=torch.float32)
            self._finish_core_hit[crossed_core] = True

        return rewards, terminations

    def _clamp_dense_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Limit per-step dense magnitude; leave intercept terminal steps (≥ threshold) untouched."""
        floor = self._rp_float("reward_dense_step_floor")
        ceil = self._rp_float("reward_dense_step_ceil")
        thresh = self._rp_float("reward_dense_terminal_threshold", 400.0)
        if floor >= 0.0 and ceil <= 0.0:
            return rewards
        rw = rewards.reshape(-1).clone()
        rw = torch.nan_to_num(
            rw,
            nan=0.0,
            posinf=ceil if ceil > 0.0 else 1.0e3,
            neginf=floor if floor < 0.0 else -1.0e3,
        )
        is_terminal = rw >= thresh
        lo = floor if floor < 0.0 else -1.0e9
        hi = ceil if ceil > 0.0 else 1.0e9
        rw = torch.where(is_terminal, rw, rw.clamp(min=lo, max=hi))
        return rw.reshape_as(rewards)
