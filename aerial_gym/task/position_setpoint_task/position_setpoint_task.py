from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import os
import time

import gymnasium as gym
from gym.spaces import Dict, Box
from typing import Tuple

from aerial_gym.task.position_setpoint_task.play_dashboard import (
    play_compact_dashboard_enabled,
    record_episode_outcomes_this_step,
)
from aerial_gym.task.position_setpoint_task.train_dashboard import (
    record_train_epoch_episodes,
    record_train_step_mean_dist,
    train_intercept_metrics_enabled,
)

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs
        self._last_step_wall_s = 0.0
        self._play_compact_eval = play_compact_dashboard_enabled()
        self._ep_return = torch.zeros(self.num_envs, device=self.device)
        self._ep_len = torch.zeros(self.num_envs, device=self.device)
        self._ep_min_body_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        # EnvManager increments obs_dict["crashes"] during physics (contact force); JIT reward overwrites it.
        # We snapshot before reward to sum per-env per episode: count of RL steps with ≥1 qualifying contact sub-step.
        self._ep_phys_contact_substeps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._ep_radius_no_contact_substeps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._ep_contact_no_radius_substeps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def _world_target_positions(self):
        """
        Prefer the first spawned env asset (intercept target) pose from Isaac Gym tensors.
        Falls back to self.target_position for plain empty variants without obstacles.
        """
        op = self.obs_dict.get("obstacle_position", None)
        if op is not None and op.shape[1] > 0:
            return op[:, 0, :]
        return self.target_position

    def _body_center_distance_to_target(self):
        """World-frame ‖target − robot root (base_link center)‖ — not prop/arm edge."""
        tgt = self._world_target_positions()
        rob = self.obs_dict["robot_position"]
        return torch.norm(tgt - rob, dim=1)

    def _effective_intercept_radius(self) -> float:
        radius = float(getattr(self.task_config, "intercept_success_radius", 0.20))
        if "INTERCEPT_SUCCESS_RADIUS" in os.environ:
            try:
                radius = float(os.environ["INTERCEPT_SUCCESS_RADIUS"])
            except ValueError:
                pass
        return radius

    def _body_center_intercept_mask(self, body_dist: torch.Tensor) -> torch.Tensor:
        """True when base_link center is within intercept_success_radius of target center."""
        return body_dist <= self._effective_intercept_radius()

    def _effective_intercept_min_force(self) -> float:
        force = float(getattr(self.task_config, "intercept_min_contact_force", 0.01))
        rp = getattr(self.task_config, "reward_parameters", None)
        if isinstance(rp, dict) and "intercept_min_contact_force" in rp:
            force = float(rp["intercept_min_contact_force"])
        if "INTERCEPT_MIN_CONTACT_FORCE" in os.environ:
            try:
                force = float(os.environ["INTERCEPT_MIN_CONTACT_FORCE"])
            except ValueError:
                pass
        return force

    def _qualifying_intercept_hit_mask(
        self, body_dist: torch.Tensor, physics_contact: torch.Tensor, contact_force_norm: torch.Tensor
    ) -> torch.Tensor:
        """Intercept success: contact + center dist ≤ radius + |F| ≥ min (same as ep-reset threshold)."""
        return (
            physics_contact
            & self._body_center_intercept_mask(body_dist)
            & (contact_force_norm >= self._effective_intercept_min_force())
        )

    def _reward_param_float(self, key: str, default: float) -> float:
        rp = getattr(self.task_config, "reward_parameters", None)
        if isinstance(rp, dict) and key in rp:
            val = rp[key]
            if isinstance(val, torch.Tensor):
                return float(val.item())
            return float(val)
        return float(default)

    def _surface_contact_center_dist(self) -> float:
        """Approximate center distance where the two rendered collision spheres touch."""
        return max(0.0, self._reward_param_float("surface_contact_center_dist_m", 0.10))

    def _near_miss_dist(self) -> float:
        radius = self._effective_intercept_radius()
        return max(radius, self._reward_param_float("near_miss_dist_m", max(radius, 0.18)))

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self._ep_return.zero_()
        self._ep_len.zero_()
        self._ep_min_body_dist.fill_(float("inf"))
        self._ep_phys_contact_substeps.zero_()
        self._ep_radius_no_contact_substeps.zero_()
        self._ep_contact_no_radius_substeps.zero_()
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = (
            0.0  # (torch.rand_like(self.target_position[env_ids]) * 10.0)
        )
        self.infos = {}
        self._ep_return[env_ids] = 0
        self._ep_len[env_ids] = 0
        self._ep_min_body_dist[env_ids] = float("inf")
        self._ep_phys_contact_substeps[env_ids] = 0
        self._ep_radius_no_contact_substeps[env_ids] = 0
        self._ep_contact_no_radius_substeps[env_ids] = 0
        self.sim_env.reset_idx(env_ids)
        return

    def render(self, mode="human"):
        """Gym-compatible render for rl-games play; refreshes Isaac viewer when requested."""
        if mode == "rgb_array":
            return None
        if mode == "human":
            self.sim_env.render(render_components="viewer")
            return None
        self.sim_env.render()
        return None

    def _finalize_episode_step_stats(self):
        """Keep per-env episode sums; reset finished envs. Optional console box."""
        term = (self.terminations > 0).reshape(-1)
        trunc = (self.truncations > 0).reshape(-1)
        done = term | trunc
        if not bool(done.any().item()):
            return

        radius = self._effective_intercept_radius()
        tr = trunc.reshape(-1).to(dtype=torch.bool, device=self.device)
        te = term.reshape(-1).to(dtype=torch.bool, device=self.device)
        dn = done.reshape(-1)
        n_done = int(dn.sum().item())
        # Success = contact with target while center dist ≤ radius AND |F| ≥ intercept_min_contact_force.
        had_center_contact = self._ep_phys_contact_substeps.reshape(-1) > 0
        succ = dn & had_center_contact
        n_succ = int(succ.sum().item())
        sum_r = self._ep_return[dn].sum().item()
        avg_r = float(sum_r) / max(n_done, 1)

        flag = os.environ.get("AERIAL_LOG_INTERCEPT", "").strip().lower()
        log_on = flag not in ("0", "false", "no", "off", "")

        if self._play_compact_eval:
            record_episode_outcomes_this_step(
                n_done,
                n_succ,
                ep_return_sum=sum_r,
                step_wall_s=self._last_step_wall_s,
            )
        elif train_intercept_metrics_enabled():
            idx_fin = dn.nonzero(as_tuple=False).reshape(-1)
            if idx_fin.numel() > 0:
                succ_on_fin = succ[idx_fin]
                succ_env_ids = idx_fin[succ_on_fin].reshape(-1).tolist()
            else:
                succ_on_fin = torch.zeros(0, dtype=torch.bool, device=self.device)
                succ_env_ids = []
            closest_vals = self._ep_min_body_dist[idx_fin]
            finite = torch.isfinite(closest_vals)
            closest_vals = closest_vals[finite]
            succ_on_fin = succ_on_fin[finite]
            failure_mask = ~succ_on_fin
            surface_center = self._surface_contact_center_dist()
            surface_gap_vals = closest_vals - surface_center
            success_closest_vals = closest_vals[succ_on_fin]
            failure_closest_vals = closest_vals[failure_mask]
            failure_surface_gap_vals = failure_closest_vals - surface_center
            radius_no_contact_ep = (self._ep_radius_no_contact_substeps[idx_fin] > 0)[finite] & failure_mask
            contact_no_radius_ep = (self._ep_contact_no_radius_substeps[idx_fin] > 0)[finite] & failure_mask
            near_miss_ep = failure_mask & (closest_vals <= self._near_miss_dist())
            record_train_epoch_episodes(
                n_done,
                n_succ,
                succ_env_ids,
                closest_dist_sum=float(closest_vals.sum().item()) if closest_vals.numel() > 0 else None,
                closest_dist_count=int(closest_vals.numel()),
                success_closest_dist_sum=(
                    float(success_closest_vals.sum().item()) if success_closest_vals.numel() > 0 else None
                ),
                success_closest_dist_count=int(success_closest_vals.numel()),
                failure_closest_dist_sum=(
                    float(failure_closest_vals.sum().item()) if failure_closest_vals.numel() > 0 else None
                ),
                failure_closest_dist_count=int(failure_closest_vals.numel()),
                surface_gap_sum=float(surface_gap_vals.sum().item()) if surface_gap_vals.numel() > 0 else None,
                surface_gap_count=int(surface_gap_vals.numel()),
                failure_surface_gap_sum=(
                    float(failure_surface_gap_vals.sum().item()) if failure_surface_gap_vals.numel() > 0 else None
                ),
                failure_surface_gap_count=int(failure_surface_gap_vals.numel()),
                near_miss_count=int(near_miss_ep.sum().item()),
                radius_no_contact_count=int(radius_no_contact_ep.sum().item()),
                contact_no_radius_count=int(contact_no_radius_ep.sum().item()),
            )
        elif log_on:
            n_trunc_c = int((dn & tr).sum().item())
            n_term_c = int((dn & (~tr) & te).sum().item())
            n_ambig = max(0, n_done - n_trunc_c - n_term_c)
            sum_steps = self._ep_len[dn].sum().item()
            avg_steps = float(sum_steps) / max(n_done, 1)
            idx_fin = dn.nonzero(as_tuple=False).squeeze(-1)
            coll_vals = self._ep_phys_contact_substeps[idx_fin]
            total_coll = int(coll_vals.sum().item())
            mean_coll = float(coll_vals.float().mean().item())
            max_coll = int(coll_vals.max().item())
            coll_pairs = ", ".join(
                f"[{int(i)}]={int(c)}" for i, c in zip(idx_fin.tolist(), coll_vals.tolist())
            )
            bar = "-" * 25
            ambig_note = f" · other/overlap: {n_ambig}" if n_ambig > 0 else ""
            print(
                f"{bar}\n"
                f"This sim step · episodes finished : {n_done} env(s)\n"
                f" Parallel env batch (play)          : {self.num_envs}\n"
                f"└ intercept (dist≤{radius:g} m, |F|≥{self._effective_intercept_min_force():g} N) : {n_succ} / {n_done} "
                f"(only among those finished on this tick)\n"
                f"  breakdown · time-limit (trunc)   : {n_trunc_c}"
                f" · other termination               : {n_term_c}"
                f"{ambig_note}\n"
                f"Episode contact-heavy substeps     : Σ={total_coll}  mean/finished={mean_coll:.2f}"
                f"  max(one env)={max_coll}\n"
                f"  (per env, sum over RL steps)|F|>thresh : {coll_pairs}\n"
                f"Avg episodic reward (finished now) : {avg_r:.2f}\n"
                f"Avg episode steps (finished now)   : {avg_steps:.1f}\n"
                f"{bar}",
                flush=True,
            )

        self._ep_return[done] = 0
        self._ep_len[done] = 0
        self._ep_min_body_dist[done] = float("inf")
        self._ep_phys_contact_substeps[done] = 0
        self._ep_radius_no_contact_substeps[done] = 0
        self._ep_contact_no_radius_substeps[done] = 0

    def step(self, actions):
        t_wall0 = time.perf_counter() if self._play_compact_eval else None
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.sim_env.step(actions=self.actions)

        _ct = self.obs_dict["crashes"].reshape(-1).to(dtype=torch.bool, device=self.device)
        _cf = torch.norm(self.obs_dict["robot_contact_force_tensor"], dim=-1)
        _bdist = self._body_center_distance_to_target()
        self._ep_min_body_dist[:] = torch.minimum(self._ep_min_body_dist, _bdist)
        if train_intercept_metrics_enabled():
            record_train_step_mean_dist(float(_bdist.mean().item()))
        _min_force_contact = _ct & (_cf >= self._effective_intercept_min_force())
        _radius_hit = self._body_center_intercept_mask(_bdist)
        _qualifying_hit = self._qualifying_intercept_hit_mask(_bdist, _ct, _cf)
        self._ep_phys_contact_substeps += _qualifying_hit.to(dtype=torch.long)
        self._ep_radius_no_contact_substeps += (_radius_hit & ~_min_force_contact).to(dtype=torch.long)
        self._ep_contact_no_radius_substeps += (_min_force_contact & ~_radius_hit).to(dtype=torch.long)

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        if getattr(self.task_config, "sync_sim_reset_with_terminations", False):
            done = (self.terminations.reshape(-1) > 0) | (self.truncations.reshape(-1) > 0)
            self.obs_dict["crashes"][:] = done.to(dtype=torch.float32)
        rw = self.rewards.reshape(-1)
        self._ep_return += rw
        self._ep_len += 1
        self._finalize_episode_step_stats()
        self.sim_env.post_reward_calculation_step()

        if t_wall0 is not None:
            self._last_step_wall_s = time.perf_counter() - t_wall0

        self.infos = {}

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
            self._world_target_positions() - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self._world_target_positions()
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )

        body_dist = self._body_center_distance_to_target()
        to_tgt = target_position - robot_position
        dist3 = torch.norm(to_tgt, dim=1).clamp(min=1e-6)
        closing_speed = ((robot_linvel * to_tgt).sum(dim=1) / dist3).clamp(min=0.0)
        if getattr(self.task_config, "no_penalty_on_intercept_contact", True):
            intercept_proximity = self._body_center_intercept_mask(body_dist).to(
                device=self.device, dtype=torch.float32
            )
        else:
            intercept_proximity = torch.zeros(
                self.sim_env.num_envs,
                device=self.device,
                dtype=torch.float32,
            )

        contact_force_norm = torch.norm(
            obs_dict["robot_contact_force_tensor"], dim=-1
        ).reshape(-1)

        return compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            contact_force_norm,
            closing_speed,
            1.0,  # curriculum multiplier slot (미사용이면 1.0 유지)
            intercept_proximity,
            self._ep_len.to(dtype=torch.float32),
            self.task_config.reward_parameters,
        )


@torch.jit.script
def _pos_exp_term(dist: torch.Tensor, gain: torch.Tensor, exp_coef: torch.Tensor) -> torch.Tensor:
    return gain * torch.exp(-exp_coef * dist * dist)


@torch.jit.script
def compute_reward(
    pos_error,
    lin_vels,
    robot_quats,
    robot_angvels,
    crashes,
    contact_force_norm,
    closing_speed_mps,
    curriculum_level_multiplier,
    intercept_proximity,
    elapsed_ep_steps,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)

    pos_reward = _pos_exp_term(dist, parameter_dict["pos_gain_1"], parameter_dict["pos_exp_1"]) + _pos_exp_term(
        dist, parameter_dict["pos_gain_2"], parameter_dict["pos_exp_2"]
    )

    switch = parameter_dict["dist_switch_m"]
    zone_mid = parameter_dict["dist_zone_mid_m"]
    # Legacy 2-zone (dist_zone_mid_m <= dist_switch_m)
    dist_far = (parameter_dict["dist_far_ref_m"] - dist) / parameter_dict["dist_far_div"]
    far_at_switch = (parameter_dict["dist_far_ref_m"] - switch) / parameter_dict["dist_far_div"]
    _sw_safe = switch.clamp(min=1e-4)  # prevent div-by-zero when switch==0
    d_in = (switch - dist).clamp(min=0.0)
    near_quad = far_at_switch + parameter_dict["dist_near_quad_gain"] * (d_in / _sw_safe) * (d_in / _sw_safe)
    dist_two_zone = torch.where(dist >= switch, dist_far, near_quad)
    # 3-zone: far d>=zone_mid | mid zone_mid>d>=switch | near d<switch (continuous at boundaries)
    r_at_mid = (parameter_dict["dist_far_ref_m"] - zone_mid) / parameter_dict["dist_far_div"]
    r_at_near = r_at_mid + (zone_mid - switch) / parameter_dict["dist_mid_div"]
    dist_far_z = (parameter_dict["dist_far_ref_m"] - dist) / parameter_dict["dist_far_div"]
    d_mid = (zone_mid - dist).clamp(min=0.0)
    dist_mid_z = r_at_mid + d_mid / parameter_dict["dist_mid_div"]
    dist_near_z = r_at_near + parameter_dict["dist_near_quad_gain"] * (d_in / _sw_safe) * (d_in / _sw_safe)
    dist_three_zone = torch.where(
        dist >= zone_mid, dist_far_z, torch.where(dist >= switch, dist_mid_z, dist_near_z)
    )
    dist_reward = torch.where(zone_mid > switch + 0.001, dist_three_zone, dist_two_zone)

    # Hyperbolic approach guidance: r = g/(d+r0) - g/(oor+r0), zero at leash, large near target.
    # Activates when dist_hype_gain > 0; fully replaces zone-based dist_reward.
    _hg = parameter_dict["dist_hype_gain"]
    _hr0 = parameter_dict["dist_hype_r0"]
    _oor_h = parameter_dict["out_of_range_dist"]
    _hype = (_hg / (dist + _hr0) - _hg / (_oor_h + _hr0)).clamp(min=0.0)
    dist_reward = torch.where(_hg > 0.001, _hype, dist_reward)

    lin_speed = torch.norm(lin_vels, dim=1)
    lin_speed_pen = parameter_dict["lin_speed_pen_amp"] * torch.tanh(
        lin_speed * parameter_dict["lin_speed_pen_tanh_scale"]
    )

    fo_start = parameter_dict["far_overshoot_pen_start_m"]
    fo_coef = parameter_dict["far_overshoot_pen_coef"]
    fo_ref = parameter_dict["far_overshoot_pen_ref_m"]
    overshoot_excess = (dist - fo_start).clamp(min=0.0)
    overshoot_pen = fo_coef * overshoot_excess * overshoot_excess / (fo_ref * fo_ref)

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = parameter_dict["upright_num"] / (parameter_dict["upright_eps"] + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = parameter_dict["ang_vel_weight"] / (torch.ones_like(spinnage) + spinnage * spinnage)

    posture = parameter_dict["posture_coupling_scale"] * (up_reward + ang_vel_reward)

    total_reward = pos_reward + dist_reward + pos_reward * posture - lin_speed_pen - overshoot_pen
    total_reward[:] = total_reward + parameter_dict["closing_speed_bonus_coef"] * closing_speed_mps
    # Massoud et al. 요격 변형: 선형 거비 −λ·dist (정지 타깃까지의 ||e||), 논문 R의 −λ·d_t 항 대응
    lm = parameter_dict["massoud_linear_dist_coef"]
    total_reward[:] = total_reward - lm * dist
    # 다중 센서 장벽이 없어, "요격 접근 구간 바깥"에서만 과속에 약한 벌점으로 근사 (옵션)
    d_ca = parameter_dict["massoud_caution_dist_m"]
    k_ca = parameter_dict["massoud_speed_caution_coef"]
    far_and_outside_intercept_shell = dist > d_ca
    total_reward[:] = total_reward - k_ca * lin_speed * far_and_outside_intercept_shell.to(
        dtype=torch.float32
    )

    far_d = parameter_dict["far_time_dist_penalty_dist_m"]
    min_far_s = parameter_dict["far_time_dist_penalty_min_steps"]
    far_coef = parameter_dict["far_time_dist_penalty_coef"]
    far_late = (dist > far_d) & (elapsed_ep_steps >= min_far_s)
    far_pen = far_coef * (dist - far_d).clamp(min=0.0)
    total_reward[:] = total_reward - far_pen * far_late.to(dtype=torch.float32)

    # Per-step dwell penalty (constant base; ramp coef should stay 0 — quadratic ramp caused reward collapse).
    time_norm = parameter_dict["time_penalty_norm_steps"]
    t_frac = elapsed_ep_steps / time_norm
    time_pen = parameter_dict["time_step_penalty"] + parameter_dict["time_ramp_penalty_coef"] * t_frac * t_frac
    total_reward[:] = total_reward - time_pen
    # Distance-proportional dwell penalty: costs more to linger far away, near-zero at target.
    # With time_dist_pen_coef > 0 this makes hovering at any range unprofitable vs. approaching.
    total_reward[:] = total_reward - parameter_dict["time_dist_pen_coef"] * dist

    total_reward[:] = curriculum_level_multiplier * total_reward

    physics_contact = crashes > 0.0

    oor = parameter_dict["out_of_range_dist"]
    crashes[:] = torch.where(dist > oor, torch.ones_like(crashes), crashes)

    penalty = parameter_dict["crash_penalty"]
    near_intercept = intercept_proximity > 0.5
    min_force = parameter_dict["intercept_min_contact_force"]
    good_intercept_touch = physics_contact & near_intercept & (contact_force_norm >= min_force)
    hit_bonus = parameter_dict["intercept_collision_bonus"]

    norm_steps = parameter_dict["intercept_hit_time_norm_steps"]
    time_eff = torch.clamp(1.0 - elapsed_ep_steps / norm_steps, min=0.0, max=1.0)
    time_bonus = parameter_dict["intercept_hit_time_bonus_amp"] * time_eff
    impact_speed_eff = torch.clamp(
        closing_speed_mps / parameter_dict["intercept_impact_speed_ref_mps"], min=0.0, max=1.0
    )
    impact_bonus = parameter_dict["intercept_impact_speed_bonus_amp"] * impact_speed_eff
    intercept_extra = hit_bonus + time_bonus + impact_bonus

    total_reward[:] = torch.where(good_intercept_touch, total_reward + intercept_extra, total_reward)

    bad_touch = physics_contact & ~good_intercept_touch
    off_bonus = parameter_dict["off_target_contact_reward"]
    pen_scale = parameter_dict["non_intercept_crash_penalty_scale"]
    total_reward[:] = torch.where(
        bad_touch,
        total_reward + off_bonus - penalty * pen_scale,
        total_reward,
    )

    gs = parameter_dict["reward_global_scale"]
    total_reward[:] = total_reward * gs

    # Moving intercept: end episode only on qualifying intercept or leash (not ground / glancing contact).
    if parameter_dict["terminate_only_qualifying_intercept"] > 0.5:
        crashes[:] = torch.where(
            good_intercept_touch | (dist > oor),
            torch.ones_like(crashes),
            torch.zeros_like(crashes),
        )

    return total_reward, crashes
