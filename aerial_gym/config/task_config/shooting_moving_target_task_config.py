from aerial_gym.config.task_config.position_setpoint_task_config import task_config as _base


class task_config(_base):
    sim_name = "moving_intercept_sim"
    env_name = "moving_intercept_env"
    robot_name = "moving_intercept_quadrotor"
    observation_space_dim = 16
    sync_sim_reset_with_terminations = True
    # 6 m arena raises the nominal intercept range; keep time budget 1.5x the old 1000-step setup.
    episode_len_steps = 1500  # sim dt≈0.01 s → ~15 s
    target_speed_min = 0.28
    target_speed_max = 0.45
    target_speed_curriculum_enable = True
    target_speed_start_min = 0.08
    target_speed_start_max = 0.16
    target_speed_curriculum_start_epoch = 400
    target_speed_curriculum_end_epoch = 1600
    curriculum_steps_per_epoch = 64
    reflect_target_at_env_bounds = True

    command_velocity_scale = 2.5
    command_yaw_rate_scale = 2.5

    # Terminal success budget = 700 + 250 + 50 = 1000.
    # Episode return can exceed 1000 because dense, milestone, and finish rewards are added.
    # Failure ep (1500-step timeout) is bounded by dense floor/ceil and should stay below success.
    reward_episode_design_max = 1000.0

    intercept_success_radius = 0.12

    reward_parameters = {
        **_base.reward_parameters,

        "pos_gain_1": 0.0,
        "pos_gain_2": 0.0,
        "posture_coupling_scale": 0.0,
        "dist_hype_gain": 0.0,
        # Disable inherited absolute-distance shaping; moving target should reward closing/progress.
        "dist_far_ref_m": 0.0,
        "dist_far_div": 1.0e6,
        "dist_switch_m": 0.0,
        "dist_zone_mid_m": 0.0,
        "dist_mid_div": 1.0e6,
        "dist_near_quad_gain": 0.0,
        "closing_speed_bonus_coef": 0.0,

        # Dense distance field: far = linear, near = continuous exponential boost.
        "dist_focus_far_ref_m": 5.0,
        "dist_focus_far_slope": 0.12,
        "dist_focus_switch_m": 1.2,
        "dist_focus_near_amp": 0.85,
        "dist_focus_near_exp": 4.0,
        "dist_focus_plateau_m": 0.30,

        # Pursuit terms stay secondary; distance itself should dominate this run.
        "range_rate_bonus_coef": 0.8,
        "dist_progress_bonus_coef": 3.0,

        "time_step_penalty": 0.0,
        "time_dist_pen_coef": 0.06,
        "range_stall_penalty_coef": 0.10,
        "range_stall_exempt_dist_m": 0.20,
        "range_open_penalty_coef": 2.0,

        # Finish funnel uses surface gap, not raw center distance.
        # Target and robot render/collision spheres touch at roughly 0.10 m center distance;
        # the strict success gate still requires contact + center distance <= intercept_success_radius.
        "surface_contact_center_dist_m": 0.10,
        "near_miss_dist_m": 0.18,
        "finish_funnel_outer_gap_m": 0.15,  # center distance ~= 0.25 m
        "finish_funnel_inner_gap_m": 0.02,  # center distance ~= 0.12 m
        "finish_funnel_outer_m": 0.25,      # legacy center-distance equivalent
        "finish_funnel_inner_m": 0.12,      # legacy center-distance equivalent
        "finish_funnel_closing_coef": 1.0,
        "finish_funnel_step_ceil": 0.50,
        "finish_core_gap_m": 0.02,
        "finish_core_dist_m": 0.12,         # legacy center-distance equivalent
        "finish_core_bonus": 60.0,

        # Keep non-terminal dense bounded; close-but-no-hit should not dominate terminal intercept.
        "reward_dense_step_floor": -0.12,
        "reward_dense_step_ceil": 1.2,
        "reward_dense_terminal_threshold": 400.0,

        "out_of_range_dist": 10.0,
        "far_overshoot_pen_start_m": 4.0,
        "far_overshoot_pen_coef": 0.12,
        "massoud_linear_dist_coef": 0.0,
        "massoud_speed_caution_coef": 0.0,
        "far_time_dist_penalty_coef": 0.0,
        "milestone_dist_1_m": 1.5,
        "milestone_bonus_1": 20.0,
        "milestone_dist_2_m": 0.75,
        "milestone_bonus_2": 35.0,
        "milestone_dist_3_m": 0.30,
        "milestone_bonus_3": 60.0,

        "intercept_collision_bonus": 700.0,
        "intercept_hit_time_bonus_amp": 250.0,
        "intercept_hit_time_norm_steps": 1500.0,
        "intercept_impact_speed_bonus_amp": 50.0,
        "time_penalty_norm_steps": 1500.0,
        "reward_global_scale": 1.0,
        "terminate_only_qualifying_intercept": 1.0,
    }
