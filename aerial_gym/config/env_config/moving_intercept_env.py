from aerial_gym.config.asset_config.env_object_config import intercept_quad_target_params


class MovingInterceptEnvCfg:
    class env:
        num_envs = 3
        num_env_actions = 0
        # 6 m cube (±3 m) — 1.5x larger moving-target intercept volume.
        env_spacing = 3.0
        num_physics_steps_per_env_step_mean = 1
        num_physics_steps_per_env_step_std = 0
        render_viewer_every_n_steps = 10
        collision_force_threshold = 0.010
        manual_camera_trigger = False
        reset_on_collision = True
        create_ground_plane = True
        ground_plane_style = "white"  # solid white floor (not Isaac Gym checkerboard)
        # Bit 1: no robot–ground contact (robot uses collision_mask=1 in moving_intercept_quadrotor).
        ground_collision_filter = 1
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        # Task updates target pose in env_asset_state_tensor each RL step; push before physics.
        write_to_sim_at_every_timestep = True

        use_warp = False
        e_s = env_spacing
        lower_bound_min = [-e_s, -e_s, -e_s]
        lower_bound_max = [-e_s, -e_s, -e_s]
        upper_bound_min = [e_s, e_s, e_s]
        upper_bound_max = [e_s, e_s, e_s]

    class env_config:
        include_asset_type = {
            "intercept_target": True,
        }

        asset_type_to_dict_map = {
            "intercept_target": intercept_quad_target_params,
        }
