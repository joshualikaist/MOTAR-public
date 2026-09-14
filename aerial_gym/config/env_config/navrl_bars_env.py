import os

from aerial_gym.config.asset_config.env_object_config import (
    NAVRL_DISTRACTOR_COUNT,
    bar_asset_params,
    navrl_distractor_box_params,
    navrl_distractor_pole_params,
    navrl_distractor_sphere_params,
    navrl_physical_target_params,
    navrl_physical_target_v2_params,
    navrl_physical_target_v3_params,
)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


# Arena footprint [m]. Default 24 preserves every v1 result byte-identically; the v2 search
# arena sets NAVRL_ARENA_XY=40 to match NavRL's 40 x 40 m map (map_range [20,20] in
# reference/NavRL isaac-training env.py:102). The height stays 3 m ground-referenced.
_ARENA_XY = _env_float("NAVRL_ARENA_XY", 24.0)
_ARENA_Z = _env_float("NAVRL_ARENA_Z", 3.0)
if not 8.0 <= _ARENA_XY <= 80.0:
    raise ValueError("NAVRL_ARENA_XY must be in [8, 80] m")
if not 2.0 <= _ARENA_Z <= 8.0:
    raise ValueError("NAVRL_ARENA_Z must be in [2, 8] m")


class NavRLBarsEnvCfg:
    """Controlled Phase-1 arena: an otherwise-empty space with a field of static bars.

    Deliberately minimal (a static-bars navigation field): no walls, no panels, no random
    clutter -- just N vertical bars the drone must fly through to reach the goal.

    Geometry is ground-referenced (z = height above the floor), so it matches the navrl_task
    height bounds (0.1..4.0 m). The drone spawns at the left edge (robot init x-ratio ~0 -> x in
    [0, ~1]) and the goal is placed on the far side at x=k (curriculum, ~5 m out to the far wall
    ~23.5 m), so every episode crosses the whole bar field. The arena is 24 x 24 m; the active
    bar count is set by task_config.density / NAVRL_NUM_BARS for Phase-2 density sweeps.
    """

    class env:
        num_envs = 64  # overridden by the task/runner num_envs
        num_env_actions = 4
        env_spacing = 5.0  # not used with warp meshes, kept for parity

        num_physics_steps_per_env_step_mean = 10
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 1
        reset_on_collision = True
        collision_force_threshold = 0.05  # [N]
        create_ground_plane = False  # no physical floor (matches env_with_obstacles); z=0 is floor level
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = True  # required for the warp LiDAR

        # Obstacle placement. "random" = legacy NavRL-style scatter with a min center-to-center
        # distance that RELAXES (*0.8) when saturated. Measured 2026-07-31
        # (tools/probe_placement_slits.py): fine through 110 bars (0 impassable slits, free space
        # fully connected), but at 150 bars the relaxation fires in 100% of layouts and produces
        # ~2.2 impassable (<0.40 m) + ~15 marginal (<0.60 m) surface gaps per layout.
        # "navrl_band" = slit-free rule mirroring the reference NavRL terrain generator's
        # good_distance() forbidden band: a candidate is accepted only if every already-placed bar
        # is either TOUCHING it (centers <= touch dist -> bars merge into a compound wall) or at
        # least gap dist away. This centre-distance rule prevents overlap but does not guarantee
        # a fixed surface corridor; diagonal maximum-size bars can leave only ~0.469 m. Saturation
        # merges instead of relaxing the forbidden centre-distance band.
        obstacle_placement_mode = (
            os.environ.get("NAVRL_PLACEMENT_MODE", "").strip().lower() or "random"
        )
        obstacle_placement_attempts_before_relax = 128
        obstacle_placement_relax_factor = 0.8
        obstacle_placement_candidate_batch_size = 32
        min_obstacle_xy_spacing = 1.5
        # navrl_band parameters (center-to-center): touch <= 0.4 guarantees overlap for every
        # footprint pair in the 0.4..0.8 m pool. gap >= 1.6 prevents AABB overlap, but does NOT
        # guarantee a 0.8 m surface corridor: with two 0.8 m squares separated diagonally the
        # theoretical corner-to-corner gap can be only ~0.469 m. This centre-only rule therefore
        # needs the separate footprint/connectivity audit before any passability claim.
        obstacle_touch_dist = _env_float("NAVRL_PLACEMENT_TOUCH_M", 0.4)
        obstacle_gap_dist = _env_float("NAVRL_PLACEMENT_GAP_M", 1.6)
        # New physical-lineage placement contract.  Uses each URDF collision footprint, forbids
        # overlap, remains valid under yaw via a circumcircle bound, and fails closed instead of
        # merging bars. Historical navrl_band checkpoints retain their original contract.
        obstacle_surface_clearance = _env_float(
            "NAVRL_PLACEMENT_SURFACE_CLEARANCE_M", 0.45
        )

        # Arena (min == max so every env is identical). Ground-referenced: z in [0, Z].
        # Default 24 x 24 x 3 m (v1); NAVRL_ARENA_XY=40 gives the NavRL-scale search arena.
        # Drone spawns at the left edge (init x-ratio ~0); general-spawn training randomizes
        # start/goal inside the bounds at runtime, so the size propagates automatically.
        lower_bound_min = [0.0, 0.0, 0.0]
        lower_bound_max = [0.0, 0.0, 0.0]
        upper_bound_min = [_ARENA_XY, _ARENA_XY, _ARENA_Z]
        upper_bound_max = [_ARENA_XY, _ARENA_XY, _ARENA_Z]

    class env_config:
        # Legacy/bounded lineages contain only bars and inject the virtual target analytically.
        # Physical mode additionally creates one dynamic PhysX target actor. Its moving mesh is
        # excluded from Warp and the actor's current OBB is ray-tested analytically, avoiding a
        # full-scene refit. Walls, panels, objects, trees stay OFF.
        _physical_target = os.environ.get("NAVRL_TARGET_DYNAMICS", "legacy").strip().lower() == "physical"
        _physical_geometry_v2 = os.environ.get(
            "NAVRL_PHYSICAL_GEOMETRY_VERSION", "v1"
        ).strip().lower() == "v2"
        # Target appearance. v3 keeps v2's inertial and collision byte for byte and changes only
        # the visual, so this selects what a camera sees and nothing else. Default v2: an existing
        # run must not change silhouette because a new asset appeared in the tree.
        _target_appearance_v3 = os.environ.get(
            "NAVRL_TARGET_APPEARANCE", "v2"
        ).strip().lower() == "v3"
        # Appearance distractors (NAVRL_DISTRACTOR_COUNT, default 0). Each shape class already
        # reports num_assets == 0 when the knob is unset, and AssetLoader skips a zero-count asset
        # type before it reads the folder or draws from the RNG, so leaving the knob unset changes
        # nothing: same assets, same order, same placement stream.
        num_distractors = NAVRL_DISTRACTOR_COUNT

        include_asset_type = {
            "distractor_sphere": True,
            "distractor_box": True,
            "distractor_pole": True,
            "physical_target": _physical_target,
            "bars": True,
        }

        # ORDER MATTERS. AssetLoader.select_and_order_assets appendleft()s every keep_in_env asset,
        # so the LAST keep_in_env asset loaded ends up at obstacle index 0. The physical target
        # must stay at index 0 (NavRLTask reads obstacle_position[:, 0] as the target actor), so
        # the distractors -- also keep_in_env -- are declared BEFORE it. The resulting layout is
        #   [target?] [distractors...] [bars...]
        # and NavRLTask's _bar_offset is 1(target) + num_distractors.
        asset_type_to_dict_map = {
            "distractor_sphere": navrl_distractor_sphere_params,
            "distractor_box": navrl_distractor_box_params,
            "distractor_pole": navrl_distractor_pole_params,
            # keep_in_env puts this at obstacle index 0; NavRLTask offsets every bar slice past it.
            "physical_target": (
                (navrl_physical_target_v3_params if _target_appearance_v3
                 else navrl_physical_target_v2_params)
                if _physical_geometry_v2
                else navrl_physical_target_params
            ),
            "bars": bar_asset_params,
        }
