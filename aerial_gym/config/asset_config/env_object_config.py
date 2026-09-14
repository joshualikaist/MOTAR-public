from aerial_gym import AERIAL_GYM_DIRECTORY

import os
import numpy as np

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
PANEL_SEMANTIC_ID = 20
FRONT_WALL_SEMANTIC_ID = 9
BACK_WALL_SEMANTIC_ID = 10
LEFT_WALL_SEMANTIC_ID = 11
RIGHT_WALL_SEMANTIC_ID = 12
BOTTOM_WALL_SEMANTIC_ID = 13
TOP_WALL_SEMANTIC_ID = 14


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


# X placement band as a fraction of the arena, shared by navrl_target_params (obstacle index 0,
# which AssetManager._placement_band reads for ALL obstacles) and bar_asset_params.
#
# The legacy 0.13..0.96 window existed only because the v1 task spawned the drone at x~0 and flew
# it left-to-right, so the spawn strip had to stay clear. v2 uses NAVRL_GENERAL_TRAIN=1 (drone AND
# target spawn uniformly over the whole arena), which voids that premise and leaves 17% of the v2
# arena permanently obstacle free -- episodes spawning there degenerate into straight-line pursuit
# no matter how high the density curriculum climbs, and reported density is inflated because the
# real flyable area exceeds the placement band. Both spawn samplers stay 1 m inside the walls and reject positions too close to a bar --
# the general drone spawn now checks each bar's own footprint surface (2026-08-27, see
# navrl_task.py's _spawn_footprint_clearance_accepted), not a flat 0.65 m from centre; the
# static-target goal fallback still uses the flat 0.65 m `goal_min_bar_clearance`. Either
# way a full-width band is safe.
# Defaults preserve v1 exactly.
_BAR_X_MIN = _env_float("NAVRL_BAR_X_MIN", 0.13)
_BAR_X_MAX = _env_float("NAVRL_BAR_X_MAX", 0.96)
if not 0.0 <= _BAR_X_MIN < _BAR_X_MAX <= 1.0:
    raise ValueError("NAVRL_BAR_X_MIN/MAX must satisfy 0 <= min < max <= 1")


class asset_state_params:
    num_assets = 1  # number of assets to include

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file = None  # if file=None, random assets will be selected. If not None, this file will be used

    min_position_ratio = [0.5, 0.5, 0.5]  # min position as a ratio of the bounds
    max_position_ratio = [0.5, 0.5, 0.5]  # max position as a ratio of the bounds

    collision_mask = 1

    disable_gravity = False
    replace_cylinder_with_capsule = (
        True  # replace collision cylinders with capsules, leads to faster/more stable simulation
    )
    flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
    density = 0.001
    angular_damping = 0.1
    linear_damping = 0.1
    max_angular_velocity = 100.0
    max_linear_velocity = 100.0
    armature = 0.001

    collapse_fixed_joints = True
    fix_base_link = True
    specific_filepath = None  # if not None, use this folder instead randomizing
    color = None
    keep_in_env = False

    body_semantic_label = 0
    link_semantic_label = 0
    per_link_semantic = False
    semantic_masked_links = {}
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # position, quat x, y, z, w

    use_collision_mesh_instead_of_visual = False
    # Isaac Gym physics and Warp ray casting do not have to contain the same asset. Dynamic
    # actors whose pose changes every physics step use an analytic sensor primitive instead of
    # forcing an expensive full-scene mesh refit.
    include_in_warp = True


class panel_asset_params(asset_state_params):
    num_assets = 3

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_position_ratio = [0.3, 0.05, 0.05]  # max position as a ratio of the bounds
    max_position_ratio = [0.85, 0.95, 0.95]  # min position as a ratio of the bounds

    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # if > -900, use this value instead of randomizing   the ratios

    min_euler_angles = [0.0, 0.0, -np.pi / 3.0]  # min euler angles
    max_euler_angles = [0.0, 0.0, np.pi / 3.0]  # max euler angles

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        0.0,
        0.0,
        -np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        0.0,
        0.0,
        np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
    color = [170, 66, 66]


class tile_asset_params(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/tile_meshes"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_position_ratio = [0.3, 0.05, 0.05]  # max position as a ratio of the bounds
    max_position_ratio = [0.85, 0.95, 0.95]  # min position as a ratio of the bounds

    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # if > -900, use this value instead of randomizing   the ratios

    min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
    max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

    min_state_ratio = [
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
    # color = [170, 66, 66]


class thin_asset_params(asset_state_params):
    num_assets = 0

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
    color = [170, 66, 66]


class tree_asset_params(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.1,
        0.1,
        0.0,
        0,
        -np.pi / 6.0,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.9,
        0.9,
        0.0,
        0,
        np.pi / 6.0,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True
    per_link_semantic = True
    keep_in_env = True

    semantic_id = -1  # TREE_SEMANTIC_ID
    color = [70, 200, 100]

    semantic_masked_links = {}


class object_asset_params(asset_state_params):
    num_assets = 35

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

    min_state_ratio = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance

    # color = [80,255,100]


class left_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "left_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = LEFT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class right_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "right_wall.urdf"

    min_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    per_link_semantic = False
    specific_filepath = "cube.urdf"
    semantic_id = RIGHT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class top_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "top_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = TOP_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class bottom_wall(asset_state_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "bottom_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BOTTOM_WALL_SEMANTIC_ID
    color = [100, 150, 150]


class front_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "front_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = FRONT_WALL_SEMANTIC_ID
    color = [100, 200, 210]


class back_wall(asset_state_params):
    num_assets = 1

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "back_wall.urdf"

    collision_mask = 1  # objects with the same collision mask will not collide

    min_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True

    collapse_fixed_joints = True
    specific_filepath = "cube.urdf"
    per_link_semantic = False
    semantic_id = BACK_WALL_SEMANTIC_ID
    color = [100, 200, 210]


INTERCEPT_TARGET_SEMANTIC_ID = 50


class intercept_quad_target_params(asset_state_params):
    """
    Intercept target: center collision sphere R=0.05 m only (no arm geometry).
    Matches chaser `quad/quad.urdf` base_link collision sphere and
    `intercept_success_radius` (center distance ≤ 0.11 m at first touch).
    """

    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "intercept_target_quad_scale.urdf"
    collision_mask = 0
    disable_gravity = True
    density = 0.000001
    replace_cylinder_with_capsule = False  # align with flying quad robot_asset
    flip_visual_attachments = True
    collapse_fixed_joints = False  # align with quad.urdf as used by base_quadrotor
    fix_base_link = True
    keep_in_env = True
    per_link_semantic = False
    semantic_id = INTERCEPT_TARGET_SEMANTIC_ID
    color = [220, 60, 60]
    # Offset from center of empty_env cube (see EmptyEnvCfg bounds ± env_spacing)
    min_state_ratio = [
        0.72,
        0.48,
        0.52,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.72,
        0.48,
        0.52,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


class navrl_target_params(asset_state_params):
    """NavRL Phase-3 moving target drone (semantic_id=50).

    A single drone-sized body the pursuer must LOCATE from its onboard sensors (no ground-truth
    position). It is registered as an env asset so its mesh is baked into the raycastable warp
    scene and shows up in the LiDAR/camera SEGMENTATION channel as id=50 (bars are id=3). It is
    kept_in_env (so it lands at obstacle index 0 -- the task slices obstacle_position[:, 1:] for
    bars and drives obstacle_position[:, 0] itself each step) and never deactivated.

    fix_base_link + disable_gravity + collision_mask=0: the body is teleported kinematically by
    NavRLTask (no physics dynamics, no crash on contact -- capture stays the geometric point test).

    The XY/Z state ratio is DELIBERATELY the same as bar_asset_params: AssetManager._placement_band
    reads asset index 0's ratio as the placement band for ALL obstacles, so a divergent ratio here
    would collapse the bar-scatter band. The initial pose is overwritten by the task at reset
    anyway, so the sampled placement is irrelevant -- only the band-protection matters.
    """

    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    file = "navrl_target_drone.urdf"

    collision_mask = 0
    disable_gravity = True
    density = 0.000001
    replace_cylinder_with_capsule = False
    flip_visual_attachments = True
    collapse_fixed_joints = False
    fix_base_link = True
    keep_in_env = True
    per_link_semantic = False
    semantic_id = INTERCEPT_TARGET_SEMANTIC_ID  # 50
    color = [220, 40, 40]

    # Same band as bar_asset_params (see docstring). _placement_band reads asset index 0 -- which
    # is THIS asset -- so the x window must track NAVRL_BAR_X_MIN/MAX or the bars keep the legacy
    # band no matter what bar_asset_params says.
    min_state_ratio = [_BAR_X_MIN, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [_BAR_X_MAX, 1.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class navrl_physical_target_params(navrl_target_params):
    """Dynamic 6-DoF target actor used by ``NAVRL_TARGET_DYNAMICS=physical``.

    The URDF is a single rigid body carrying the ref5in design-point mass/inertia and exact
    collision box. A task-side four-motor controller applies the equivalent body wrench at every
    0.01 s PhysX step. The same oriented box is ray-tested by camera and LiDAR, so no stale Warp
    mesh or sphere proxy can disagree with contact geometry.
    """

    collision_mask = 0
    disable_gravity = False
    density = 1.0
    collapse_fixed_joints = True
    fix_base_link = False
    keep_in_env = True
    include_in_warp = False


class navrl_physical_target_v2_params(navrl_physical_target_params):
    """Fresh-only target whose collision proxy rounds the declared prop span outward."""

    file = "navrl_target_drone_v2.urdf"


class navrl_physical_target_v3_params(navrl_physical_target_v2_params):
    """v2's physics with a quadrotor silhouette instead of a solid box.

    Same inertial, same collision box, same single base_link: the simulator reads collision only,
    in asset_loader._urdf_collision_half_extents and in NavRLTask's base_link box lookup, so
    spawn, contact and dynamics are untouched and a test calls both extractors to say so.

    What changes is what a camera sees. The target was a featureless box that shared one colour
    with every distractor, which makes colour enough to find it. The silhouette now has arms and
    propeller discs, generated by tools/generate_shared_airframe.py from dimensions that already
    existed: motor positions and cylinders from quad_navrl_ref5in_v2.urdf, the propeller radius
    from PROP_RADIUS_5IN_M, and the body box from that URDF's own inertia approximation.

    Measured, not assumed: width goes from 0.283 m to 0.2826 m, but projected area falls to 54-70%
    of the box across 21 viewing directions, because a quadrotor has gaps between its arms and a
    box does not. Apparent linear size therefore moves to about 0.77 of the box. Apparent size
    against range is a detector cue, so any comparison across this change is confounded by it
    unless a size-matched control is included.

    Opt in with NAVRL_TARGET_APPEARANCE=v3. The default stays v2 so no existing run changes.
    """

    file = "navrl_target_drone_v3.urdf"


class bar_asset_params(asset_state_params):
    """Variable-size vertical obstacle bars ("막대") for the controlled NavRL bars environment.

    Bars are randomly picked (file=None) from a pool of generated box URDFs whose horizontal
    footprint (width, depth) is random in [0.4, 0.8] m and whose height is fixed at 2 m
    (see resources/models/environment_assets/bars, generated by tools/generate_bar_assets.py).
    Each box is centered at its origin, so the z-ratio puts the center at 1 m => the bar spans
    0..2 m in the ground-referenced [0, 3] m arena. num_assets bars are scattered in a central
    band so the drone (which spawns near the low-x edge and flies at 1 m altitude) must weave
    through them to reach nearby goals. No walls, no panels -- an otherwise empty space.

    Non-overlap and a minimum inter-bar gap are enforced separately by the environment via
    NavRLBarsEnvCfg.env.min_obstacle_xy_spacing (a minimum center-to-center XY distance).
    """

    # Build-time ceiling for the density sweep. Runtime active bars are controlled by
    # task_config.density / NAVRL_NUM_BARS; inactive bars are moved to -1000 by AssetManager.
    num_assets = max(0, _env_int("NAVRL_MAX_BARS", 150))

    # Bar pool selection. "bars" = legacy 2.0 m pool (fly-over possible in a 3 m arena);
    # "bars_h3" = full-height 3.0 m pool (v2 search arena: removes the fly-over option,
    # matching NavRL's mostly-unoverflyable obstacles). Same footprints/seed either way, so
    # this changes ONLY the height. The bar z-ratio must put the box center at height/2:
    # 2 m pool -> 1.0 m (ratio 1/3 in a 3 m arena), 3 m pool -> 1.5 m (ratio 1/2).
    _bar_pool = (os.environ.get("NAVRL_BAR_POOL", "").strip() or "bars")
    _bar_z_ratio = 0.5 if _bar_pool == "bars_h3" else 0.3333

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/{_bar_pool}"
    file = None  # random pick from the pool -> each bar gets a random footprint

    collision_mask = 1  # bars do not collide with each other

    # [x_ratio, y_ratio, z_ratio, roll, pitch, yaw, 1.0, vx,vy,vz, wx,wy,wz]
    # x band starts past the drone spawn strip (drone x-ratio ~0, x <~ 1); z ratio places the box
    # center at bar_height/2 (pool-dependent, see _bar_z_ratio) so bars stand on the floor; upright.
    min_state_ratio = [
        _BAR_X_MIN,
        0.0,
        _bar_z_ratio,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        _BAR_X_MAX,
        1.0,
        _bar_z_ratio,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False

    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = OBJECT_SEMANTIC_ID
    color = [170, 66, 66]


# ---------------------------------------------------------------------------------------------
# Appearance distractors (D9 of docs/archive/development_directions_2026-08.md).
#
# The bootstrap detector is a 1x1 conv computing 3.0*R - 2.0*G - 2.0*B - 0.9 (a red detector),
# and the renderer paints the target a flat [0.88, 0.08, 0.045] while NOTHING else in the scene
# is red. "red pixel => target" is therefore true by construction, and the ~99.77% frame precision
# v7 reports was measured in a world containing zero objects that resemble the target. These
# assets add that missing axis: static, collidable bodies the renderer paints the SAME nominal
# target colour, so the colour rule fires on something that is not the target.
#
# DEFAULT OFF. NAVRL_DISTRACTOR_COUNT defaults to 0, every class below then reports
# num_assets == 0, and AssetLoader.select_and_order_assets skips a zero-count asset type before
# it touches the filesystem or the RNG -- so with the knob unset the built world, the asset
# ordering and the placement stream are bit-identical to the historical ones.
DISTRACTOR_SEMANTIC_ID = 51  # target = 50, bars/objects = 3

_DISTRACTOR_SHAPES = ("sphere", "box", "pole")

# Total number of distractors per env. 0 = the historical world.
NAVRL_DISTRACTOR_COUNT = max(0, _env_int("NAVRL_DISTRACTOR_COUNT", 0))

# Shape mix, as a comma-separated subset of _DISTRACTOR_SHAPES. The count is dealt out
# round-robin over this list in the order written, so "sphere" alone gives an all-sphere field
# and the default gives a 1:1:1 mix (rounded in list order).
_DISTRACTOR_SHAPE_MIX_RAW = (
    os.environ.get("NAVRL_DISTRACTOR_SHAPES", "").strip() or "sphere,box,pole"
)
_DISTRACTOR_SHAPE_MIX = tuple(
    s.strip().lower() for s in _DISTRACTOR_SHAPE_MIX_RAW.split(",") if s.strip()
)
if not _DISTRACTOR_SHAPE_MIX:
    raise ValueError("NAVRL_DISTRACTOR_SHAPES must name at least one shape")
_unknown_shapes = [s for s in _DISTRACTOR_SHAPE_MIX if s not in _DISTRACTOR_SHAPES]
if _unknown_shapes:
    raise ValueError(
        "NAVRL_DISTRACTOR_SHAPES contains unknown shape(s) %s; valid shapes are %s"
        % (", ".join(sorted(set(_unknown_shapes))), ", ".join(_DISTRACTOR_SHAPES))
    )


def _distractor_shape_counts(total, mix):
    """Deal `total` distractors round-robin over `mix`, in the order `mix` is written."""
    counts = {shape: 0 for shape in _DISTRACTOR_SHAPES}
    if total <= 0:
        return counts
    for i in range(int(total)):
        counts[mix[i % len(mix)]] += 1
    return counts


_DISTRACTOR_COUNTS = _distractor_shape_counts(NAVRL_DISTRACTOR_COUNT, _DISTRACTOR_SHAPE_MIX)


class _navrl_distractor_base(asset_state_params):
    """Static, collidable body painted the nominal target colour by the NavRL renderer.

    Physically real, not a decal: the URDF collision primitive equals the visual primitive, so the
    LiDAR returns it, the drone can crash into it, and it occludes the target exactly like a bar.
    ``include_in_warp`` therefore stays True -- the renderer's distractor mask is a mesh ray cast
    keyed on ``semantic_id``, so an asset excluded from Warp would be invisible to the camera
    while still being solid, which is the one combination that would be a lie.

    ``keep_in_env = True`` puts these right behind the target (index 0) and in front of the bars,
    so they exist at every density: AssetManager parks assets beyond
    ``num_obstacles_in_env = _bar_offset + n_bars_active`` at -1000, and a distractor placed after
    the bars would be parked whenever the density curriculum ran below the build-time bar ceiling.
    NavRLTask widens ``_bar_offset`` by the distractor count so every bar slice keeps addressing
    bars only.

    The XY state ratio is the SAME band as bar_asset_params / navrl_target_params, for the reason
    documented there: AssetManager._placement_band reads asset index 0's ratio for ALL obstacles,
    and in a legacy (non-physical-target) lineage index 0 is one of THESE.
    """

    num_assets = 0
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

    collision_mask = 1  # same as bars: distractors do not collide with each other
    disable_gravity = True
    density = 0.000001
    replace_cylinder_with_capsule = False
    flip_visual_attachments = True
    collapse_fixed_joints = True
    fix_base_link = True
    keep_in_env = True
    per_link_semantic = False
    semantic_id = DISTRACTOR_SEMANTIC_ID
    color = [224, 20, 11]  # 255 * [0.88, 0.08, 0.045], for the Isaac Gym viewer only

    # Per-shape state ratios are written out below as literal 13-vectors, exactly as
    # bar_asset_params does:
    #   [x_ratio, y_ratio, z_ratio, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]
    # The z ratio puts the body's centre at its intended height in the default 3 m arena, the
    # same assumption bar_asset_params._bar_z_ratio makes.


class navrl_distractor_sphere_params(_navrl_distractor_base):
    """0.15 m sphere -- deliberately the target's own camera_target_radius.

    Centre at 1.0 m in the default 3 m arena: the pursuer's cruise altitude, so it sits in the
    forward camera's FOV rather than at the edge of it.
    """

    num_assets = _DISTRACTOR_COUNTS["sphere"]
    file = "navrl_distractor_sphere.urdf"
    min_state_ratio = [_BAR_X_MIN, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [_BAR_X_MAX, 1.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class navrl_distractor_box_params(_navrl_distractor_base):
    """0.30 m cube -- target-scale, different aspect ratio. Same cruise-altitude centre."""

    num_assets = _DISTRACTOR_COUNTS["box"]
    file = "navrl_distractor_box.urdf"
    min_state_ratio = [_BAR_X_MIN, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [_BAR_X_MAX, 1.0, 0.3333, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class navrl_distractor_pole_params(_navrl_distractor_base):
    """0.06 m radius x 1.60 m pole -- taller and thinner than the target.

    Floor-standing like a bar: the centre sits at h/2 = 0.80 m, i.e. 0.80/3 of the default arena.
    """

    num_assets = _DISTRACTOR_COUNTS["pole"]
    file = "navrl_distractor_pole.urdf"
    min_state_ratio = [_BAR_X_MIN, 0.0, 0.26667, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [_BAR_X_MAX, 1.0, 0.26667, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
