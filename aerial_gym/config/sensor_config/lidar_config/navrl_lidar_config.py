import os

from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)

_NAVRL_VISION = os.environ.get("NAVRL_VISION", "0").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)

# LiDAR sensing range. NavRL's spec is 4.0 m (kept as the default for the GT-injected LiDAR task).
# The sensor-only perception task needs a longer obstacle horizon than 4 m -- at 2 m/s a bar becomes
# visible only ~2 s before impact, so PPO learned to hover rather than risk approach (see
# CRASH_TUNING_LOG.md). NAVRL_LIDAR_RANGE overrides it (e.g. 8). MUST match the perception module's
# lidar_max_range (navrl_task_config.perception.lidar_max_range reads the SAME env var).
_NAVRL_HBEAMS = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "").strip() or 36)
_NAVRL_VBEAMS = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "").strip() or 4)

try:
    _NAVRL_LIDAR_RANGE = float(os.environ.get("NAVRL_LIDAR_RANGE", "").strip() or "4.0")
except ValueError:
    _NAVRL_LIDAR_RANGE = 4.0


class NavRLLidarConfig(BaseLidarConfig):
    """Ray-cast LiDAR matching NavRL's RayCaster (Xu et al., "NavRL", RA-L 2025).

    Reference (reference/NavRL/isaac-training/training/cfg/drone.yaml):
        lidar_range  = 4.0          # meters
        lidar_vfov   = [-10, 20]    # degrees
        lidar_vbeams = 4            # vertical beams
        lidar_hres   = 10           # degrees  -> 360/10 = 36 horizontal beams
    The RayCaster is attached yaw-only to the drone body.
    """

    # --- scan resolution: NavRL spec is 36 horizontal x 4 vertical (lidar_hres=10, lidar_vbeams=4).
    # The horizontal count is overridable because it sets the ANGULAR QUANTIZATION of every obstacle
    # token the policy sees: token positions are built from range/angle geometry, so their lateral
    # error is about half a bin (~0.44 m at 5 m with 10 deg bins). The bar_contact probe measured a
    # 0.57 m token error, matching that prediction -- so 72 beams (5 deg) halves it.
    # MUST equal navrl_perception.HBEAMS/VBEAMS: both read these same env vars, and a mismatch would
    # reshape the scan tensor into the wrong view.
    width = _NAVRL_HBEAMS  # horizontal beams
    height = _NAVRL_VBEAMS  # vertical beams

    # --- vertical field of view: NavRL lidar_vfov = [-10, 20] -> beams at {-10, 0, 10, 20} deg
    vertical_fov_deg_min = -10.0
    vertical_fov_deg_max = 20.0

    # --- horizontal 360 deg.
    # Aerial Gym generates rays with inclusive endpoints (linspace over [min, max]). A full
    # [-180, 180] span would duplicate the +/-180 ray and give 360/(width-1) spacing. Stopping one
    # bin short of a full turn yields exactly `width` distinct beams at 360/width deg: at width=36
    # that is [-170, 180] at 10 deg (NavRL's lidar_hres=10), at width=72 it is [-175, 180] at 5 deg.
    # The absolute azimuth offset is irrelevant (drone-relative scan), but the SPACING must match
    # navrl_perception._lidar_angles, which derives its bearings the same way.
    horizontal_fov_deg_min = -180.0 + 360.0 / _NAVRL_HBEAMS
    horizontal_fov_deg_max = 180.0

    # --- range: NavRL lidar_range = 4.0 m (override with NAVRL_LIDAR_RANGE for the perception task)
    max_range = _NAVRL_LIDAR_RANGE
    min_range = 0.2

    # Out-of-range fill (recomputed here because the base class evaluated these against its own
    # max_range=10.0; with normalize_range=True a no-hit ray should map to 1.0 after dividing by
    # max_range, so the far fill must equal *this* max_range).
    normalize_range = True
    far_out_of_range_value = max_range  # -> 1.0 after normalization (no obstacle in that ray)
    near_out_of_range_value = -max_range

    # --- attach yaw-only: scan plane stays level under body roll/pitch (NavRL attach_yaw_only)
    yaw_only_attach = True

    # --- obstacle + target range/semantics. In vision mode the nearest return can be either an
    # environment mesh (bar/wall) or the moving target sphere; semantic id 50 identifies target
    # returns. This complements, rather than replaces, the forward camera observation.
    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = _NAVRL_VISION
    inject_target = _NAVRL_VISION
    target_radius = 0.20
    # Physical mode replaces the historical sphere with the actor's exact oriented collision box.
    target_box_half_extents = [0.14, 0.14, 0.06]
    target_use_oriented_box = (
        os.environ.get("NAVRL_TARGET_DYNAMICS", "legacy").strip().lower() == "physical"
    )
    target_semantic_id = 50

    # --- NavRL ray caster: sensor at body center, no placement randomization, no noise
    randomize_placement = False
    min_translation = [0.0, 0.0, 0.0]
    max_translation = [0.0, 0.0, 0.0]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]
    nominal_position = [0.0, 0.0, 0.0]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]
    euler_frame_rot_deg = [0.0, 0.0, 0.0]

    class sensor_noise:
        enable_sensor_noise = False
        std_a = 0.0
        std_b = 0.0
        std_c = 0.0
        mean_offset = 0.0
        pixel_dropout_prob = 0.0
