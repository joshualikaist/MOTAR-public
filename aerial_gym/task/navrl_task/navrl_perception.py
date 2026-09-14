"""Perception front-end for the NavRL++-Target policy.

The actor never receives RGB-D pixels, LiDAR semantic IDs, or simulator target state.  This
module consumes raw RGB-D and raw LiDAR ranges, detects the target from appearance, associates
camera and LiDAR measurements, tracks target position/velocity/covariance, and emits a fixed-size
structured history for the Transformer policy.

The simulator may use ground truth to *render* sensor pixels, just as a real scene produces sensor
measurements.  Ground truth is intentionally absent from every public method in this file.
"""

import hashlib
import math
import os
import pathlib

import torch
import torch.nn as nn

from aerial_gym.task.navrl_task.navrl_corridor import CORRIDOR_DIM, extract_corridor_tokens
from aerial_gym.task.navrl_task.navrl_search_state import (
    SEARCH_BELIEF_DIM,
    SEARCH_COVERAGE_DIM,
    SEARCH_MODE_DIM,
    SEARCH_STATES,
    SearchGrid,
    search_feature_dim,
)


ROBOT_HISTORY = 5
TARGET_HISTORY = 5
OBSTACLE_HISTORY = 5
# Obstacle-token capacity. Env-overridable so it can be swept; changing it changes
# STRUCTURED_OBS_DIM and therefore requires a FRESH policy (no warm-start across capacities).
MAX_OBSTACLES = int(os.environ.get("NAVRL_MAX_OBSTACLES", "").strip() or 5)
ROBOT_DIM = 10
TARGET_DIM = 16
OBSTACLE_DIM = 12

# Agreement window that declares a sensor return "the target, not static geometry", used to carve
# the target out of the obstacle map. Named constants because P2's latency-aware variant must
# reproduce the same window from the PREDICTED target pose; two literals would silently drift.
TARGET_LIKE_ANGLE_RAD = math.radians(15.0)
TARGET_LIKE_RANGE_TOL_M = 0.55

# Standard deviation standing in for "this sensor said nothing about this direction". Large
# enough that the Kalman gain perpendicular to the measurement ray is numerically zero, while
# keeping R well-conditioned.
LIDAR_UNOBSERVED_SIGMA_M = 50.0

# LiDAR scan resolution. HBEAMS is THE source of truth for the horizontal beam count and must equal
# navrl_lidar_config.width (both read NAVRL_LIDAR_HBEAMS). Changing it changes STRUCTURED_OBS_DIM,
# hence a fresh policy is required. The old claim that token error was dominated by angular
# quantization was refuted by evaluation; see WORKLOG 2026-07-27 and bar-probe v2.
HBEAMS = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "").strip() or 36)
VBEAMS = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "").strip() or 4)
# Angular half-width blanked around an accepted obstacle token, in DEGREES (not bins: the bin size
# changes with HBEAMS). It stops one wide bar from consuming several token slots, but it also caps
# how many tokens can EVER be filled at 360/(2*this). The original +-2 bins at 36 beams was +-20 deg
# = at most ~7 tokens, so raising MAX_OBSTACLES alone silently wasted the extra slots. At +-10 deg
# the ceiling is 18, comfortably above any capacity we intend to sweep.
OBSTACLE_SUPPRESS_DEG = float(os.environ.get("NAVRL_OBSTACLE_SUPPRESS_DEG", "").strip() or 10.0)
# Obstacle proposal policy. ``greedy_suppress`` preserves the historical nearest-bearing selector.
# ``cluster_sector`` first joins adjacent ray endpoints that plausibly lie on one physical surface,
# then reserves one nearest cluster per angular sector. Empty sectors are filled from the remaining
# nearest clusters. Both modes emit the same [MAX_OBSTACLES, OBSTACLE_DIM] shape.
OBSTACLE_SELECTOR = (
    os.environ.get("NAVRL_OBSTACLE_SELECTOR", "").strip().lower() or "greedy_suppress"
)
OBSTACLE_CLUSTER_GAP_M = float(
    os.environ.get("NAVRL_OBSTACLE_CLUSTER_GAP_M", "").strip() or 0.45
)
OBSTACLE_SECTORS = int(
    os.environ.get("NAVRL_OBSTACLE_SECTORS", "").strip() or MAX_OBSTACLES
)
# Angular sector (centered on body-forward, in DEGREES) the obstacle tokens are selected from.
# 360 = the original behavior: pick the nearest MAX_OBSTACLES bars anywhere around the drone.
# Why narrowing helps: the probe measured ~16 bars inside the horizon against 8 token slots, so a
# 360-degree search represents only ~50% of them -- which is exactly the measured hit_in_tokens
# (0.40-0.53). Spending slots on bars BEHIND the drone is what starves the ones ahead of it.
# Restricting the sector raises coverage geometrically (240 deg -> ~75%, 180 deg -> ~100%) WITHOUT
# changing the observation width, so a policy can warm-start across this change.
# Omnidirectional awareness is not lost: the static scan token still carries all HBEAMS bearings.
OBSTACLE_FOV_DEG = float(os.environ.get("NAVRL_OBSTACLE_FOV_DEG", "").strip() or 360.0)
if MAX_OBSTACLES <= 0:
    raise ValueError("NAVRL_MAX_OBSTACLES must be positive")
if HBEAMS <= 0 or VBEAMS <= 0:
    raise ValueError("NAVRL_LIDAR_HBEAMS and NAVRL_LIDAR_VBEAMS must be positive")
if not math.isfinite(OBSTACLE_SUPPRESS_DEG) or OBSTACLE_SUPPRESS_DEG < 0.0:
    raise ValueError("NAVRL_OBSTACLE_SUPPRESS_DEG must be finite and non-negative")
if OBSTACLE_SELECTOR not in ("greedy_suppress", "cluster_sector", "ttc_sector"):
    raise ValueError(
        "NAVRL_OBSTACLE_SELECTOR must be 'greedy_suppress', 'cluster_sector' or 'ttc_sector'"
    )
# ttc_sector only: a cluster that the drone is not closing on is ranked by this fallback time
# instead of being dropped, so a slow/hovering drone still tokenizes its surroundings by proximity
# rather than emitting an empty set.
OBSTACLE_TTC_IDLE_S = float(
    os.environ.get("NAVRL_OBSTACLE_TTC_IDLE_S", "").strip() or 30.0
)
# Speed below which closing rate is treated as unreliable and selection degrades to nearest-first.
OBSTACLE_TTC_MIN_SPEED = float(
    os.environ.get("NAVRL_OBSTACLE_TTC_MIN_SPEED", "").strip() or 0.15
)
if not math.isfinite(OBSTACLE_TTC_IDLE_S) or OBSTACLE_TTC_IDLE_S <= 0.0:
    raise ValueError("NAVRL_OBSTACLE_TTC_IDLE_S must be finite and positive")
if not math.isfinite(OBSTACLE_TTC_MIN_SPEED) or OBSTACLE_TTC_MIN_SPEED < 0.0:
    raise ValueError("NAVRL_OBSTACLE_TTC_MIN_SPEED must be finite and non-negative")
if not math.isfinite(OBSTACLE_CLUSTER_GAP_M) or OBSTACLE_CLUSTER_GAP_M <= 0.0:
    raise ValueError("NAVRL_OBSTACLE_CLUSTER_GAP_M must be finite and positive")
if OBSTACLE_SECTORS <= 0 or OBSTACLE_SECTORS > MAX_OBSTACLES:
    raise ValueError("NAVRL_OBSTACLE_SECTORS must be in 1..NAVRL_MAX_OBSTACLES")
if not math.isfinite(OBSTACLE_FOV_DEG) or not 0.0 < OBSTACLE_FOV_DEG <= 360.0:
    raise ValueError("NAVRL_OBSTACLE_FOV_DEG must be in (0, 360]")


def obstacle_selector_provenance(selector, configured_fov_deg):
    """Return the candidate FOV and whether angular suppression actually executes."""
    selector = str(selector).strip().lower()
    configured_fov_deg = float(configured_fov_deg)
    if selector not in ("greedy_suppress", "cluster_sector", "ttc_sector"):
        raise ValueError("unsupported obstacle selector: %s" % selector)
    if not math.isfinite(configured_fov_deg) or not 0.0 < configured_fov_deg <= 360.0:
        raise ValueError("configured_fov_deg must be in (0, 360]")
    return {
        "effective_fov_deg": 360.0 if selector == "ttc_sector" else configured_fov_deg,
        "suppress_active": selector == "greedy_suppress",
    }


# Provenance must describe the selector's actual candidate set. TTC intentionally ranks all 360
# degrees, while cluster_sector/greedy_suppress obey the configured FOV. Suppression is only used
# by the legacy greedy selector; logging it as active for cluster/TTC previously misdescribed runs.
_OBSTACLE_PROVENANCE = obstacle_selector_provenance(OBSTACLE_SELECTOR, OBSTACLE_FOV_DEG)
OBSTACLE_EFFECTIVE_FOV_DEG = _OBSTACLE_PROVENANCE["effective_fov_deg"]
OBSTACLE_SUPPRESS_ACTIVE = _OBSTACLE_PROVENANCE["suppress_active"]
# Corridor (free-gap) affordance tokens -- see navrl_corridor.py. 0 (the default) keeps the
# historical 898-D schema byte-identical; any positive count APPENDS
# CORRIDOR_TOKENS * CORRIDOR_DIM features at the END of the observation, so every existing
# segment keeps its offset and a checkpoint can be warm-started by expanding only the input
# projections (see runner._expand_corridor_checkpoint). Changing this is a schema change:
# provenance is recorded in env_state as cfg_corridor_tokens.
CORRIDOR_TOKENS = int(os.environ.get("NAVRL_CORRIDOR_TOKENS", "").strip() or 0)
CORRIDOR_HORIZON_M = float(os.environ.get("NAVRL_CORRIDOR_HORIZON_M", "").strip() or 6.0)
CORRIDOR_MIN_WIDTH_M = float(os.environ.get("NAVRL_CORRIDOR_MIN_WIDTH_M", "").strip() or 0.55)
if CORRIDOR_TOKENS < 0 or CORRIDOR_TOKENS > 16:
    raise ValueError("NAVRL_CORRIDOR_TOKENS must be in 0..16")
if not math.isfinite(CORRIDOR_HORIZON_M) or CORRIDOR_HORIZON_M <= 0.0:
    raise ValueError("NAVRL_CORRIDOR_HORIZON_M must be finite and positive")
if not math.isfinite(CORRIDOR_MIN_WIDTH_M) or CORRIDOR_MIN_WIDTH_M < 0.0:
    raise ValueError("NAVRL_CORRIDOR_MIN_WIDTH_M must be finite and non-negative")
CORRIDOR_OBS_DIM = CORRIDOR_TOKENS * CORRIDOR_DIM
STATIC_DIM = VBEAMS * HBEAMS

# Opt-in active-search geofence token. Ranges are four body-ray distances
# (forward/left/back/right) to a known arena geofence, obtainable from VIO/GPS plus a mapped flight
# boundary. Disabled is byte-identical to the historical schema. Enabled appends four normalized
# ranges and four validity flags and therefore requires a fresh policy.
GEOFENCE_ACTOR = os.environ.get("NAVRL_GEOFENCE_ACTOR", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
GEOFENCE_RAYS = 4
GEOFENCE_DIM = 2 * GEOFENCE_RAYS if GEOFENCE_ACTOR else 0
GEOFENCE_NOISE_STD_M = float(
    os.environ.get("NAVRL_GEOFENCE_NOISE_STD_M", "").strip() or 0.0
)
GEOFENCE_DROPOUT = float(os.environ.get("NAVRL_GEOFENCE_DROPOUT", "").strip() or 0.0)
# Evaluation-only mechanism ablation. It preserves the 906-D schema while replacing all ranges by
# the declared missing-value sentinel and clearing validity. Training launchers never set it.
GEOFENCE_FORCE_INVALID = os.environ.get(
    "NAVRL_GEOFENCE_FORCE_INVALID", "0"
).strip().lower() in ("1", "true", "yes", "on")
# Explicit blind-search state. ``off`` leaves the historical actor bytes and token count intact;
# geofence reuses the existing 8-D token; coverage/belief append one new search token.
SEARCH_STATE = os.environ.get("NAVRL_SEARCH_STATE", "off").strip().lower() or "off"
if SEARCH_STATE not in SEARCH_STATES:
    raise ValueError("NAVRL_SEARCH_STATE must be off|geofence|coverage|belief")
if SEARCH_STATE != "off" and not GEOFENCE_ACTOR:
    raise ValueError(
        "NAVRL_SEARCH_STATE=%s requires NAVRL_GEOFENCE_ACTOR=1" % SEARCH_STATE
    )
SEARCH_DIM = search_feature_dim(SEARCH_STATE)
SEARCH_STATE_FORCE_INVALID = os.environ.get(
    "NAVRL_SEARCH_STATE_FORCE_INVALID", "0"
).strip().lower() in ("1", "true", "yes", "on")
if not math.isfinite(GEOFENCE_NOISE_STD_M) or GEOFENCE_NOISE_STD_M < 0.0:
    raise ValueError("NAVRL_GEOFENCE_NOISE_STD_M must be finite and non-negative")
if not math.isfinite(GEOFENCE_DROPOUT) or not 0.0 <= GEOFENCE_DROPOUT <= 1.0:
    raise ValueError("NAVRL_GEOFENCE_DROPOUT must be in [0,1]")


def lidar_bin_bearings(device=None):
    """Physical body-frame bearing of each horizontal scan bin, in radians.

    THE single source of truth for bin -> bearing. It mirrors the warp ray generator
    (warp_lidar.py:67: azimuth = hfov_max - span * j/(W-1)), so the bearing DECREASES with the bin
    index: bin 0 = +180 deg, bin W-1 = -(180 - 360/W) deg.

    History: this module used to assume the INCREASING convention (linspace(-180+bin, +180)),
    which is the mirror image (assumed = bin_deg - true). Every obstacle token was therefore
    emitted on the wrong side of the drone, the camera fusion ghosted forward bars onto the
    opposite bearing, and the LiDAR target re-association read the mirrored bin. Physically
    adjudicated with tools/probe_lidar_bearing.py: back-projecting real returns through the
    increasing table lands on a GT bar only 13.9% of the time (mean error 2.5 m); through this
    decreasing table 94.8% (mean error 0.47 m ~= bar surface-to-center). Any edit here must keep
    tests/test_lidar_bearing_convention.py and that probe passing.
    """
    bin_rad = 2.0 * math.pi / HBEAMS
    return torch.linspace(math.pi, -math.pi + bin_rad, HBEAMS, device=device)


def camera_no_return_to_lidar_range(camera_col_min, camera_max_range, lidar_max_range):
    """Map camera columns that saw NOTHING to the LiDAR no-return value before min-fusion.

    The obstacle depth camera fills no-hit rays with its own far plane (camera_obstacle_max_range,
    10 m). When the LiDAR range exceeds that (12 m), min-fusing the raw fill value stamps a phantom
    10 m wall across the whole forward sector: the actor can never observe true free space ahead,
    and the token selector manufactures obstacles out of the camera's horizon. A column at (or
    beyond) the camera far plane carries no obstacle information, so it must fuse as lidar_max
    (= no return), leaving the LiDAR's own measurement untouched.
    """
    no_return = camera_col_min >= float(camera_max_range) - 1e-3
    return torch.where(
        no_return, torch.full_like(camera_col_min, float(lidar_max_range)), camera_col_min
    )


def select_cluster_sector_obstacles(
    nearest,
    bearings,
    *,
    max_range,
    max_obstacles,
    token_fov_deg,
    cluster_gap_m,
    num_sectors,
):
    """Select distinct, angularly distributed LiDAR surface clusters.

    ``nearest`` is [batch, horizontal_beams]. Adjacent valid ray endpoints are assigned to the same
    cluster when their Euclidean separation is at most ``cluster_gap_m``. One nearest cluster is
    reserved per fixed body-frame angular sector, then empty sector capacity is filled with the
    nearest still-unselected clusters. Returned proposals are sorted by range so a same-shape
    warm-start retains the historical nearest-first slot convention as closely as possible.

    The intended experiment uses a forward 240-degree FOV, so its eligible bearings form one
    contiguous interval away from the +/-pi scan seam. Full-360 operation deliberately treats the
    seam as a boundary; the static scan token still retains the complete circular geometry.
    """
    if nearest.ndim != 2 or bearings.ndim != 1 or nearest.shape[1] != bearings.shape[0]:
        raise ValueError("nearest must be [batch, beams] and bearings must be [beams]")
    if max_obstacles <= 0 or num_sectors <= 0 or num_sectors > max_obstacles:
        raise ValueError("sector count must be in 1..max_obstacles")
    if not 0.0 < float(token_fov_deg) <= 360.0:
        raise ValueError("token_fov_deg must be in (0, 360]")
    if not math.isfinite(float(cluster_gap_m)) or float(cluster_gap_m) <= 0.0:
        raise ValueError("cluster_gap_m must be finite and positive")

    batch, beams = nearest.shape
    device = nearest.device
    max_range = float(max_range)
    half_fov = math.radians(float(token_fov_deg) * 0.5)
    eligible = nearest < max_range * 0.995
    if token_fov_deg < 359.9:
        eligible &= bearings.abs().view(1, beams) <= half_fov + 1e-7

    # A range jump alone is a poor object boundary: two rays on one nearby cylinder can differ in
    # range noticeably. Endpoint distance combines angular separation and radial change in metres.
    x = nearest * torch.cos(bearings).view(1, beams)
    y = nearest * torch.sin(bearings).view(1, beams)
    endpoint_gap = torch.sqrt(
        (x[:, 1:] - x[:, :-1]).square() + (y[:, 1:] - y[:, :-1]).square()
    )
    linked_to_previous = (
        eligible[:, 1:]
        & eligible[:, :-1]
        & (endpoint_gap <= float(cluster_gap_m))
    )
    boundary = torch.ones((batch, beams), dtype=torch.bool, device=device)
    boundary[:, 1:] = ~linked_to_previous
    cluster_id = boundary.long().cumsum(dim=1)

    remaining = eligible.clone()
    rows = torch.arange(batch, device=device)
    sector_ranges = []
    sector_indices = []
    sector_valid = []
    sector_width = (2.0 * half_fov) / float(num_sectors)

    for sector in range(num_sectors):
        low = -half_fov + sector * sector_width
        high = low + sector_width
        if sector == num_sectors - 1:
            in_sector = (bearings >= low - 1e-7) & (bearings <= high + 1e-7)
        else:
            in_sector = (bearings >= low - 1e-7) & (bearings < high - 1e-7)
        work = nearest.masked_fill(~(remaining & in_sector.view(1, beams)), max_range)
        picked_range, picked_idx = work.min(dim=1)
        picked_valid = picked_range < max_range * 0.995
        sector_ranges.append(picked_range)
        sector_indices.append(picked_idx)
        sector_valid.append(picked_valid)

        picked_cluster = cluster_id[rows, picked_idx]
        remove = cluster_id.eq(picked_cluster.unsqueeze(1)) & picked_valid.unsqueeze(1)
        remaining &= ~remove

    # Generate enough global candidates to fill every empty sector. They are lower-priority than
    # all valid sector representatives, irrespective of range.
    fallback_ranges = []
    fallback_indices = []
    fallback_valid = []
    for _ in range(max_obstacles):
        work = nearest.masked_fill(~remaining, max_range)
        picked_range, picked_idx = work.min(dim=1)
        picked_valid = picked_range < max_range * 0.995
        fallback_ranges.append(picked_range)
        fallback_indices.append(picked_idx)
        fallback_valid.append(picked_valid)

        picked_cluster = cluster_id[rows, picked_idx]
        remove = cluster_id.eq(picked_cluster.unsqueeze(1)) & picked_valid.unsqueeze(1)
        remaining &= ~remove

    ranges = torch.stack(sector_ranges + fallback_ranges, dim=1)
    indices = torch.stack(sector_indices + fallback_indices, dim=1)
    valid = torch.stack(sector_valid + fallback_valid, dim=1)
    source_is_fallback = torch.cat(
        [
            torch.zeros(num_sectors, device=device, dtype=nearest.dtype),
            torch.ones(max_obstacles, device=device, dtype=nearest.dtype),
        ]
    ).view(1, -1)
    priority = source_is_fallback * 2.0 + ranges / max_range
    priority = priority.masked_fill(~valid, float("inf"))
    keep = priority.argsort(dim=1)[:, :max_obstacles]
    ranges = ranges.gather(1, keep)
    indices = indices.gather(1, keep)
    valid = valid.gather(1, keep)

    # Preserve nearest-first slot semantics for the downstream flattened MLP.
    range_order = ranges.masked_fill(~valid, float("inf")).argsort(dim=1)
    ranges = ranges.gather(1, range_order)
    indices = indices.gather(1, range_order)
    valid = valid.gather(1, range_order)
    return ranges, indices, valid


def select_ttc_obstacles(
    nearest,
    bearings,
    body_vel_xy,
    *,
    max_range,
    max_obstacles,
    cluster_gap_m,
    idle_ttc_s=30.0,
    min_speed=0.15,
):
    """Select the obstacle clusters the drone is most imminently going to hit.

    ``cluster_sector`` allocates slots by BEARING: one cluster per fixed forward sector. That is a
    proxy for threat, and the crash probe measured where the proxy breaks (run ppo_260731_1722,
    2300 epochs, 70 bars): 23.6% of the bars actually struck were outside the 240-degree token
    window entirely, and of those inside it, 11.7% still received no token because a sector only
    reserves its single nearest cluster. Both losses share one cause -- a bar is dangerous because
    the drone is MOVING INTO it, and bearing alone does not encode that. While searching, the drone
    yaws hard, so a bar that was forward moments ago sits behind the window while the velocity
    vector still points at it.

    This selector ranks clusters by time-to-collision instead:

        closing = -d(range)/dt ~= body_velocity . unit_vector_to_cluster
        ttc     = range / closing            when closing > 0
                = idle_ttc_s + range/max_range  otherwise (receding: ordered last, by proximity)

    Consequences: a receding bar dead ahead yields its slot to an approaching bar off to the side;
    a second cluster in one sector can be tokenized when it is the more urgent one; and no bearing
    is excluded a priori, so the rear blind sector disappears without widening any sensor. Below
    ``min_speed`` the closing estimate is dominated by noise, so ranking degrades continuously to
    nearest-first -- which is exactly ``cluster_sector`` behaviour at a standstill.

    Shape and semantics of the return match ``select_cluster_sector_obstacles`` exactly
    ([batch, max_obstacles] ranges/indices/valid, sorted nearest-first), so this is a same-width
    swap that a trained policy can warm-start across.
    """
    if nearest.ndim != 2 or bearings.ndim != 1 or nearest.shape[1] != bearings.shape[0]:
        raise ValueError("nearest must be [batch, beams] and bearings must be [beams]")
    if body_vel_xy.ndim != 2 or body_vel_xy.shape[0] != nearest.shape[0] or body_vel_xy.shape[1] != 2:
        raise ValueError("body_vel_xy must be [batch, 2]")
    if max_obstacles <= 0:
        raise ValueError("max_obstacles must be positive")
    if not math.isfinite(float(cluster_gap_m)) or float(cluster_gap_m) <= 0.0:
        raise ValueError("cluster_gap_m must be finite and positive")

    batch, beams = nearest.shape
    device = nearest.device
    max_range = float(max_range)
    eligible = nearest < max_range * 0.995

    # Identical clustering to cluster_sector: adjacent endpoints within cluster_gap_m are one
    # physical surface. Keeping this shared means the A/B isolates the RANKING, not the grouping.
    cos_b = torch.cos(bearings).view(1, beams)
    sin_b = torch.sin(bearings).view(1, beams)
    x = nearest * cos_b
    y = nearest * sin_b
    endpoint_gap = torch.sqrt(
        (x[:, 1:] - x[:, :-1]).square() + (y[:, 1:] - y[:, :-1]).square()
    )
    linked_to_previous = eligible[:, 1:] & eligible[:, :-1] & (endpoint_gap <= float(cluster_gap_m))
    boundary = torch.ones((batch, beams), dtype=torch.bool, device=device)
    boundary[:, 1:] = ~linked_to_previous
    cluster_id = boundary.long().cumsum(dim=1)

    # Closing speed toward each bearing. Bars are static, so the drone's own motion is the whole
    # closing rate; a moving target is never in this scan (it is a virtual point, no mesh).
    closing = body_vel_xy[:, 0:1] * cos_b + body_vel_xy[:, 1:2] * sin_b
    speed = body_vel_xy.norm(dim=1, keepdim=True)
    approaching = (closing > 1e-3) & (speed > float(min_speed))
    ttc = torch.where(
        approaching,
        nearest / closing.clamp(min=1e-3),
        float(idle_ttc_s) + nearest / max_range,
    )
    ttc = ttc.masked_fill(~eligible, float("inf"))

    remaining = eligible.clone()
    rows = torch.arange(batch, device=device)
    picked_ranges, picked_indices, picked_valid = [], [], []
    for _ in range(max_obstacles):
        work = ttc.masked_fill(~remaining, float("inf"))
        _, idx = work.min(dim=1)
        r = nearest[rows, idx]
        valid = remaining[rows, idx] & (r < max_range * 0.995)
        picked_ranges.append(r)
        picked_indices.append(idx)
        picked_valid.append(valid)
        # Consume the whole cluster so one wide surface cannot occupy several slots.
        picked_cluster = cluster_id[rows, idx]
        remaining &= ~(cluster_id.eq(picked_cluster.unsqueeze(1)) & valid.unsqueeze(1))

    ranges = torch.stack(picked_ranges, dim=1)
    indices = torch.stack(picked_indices, dim=1)
    valid = torch.stack(picked_valid, dim=1)

    # Preserve nearest-first slot semantics for the downstream flattened MLP.
    order = ranges.masked_fill(~valid, float("inf")).argsort(dim=1)
    return ranges.gather(1, order), indices.gather(1, order), valid.gather(1, order)


STRUCTURED_OBS_DIM = (
    ROBOT_HISTORY * ROBOT_DIM
    + TARGET_HISTORY * TARGET_DIM
    + OBSTACLE_HISTORY * MAX_OBSTACLES * OBSTACLE_DIM
    + STATIC_DIM
    + CORRIDOR_OBS_DIM
    + GEOFENCE_DIM
    + SEARCH_DIM
)


def body_geofence_features(
    drone_pos_w,
    vehicle_quat,
    env_bounds_min,
    env_bounds_max,
    noise_std_m=0.0,
    dropout=0.0,
):
    """Return [F,L,B,R] ray ranges plus validity, normalized by the XY diagonal."""
    if env_bounds_min is None or env_bounds_max is None:
        raise ValueError("geofence actor requires per-environment bounds")
    if env_bounds_min.shape[0] != drone_pos_w.shape[0] or env_bounds_max.shape[0] != drone_pos_w.shape[0]:
        raise ValueError("geofence bounds batch must match drone batch")
    n = drone_pos_w.shape[0]
    body_dirs = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=drone_pos_w.dtype,
        device=drone_pos_w.device,
    )
    dirs_w = _quat_rotate_xyzw(
        vehicle_quat[:, None, :].expand(-1, GEOFENCE_RAYS, -1).reshape(-1, 4),
        body_dirs[None, :, :].expand(n, -1, -1).reshape(-1, 3),
    ).reshape(n, GEOFENCE_RAYS, 3)[:, :, :2]
    pos = drone_pos_w[:, None, :2]
    lo = env_bounds_min[:, None, :2]
    hi = env_bounds_max[:, None, :2]
    eps = 1e-7
    t = torch.where(
        dirs_w > eps,
        (hi - pos) / dirs_w.clamp(min=eps),
        torch.where(
            dirs_w < -eps,
            (lo - pos) / dirs_w.clamp(max=-eps),
            torch.full_like(dirs_w, float("inf")),
        ),
    )
    ranges = t.masked_fill(t < 0.0, float("inf")).min(dim=2).values
    # This is an always-on sensor contract, not generic perception augmentation: configured noise
    # and dropout apply in both rollout and evaluation. Zero/zero is the causal first A/B.
    if float(noise_std_m) > 0.0:
        ranges = ranges + torch.randn_like(ranges) * float(noise_std_m)
    span = (env_bounds_max[:, :2] - env_bounds_min[:, :2]).norm(dim=1, keepdim=True)
    ranges = (ranges.clamp(min=0.0) / span.clamp(min=1e-6)).clamp(max=1.0)
    valid = torch.ones_like(ranges)
    if float(dropout) > 0.0:
        valid = (torch.rand_like(ranges) >= float(dropout)).to(ranges.dtype)
        ranges = torch.where(valid.bool(), ranges, torch.ones_like(ranges))
    return torch.cat([ranges, valid], dim=1)


def _quat_rotate_xyzw(q, v):
    qvec = q[:, :3]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return v + 2.0 * (q[:, 3:4] * uv + uuv)


def _quat_mul_xyzw(a, b):
    """Hamilton product of xyzw quaternion rows: rotation b followed by rotation a."""
    ax, ay, az, aw = a.unbind(dim=1)
    bx, by, bz, bw = b.unbind(dim=1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=1,
    )


def _quat_rotate_inverse_xyzw(q, v):
    qi = torch.cat([-q[:, :3], q[:, 3:4]], dim=1)
    return _quat_rotate_xyzw(qi, v)


class SpatialTargetSegmenter(nn.Module):
    """Small conv head (7x7 receptive field) for appearance-randomised scenes.

    The preregistered Gate 3 escalation: the 1x1 per-pixel colour rule failed the offline gate
    under the declared appearance envelope (pixel precision 0.17 at hue +/-60, light +/-0.5 --
    a per-pixel rule cannot separate an arbitrary-hue target from a jittered background), which
    opens exactly this step. ~2.9k parameters keeps 128 parallel camera streams cheap; the
    dilated second layer buys spatial context (blob vs bar) without pooling.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, rgb, depth, max_depth):
        depth_channel = (depth / max_depth).clamp(0.0, 1.0).unsqueeze(1)
        return torch.sigmoid(self.net(torch.cat([rgb, depth_channel], dim=1))).squeeze(1)

    def forward_logits(self, features):
        """Logits from the pre-concatenated 4-channel input (offline trainer contract)."""
        return self.net(features)


class SpatialTargetSegmenterWide(nn.Module):
    """9x9-receptive-field, 24-channel head (~11.3k params) for the appearance envelope.

    The confirmatory v6 run showed the 7x7/16ch head is capacity-bound under the declared
    envelope: no validation operating point reached the 0.95 recall gate (best 0.925), with the
    misses concentrated in hard frames (far/small/strong-blur draws). One extra 3x3 stage and
    +50% width buys context and capacity while staying trivially cheap for 128 parallel streams.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, kernel_size=1),
        )

    def forward(self, rgb, depth, max_depth):
        depth_channel = (depth / max_depth).clamp(0.0, 1.0).unsqueeze(1)
        return torch.sigmoid(self.net(torch.cat([rgb, depth_channel], dim=1))).squeeze(1)

    def forward_logits(self, features):
        """Logits from the pre-concatenated 4-channel input (offline trainer contract)."""
        return self.net(features)


def build_target_segmenter(architecture):
    """Map an artifact's meta.architecture tag to its module class (default: legacy 1x1)."""
    tag = str(architecture)
    # Longest prefix first: "SpatialTargetSegmenterWide" also startswith the narrow tag.
    if tag.startswith("SpatialTargetSegmenterWide"):
        return SpatialTargetSegmenterWide()
    if tag.startswith("SpatialTargetSegmenter"):
        return SpatialTargetSegmenter()
    return AppearanceTargetSegmenter()


class AppearanceTargetSegmenter(nn.Module):
    """Tiny learnable RGB-D pixel classifier with a usable red-target bootstrap.

    A trained checkpoint can replace the bootstrap weights through ``load_state_dict``.  The
    default is deterministic and recognizes appearance only; it has no access to simulator class
    IDs.  Keeping this head tiny makes 256 parallel camera streams practical on an 8 GB GPU.
    """

    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[0, 0, 0, 0] = 3.0
            self.classifier.weight[0, 1, 0, 0] = -2.0
            self.classifier.weight[0, 2, 0, 0] = -2.0
            self.classifier.bias.fill_(-0.9)

    def forward(self, rgb, depth, max_depth):
        depth_channel = (depth / max_depth).clamp(0.0, 1.0).unsqueeze(1)
        return torch.sigmoid(self.classifier(torch.cat([rgb, depth_channel], dim=1))).squeeze(1)

    def forward_logits(self, features):
        """Logits from the pre-concatenated 4-channel input (offline trainer contract)."""
        return self.classifier(features)


class BatchedConstantVelocityTracker:
    """GPU-vectorized 3-D constant-velocity Kalman filter."""

    def __init__(self, num_envs, device, dt, memory_s):
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)
        self.memory_s = float(memory_s)
        self.state = torch.zeros(self.num_envs, 6, device=device)
        self.cov = torch.eye(6, device=device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.age = torch.full((self.num_envs,), self.memory_s, device=device)

        self.F = torch.eye(6, device=device)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        self.H = torch.zeros(3, 6, device=device)
        self.H[:, :3] = torch.eye(3, device=device)
        self.I = torch.eye(6, device=device)
        q_pos = 0.025 * self.dt * self.dt
        q_vel = 0.25 * self.dt
        self.Q = torch.diag(torch.tensor([q_pos] * 3 + [q_vel] * 3, device=device))

    def reset_idx(self, env_ids):
        self.state[env_ids] = 0.0
        self.cov[env_ids] = torch.eye(6, device=self.device)
        self.active[env_ids] = False
        self.age[env_ids] = self.memory_s

    def step(self, measurement_world, visible, measurement_var):
        active = self.active
        if bool(active.any()):
            x = self.state[active]
            p = self.cov[active]
            self.state[active] = torch.matmul(x, self.F.t())
            self.cov[active] = self.F.unsqueeze(0).matmul(p).matmul(self.F.t()) + self.Q
            self.age[active] += self.dt

        new = visible & ~self.active
        if bool(new.any()):
            self.state[new, :3] = measurement_world[new]
            self.state[new, 3:] = 0.0
            p0 = torch.zeros(int(new.sum()), 6, 6, device=self.device)
            p0[:, :3, :3] = torch.diag_embed(measurement_var[new])
            p0[:, 3:, 3:] = torch.eye(3, device=self.device).unsqueeze(0) * 1.0
            self.cov[new] = p0
            self.active[new] = True
            self.age[new] = 0.0

        self.correct(measurement_world, visible & ~new, measurement_var)

        expired = self.active & (self.age > self.memory_s)
        self.active[expired] = False
        return self.state, self.cov, self.active, self.age

    def correct(self, measurement_world, visible, measurement_var, measurement_cov=None, reset_age=True):
        """Apply a second sensor correction without advancing time again.

        `measurement_cov` supplies a full 3x3 noise matrix instead of a diagonal one. A sensor
        that constrains only one direction -- a range along a known bearing, say -- has an
        anisotropic R, and passing it as a diagonal understates the uncertainty in the axes it
        never measured, shrinking their covariance for free.
        """
        update = visible & self.active
        if bool(update.any()):
            x = self.state[update]
            p = self.cov[update]
            z = measurement_world[update]
            innovation = z - x[:, :3]
            r = (
                measurement_cov[update]
                if measurement_cov is not None
                else torch.diag_embed(measurement_var[update])
            )
            s = p[:, :3, :3] + r
            k = p[:, :, :3].matmul(torch.inverse(s))
            self.state[update] = x + k.matmul(innovation.unsqueeze(2)).squeeze(2)
            ikh = self.I.unsqueeze(0) - k.matmul(self.H.unsqueeze(0))
            self.cov[update] = ikh.matmul(p).matmul(ikh.transpose(1, 2)) + k.matmul(r).matmul(
                k.transpose(1, 2)
            )
            if reset_age:
                self.age[update] = 0.0


class NavRLPerceptionModule:
    """RGB-D/LiDAR fusion, tracking, uncertainty, and 2-second history encoding."""

    def __init__(self, num_envs, device, cfg, step_dt, camera_cfg):
        self.num_envs = int(num_envs)
        self.device = device
        self.cfg = cfg
        self.step_dt = float(step_dt)
        self.max_camera_range = float(camera_cfg.detector_max_range)
        self.lidar_max_range = float(getattr(cfg, "lidar_max_range", 4.0))
        # Far plane of the obstacle depth camera: columns at this value saw NOTHING and must not
        # min-fuse into a longer-range LiDAR scan (see camera_no_return_to_lidar_range).
        self.camera_obstacle_max_range = float(
            getattr(camera_cfg, "camera_obstacle_max_range", self.lidar_max_range)
        )
        self.hfov = math.radians(float(camera_cfg.detector_hfov_deg))
        self.vfov = math.radians(float(camera_cfg.detector_vfov_deg))
        self.width = int(camera_cfg.camera_width)
        self.height = int(camera_cfg.camera_height)
        self.fx = self.width / (2.0 * math.tan(self.hfov * 0.5))
        self.fy = self.height / (2.0 * math.tan(self.vfov * 0.5))
        self.cx = (self.width - 1) * 0.5
        self.cy = (self.height - 1) * 0.5
        # DETECT resolution: where the target measurement comes from. Equal to the camera
        # resolution by default, in which case every expression below is the historical one on
        # the historical operands and `detect_decoupled` is False.
        self.detect_width = int(getattr(camera_cfg, "detect_width", self.width))
        self.detect_height = int(getattr(camera_cfg, "detect_height", self.height))
        self.detect_decoupled = (self.detect_width != self.width) or (
            self.detect_height != self.height
        )
        # WHICH INTRINSICS GO WHERE. fx = (W/2)/tan(hfov/2) is resolution-dependent, so:
        #   detect_fx/fy/cx/cy  -- anything derived from the DETECT-resolution mask: the target
        #       centroid -> bearing/elevation ray, the injected bearing-noise centroid offset
        #       (du = -fx*dbearing), and the lateral measurement sigma (surface_range/fx is the
        #       metres-per-pixel of the image the centroid was measured on).
        #   fx/fy/cx/cy         -- anything derived from the CAMERA-resolution image: the profile
        #       head's centroid comparison. (The obstacle-map column mapping and the target-pixel
        #       reconstruction use hfov and width directly, not fx, and stay camera-resolution.)
        # Using the camera fx on a detect-resolution centroid would scale every bearing by
        # width/detect_width -- a silent, systematic bearing bias, not noise.
        self.detect_fx = self.detect_width / (2.0 * math.tan(self.hfov * 0.5))
        self.detect_fy = self.detect_height / (2.0 * math.tan(self.vfov * 0.5))
        self.detect_cx = (self.detect_width - 1) * 0.5
        self.detect_cy = (self.detect_height - 1) * 0.5
        self.camera_offset = torch.tensor(
            camera_cfg.camera_translation, dtype=torch.float32, device=device
        ).view(1, 3)
        self.target_radius = float(camera_cfg.camera_target_radius)
        self.min_pixels = int(getattr(cfg, "min_target_pixels", 2))
        self.pixel_threshold = float(getattr(cfg, "pixel_threshold", 0.55))
        self.dropout_prob = float(getattr(cfg, "detection_dropout_prob", 0.0))
        self.detection_latency_s = float(getattr(cfg, "detection_latency_s", 0.0))
        self.range_error_m = float(getattr(cfg, "range_error_m", 0.0))
        # Depth noise model order (prereg_2026-09-04_depth_noise_model_order). Default "linear"
        # reproduces the pre-2026-09-04 sigma_r bit for bit; "stereo" swaps ONLY the linear range
        # term for Intel's quadratic stereo formula. sigma_lat is untouched in both modes -- it is
        # bearing error times range, so linear in r is structurally correct there.
        self.depth_noise_model = (
            str(getattr(cfg, "depth_noise_model", "linear") or "linear").strip().lower()
        )
        if self.depth_noise_model not in ("linear", "stereo"):
            raise ValueError(
                "depth_noise_model must be linear|stereo, got %r" % self.depth_noise_model
            )
        # Precompute D^2 * subpixel / (fx_sensor * baseline) as a single coefficient on r^2.
        self.depth_stereo_coeff = float(getattr(cfg, "depth_stereo_subpixel", 0.08)) / (
            float(getattr(cfg, "depth_stereo_fx_px", 447.0))
            * float(getattr(cfg, "depth_stereo_baseline_m", 0.095))
        )
        # Latency compensation (P0/P1, WORKLOG 2026-08-05). Both default off; both are no-ops
        # when detection_latency_s is 0, so a clean run is byte-identical with them set.
        self.latency_compensate = bool(getattr(cfg, "latency_compensate", False))
        self.latency_lidar_backup = bool(getattr(cfg, "latency_lidar_backup", False))
        # P2: how the obstacle map is edited while the detection is stale.
        self.latency_obstacle_fix = (
            str(getattr(cfg, "latency_obstacle_fix", "off") or "off").strip().lower()
        )
        if self.latency_obstacle_fix not in ("off", "predict", "skip"):
            raise ValueError(
                "latency_obstacle_fix must be off|predict|skip, got "
                f"{self.latency_obstacle_fix!r}"
            )
        # P3: lift the delayed measurement to world with the pose it was TAKEN at, not the
        # current one. Set alongside the ring-buffer read so the index math happens exactly once.
        self.latency_ego_motion_fix = bool(getattr(cfg, "latency_ego_motion_fix", False))
        # 검증 3: perturbations of P3's exact-timestamp/exact-pose premise.
        self.pose_clock_offset_s = float(getattr(cfg, "pose_clock_offset_s", 0.0))
        self.pose_noise_pos_m = float(getattr(cfg, "pose_noise_pos_m", 0.0))
        self.pose_noise_yaw_deg = float(getattr(cfg, "pose_noise_yaw_deg", 0.0))
        self.pose_noise_seed = int(getattr(cfg, "pose_noise_seed", 9163))
        self._pose_noise_generator = torch.Generator(device=device)
        self._pose_noise_generator.manual_seed(self.pose_noise_seed)
        self._pose_premise_active = (
            self.pose_clock_offset_s != 0.0
            or self.pose_noise_pos_m > 0.0
            or self.pose_noise_yaw_deg > 0.0
        )
        # 4-1 결합 진단: v7-shaped synthetic noise on the analytic detector's output.
        # See docs/prereg_2026-08-13_detector_coupling.md. Scale multiplies the two sigmas and the
        # miss-entry probability but NOT p10, so the dose ladder changes magnitude while keeping
        # the miss run-length distribution (the part iid noise gets wrong) fixed.
        _dn_scale = float(getattr(cfg, "detector_noise_scale", 1.0))
        self.detector_noise_bearing_std_rad = (
            float(getattr(cfg, "detector_noise_bearing_std_rad", 0.0)) * _dn_scale
        )
        self.detector_noise_range_std_m = (
            float(getattr(cfg, "detector_noise_range_std_m", 0.0)) * _dn_scale
        )
        self.detector_noise_dropout_p01 = min(
            1.0, float(getattr(cfg, "detector_noise_dropout_p01", 0.0)) * _dn_scale
        )
        self.detector_noise_dropout_p10 = float(getattr(cfg, "detector_noise_dropout_p10", 1.0))
        self.detector_noise_scale = _dn_scale
        # AR(1) + heteroscedastic range error, matched to the profiled structure. rho is NOT
        # scaled by the dose ladder: the ladder must vary magnitude only, leaving the correlation
        # structure fixed, or the rungs are three different noise families.
        self.detector_noise_range_rho = float(getattr(cfg, "detector_noise_range_rho", 0.0))
        self.detector_noise_range_bias_m = (
            float(getattr(cfg, "detector_noise_range_bias_m", 0.0)) * _dn_scale
        )
        self._detector_noise_range_ar = torch.zeros(self.num_envs, device=device)
        edges, mults = [], []
        for chunk in str(
            getattr(cfg, "detector_noise_range_sigma_profile", "") or ""
        ).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            hi, mult = chunk.split(":")
            edges.append(float(hi))
            mults.append(float(mult))
        self._detector_noise_sigma_edges = (
            torch.tensor(edges, device=device) if edges else None
        )
        self._detector_noise_sigma_mults = (
            torch.tensor(mults, device=device) if mults else None
        )
        bias_edges, bias_values = [], []
        for chunk in str(
            getattr(cfg, "detector_noise_range_bias_profile", "") or ""
        ).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            hi, value = chunk.split(":")
            bias_edges.append(float(hi))
            # The dose ladder scales systematic and random error together.
            bias_values.append(float(value) * _dn_scale)
        self._detector_noise_bias_edges = (
            torch.tensor(bias_edges, device=device) if bias_edges else None
        )
        self._detector_noise_bias_values = (
            torch.tensor(bias_values, device=device) if bias_values else None
        )
        self.detector_noise_seed = int(getattr(cfg, "detector_noise_seed", 9409))
        self._detector_noise_generator = torch.Generator(device=device)
        self._detector_noise_generator.manual_seed(self.detector_noise_seed)
        # Persistent Markov state: True = currently in a miss run. Not reset per episode -- the
        # chain models the detector, not the episode, and rewinding it on every reset would bias
        # the run-length distribution toward short runs.
        self._detector_noise_missing = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._detector_noise_active = (
            self.detector_noise_bearing_std_rad > 0.0
            or self.detector_noise_range_std_m > 0.0
            or self.detector_noise_dropout_p01 > 0.0
        )
        # Profiling: paired evaluation of a second detector on identical frames (analytic drives,
        # the profile head only observes). Populated by _detect_rgbd, written out at exit.
        self.detector_profile_records = []
        self._profile_segmenter = None
        self._detector_profile_out = str(
            os.environ.get("NAVRL_DETPROFILE_OUT", "") or ""
        ).strip()
        # Bounded so a long campaign cannot grow this without limit. 4,000 steps x num_envs is
        # already ~500k paired samples, far more than the quantiles and run-length histogram need.
        self._detector_profile_max_steps = int(
            os.environ.get("NAVRL_DETPROFILE_MAX_STEPS", "4000") or 4000
        )
        self._latency_delayed_pose = None
        # Backfill the target pixel mask from the fused bearing/range when the camera did not
        # deliver one but the track is alive (LiDAR-only frames). Off by default until measured.
        self.target_mask_backfill = bool(getattr(cfg, "target_mask_backfill", False))
        # H2 probe: LiDAR association is the ONLY target measurement on a camera-missed frame,
        # but its bearing is quantised to a 360/HBEAMS bin -- 5 deg here, which is 0.44 m of
        # lateral error at 5 m, larger than the ego-motion error P3 removed. Disabling it makes
        # the tracker coast on its constant-velocity prediction instead of correcting with a
        # coarse measurement, which is the A/B that decides whether the fallback helps or hurts.
        self.lidar_target_assoc = bool(getattr(cfg, "lidar_target_assoc", True))
        # H3: restrict the LiDAR correction to the direction it actually measures.
        self.lidar_range_only_update = bool(getattr(cfg, "lidar_range_only_update", False))
        # >0 replaces the covariance-scaled association gate with a constant, in metres.
        self.lidar_assoc_gate_m = float(getattr(cfg, "lidar_assoc_gate_m", 0.0))
        # H4 flag probe: keep the LiDAR range correction but stop it from reporting the target
        # as SEEN -- no age reset, no visibility, no confidence. Separates the state update
        # (measurably ~harmless: H3/gate arms) from the "visible, just seen" flags the policy
        # reads about a bearing the filter predicted itself.
        self.lidar_silent_correct = bool(getattr(cfg, "lidar_silent_correct", False))
        self.rgb_noise_std = float(getattr(cfg, "rgb_noise_std", 0.015))
        self.depth_noise_std = float(getattr(cfg, "depth_noise_std", 0.02))
        self.empirical_error = None
        empirical_model_path = str(getattr(cfg, "empirical_error_model", "") or "").strip()
        if empirical_model_path:
            incompatible = []
            if bool(getattr(cfg, "enable_perturbations", False)):
                incompatible.append("NAVRL_PERCEPTION_PERTURB")
            if self._detector_noise_active:
                incompatible.append("NAVRL_DETNOISE_*")
            if self.detection_latency_s != 0.0:
                incompatible.append("NAVRL_DETECTION_LATENCY_S")
            if self.range_error_m != 0.0:
                incompatible.append("NAVRL_RANGE_ERROR_M")
            if str(getattr(cfg, "detector_checkpoint", "") or "").strip():
                incompatible.append("NAVRL_DETECTOR_CHECKPOINT")
            if self._pose_premise_active:
                incompatible.append("NAVRL_POSE_{CLOCK,NOISE}_*")
            if incompatible:
                raise RuntimeError(
                    "P9 empirical error is mutually exclusive with: " + ", ".join(incompatible)
                )
            from aerial_gym.task.navrl_task.navrl_empirical_error import EmpiricalPerceptionError

            self.empirical_error = EmpiricalPerceptionError(
                empirical_model_path,
                getattr(cfg, "empirical_error_model_sha256", ""),
                self.num_envs,
                self.device,
                self.step_dt,
                getattr(cfg, "empirical_error_seed", 1701),
            )
        self.history_stride = max(1, int(round(float(cfg.history_interval_s) / self.step_dt)))
        self.step_count = 0
        self._latency_steps = max(0, int(round(self.detection_latency_s / self.step_dt)))
        # +3 keeps one slot of pose history on EACH side of the capture slot, so a clock-offset
        # read of up to one step in either direction never wraps onto overwritten data.
        empirical_latency_steps = (
            self.empirical_error.max_latency_steps if self.empirical_error is not None else 0
        )
        self._latency_slots = max(self._latency_steps, empirical_latency_steps) + 3
        self._latency_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._latency_meas_vehicle = torch.zeros(
            self.num_envs, self._latency_slots, 3, dtype=torch.float32, device=device
        )
        self._latency_surface_range = torch.zeros(
            self.num_envs, self._latency_slots, dtype=torch.float32, device=device
        )
        self._latency_bearing = torch.zeros(
            self.num_envs, self._latency_slots, dtype=torch.float32, device=device
        )
        self._latency_visible = torch.zeros(
            self.num_envs, self._latency_slots, dtype=torch.bool, device=device
        )
        self._latency_confidence = torch.zeros(
            self.num_envs, self._latency_slots, dtype=torch.float32, device=device
        )
        self._latency_mask = torch.zeros(
            self.num_envs,
            self._latency_slots,
            self.height,
            self.width,
            dtype=torch.bool,
            device=device,
        )
        # Pose of the observer at the moment each buffered detection was taken (P3). The quat
        # buffer holds identity rather than zeros so an unread slot still rotates validly.
        self._latency_drone_pos = torch.zeros(
            self.num_envs, self._latency_slots, 3, dtype=torch.float32, device=device
        )
        self._latency_drone_quat = torch.zeros(
            self.num_envs, self._latency_slots, 4, dtype=torch.float32, device=device
        )
        self._latency_drone_quat[..., 3] = 1.0
        self._env_ids = torch.arange(self.num_envs, device=device)

        self.segmenter = AppearanceTargetSegmenter().to(device).eval()
        checkpoint = str(getattr(cfg, "detector_checkpoint", "") or "").strip()
        if checkpoint:
            # The eval harness hashes the detector and exports the digest expecting it to be
            # checked here; until 2026-08-06 nothing read it, so every result recorded a
            # detector SHA that was never compared against the bytes actually loaded.
            expected_sha = os.environ.get("NAVRL_EXPECTED_DETECTOR_SHA256", "").strip().lower()
            if expected_sha:
                digest = hashlib.sha256()
                with open(checkpoint, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_sha = digest.hexdigest()
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        "detector checkpoint SHA mismatch: %s expected %s, loaded %s"
                        % (checkpoint, expected_sha, actual_sha)
                    )
            state = torch.load(checkpoint, map_location=device)
            architecture = ""
            if isinstance(state, dict):
                architecture = str((state.get("meta") or {}).get("architecture", ""))
            # Artifacts carry their architecture; a spatial checkpoint must not be forced into
            # the 1x1 head (strict load would fail loudly, but constructing the right class is
            # the contract, not the error message).
            self.segmenter = build_target_segmenter(architecture).to(device).eval()
            self.segmenter.load_state_dict(state.get("model", state), strict=True)

        # 4-1 profiling: a SECOND head evaluated on the same frame, observing only. Its outputs
        # never touch the tracker, the map or the observation -- the arm being profiled must
        # behave exactly as it does without profiling, or the errors are measured on a trajectory
        # the profiled detector partly caused.
        profile_ckpt = str(getattr(cfg, "detector_profile_checkpoint", "") or "").strip()
        if profile_ckpt:
            p_state = torch.load(profile_ckpt, map_location=device)
            p_arch = ""
            if isinstance(p_state, dict):
                p_arch = str((p_state.get("meta") or {}).get("architecture", ""))
            self._profile_segmenter = build_target_segmenter(p_arch).to(device).eval()
            self._profile_segmenter.load_state_dict(
                p_state.get("model", p_state), strict=True
            )
            if self._detector_profile_out:
                # atexit rather than an explicit drain: the evaluator owns the run loop and there
                # is no hook in it for "campaign finished", so tying the write to interpreter exit
                # is the only way to capture a full run without editing the run loop itself.
                import atexit

                atexit.register(self._dump_detector_profile)

        if self.detect_decoupled:
            self._assert_detect_decoupling_is_equivalent(cfg)

        self.tracker = BatchedConstantVelocityTracker(
            num_envs, device, step_dt, float(camera_cfg.tracker_memory_s)
        )
        # S1 shadow-mode counterfactual (docs/prereg_2026-09-03_s1_structure_fix_shadow.md).
        # Read-only beside the live path: its own tracker instance, its own diagnostics keys.
        self._s1_shadow = None
        self._s1_shadow_frame = None
        if os.environ.get("NAVRL_S1_SHADOW", "0").strip() == "1":
            from aerial_gym.task.navrl_task.shadow_association import ShadowAssociator

            if self.detect_decoupled:
                raise RuntimeError(
                    "NAVRL_S1_SHADOW requires detect resolution == camera resolution: the "
                    "shadow candidates are cut from the camera-resolution mask and a decoupled "
                    "detect stream would make the two paths measure different frames"
                )
            if float(getattr(camera_cfg, "detection_latency_s", 0.0)) != 0.0:
                raise RuntimeError(
                    "NAVRL_S1_SHADOW requires zero detection latency: the shadow path "
                    "classifies the current frame and a delayed live path would break "
                    "same-frame comparability"
                )
            self._s1_shadow = ShadowAssociator(
                BatchedConstantVelocityTracker(
                    num_envs, device, step_dt, float(camera_cfg.tracker_memory_s)
                ),
                camera_offset=self.camera_offset,
                target_radius=self.target_radius,
                min_pixels=self.min_pixels,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                detect_fx=self.detect_fx,
                dt=step_dt,
            )
        self.robot_history = torch.zeros(
            self.num_envs, ROBOT_HISTORY, ROBOT_DIM, device=device
        )
        self.target_history = torch.zeros(
            self.num_envs, TARGET_HISTORY, TARGET_DIM, device=device
        )
        self.obstacle_history = torch.zeros(
            self.num_envs, OBSTACLE_HISTORY, MAX_OBSTACLES, OBSTACLE_DIM, device=device
        )
        self.last_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.last_confidence = torch.zeros(self.num_envs, device=device)
        # Exact target-return association used to build the actor's obstacle representation.
        # Safety layers may reuse this sensor-derived mask; they must never inspect the simulator
        # semantic-ID buffer independently of the perception front-end.
        self.last_target_like = torch.zeros(
            self.num_envs, VBEAMS, HBEAMS, dtype=torch.bool, device=device
        )
        self.search = (
            SearchGrid(
                self.num_envs,
                arena_bounds=None,
                cell_m=2.0,
                device=device,
                search_state=SEARCH_STATE,
                step_dt=self.step_dt,
                camera_hfov_rad=self.hfov,
                camera_range_m=self.max_camera_range,
                depth_far_m=self.camera_obstacle_max_range,
                detection_probability=0.9,
                target_speed_prior_mps=1.25,
            )
            if SEARCH_DIM > 0
            else None
        )

        self._u = torch.arange(self.width, device=device, dtype=torch.float32).view(1, 1, -1)
        self._v = torch.arange(self.height, device=device, dtype=torch.float32).view(1, -1, 1)
        # Vehicle-frame bearing of each depth COLUMN -- the exact inverse of the lidar-angle ->
        # column map used below (camera_u = (hfov/2 - angle)/hfov * (width-1)). Only P2's
        # "predict" mode reads it, to rebuild the target pixel mask from the predicted bearing.
        self._pixel_angles = (
            self.hfov * 0.5
            - torch.arange(self.width, device=device, dtype=torch.float32)
            / max(self.width - 1, 1)
            * self.hfov
        )
        # Bearings of the horizontal beams. The span stops 360/HBEAMS short of a full turn so the
        # first and last ray are not the same direction (matching navrl_lidar_config's fov choice):
        # HBEAMS=36 -> [-170, 180] deg at 10 deg spacing, HBEAMS=72 -> [-175, 180] at 5 deg.
        self._bin_deg = 360.0 / HBEAMS
        # Physical bin bearings (DECREASING with index, matching the warp ray generator).
        # See lidar_bin_bearings() for the mirror-bug history and the adjudication probe.
        self._lidar_angles = lidar_bin_bearings(device)
        # Suppression window converted from degrees to bins (at least 1 so a token always blanks
        # its own bin, never selecting the same bearing twice).
        self._suppress_bins = max(1, int(round(OBSTACLE_SUPPRESS_DEG / self._bin_deg)))
        # Bearings eligible to become obstacle tokens (see OBSTACLE_FOV_DEG). None = all of them.
        if OBSTACLE_FOV_DEG < 359.9:
            self._token_bearing_mask = (
                self._lidar_angles.abs() <= math.radians(OBSTACLE_FOV_DEG * 0.5)
            )
        else:
            self._token_bearing_mask = None

    def reset_idx(self, env_ids):
        self.tracker.reset_idx(env_ids)
        if self._s1_shadow is not None:
            self._s1_shadow.reset_idx(env_ids)
        self.robot_history[env_ids] = 0.0
        self.target_history[env_ids] = 0.0
        self.obstacle_history[env_ids] = 0.0
        self.last_visible[env_ids] = False
        self.last_confidence[env_ids] = 0.0
        self.last_target_like[env_ids] = False
        if self.search is not None:
            self.search.reset(env_ids)
        if self.empirical_error is not None:
            self.empirical_error.reset_idx(env_ids)
        self._latency_step[env_ids] = 0
        self._latency_meas_vehicle[env_ids] = 0.0
        self._latency_surface_range[env_ids] = 0.0
        self._latency_bearing[env_ids] = 0.0
        self._latency_visible[env_ids] = False
        self._latency_confidence[env_ids] = 0.0
        self._latency_mask[env_ids] = False
        self._latency_drone_pos[env_ids] = 0.0
        self._latency_drone_quat[env_ids] = 0.0
        self._latency_drone_quat[env_ids, :, 3] = 1.0

    def _apply_detection_latency(
        self,
        measurement_vehicle,
        surface_range,
        bearing,
        visible,
        confidence,
        mask,
        drone_pos_w=None,
        vehicle_quat=None,
        latency_steps=None,
    ):
        self._latency_delayed_pose = None
        if latency_steps is None and self._latency_steps <= 0:
            return measurement_vehicle, surface_range, bearing, visible, confidence, mask

        if latency_steps is None:
            latency_steps = self._latency_steps
        elif self._latency_steps != 0:
            raise RuntimeError("P9 variable latency cannot be combined with fixed latency")

        write_idx = self._latency_step % self._latency_slots
        self._latency_meas_vehicle[self._env_ids, write_idx] = measurement_vehicle
        self._latency_surface_range[self._env_ids, write_idx] = surface_range
        self._latency_bearing[self._env_ids, write_idx] = bearing
        self._latency_visible[self._env_ids, write_idx] = visible
        self._latency_confidence[self._env_ids, write_idx] = confidence
        self._latency_mask[self._env_ids, write_idx] = mask
        if drone_pos_w is not None and vehicle_quat is not None:
            self._latency_drone_pos[self._env_ids, write_idx] = drone_pos_w
            self._latency_drone_quat[self._env_ids, write_idx] = vehicle_quat
        self._latency_step += 1
        ready = self._latency_step > latency_steps
        read_idx = (self._latency_step - latency_steps - 1) % self._latency_slots
        delayed_visible = self._latency_visible[self._env_ids, read_idx] & ready
        delayed_mask = self._latency_mask[self._env_ids, read_idx] & delayed_visible.view(
            -1, 1, 1
        )
        delayed_confidence = self._latency_confidence[self._env_ids, read_idx]
        # Published from the SAME read index as the measurement, so the pose and the detection
        # can never come from different steps. observe() consumes it only when P3 is enabled.
        if drone_pos_w is not None and vehicle_quat is not None:
            if self._pose_premise_active and isinstance(latency_steps, int):
                self._latency_delayed_pose = self._perturbed_capture_pose()
            else:
                self._latency_delayed_pose = (
                    self._latency_drone_pos[self._env_ids, read_idx],
                    self._latency_drone_quat[self._env_ids, read_idx],
                )
        return (
            self._latency_meas_vehicle[self._env_ids, read_idx],
            self._latency_surface_range[self._env_ids, read_idx],
            self._latency_bearing[self._env_ids, read_idx],
            delayed_visible,
            torch.where(
                delayed_visible,
                delayed_confidence,
                torch.zeros_like(confidence),
            ),
            delayed_mask,
        )

    def _perturbed_capture_pose(self):
        """The capture-time pose as imperfect hardware would deliver it (검증 3).

        Clock offset: the pose is read at capture-time + offset, with linear interpolation
        between the two bracketing odometry samples (positions lerped, quaternions
        sign-aligned nlerp -- inter-step rotations are small). offset = +tau lands on the
        CURRENT pose and therefore reproduces the naive pre-P3 transform exactly, which the
        sensitivity ladder uses as a built-in anchor. Odometry noise then perturbs the pose
        itself: gaussian position error per axis and a yaw error about world z.
        """
        offset_steps = self.pose_clock_offset_s / self.step_dt
        lo = math.floor(offset_steps)
        frac = offset_steps - lo
        newest = (self._latency_step - 1).clamp(min=0)
        base_abs = self._latency_step - self._latency_steps - 1
        a_abs = torch.minimum((base_abs + lo).clamp(min=0), newest)
        b_abs = torch.minimum((base_abs + lo + 1).clamp(min=0), newest)
        idx_a = a_abs % self._latency_slots
        idx_b = b_abs % self._latency_slots
        pos_a = self._latency_drone_pos[self._env_ids, idx_a]
        pos_b = self._latency_drone_pos[self._env_ids, idx_b]
        pos = torch.lerp(pos_a, pos_b, float(frac))
        quat_a = self._latency_drone_quat[self._env_ids, idx_a]
        quat_b = self._latency_drone_quat[self._env_ids, idx_b]
        quat_b = torch.where(
            (quat_a * quat_b).sum(dim=1, keepdim=True) < 0.0, -quat_b, quat_b
        )
        quat = torch.lerp(quat_a, quat_b, float(frac))
        quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9)
        if self.pose_noise_pos_m > 0.0:
            pos_noise = torch.randn(
                pos.shape,
                dtype=pos.dtype,
                device=pos.device,
                generator=self._pose_noise_generator,
            )
            pos = pos + pos_noise * self.pose_noise_pos_m
        if self.pose_noise_yaw_deg > 0.0:
            half = (
                torch.randn(
                    self.num_envs,
                    dtype=quat.dtype,
                    device=quat.device,
                    generator=self._pose_noise_generator,
                )
                * math.radians(self.pose_noise_yaw_deg)
                * 0.5
            )
            zeros = torch.zeros_like(half)
            noise = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=1)
            quat = _quat_mul_xyzw(noise, quat)
        return pos, quat

    def _detector_noise_range(self, surface_range):
        """AR(1), range-dependent range error matched to the profiled v7 statistics.

        The seed-419 profile measured lag-1 autocorrelation 0.644 and a std that varies 0.23-1.07 m
        with measured range. Injecting white homoscedastic noise with the same marginal would be a
        materially weaker perturbation: a constant-velocity Kalman filter averages independent
        errors away across frames, whereas a correlated error walks the track off and holds it
        there. Matching only the marginal would therefore under-reproduce v7 and leave a null
        result unable to distinguish "not coupling" from "my noise was too easy".

            e_t = rho * e_{t-1} + sqrt(1 - rho^2) * w_t,   w_t ~ N(0, sigma(range)^2)

        The sqrt(1 - rho^2) keeps the stationary variance equal to sigma^2, so rho changes the
        correlation without changing the marginal the profile pinned.
        """
        sigma = torch.full_like(surface_range, self.detector_noise_range_std_m)
        if self._detector_noise_sigma_edges is not None:
            # Piecewise-constant multiplier, selected by which profiled band the range falls in.
            idx = torch.bucketize(surface_range, self._detector_noise_sigma_edges)
            idx = idx.clamp(max=self._detector_noise_sigma_mults.numel() - 1)
            sigma = sigma * self._detector_noise_sigma_mults[idx]

        white = torch.randn(
            self.num_envs, device=self.device, dtype=surface_range.dtype,
            generator=self._detector_noise_generator,
        )
        rho = self.detector_noise_range_rho
        if rho > 0.0:
            self._detector_noise_range_ar = (
                rho * self._detector_noise_range_ar
                + math.sqrt(max(0.0, 1.0 - rho * rho)) * white
            )
            unit = self._detector_noise_range_ar
        else:
            unit = white
        bias = torch.full_like(surface_range, self.detector_noise_range_bias_m)
        if self._detector_noise_bias_edges is not None:
            # Select from the CLEAN analytic range, exactly as the profile bins were defined.
            # A separate edge vector is intentional: it makes the contract explicit and prevents
            # a future sigma-profile edit from silently changing which mean is injected.
            bias_idx = torch.bucketize(surface_range, self._detector_noise_bias_edges)
            bias_idx = bias_idx.clamp(max=self._detector_noise_bias_values.numel() - 1)
            bias = self._detector_noise_bias_values[bias_idx]
        return (surface_range + unit * sigma + bias).clamp(0.0, self.max_camera_range)

    def _detector_noise_visibility(self, visible):
        """Two-state Markov miss process on the detection flag.

        iid Bernoulli dropout with the right marginal still gets the WRONG temporal structure: a
        real detector that loses the target holds the miss for several frames, and the tracker's
        response to one isolated miss is nothing like its response to a four-frame run (it coasts
        on constant velocity while the covariance grows, and hands the frame to the LiDAR
        fallback). Matching only the marginal would therefore under-reproduce the effect and make
        a null result uninterpretable -- the same failure that made verification 3's step-wise iid
        Gaussian an incomplete odometry model.

        p01 = P(enter a miss | currently seen); p10 = P(leave a miss | currently missing).
        Stationary miss rate p01/(p01+p10), mean miss run-length 1/p10.
        """
        draw = torch.rand(
            self.num_envs, device=self.device,
            generator=self._detector_noise_generator,
        )
        was_missing = self._detector_noise_missing
        enter = (~was_missing) & (draw < self.detector_noise_dropout_p01)
        stay = was_missing & (draw >= self.detector_noise_dropout_p10)
        self._detector_noise_missing = enter | stay
        return visible & ~self._detector_noise_missing

    def _dump_detector_profile(self):
        """Write the paired-detector records as one npz. Registered with atexit; must never raise
        into the interpreter shutdown path, so failures are reported and swallowed."""
        try:
            records = self.detector_profile_records
            if not records or not self._detector_profile_out:
                return
            import numpy as np

            keys = list(records[0].keys())
            arrays = {
                # (steps, num_envs) so the run-length analysis can walk the time axis per env.
                k: torch.stack([r[k] for r in records]).numpy() for k in keys
            }
            path = pathlib.Path(self._detector_profile_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            print(
                f"[detprofile] wrote {path} "
                f"({len(records)} steps x {self.num_envs} envs)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - atexit must not raise
            print(f"[detprofile] dump failed: {exc}", flush=True)

    def _record_detector_profile(self, rgb, depth, mask, count, visible):
        """Score the profile head on the same frame and store its error against the live head.

        Runs strictly read-only: nothing here feeds the tracker, the map or the observation, so
        the trajectory stays exactly the one the live detector produces. That is deliberate --
        profiling v7 on a v7-driven trajectory would measure its error on states it partly caused,
        and the injected-noise arms are driven by the analytic detector, so the analytic-driven
        trajectory is the matched reference. The cost is a covariate-shift caveat, recorded in the
        profile summary rather than silently ignored.
        """
        with torch.no_grad():
            p_score = self._profile_segmenter(rgb, depth, self.max_camera_range)
        p_mask = (p_score >= self.pixel_threshold) & (depth < self.max_camera_range)
        p_count = p_mask.sum(dim=(1, 2))
        p_visible = p_count >= self.min_pixels
        p_mask = p_mask & p_visible.view(-1, 1, 1)

        def _centroid_range(m, c):
            denom = c.clamp(min=1).float()
            mf = m.float()
            return (
                (mf * self._u).sum(dim=(1, 2)) / denom,
                (mf * self._v).sum(dim=(1, 2)) / denom,
                (depth * mf).sum(dim=(1, 2)) / denom,
            )

        ref_mask = mask & visible.view(-1, 1, 1)
        u_a, v_a, r_a = _centroid_range(ref_mask, count)
        u_b, v_b, r_b = _centroid_range(p_mask, p_count)
        # Small-angle bearing from the pixel centroid: atan(-(u - cx) / fx).
        bearing_a = torch.atan(-(u_a - self.cx) / self.fx)
        bearing_b = torch.atan(-(u_b - self.cx) / self.fx)

        both = visible & p_visible
        self.detector_profile_records.append({
            "ref_visible": visible.detach().to("cpu", torch.bool).clone(),
            "profile_visible": p_visible.detach().to("cpu", torch.bool).clone(),
            "both_visible": both.detach().to("cpu", torch.bool).clone(),
            "bearing_err": (bearing_b - bearing_a).detach().to("cpu", torch.float32).clone(),
            "range_err": (r_b - r_a).detach().to("cpu", torch.float32).clone(),
            "ref_range": r_a.detach().to("cpu", torch.float32).clone(),
            "ref_count": count.detach().to("cpu", torch.int32).clone(),
            "profile_count": p_count.detach().to("cpu", torch.int32).clone(),
        })

    def _assert_detect_decoupling_is_equivalent(self, cfg):
        """Refuse the decoupling whenever "segment the high-res RGB" stops equalling "read the
        high-res target mask".

        The decoupling rests on one claim: with the renderer's flat target paint and the
        bootstrap colour rule, segmenting a high-resolution render selects EXACTLY the target
        pixels the ray-cast already found, and nothing else. Everything that can falsify that is
        refused here rather than approximated.

        Refused:
          detector_checkpoint            a learned head is not the flat-colour rule; nothing
                                         proves it fires on exactly the painted pixels (and a
                                         spatial head's answer depends on the neighbourhood,
                                         which a per-frame scalar summary cannot carry at all).
          detector_profile_checkpoint    the profile head scores the CAMERA-resolution image; a
                                         paired error against a detect-resolution reference is
                                         two different measurements, not a comparison.
          a non-bootstrap / depth-weighted 1x1 head
                                         the target-pixel score must be one number per env; a
                                         non-zero depth weight makes it vary per pixel.
          rgb_noise_std / depth_noise_std with perturbations enabled
                                         per-PIXEL iid noise flips individual pixels across the
                                         decision boundary. A high-resolution mask can only
                                         reproduce that statistically, never frame-for-frame,
                                         and the flip rate itself depends on the pixel count.
          detection_latency_s > 0        the latency ring buffers the camera-resolution pixel
                                         mask; the detect-resolution count/centroid would have
                                         to be delayed in lockstep, which is not implemented.

        Checked and ALLOWED (they act on per-env scalars, identically at either resolution):
          detection_dropout_prob, range_error_m, detector_noise_* (bearing noise is injected as
          a centroid offset scaled by detect_fx, so it stays an angle), latency_* compensation
          flags at zero latency, lidar_* association flags, target_mask_backfill (it rebuilds a
          CAMERA-resolution pixel mask from the fused bearing/range, and simply gets a better
          bearing), pixel_threshold, min_target_pixels.
        """
        offenders = []
        if str(getattr(cfg, "detector_checkpoint", "") or "").strip():
            offenders.append("detector_checkpoint is set")
        if str(getattr(cfg, "detector_profile_checkpoint", "") or "").strip():
            offenders.append("detector_profile_checkpoint is set")
        if not isinstance(self.segmenter, AppearanceTargetSegmenter):
            offenders.append(
                "segmenter is %s, not the flat-colour bootstrap head"
                % type(self.segmenter).__name__
            )
        else:
            depth_weight = float(self.segmenter.classifier.weight[0, 3].abs().max())
            if depth_weight > 0.0:
                offenders.append(
                    "segmenter depth-channel weight is %g (must be 0 so the target-pixel score "
                    "is one number per env)" % depth_weight
                )
        if bool(getattr(cfg, "enable_perturbations", False)):
            if self.rgb_noise_std > 0.0:
                offenders.append("rgb_noise_std=%g with perturbations on" % self.rgb_noise_std)
            if self.depth_noise_std > 0.0:
                offenders.append(
                    "depth_noise_std=%g with perturbations on" % self.depth_noise_std
                )
        if self._latency_steps > 0:
            offenders.append("detection_latency_s=%g" % self.detection_latency_s)
        if offenders:
            raise RuntimeError(
                "NavRL detect-resolution decoupling (%dx%d detect vs %dx%d camera) is NOT "
                "equivalent under: %s. Set NAVRL_DETECT_WIDTH/HEIGHT equal to "
                "NAVRL_CAMERA_WIDTH/HEIGHT, or turn the listed perturbation(s) off."
                % (
                    self.detect_width,
                    self.detect_height,
                    self.width,
                    self.height,
                    "; ".join(offenders),
                )
            )

    def _consume_detect_frame(self):
        """Take this step's detect-resolution frame from the renderer channel and validate it."""
        from aerial_gym.task.navrl_task.navrl_detector import DETECT_CHANNEL

        frame = DETECT_CHANNEL.consume()
        if (
            int(frame["width"]) != self.detect_width
            or int(frame["height"]) != self.detect_height
        ):
            raise RuntimeError(
                "detect-resolution frame is %dx%d but this perception module expects %dx%d"
                % (frame["width"], frame["height"], self.detect_width, self.detect_height)
            )
        if int(frame["num_envs"]) != self.num_envs:
            raise RuntimeError(
                "detect-resolution frame has %d envs, expected %d"
                % (frame["num_envs"], self.num_envs)
            )
        if float(frame["far_plane"]) != self.max_camera_range:
            # The camera-resolution mask applies `depth < max_camera_range` per pixel. The
            # detect-resolution mask gets that for free ONLY because the renderer's far plane is
            # the same number: a target pixel's depth is strictly below it by construction.
            raise RuntimeError(
                "detect-resolution renderer far plane %g != perception max_camera_range %g"
                % (frame["far_plane"], self.max_camera_range)
            )
        return frame

    def _detect_rgbd(self, rgb, depth, training, drone_pos_w=None, vehicle_quat=None):
        if (
            self.detect_decoupled
            and training
            and (self.rgb_noise_std > 0.0 or self.depth_noise_std > 0.0)
        ):
            # Backstop for a caller that passes training=True while cfg.enable_perturbations is
            # off (the __init__ guard cannot see that). Per-pixel image noise has no
            # detect-resolution counterpart; see _assert_detect_decoupling_is_equivalent.
            raise RuntimeError(
                "detect-resolution decoupling is not equivalent under per-pixel image noise "
                "(rgb_noise_std=%g, depth_noise_std=%g) with training=True"
                % (self.rgb_noise_std, self.depth_noise_std)
            )
        if training and self.rgb_noise_std > 0.0:
            rgb = (rgb + torch.randn_like(rgb) * self.rgb_noise_std).clamp(0.0, 1.0)
        if training and self.depth_noise_std > 0.0:
            depth = (depth + torch.randn_like(depth) * self.depth_noise_std).clamp(
                0.0, self.max_camera_range
            )
        with torch.no_grad():
            score = self.segmenter(rgb, depth, self.max_camera_range)
        mask = (score >= self.pixel_threshold) & (depth < self.max_camera_range)
        if self._s1_shadow is not None:
            # Stash BEFORE the live path's in-place `mask &= visible` gating below, so the
            # shadow candidates see the raw threshold mask of this exact frame. clone() keeps
            # the shadow strictly read-only with respect to the live tensor.
            self._s1_shadow_frame = (mask.clone(), depth)
        count = mask.sum(dim=(1, 2))
        visible = count >= self.min_pixels
        if (
            self._profile_segmenter is not None
            and len(self.detector_profile_records) < self._detector_profile_max_steps
        ):
            self._record_detector_profile(rgb, depth, mask, count, visible)
        detect_frame = None
        if self.detect_decoupled:
            # The target measurement moves to the detect-resolution mask. `mask` above stays the
            # CAMERA-resolution one and keeps its jobs: it is what gets blanked out of the
            # obstacle depth map, and what the latency ring buffers. Only count/centroid/range
            # (and therefore visibility, bearing, elevation and confidence) come from the frame.
            detect_frame = self._consume_detect_frame()
            # Perception is not told what a target is: it runs its OWN segmenter, at its own
            # threshold, on the RGB value a target pixel of that frame carries. The value is one
            # number per env only because the fail-closed guard has ruled out every knob that
            # would make the paint non-flat.
            with torch.no_grad():
                detect_score = self.segmenter(
                    detect_frame["rgb"].view(self.num_envs, 3, 1, 1),
                    detect_frame["depth_probe"].view(self.num_envs, 1, 1),
                    self.max_camera_range,
                ).view(self.num_envs)
            # `depth < max_camera_range` needs no separate test: a detect-resolution target pixel
            # is strictly inside the renderer far plane by construction, and _consume_detect_frame
            # has already checked that far plane equals max_camera_range.
            count = torch.where(
                detect_score >= self.pixel_threshold,
                detect_frame["count"],
                torch.zeros_like(detect_frame["count"]),
            )
            visible = count >= self.min_pixels
        if training and self.dropout_prob > 0.0:
            visible &= torch.rand(self.num_envs, device=self.device) >= self.dropout_prob
        if self._detector_noise_active:
            visible = self._detector_noise_visibility(visible)
        denom = count.clamp(min=1).float()
        empirical_latency_steps = None
        if self.empirical_error is not None:
            clean_visible = visible.clone()
            if detect_frame is not None:
                bbox = detect_frame["bbox"]
                bbox_width = (bbox[:, 2] - bbox[:, 0] + 1.0).clamp(min=0.0)
                bbox_height = (bbox[:, 3] - bbox[:, 1] + 1.0).clamp(min=0.0)
                clean_u = detect_frame["u_sum"] / denom
                clean_v = detect_frame["v_sum"] / denom
                clean_surface_range = detect_frame["depth_sum"] / denom
            else:
                raw_mf = mask.float()
                clean_u = (raw_mf * self._u).sum(dim=(1, 2)) / denom
                clean_v = (raw_mf * self._v).sum(dim=(1, 2)) / denom
                clean_surface_range = (depth * raw_mf).sum(dim=(1, 2)) / denom
                occupied_cols = mask.any(dim=1)
                occupied_rows = mask.any(dim=2)
                any_pixel = count > 0
                u_min = occupied_cols.float().argmax(dim=1)
                u_max = self.width - 1 - occupied_cols.float().flip(1).argmax(dim=1)
                v_min = occupied_rows.float().argmax(dim=1)
                v_max = self.height - 1 - occupied_rows.float().flip(1).argmax(dim=1)
                bbox_width = torch.where(any_pixel, u_max - u_min + 1.0, 0.0)
                bbox_height = torch.where(any_pixel, v_max - v_min + 1.0, 0.0)
            size_px = (bbox_width * bbox_height).clamp(min=0.0).sqrt()
            empirical_state, empirical_offset, empirical_latency_steps = (
                self.empirical_error.sample(size_px, clean_visible)
            )
            # FALSE_LOCK remains a policy-facing measurement, but it is not the target mask and
            # therefore must not carve the real target out of the obstacle depth map.
            visible = clean_visible & (empirical_state != 2)
            map_visible = clean_visible & (empirical_state == 0)
            mask &= map_visible.view(-1, 1, 1)
            gate = visible.float()
            u = (clean_u + empirical_offset[:, 0] * self.detect_width).clamp(
                0.0, float(self.detect_width - 1)
            ) * gate
            v = (clean_v + empirical_offset[:, 1] * self.detect_height).clamp(
                0.0, float(self.detect_height - 1)
            ) * gate
            surface_range = clean_surface_range * gate
            if detect_frame is not None:
                self._last_detect_count = torch.where(
                    visible, count, torch.zeros_like(count)
                )
        else:
            mask &= visible.view(-1, 1, 1)
            if detect_frame is not None:
                # Same gating as the camera-resolution path, where `mask &= visible` above zeroes
                # every sum for an env that is not visible.
                gate = visible.float()
                self._last_detect_count = torch.where(
                    visible, count, torch.zeros_like(count)
                )
                u = gate * detect_frame["u_sum"] / denom
                v = gate * detect_frame["v_sum"] / denom
                surface_range = gate * detect_frame["depth_sum"] / denom
            else:
                mf = mask.float()
                u = (mf * self._u).sum(dim=(1, 2)) / denom
                v = (mf * self._v).sum(dim=(1, 2)) / denom
                surface_range = (depth * mf).sum(dim=(1, 2)) / denom
        if training and self.range_error_m != 0.0:
            surface_range = (surface_range + self.range_error_m).clamp(
                0.0, self.max_camera_range
            )
        if self._detector_noise_active:
            # Bearing error is injected as a CENTROID error, not by rewriting `bearing` after the
            # fact: bearing is derived from u below via the ray, and writing it directly would
            # desync it from measurement_vehicle -- the KF would see the clean position while only
            # the obstacle-map carve-out saw the perturbed angle. Perturbing u moves both, which is
            # also what a real detector error does.
            #   bearing ~= atan(-(u - cx) / fx)  ->  d(bearing)/du ~= -1/fx  ->  du = -fx * dbearing
            if self.detector_noise_bearing_std_rad > 0.0:
                d_bearing = torch.randn(
                    self.num_envs, device=self.device, dtype=u.dtype,
                    generator=self._detector_noise_generator,
                ) * self.detector_noise_bearing_std_rad
                u = u - self.detect_fx * d_bearing
            if self.detector_noise_range_std_m > 0.0:
                surface_range = self._detector_noise_range(surface_range)
        if detect_frame is not None:
            # Mean segmenter score over the mask, which is that one score wherever the mask is.
            confidence = gate * detect_score
        elif self.empirical_error is not None:
            confidence = gate * (score * raw_mf).sum(dim=(1, 2)) / denom
        else:
            confidence = (score * mf).sum(dim=(1, 2)) / denom
        confidence *= (count.float() / max(1.0, float(self.min_pixels * 4))).clamp(max=1.0)
        confidence = torch.where(visible, confidence, torch.zeros_like(confidence))

        # Detect-resolution centroid -> detect-resolution intrinsics (identical floats to
        # self.cx/fx when the resolutions are equal).
        ray = torch.stack(
            [
                torch.ones_like(u),
                -(u - self.detect_cx) / self.detect_fx,
                -(v - self.detect_cy) / self.detect_fy,
            ],
            dim=1,
        )
        ray = ray / ray.norm(dim=1, keepdim=True).clamp(min=1e-6)
        center_range = surface_range + self.target_radius
        measurement_vehicle = self.camera_offset + ray * center_range.unsqueeze(1)
        bearing = torch.atan2(measurement_vehicle[:, 1], measurement_vehicle[:, 0])
        return self._apply_detection_latency(
            measurement_vehicle,
            surface_range,
            bearing,
            visible,
            confidence,
            mask,
            drone_pos_w=drone_pos_w,
            vehicle_quat=vehicle_quat,
            latency_steps=empirical_latency_steps,
        )

    def _latency_corrected_map_inputs(
        self,
        depth,
        target_pixels,
        target_surface_range,
        target_bearing,
        visible,
        drone_pos_w,
        vehicle_quat,
    ):
        """P2: fix WHERE the obstacle map gets the target carved out of it under latency.

        `_fuse_static_and_extract_obstacles` deletes sensor returns that look like the target --
        LiDAR bins inside the target_like window, and depth pixels inside `target_pixels` -- so
        that the target is never proposed as an obstacle. Every one of those inputs is DELAYED,
        so under latency the deletion happens at the target's OLD bearing/range, where a real bar
        may stand: the map loses that bar and the drone flies into it. This is the channel that
        tripled bar contacts (337 -> 931) in the 0.1 s arm even though LiDAR itself is undelayed,
        and neither P0 (policy-facing output only) nor P1 (association gate) touched it.

        "predict" rebuilds the deletion window from the tracker's forward-predicted target pose,
        i.e. deletes at where the target is NOW. "skip" stops deleting while the detection is
        stale: the target survives in the map as a phantom obstacle (costing capture), but no
        real bar is ever erased (bounding crash). Returns the inputs unchanged when off or when
        there is no latency, so clean runs are bit-identical.
        """
        if self.latency_obstacle_fix == "off" or self._latency_steps <= 0:
            return target_pixels, target_surface_range, target_bearing, visible
        if self.latency_obstacle_fix == "skip":
            # visible=False disables both the target_like carve-out and (via the empty pixel
            # mask) the depth blanking, without touching the fused scan geometry itself.
            return (
                torch.zeros_like(target_pixels),
                target_surface_range,
                target_bearing,
                torch.zeros_like(visible),
            )

        # "predict": same window, relocated to the tracker's estimate of the target NOW. This is
        # the P0 extrapolation reused on the map path -- deliberately the same expression, so the
        # two compensations cannot disagree about where the target is.
        state = self.tracker.state
        pred_pos_w = state[:, :3] + state[:, 3:] * self.detection_latency_s
        rel = _quat_rotate_inverse_xyzw(vehicle_quat, pred_pos_w - drone_pos_w)
        pred_bearing = torch.atan2(rel[:, 1], rel[:, 0])
        # surface_range is measured from the CAMERA to the near face, matching _detect_rgbd's
        # center_range = surface_range + target_radius decomposition.
        pred_surface = (rel - self.camera_offset).norm(dim=1) - self.target_radius
        pred_surface = pred_surface.clamp(min=0.0)
        # Only relocate where the tracker actually has a target; elsewhere carve nothing rather
        # than carve at an arbitrary place.
        pred_visible = visible & self.tracker.active
        pred_pixels = self._reconstruct_target_pixels(
            depth, pred_bearing, pred_surface, pred_visible
        )
        return pred_pixels, pred_surface, pred_bearing, pred_visible

    def _reconstruct_target_pixels(self, depth, bearing, surface_range, gate):
        """Which depth pixels the target would occupy at a given bearing/range.

        Uses the same agreement window as the LiDAR target_like carve-out, so the two halves of
        the obstacle map agree about what counts as the target.
        """
        pixel_delta = torch.atan2(
            torch.sin(self._pixel_angles.view(1, 1, -1) - bearing.view(-1, 1, 1)),
            torch.cos(self._pixel_angles.view(1, 1, -1) - bearing.view(-1, 1, 1)),
        ).abs()
        return (
            gate.view(-1, 1, 1)
            & (pixel_delta < TARGET_LIKE_ANGLE_RAD)
            & ((depth - surface_range.view(-1, 1, 1)).abs() < TARGET_LIKE_RANGE_TOL_M)
        )

    def _fuse_static_and_extract_obstacles(
        self,
        lidar_m,
        raw_depth,
        target_pixels,
        target_surface_range,
        target_bearing,
        visible,
        drone_vel_w=None,
        vehicle_quat=None,
    ):
        scan = lidar_m.view(self.num_envs, VBEAMS, HBEAMS).clone()

        # Camera-associated LiDAR returns are target evidence, not static obstacles. No semantic
        # ID is consulted: association uses only bearing and metric range agreement.
        angle_delta = torch.atan2(
            torch.sin(self._lidar_angles.view(1, 1, HBEAMS) - target_bearing.view(-1, 1, 1)),
            torch.cos(self._lidar_angles.view(1, 1, HBEAMS) - target_bearing.view(-1, 1, 1)),
        ).abs()
        target_like = (
            visible.view(-1, 1, 1)
            & (angle_delta < TARGET_LIKE_ANGLE_RAD)
            & ((scan - target_surface_range.view(-1, 1, 1)).abs() < TARGET_LIKE_RANGE_TOL_M)
        )
        self.last_target_like[:] = target_like
        scan = torch.where(target_like, torch.full_like(scan, self.lidar_max_range), scan)

        # Fuse forward RGB-D geometry into the 360-degree LiDAR distance representation.
        depth_no_target = torch.where(
            target_pixels, torch.full_like(raw_depth, self.lidar_max_range), raw_depth
        )
        camera_col_min = camera_no_return_to_lidar_range(
            depth_no_target.amin(dim=1), self.camera_obstacle_max_range, self.lidar_max_range
        ).clamp(0.0, self.lidar_max_range)
        inside = self._lidar_angles.abs() <= self.hfov * 0.5
        camera_u = ((self.hfov * 0.5 - self._lidar_angles) / self.hfov * (self.width - 1)).round()
        camera_u = camera_u.long().clamp(0, self.width - 1)
        camera_ranges = camera_col_min[:, camera_u]
        fused_camera = camera_ranges.unsqueeze(1).expand(-1, VBEAMS, -1)
        scan[:, :, inside] = torch.minimum(scan[:, :, inside], fused_camera[:, :, inside])
        static_state = (scan / self.lidar_max_range).clamp(0.0, 1.0)

        # Nearest angularly separated obstacle proposals. Bars are static, hence velocity=0;
        # position and covariance come from range/angle geometry rather than simulator positions.
        nearest = scan.amin(dim=1)
        # Diagnostics only (no grad, no copy cost on the hot path): the per-bearing obstacle range
        # actually seen this step. The bar_contact probe uses it to measure how crowded the scene was
        # at the moment of a collision -- i.e. whether MAX_OBSTACLES truncated away the bar that hit.
        self.last_scan_nearest = nearest
        tokens = torch.zeros(
            self.num_envs, MAX_OBSTACLES, OBSTACLE_DIM, device=self.device
        )
        rows = torch.arange(self.num_envs, device=self.device)
        if OBSTACLE_SELECTOR == "ttc_sector":
            # Body-frame planar velocity drives the closing-rate ranking. Same rotation the robot
            # observation uses below, so the two views of the drone's motion cannot disagree.
            if vehicle_quat is None or drone_vel_w is None:
                raise ValueError(
                    "ttc_sector requires vehicle_quat and drone_vel_w for closing-rate ranking"
                )
            body_vel = _quat_rotate_inverse_xyzw(vehicle_quat, drone_vel_w)
            selected_ranges, selected_indices, selected_valid = select_ttc_obstacles(
                nearest,
                self._lidar_angles,
                body_vel[:, :2],
                max_range=self.lidar_max_range,
                max_obstacles=MAX_OBSTACLES,
                cluster_gap_m=OBSTACLE_CLUSTER_GAP_M,
                idle_ttc_s=OBSTACLE_TTC_IDLE_S,
                min_speed=OBSTACLE_TTC_MIN_SPEED,
            )
        elif OBSTACLE_SELECTOR == "cluster_sector":
            selected_ranges, selected_indices, selected_valid = (
                select_cluster_sector_obstacles(
                    nearest,
                    self._lidar_angles,
                    max_range=self.lidar_max_range,
                    max_obstacles=MAX_OBSTACLES,
                    token_fov_deg=OBSTACLE_FOV_DEG,
                    cluster_gap_m=OBSTACLE_CLUSTER_GAP_M,
                    num_sectors=OBSTACLE_SECTORS,
                )
            )
        else:
            work = nearest.clone()
            if self._token_bearing_mask is not None:
                # Blank bearings outside the token sector so they can never win a slot. Applied to
                # the SELECTION copy only -- `static_state` still encodes the full 360-degree scan.
                work = work.masked_fill(
                    ~self._token_bearing_mask.view(1, -1), self.lidar_max_range
                )
            selected_ranges = torch.full(
                (self.num_envs, MAX_OBSTACLES), self.lidar_max_range, device=self.device
            )
            selected_indices = torch.zeros(
                (self.num_envs, MAX_OBSTACLES), dtype=torch.long, device=self.device
            )
            selected_valid = torch.zeros(
                (self.num_envs, MAX_OBSTACLES), dtype=torch.bool, device=self.device
            )
            for slot in range(MAX_OBSTACLES):
                r, idx = work.min(dim=1)
                valid = r < self.lidar_max_range * 0.995
                selected_ranges[:, slot] = r
                selected_indices[:, slot] = idx
                selected_valid[:, slot] = valid
                for off in range(-self._suppress_bins, self._suppress_bins + 1):
                    work[rows, (idx + off) % HBEAMS] = self.lidar_max_range

        for slot in range(MAX_OBSTACLES):
            r = selected_ranges[:, slot]
            idx = selected_indices[:, slot]
            valid = selected_valid[:, slot]
            a = self._lidar_angles[idx]
            pos = torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], dim=1)
            radial_sigma = 0.04 + 0.02 * r
            lateral_sigma = 0.05 + r * math.tan(math.radians(5.0))
            cov = torch.stack(
                [radial_sigma.square(), lateral_sigma.square(), torch.full_like(r, 0.12)], dim=1
            )
            feat = torch.cat(
                [
                    pos / self.lidar_max_range,
                    torch.zeros_like(pos),
                    torch.full_like(r.unsqueeze(1), 0.30 / self.lidar_max_range),
                    (1.0 - r / self.lidar_max_range).clamp(0.0, 1.0).unsqueeze(1),
                    cov / (self.lidar_max_range * self.lidar_max_range),
                    torch.ones_like(r.unsqueeze(1)),
                ],
                dim=1,
            )
            tokens[:, slot] = feat * valid.unsqueeze(1)
        return static_state.reshape(self.num_envs, -1), tokens

    def _associate_lidar_target(self, lidar_m, drone_pos_w, vehicle_quat, camera_visible):
        """Associate a raw LiDAR return with the predicted camera-initialized target track."""
        rel = _quat_rotate_inverse_xyzw(
            vehicle_quat, self.tracker.state[:, :3] - drone_pos_w
        )
        predicted_center_range = rel.norm(dim=1).clamp(min=1e-6)
        predicted_surface_range = (predicted_center_range - self.target_radius).clamp(min=0.0)
        bearing = torch.atan2(rel[:, 1], rel[:, 0])
        angular_error = torch.atan2(
            torch.sin(self._lidar_angles.view(1, -1) - bearing.unsqueeze(1)),
            torch.cos(self._lidar_angles.view(1, -1) - bearing.unsqueeze(1)),
        ).abs()
        h_idx = angular_error.argmin(dim=1)
        raw_scan = lidar_m.view(self.num_envs, VBEAMS, HBEAMS)
        rows = torch.arange(self.num_envs, device=self.device)
        ray_ranges = raw_scan[rows, :, h_idx]
        measured_surface, _ = ray_ranges.min(dim=1)
        pos_sigma = torch.diagonal(self.tracker.cov[:, :3, :3], dim1=1, dim2=2).sum(
            dim=1
        ).sqrt()
        gate = (0.35 + 2.0 * pos_sigma).clamp(max=1.0)
        # Scaling the association window by the track covariance means an honest covariance buys
        # a WIDER mis-association window: with the range-only update the gate hits its 1.0 m cap
        # within five blind steps, against ~0.67 m when the covariance was frozen. A gate is a
        # statement about measurement precision, not about how lost the track is, so allow a
        # constant one (WORKLOG 2026-08-07).
        if self.lidar_assoc_gate_m > 0.0:
            gate = torch.full_like(gate, self.lidar_assoc_gate_m)
        valid = (
            self.tracker.active
            & ~camera_visible
            & (predicted_surface_range < self.lidar_max_range)
            & (measured_surface < self.lidar_max_range * 0.995)
            & ((measured_surface - predicted_surface_range).abs() < gate)
        )
        center_range = measured_surface + self.target_radius
        measured_vehicle = torch.stack(
            [
                center_range * torch.cos(bearing),
                center_range * torch.sin(bearing),
                rel[:, 2],
            ],
            dim=1,
        )
        measurement_world = drone_pos_w + _quat_rotate_xyzw(vehicle_quat, measured_vehicle)
        lidar_var = torch.tensor(
            [0.08**2, 0.15**2, 0.20**2], device=self.device
        ).view(1, 3).expand(self.num_envs, -1)
        # H3: only `measured_surface` is measured here. The bearing and the vertical component of
        # `measured_vehicle` are the tracker's OWN prediction read back, so feeding all three to a
        # diagonal-R update hands the filter information it never received: the lateral and
        # vertical covariance stop growing even though nothing observed them. Measured on a target
        # the camera has lost, the reported lateral sigma sits at 0.09 m while the true error runs
        # to 3.27 m over 20 steps -- and the policy reads that covariance in its target token, so
        # it is told "certain" exactly when it should be told "lost". Building R as
        # sigma_r^2*uu^T + sigma_perp^2*(I - uu^T) about the measurement ray keeps the range
        # update and leaves the unobserved directions to process noise, which is what an
        # anisotropic sensor actually justifies.
        lidar_cov = None
        if self.lidar_range_only_update:
            ray = measurement_world - drone_pos_w
            unit = ray / ray.norm(dim=1, keepdim=True).clamp(min=1e-6)
            outer = unit.unsqueeze(2) * unit.unsqueeze(1)
            eye = torch.eye(3, device=self.device).unsqueeze(0)
            lidar_cov = (
                lidar_var[:, 0].view(-1, 1, 1) * outer
                + LIDAR_UNOBSERVED_SIGMA_M**2 * (eye - outer)
            )
        self.tracker.correct(
            measurement_world,
            valid,
            lidar_var,
            measurement_cov=lidar_cov,
            reset_age=not self.lidar_silent_correct,
        )
        confidence = torch.exp(
            -(measured_surface - predicted_surface_range).abs() / gate.clamp(min=1e-3)
        ) * valid.float()
        return valid, confidence, measured_surface, bearing

    def _target_features(
        self, drone_pos_w, drone_vel_w, vehicle_quat, camera_confidence, lidar_confidence
    ):
        state, cov, active, age = (
            self.tracker.state,
            self.tracker.cov,
            self.tracker.active,
            self.tracker.age,
        )
        pos_world = state[:, :3]
        # P0 latency compensation: the KF is corrected with measurements that are
        # detection_latency_s OLD, so its position estimate trails the true target by ~v*tau.
        # Extrapolate the POLICY-FACING position to "now" with the filter's own velocity.
        # Output-side only: KF internals, covariance, age, LiDAR association, and diagnostics
        # are untouched, and the observation stays the same width.
        if self.latency_compensate and self._latency_steps > 0:
            pos_world = pos_world + state[:, 3:] * self.detection_latency_s
        rel_pos = _quat_rotate_inverse_xyzw(vehicle_quat, pos_world - drone_pos_w)
        rel_vel = _quat_rotate_inverse_xyzw(vehicle_quat, state[:, 3:] - drone_vel_w)
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        pos_var = diag[:, :3] / (self.max_camera_range * self.max_camera_range)
        vel_var = diag[:, 3:] / 4.0
        features = torch.cat(
            [
                rel_pos / self.max_camera_range,
                rel_vel / 2.0,
                pos_var.clamp(0.0, 1.0),
                vel_var.clamp(0.0, 1.0),
                camera_confidence.unsqueeze(1),
                lidar_confidence.unsqueeze(1),
                (age / self.tracker.memory_s).clamp(0.0, 1.0).unsqueeze(1),
                torch.full_like(age.unsqueeze(1), self.target_radius / self.lidar_max_range),
            ],
            dim=1,
        )
        return features * active.unsqueeze(1)

    def _update_histories(self, robot_now, target_now, obstacles_now):
        if self.step_count % self.history_stride == 0:
            self.robot_history = torch.roll(self.robot_history, shifts=-1, dims=1)
            self.target_history = torch.roll(self.target_history, shifts=-1, dims=1)
            self.obstacle_history = torch.roll(self.obstacle_history, shifts=-1, dims=1)
        self.robot_history[:, -1] = robot_now
        self.target_history[:, -1] = target_now
        self.obstacle_history[:, -1] = obstacles_now
        self.step_count += 1

    def observe(
        self,
        rgb,
        depth,
        lidar_m,
        drone_pos_w,
        drone_vel_w,
        vehicle_quat,
        yaw_rate,
        previous_action,
        max_velocity,
        flight_altitude,
        training=True,
        env_bounds_min=None,
        env_bounds_max=None,
    ):
        """Return actor-safe structured observation and sensor diagnostics."""
        meas_vehicle, surface_range, bearing, visible, confidence, pixels = self._detect_rgbd(
            rgb, depth, training, drone_pos_w=drone_pos_w, vehicle_quat=vehicle_quat
        )
        # P3 ego-motion compensation: a DELAYED detection was taken in the vehicle frame of
        # t-tau, so lifting it to world with the pose at t injects the drone's own motion over
        # tau into every KF correction -- 0.23 m of translation at the measured 2.33 m/s mean
        # speed, plus yaw applied straight to the bearing. Both dominate the <=0.15 m target lag
        # that P0 removes, which is why P0 alone changed nothing (WORKLOG 2026-08-06). Lifting
        # with the buffered capture-time pose leaves a measurement that is accurate but tau old,
        # i.e. exactly the residual P0 was written for.
        meas_pos_w, meas_quat = drone_pos_w, vehicle_quat
        if self.latency_ego_motion_fix and self._latency_delayed_pose is not None:
            meas_pos_w, meas_quat = self._latency_delayed_pose
        meas_world = meas_pos_w + _quat_rotate_xyzw(meas_quat, meas_vehicle)
        if self.detect_decoupled:
            # The measurement was made on the detect-resolution mask, so its pixel support --
            # which sets the shot-noise term of sigma_r -- is the detect-resolution count, not
            # the camera-resolution mask that is still used for the obstacle map.
            pixel_count = self._last_detect_count.float().clamp(min=1.0)
        else:
            pixel_count = pixels.sum(dim=(1, 2)).float().clamp(min=1.0)
        if self.depth_noise_model == "stereo":
            # Intel: RMS = D^2 * subpixel / (fx_sensor * baseline). Only the range term changes;
            # the 0.04 floor and the 0.15/sqrt(px) centroid shot-noise term are properties of the
            # detection, not of the stereo depth, so they carry over unchanged.
            range_term = self.depth_stereo_coeff * surface_range.square()
        else:
            range_term = 0.012 * surface_range
        sigma_r = 0.04 + range_term + 0.15 / pixel_count.sqrt()
        # metres per pixel of the image the centroid was actually measured on.
        sigma_lat = 0.03 + surface_range / max(self.detect_fx, 1.0)
        measurement_var = torch.stack(
            [sigma_r.square(), sigma_lat.square(), sigma_lat.square()], dim=1
        )
        self.tracker.step(meas_world, visible, measurement_var)
        if self._s1_shadow is not None:
            # Same-frame counterfactual: same raw mask, same pose lift as the live measurement
            # (latency is asserted zero at construction, so meas_pos_w/meas_quat are the current
            # pose). Results go ONLY into diagnostics; the live tracker was stepped above.
            s1_mask, s1_depth = self._s1_shadow_frame
            self._s1_shadow_frame = None
            k = self._s1_shadow.top_k
            s1_quat = meas_quat.repeat_interleave(k, dim=0)
            s1_pos = meas_pos_w.repeat_interleave(k, dim=0)
            s1_meas_world, s1_visible, s1_initialized, s1_num_candidates = self._s1_shadow.step(
                s1_mask, s1_depth,
                lambda vehicle_xyz: s1_pos + _quat_rotate_xyzw(s1_quat, vehicle_xyz),
            )
        # P1 latency compensation: `visible` here is the DELAYED camera flag (the ring buffer in
        # _apply_detection_latency already ran inside _detect_rgbd). The historical
        # `~camera_visible` gate inside _associate_lidar_target then blocks the fresh-LiDAR
        # correction exactly when the stale camera claims sight -- the structural reason latency
        # was catastrophic (-42.7 pp) while range error was benign. With the backup enabled a
        # delayed detection no longer vetoes the LiDAR path; sim LiDAR has no latency.
        lidar_camera_gate = visible
        if self.latency_lidar_backup and self._latency_steps > 0:
            lidar_camera_gate = torch.zeros_like(visible)
        if self.lidar_target_assoc:
            lidar_visible, lidar_confidence, lidar_surface, lidar_bearing = (
                self._associate_lidar_target(
                    lidar_m, drone_pos_w, vehicle_quat, lidar_camera_gate
                )
            )
        else:
            lidar_visible = torch.zeros_like(visible)
            lidar_confidence = torch.zeros_like(confidence)
            lidar_surface = torch.zeros_like(surface_range)
            lidar_bearing = torch.zeros_like(bearing)
        if self.lidar_silent_correct:
            # The correction has already been applied inside _associate_lidar_target; from here
            # on the association contributes no visibility, no confidence, and no map edits.
            lidar_visible = torch.zeros_like(lidar_visible)
            lidar_confidence = torch.zeros_like(lidar_confidence)
        fused_visible = visible | lidar_visible
        fused_surface = torch.where(lidar_visible, lidar_surface, surface_range)
        fused_bearing = torch.where(lidar_visible, lidar_bearing, bearing)

        map_pixels, map_surface, map_bearing, map_visible = self._latency_corrected_map_inputs(
            depth, pixels, fused_surface, fused_bearing, fused_visible, drone_pos_w, vehicle_quat
        )
        # The obstacle map is edited in two places that use DIFFERENT gates: the LiDAR
        # target_like carve-out is gated on fused_visible (camera OR LiDAR), while the depth
        # blanking is gated on the camera-only pixel mask. Whenever LiDAR alone is holding the
        # track -- every dropped camera frame -- the LiDAR half removes the target but the camera
        # half puts it straight back, so the target becomes a phantom obstacle dead ahead and
        # consumes one of the 8 obstacle tokens exactly when the drone should be closing in.
        # Backfilling the mask from the fused bearing/range makes both halves agree.
        if self.target_mask_backfill:
            camera_mask_empty = ~map_pixels.any(dim=1).any(dim=1)
            map_pixels = torch.where(
                (map_visible & camera_mask_empty).view(-1, 1, 1),
                self._reconstruct_target_pixels(depth, map_bearing, map_surface, map_visible),
                map_pixels,
            )
        static_state, obstacles_now = self._fuse_static_and_extract_obstacles(
            lidar_m,
            depth,
            map_pixels,
            map_surface,
            map_bearing,
            map_visible,
            drone_vel_w=drone_vel_w,
            vehicle_quat=vehicle_quat,
        )
        target_now = self._target_features(
            drone_pos_w, drone_vel_w, vehicle_quat, confidence, lidar_confidence
        )
        if self.search is not None:
            if self.search.cell_centres is None:
                self.search.set_arena_bounds(env_bounds_min, env_bounds_max)
            self.search.update(
                drone_pos_w,
                vehicle_quat,
                depth,
                self.tracker.state,
                self.tracker.cov,
                self.tracker.active,
            )
        robot_now = torch.cat(
            [
                _quat_rotate_inverse_xyzw(vehicle_quat, drone_vel_w) / max_velocity,
                yaw_rate.unsqueeze(1),
                previous_action,
                (drone_pos_w[:, 2:3] / max(flight_altitude * 3.0, 1e-6)),
                torch.ones(self.num_envs, 1, device=self.device),
            ],
            dim=1,
        )
        self._update_histories(robot_now, target_now, obstacles_now)
        self.last_visible[:] = fused_visible
        self.last_confidence[:] = torch.maximum(confidence, lidar_confidence)

        obs_parts = [
            static_state,
            self.obstacle_history.reshape(self.num_envs, -1),
            self.robot_history.reshape(self.num_envs, -1),
            self.target_history.reshape(self.num_envs, -1),
        ]
        if CORRIDOR_TOKENS > 0:
            # Free-gap affordance from the SAME fused profile the obstacle tokens use
            # (last_scan_nearest is set inside _fuse_static_and_extract_obstacles above).
            # Appended LAST so all pre-existing segment offsets stay checkpoint-compatible.
            corridor_tokens, _ = extract_corridor_tokens(
                self.last_scan_nearest,
                self._lidar_angles,
                max_range=self.lidar_max_range,
                num_corridors=CORRIDOR_TOKENS,
                fov_deg=OBSTACLE_FOV_DEG,
                horizon_m=CORRIDOR_HORIZON_M,
                min_width_m=CORRIDOR_MIN_WIDTH_M,
            )
            obs_parts.append(corridor_tokens.reshape(self.num_envs, -1))
        if GEOFENCE_ACTOR:
            geofence = body_geofence_features(
                drone_pos_w,
                vehicle_quat,
                env_bounds_min,
                env_bounds_max,
                noise_std_m=GEOFENCE_NOISE_STD_M,
                dropout=GEOFENCE_DROPOUT,
            )
            if GEOFENCE_FORCE_INVALID or SEARCH_STATE_FORCE_INVALID:
                geofence[:, :GEOFENCE_RAYS] = 1.0
                geofence[:, GEOFENCE_RAYS:] = 0.0
            obs_parts.append(geofence)
        if self.search is not None:
            obs_parts.append(
                self.search.features(
                    drone_pos_w,
                    vehicle_quat,
                    force_invalid=SEARCH_STATE_FORCE_INVALID,
                )
            )
        obs = torch.cat(obs_parts, dim=1)
        if obs.shape[1] != STRUCTURED_OBS_DIM:
            raise RuntimeError(
                "perception observation schema drift: got %d, expected %d"
                % (obs.shape[1], STRUCTURED_OBS_DIM)
            )
        diagnostics = {
            "visible": fused_visible,
            "camera_visible": visible,
            "lidar_visible": lidar_visible,
            "confidence": torch.maximum(confidence, lidar_confidence),
            # The CAMERA half of the two fields above, unfused. Both are already computed above --
            # these keys add references, not work. They exist because the appearance-distractor
            # measurement (docs/prereg_2026-09-01_distractor_envelope.md section 4) classifies the
            # position the RGB-D detector reported, and the fused fields would attribute a
            # LiDAR-only track to the camera. Diagnostics are evaluator-facing: the actor
            # observation is `obs`, which is built above and never reads this dict.
            "camera_confidence": confidence,
            # `meas_world` IS the detector's reported 3D measurement -- the same tensor handed to
            # self.tracker.step() as the KF observation. Exporting the tracker's posterior instead
            # would measure the filter, not the detector.
            "camera_measurement_world": meas_world,
            "target_pixels": pixel_count,
            "track_age": self.tracker.age,
            "track_covariance": self.tracker.cov,
            **(
                {
                    "s1_shadow_measurement_world": s1_meas_world,
                    "s1_shadow_visible": s1_visible,
                    "s1_shadow_initialized": s1_initialized,
                    "s1_shadow_num_candidates": s1_num_candidates,
                }
                if self._s1_shadow is not None
                else {}
            ),
            "search_state_masked": torch.full(
                (self.num_envs,),
                SEARCH_STATE_FORCE_INVALID,
                dtype=torch.bool,
                device=self.device,
            ),
        }
        if self.search is not None:
            diagnostics.update(self.search.diagnostics())
        return obs, diagnostics
