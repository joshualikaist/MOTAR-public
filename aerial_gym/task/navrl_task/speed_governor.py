"""Sensor-only horizontal speed governor for dense NavRL obstacle fields.

The policy already commands a body-frame velocity.  This module never changes its direction or
uses simulator geometry: it only scales the horizontal norm from the current LiDAR scan.  Keeping
the math here independent of Isaac Gym makes the safety contract CPU-testable.
"""

from dataclasses import dataclass
import math

import torch


VALID_SPEED_GOVERNOR_MODES = (
    "off", "fixed", "clearance", "ttc", "riskcap", "stopcap",
    # A4 baselines (prereg pending). Both answer "is the corridor the problem?" by changing
    # only the GEOMETRY of the measurement, keeping the stopping-distance law identical.
    "omni",     # same law, but clearance is the nearest return in ANY bearing
    "dwa_arc",  # same law, but clearance is measured along the arc the vehicle is on
    "riskcap_arc",  # A7: riskcap law with the same arc clearance as dwa_arc
)


def _finite_float(environ, name, default, *, minimum=None, strict_minimum=False):
    raw = str(environ.get(name, "")).strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric; got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if minimum is not None:
        bad = value <= minimum if strict_minimum else value < minimum
        if bad:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}; got {value!r}")
    return value


@dataclass(frozen=True)
class SpeedGovernorConfig:
    mode: str = "off"
    fixed_cap_mps: float = 2.0
    free_speed_cap_mps: float = math.sqrt(2.0) * 2.5
    path_half_width_m: float = 0.45
    hard_margin_m: float = 0.45
    slow_distance_m: float = 3.0
    release_distance_m: float = 5.0
    ttc_s: float = 1.0
    brake_mps2: float = 2.0
    reaction_s: float = 0.1
    # Plan I3 (docs/plans/lateral_contact_density_plan_2026-09-05.md section 4, series L). Each is
    # a single additive change on top of ANY base law, so one arm changes one thing.
    #   L2  width_per_mps:   corridor half-width grows with requested speed (Nav2 VelocityPolygon)
    #   L3  lateral_*:       a second, softer cap from the nearest return BESIDE the vehicle
    #                        (RSS lateral rule); floor > 0 so it cannot deadlock like omni
    #   L5  yaw_cap_*:       yaw-rate magnitude is scaled down when something is close beside
    # All are off at their defaults (0.0), which reproduces every result before this change.
    width_per_mps: float = 0.0
    #   L8  width_per_open_m / width_open_ref_m: the tube widens where the vehicle SEES open space.
    #       L1 found the best width differs by density (1.2 m at 70 bars, 0.6 m at 205), but bar
    #       count is a simulator parameter no vehicle can observe. Nearest sensed surface can:
    #       it falls 2.06 -> 1.18 m from 70 to 205 bars with no overlap between densities, so the
    #       same rule is implementable on hardware. Zero gain reproduces the fixed width exactly.
    width_per_open_m: float = 0.0
    width_open_ref_m: float = 1.2
    width_max_m: float = 3.0
    lateral_margin_m: float = 0.0
    lateral_span_m: float = 2.0
    lateral_floor_mps: float = 1.0
    yaw_cap_radps: float = 0.0
    yaw_cap_margin_m: float = 1.0

    @property
    def open_width_enabled(self):
        return self.width_per_open_m > 0.0

    @property
    def lateral_channel_enabled(self):
        return self.lateral_margin_m > 0.0

    @property
    def yaw_cap_enabled(self):
        return self.yaw_cap_radps > 0.0

    @classmethod
    def from_environ(cls, environ):
        mode = str(environ.get("NAVRL_SPEED_GOVERNOR", "off")).strip().lower() or "off"
        if mode not in VALID_SPEED_GOVERNOR_MODES:
            raise ValueError(
                "NAVRL_SPEED_GOVERNOR must be one of %s; got %r"
                % (", ".join(VALID_SPEED_GOVERNOR_MODES), mode)
            )
        result = cls(
            mode=mode,
            fixed_cap_mps=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_FIXED_MPS", 2.0, minimum=0.0,
                strict_minimum=True,
            ),
            free_speed_cap_mps=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_FREE_MPS", math.sqrt(2.0) * 2.5,
                minimum=0.0, strict_minimum=True,
            ),
            path_half_width_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M", 0.45,
                minimum=0.0, strict_minimum=True,
            ),
            hard_margin_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_MARGIN_M", 0.45, minimum=0.0,
            ),
            slow_distance_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_SLOW_M", 3.0, minimum=0.0,
                strict_minimum=True,
            ),
            release_distance_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_RELEASE_M", 5.0, minimum=0.0,
                strict_minimum=True,
            ),
            ttc_s=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_TTC_S", 1.0, minimum=0.0,
                strict_minimum=True,
            ),
            brake_mps2=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2", 2.0, minimum=0.0,
                strict_minimum=True,
            ),
            reaction_s=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_REACTION_S", 0.1, minimum=0.0,
            ),
            width_per_mps=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_WIDTH_PER_MPS", 0.0, minimum=0.0,
            ),
            width_per_open_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_WIDTH_PER_OPEN_M", 0.0, minimum=0.0,
            ),
            width_open_ref_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_WIDTH_OPEN_REF_M", 1.2, minimum=0.0,
                strict_minimum=True,
            ),
            width_max_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_WIDTH_MAX_M", 3.0, minimum=0.0,
                strict_minimum=True,
            ),
            lateral_margin_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_LATERAL_MARGIN_M", 0.0, minimum=0.0,
            ),
            lateral_span_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_LATERAL_SPAN_M", 2.0, minimum=0.0,
                strict_minimum=True,
            ),
            lateral_floor_mps=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_LATERAL_FLOOR_MPS", 1.0, minimum=0.0,
            ),
            yaw_cap_radps=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_YAW_CAP_RADPS", 0.0, minimum=0.0,
            ),
            yaw_cap_margin_m=_finite_float(
                environ, "NAVRL_SPEED_GOVERNOR_YAW_CAP_MARGIN_M", 1.0, minimum=0.0,
                strict_minimum=True,
            ),
        )
        if result.slow_distance_m <= result.hard_margin_m:
            raise ValueError(
                "NAVRL_SPEED_GOVERNOR_SLOW_M must exceed NAVRL_SPEED_GOVERNOR_MARGIN_M"
            )
        if result.mode in ("riskcap", "riskcap_arc"):
            if result.release_distance_m <= result.slow_distance_m:
                raise ValueError(
                    "NAVRL_SPEED_GOVERNOR_RELEASE_M must exceed NAVRL_SPEED_GOVERNOR_SLOW_M "
                    "for riskcap"
                )
            if result.free_speed_cap_mps < result.fixed_cap_mps:
                raise ValueError(
                    "NAVRL_SPEED_GOVERNOR_FREE_MPS must be >= NAVRL_SPEED_GOVERNOR_FIXED_MPS "
                    "for riskcap"
                )
        if result.lateral_channel_enabled and result.lateral_floor_mps > result.free_speed_cap_mps:
            raise ValueError(
                "NAVRL_SPEED_GOVERNOR_LATERAL_FLOOR_MPS must not exceed NAVRL_SPEED_GOVERNOR_FREE_MPS"
            )
        if result.mode == "stopcap":
            if result.brake_mps2 <= 0.0:
                raise ValueError(
                    "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2 must be strictly positive for stopcap"
                )
        return result


def directional_lidar_clearance(
    lidar_m,
    bearings_rad,
    command_xy,
    *,
    max_range_m,
    path_half_width_m,
    target_return_mask=None,
    vertical_fov_deg=(20.0, -10.0),
):
    """Return nearest sensor surface inside the swept horizontal command corridor.

    ``lidar_m`` is ``[N,V,H]`` slant range, ``bearings_rad`` is ``[H]``, and commands are in the
    same vehicle frame. ``vertical_fov_deg`` follows tensor row order (row 0, last row), which is
    +20 to -10 degrees for the NavRL Warp LiDAR. Target returns are removed because the target is a
    capture object, not a collision obstacle. No-return rays remain no-return after projection.
    """

    if lidar_m.ndim != 3:
        raise ValueError("lidar_m must be [batch, vertical_beams, horizontal_beams]")
    if command_xy.ndim != 2 or command_xy.shape[0] != lidar_m.shape[0] or command_xy.shape[1] != 2:
        raise ValueError("command_xy must be [batch, 2]")
    if bearings_rad.ndim != 1 or bearings_rad.shape[0] != lidar_m.shape[2]:
        raise ValueError("bearings_rad must match the horizontal beam dimension")
    if target_return_mask is not None and target_return_mask.shape != lidar_m.shape:
        raise ValueError("target_return_mask must match lidar_m")

    max_range = float(max_range_m)
    finite = torch.isfinite(lidar_m)
    valid = finite & (lidar_m >= 0.0) & (lidar_m < max_range * 0.995)
    if target_return_mask is not None:
        valid &= ~target_return_mask.bool()
    safe_range = torch.where(valid, lidar_m, torch.full_like(lidar_m, max_range))

    vertical_angles = torch.linspace(
        math.radians(float(vertical_fov_deg[0])),
        math.radians(float(vertical_fov_deg[1])),
        lidar_m.shape[1],
        device=lidar_m.device,
        dtype=lidar_m.dtype,
    )
    horizontal_range = safe_range * torch.cos(vertical_angles).view(1, -1, 1)
    horizontal_range = torch.where(valid, horizontal_range, torch.full_like(horizontal_range, max_range))
    nearest = horizontal_range.amin(dim=1)
    ray_valid = valid.any(dim=1)

    requested_speed = command_xy.norm(dim=1)
    command_bearing = torch.atan2(command_xy[:, 1], command_xy[:, 0])
    delta = torch.atan2(
        torch.sin(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
        torch.cos(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
    )
    forward = nearest * torch.cos(delta)
    lateral = (nearest * torch.sin(delta)).abs()
    in_path = (
        ray_valid
        & (forward > 0.0)
        & (lateral <= _half_width_column(path_half_width_m, lateral))
    )
    clearance = torch.where(
        in_path, forward, torch.full_like(forward, max_range)
    ).amin(dim=1)
    return torch.where(
        requested_speed > 1e-6,
        clearance.clamp(min=0.0, max=max_range),
        torch.full_like(clearance, max_range),
    )


def _half_width_column(path_half_width_m, like):
    """A scalar half-width, or a per-env [N] tensor broadcast against a [N, H] ray matrix."""
    if isinstance(path_half_width_m, torch.Tensor):
        return path_half_width_m.to(like.dtype).view(-1, 1)
    return float(path_half_width_m)


def speed_dependent_half_width(config, requested_speed, open_m=None):
    """Corridor half-width for this step.

    L2 adds k_v * |v_requested|; L8 adds k_o * (open - ref), where `open` is the nearest sensed
    surface in ANY bearing -- the vehicle's own read of how cluttered it is. Both default to 0,
    which returns the scalar w0 and reproduces every earlier arm bit for bit.
    """
    k_v, k_o = float(config.width_per_mps), float(config.width_per_open_m)
    if k_v <= 0.0 and k_o <= 0.0:
        return float(config.path_half_width_m)
    width = float(config.path_half_width_m)
    if k_v > 0.0:
        width = width + k_v * requested_speed
    if k_o > 0.0:
        if open_m is None:
            raise ValueError("open-space width is enabled but no clutter measurement was supplied")
        width = width + k_o * (open_m - float(config.width_open_ref_m))
    # The sensed-clutter term only WIDENS. Verification of the first L8 run showed that letting it
    # narrow below w0 (old floor 0.1 m) shrank the tube on 44-45% of frames at 205 bars -- exactly
    # where the width sweep says a wider tube is optimal -- so the term now floors at w0.
    floor = float(config.path_half_width_m)
    return width.clamp(floor, float(config.width_max_m)) if hasattr(width, "clamp") else width


LATERAL_SECTOR_DEG = (15.0, 165.0)   # same sector the contact records call "beside"


def lateral_clearance(
    lidar_m, bearings_rad, command_xy, *, max_range_m, target_return_mask=None,
    sector_deg=LATERAL_SECTOR_DEG,
):
    """L3: nearest sensor surface BESIDE the commanded direction (either side), horizontally
    projected like the corridor. Rays inside +-15 deg of the command are the corridor's job; rays
    beyond 165 deg are behind. No-return rays stay no-return, exactly as in the corridor."""
    if lidar_m.ndim != 3:
        raise ValueError("lidar_m must be [batch, vertical_beams, horizontal_beams]")
    max_range = float(max_range_m)
    finite = torch.isfinite(lidar_m)
    valid = finite & (lidar_m >= 0.0) & (lidar_m < max_range * 0.995)
    if target_return_mask is not None:
        valid &= ~target_return_mask.bool()
    nearest = _horizontal_nearest(lidar_m, valid, max_range)
    ray_valid = valid.any(dim=1)
    command_bearing = torch.atan2(command_xy[:, 1], command_xy[:, 0])
    delta = torch.atan2(
        torch.sin(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
        torch.cos(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
    ).abs()
    lo, hi = math.radians(float(sector_deg[0])), math.radians(float(sector_deg[1]))
    beside = ray_valid & (delta >= lo) & (delta <= hi)
    return torch.where(beside, nearest, torch.full_like(nearest, max_range)).amin(dim=1)


def lateral_cap(lateral_clearance_m, config):
    """L3 cap: floor at lateral_floor_mps, released linearly to free_speed_cap over
    [lateral_margin_m, lateral_margin_m + lateral_span_m]. The floor is what keeps this from
    reproducing omni's deadlock: something is always beside the vehicle in clutter."""
    release = (
        (lateral_clearance_m - float(config.lateral_margin_m)) / float(config.lateral_span_m)
    ).clamp(0.0, 1.0)
    return float(config.lateral_floor_mps) + release * (
        float(config.free_speed_cap_mps) - float(config.lateral_floor_mps)
    )


def yaw_scale(lateral_clearance_m, yaw_rate_max, config):
    """L5: multiplicative scale in (0, 1] on the commanded yaw-rate magnitude. Full authority when
    the nearest beside-return is farther than yaw_cap_margin_m; capped at yaw_cap_radps inside.
    Never changes the sign, so it is magnitude-only in yaw exactly as the governor is in xy."""
    cap = float(config.yaw_cap_radps) / max(1e-6, float(yaw_rate_max))
    close = lateral_clearance_m < float(config.yaw_cap_margin_m)
    return torch.where(close, torch.full_like(lateral_clearance_m, min(1.0, cap)),
                       torch.ones_like(lateral_clearance_m))


def _horizontal_nearest(lidar_m, valid, max_range_m, vertical_fov_deg=(20.0, -10.0)):
    """Per-bearing nearest range projected onto the horizontal plane.

    Shared with directional_lidar_clearance so that the A4 arms differ from the corridor in
    geometry ONLY. Forgetting this projection makes an arm look 6% further-sighted than the
    corridor at the +20 degree row, which would confound the comparison it exists to make.
    """
    vertical_angles = torch.linspace(
        math.radians(float(vertical_fov_deg[0])),
        math.radians(float(vertical_fov_deg[1])),
        lidar_m.shape[1],
        device=lidar_m.device,
        dtype=lidar_m.dtype,
    )
    safe = torch.where(valid, lidar_m, torch.full_like(lidar_m, float(max_range_m)))
    horiz = safe * torch.cos(vertical_angles).view(1, -1, 1)
    horiz = torch.where(valid, horiz, torch.full_like(horiz, float(max_range_m)))
    return horiz.amin(dim=1)


def omnidirectional_clearance(lidar_m, *, max_range_m, target_return_mask=None):
    """Nearest sensor surface in ANY bearing -- the corridor removed, nothing else changed.

    This is the crude answer the contact forensics invites: 57-58% of contacts were lateral to a
    0.45 m half-width corridor, so what happens if the corridor is simply dropped? It is the
    baseline a more careful geometric test has to beat, and the S1/Q1 history says that is not a
    formality -- riskcap did not beat a constant cap.
    """
    finite = torch.isfinite(lidar_m)
    valid = finite & (lidar_m >= 0.0) & (lidar_m < float(max_range_m) * 0.995)
    if target_return_mask is not None:
        valid &= ~target_return_mask.bool()
    if lidar_m.ndim != 3:
        raise ValueError("lidar_m must be [batch, vertical_beams, horizontal_beams]")
    return _horizontal_nearest(lidar_m, valid, max_range_m).amin(dim=1)


def arc_clearance(
    lidar_m,
    bearings_rad,
    command_xy,
    yaw_rate,
    *,
    max_range_m,
    path_half_width_m,
    target_return_mask=None,
):
    """Distance to the nearest obstacle along the CIRCULAR ARC the vehicle is turning onto.

    Fox, Burgard & Thrun (IEEE RAM 1997) Eq. 14 measures dist(v, omega) on the arc, not down a
    straight corridor, and their section 2 argues the straight-line decomposition is only
    justifiable "if infinite forces can be asserted on the robot". This arm quantifies that
    objection on our own data: same stopping law, same scan, arc instead of a line.

    For speed v and yaw rate w the instantaneous arc has radius R = v / w. A ray at bearing
    `delta` off the command intersects that arc where the chord subtends the same angle, giving
    an along-arc distance of R * 2 * delta for the ray that touches it. We keep the corridor's
    half-width as the tube radius about the arc so that only the geometry changes.
    """
    if lidar_m.ndim != 3:
        raise ValueError("lidar_m must be [batch, vertical_beams, horizontal_beams]")
    max_range = float(max_range_m)
    finite = torch.isfinite(lidar_m)
    valid = finite & (lidar_m >= 0.0) & (lidar_m < max_range * 0.995)
    if target_return_mask is not None:
        valid &= ~target_return_mask.bool()
    nearest = _horizontal_nearest(lidar_m, valid, max_range)
    ray_valid = valid.any(dim=1)

    speed = command_xy.norm(dim=1)
    command_bearing = torch.atan2(command_xy[:, 1], command_xy[:, 0])
    delta = torch.atan2(
        torch.sin(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
        torch.cos(bearings_rad.view(1, -1) - command_bearing.view(-1, 1)),
    )
    # Signed turn radius; a near-zero yaw rate degenerates to the straight corridor by construction.
    w = yaw_rate.view(-1, 1)
    straight = w.abs() < 1e-3
    radius = torch.where(straight, torch.full_like(w, 1e6), speed.view(-1, 1) / w)

    # Perpendicular offset of the ray endpoint from the arc, and the along-arc travel to reach it.
    px = nearest * torch.cos(delta)
    py = nearest * torch.sin(delta)
    perp = (torch.sqrt(px.square() + (py - radius).square()) - radius.abs()).abs()
    # A7 M2: at the straight limit the huge-radius surrogate loses the +-0.45 m boundary to
    # float32 cancellation (12 m vs 3.97 m on a ray at lateral 0.46 m). Use the exact lateral
    # offset there so yaw rate 0 reproduces directional_lidar_clearance bit for bit.
    perp = torch.where(straight, py.abs(), perp)
    along = torch.where(
        straight, px, (radius.abs() * (2.0 * delta.abs())).clamp(max=max_range)
    )
    on_arc = ray_valid & (along > 0.0) & (perp <= _half_width_column(path_half_width_m, perp))
    clearance = torch.where(on_arc, along, torch.full_like(along, max_range)).amin(dim=1)
    return torch.where(
        speed > 1e-6, clearance.clamp(0.0, max_range), torch.full_like(clearance, max_range)
    )


def apply_speed_governor(command_xy, clearance_m, config, lateral_clearance_m=None):
    """Scale horizontal velocity and return tensors required for causal diagnostics.

    ``lateral_clearance_m`` (L3) is optional; when the lateral channel is enabled the final cap
    is the minimum of the base law's cap and the lateral cap. Direction is never changed.
    """

    if command_xy.ndim != 2 or command_xy.shape[1] != 2:
        raise ValueError("command_xy must be [batch, 2]")
    if clearance_m.ndim != 1 or clearance_m.shape[0] != command_xy.shape[0]:
        raise ValueError("clearance_m must be [batch]")
    if config.lateral_channel_enabled and lateral_clearance_m is None:
        raise ValueError("lateral channel is enabled but no lateral clearance was supplied")
    requested = command_xy.norm(dim=1)
    usable = (clearance_m - float(config.hard_margin_m)).clamp(min=0.0)

    if config.mode == "off":
        cap = requested
    elif config.mode == "fixed":
        cap = torch.full_like(requested, float(config.fixed_cap_mps))
    elif config.mode == "clearance":
        span = float(config.slow_distance_m - config.hard_margin_m)
        cap = float(config.free_speed_cap_mps) * (usable / span).clamp(0.0, 1.0)
    elif config.mode == "ttc":
        cap = (usable / float(config.ttc_s)).clamp(
            min=0.0, max=float(config.free_speed_cap_mps)
        )
    elif config.mode in ("riskcap", "riskcap_arc"):
        # Minimum-intervention filter: preserve the fixed-2.0 positive control in clutter, but
        # release it smoothly in genuinely open command directions. Unlike clearance/TTC this
        # never creates a forced zero-speed deadlock; a policy request below the cap is untouched.
        release_span = float(config.release_distance_m - config.slow_distance_m)
        release = (
            (clearance_m - float(config.slow_distance_m)) / release_span
        ).clamp(0.0, 1.0)
        cap = float(config.fixed_cap_mps) + release * (
            float(config.free_speed_cap_mps) - float(config.fixed_cap_mps)
        )
    elif config.mode in ("stopcap", "omni", "dwa_arc"):
        # Stopping-distance admissible cap (DWA admissibility; RSS longitudinal rule with
        # a_accel=0 and a static obstacle): the largest v with
        #   usable >= v*reaction + v^2/(2*brake),
        # solved for v. Unlike riskcap there is no floor -- the cap reaches zero exactly at
        # usable=0, so stopping_margin_executed >= 0 holds by construction (up to inter-step
        # clearance change). hard_margin_m is live again in this mode via `usable`.
        brake = float(config.brake_mps2)
        reaction_reach = brake * float(config.reaction_s)
        cap = (
            torch.sqrt(reaction_reach * reaction_reach + 2.0 * brake * usable)
            - reaction_reach
        ).clamp(min=0.0, max=float(config.free_speed_cap_mps))
    else:  # Config construction is fail-closed, but keep direct callers safe.
        raise ValueError(f"unsupported speed governor mode: {config.mode!r}")

    if config.lateral_channel_enabled:
        lat_cap = lateral_cap(lateral_clearance_m, config)
        cap = torch.minimum(cap, lat_cap)
    else:
        lat_cap = torch.full_like(cap, float("inf"))
    executed_speed = torch.minimum(requested, cap)
    scale = torch.where(
        requested > 1e-6,
        executed_speed / requested.clamp(min=1e-6),
        torch.ones_like(requested),
    )
    governed = command_xy * scale.unsqueeze(1)

    def stopping_margin(speed):
        stopping_distance = (
            speed * float(config.reaction_s)
            + speed.square() / (2.0 * float(config.brake_mps2))
        )
        return usable - stopping_distance

    ttc_requested = torch.where(
        requested > 1e-6,
        usable / requested.clamp(min=1e-6),
        torch.full_like(requested, float("inf")),
    )
    return governed, {
        "requested_speed_mps": requested,
        "executed_speed_mps": executed_speed,
        "speed_cap_mps": cap,
        "scale": scale,
        "clearance_m": clearance_m,
        "ttc_requested_s": ttc_requested,
        "stopping_margin_requested_m": stopping_margin(requested),
        "stopping_margin_executed_m": stopping_margin(executed_speed),
        "lateral_cap_mps": lat_cap,
        "lateral_clearance_m": (
            lateral_clearance_m if lateral_clearance_m is not None
            else torch.full_like(cap, float("inf"))
        ),
    }
