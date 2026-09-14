"""Exact soft-envelope geometry for the fresh braking-aware target route.

This module is deliberately simulator independent.  The NumPy path is used by reset-time A*,
while the Torch path certifies ordinary GPU control intervals.  Both represent obstacles as
*closed* axis-aligned boxes and arena limits as an open admissible centre region.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np
import torch


TARGET_ROUTE_BRAKING_GEOMETRY_SCHEMA = "navrl_target_route_soft_aabb_v3"


@dataclass(frozen=True)
class SoftEnvelopeSpec:
    wall_margin_m: float
    boundary_reserve_m: float
    tracking_margin_m: float

    def validate(self) -> None:
        for name, value in (
            ("wall_margin_m", self.wall_margin_m),
            ("boundary_reserve_m", self.boundary_reserve_m),
            ("tracking_margin_m", self.tracking_margin_m),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def boundary_margin_m(self) -> float:
        return float(self.wall_margin_m) + float(self.boundary_reserve_m)


def validate_brake_lookup(speed_knots: Sequence[float], distance_knots: Sequence[float]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """CPU fail-closed check for the monotone ceiling lookup.  Interpolation is forbidden."""
    try:
        speeds = tuple(float(value) for value in speed_knots)
        distances = tuple(float(value) for value in distance_knots)
    except (TypeError, ValueError) as exc:
        raise ValueError("braking lookup is not a finite numeric sequence") from exc
    if len(speeds) == 0 or len(speeds) != len(distances):
        raise ValueError("braking lookup must contain matching non-empty vectors")
    if any(not math.isfinite(value) or value < 0.0 for value in speeds):
        raise ValueError("braking speed knots must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0.0 for value in distances):
        raise ValueError("braking distance knots must be finite and non-negative")
    if any(later <= earlier for earlier, later in zip(speeds, speeds[1:])):
        raise ValueError("braking speed knots must be strictly increasing")
    if any(later < earlier for earlier, later in zip(distances, distances[1:])):
        raise ValueError("braking distance knots must be monotone non-decreasing")
    return speeds, distances


def numpy_soft_envelope(
    arena_lo_xy,
    arena_hi_xy,
    bars_half_extents_xy,
    support_xy,
    spec: SoftEnvelopeSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return admissible centre bounds and inflated bar half extents."""
    spec.validate()
    lo = np.asarray(arena_lo_xy, dtype=np.float64)
    hi = np.asarray(arena_hi_xy, dtype=np.float64)
    support = np.asarray(support_xy, dtype=np.float64)
    half = np.asarray(bars_half_extents_xy, dtype=np.float64)
    if lo.shape != (2,) or hi.shape != (2,) or support.shape != (2,):
        raise ValueError("arena bounds and support must be XY pairs")
    half = half.reshape((-1, 2))
    if (
        not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or not np.isfinite(support).all()
        or not np.isfinite(half).all()
        or np.any(hi <= lo)
        or np.any(support < 0.0)
        or np.any(half < 0.0)
    ):
        raise ValueError("invalid soft-envelope geometry")
    boundary = spec.boundary_margin_m
    return (
        lo + boundary + support,
        hi - boundary - support,
        half + support[None, :] + float(spec.tracking_margin_m),
    )


def torch_soft_envelope(
    arena_lo_xy,
    arena_hi_xy,
    bars_half_extents_xy,
    support_xy,
    spec: SoftEnvelopeSpec,
):
    """Torch equivalent of :func:`numpy_soft_envelope`, preserving device and dtype."""
    spec.validate()
    if arena_lo_xy.ndim != 2 or arena_lo_xy.shape[1] != 2:
        raise ValueError("arena bounds must have shape [N,2]")
    if arena_hi_xy.shape != arena_lo_xy.shape or support_xy.shape != arena_lo_xy.shape:
        raise ValueError("arena bounds and support must have matching shapes")
    if (
        bars_half_extents_xy.ndim != 3
        or bars_half_extents_xy.shape[0] != arena_lo_xy.shape[0]
        or bars_half_extents_xy.shape[2] != 2
    ):
        raise ValueError("bar half extents must have shape [N,B,2]")
    finite = (
        torch.isfinite(arena_lo_xy).all(dim=1)
        & torch.isfinite(arena_hi_xy).all(dim=1)
        & torch.isfinite(support_xy).all(dim=1)
        & (support_xy >= 0.0).all(dim=1)
        & (arena_hi_xy > arena_lo_xy).all(dim=1)
        & torch.isfinite(bars_half_extents_xy).all(dim=(1, 2))
        & (bars_half_extents_xy >= 0.0).all(dim=(1, 2))
    )
    boundary = spec.boundary_margin_m
    lo = arena_lo_xy + boundary + support_xy
    hi = arena_hi_xy - boundary - support_xy
    half = bars_half_extents_xy + support_xy.unsqueeze(1) + float(spec.tracking_margin_m)
    # Invalid rows get an empty open set so later queries fail closed without raising on GPU.
    lo = torch.where(finite.unsqueeze(1), lo, torch.ones_like(lo))
    hi = torch.where(finite.unsqueeze(1), hi, torch.zeros_like(hi))
    half = torch.where(finite.view(-1, 1, 1), half, half + 1.0e9)
    return lo, hi, half


def _numpy_segment_hits_closed_aabb(p0, p1, lo, hi, epsilon=1e-12) -> bool:
    direction = p1 - p0
    enter, leave = 0.0, 1.0
    for axis in (0, 1):
        if abs(direction[axis]) <= epsilon:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False
            continue
        ta = (lo[axis] - p0[axis]) / direction[axis]
        tb = (hi[axis] - p0[axis]) / direction[axis]
        if ta > tb:
            ta, tb = tb, ta
        enter = max(enter, ta)
        leave = min(leave, tb)
        if enter > leave:
            return False
    return enter <= leave and leave >= 0.0 and enter <= 1.0


def numpy_segments_soft_safe(p0, p1, admissible_lo, admissible_hi, bars_xy, inflated_half) -> bool:
    """Continuous closed-AABB safety for one NumPy segment.  Malformed geometry is unsafe."""
    try:
        p0 = np.asarray(p0, dtype=np.float64).reshape(2)
        p1 = np.asarray(p1, dtype=np.float64).reshape(2)
        admissible_lo = np.asarray(admissible_lo, dtype=np.float64).reshape(2)
        admissible_hi = np.asarray(admissible_hi, dtype=np.float64).reshape(2)
        bars = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
        half = np.asarray(inflated_half, dtype=np.float64).reshape((-1, 2))
    except (TypeError, ValueError):
        return False
    if (
        not np.isfinite(p0).all()
        or not np.isfinite(p1).all()
        or not np.isfinite(admissible_lo).all()
        or not np.isfinite(admissible_hi).all()
        or not np.isfinite(bars).all()
        or not np.isfinite(half).all()
        or np.any(admissible_hi <= admissible_lo)
        or bars.shape != half.shape
        or np.any(half < 0.0)
    ):
        return False
    if np.any(p0 <= admissible_lo) or np.any(p0 >= admissible_hi):
        return False
    if np.any(p1 <= admissible_lo) or np.any(p1 >= admissible_hi):
        return False
    for center, extent in zip(bars, half):
        if _numpy_segment_hits_closed_aabb(p0, p1, center - extent, center + extent):
            return False
    return True


def _torch_segment_hits_closed_aabb(p0, p1, centers, half, epsilon=1e-9):
    """Vectorised slab test for [N,C,2] segments against [N,B,2] closed boxes."""
    if p0.shape != p1.shape or p0.ndim != 3 or p0.shape[2] != 2:
        raise ValueError("segments must have shape [N,C,2]")
    if centers.ndim != 3 or centers.shape[0] != p0.shape[0] or centers.shape[2] != 2:
        raise ValueError("centers must have shape [N,B,2]")
    if half.shape != centers.shape:
        raise ValueError("half extents must match centers")
    if centers.shape[1] == 0:
        return torch.zeros(p0.shape[:2], dtype=torch.bool, device=p0.device)
    start = p0.unsqueeze(2)
    direction = (p1 - p0).unsqueeze(2)
    box_lo = centers.unsqueeze(1) - half.unsqueeze(1)
    box_hi = centers.unsqueeze(1) + half.unsqueeze(1)
    parallel = direction.abs() <= float(epsilon)
    parallel_inside = (start >= box_lo) & (start <= box_hi)
    axis_possible = (~parallel) | parallel_inside
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    ta = (box_lo - start) / safe_direction
    tb = (box_hi - start) / safe_direction
    axis_enter = torch.minimum(ta, tb)
    axis_exit = torch.maximum(ta, tb)
    axis_enter = torch.where(parallel, torch.full_like(axis_enter, float("-inf")), axis_enter)
    axis_exit = torch.where(parallel, torch.full_like(axis_exit, float("inf")), axis_exit)
    enter = axis_enter.amax(dim=3)
    leave = axis_exit.amin(dim=3)
    hit = axis_possible.all(dim=3) & (enter <= leave) & (leave >= 0.0) & (enter <= 1.0)
    return hit.any(dim=2)


def torch_segments_soft_safe(p0, p1, admissible_lo, admissible_hi, bars_xy, inflated_half):
    """Continuous closed-AABB safety for a candidate batch [N,C,2]."""
    finite = (
        torch.isfinite(p0).all(dim=2)
        & torch.isfinite(p1).all(dim=2)
        & torch.isfinite(admissible_lo).all(dim=1).unsqueeze(1)
        & torch.isfinite(admissible_hi).all(dim=1).unsqueeze(1)
        & (admissible_hi > admissible_lo).all(dim=1).unsqueeze(1)
        & torch.isfinite(bars_xy).all(dim=(1, 2)).unsqueeze(1)
        & torch.isfinite(inflated_half).all(dim=(1, 2)).unsqueeze(1)
        & (inflated_half >= 0.0).all(dim=(1, 2)).unsqueeze(1)
    )
    inside = (
        (p0 > admissible_lo.unsqueeze(1))
        & (p0 < admissible_hi.unsqueeze(1))
        & (p1 > admissible_lo.unsqueeze(1))
        & (p1 < admissible_hi.unsqueeze(1))
    ).all(dim=2)
    return finite & inside & ~_torch_segment_hits_closed_aabb(p0, p1, bars_xy, inflated_half)


def numpy_ceiling_stop_distance(speed, speed_knots: Sequence[float], distance_knots: Sequence[float]):
    speeds, distances = validate_brake_lookup(speed_knots, distance_knots)
    speed = np.asarray(speed, dtype=np.float64)
    index = np.searchsorted(np.asarray(speeds), speed, side="left")
    valid = np.isfinite(speed) & (index < len(speeds))
    clamped = np.minimum(index, len(speeds) - 1)
    distance = np.asarray(distances, dtype=np.float64)[clamped]
    distance = np.where(valid, distance, np.full_like(distance, np.nan))
    return distance, valid


def ceiling_stop_distance(speed, speed_knots: Sequence[float], distance_knots: Sequence[float]):
    """Conservative GPU ceiling lookup; values above the final certified knot are invalid."""
    speeds = torch.as_tensor(speed_knots, dtype=speed.dtype, device=speed.device)
    distances = torch.as_tensor(distance_knots, dtype=speed.dtype, device=speed.device)
    if speeds.ndim != 1 or distances.shape != speeds.shape or speeds.numel() == 0:
        raise ValueError("braking lookup must contain matching non-empty vectors")
    # Values are validated once against the raw receipt by NavRLTask.  Deliberately avoid
    # tensor truth conversion here: that would synchronize the GPU on every policy interval.
    index = torch.bucketize(speed.contiguous(), speeds, right=False)
    valid = torch.isfinite(speed) & (index < speeds.numel())
    clamped = index.clamp(max=speeds.numel() - 1)
    return distances[clamped], valid


def terminal_stop_certificate(
    position_xy,
    velocity_xy,
    admissible_lo,
    admissible_hi,
    bars_xy,
    inflated_half,
    speed_knots,
    distance_knots,
    lateral_tube_m: float,
):
    """Certify a zero-command stop segment for [N,C,2] candidate terminal states."""
    lateral = float(lateral_tube_m)
    if not math.isfinite(lateral) or lateral < 0.0:
        raise ValueError("lateral stopping tube must be finite and non-negative")
    finite = torch.isfinite(velocity_xy).all(dim=2) & torch.isfinite(position_xy).all(dim=2)
    speed = velocity_xy.norm(dim=2)
    distance, covered = ceiling_stop_distance(speed, speed_knots, distance_knots)
    direction = velocity_xy / speed.unsqueeze(2).clamp(min=1e-9)
    stop = position_xy + direction * distance.unsqueeze(2)
    safe = torch_segments_soft_safe(
        position_xy,
        stop,
        admissible_lo + lateral,
        admissible_hi - lateral,
        bars_xy,
        inflated_half + lateral,
    )
    return safe & covered & finite, stop, distance
