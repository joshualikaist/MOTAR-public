"""Deterministic, simulator-independent global routes for the physical NavRL target.

The planner is intentionally outside the policy.  It gives the *target actor* a collision-safe
velocity reference; it never exposes ground-truth geometry to the pursuer actor or critic.

Planning is performed only at reset, goal replacement, or explicit invalidation.  Selected
environment tensors are copied to CPU once per planning batch, while route following uses a
padded GPU waypoint cache and remains vectorised at every ordinary RL step.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib.util
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _load_target_route_geometry():
    name = "_navrl_target_route_geometry_standalone"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("target_route_geometry.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GEO = _load_target_route_geometry()


TARGET_ROUTE_MODE_OFF = "off"
TARGET_ROUTE_MODE_GLOBAL_ASTAR = "global_astar_v1"
# Keep the legacy route-arm tuple immutable; the recovery lineage is accepted explicitly by
# NavRLTask and never aliases the v1 mode.
TARGET_ROUTE_MODE_RECOVERY = "global_astar_recovery_v2"
TARGET_ROUTE_MODE_BRAKING_V3 = "global_astar_braking_v3"
TARGET_ROUTE_MODES = (TARGET_ROUTE_MODE_OFF, TARGET_ROUTE_MODE_GLOBAL_ASTAR)
TARGET_ROUTE_MODEL = "physx_ref5in_6dof_global_astar_aabb_v1"
TARGET_ROUTE_RECOVERY_MODEL = "physx_ref5in_6dof_global_astar_aabb_v2_two_envelope_recovery"
TARGET_ROUTE_RECOVERY_SCHEMA = "navrl_target_route_two_envelope_recovery_v1"
TARGET_ROUTE_BRAKING_V3_MODEL = "physx_ref5in_6dof_global_astar_aabb_v3_braking"
TARGET_ROUTE_BRAKING_V3_SCHEMA = "navrl_target_route_braking_aware_v3"
TARGET_ROUTE_HARD_EPSILON_M = 1e-4
# Derived reachable-tube reserve for the prospective CONNECT RL-step chord: the fixed 0.1 s
# command interval and the 45-degree horizontal acceleration bound give
# ceil(g*tan(45deg)*0.1^2/8) = 0.0122625 m, rounded upward.
TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M = 0.0123

RECOVERY_NORMAL = 0
RECOVERY_BRAKE = 1
RECOVERY_CONNECT = 2
RECOVERY_ROUTE = 3
RECOVERY_NO_CONNECTOR = 4

BRAKING_V3_ROUTE = 0
BRAKING_V3_PREBRAKE = 1
BRAKING_V3_STOP_TURN_GO = 2

_NEIGHBORS = (
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class RoutePlannerConfig:
    resolution_m: float = 0.25
    tracking_margin_m: float = 0.45
    boundary_margin_m: float = 1.25
    max_expansions: int = 50000
    max_waypoints: int = 128
    replan_cooldown_steps: int = 10
    goal_tolerance_m: float = 0.05
    goal_exclusion_radius_m: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("route resolution must be finite and positive")
        if not math.isfinite(self.tracking_margin_m) or self.tracking_margin_m < 0.0:
            raise ValueError("route tracking margin must be finite and non-negative")
        if not math.isfinite(self.boundary_margin_m) or self.boundary_margin_m < 0.0:
            raise ValueError("route boundary margin must be finite and non-negative")
        if self.max_expansions <= 0 or self.max_waypoints < 2:
            raise ValueError("route max_expansions and max_waypoints must be positive")
        if self.replan_cooldown_steps < 1:
            raise ValueError("route replan cooldown must be at least one step")
        if not math.isfinite(self.goal_tolerance_m) or self.goal_tolerance_m < 0.0:
            raise ValueError("route goal tolerance must be finite and non-negative")
        if (
            not math.isfinite(self.goal_exclusion_radius_m)
            or self.goal_exclusion_radius_m <= 0.0
        ):
            raise ValueError("route goal exclusion radius must be finite and positive")


@dataclass(frozen=True)
class RoutePlan:
    status: str
    waypoints_xy: np.ndarray
    expanded_nodes: int
    raw_grid_nodes: int
    smoothed_nodes: int
    path_length_m: float

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def conservative_xy_support_from_box(box_xyz: Sequence[float]) -> np.ndarray:
    """Return a finite XY support envelope safe under every future tilt/yaw orientation.

    The world-axis support of an oriented box cannot exceed its 3-D half-diagonal. Using this
    circumscribed radius on both horizontal axes is slightly conservative at the declared 45-deg
    tilt limit, but unlike current-pose support it cannot become invalid after the route is cached.
    """
    size = np.asarray(box_xyz, dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("physical target box_xyz must contain three finite positive sizes")
    radius = 0.5 * float(np.linalg.norm(size))
    return np.array([radius, radius], dtype=np.float64)


def _finite_xy(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite XY pair")
    return result


def _segment_intersects_closed_aabb(p0, p1, lo, hi, epsilon=1e-12) -> bool:
    """Exact slab test. Touching an inflated AABB is unsafe (closed-set collision)."""
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


def segment_is_safe(
    p0,
    p1,
    admissible_lo,
    admissible_hi,
    bars_xy,
    inflated_half_extents_xy,
) -> bool:
    """Continuous line-of-sight contract used by smoothing and unit tests."""
    p0 = _finite_xy(p0, "p0")
    p1 = _finite_xy(p1, "p1")
    admissible_lo = _finite_xy(admissible_lo, "admissible_lo")
    admissible_hi = _finite_xy(admissible_hi, "admissible_hi")
    if np.any(admissible_hi <= admissible_lo):
        return False
    # The admissible arena is convex; endpoint inclusion proves the entire segment is inside.
    if np.any(p0 <= admissible_lo) or np.any(p0 >= admissible_hi):
        return False
    if np.any(p1 <= admissible_lo) or np.any(p1 >= admissible_hi):
        return False
    bars_xy = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
    half = np.asarray(inflated_half_extents_xy, dtype=np.float64).reshape((-1, 2))
    if bars_xy.shape != half.shape or not np.isfinite(bars_xy).all() or not np.isfinite(half).all():
        return False
    if np.any(half < 0.0):
        return False
    for center, extent in zip(bars_xy, half):
        if _segment_intersects_closed_aabb(p0, p1, center - extent, center + extent):
            return False
    return True


def _point_segment_distance(point, p0, p1) -> float:
    delta = p1 - p0
    denominator = float(np.dot(delta, delta))
    fraction = (
        float(np.clip(np.dot(point - p0, delta) / denominator, 0.0, 1.0))
        if denominator > 1e-24
        else 0.0
    )
    return float(np.linalg.norm(point - (p0 + fraction * delta)))


def _point_aabb_distance(point, lo, hi) -> float:
    offset = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(offset))


def _segment_aabb_distance(p0, p1, lo, hi) -> float:
    """Exact Euclidean clearance between a segment and a closed 2-D AABB."""
    if _segment_intersects_closed_aabb(p0, p1, lo, hi):
        return 0.0
    corners = (
        np.array([lo[0], lo[1]]),
        np.array([lo[0], hi[1]]),
        np.array([hi[0], lo[1]]),
        np.array([hi[0], hi[1]]),
    )
    return min(
        _point_aabb_distance(p0, lo, hi),
        _point_aabb_distance(p1, lo, hi),
        *(_point_segment_distance(corner, p0, p1) for corner in corners),
    )


def route_handoff_clearance_certificates(
    waypoints_xy,
    admissible_lo,
    admissible_hi,
    bars_xy,
    inflated_half_extents_xy,
    safety_epsilon_m: float = TARGET_ROUTE_HARD_EPSILON_M,
) -> np.ndarray:
    """Certify safe early/overshoot handoff balls for every outgoing route segment.

    If ``q`` is within radius ``r`` of waypoint ``p``, the segment ``q -> next`` stays within
    ``r`` of the planned segment ``p -> next`` under equal interpolation.  Therefore any radius
    strictly below that planned segment's exact obstacle/boundary clearance proves the connector
    safe without retaining bar geometry in the ordinary GPU control path.
    """
    points = np.asarray(waypoints_xy, dtype=np.float64).reshape((-1, 2))
    lo = _finite_xy(admissible_lo, "admissible_lo")
    hi = _finite_xy(admissible_hi, "admissible_hi")
    bars = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
    half = np.asarray(inflated_half_extents_xy, dtype=np.float64).reshape((-1, 2))
    epsilon = float(safety_epsilon_m)
    if (
        points.shape[0] < 2
        or bars.shape != half.shape
        or not np.isfinite(points).all()
        or not np.isfinite(bars).all()
        or not np.isfinite(half).all()
        or np.any(half < 0.0)
        or np.any(hi <= lo)
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError("invalid route handoff certificate geometry")
    result = np.zeros(points.shape[0], dtype=np.float64)
    for index, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        if not segment_is_safe(p0, p1, lo, hi, bars, half):
            raise ValueError("cannot certify an unsafe planned route segment")
        boundary_clearance = min(
            float(np.min(p0 - lo)),
            float(np.min(p1 - lo)),
            float(np.min(hi - p0)),
            float(np.min(hi - p1)),
        )
        obstacle_clearance = min(
            (
                _segment_aabb_distance(p0, p1, center - extent, center + extent)
                for center, extent in zip(bars, half)
            ),
            default=float("inf"),
        )
        result[index] = max(0.0, min(boundary_clearance, obstacle_clearance) - epsilon)
    # The final waypoint has no outgoing connector; its radius is deliberately zero and unused.
    return result


def nearest_soft_free_anchor(
    point,
    bars_xy,
    bar_half_extents_xy,
    arena_lo_xy,
    arena_hi_xy,
    support_xy,
    hard_boundary_margin_m,
    soft_boundary_margin_m,
    resolution_m=0.25,
    radius_cells=3,
    tracking_margin_m=0.45,
    soft_hysteresis_m=0.25,
    hard_epsilon_m=TARGET_ROUTE_HARD_EPSILON_M,
):
    """Find the deterministic nearest 7x7 soft-free anchor with a hard-safe connector.

    This is intentionally the same bounded search used by route-recovery forensics.  It is
    geometry-only: it does not move an actor and returns no fallback point on malformed input.
    Soft occupancy is the planner's closed AABB inflation; the connector uses the exact slab
    segment test against the *hard* inflation.
    """
    try:
        point = _finite_xy(point, "point")
        bars = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
        bar_half = np.asarray(bar_half_extents_xy, dtype=np.float64).reshape((-1, 2))
        lo = _finite_xy(arena_lo_xy, "arena_lo_xy")
        hi = _finite_xy(arena_hi_xy, "arena_hi_xy")
        support = _finite_xy(support_xy, "support_xy")
        hard_boundary = float(hard_boundary_margin_m)
        soft_boundary = float(soft_boundary_margin_m)
        resolution = float(resolution_m)
        radius = int(radius_cells)
        tracking = float(tracking_margin_m)
        hysteresis = float(soft_hysteresis_m)
        hard_epsilon = float(hard_epsilon_m)
    except (TypeError, ValueError, FloatingPointError):
        return {"exists": None, "xy_m": None, "cell_ij": None,
                "distance_m": None, "hard_connector_safe": None,
                "hard_connector_clearance_m": None}
    if (
        bars.shape != bar_half.shape
        or not np.isfinite(bars).all()
        or not np.isfinite(bar_half).all()
        or np.any(bar_half < 0.0)
        or np.any(support < 0.0)
        or np.any(hi <= lo)
        or not math.isfinite(hard_boundary)
        or not math.isfinite(soft_boundary)
        or hard_boundary < 0.0
        or soft_boundary < 0.0
        or not math.isfinite(resolution)
        or resolution <= 0.0
        or radius < 0
        or not math.isfinite(tracking)
        or tracking < 0.0
        or not math.isfinite(hysteresis)
        or hysteresis < 0.0
        or not math.isfinite(hard_epsilon)
        or hard_epsilon < 0.0
    ):
        return {"exists": None, "xy_m": None, "cell_ij": None,
                "distance_m": None, "hard_connector_safe": None,
                "hard_connector_clearance_m": None}
    extent = hi - lo
    shape = np.maximum(1, np.floor(extent / resolution).astype(np.int64))
    axes = tuple(
        lo[k] + (np.arange(int(shape[k]), dtype=np.float64) + 0.5) * resolution
        for k in (0, 1)
    )
    base = tuple(int(np.argmin(np.abs(axes[k] - point[k]))) for k in (0, 1))
    hard_lo = lo + hard_boundary + support + hard_epsilon
    hard_hi = hi - hard_boundary - support - hard_epsilon
    # Anchor clearance includes the registered release hysteresis.  A merely soft-free anchor
    # could otherwise leave CONNECT at its threshold and immediately re-enter it.
    soft_lo = lo + soft_boundary + support + hysteresis
    soft_hi = hi - soft_boundary - support - hysteresis
    hard_half = bar_half + support[None, :] + hard_epsilon
    soft_half = hard_half + tracking + hysteresis
    candidates = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            i, j = base[0] + di, base[1] + dj
            if not (0 <= i < shape[0] and 0 <= j < shape[1]):
                continue
            anchor = np.array([axes[0][i], axes[1][j]], dtype=np.float64)
            soft_free = bool(np.all(anchor > soft_lo) and np.all(anchor < soft_hi))
            soft_free &= all(
                not _segment_intersects_closed_aabb(anchor, anchor, center - extent, center + extent)
                for center, extent in zip(bars, soft_half)
            )
            if soft_free and segment_is_safe(point, anchor, hard_lo, hard_hi, bars, hard_half):
                candidates.append((float(np.dot(anchor - point, anchor - point)), i, j, anchor))
    if not candidates:
        return {"exists": False, "xy_m": None, "cell_ij": None,
                "distance_m": None, "hard_connector_safe": False,
                "hard_connector_clearance_m": None}
    _, i, j, anchor = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    connector = min(
        (_segment_aabb_distance(point, anchor, center - extent, center + extent)
         for center, extent in zip(bars, hard_half)),
        default=float("inf"),
    )
    connector = min(
        connector,
        float(np.minimum(anchor - hard_lo, hard_hi - anchor).min()),
        float(np.minimum(point - hard_lo, hard_hi - point).min()),
    )
    return {
        "exists": True,
        "xy_m": anchor.tolist(),
        "cell_ij": [int(i), int(j)],
        "distance_m": float(np.linalg.norm(anchor - point)),
        "hard_connector_safe": True,
        "hard_connector_clearance_m": float(connector),
    }


class DeterministicAStarRoutePlanner:
    """CPU A* with exact AABB inflation and safe line-of-sight smoothing."""

    def __init__(self, config: RoutePlannerConfig):
        config.validate()
        self.config = config

    def _grid(self, arena_lo, arena_hi):
        extent = arena_hi - arena_lo
        shape = np.maximum(1, np.floor(extent / self.config.resolution_m).astype(np.int64))
        axes = tuple(
            arena_lo[axis]
            + (np.arange(int(shape[axis]), dtype=np.float64) + 0.5)
            * self.config.resolution_m
            for axis in (0, 1)
        )
        return axes, (int(shape[0]), int(shape[1]))

    @staticmethod
    def _cell(point, axes):
        return tuple(
            int(np.argmin(np.abs(axis - point[k])))
            for k, axis in enumerate(axes)
        )

    def _anchor_cell(
        self, point, axes, free, admissible_lo, admissible_hi, bars, inflated_half
    ):
        """Map a continuous safe endpoint to a nearby grid node with a proven safe connector.

        A safe point may lie just outside the inflated AABB while its nearest cell centre lies
        inside it. Rejecting that endpoint is a rasterisation artefact, not a geometric failure.
        Search a bounded deterministic neighbourhood and retain the continuous slab test.
        """
        base_i, base_j = self._cell(point, axes)
        candidates = []
        for di in range(-3, 4):
            for dj in range(-3, 4):
                i, j = base_i + di, base_j + dj
                if 0 <= i < free.shape[0] and 0 <= j < free.shape[1] and free[i, j]:
                    cell_point = np.array([axes[0][i], axes[1][j]], dtype=np.float64)
                    candidates.append((float(np.sum((cell_point - point) ** 2)), i, j))
        for _, i, j in sorted(candidates):
            cell_point = np.array([axes[0][i], axes[1][j]], dtype=np.float64)
            if segment_is_safe(
                point, cell_point, admissible_lo, admissible_hi, bars, inflated_half
            ):
                return (i, j)
        return None

    def plan(
        self,
        start_xy,
        goal_xy,
        bars_xy,
        bars_half_extents_xy,
        arena_lo_xy,
        arena_hi_xy,
        target_support_xy,
    ) -> RoutePlan:
        try:
            start = _finite_xy(start_xy, "start_xy")
            goal = _finite_xy(goal_xy, "goal_xy")
            arena_lo = _finite_xy(arena_lo_xy, "arena_lo_xy")
            arena_hi = _finite_xy(arena_hi_xy, "arena_hi_xy")
            support = _finite_xy(target_support_xy, "target_support_xy")
            bars = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
            half = np.asarray(bars_half_extents_xy, dtype=np.float64).reshape((-1, 2))
        except (TypeError, ValueError):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)
        if (
            bars.shape != half.shape
            or not np.isfinite(bars).all()
            or not np.isfinite(half).all()
            or np.any(half < 0.0)
            or np.any(support < 0.0)
            or np.any(arena_hi <= arena_lo)
        ):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)

        admissible_lo = arena_lo + self.config.boundary_margin_m + support
        admissible_hi = arena_hi - self.config.boundary_margin_m - support
        inflated_half = half + support[None, :] + self.config.tracking_margin_m
        if not segment_is_safe(start, start, admissible_lo, admissible_hi, bars, inflated_half):
            return RoutePlan("unsafe_start", np.empty((0, 2)), 0, 0, 0, 0.0)
        if not segment_is_safe(goal, goal, admissible_lo, admissible_hi, bars, inflated_half):
            return RoutePlan("unsafe_goal", np.empty((0, 2)), 0, 0, 0, 0.0)
        if segment_is_safe(start, goal, admissible_lo, admissible_hi, bars, inflated_half):
            points = np.stack((start, goal))
            return RoutePlan("ok", points, 0, 2, 2, float(np.linalg.norm(goal - start)))

        axes, shape = self._grid(arena_lo, arena_hi)
        xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
        free = (
            (xx > admissible_lo[0])
            & (xx < admissible_hi[0])
            & (yy > admissible_lo[1])
            & (yy < admissible_hi[1])
        )
        for center, extent in zip(bars, inflated_half):
            free &= ~(
                (np.abs(xx - center[0]) <= extent[0])
                & (np.abs(yy - center[1]) <= extent[1])
            )
        start_cell = self._anchor_cell(
            start, axes, free, admissible_lo, admissible_hi, bars, inflated_half
        )
        goal_cell = self._anchor_cell(
            goal, axes, free, admissible_lo, admissible_hi, bars, inflated_half
        )
        if start_cell is None:
            return RoutePlan("unsafe_start_cell", np.empty((0, 2)), 0, 0, 0, 0.0)
        if goal_cell is None:
            return RoutePlan("unsafe_goal_cell", np.empty((0, 2)), 0, 0, 0, 0.0)

        g_score = np.full(shape, np.inf, dtype=np.float64)
        parent_i = np.full(shape, -1, dtype=np.int32)
        parent_j = np.full(shape, -1, dtype=np.int32)
        g_score[start_cell] = 0.0

        def heuristic(i, j):
            return math.hypot(i - goal_cell[0], j - goal_cell[1])

        queue = [(heuristic(*start_cell), 0.0, start_cell[0], start_cell[1])]
        expanded = 0
        while queue:
            _, cost, i, j = heapq.heappop(queue)
            if cost != g_score[i, j]:
                continue
            expanded += 1
            if expanded > self.config.max_expansions:
                return RoutePlan("expansion_limit", np.empty((0, 2)), expanded, 0, 0, 0.0)
            if (i, j) == goal_cell:
                break
            for di, dj, step_cost in _NEIGHBORS:
                ii, jj = i + di, j + dj
                if not (0 <= ii < shape[0] and 0 <= jj < shape[1] and free[ii, jj]):
                    continue
                # No diagonal corner cutting. Exact smoothing below applies the stronger AABB test.
                if di and dj and (not free[i + di, j] or not free[i, j + dj]):
                    continue
                candidate = cost + step_cost
                if candidate < g_score[ii, jj]:
                    g_score[ii, jj] = candidate
                    parent_i[ii, jj], parent_j[ii, jj] = i, j
                    priority = candidate + heuristic(ii, jj)
                    heapq.heappush(queue, (priority, candidate, ii, jj))
        if not np.isfinite(g_score[goal_cell]):
            return RoutePlan("no_path", np.empty((0, 2)), expanded, 0, 0, 0.0)

        cells = [goal_cell]
        cursor = goal_cell
        while cursor != start_cell:
            cursor = (int(parent_i[cursor]), int(parent_j[cursor]))
            if cursor[0] < 0 or cursor[1] < 0:
                return RoutePlan("parent_chain_invalid", np.empty((0, 2)), expanded, 0, 0, 0.0)
            cells.append(cursor)
        cells.reverse()
        # Preserve both endpoint anchor cells. Omitting an anchor can join the continuous
        # endpoint directly to the *second* grid node and cut an inflated AABB corner even though
        # endpoint->anchor and anchor->neighbour are each safe.
        raw = [start]
        for i, j in cells:
            point = np.array([axes[0][i], axes[1][j]])
            if np.linalg.norm(point - raw[-1]) > 1e-9:
                raw.append(point)
        if np.linalg.norm(goal - raw[-1]) > 1e-9:
            raw.append(goal)

        # Greedy farthest-visible smoothing. Every accepted edge passes the continuous exact-AABB
        # contract; no waypoint is removed merely because the raster path looked clear.
        smoothed = [raw[0]]
        anchor = 0
        while anchor < len(raw) - 1:
            candidate = len(raw) - 1
            while candidate > anchor + 1 and not segment_is_safe(
                raw[anchor], raw[candidate], admissible_lo, admissible_hi, bars, inflated_half
            ):
                candidate -= 1
            if not segment_is_safe(
                raw[anchor], raw[candidate], admissible_lo, admissible_hi, bars, inflated_half
            ):
                return RoutePlan("smoothing_invalid", np.empty((0, 2)), expanded, len(raw), 0, 0.0)
            smoothed.append(raw[candidate])
            anchor = candidate
        points = np.asarray(smoothed, dtype=np.float64)
        if points.shape[0] > self.config.max_waypoints:
            return RoutePlan("waypoint_limit", np.empty((0, 2)), expanded, len(raw), len(points), 0.0)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return RoutePlan("ok", points, expanded, len(raw), len(points), length)

    def plan_to_connected_goal(
        self,
        start_xy,
        bars_xy,
        bars_half_extents_xy,
        arena_lo_xy,
        arena_hi_xy,
        target_support_xy,
        min_goal_distance_m: float,
        selector: float,
        excluded_goal_xy=None,
        exclusion_radius_m: float = 0.0,
    ) -> RoutePlan:
        """Choose a sufficiently distant goal proven reachable by the same planner contract.

        Candidate order is deterministic for ``selector``. A returned goal is in the start's
        connected component by construction because ``plan`` has produced a valid path to it.
        Failure returns no route; it never falls back to an unproved random waypoint.
        """
        try:
            start = _finite_xy(start_xy, "start_xy")
            arena_lo = _finite_xy(arena_lo_xy, "arena_lo_xy")
            arena_hi = _finite_xy(arena_hi_xy, "arena_hi_xy")
            selector = float(selector)
            minimum = float(min_goal_distance_m)
            exclusion_radius = float(exclusion_radius_m)
            excluded_goal = (
                None
                if excluded_goal_xy is None
                else _finite_xy(excluded_goal_xy, "excluded_goal_xy")
            )
        except (TypeError, ValueError):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)
        if (
            not math.isfinite(selector)
            or not math.isfinite(minimum)
            or minimum <= 0.0
            or not math.isfinite(exclusion_radius)
            or exclusion_radius < 0.0
            or (excluded_goal is not None and exclusion_radius <= 0.0)
        ):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)
        try:
            bars = np.asarray(bars_xy, dtype=np.float64).reshape((-1, 2))
            half = np.asarray(bars_half_extents_xy, dtype=np.float64).reshape((-1, 2))
            support = _finite_xy(target_support_xy, "target_support_xy")
        except (TypeError, ValueError):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)
        if (
            bars.shape != half.shape
            or not np.isfinite(bars).all()
            or not np.isfinite(half).all()
            or np.any(half < 0.0)
            or np.any(support < 0.0)
        ):
            return RoutePlan("invalid_input", np.empty((0, 2)), 0, 0, 0, 0.0)

        admissible_lo = arena_lo + self.config.boundary_margin_m + support
        admissible_hi = arena_hi - self.config.boundary_margin_m - support
        inflated_half = half + support[None, :] + self.config.tracking_margin_m
        if not segment_is_safe(start, start, admissible_lo, admissible_hi, bars, inflated_half):
            return RoutePlan("unsafe_start", np.empty((0, 2)), 0, 0, 0, 0.0)
        axes, shape = self._grid(arena_lo, arena_hi)
        xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
        free = (
            (xx > admissible_lo[0])
            & (xx < admissible_hi[0])
            & (yy > admissible_lo[1])
            & (yy < admissible_hi[1])
        )
        for center, extent in zip(bars, inflated_half):
            free &= ~(
                (np.abs(xx - center[0]) <= extent[0])
                & (np.abs(yy - center[1]) <= extent[1])
            )
        start_cell = self._anchor_cell(
            start, axes, free, admissible_lo, admissible_hi, bars, inflated_half
        )
        if start_cell is None:
            return RoutePlan("unsafe_start_cell", np.empty((0, 2)), 0, 0, 0, 0.0)

        # One deterministic Dijkstra expansion gives both the connected component and the parent
        # tree. Unlike the previous candidate loop, occupancy and search are never repeated up to
        # 64 times for one reset.
        distance = np.full(shape, np.inf, dtype=np.float64)
        parent_i = np.full(shape, -1, dtype=np.int32)
        parent_j = np.full(shape, -1, dtype=np.int32)
        distance[start_cell] = 0.0
        queue = [(0.0, start_cell[0], start_cell[1])]
        expanded = 0
        while queue:
            cost, i, j = heapq.heappop(queue)
            if cost != distance[i, j]:
                continue
            expanded += 1
            if expanded > self.config.max_expansions:
                return RoutePlan(
                    "expansion_limit", np.empty((0, 2)), expanded, 0, 0, 0.0
                )
            for di, dj, step_cost in _NEIGHBORS:
                ii, jj = i + di, j + dj
                if not (0 <= ii < shape[0] and 0 <= jj < shape[1] and free[ii, jj]):
                    continue
                if di and dj and (not free[i + di, j] or not free[i, j + dj]):
                    continue
                candidate = cost + step_cost
                if candidate < distance[ii, jj]:
                    distance[ii, jj] = candidate
                    parent_i[ii, jj], parent_j[ii, jj] = i, j
                    heapq.heappush(queue, (candidate, ii, jj))

        reachable_i, reachable_j = np.nonzero(np.isfinite(distance))
        reachable_points = np.stack(
            (axes[0][reachable_i], axes[1][reachable_j]), axis=1
        )
        distant = np.linalg.norm(reachable_points - start[None, :], axis=1) >= minimum
        if excluded_goal is not None:
            distant &= (
                np.linalg.norm(reachable_points - excluded_goal[None, :], axis=1)
                > exclusion_radius
            )
        reachable_i, reachable_j = reachable_i[distant], reachable_j[distant]
        if not len(reachable_i):
            status = "no_alternative_goal" if excluded_goal is not None else "no_connected_goal"
            return RoutePlan(status, np.empty((0, 2)), 0, 0, 0, 0.0)
        choice = min(len(reachable_i) - 1, int(math.floor((selector % 1.0) * len(reachable_i))))
        goal_cell = (int(reachable_i[choice]), int(reachable_j[choice]))
        goal = np.array([axes[0][goal_cell[0]], axes[1][goal_cell[1]]])
        cells = [goal_cell]
        cursor = goal_cell
        while cursor != start_cell:
            cursor = (int(parent_i[cursor]), int(parent_j[cursor]))
            if cursor[0] < 0 or cursor[1] < 0:
                return RoutePlan(
                    "parent_chain_invalid", np.empty((0, 2)), expanded, 0, 0, 0.0
                )
            cells.append(cursor)
        cells.reverse()
        raw = [start]
        for i, j in cells:
            point = np.array([axes[0][i], axes[1][j]])
            if np.linalg.norm(point - raw[-1]) > 1e-9:
                raw.append(point)
        smoothed = [raw[0]]
        anchor = 0
        while anchor < len(raw) - 1:
            candidate = len(raw) - 1
            while candidate > anchor + 1 and not segment_is_safe(
                raw[anchor], raw[candidate], admissible_lo, admissible_hi, bars, inflated_half
            ):
                candidate -= 1
            if not segment_is_safe(
                raw[anchor], raw[candidate], admissible_lo, admissible_hi, bars, inflated_half
            ):
                return RoutePlan(
                    "smoothing_invalid", np.empty((0, 2)), expanded, len(raw), 0, 0.0
                )
            smoothed.append(raw[candidate])
            anchor = candidate
        points = np.asarray(smoothed, dtype=np.float64)
        if len(points) > self.config.max_waypoints:
            return RoutePlan(
                "waypoint_limit", np.empty((0, 2)), expanded, len(raw), len(points), 0.0
            )
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return RoutePlan("ok", points, expanded, len(raw), len(points), length)


class BatchedTargetRouteManager:
    """Per-environment cached routes with GPU-vectorised waypoint following."""

    STATUS_CODES = {
        "unplanned": 0,
        "ok": 1,
        "invalid_input": 2,
        "unsafe_start": 3,
        "unsafe_goal": 4,
        "unsafe_start_cell": 5,
        "unsafe_goal_cell": 6,
        "no_path": 7,
        "expansion_limit": 8,
        "parent_chain_invalid": 9,
        "smoothing_invalid": 10,
        "waypoint_limit": 11,
        "local_step_infeasible": 12,
        "support_contract_changed": 13,
        "goal_changed": 14,
        "no_connected_goal": 15,
        "no_alternative_goal": 16,
        "same_goal_reselected": 17,
        "soft_envelope_violation": 18,
        "recovery_no_connector": 19,
        "recovery_hard_breach": 20,
        "recovery_local_infeasible_soft_free": 21,
        "recovery_brake_timeout": 22,
        "recovery_connect_timeout": 23,
    }

    def __init__(self, num_envs: int, device, config: RoutePlannerConfig, *, recovery_enabled=False, braking_v3_enabled=False):
        config.validate()
        if bool(recovery_enabled) and bool(braking_v3_enabled):
            raise ValueError("recovery-v2 and braking-v3 cannot share a manager")
        self.num_envs = int(num_envs)
        self.device = device
        self.config = config
        self.recovery_enabled = bool(recovery_enabled)
        self.braking_v3_enabled = bool(braking_v3_enabled)
        self.planner = DeterministicAStarRoutePlanner(config)
        self.waypoints = torch.zeros(
            (self.num_envs, config.max_waypoints, 2), dtype=torch.float32, device=device
        )
        self.length = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.cursor = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.goal = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.segment_start = torch.zeros_like(self.goal)
        self.handoff_clearance = torch.zeros(
            (self.num_envs, config.max_waypoints), dtype=torch.float32, device=device
        )
        self.completion_reported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.goal_change_reported = torch.zeros_like(self.completion_reported)
        self.support_change_reported = torch.zeros_like(self.completion_reported)
        self.planned_support = torch.zeros_like(self.goal)
        self.next_replan_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.status_code = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.recovery_state = torch.full(
            (self.num_envs,), RECOVERY_NORMAL, dtype=torch.long, device=device
        )
        self.recovery_anchor = torch.zeros_like(self.goal)
        self.recovery_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.recovery_brake_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.recovery_connect_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.recovery_connect_timeout_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.recovery_entries = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_brake_intervals = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_connect_intervals = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_no_connector_count = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_hard_breach_count = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_brake_timeout_count = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_connect_timeout_count = torch.zeros((), dtype=torch.long, device=device)
        self.recovery_route_resumes = torch.zeros((), dtype=torch.long, device=device)
        self.plan_attempts = 0
        self.plan_successes = 0
        self.replan_attempts = 0
        self.no_path_count = 0
        self.invalid_count = 0
        # Runtime counters remain device-side so the normal control path does not
        # force a GPU synchronization every interval.  They are materialized only
        # when bulk-evaluation diagnostics are exported.
        self.runtime_invalid_count = torch.zeros((), dtype=torch.long, device=device)
        self.local_step_invalidations = torch.zeros((), dtype=torch.long, device=device)
        self.support_contract_invalidations = torch.zeros(
            (), dtype=torch.long, device=device
        )
        self.goal_changed_invalidations = torch.zeros(
            (), dtype=torch.long, device=device
        )
        self.goal_completions = torch.zeros((), dtype=torch.long, device=device)
        self.fallback_intervals = torch.zeros((), dtype=torch.long, device=device)
        self.connected_goal_replans = 0
        self.same_goal_reselection_count = 0
        self.expanded_nodes = 0
        self.raw_waypoints = 0
        self.smoothed_waypoints = 0
        self.planning_batches = 0
        self.planning_envs = 0
        self.total_planning_wall_s = 0.0
        self.max_batch_wall_s = 0.0
        self.max_batch_size = 0
        self.braking_v3_state = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.v3_reset_plan_attempts = torch.zeros((), dtype=torch.long, device=device)
        self.v3_goal_completion_plan_attempts = torch.zeros((), dtype=torch.long, device=device)
        self.v3_runtime_replan_attempts = torch.zeros((), dtype=torch.long, device=device)
        self.v3_runtime_replan_unsafe_start_count = torch.zeros((), dtype=torch.long, device=device)
        self.v3_soft_envelope_exit_count = torch.zeros((), dtype=torch.long, device=device)
        self.v3_accepted_terminal_stop_certificate_count = torch.zeros((), dtype=torch.long, device=device)
        self.v3_accepted_command_count = torch.zeros((), dtype=torch.long, device=device)
        self.v3_prebrake_intervals = torch.zeros((), dtype=torch.long, device=device)
        self.v3_certificate_failure_count = torch.zeros((), dtype=torch.long, device=device)
        self.v3_stop_turn_go_intervals = torch.zeros((), dtype=torch.long, device=device)
        self.v3_corner_prebrake_intervals = torch.zeros((), dtype=torch.long, device=device)

    def reset_idx(self, env_ids) -> None:
        if len(env_ids) == 0:
            return
        self.length[env_ids] = 0
        self.cursor[env_ids] = 0
        self.handoff_clearance[env_ids] = 0.0
        self.valid[env_ids] = False
        self.completion_reported[env_ids] = False
        self.goal_change_reported[env_ids] = False
        self.support_change_reported[env_ids] = False
        self.status_code[env_ids] = self.STATUS_CODES["unplanned"]
        self.next_replan_step[env_ids] = 0
        self.recovery_state[env_ids] = RECOVERY_NORMAL
        self.recovery_anchor[env_ids] = 0.0
        self.recovery_age_steps[env_ids] = 0
        self.recovery_brake_age_steps[env_ids] = 0
        self.recovery_connect_age_steps[env_ids] = 0
        self.recovery_connect_timeout_steps[env_ids] = 0
        self.braking_v3_state[env_ids] = BRAKING_V3_ROUTE

    def invalidate(self, mask, reason: str, current_step: int) -> None:
        if reason not in self.STATUS_CODES:
            raise ValueError(f"unknown route invalidation reason {reason!r}")
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("route invalidation mask must have shape [N] and bool dtype")
        self.valid[mask] = False
        self.status_code[mask] = self.STATUS_CODES[reason]
        self.next_replan_step[mask] = int(current_step) + self.config.replan_cooldown_steps
        self.runtime_invalid_count += mask.sum()
        if reason == "local_step_infeasible":
            self.local_step_invalidations += mask.sum()

    def enter_recovery(self, mask, current_step: int) -> None:
        """Latch a soft-envelope violation without changing the physical actor pose."""
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("recovery mask must have shape [N] and bool dtype")
        fresh = mask & ((self.recovery_state == RECOVERY_NORMAL) | (self.recovery_state == RECOVERY_ROUTE))
        if bool(fresh.any()):
            self.recovery_entries += fresh.sum()
        self.recovery_state[fresh] = RECOVERY_BRAKE
        self.recovery_age_steps[fresh] = 0
        self.recovery_brake_age_steps[fresh] = 0
        self.recovery_connect_age_steps[fresh] = 0
        self.recovery_connect_timeout_steps[fresh] = 0
        self.valid[fresh] = False
        self.status_code[fresh] = self.STATUS_CODES["soft_envelope_violation"]
        self.next_replan_step[fresh] = int(current_step)
        self.runtime_invalid_count += fresh.sum()

    def recovery_anchor_idx(
        self,
        env_ids,
        position_xy,
        bars_xy,
        bars_half_extents_xy,
        arena_lo_xy,
        arena_hi_xy,
        support_xy,
        hard_boundary_margin_m: float,
        soft_boundary_margin_m: float,
    ) -> torch.Tensor:
        """Populate deterministic anchors; rows without a certified connector stay false."""
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) == 0:
            return result
        selected = [
            value[env_ids].detach().to("cpu", dtype=torch.float64).numpy()
            for value in (
                position_xy, bars_xy, bars_half_extents_xy, arena_lo_xy,
                arena_hi_xy, support_xy,
            )
        ]
        positions, bars, half, lows, highs, supports = selected
        # NavRL stores arena bounds as XYZ even though route recovery is an XY planner.
        # Project here as a defensive API boundary; adding a 2-D support vector to XYZ bounds
        # otherwise raises before the first recovery interval.
        lows = lows[..., :2]
        highs = highs[..., :2]
        ids = env_ids.detach().to("cpu", dtype=torch.long).tolist()
        for local, env_id in enumerate(ids):
            anchor = nearest_soft_free_anchor(
                positions[local], bars[local], half[local], lows[local], highs[local],
                supports[local], hard_boundary_margin_m, soft_boundary_margin_m,
                resolution_m=self.config.resolution_m,
                radius_cells=3,
                tracking_margin_m=self.config.tracking_margin_m,
                soft_hysteresis_m=self.config.resolution_m,
                hard_epsilon_m=TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
            )
            if anchor.get("exists") is True and anchor.get("hard_connector_safe") is True:
                self.recovery_anchor[env_id] = torch.as_tensor(
                    anchor["xy_m"], dtype=self.recovery_anchor.dtype, device=self.device
                )
                result[env_id] = True
        return result

    def brake_connector_idx(
        self,
        env_ids,
        position_xy,
        velocity_xy,
        bars_xy,
        bars_half_extents_xy,
        arena_lo_xy,
        arena_hi_xy,
        support_xy,
        hard_boundary_margin_m: float,
        decel_mps2: float,
        brake_speed_samples_mps=None,
        brake_stop_distance_samples_m=None,
        certified_lateral_tube_m=0.0,
    ) -> torch.Tensor:
        """Certify the straight swept stopping segment in the hard envelope.

        Recovery callers must provide the validated monotone speed/stop-distance lookup. The
        deceleration formula remains only as a legacy planner API fallback and is not a recovery
        certificate.
        """
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) == 0:
            return result
        if not math.isfinite(float(certified_lateral_tube_m)) or float(certified_lateral_tube_m) < 0.0:
            return result
        use_lookup = brake_speed_samples_mps is not None or brake_stop_distance_samples_m is not None
        if use_lookup:
            try:
                sample_speeds = np.asarray(brake_speed_samples_mps, dtype=np.float64).reshape(-1)
                sample_distances = np.asarray(brake_stop_distance_samples_m, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return result
            if (
                sample_speeds.size == 0 or sample_speeds.shape != sample_distances.shape
                or not np.isfinite(sample_speeds).all() or not np.isfinite(sample_distances).all()
                or np.any(sample_speeds <= 0.0) or np.any(sample_distances < 0.0)
                or np.any(np.diff(sample_speeds) <= 0.0) or np.any(np.diff(sample_distances) < 0.0)
            ):
                return result
        elif not math.isfinite(float(decel_mps2)) or float(decel_mps2) <= 0.0:
            return result
        selected = [
            value[env_ids].detach().to("cpu", dtype=torch.float64).numpy()
            for value in (
                position_xy, velocity_xy, bars_xy, bars_half_extents_xy,
                arena_lo_xy, arena_hi_xy, support_xy,
            )
        ]
        positions, velocities, bars, half, lows, highs, supports = selected
        lows = lows[..., :2]
        highs = highs[..., :2]
        ids = env_ids.detach().to("cpu", dtype=torch.long).tolist()
        for local, env_id in enumerate(ids):
            speed = float(np.linalg.norm(velocities[local]))
            if use_lookup:
                sample_index = int(np.searchsorted(sample_speeds, speed, side="left"))
                if sample_index >= sample_speeds.size:
                    continue
                # Ceiling lookup is conservative for a monotone measured stop-distance curve;
                # interpolation is deliberately avoided because it is not a certificate.
                distance = float(sample_distances[sample_index])
            else:
                distance = speed * speed / (2.0 * float(decel_mps2))
            direction = velocities[local] / speed if speed > 1e-9 else np.zeros(2)
            stop = positions[local] + direction * distance
            hard_margin = (
                TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M
                + float(certified_lateral_tube_m)
            )
            hard_lo = lows[local] + float(hard_boundary_margin_m) + supports[local] + hard_margin
            hard_hi = highs[local] - float(hard_boundary_margin_m) - supports[local] - hard_margin
            hard_half = half[local] + supports[local][None, :] + hard_margin
            result[env_id] = segment_is_safe(
                positions[local], stop, hard_lo, hard_hi, bars[local], hard_half
            )
        return result

    def mark_no_connector(
        self, mask, hard_breach: bool = False, timeout_kind: Optional[str] = None
    ) -> None:
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("recovery mask must have shape [N] and bool dtype")
        if timeout_kind not in (None, "brake", "connect"):
            raise ValueError("timeout_kind must be None, brake, or connect")
        self.recovery_state[mask] = RECOVERY_NO_CONNECTOR
        self.recovery_brake_age_steps[mask] = 0
        self.recovery_connect_age_steps[mask] = 0
        self.recovery_connect_timeout_steps[mask] = 0
        self.valid[mask] = False
        reason = (
            "recovery_hard_breach" if hard_breach
            else "recovery_brake_timeout" if timeout_kind == "brake"
            else "recovery_connect_timeout" if timeout_kind == "connect"
            else "recovery_no_connector"
        )
        self.status_code[mask] = self.STATUS_CODES[reason]
        if bool(mask.any()):
            self.recovery_no_connector_count += mask.sum()
            if hard_breach:
                self.recovery_hard_breach_count += mask.sum()
            if timeout_kind == "brake":
                self.recovery_brake_timeout_count += mask.sum()
            if timeout_kind == "connect":
                self.recovery_connect_timeout_count += mask.sum()

    def mark_local_infeasible_soft_free(self, mask) -> None:
        """Keep a local dynamics failure fail-closed even when geometry is soft-free."""
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("recovery mask must have shape [N] and bool dtype")
        self.recovery_state[mask] = RECOVERY_NO_CONNECTOR
        self.valid[mask] = False
        self.status_code[mask] = self.STATUS_CODES["recovery_local_infeasible_soft_free"]
        if bool(mask.any()):
            self.recovery_no_connector_count += mask.sum()

    def mark_route_resume(self, mask) -> None:
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("recovery mask must have shape [N] and bool dtype")
        self.recovery_state[mask] = RECOVERY_ROUTE
        self.recovery_route_resumes += mask.sum()

    def needs_replan(self, goal_xy, support_xy, current_step: int):
        if goal_xy.shape != self.goal.shape or support_xy.shape != self.planned_support.shape:
            raise ValueError("goal/support must have shape [N, 2]")
        goal_changed = (goal_xy - self.goal).norm(dim=1) > self.config.goal_tolerance_m
        support_changed = (support_xy > self.planned_support + 1e-6).any(dim=1)
        goal_event = goal_changed & ~self.goal_change_reported
        support_event = support_changed & ~self.support_change_reported
        self.goal_change_reported |= goal_changed
        self.support_change_reported |= support_changed
        self.goal_changed_invalidations += goal_event.sum()
        self.support_contract_invalidations += support_event.sum()
        self.runtime_invalid_count += goal_event.sum() + support_event.sum()
        self.status_code[goal_changed] = self.STATUS_CODES["goal_changed"]
        self.status_code[support_changed] = self.STATUS_CODES["support_contract_changed"]
        self.valid[goal_changed | support_changed] = False
        cooldown_done = self.next_replan_step <= int(current_step)
        return (~self.valid) & cooldown_done

    def plan_idx(
        self,
        env_ids,
        start_xy,
        goal_xy,
        bars_xy,
        bars_half_extents_xy,
        arena_lo_xy,
        arena_hi_xy,
        support_xy,
        current_step: int,
        is_replan: bool = False,
        connected_goal_selector=None,
        min_goal_distance_m: float = 0.0,
        excluded_goal_xy=None,
        goal_exclusion_radius_m: float = 0.0,
        plan_cause: Optional[str] = None,
    ) -> Dict[str, int]:
        if len(env_ids) == 0:
            return {}
        batch_started = time.perf_counter()
        batch_size = len(env_ids)
        ids = env_ids.detach().to("cpu", dtype=torch.long).tolist()
        selected = [
            value.detach().to("cpu", dtype=torch.float64).numpy()
            for value in (
                start_xy[env_ids],
                goal_xy[env_ids],
                bars_xy[env_ids],
                bars_half_extents_xy[env_ids],
                arena_lo_xy[env_ids],
                arena_hi_xy[env_ids],
                support_xy[env_ids],
            )
        ]
        starts, goals, bars, half, arena_lo, arena_hi, support = selected
        selectors = None
        if connected_goal_selector is not None:
            # One batch transfer avoids one synchronizing .item() call per environment.
            selectors = (
                connected_goal_selector[env_ids]
                .detach().to("cpu", dtype=torch.float64).numpy()
            )
        excluded_goals = None
        if excluded_goal_xy is not None:
            if excluded_goal_xy.shape != self.goal.shape:
                raise ValueError("excluded_goal_xy must have shape [N,2]")
            if connected_goal_selector is None or float(goal_exclusion_radius_m) <= 0.0:
                raise ValueError(
                    "goal exclusion requires connected-goal planning and a positive radius"
                )
            excluded_goals = (
                excluded_goal_xy[env_ids]
                .detach().to("cpu", dtype=torch.float64).numpy()
            )
        batch_counts: Dict[str, int] = {}
        if plan_cause not in (None, "reset", "goal_completion", "runtime_replan"):
            raise ValueError("plan_cause must be reset, goal_completion, runtime_replan, or None")
        for local, env_id in enumerate(ids):
            if self.braking_v3_enabled:
                # Fail closed on NaN/malformed rows without invoking A*.  v1 still uses
                # segment_is_safe, which raises on non-finite endpoints.
                try:
                    admissible_lo, admissible_hi, inflated_half = _GEO.numpy_soft_envelope(
                        arena_lo[local], arena_hi[local], half[local], support[local],
                        _GEO.SoftEnvelopeSpec(
                            wall_margin_m=0.0,
                            boundary_reserve_m=self.config.boundary_margin_m,
                            tracking_margin_m=self.config.tracking_margin_m,
                        ),
                    )
                    start_safe = _GEO.numpy_segments_soft_safe(
                        starts[local], starts[local], admissible_lo, admissible_hi,
                        bars[local], inflated_half,
                    )
                except ValueError:
                    start_safe = False
                if not start_safe:
                    self.plan_attempts += 1
                    self.replan_attempts += int(is_replan)
                    self.invalid_count += 1
                    self.valid[env_id] = False
                    self.length[env_id] = 0
                    self.handoff_clearance[env_id] = 0.0
                    self.status_code[env_id] = self.STATUS_CODES["unsafe_start"]
                    self.next_replan_step[env_id] = (
                        int(current_step) + self.config.replan_cooldown_steps
                    )
                    self.goal[env_id] = goal_xy[env_id]
                    self.planned_support[env_id] = support_xy[env_id]
                    batch_counts["unsafe_start"] = batch_counts.get("unsafe_start", 0) + 1
                    if plan_cause == "reset":
                        self.v3_reset_plan_attempts += 1
                    elif plan_cause == "goal_completion":
                        self.v3_goal_completion_plan_attempts += 1
                    elif plan_cause == "runtime_replan":
                        self.v3_runtime_replan_attempts += 1
                        self.v3_runtime_replan_unsafe_start_count += 1
                    continue
            if connected_goal_selector is None:
                result = self.planner.plan(
                    starts[local], goals[local], bars[local], half[local],
                    arena_lo[local], arena_hi[local], support[local]
                )
            else:
                selector = float(selectors[local])
                result = self.planner.plan_to_connected_goal(
                    starts[local], bars[local], half[local], arena_lo[local],
                    arena_hi[local], support[local], min_goal_distance_m, selector,
                    excluded_goal_xy=(
                        excluded_goals[local] if excluded_goals is not None else None
                    ),
                    exclusion_radius_m=(
                        float(goal_exclusion_radius_m) if excluded_goals is not None else 0.0
                    ),
                )
            if result.valid and excluded_goals is not None and (
                np.linalg.norm(result.waypoints_xy[-1] - excluded_goals[local])
                <= float(goal_exclusion_radius_m)
            ):
                # Defense in depth: a planner regression cannot reintroduce the failed goal.
                self.same_goal_reselection_count += 1
                result = RoutePlan(
                    "same_goal_reselected", np.empty((0, 2)),
                    result.expanded_nodes, result.raw_grid_nodes,
                    result.smoothed_nodes, 0.0,
                )
            self.plan_attempts += 1
            self.replan_attempts += int(is_replan)
            if self.braking_v3_enabled:
                if plan_cause == "reset":
                    self.v3_reset_plan_attempts += 1
                elif plan_cause == "goal_completion":
                    self.v3_goal_completion_plan_attempts += 1
                elif plan_cause == "runtime_replan":
                    self.v3_runtime_replan_attempts += 1
            self.connected_goal_replans += int(is_replan and selectors is not None)
            self.expanded_nodes += result.expanded_nodes
            self.raw_waypoints += result.raw_grid_nodes
            self.smoothed_waypoints += result.smoothed_nodes
            batch_counts[result.status] = batch_counts.get(result.status, 0) + 1
            selected_goal = (
                torch.as_tensor(
                    result.waypoints_xy[-1], dtype=self.goal.dtype, device=self.device
                )
                if result.valid
                else goal_xy[env_id]
            )
            self.goal[env_id] = selected_goal
            self.planned_support[env_id] = support_xy[env_id]
            self.goal_change_reported[env_id] = False
            self.support_change_reported[env_id] = False
            self.next_replan_step[env_id] = int(current_step) + self.config.replan_cooldown_steps
            self.status_code[env_id] = self.STATUS_CODES[result.status]
            if not result.valid:
                self.valid[env_id] = False
                self.length[env_id] = 0
                self.handoff_clearance[env_id] = 0.0
                self.no_path_count += int(result.status == "no_path")
                self.invalid_count += int(result.status != "no_path")
                continue
            count = result.waypoints_xy.shape[0] - 1  # exclude current start
            route = torch.as_tensor(
                result.waypoints_xy[1:], dtype=self.waypoints.dtype, device=self.device
            )
            self.waypoints[env_id].zero_()
            self.waypoints[env_id, :count] = route
            admissible_lo = arena_lo[local] + self.config.boundary_margin_m + support[local]
            admissible_hi = arena_hi[local] - self.config.boundary_margin_m - support[local]
            inflated_half = (
                half[local] + support[local][None, :] + self.config.tracking_margin_m
            )
            certificates = route_handoff_clearance_certificates(
                result.waypoints_xy,
                admissible_lo,
                admissible_hi,
                bars[local],
                inflated_half,
            )
            cached_certificates = torch.as_tensor(
                certificates[1:], dtype=self.handoff_clearance.dtype, device=self.device
            )
            self.handoff_clearance[env_id].zero_()
            self.handoff_clearance[env_id, :count] = cached_certificates
            self.segment_start[env_id] = start_xy[env_id]
            self.length[env_id] = count
            self.cursor[env_id] = 0
            self.completion_reported[env_id] = False
            self.valid[env_id] = count > 0
            self.plan_successes += 1
        elapsed = time.perf_counter() - batch_started
        self.planning_batches += 1
        self.planning_envs += batch_size
        self.total_planning_wall_s += elapsed
        self.max_batch_wall_s = max(self.max_batch_wall_s, elapsed)
        self.max_batch_size = max(self.max_batch_size, batch_size)
        return batch_counts

    def braking_v3_follow_reference(
        self,
        position_xy,
        velocity_xy,
        cruise_speed,
        reach_m: float,
        admissible_lo,
        admissible_hi,
        bars_xy,
        inflated_half,
        max_turn_rate,
        stop_speed_mps: float,
        brake_speed_knots,
        brake_distance_knots,
        dt: float,
        count_intervals=True,
    ):
        """GPU tangent/cross-track follower with certified fillets and stop-turn-go."""
        if position_xy.shape != self.goal.shape or cruise_speed.shape != (self.num_envs,):
            raise ValueError("position must be [N,2] and speed [N]")
        rows = torch.arange(self.num_envs, device=self.device)
        speed_now = velocity_xy.norm(dim=1)
        stopped = speed_now <= float(stop_speed_mps)
        stop_turn_go = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        corner_prebrake = torch.zeros_like(stop_turn_go)
        used_fillet = torch.zeros_like(stop_turn_go)
        for _ in range(4):
            safe_cursor = torch.minimum(self.cursor, (self.length - 1).clamp(min=0))
            waypoint = self.waypoints[rows, safe_cursor]
            next_cursor = torch.minimum(safe_cursor + 1, (self.length - 1).clamp(min=0))
            next_waypoint = self.waypoints[rows, next_cursor]
            has_next = self.valid & (self.cursor + 1 < self.length)
            segment = waypoint - self.segment_start
            offset = position_xy - waypoint
            cross_track = torch.abs(offset[:, 0] * segment[:, 1] - offset[:, 1] * segment[:, 0])
            cross_track = cross_track / segment.norm(dim=1).clamp(min=1e-6)
            passed = ((offset * segment).sum(dim=1) >= 0.0) & (cross_track <= float(reach_m))
            connector_certified = (
                (waypoint - position_xy).norm(dim=1)
                <= self.handoff_clearance[rows, safe_cursor]
            )
            reached = self.valid & (
                ((waypoint - position_xy).norm(dim=1) <= float(reach_m)) | passed
            ) & connector_certified
            fillet_safe = has_next & _GEO.torch_segments_soft_safe(
                position_xy.unsqueeze(1),
                next_waypoint.unsqueeze(1),
                admissible_lo,
                admissible_hi,
                bars_xy,
                inflated_half,
            )[:, 0]
            next_tangent = next_waypoint - waypoint
            current_heading = torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])
            next_heading = torch.atan2(next_tangent[:, 1], next_tangent[:, 0])
            heading_error = torch.atan2(
                torch.sin(next_heading - current_heading),
                torch.cos(next_heading - current_heading),
            ).abs()
            turn_budget = max_turn_rate * float(dt)
            need_stop_turn = (
                has_next & reached & ~fillet_safe & (heading_error > turn_budget) & ~stopped
            )
            skip_corner = has_next & reached & fillet_safe
            can_advance = reached & (self.cursor + 1 < self.length) & (
                skip_corner | ~need_stop_turn | stopped
            )
            self.segment_start[can_advance] = waypoint[can_advance]
            self.cursor += can_advance.long()
            stop_turn_go = need_stop_turn & ~can_advance
            used_fillet = skip_corner & can_advance
        safe_cursor = torch.minimum(self.cursor, (self.length - 1).clamp(min=0))
        waypoint = self.waypoints[rows, safe_cursor]
        remaining = (waypoint - position_xy).norm(dim=1)
        stop_distance, covered = _GEO.ceiling_stop_distance(
            speed_now, brake_speed_knots, brake_distance_knots
        )
        next_cursor = torch.minimum(safe_cursor + 1, (self.length - 1).clamp(min=0))
        next_waypoint = self.waypoints[rows, next_cursor]
        has_next = self.valid & (self.cursor + 1 < self.length)
        next_tangent = next_waypoint - waypoint
        current_heading = torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])
        next_heading = torch.atan2(next_tangent[:, 1], next_tangent[:, 0])
        heading_error = torch.atan2(
            torch.sin(next_heading - current_heading),
            torch.cos(next_heading - current_heading),
        ).abs()
        upcoming_turn = has_next & (heading_error > (max_turn_rate * float(dt)))
        corner_prebrake = upcoming_turn & covered & (remaining <= stop_distance)
        delta = waypoint - position_xy
        complete = self.valid & (self.cursor + 1 >= self.length) & (
            delta.norm(dim=1) <= float(reach_m)
        )
        new_completion = complete & ~self.completion_reported
        self.goal_completions += new_completion.sum()
        self.completion_reported |= complete
        active = self.valid & ~complete
        tangent = waypoint - self.segment_start
        tangent_norm = tangent.norm(dim=1, keepdim=True).clamp(min=1e-6)
        path_tangent = tangent / tangent_norm
        # Cross-track: point onto the current segment, then along the certified tangent.
        along = ((position_xy - self.segment_start) * path_tangent).sum(dim=1, keepdim=True)
        along = along.clamp(min=0.0)
        projected = self.segment_start + path_tangent * along
        to_path = projected - position_xy
        desired_dir = path_tangent + to_path
        desired_dir = desired_dir / desired_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        toward_waypoint = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
        use_tangent = (tangent.norm(dim=1) > 1e-6) & active
        heading = torch.where(use_tangent.unsqueeze(1), desired_dir, toward_waypoint)
        speed_limit = torch.where(
            active & ~stop_turn_go & ~corner_prebrake,
            cruise_speed.clamp(min=0.0),
            torch.zeros_like(cruise_speed),
        )
        velocity = heading * speed_limit.unsqueeze(1)
        if count_intervals:
            self.fallback_intervals += (~self.valid & (cruise_speed > 1e-6)).sum()
            self.v3_stop_turn_go_intervals += (self.valid & stop_turn_go).sum()
            self.v3_corner_prebrake_intervals += (self.valid & corner_prebrake).sum()
        self.braking_v3_state = torch.where(
            stop_turn_go,
            torch.full_like(self.braking_v3_state, BRAKING_V3_STOP_TURN_GO),
            torch.where(
                corner_prebrake,
                torch.full_like(self.braking_v3_state, BRAKING_V3_PREBRAKE),
                torch.full_like(self.braking_v3_state, BRAKING_V3_ROUTE),
            ),
        )
        return velocity, speed_limit, active, complete, stop_turn_go, corner_prebrake, used_fillet

    def velocity_reference(self, position_xy, speed, reach_m: float):
        if position_xy.shape != self.goal.shape or speed.shape != (self.num_envs,):
            raise ValueError("position must be [N,2] and speed [N]")
        rows = torch.arange(self.num_envs, device=self.device)
        # Smoothed routes contain few points, but advance repeatedly so a slow/reset target cannot
        # spend extra control intervals commanding a waypoint already inside the reach radius.
        for _ in range(4):
            safe_cursor = torch.minimum(self.cursor, (self.length - 1).clamp(min=0))
            waypoint = self.waypoints[rows, safe_cursor]
            segment = waypoint - self.segment_start
            offset = position_xy - waypoint
            cross_track = torch.abs(offset[:, 0] * segment[:, 1] - offset[:, 1] * segment[:, 0])
            cross_track /= segment.norm(dim=1).clamp(min=1e-6)
            passed = ((offset * segment).sum(dim=1) >= 0.0) & (
                cross_track <= float(reach_m)
            )
            connector_certified = (
                (waypoint - position_xy).norm(dim=1)
                <= self.handoff_clearance[rows, safe_cursor]
            )
            reached = self.valid & (
                ((waypoint - position_xy).norm(dim=1) <= float(reach_m)) | passed
            ) & connector_certified
            can_advance = reached & (self.cursor + 1 < self.length)
            self.segment_start[can_advance] = waypoint[can_advance]
            self.cursor += can_advance.long()
        safe_cursor = torch.minimum(self.cursor, (self.length - 1).clamp(min=0))
        waypoint = self.waypoints[rows, safe_cursor]
        delta = waypoint - position_xy
        final_segment = waypoint - self.segment_start
        final_offset = position_xy - waypoint
        final_cross_track = torch.abs(
            final_offset[:, 0] * final_segment[:, 1]
            - final_offset[:, 1] * final_segment[:, 0]
        ) / final_segment.norm(dim=1).clamp(min=1e-6)
        passed_final = ((final_offset * final_segment).sum(dim=1) >= 0.0) & (
            final_cross_track <= float(reach_m)
        )
        complete = self.valid & (self.cursor + 1 >= self.length) & (
            (delta.norm(dim=1) <= float(reach_m)) | passed_final
        )
        new_completion = complete & ~self.completion_reported
        self.goal_completions += new_completion.sum()
        self.completion_reported |= complete
        active = self.valid & ~complete
        direction = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-6)
        velocity = torch.where(
            active.unsqueeze(1), direction * speed.clamp(min=0.0).unsqueeze(1),
            torch.zeros_like(position_xy)
        )
        self.fallback_intervals += (~self.valid & (speed > 1e-6)).sum()
        return velocity, active, complete

    def diagnostics(self) -> Dict[str, object]:
        reverse = {value: key for key, value in self.STATUS_CODES.items()}
        codes, counts = torch.unique(self.status_code, return_counts=True)
        status_counts = {
            reverse.get(int(code), f"unknown_{int(code)}"): int(count)
            for code, count in zip(codes.detach().cpu(), counts.detach().cpu())
        }
        payload = {
            "mode": TARGET_ROUTE_MODE_RECOVERY if self.recovery_enabled else TARGET_ROUTE_MODE_GLOBAL_ASTAR,
            "plan_attempts": self.plan_attempts,
            "plan_successes": self.plan_successes,
            "replan_attempts": self.replan_attempts,
            "connected_goal_replans": self.connected_goal_replans,
            "same_goal_reselection_count": self.same_goal_reselection_count,
            "no_path_count": self.no_path_count,
            "invalid_count": self.invalid_count + int(self.runtime_invalid_count.item()),
            "local_step_invalidations": int(self.local_step_invalidations.item()),
            "fallback_intervals": int(self.fallback_intervals.item()),
            "goal_completions": int(self.goal_completions.item()),
            "invalidation_counts": {
                "local_step_infeasible": int(self.local_step_invalidations.item()),
                "support_contract_changed": int(self.support_contract_invalidations.item()),
                "goal_changed": int(self.goal_changed_invalidations.item()),
            },
            "planning_batches": self.planning_batches,
            "planning_envs": self.planning_envs,
            "total_planning_wall_s": self.total_planning_wall_s,
            "max_batch_wall_s": self.max_batch_wall_s,
            "max_batch_size": self.max_batch_size,
            "planning_wall_ms_per_env": (
                1000.0 * self.total_planning_wall_s / self.planning_envs
                if self.planning_envs else 0.0
            ),
            "expanded_nodes": self.expanded_nodes,
            "raw_waypoints": self.raw_waypoints,
            "smoothed_waypoints": self.smoothed_waypoints,
            "currently_valid": int(self.valid.sum().item()),
            "status_counts": status_counts,
        }
        if self.braking_v3_enabled:
            payload.update({
                "mode": TARGET_ROUTE_MODE_BRAKING_V3,
                "model": TARGET_ROUTE_BRAKING_V3_MODEL,
                "schema": TARGET_ROUTE_BRAKING_V3_SCHEMA,
                "v3_diagnostics": self.v3_gate_diagnostics(),
            })
            return payload
        if not self.recovery_enabled:
            return payload
        payload.update({
            "model": TARGET_ROUTE_RECOVERY_MODEL,
            "recovery_schema": TARGET_ROUTE_RECOVERY_SCHEMA,
            "recovery_hard_envelope": "closed_aabb_support_v1",
            "recovery_soft_envelope": "closed_aabb_support_plus_tracking_v1",
            "recovery_hard_epsilon_m": 1e-4,
            "recovery_reachable_tube_margin_m": TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
            "recovery_hysteresis_m": self.config.resolution_m,
            "recovery_anchor_radius_cells": 3,
            "recovery_age_steps_max": int(self.recovery_age_steps.max().item()),
            "recovery_brake_age_steps_max": int(self.recovery_brake_age_steps.max().item()),
            "recovery_connect_age_steps_max": int(self.recovery_connect_age_steps.max().item()),
            "recovery_connect_timeout_steps_max": int(self.recovery_connect_timeout_steps.max().item()),
            "recovery_state_counts": {
                "normal": int((self.recovery_state == RECOVERY_NORMAL).sum().item()),
                "brake": int((self.recovery_state == RECOVERY_BRAKE).sum().item()),
                "connect": int((self.recovery_state == RECOVERY_CONNECT).sum().item()),
                "route": int((self.recovery_state == RECOVERY_ROUTE).sum().item()),
                "no_connector": int((self.recovery_state == RECOVERY_NO_CONNECTOR).sum().item()),
            },
            "recovery_entries": int(self.recovery_entries.item()),
            "recovery_brake_intervals": int(self.recovery_brake_intervals.item()),
            "recovery_connect_intervals": int(self.recovery_connect_intervals.item()),
            "recovery_no_connector_count": int(self.recovery_no_connector_count.item()),
            "recovery_hard_breach_count": int(self.recovery_hard_breach_count.item()),
            "recovery_brake_timeout_count": int(self.recovery_brake_timeout_count.item()),
            "recovery_connect_timeout_count": int(self.recovery_connect_timeout_count.item()),
            "recovery_route_resumes": int(self.recovery_route_resumes.item()),
        })
        return payload

    def v3_gate_diagnostics(self) -> Dict[str, object]:
        """Cause-separated telemetry consumed by the braking-v3 gate verifier."""
        return {
            "runtime_replan_unsafe_start_count": int(self.v3_runtime_replan_unsafe_start_count.item()),
            "soft_envelope_exit_count": int(self.v3_soft_envelope_exit_count.item()),
            "accepted_terminal_stop_certificate_count": int(
                self.v3_accepted_terminal_stop_certificate_count.item()
            ),
            "accepted_command_count": int(self.v3_accepted_command_count.item()),
            "plan_attempt_count": int(self.plan_attempts),
            "plan_success_count": int(self.plan_successes),
            "fallback_interval_count": int(self.fallback_intervals.item()),
            "goal_completion_count": int(self.goal_completions.item()),
            "reset_plan_attempt_count": int(self.v3_reset_plan_attempts.item()),
            "goal_completion_plan_attempt_count": int(self.v3_goal_completion_plan_attempts.item()),
            "runtime_replan_attempt_count": int(self.v3_runtime_replan_attempts.item()),
            "prebrake_interval_count": int(self.v3_prebrake_intervals.item()),
            "certificate_failure_count": int(self.v3_certificate_failure_count.item()),
            "stop_turn_go_interval_count": int(self.v3_stop_turn_go_intervals.item()),
            "corner_prebrake_interval_count": int(self.v3_corner_prebrake_intervals.item()),
        }
