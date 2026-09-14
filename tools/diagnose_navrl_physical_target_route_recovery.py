#!/usr/bin/env python3
"""Evaluation-only forensics for routed physical-target recovery.

The routed target transition is deliberately *not* instrumented in the simulator source.  This
tool attaches an observer to a child process at evaluation time, records the arguments already
passed to the route manager, and computes geometry on CPU.  It therefore cannot change target
commands, planner choices, observations, rewards, termination, PPO, or the attempt2 evaluator.

The default frozen probe is the route-on low/high-speed slice of the attempt2 density grid:
70/150/205/300 bars x 0.6/1.5 m/s, seed 827, 32 environments, 300 steps.  It is intentionally
small enough to diagnose the reported 0.125--0.635% invalidation and 32.6--85.0% fallback
range while retaining every density knot and both speed endpoints.  The planner's 0.25 m grid
and its +/-3-cell anchor neighbourhood are reused; no .45 m threshold is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types
from typing import Any, Callable, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 827
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20
DENSITIES = (70, 150, 205, 300)
SPEEDS = (0.6, 1.5)
GRID_RESOLUTION_M = 0.25
TRACKING_MARGIN_M = 0.45
ANCHOR_RADIUS_CELLS = 3
ATTEMPT2_SUMMARY_SHA256 = "e5e4560464dc3a2080d904c2f8d2247e0c65e671dd63ea08d1b507ec65fc7197"
ATTEMPT2_SUMMARY = ROOT / "results/navrl_physical_target_routed_gate_seed827_attempt2/summary.json"
OUTPUT_ROOT = ROOT / "results/navrl_physical_target_route_recovery_forensics_seed827"
SCHEMA = "navrl_physical_target_route_recovery_forensics_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorization_token(partial_directory: Path) -> str:
    return sha256(Path(__file__).resolve()) + ":" + partial_directory.name


def runtime_source_manifest() -> tuple[list[dict[str, Any]], str]:
    paths = [path for base in (ROOT / "aerial_gym", ROOT / "resources")
             for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    paths.append(Path(__file__).resolve())
    entries = [{"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path),
                "size": path.stat().st_size} for path in sorted(set(paths))]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return entries, hashlib.sha256(encoded).hexdigest()


def _finite_xy(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(2)
    if not np.isfinite(result).all():
        raise ValueError("non-finite XY")
    return result


def _point_aabb_distance(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    offset = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(offset))


def _segment_intersects(p0, p1, lo, hi, epsilon=1e-12) -> bool:
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
        enter, leave = max(enter, ta), min(leave, tb)
        if enter > leave:
            return False
    return enter <= leave and leave >= 0.0 and enter <= 1.0


def _point_segment_distance(point, p0, p1) -> float:
    delta = p1 - p0
    denominator = float(np.dot(delta, delta))
    fraction = float(np.clip(np.dot(point - p0, delta) / denominator, 0.0, 1.0)) if denominator > 1e-24 else 0.0
    return float(np.linalg.norm(point - (p0 + fraction * delta)))


def _segment_aabb_distance(p0, p1, lo, hi) -> float:
    if _segment_intersects(p0, p1, lo, hi):
        return 0.0
    corners = (np.array([lo[0], lo[1]]), np.array([lo[0], hi[1]]),
               np.array([hi[0], lo[1]]), np.array([hi[0], hi[1]]))
    return min(_point_aabb_distance(p0, lo, hi), _point_aabb_distance(p1, lo, hi),
               *(_point_segment_distance(corner, p0, p1) for corner in corners))


def _point_polyline_distance(point, points) -> Optional[float]:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    return min(_point_segment_distance(point, a, b) for a, b in zip(points[:-1], points[1:]))


def _segment_is_safe(p0, p1, lo, hi, bars, half) -> bool:
    # Match target_route_planner.segment_is_safe: boundaries are closed and touching is unsafe.
    if np.any(p0 <= lo) or np.any(p0 >= hi) or np.any(p1 <= lo) or np.any(p1 >= hi):
        return False
    return not any(_segment_intersects(p0, p1, center - extent, center + extent)
                   for center, extent in zip(bars, half))


def geometry_metrics(point, bars, bar_half, bounds_lo, bounds_hi, support, boundary_margin,
                     route=None, active_segment=None,
                     soft_boundary_margin=None, route_clearance=None) -> dict[str, Any]:
    """Return exact support-only and support+tracking clearances, fail-closed on bad input."""
    try:
        point = _finite_xy(point)
        bars = np.asarray(bars, dtype=np.float64).reshape((-1, 2))
        bar_half = np.asarray(bar_half, dtype=np.float64).reshape((-1, 2))
        bounds_lo, bounds_hi = _finite_xy(bounds_lo), _finite_xy(bounds_hi)
        support = _finite_xy(support)
        boundary_margin = float(boundary_margin)
        soft_boundary_margin = boundary_margin if soft_boundary_margin is None else float(soft_boundary_margin)
        if bars.shape != bar_half.shape or not np.isfinite(bars).all() or not np.isfinite(bar_half).all():
            raise ValueError("invalid bars")
        if (np.any(bar_half < 0) or np.any(support < 0)
                or not np.isfinite(boundary_margin) or not np.isfinite(soft_boundary_margin)):
            raise ValueError("invalid geometry")
        hard_half = bar_half + support
        soft_half = hard_half + TRACKING_MARGIN_M
        hard_bar_value = min((_point_aabb_distance(point, c - e, c + e)
                              for c, e in zip(bars, hard_half)), default=None)
        soft_bar_value = min((_point_aabb_distance(point, c - e, c + e)
                              for c, e in zip(bars, soft_half)), default=None)
        hard_bar = float("inf") if hard_bar_value is None else hard_bar_value
        soft_bar = float("inf") if soft_bar_value is None else soft_bar_value
        rounded_soft_value = min((float(np.linalg.norm(
            np.maximum(np.abs(point - c) - (e + support), 0.0)))
            for c, e in zip(bars, bar_half)), default=float("inf"))
        local_soft_free_rounded = rounded_soft_value >= TRACKING_MARGIN_M + 1e-4
        route_soft_free_aabb = soft_bar > 0.0
        hard_lo, hard_hi = bounds_lo + boundary_margin + support, bounds_hi - boundary_margin - support
        # The fixed arena reserve is separate from the obstacle tracking reserve: runtime hard
        # uses wall_margin + support, while the cached route uses route boundary_margin + support.
        # Do not add 0.45 m to a boundary a second time.
        soft_lo, soft_hi = bounds_lo + soft_boundary_margin + support, bounds_hi - soft_boundary_margin - support
        hard_boundary = float(np.minimum(point - hard_lo, hard_hi - point).min())
        soft_boundary = float(np.minimum(point - soft_lo, soft_hi - point).min())
        hard = min(hard_bar, hard_boundary)
        soft = min(soft_bar, soft_boundary)
        obstacle_bad = soft_bar <= 0.0
        boundary_bad = soft_boundary <= 0.0
        reason = ("both" if obstacle_bad and boundary_bad else "obstacle" if obstacle_bad
                  else "boundary" if boundary_bad else "none")
        out = {
            "hard_clearance_m": float(hard), "soft_clearance_m": float(soft),
            "hard_bar_clearance_m": None if hard_bar_value is None else float(hard_bar),
            "soft_bar_clearance_m": None if soft_bar_value is None else float(soft_bar),
            "hard_boundary_clearance_m": float(hard_boundary),
            "soft_boundary_clearance_m": float(soft_boundary),
            "unsafe_start_reason": reason,
            "local_soft_bar_distance_rounded_m": None if not np.isfinite(rounded_soft_value) else rounded_soft_value,
            "local_soft_free_rounded": bool(local_soft_free_rounded),
            "route_soft_free_aabb": bool(route_soft_free_aabb),
            "rounded_vs_aabb_soft_disagreement": bool(local_soft_free_rounded != route_soft_free_aabb),
            "hard_free_exact": bool(hard > 0.0),
            "soft_free": bool(soft > 0.0),
            "runtime_hard_free_epsilon_1e-4": bool(hard > 1e-4),
            "runtime_vs_exact_free_disagreement": bool((hard > 0.0) != (hard > 1e-4)),
        }
        if active_segment is not None:
            a, b = (_finite_xy(active_segment[0]), _finite_xy(active_segment[1]))
            out["route_cross_track_error_m"] = _point_segment_distance(point, a, b)
        if route is not None:
            points = np.asarray(route, dtype=np.float64).reshape((-1, 2))
            if len(points) >= 2 and np.isfinite(points).all():
                out["route_polyline_error_m"] = _point_polyline_distance(point, points)
            else:
                out["route_polyline_error_m"] = None
        if route is not None:
            points = np.asarray(route, dtype=np.float64).reshape((-1, 2))
            if len(points) >= 2 and np.isfinite(points).all():
                if route_clearance is not None:
                    out["route_hard_min_segment_clearance_m"] = route_clearance.get(
                        "route_hard_min_segment_clearance_m")
                    out["route_soft_min_segment_clearance_m"] = route_clearance.get(
                        "route_soft_min_segment_clearance_m")
                else:
                    hard_segment = []
                    soft_segment = []
                    for a, b in zip(points[:-1], points[1:]):
                        hard_segment.append(min((_segment_aabb_distance(a, b, c - e, c + e)
                                                 for c, e in zip(bars, hard_half)), default=float("inf")))
                        soft_segment.append(min((_segment_aabb_distance(a, b, c - e, c + e)
                                                 for c, e in zip(bars, soft_half)), default=float("inf")))
                        hard_segment.append(float(np.minimum(np.minimum(a - hard_lo, hard_hi - a),
                                                             np.minimum(b - hard_lo, hard_hi - b)).min()))
                        soft_segment.append(float(np.minimum(np.minimum(a - soft_lo, soft_hi - a),
                                                             np.minimum(b - soft_lo, soft_hi - b)).min()))
                    out["route_hard_min_segment_clearance_m"] = float(min(hard_segment))
                    out["route_soft_min_segment_clearance_m"] = float(min(soft_segment))
            else:
                out["route_hard_min_segment_clearance_m"] = None
                out["route_soft_min_segment_clearance_m"] = None
        return out
    except (TypeError, ValueError, FloatingPointError):
        # A diagnostic must never turn malformed state into a reassuring zero.
        return {key: None for key in ("hard_clearance_m", "soft_clearance_m",
                                      "hard_bar_clearance_m", "soft_bar_clearance_m",
                                      "hard_boundary_clearance_m", "soft_boundary_clearance_m",
                                      "route_cross_track_error_m", "route_polyline_error_m",
                                      "route_hard_min_segment_clearance_m",
                                      "route_soft_min_segment_clearance_m",
                                      "unsafe_start_reason", "hard_free_exact", "soft_free",
                                      "runtime_hard_free_epsilon_1e-4",
                                      "runtime_vs_exact_free_disagreement",
                                      "local_soft_bar_distance_rounded_m",
                                      "local_soft_free_rounded", "route_soft_free_aabb",
                                      "rounded_vs_aabb_soft_disagreement")}


def nearest_soft_free_anchor(point, bars, bar_half, bounds_lo, bounds_hi, support,
                             boundary_margin, resolution=GRID_RESOLUTION_M,
                             radius_cells=ANCHOR_RADIUS_CELLS,
                             soft_boundary_margin=None) -> dict[str, Any]:
    """Search the planner-equivalent 7x7 neighbourhood for a soft-free, hard-safe connector."""
    try:
        point = _finite_xy(point)
        bars = np.asarray(bars, dtype=np.float64).reshape((-1, 2))
        bar_half = np.asarray(bar_half, dtype=np.float64).reshape((-1, 2))
        lo, hi = _finite_xy(bounds_lo), _finite_xy(bounds_hi)
        support = _finite_xy(support)
        soft_boundary_margin = boundary_margin if soft_boundary_margin is None else soft_boundary_margin
        extent = hi - lo
        shape = np.maximum(1, np.floor(extent / float(resolution)).astype(np.int64))
        axes = tuple(lo[k] + (np.arange(int(shape[k])) + 0.5) * float(resolution) for k in (0, 1))
        base = tuple(int(np.argmin(np.abs(axes[k] - point[k]))) for k in (0, 1))
        soft_lo, soft_hi = lo + float(soft_boundary_margin) + support, hi - float(soft_boundary_margin) - support
        hard_lo, hard_hi = lo + boundary_margin + support, hi - boundary_margin - support
        soft_half = bar_half + support[None, :] + TRACKING_MARGIN_M
        hard_half = bar_half + support[None, :]
        candidates = []
        for di in range(-int(radius_cells), int(radius_cells) + 1):
            for dj in range(-int(radius_cells), int(radius_cells) + 1):
                i, j = base[0] + di, base[1] + dj
                if not (0 <= i < shape[0] and 0 <= j < shape[1]):
                    continue
                anchor = np.array([axes[0][i], axes[1][j]], dtype=np.float64)
                soft_free = np.all(anchor > soft_lo) and np.all(anchor < soft_hi)
                soft_free &= all(not _segment_intersects(anchor, anchor, c - e, c + e)
                                 for c, e in zip(bars, soft_half))
                if soft_free and _segment_is_safe(point, anchor, hard_lo, hard_hi, bars, hard_half):
                    candidates.append((float(np.dot(anchor - point, anchor - point)), i, j, anchor))
        if not candidates:
            return {"exists": False, "distance_m": None, "xy_m": None,
                    "hard_connector_safe": False, "hard_connector_clearance_m": None}
        _, i, j, anchor = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        connector = min(_segment_aabb_distance(point, anchor, c - e, c + e)
                        for c, e in zip(bars, hard_half)) if len(bars) else float("inf")
        connector = min(connector, float(np.minimum(anchor - hard_lo, hard_hi - anchor).min()),
                        float(np.minimum(point - hard_lo, hard_hi - point).min()))
        return {"exists": True, "distance_m": float(np.linalg.norm(anchor - point)),
                "xy_m": anchor.tolist(), "cell_ij": [int(i), int(j)],
                "hard_connector_safe": True, "hard_connector_clearance_m": float(connector)}
    except (TypeError, ValueError, FloatingPointError):
        return {"exists": None, "distance_m": None, "xy_m": None,
                "hard_connector_safe": None, "hard_connector_clearance_m": None}


class RouteForensicsRecorder:
    """CPU observer attached only by the diagnostic child process."""

    def __init__(self, manager, geometry_provider: Callable[[], Mapping[str, Any]], step_provider: Callable[[], int]):
        self.manager = manager
        self.geometry_provider = geometry_provider
        self.step_provider = step_provider
        self.routes: dict[int, np.ndarray] = {}
        self.route_clearances: dict[int, dict[str, Optional[float]]] = {}
        self.origins: dict[int, dict[str, Any]] = {}
        self.last_speed: dict[int, float] = {}
        self.next_id = 0
        self.events: list[dict[str, Any]] = []
        self.fallback_by_origin: dict[str, dict[str, Any]] = {}
        self.observer_wall_s = 0.0
        self.observer_compute_wall_s = 0.0
        self.observer_transfer_wall_s = 0.0

    def _materialize_geometry(self):
        started = time.perf_counter()
        source = self.geometry_provider()
        g = {key: self._array(value) if key not in ("hard_boundary_margin", "soft_boundary_margin")
             else value for key, value in source.items()}
        self.observer_transfer_wall_s += time.perf_counter() - started
        return g

    @staticmethod
    def _array(value):
        if value is None:
            return None
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def _snapshot_from_geometry(self, env, g, point=None, route=None, active=None, speed=None,
                                include_anchor=True, route_clearance=None):
        started = time.perf_counter()
        pos = g["position"][env, :2] if point is None else point
        bars, half = g["bars"][env], g["bar_half"][env]
        lo, hi = g["bounds_lo"][env], g["bounds_hi"][env]
        support = g["support"][env]
        row = geometry_metrics(pos, bars, half, lo, hi, support,
                               float(g["hard_boundary_margin"]), route, active,
                               soft_boundary_margin=float(g["soft_boundary_margin"]),
                               route_clearance=route_clearance)
        if speed is None and g.get("speed") is not None:
            speed = g["speed"][env]
        row["target_speed_mps"] = None if speed is None else float(speed)
        realized = g.get("realized_velocity")
        if realized is not None:
            row["realized_target_speed_mps"] = float(np.linalg.norm(realized[env, :2]))
        if include_anchor:
            row["nearest_soft_free_anchor"] = nearest_soft_free_anchor(
                pos, bars, half, lo, hi, support, float(g["hard_boundary_margin"]),
                soft_boundary_margin=float(g["soft_boundary_margin"]))
        self.observer_compute_wall_s += time.perf_counter() - started
        return row

    def _snapshot(self, env: int, point=None, route=None, active=None, speed=None,
                  include_anchor=True):
        source = self.geometry_provider()
        g = {key: self._array(value) if key not in ("hard_boundary_margin", "soft_boundary_margin")
             else value for key, value in source.items()}
        return self._snapshot_from_geometry(env, g, point, route, active, speed, include_anchor)

    def _event(self, env, kind, reason=None, origin=None, **extra):
        row = {"event_id": self.next_id, "env": int(env), "step": int(self.step_provider()),
               "event": kind, "reason": reason, "origin_invalidation_id": origin}
        row.update(extra)
        self.events.append(row)
        self.next_id += 1
        return row["event_id"]

    def invalidation(self, mask, reason):
        started = time.perf_counter()
        g = self._materialize_geometry()
        for env in np.flatnonzero(self._array(mask)):
            env = int(env)
            origin_id = self._event(env, "invalidation", reason=reason,
                                    **self._snapshot_from_geometry(env, g, route=self.routes.get(env),
                                                                    speed=self.last_speed.get(env),
                                                                    include_anchor=False,
                                                                    route_clearance=self.route_clearances.get(env)))
            self.origins[env] = {"event_id": origin_id, "step": int(self.step_provider()),
                                 "reason": reason, "intervals": 0}
        self.observer_wall_s += time.perf_counter() - started

    def plan(self, env_ids, starts, bars, half, bounds_lo, bounds_hi, support, *, is_replan, before_routes):
        started = time.perf_counter()
        status_reverse = {int(v): k for k, v in self.manager.STATUS_CODES.items()}
        statuses = self._array(self.manager.status_code)
        g = self._materialize_geometry()
        before_clearances = dict(self.route_clearances)
        for local, env_value in enumerate(self._array(env_ids).astype(int).tolist()):
            env = int(env_value)
            status = status_reverse.get(int(statuses[env]), "unknown")
            old_route = before_routes.get(env)
            route = None
            if bool(self._array(self.manager.valid)[env]):
                count = int(self._array(self.manager.length)[env])
                route = np.concatenate((np.asarray(starts[env], dtype=float).reshape(1, 2),
                                        self._array(self.manager.waypoints)[env, :count]), axis=0)
                self.routes[env] = route
                # Compute route-vs-bar segment clearance once at plan time. Fallback telemetry
                # reuses this immutable snapshot and therefore cannot repeat bars x segments.
                route_row = geometry_metrics(
                    np.asarray(starts[env]), g["bars"][env], g["bar_half"][env],
                    g["bounds_lo"][env], g["bounds_hi"][env], g["support"][env],
                    float(g["hard_boundary_margin"]), route=route,
                    soft_boundary_margin=float(g["soft_boundary_margin"]))
                self.route_clearances[env] = {
                    key: route_row.get(key) for key in (
                        "route_hard_min_segment_clearance_m",
                        "route_soft_min_segment_clearance_m")}
            snapshot = self._snapshot_from_geometry(
                env, g, point=np.asarray(starts[env]), route=old_route,
                speed=self.last_speed.get(env),
                include_anchor=status in ("unsafe_start", "unsafe_start_cell"),
                route_clearance=before_clearances.get(env))
            row = {"plan_status": status, "replan": bool(is_replan),
                   "snapshot": snapshot, **snapshot}
            self._event(env, "replan" if is_replan else "initial_plan", reason=status,
                        origin=self.origins.get(env, {}).get("event_id"), **row)
            if status == "ok":
                self.origins.pop(env, None)
        self.observer_wall_s += time.perf_counter() - started

    def fallback(self, positions, speeds, active_segment):
        started = time.perf_counter()
        g = self._materialize_geometry()
        valid = self._array(self.manager.valid).astype(bool)
        for env, (position, speed) in enumerate(zip(self._array(positions), self._array(speeds))):
            self.last_speed[env] = float(speed)
            if valid[env] or float(speed) <= 1e-6:
                continue
            origin = self.origins.get(env)
            if origin is not None:
                origin["intervals"] += 1
                age = int(self.step_provider()) - int(origin["step"]) + 1
                key = str(origin["event_id"])
                aggregate = self.fallback_by_origin.setdefault(
                    key, {"invalidation_event_id": origin["event_id"], "reason": origin["reason"],
                          "env": env, "intervals": 0, "max_age_steps": 0})
                aggregate["intervals"] += 1
                aggregate["max_age_steps"] = max(aggregate["max_age_steps"], age)
            else:
                age, key = None, None
            # One device->host snapshot per velocity-reference call; fallback events do not run
            # the expensive 7x7 anchor search. Anchors are required only for unsafe-start replans.
            row = self._snapshot_from_geometry(env, g, point=position[:2], route=self.routes.get(env),
                                                active=active_segment[env], speed=speed,
                                                include_anchor=False,
                                                route_clearance=self.route_clearances.get(env))
            row["fallback_age_steps"] = age
            self._event(env, "fallback", origin=key, **row)
        self.observer_wall_s += time.perf_counter() - started

    def reset(self, env_ids):
        for env in self._array(env_ids).astype(int).tolist():
            self._event(int(env), "reset", reason="episode_reset")
            self.routes.pop(int(env), None)
            self.route_clearances.pop(int(env), None)
            self.origins.pop(int(env), None)
            self.last_speed.pop(int(env), None)


def attach_observer(task) -> RouteForensicsRecorder:
    manager = task._target_route_manager
    if manager is None:
        raise RuntimeError("route manager is disabled; diagnostic requires global_astar_v1")
    def geometry_provider():
        return {"position": task.target_position, "bars": task.obs_dict["obstacle_position"][:, task._bar_offset:task._bar_offset + task.n_bars_active, :2],
                "bar_half": task.obs_dict["asset_collision_half_extents"][:, task._bar_offset:task._bar_offset + task.n_bars_active, :2],
                "bounds_lo": task.obs_dict["env_bounds_min"][:, :2], "bounds_hi": task.obs_dict["env_bounds_max"][:, :2],
                "support": task._target_route_support_xy,
                "speed": task._tm_speed,
                "hard_boundary_margin": float(task.cur.wall_margin),
                "soft_boundary_margin": manager.config.boundary_margin_m,
                "realized_velocity": task.target_vel_w}
    recorder = RouteForensicsRecorder(manager, geometry_provider, lambda: int(task.num_task_steps))
    original_invalidate, original_needs, original_plan, original_velocity, original_reset = manager.invalidate, manager.needs_replan, manager.plan_idx, manager.velocity_reference, manager.reset_idx
    def invalidate(mask, reason, current_step):
        recorder.invalidation(mask, reason)
        return original_invalidate(mask, reason, current_step)
    def needs_replan(*args, **kwargs):
        before = recorder._array(manager.valid).copy()
        result = original_needs(*args, **kwargs)
        after_status = recorder._array(manager.status_code)
        reverse = {int(v): k for k, v in manager.STATUS_CODES.items()}
        newly_invalid = before & ~recorder._array(manager.valid)
        for reason in ("goal_changed", "support_contract_changed"):
            mask = newly_invalid & np.array(
                [reverse.get(int(code), "unknown") == reason for code in after_status], dtype=bool)
            if bool(mask.any()):
                recorder.invalidation(mask, reason)
        return result
    def plan_idx(env_ids, start_xy, goal_xy, bars_xy, bars_half_extents_xy, arena_lo_xy, arena_hi_xy, support_xy, current_step, is_replan=False, **kwargs):
        before = dict(recorder.routes)
        result = original_plan(env_ids, start_xy, goal_xy, bars_xy, bars_half_extents_xy, arena_lo_xy, arena_hi_xy, support_xy, current_step, is_replan=is_replan, **kwargs)
        recorder.plan(env_ids, recorder._array(start_xy), recorder._array(bars_xy), recorder._array(bars_half_extents_xy), recorder._array(arena_lo_xy), recorder._array(arena_hi_xy), recorder._array(support_xy), is_replan=is_replan, before_routes=before)
        return result
    def velocity_reference(position_xy, speed, reach_m):
        result = original_velocity(position_xy, speed, reach_m)
        rows = np.arange(manager.num_envs)
        cursor = np.minimum(recorder._array(manager.cursor), np.maximum(recorder._array(manager.length) - 1, 0))
        waypoints = recorder._array(manager.waypoints)[rows, cursor]
        starts = recorder._array(manager.segment_start)
        recorder.fallback(position_xy, speed, list(zip(starts, waypoints)))
        return result
    def reset_idx(env_ids):
        recorder.reset(env_ids)
        return original_reset(env_ids)
    manager.invalidate, manager.needs_replan, manager.plan_idx, manager.velocity_reference, manager.reset_idx = invalidate, needs_replan, plan_idx, velocity_reference, reset_idx
    return recorder


def frozen_contract() -> dict[str, Any]:
    if not ATTEMPT2_SUMMARY.is_file() or sha256(ATTEMPT2_SUMMARY) != ATTEMPT2_SUMMARY_SHA256:
        raise RuntimeError("attempt2 summary provenance mismatch; refusing diagnostic")
    return {"seed": SEED, "envs": ENVS, "steps": STEPS, "warmup_steps": WARMUP_STEPS,
            "route_mode": "global_astar_v1", "target_dynamics": "physical", "pattern": "waypoint",
            "densities": list(DENSITIES), "speeds_mps": list(SPEEDS), "grid_resolution_m": GRID_RESOLUTION_M,
            "tracking_margin_m": TRACKING_MARGIN_M, "anchor_radius_cells": ANCHOR_RADIUS_CELLS,
            "runtime_wall_margin_m": 0.50, "route_boundary_margin_m": 1.25,
            "boundary_soft_minus_hard_m": 0.75,
            "attempt2_summary_sha256": ATTEMPT2_SUMMARY_SHA256}


def _configure(density: int, speed: float) -> None:
    values = {
        "AERIAL_GYM_SIM_NAME": "base_sim", "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical", "NAVRL_TARGET_ROUTE_MODE": "global_astar_v1",
        "NAVRL_TARGET_PATTERN": "waypoint", "NAVRL_TARGET_SPEED": str(speed),
        "NAVRL_TARGET_SPEED_FINAL": str(speed), "NAVRL_TARGET_SPEED_MIN": str(speed),
        "NAVRL_TARGET_SPEED_RAMP_EPOCHS": "1", "NAVRL_TARGET_MAX_ACCEL": "4.0",
        "NAVRL_TARGET_MAX_TURN_RATE_DEG": "150.0", "NAVRL_TARGET_LOOKAHEAD_S": "1.0",
        "NAVRL_TARGET_OBSTACLE_CLEARANCE": "0.77", "NAVRL_TARGET_MASS_KG": "1.20",
        "NAVRL_TARGET_MOTOR_ARM_XY_M": "0.0777817", "NAVRL_TARGET_MAX_MOTOR_THRUST_N": "9.60",
        "NAVRL_TARGET_MOTOR_TAU_S": "0.04", "NAVRL_TARGET_YAW_TORQUE_RATIO_M": "0.01",
        "NAVRL_TARGET_MAX_TILT_DEG": "45.0", "NAVRL_TARGET_VEL_KP": "2.5",
        "NAVRL_TARGET_ALT_KP": "4.0", "NAVRL_TARGET_TRACKING_MARGIN_M": "0.45",
        "NAVRL_TARGET_BOUNDARY_MARGIN_M": "0.75", "NAVRL_TARGET_ROUTE_RESOLUTION_M": "0.25",
        "NAVRL_TARGET_ROUTE_MAX_EXPANSIONS": "50000", "NAVRL_TARGET_ROUTE_MAX_WAYPOINTS": "128",
        "NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS": "10", "NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M": "0.05",
        "NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M": "6.0", "NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M": "1.0",
        # Match attempt2: constructor RNG sees the 70-bar base; run_cell changes active bars only
        # after task.seed(827), preserving reset/layout provenance for every density.
        "NAVRL_NUM_BARS": "70", "NAVRL_MAX_BARS": "300", "NAVRL_DENSITY_CURRICULUM": "0",
        "NAVRL_VISION": "0", "NAVRL_PERCEPTION": "0", "NAVRL_GENERAL_TRAIN": "1",
        "NAVRL_ARENA_XY": "40", "NAVRL_ARENA_Z": "3", "NAVRL_BAR_POOL": "bars_h3",
        "NAVRL_BAR_X_MIN": "0", "NAVRL_BAR_X_MAX": "1", "NAVRL_PLACEMENT_MODE": "navrl_band",
        "NAVRL_PLACEMENT_TOUCH_M": "0.4", "NAVRL_PLACEMENT_GAP_M": "1.6",
    }
    os.environ.update(values)


def build_child_environment() -> dict[str, str]:
    """Mirror the attempt2 hermetic launcher: selected conda bin first, stale NAVRL removed."""
    python_bin = Path(sys.executable).resolve().parent
    ninja = python_bin / "ninja"
    if not ninja.is_file() or not os.access(str(ninja), os.X_OK):
        raise RuntimeError("selected Python environment has no executable ninja")
    child = {key: value for key, value in os.environ.items()
             if not key.startswith("NAVRL_") and key != "AERIAL_GYM_SIM_NAME"}
    old_path = child.get("PATH", "")
    child["PATH"] = os.pathsep.join([str(python_bin)] + [part for part in old_path.split(os.pathsep)
                                                         if part and part != str(python_bin)])
    child["PYTHONNOUSERSITE"] = "1"
    child["PYTHONPATH"] = str(ROOT)
    return child


def _file_provenance(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"runtime provenance file is missing: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def _require_float32_close(actual, expected, label: str, tolerance: float = 1e-6) -> None:
    """Accept only documented float32 materialization error, never scalar config drift."""
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    max_error = (float(np.max(np.abs(actual_values - expected_values)))
                 if actual_values.shape == expected_values.shape and actual_values.size else 0.0)
    if (actual_values.shape != expected_values.shape
            or not np.isfinite(actual_values).all()
            or max_error > tolerance):
        raise RuntimeError(f"instantiated float32 contract drift: {label}")


def runtime_software_provenance(torch_module, import_origin: Path,
                                task_origin: Path, planner_origin: Path) -> dict[str, Any]:
    """Capture the external runtime identities that source manifests cannot describe."""
    python_path = Path(sys.executable).resolve()
    ninja_path = (python_path.parent / "ninja").resolve()
    if not ninja_path.is_file() or not os.access(str(ninja_path), os.X_OK):
        raise RuntimeError("selected Python environment has no executable ninja")
    ninja_version = subprocess.run([str(ninja_path), "--version"], check=True,
                                   capture_output=True, text=True).stdout.strip()
    isaac = sys.modules.get("isaacgym")
    isaac_path = Path(getattr(isaac, "__file__", "")).resolve() if isaac is not None else None
    if isaac_path is None or not isaac_path.is_file():
        raise RuntimeError("Isaac Gym import origin is unavailable")
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    current_device = int(torch_module.cuda.current_device()) if cuda_available else None
    gpu_names = ([str(torch_module.cuda.get_device_name(i)) for i in range(device_count)]
                 if cuda_available else [])
    if not cuda_available or current_device is None or not gpu_names:
        raise RuntimeError("CUDA/GPU identity is unavailable; refusing forensic cell")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi is unavailable; refusing forensic cell")
    nvidia_query = subprocess.run(
        [nvidia_smi, "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.strip()
    if not nvidia_query:
        raise RuntimeError("nvidia-smi returned no GPU identity")
    nvidia_rows = [line.strip() for line in nvidia_query.splitlines() if line.strip()]
    if len(nvidia_rows) < device_count:
        raise RuntimeError("nvidia-smi GPU identity count disagrees with torch")
    driver_version = None
    get_driver = getattr(getattr(torch_module, "_C", None), "_cuda_getDriverVersion", None)
    if get_driver is not None and cuda_available:
        driver_version = int(get_driver())
    if driver_version is None:
        driver_version = nvidia_rows[0].split(",", 1)[0].strip()
    return {
        "python": {"executable": str(python_path), "executable_sha256": sha256(python_path),
                   "version": sys.version, "implementation": sys.implementation.name},
        "torch": {"version": str(torch_module.__version__),
                   "origin": str(Path(torch_module.__file__).resolve()),
                   "origin_sha256": sha256(Path(torch_module.__file__).resolve()),
                   "compiled_cuda_version": str(torch_module.version.cuda)},
        "isaac_gym": _file_provenance(isaac_path),
        "ninja": {"path": str(ninja_path), "sha256": sha256(ninja_path),
                  "version": ninja_version},
        "cuda": {"available": cuda_available, "device_count": device_count,
                  "current_device": current_device, "gpu_names": gpu_names,
                  "driver_version": driver_version, "nvidia_smi": {
                      "path": str(Path(nvidia_smi).resolve()),
                      "sha256": sha256(Path(nvidia_smi).resolve()),
                      "query": nvidia_query}},
        "repo_modules": {
            "aerial_gym": _file_provenance(import_origin),
            "navrl_task": _file_provenance(task_origin),
            "target_route_planner": _file_provenance(planner_origin),
            "target_motion": _file_provenance(ROOT / "aerial_gym/task/navrl_task/target_motion.py"),
            "physical_target_controller": _file_provenance(ROOT / "aerial_gym/task/navrl_task/physical_target.py"),
        },
    }


def runtime_contract_attestation(task, density: int, speed: float) -> dict[str, Any]:
    """Record values observed after task construction/reset, not merely environment labels."""
    physics = task._runtime_physics_contract()
    arena = task._arena_contract()
    bounds = task.obs_dict["env_bounds_max"][0] - task.obs_dict["env_bounds_min"][0]
    bounds_xyz = [float(value) for value in bounds.detach().cpu().tolist()]
    tm = task.tm
    route = task._target_route_manager
    controller = task._target_controller
    robot = task._robot_provenance
    sim_cfg = task.sim_env.sim_config
    sim_type = sim_cfg if isinstance(sim_cfg, type) else type(sim_cfg)
    sim_module = Path(__import__(sim_type.__module__, fromlist=["__file__"]).__file__).resolve()
    contract = {
        "density": int(task.n_bars_active), "requested_density": int(density),
        "speed_mps": float(speed), "route_mode": str(task._target_route_mode),
        "target_dynamics": str(task._target_dynamics), "target_pattern": str(tm.pattern),
        "target": {"max_accel_mps2": float(tm.max_accel),
                    "max_turn_rate_degps": float(tm.max_turn_rate_deg),
                    "lookahead_s": float(tm.avoidance_lookahead_s),
                    "speed_min_mps": float(tm.speed_min), "speed_final_mps": float(tm.speed_final),
                    "speed_fixed_mps": float(tm.speed_fixed),
                    "physical_mass_kg": float(tm.physical_mass),
                    "motor_arm_xy_m": float(tm.physical_motor_arm_xy),
                    "max_motor_thrust_n": float(tm.physical_max_motor_thrust),
                    "motor_tau_s": float(tm.physical_motor_tau),
                    "velocity_kp": float(controller.velocity_kp),
                    "altitude_kp": float(controller.altitude_kp),
                    "attitude_kp": [float(v) for v in controller.attitude_kp[0].detach().cpu().tolist()],
                    "rate_kp": [float(v) for v in controller.rate_kp[0].detach().cpu().tolist()],
                    "yaw_torque_ratio_m": float(tm.physical_yaw_torque_ratio),
                    "max_tilt_deg": float(tm.physical_max_tilt_deg),
                    "box_xyz_m": [float(v) for v in tm.physical_box_xyz],
                    "obstacle_clearance_m": float(tm.obstacle_clearance),
                    "tracking_margin_m": float(tm.physical_tracking_margin),
                    "boundary_margin_m": float(tm.physical_boundary_margin)},
        "route": {"resolution_m": float(route.config.resolution_m),
                  "support_xy_m": [float(v) for v in task._target_route_support_xy[0].detach().cpu().tolist()],
                  "replan_cooldown_steps": int(route.config.replan_cooldown_steps),
                  "goal_tolerance_m": float(route.config.goal_tolerance_m),
                  "min_goal_distance_m": float(tm.route_min_goal_distance_m),
                  "goal_exclusion_radius_m": float(route.config.goal_exclusion_radius_m),
                  "max_expansions": int(route.config.max_expansions),
                  "max_waypoints": int(route.config.max_waypoints)},
        "arena": {"bounds_xyz_m": bounds_xyz, **arena},
        "bar_capacity": int(task.obs_dict["obstacle_position"].shape[1] - task._bar_offset),
        "physics": physics,
        "robot": {key: robot[key] for key in (
            "robot_name", "robot_config_path", "robot_config_sha256",
            "robot_asset_path", "robot_asset_sha256") if key in robot},
        "sim": {"config_class": str(sim_type.__name__), "config_module": sim_type.__module__,
                "config_path": str(sim_module), "config_sha256": sha256(sim_module)},
    }
    expected = {"density": density, "speed_mps": speed, "route_mode": "global_astar_v1",
                "target_dynamics": "physical", "target_pattern": "waypoint"}
    for key, value in expected.items():
        if contract[key] != value:
            raise RuntimeError(f"instantiated runtime contract drift: {key}={contract[key]!r}")
    if bounds_xyz != [40.0, 40.0, 3.0]:
        raise RuntimeError(f"instantiated arena bounds drift: {bounds_xyz!r}")
    exact = {
        "target.max_accel_mps2": (contract["target"]["max_accel_mps2"], 4.0),
        "target.max_turn_rate_degps": (contract["target"]["max_turn_rate_degps"], 150.0),
        "target.lookahead_s": (contract["target"]["lookahead_s"], 1.0),
        "target.physical_mass_kg": (contract["target"]["physical_mass_kg"], 1.20),
        "target.motor_arm_xy_m": (contract["target"]["motor_arm_xy_m"], 0.0777817),
        "target.max_motor_thrust_n": (contract["target"]["max_motor_thrust_n"], 9.60),
        "target.motor_tau_s": (contract["target"]["motor_tau_s"], 0.04),
        "target.velocity_kp": (contract["target"]["velocity_kp"], 2.5),
        "target.altitude_kp": (contract["target"]["altitude_kp"], 4.0),
        "target.yaw_torque_ratio_m": (contract["target"]["yaw_torque_ratio_m"], 0.01),
        "target.obstacle_clearance_m": (contract["target"]["obstacle_clearance_m"], 0.77),
        "target.max_tilt_deg": (contract["target"]["max_tilt_deg"], 45.0),
        "target.tracking_margin_m": (contract["target"]["tracking_margin_m"], 0.45),
        "target.boundary_margin_m": (contract["target"]["boundary_margin_m"], 0.75),
        "route.resolution_m": (contract["route"]["resolution_m"], 0.25),
        "route.replan_cooldown_steps": (contract["route"]["replan_cooldown_steps"], 10),
        "route.min_goal_distance_m": (contract["route"]["min_goal_distance_m"], 6.0),
        "route.goal_tolerance_m": (contract["route"]["goal_tolerance_m"], 0.05),
        "route.goal_exclusion_radius_m": (contract["route"]["goal_exclusion_radius_m"], 1.0),
        "route.max_expansions": (contract["route"]["max_expansions"], 50000),
        "route.max_waypoints": (contract["route"]["max_waypoints"], 128),
        "physics.physics_dt_s": (contract["physics"]["physics_dt_s"], 0.01),
        "physics.physics_substeps": (contract["physics"]["physics_substeps"], 1),
        "physics.physics_steps_per_rl_step": (contract["physics"]["physics_steps_per_rl_step"], 10),
        "physics.rl_step_dt_s": (contract["physics"]["rl_step_dt_s"], 0.10),
    }
    for name, (actual, expected_value) in exact.items():
        if abs(float(actual) - float(expected_value)) > 1e-9:
            raise RuntimeError(f"instantiated runtime contract drift: {name}={actual!r}")
    if abs(float(contract["target"]["speed_fixed_mps"]) - float(speed)) > 1e-9:
        raise RuntimeError("instantiated target speed contract drift")
    if contract["bar_capacity"] != 300 or contract["target"]["box_xyz_m"] != [0.28, 0.28, 0.12]:
        raise RuntimeError("instantiated bar capacity/target box contract drift")
    _require_float32_close(contract["target"]["attitude_kp"], [0.08, 0.08, 0.04], "attitude_kp")
    _require_float32_close(contract["target"]["rate_kp"], [0.04, 0.04, 0.03], "rate_kp")
    _require_float32_close(contract["route"]["support_xy_m"],
                           [0.2068816086567407, 0.2068816086567407], "support_xy_m")
    if (contract["target"]["speed_min_mps"] != float(speed)
            or contract["target"]["speed_final_mps"] != float(speed)):
        raise RuntimeError("instantiated speed min/final contract drift")
    if (contract["arena"]["cfg_bar_pool"] != "bars_h3"
            or contract["arena"]["cfg_placement_mode"] != "navrl_band"
            or contract["arena"]["cfg_placement_gap_m"] != 1.6
            or contract["arena"]["cfg_placement_touch_m"] != 0.4
            or contract["arena"]["cfg_bar_x_min"] != 0.0
            or contract["arena"]["cfg_bar_x_max"] != 1.0):
        raise RuntimeError("instantiated bar placement contract drift")
    if (contract["sim"]["config_class"] != "BaseSimConfig"
            or not contract["sim"]["config_path"].endswith("aerial_gym/config/sim_config/base_sim_config.py")
            or contract["robot"].get("robot_name") != "navrl_ref5in_quad"
            or not contract["robot"].get("robot_asset_path", "").endswith("resources/robots/quad/quad_navrl_ref5in.urdf")):
        raise RuntimeError("instantiated sim/robot provenance contract drift")
    return contract


def run_cell(density: int, speed: float, output: Path) -> None:
    _configure(density, speed)
    sys.argv[:] = [sys.argv[0]]
    from aerial_gym.registry.task_registry import task_registry
    import aerial_gym
    import aerial_gym.task.navrl_task.navrl_task as navrl_task_module
    import aerial_gym.task.navrl_task.target_route_planner as route_planner_module
    import torch
    import_origin = Path(aerial_gym.__file__).resolve()
    task_origin = Path(navrl_task_module.__file__).resolve()
    planner_origin = Path(route_planner_module.__file__).resolve()
    expected_origins = {
        import_origin: ROOT / "aerial_gym/__init__.py",
        task_origin: ROOT / "aerial_gym/task/navrl_task/navrl_task.py",
        planner_origin: ROOT / "aerial_gym/task/navrl_task/target_route_planner.py",
    }
    if any(actual != expected for actual, expected in expected_origins.items()):
        raise RuntimeError("aerial_gym import escaped or drifted from the worktree")
    _, runtime_manifest_sha = runtime_source_manifest()
    task = task_registry.make_task("navrl_task", seed=SEED, num_envs=ENVS, headless=True, use_warp=True)
    task.seed(SEED)
    task._set_active_bars(density)
    boundary_attestation = {
        "runtime_wall_margin_m": float(task.cur.wall_margin),
        "route_boundary_margin_m": float(task._target_route_manager.config.boundary_margin_m),
        "physical_boundary_margin_m": float(task.tm.physical_boundary_margin),
        "tracking_margin_m": float(task.tm.physical_tracking_margin),
    }
    if (abs(boundary_attestation["runtime_wall_margin_m"] - 0.50) > 1e-9
            or abs(boundary_attestation["route_boundary_margin_m"] - 1.25) > 1e-9
            or abs(boundary_attestation["physical_boundary_margin_m"] - 0.75) > 1e-9
            or abs(boundary_attestation["tracking_margin_m"] - TRACKING_MARGIN_M) > 1e-9):
        raise RuntimeError("boundary/support contract drift; refusing forensic cell")
    recorder = attach_observer(task)
    task.reset()
    runtime_contract = runtime_contract_attestation(task, density, speed)
    software_provenance = runtime_software_provenance(
        torch, import_origin, task_origin, planner_origin)
    zero_policy_action = torch.zeros((ENVS, 4), device=task.device)
    rollout_started = time.perf_counter()
    for _ in range(STEPS):
        interval_start_step = int(task.num_task_steps)
        task._target_controller.begin_control_interval()
        task._advance_target()
        # Keep canonical NavRLTask ordering: target advance, neutral policy mapping, physics,
        # failed-env reset, then exactly one task-clock increment.
        command = task.transform_action_to_command(zero_policy_action)
        if tuple(command.shape) != tuple(zero_policy_action.shape) or not bool(command.isfinite().all()):
            raise RuntimeError("canonical neutral pursuer command contract drift")
        task.sim_env.step(actions=command)
        contact = task._target_controller.contact_seen.clone()
        bmin, bmax = task.obs_dict["env_bounds_min"], task.obs_dict["env_bounds_max"]
        support = task._physical_target_support_xyz()
        invalid = ((task.target_position[:, :2] - support[:, :2] < bmin[:, :2])
                   | (task.target_position[:, :2] + support[:, :2] > bmax[:, :2])).any(dim=1)
        invalid |= ((task.target_position[:, 2] - support[:, 2] < bmin[:, 2])
                    | (task.target_position[:, 2] + support[:, 2] > bmax[:, 2])
                    | ~torch.isfinite(task.target_position).all(dim=1))
        failed = contact | invalid
        if bool(failed.any()):
            failed_ids = failed.nonzero(as_tuple=False).squeeze(-1)
            task.sim_env.reset_idx(failed_ids)
            task.reset_idx(failed_ids)
        if int(task.num_task_steps) != interval_start_step:
            raise RuntimeError("task clock changed before canonical forensic increment")
        task.num_task_steps += 1
        if int(task.num_task_steps) != interval_start_step + 1:
            raise RuntimeError("task clock did not advance exactly once")
    rollout_wall_s = time.perf_counter() - rollout_started
    payload = {"schema": SCHEMA + "_cell", "contract": frozen_contract(), "density": density,
               "speed_mps": speed, "runtime_contract": runtime_contract,
               "software_provenance": software_provenance,
               "source": {"git_head": os.popen("git rev-parse HEAD").read().strip(),
               "tool_sha256": sha256(Path(__file__).resolve()),
               "import_origin": str(import_origin), "import_origin_sha256": sha256(import_origin),
               "planner_path": str(planner_origin), "planner_sha256": sha256(planner_origin),
               "task_path": str(task_origin), "task_sha256": sha256(task_origin),
               "runtime_manifest_sha256": runtime_manifest_sha},
               "boundary_attestation": boundary_attestation,
               "neutral_pursuer_command_contract": {
                   "policy_action": "all_zero_[N,4]",
                   "mapping": "NavRLTask.transform_action_to_command",
                   "mapping_order": "after_target_advance_before_sim_step",
                   "mapping_calls": STEPS,
               },
               "observer": {"event_count": len(recorder.events),
                            "wall_s": recorder.observer_wall_s,
                            "compute_wall_s": recorder.observer_compute_wall_s,
                            "transfer_wall_s": recorder.observer_transfer_wall_s,
                            "rollout_wall_s": rollout_wall_s,
                            "wall_share": recorder.observer_wall_s / max(rollout_wall_s, 1e-9),
                            "anchor_search_events": sum(
                                int("nearest_soft_free_anchor" in event) for event in recorder.events)},
               "events": recorder.events, "fallback_by_origin": list(recorder.fallback_by_origin.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preflight() -> int:
    contract = frozen_contract()
    if tuple(contract["densities"]) != DENSITIES or tuple(contract["speeds_mps"]) != SPEEDS:
        raise RuntimeError("frozen cell contract drift")
    print(json.dumps({"schema": SCHEMA + "_preflight", "contract": contract}, indent=2))
    return 0


def verify_receipt(directory: Path) -> int:
    """Verify the separate forensic receipt without touching attempt2 artifacts."""
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("forensic receipt is missing")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA + "_receipt":
        raise RuntimeError("forensic receipt schema mismatch")
    if payload.get("contract") != frozen_contract():
        raise RuntimeError("forensic receipt frozen contract mismatch")
    if payload.get("tool_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("forensic tool hash mismatch")
    recorded_head = payload.get("git_head")
    if (not isinstance(recorded_head, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40,64}", recorded_head)):
        raise RuntimeError("forensic receipt has no valid recorded git commit")
    commit_check = subprocess.run(["git", "cat-file", "-e", recorded_head + "^{commit}"],
                                  cwd=ROOT, check=False, capture_output=True, text=True)
    if commit_check.returncode != 0:
        raise RuntimeError("forensic recorded git commit is missing")
    # worktree_clean is an execution-time claim.  Post-run verification intentionally does not
    # require the current HEAD to equal the recorded commit: committing/cherry-picking the result
    # must not invalidate it when the bound runtime bytes are unchanged.
    if payload.get("worktree_clean") is not True:
        raise RuntimeError("forensic receipt execution was not clean")
    runtime_manifest, runtime_manifest_sha = runtime_source_manifest()
    manifest_hashes = {entry["path"]: entry["sha256"] for entry in runtime_manifest}
    if payload.get("runtime_manifest_sha256") != runtime_manifest_sha:
        raise RuntimeError("forensic runtime source manifest drift")
    if payload.get("runtime_source_manifest") != runtime_manifest:
        raise RuntimeError("forensic runtime source manifest entries drift")
    software = payload.get("software_provenance")
    if not isinstance(software, dict):
        raise RuntimeError("forensic software provenance is missing")
    for group in ("python", "torch", "isaac_gym", "ninja"):
        record = software.get(group, {})
        path_value = record.get("path") or record.get("executable") or record.get("origin")
        digest = record.get("sha256") or record.get("executable_sha256") or record.get("origin_sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise RuntimeError(f"forensic {group} provenance is incomplete")
        if sha256(Path(path_value).resolve()) != digest:
            raise RuntimeError(f"forensic {group} runtime bytes drift")
    nvidia = (software.get("cuda") or {}).get("nvidia_smi") or {}
    nvidia_path = Path(nvidia.get("path", "")).resolve()
    if (not nvidia_path.is_file()
            or sha256(nvidia_path) != nvidia.get("sha256")
            or not nvidia.get("query")):
        raise RuntimeError("forensic nvidia-smi provenance drift")
    for record in software.get("repo_modules", {}).values():
        path = Path(record.get("path", "")).resolve()
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError("forensic runtime module bytes drift")
    if payload.get("attempt2_artifacts_read_only") is not True or payload.get("original_evaluator_unchanged") is not True:
        raise RuntimeError("forensic receipt claim-boundary markers missing")
    expected = {(density, speed) for density in DENSITIES for speed in SPEEDS}
    observed = set()
    recorded_runtime_contracts = {(int(row.get("density")), float(row.get("speed_mps"))): row.get("contract")
                                 for row in payload.get("runtime_contracts", [])}
    if set(recorded_runtime_contracts) != expected:
        raise RuntimeError("forensic receipt runtime contract grid mismatch")
    for entry in payload.get("cells", []):
        density, speed = int(entry["density"]), float(entry["speed_mps"])
        if (density, speed) in observed or (density, speed) not in expected:
            raise RuntimeError("forensic receipt cell grid mismatch")
        observed.add((density, speed))
        path = (directory / entry["path"]).resolve()
        if directory.resolve() not in path.parents or "routed_gate_seed827_attempt2" in str(path):
            raise RuntimeError("forensic cell escapes its output directory")
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise RuntimeError("forensic cell hash mismatch")
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("schema") != SCHEMA + "_cell" or cell.get("contract") != frozen_contract():
            raise RuntimeError("forensic cell contract mismatch")
        source = cell.get("source", {})
        expected_paths = {
            "import_origin": ROOT / "aerial_gym/__init__.py",
            "planner_path": ROOT / "aerial_gym/task/navrl_task/target_route_planner.py",
            "task_path": ROOT / "aerial_gym/task/navrl_task/navrl_task.py",
        }
        for source_key in ("import_origin", "planner_path", "task_path"):
            source_path = Path(source.get(source_key, "")).resolve()
            if source_path != expected_paths[source_key]:
                raise RuntimeError("forensic imported source path mismatch")
        if source.get("import_origin_sha256") != sha256(Path(source["import_origin"]).resolve()):
            raise RuntimeError("forensic import-origin hash mismatch")
        if source.get("planner_sha256") != sha256(Path(source["planner_path"]).resolve()):
            raise RuntimeError("forensic planner hash mismatch")
        if source.get("task_sha256") != sha256(Path(source["task_path"]).resolve()):
            raise RuntimeError("forensic task hash mismatch")
        if source.get("runtime_manifest_sha256") != runtime_manifest_sha:
            raise RuntimeError("forensic cell runtime manifest mismatch")
        if source.get("tool_sha256") != payload.get("tool_sha256"):
            raise RuntimeError("forensic cell tool hash mismatch")
        if source.get("git_head") != payload.get("git_head"):
            raise RuntimeError("forensic cell git head mismatch")
        if cell.get("software_provenance") != software:
            raise RuntimeError("forensic cell software provenance mismatch")
        runtime_contract = cell.get("runtime_contract")
        if not isinstance(runtime_contract, dict):
            raise RuntimeError("forensic cell runtime contract is missing")
        if (runtime_contract.get("density") != density
                or abs(float(runtime_contract.get("speed_mps")) - speed) > 1e-12
                or runtime_contract.get("requested_density") != density):
            raise RuntimeError("forensic instantiated density/speed contract mismatch")
        if recorded_runtime_contracts[(density, speed)] != runtime_contract:
            raise RuntimeError("forensic receipt runtime contract drift")
        robot_contract = runtime_contract.get("robot", {})
        for path_key, hash_key in (("robot_config_path", "robot_config_sha256"),
                                   ("robot_asset_path", "robot_asset_sha256")):
            relative = robot_contract.get(path_key)
            if not isinstance(relative, str) or manifest_hashes.get(relative) != robot_contract.get(hash_key):
                raise RuntimeError("forensic robot source is not bound to runtime manifest")
    if observed != expected:
        raise RuntimeError("forensic receipt is missing a frozen cell")
    if directory.resolve() == OUTPUT_ROOT.resolve():
        marker = directory / ".COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError("forensic completion marker is missing")
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        if (marker_payload.get("schema") != SCHEMA + "_complete"
                or marker_payload.get("receipt_sha256") != sha256(receipt_path)
                or marker_payload.get("summary_sha256") != sha256(directory / "summary.json")):
            raise RuntimeError("forensic completion marker hash mismatch")
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = _build_summary(directory, payload)
        if stored_summary != expected_summary:
            raise RuntimeError("forensic summary semantic/hash mismatch")
    print(json.dumps({"verified": True, "cells": len(observed), "receipt": str(receipt_path)}))
    return 0


def _build_summary(directory: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the descriptive summary from verified cells without writing files."""
    event_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    unsafe_counts: dict[str, int] = {}
    rounded_aabb_disagreements = 0
    fallback_intervals = 0
    local_fallback_intervals = 0
    plan_status_counts = {"initial_plan": {}, "replan": {}}
    unsafe_replan_by_origin = {}
    unsafe_replan_unattributed = []
    unsafe_replan_other_origin = []
    origin_reasons = {}
    per_cell = []
    invalidation_count = 0
    metric_values: dict[str, list[float]] = {key: [] for key in (
        "hard_clearance_m", "soft_clearance_m", "route_cross_track_error_m",
        "route_polyline_error_m", "fallback_age_steps", "target_speed_mps",
        "nearest_anchor_distance_m")}
    for entry in receipt["cells"]:
        cell = json.loads((directory / entry["path"]).read_text(encoding="utf-8"))
        cell_key = "%s@%s" % (entry["density"], entry["speed_mps"])
        cell_events = cell.get("events", [])
        cell_status_counts, cell_reason_counts = {}, {}
        for event in cell_events:
            if event.get("event") in ("initial_plan", "replan"):
                status = str(event.get("plan_status", event.get("reason")))
                cell_status_counts[status] = cell_status_counts.get(status, 0) + 1
            if event.get("reason") is not None:
                reason_value = str(event["reason"])
                cell_reason_counts[reason_value] = cell_reason_counts.get(reason_value, 0) + 1
        origin_detail = {}
        for event in cell_events:
            if event.get("event") == "invalidation":
                origin_id = str(event.get("event_id"))
                origin_detail[origin_id] = {
                    "origin_id": origin_id, "origin_reason": event.get("reason"),
                    "first_unsafe_status": None, "unsafe_replan_count": 0,
                    "repeats": 0, "fallback_intervals": 0, "local_fallback_intervals": 0,
                    "max_fallback_age_steps": 0,
                }
        for event in cell_events:
            if (event.get("event") == "replan"
                    and event.get("plan_status", event.get("reason")) in ("unsafe_start", "unsafe_start_cell")):
                origin_id = event.get("origin_invalidation_id")
                if origin_id is None or str(origin_id) not in origin_detail:
                    continue
                detail = origin_detail[str(origin_id)]
                detail["unsafe_replan_count"] += 1
                if detail["first_unsafe_status"] is None:
                    detail["first_unsafe_status"] = event.get("plan_status", event.get("reason"))
                else:
                    detail["repeats"] += 1
        for row in cell.get("fallback_by_origin", []):
            origin_id = str(row.get("invalidation_event_id"))
            if origin_id in origin_detail:
                detail = origin_detail[origin_id]
                intervals = int(row.get("intervals", 0))
                detail["fallback_intervals"] += intervals
                if row.get("reason") == "local_step_infeasible":
                    detail["local_fallback_intervals"] += intervals
                detail["max_fallback_age_steps"] = max(
                    detail["max_fallback_age_steps"], int(row.get("max_age_steps", 0)))
        cell_local_ids = {str(event.get("event_id")) for event in cell_events
                          if event.get("event") == "invalidation"
                          and event.get("reason") == "local_step_infeasible"}
        cell_local_origins = {str(event.get("origin_invalidation_id")) for event in cell_events
                              if event.get("event") == "replan"
                              and event.get("plan_status", event.get("reason")) in ("unsafe_start", "unsafe_start_cell")
                              and str(event.get("origin_invalidation_id")) in cell_local_ids}
        for event in cell.get("events", []):
            kind = str(event.get("event"))
            event_counts[kind] = event_counts.get(kind, 0) + 1
            reason = event.get("reason")
            if kind == "invalidation":
                origin_reasons[cell_key + "#" + str(event.get("event_id"))] = reason
            if reason is not None:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
            if kind in plan_status_counts:
                status = str(event.get("plan_status", reason))
                bucket = plan_status_counts[kind]
                bucket[status] = bucket.get(status, 0) + 1
                if kind == "replan" and status in ("unsafe_start", "unsafe_start_cell"):
                    origin = event.get("origin_invalidation_id")
                    if origin is None:
                        unsafe_replan_unattributed.append(event)
                    elif origin_reasons.get(cell_key + "#" + str(origin)) == "local_step_infeasible":
                        unsafe_replan_by_origin.setdefault(cell_key + "#" + str(origin), event)
                    else:
                        unsafe_replan_other_origin.append(event)
            if kind == "invalidation" and reason == "local_step_infeasible":
                invalidation_count += 1
            unsafe = event.get("unsafe_start_reason")
            if unsafe not in (None, "none"):
                unsafe_counts[str(unsafe)] = unsafe_counts.get(str(unsafe), 0) + 1
            rounded_aabb_disagreements += int(bool(event.get("rounded_vs_aabb_soft_disagreement")))
            for key in metric_values:
                value = event.get(key)
                if key == "nearest_anchor_distance_m":
                    value = (event.get("nearest_soft_free_anchor") or {}).get("distance_m")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_values[key].append(float(value))
        fallback_intervals += sum(int(row.get("intervals", 0)) for row in cell.get("fallback_by_origin", []))
        local_fallback_intervals += sum(int(row.get("intervals", 0)) for row in cell.get("fallback_by_origin", [])
                                        if row.get("reason") == "local_step_infeasible")
        cell_local_fallback = sum(int(row.get("intervals", 0)) for row in cell.get("fallback_by_origin", [])
                                  if row.get("reason") == "local_step_infeasible")
        cell_local_invalidations = sum(1 for event in cell_events
                                       if event.get("event") == "invalidation"
                                       and event.get("reason") == "local_step_infeasible")
        per_cell.append({"density": entry["density"], "speed_mps": entry["speed_mps"],
                         "event_counts": {kind: sum(1 for event in cell_events if event.get("event") == kind)
                                           for kind in sorted({event.get("event") for event in cell_events})},
                         "plan_status_counts": cell_status_counts, "reason_counts": cell_reason_counts,
                         "unique_unsafe_start_replan_origins": len(cell_local_origins),
                         "origins": [origin_detail[key] for key in sorted(origin_detail)],
                         "fallback_intervals_local": cell_local_fallback,
                         "local_invalidation_count": cell_local_invalidations,
                         "fallback_amplification_local": cell_local_fallback / max(1, cell_local_invalidations)})
    unsafe_replan_rows = list(unsafe_replan_by_origin.values())
    unsafe_count = len(unsafe_replan_rows)
    hard_free_soft_unsafe = sum(bool(row.get("hard_free_exact")) and not bool(row.get("soft_free"))
                                for row in unsafe_replan_rows)
    anchor_rows = [row.get("nearest_soft_free_anchor", {}) for row in unsafe_replan_rows]
    # Denominator is every unsafe-start replan, including rows with no candidate anchor. Using
    # only exists rows would make the connector fraction tautologically one.
    connector_total = unsafe_count
    connector_safe = sum(bool(row.get("exists")) and bool(row.get("hard_connector_safe"))
                       for row in anchor_rows)
    def wilson_lower(success, total):
        if not total:
            return None
        z = 1.96
        p = float(success) / total
        denominator = 1.0 + z * z / total
        center = p + z * z / (2.0 * total)
        spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        return float((center - spread) / denominator)
    hard_soft_ratio = hard_free_soft_unsafe / max(1, unsafe_count)
    connector_ratio = connector_safe / max(1, connector_total)
    amplification = local_fallback_intervals / max(1, invalidation_count)
    decision = "RECOVERY_DOMINANT" if (
        unsafe_count >= 64 and wilson_lower(hard_free_soft_unsafe, unsafe_count) is not None
        and wilson_lower(hard_free_soft_unsafe, unsafe_count) > 0.5
        and wilson_lower(connector_safe, connector_total) is not None
        and wilson_lower(connector_safe, connector_total) > 0.5
        and amplification > 10.0
    ) else ("HARD_UNSAFE" if hard_soft_ratio <= 0.5 else "ANCHOR_INSUFFICIENT" if connector_ratio <= 0.5 else "INCONCLUSIVE")
    stats = {}
    for key, values in metric_values.items():
        if values:
            stats[key] = {"count": len(values), "median": float(np.median(values)),
                          "p90": float(np.quantile(values, 0.90)), "max": float(max(values))}
        else:
            stats[key] = {"count": 0, "median": None, "p90": None, "max": None}
    return {"schema": SCHEMA + "_summary", "contract": frozen_contract(),
               "receipt_sha256": sha256(directory / "receipt.json"),
               "event_counts": event_counts, "reason_counts": reason_counts,
               "unsafe_start_reason_counts": unsafe_counts,
               "rounded_vs_aabb_soft_disagreements": rounded_aabb_disagreements,
               "fallback_intervals": fallback_intervals, "metric_stats": stats,
               "local_fallback_intervals": local_fallback_intervals,
               "plan_status_counts": plan_status_counts,
               "per_cell": per_cell,
               "decision_rule": {
                   "unsafe_start_replan_n": unsafe_count,
                   "unsafe_start_replan_repeats": max(
                       0, sum(1 for cell_entry in receipt["cells"]
                              for event in json.loads((directory / cell_entry["path"]).read_text(encoding="utf-8")).get("events", [])
                              if event.get("event") == "replan"
                              and event.get("plan_status", event.get("reason")) in ("unsafe_start", "unsafe_start_cell"))
                       - unsafe_count - len(unsafe_replan_unattributed)
                       - len(unsafe_replan_other_origin)),
                   "unsafe_start_replan_unattributed": len(unsafe_replan_unattributed),
                   "unsafe_start_replan_other_origin": len(unsafe_replan_other_origin),
                   "unsafe_start_replan_min_n": 64,
                   "hard_free_soft_unsafe_ratio": hard_soft_ratio,
                   "hard_free_soft_unsafe_wilson95_lower": wilson_lower(hard_free_soft_unsafe, unsafe_count),
                   "exact_hard_connector_ratio": connector_ratio,
                   "exact_hard_connector_wilson95_lower": wilson_lower(connector_safe, connector_total),
                   "attributed_fallback_to_local_invalidation": amplification,
                   "cooldown_steps": 10, "amplification_gt_cooldown": amplification > 10.0,
                   "verdict": decision,
               },
               "interpretation": "descriptive_only_no_gate_or_tuning_authority"}


def summarize(directory: Path, *, internal_partial: bool = False) -> int:
    """Create the descriptive summary only after receipt and every cell verify."""
    directory = directory.resolve()
    if (not internal_partial or directory == OUTPUT_ROOT.resolve()
            or directory.parent != OUTPUT_ROOT.parent
            or not directory.name.startswith(OUTPUT_ROOT.name + ".partial-")):
        raise RuntimeError("completed forensic artifacts are immutable; summarize only an authorized partial run")
    verify_receipt(directory)
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    summary = _build_summary(directory, receipt)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(directory / "summary.json"),
                      "fallback_intervals": summary["fallback_intervals"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--_cell-density", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_cell-speed", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--_auth-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify:
        return verify_receipt(Path(args.output).resolve())
    if args.summarize:
        raise RuntimeError("--summarize is internal-only; completed forensic artifacts are immutable")
    if args.preflight or not args.run and args._cell_density is None:
        return preflight()
    out = Path(args.output).resolve()
    if args._cell_density is not None:
        partial = out.parent
        if (args._auth_token != authorization_token(partial)
                or partial.parent != OUTPUT_ROOT.parent
                or not partial.name.startswith(OUTPUT_ROOT.name + ".partial-")
                or out.exists() or "routed_gate_seed827_attempt2" in str(out)):
            raise RuntimeError("unauthorized or unsafe forensic child output")
        run_cell(args._cell_density, float(args._cell_speed), out)
        return 0
    if out != OUTPUT_ROOT or out.exists():
        raise RuntimeError("forensic run requires a fresh canonical OUTPUT_ROOT")
    if list(OUTPUT_ROOT.parent.glob(OUTPUT_ROOT.name + ".partial-*")):
        raise RuntimeError("forensic partial output exists; refusing rerun or mixed cells")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError("forensic run requires a clean committed worktree")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    partial = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".partial-" + str(os.getpid()))
    if partial.exists():
        raise RuntimeError("forensic partial output already exists; refusing rerun")
    partial.mkdir(parents=True)
    out = partial
    token = authorization_token(partial)
    runtime_manifest, runtime_manifest_sha = runtime_source_manifest()
    cells = []
    runtime_contracts = []
    software_provenance = None
    for density in DENSITIES:
        for speed in SPEEDS:
            cell = out / f"route_on__speed_{str(speed).replace('.', 'p')}__bars_{density}.json"
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--_cell-density", str(density), "--_cell-speed", str(speed),
                       "--output", str(cell), "--_auth-token", token]
            completed = subprocess.run(command, cwd=ROOT, env=build_child_environment(), check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"forensic child failed: density={density} speed={speed}")
            cell_payload = json.loads(cell.read_text(encoding="utf-8"))
            cell_software = cell_payload.get("software_provenance")
            if software_provenance is None:
                software_provenance = cell_software
            elif cell_software != software_provenance:
                raise RuntimeError("forensic child software provenance differs across cells")
            runtime_contracts.append({"density": density, "speed_mps": speed,
                                      "contract": cell_payload.get("runtime_contract")})
            cells.append({"density": density, "speed_mps": speed, "path": str(cell.relative_to(out)), "sha256": sha256(cell)})
    if software_provenance is None:
        raise RuntimeError("forensic children produced no software provenance")
    receipt = {"schema": SCHEMA + "_receipt", "contract": frozen_contract(), "tool_sha256": sha256(Path(__file__).resolve()),
               "git_head": git_head, "worktree_clean": True,
               "runtime_manifest_sha256": runtime_manifest_sha,
               "runtime_source_manifest": runtime_manifest,
               "software_provenance": software_provenance,
               "runtime_contracts": runtime_contracts,
               "cells": cells, "attempt2_artifacts_read_only": True, "original_evaluator_unchanged": True}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summarize(out, internal_partial=True)
    verify_receipt(out)
    marker = out / ".COMPLETE.json"
    marker_tmp = out / ".COMPLETE.json.tmp"
    marker_tmp.write_text(json.dumps({"schema": SCHEMA + "_complete",
                                      "receipt_sha256": sha256(out / "receipt.json"),
                                      "summary_sha256": sha256(out / "summary.json")}, indent=2) + "\n",
                          encoding="utf-8")
    os.replace(marker_tmp, marker)
    os.replace(out, OUTPUT_ROOT)
    verify_receipt(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
