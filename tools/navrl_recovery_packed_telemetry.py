#!/usr/bin/env python3
"""Evaluation-only packed telemetry for the two-envelope physical-target gate.

The observer is deliberately outside the task.  It wraps the already installed physics callback
and samples tensors on the device; policy observations, rewards, actions, resets and target state
are never written.  One compact NPZ is transferred at the end of each cell.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Mapping

import numpy as np
import torch


SCHEMA = "navrl_target_recovery_packed_telemetry_v1"
STATE_OFF = -1
STATE_NORMAL = 0
STATE_BRAKE = 1
STATE_CONNECT = 2
STATE_ROUTE = 3
STATE_NO_CONNECTOR = 4
STATUS_BRAKE_TIMEOUT = 22
STATUS_CONNECT_TIMEOUT = 23
STATUS_TIMEOUT = STATUS_BRAKE_TIMEOUT  # compatibility name for existing synthetic fixtures
HARD_EPSILON_M = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = path.resolve()
    if path.exists():
        raise RuntimeError("refusing to overwrite telemetry: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    if temporary.exists():
        raise RuntimeError("stale telemetry temporary exists: %s" % temporary)
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _signed_aabb_margin(position_xy, bars_xy, half_xy, lo_xy, hi_xy):
    """Signed exact-AABB margin; positive is free and touching is zero."""
    wall = torch.minimum(position_xy - lo_xy, hi_xy - position_xy).amin(dim=1)
    if bars_xy.shape[1] == 0:
        obstacle = torch.full_like(wall, float("inf"))
    else:
        delta = (position_xy.unsqueeze(1) - bars_xy).abs() - half_xy
        inside = (delta <= 0.0).all(dim=2)
        outside = delta.clamp(min=0.0).norm(dim=2)
        inside_depth = delta.amax(dim=2)
        obstacle = torch.where(inside, inside_depth, outside).amin(dim=1)
    return torch.minimum(wall, obstacle)


def _signed_aabb_margin_reason(position_xy, bars_xy, half_xy, lo_xy, hi_xy):
    """Return signed margin and stable reason (wall 1..4, bar 1000+index)."""
    wall_values = torch.stack(
        (position_xy[:, 0] - lo_xy[:, 0], hi_xy[:, 0] - position_xy[:, 0],
         position_xy[:, 1] - lo_xy[:, 1], hi_xy[:, 1] - position_xy[:, 1]), dim=1
    )
    wall, wall_index = wall_values.min(dim=1)
    wall_reason = wall_index.to(torch.int16) + 1
    if bars_xy.shape[1] == 0:
        return wall, wall_reason
    delta = (position_xy.unsqueeze(1) - bars_xy).abs() - half_xy
    inside = (delta <= 0.0).all(dim=2)
    per_bar = torch.where(inside, delta.amax(dim=2), delta.clamp(min=0.0).norm(dim=2))
    obstacle, bar_index = per_bar.min(dim=1)
    use_bar = obstacle <= wall
    reason = torch.where(use_bar, bar_index.to(torch.int16) + 1000, wall_reason)
    return torch.minimum(wall, obstacle), reason


def _geometry_valid(position_xy, bars_xy, half_xy, lo_xy, hi_xy):
    valid = torch.isfinite(position_xy).all(dim=1)
    valid &= torch.isfinite(lo_xy).all(dim=1) & torch.isfinite(hi_xy).all(dim=1)
    valid &= (hi_xy > lo_xy).all(dim=1)
    if bars_xy.shape[1]:
        valid &= torch.isfinite(bars_xy).all(dim=(1, 2))
        valid &= torch.isfinite(half_xy).all(dim=(1, 2))
        valid &= (half_xy >= 0.0).all(dim=(1, 2))
    return valid


def _segments_hit_closed_aabb(p0, p1, bars, half, hard_epsilon_m=HARD_EPSILON_M):
    """Vectorized [N,C] continuous segment/AABB intersection."""
    if bars.shape[1] == 0:
        return torch.zeros(p0.shape[:2], dtype=torch.bool, device=p0.device)
    start = p0.unsqueeze(2)
    direction = (p1 - p0).unsqueeze(2)
    box_lo = bars.unsqueeze(1) - half.unsqueeze(1) - float(hard_epsilon_m)
    box_hi = bars.unsqueeze(1) + half.unsqueeze(1) + float(hard_epsilon_m)
    parallel = direction.abs() <= 1e-9
    parallel_inside = (~parallel) | ((start >= box_lo) & (start <= box_hi))
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    t0 = (box_lo - start) / safe_direction
    t1 = (box_hi - start) / safe_direction
    axis_enter = torch.where(parallel, torch.full_like(t0, float("-inf")), torch.minimum(t0, t1))
    axis_exit = torch.where(parallel, torch.full_like(t0, float("inf")), torch.maximum(t0, t1))
    enter = axis_enter.amax(dim=3)
    leave = axis_exit.amin(dim=3)
    hit = parallel_inside.all(dim=3) & (enter <= leave) & (leave >= 0.0) & (enter <= 1.0)
    return hit.any(dim=2)


class _SubstepProxy:
    """Transparent task physics callback that records after the real controller callback."""

    def __init__(self, observer, controller):
        self.observer = observer
        self.controller = controller

    def __call__(self):
        return self.controller()

    def post_physics_step(self):
        self.controller.post_physics_step()
        self.observer.record_substep()

    def __getattr__(self, name):
        return getattr(self.controller, name)


class RecoveryPackedObserver:
    """Fixed-shape device-side observer for one density/speed/route cell."""

    def __init__(self, task, steps: int, physics_substeps: int):
        self.task = task
        self.ctrl = task._target_controller
        self.device = task.device
        self.steps = int(steps)
        self.physics_substeps = int(physics_substeps)
        self.envs = int(task.num_envs)
        if self.steps <= 0 or self.physics_substeps <= 0 or self.envs <= 0:
            raise ValueError("observer dimensions must be positive")
        self.route_enabled = bool(getattr(task, "_target_route_recovery_enabled", False))
        from aerial_gym.task.navrl_task.target_route_planner import (
            TARGET_ROUTE_HARD_EPSILON_M,
            TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
        )
        box = [float(value) for value in task.tm.physical_box_xyz]
        declared_support = 0.5 * math.sqrt(sum(value * value for value in box))
        self.support_xy = torch.full(
            (self.envs, 2), declared_support, dtype=task.target_position.dtype,
            device=self.device,
        )
        self.hard_reserve_m = float(
            TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M
        )
        measured_stop = os.environ.get("NAVRL_TARGET_RECOVERY_STOP_DISTANCE_P95_M", "")
        self.measured_stop_distance_p95_m = (
            float(measured_stop) if self.route_enabled and measured_stop else float("nan")
        )
        self.interval = -1
        self.substep = 0
        self._in_advance = False
        self._reset_calls_in_advance = torch.zeros(
            (self.steps,), dtype=torch.int32, device=self.device
        )
        self._install_reset_spies()
        self.original_callback = task.sim_env.physics_step_callback
        if self.original_callback is not self.ctrl:
            raise RuntimeError("physical target is not the installed physics callback")
        task.sim_env.set_physics_step_callback(_SubstepProxy(self, self.ctrl))

        shape = (self.steps, self.envs)
        subshape = (self.steps, self.physics_substeps, self.envs)
        self.i16 = {
            name: torch.full(shape, -1, dtype=torch.int16, device=self.device)
            for name in (
                "state_before", "state_after", "age_before", "age_after", "brake_age_before",
                "brake_age_after", "connect_age_before", "connect_age_after",
                "connect_timeout_steps", "status_after",
                "hard_reason_before", "soft_reason_before", "hard_reason_after",
                "soft_reason_after",
                "anchor_cell_i", "anchor_cell_j", "candidate_count", "candidate_horizon_steps",
                "candidate_selected_index", "candidate_safe_prefix_steps",
                "candidate_full_horizon_safe",
            )
        }
        self.i32 = {
            name: torch.zeros(shape, dtype=torch.int32, device=self.device)
            for name in (
                "entry_delta", "resume_delta", "no_connector_delta", "hard_breach_delta",
                "brake_timeout_delta", "connect_timeout_delta",
                "timeout_event", "direct_position_write", "reset_call_during_advance",
                "runner_reset_after_interval", "candidate_binding_error",
            )
        }
        self.f32 = {
            name: torch.full(shape, float("nan"), dtype=torch.float32, device=self.device)
            for name in (
                "hard_margin_before_m", "soft_margin_before_m", "hard_margin_after_m",
                "soft_margin_after_m", "speed_before_mps", "speed_after_mps",
                "stop_distance_m", "stop_margin_m", "anchor_distance_m",
                "anchor_distance_after_m", "connector_clearance_m", "formula_stop_distance_m",
                "planned_first_progress_m", "planned_horizon_progress_m",
            )
        }
        self.xy = {
            name: torch.full(shape + (2,), float("nan"), dtype=torch.float32, device=self.device)
            for name in ("position_before_xy", "position_after_xy", "velocity_before_xy",
                         "velocity_after_xy", "command_xy", "anchor_xy", "anchor_before_xy")
        }
        self.sub_f32 = {
            name: torch.full(subshape, float("nan"), dtype=torch.float32, device=self.device)
            for name in (
                "hard_margin_m", "support_xy_m", "contact_force_n", "velocity_error_mps",
                "tilt_deg",
            )
        }
        self.sub_i8 = {
            name: torch.zeros(subshape, dtype=torch.int8, device=self.device)
            for name in ("geometry_valid", "obb_valid", "motor_saturated", "watchdog_breach")
        }
        self._before_position = None
        self._before_counters = None
        self._exact_calls = []
        self._candidate_event_pairs = []
        self._candidate_cpu_s = 0.0
        self._certificate_calls = 0
        self._connector_cpu_s = 0.0
        self._connector_clearance_cache = torch.full(
            (self.envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self._install_candidate_probe()

    def _install_candidate_probe(self):
        self._task_module = importlib.import_module(self.task.__class__.__module__)
        self._original_bounded_step = self._task_module.bounded_drone_target_step

        def observed(*args, **kwargs):
            result = self._original_bounded_step(*args, **kwargs)
            exact = bool(kwargs.get("exact_aabb_clearance", False))
            if not exact and len(args) >= 15:
                exact = bool(args[14])
            if exact:
                self._certificate_calls += 1
                if len(result) != 5 or not isinstance(result[4], Mapping):
                    raise RuntimeError("exact CONNECT call omitted its selected-candidate certificate")
                certificate = result[4]
                old_xy = kwargs.get("old_xy", args[0] if args else None)
                required = {
                    "selected_index", "candidate_count", "horizon_steps",
                    "safe_prefix_steps", "full_horizon_safe", "immediate_feasible",
                    "selected_final_position_xy", "row_ids",
                }
                if old_xy is None or not required.issubset(certificate):
                    raise RuntimeError("CONNECT selected-candidate certificate is incomplete")
                self._exact_calls.append({
                    "old_xy": old_xy.detach().clone(),
                    "selected_first_xy": result[0].detach().clone(),
                    "selected_final_xy": certificate["selected_final_position_xy"].detach().clone(),
                    "row_ids": certificate["row_ids"].detach().clone(),
                    "selected_index": certificate["selected_index"].detach().clone(),
                    "count": int(certificate["candidate_count"]),
                    "horizon": int(certificate["horizon_steps"]),
                    "safe_prefix": certificate["safe_prefix_steps"],
                    "full_horizon_safe": certificate["full_horizon_safe"],
                    "selected_binding_error": torch.zeros_like(
                        certificate["full_horizon_safe"], dtype=torch.bool
                    ),
                })
            return result

        self._task_module.bounded_drone_target_step = observed

    def _install_reset_spies(self):
        self._task_reset_idx = self.task.reset_idx
        self._sim_reset_idx = self.task.sim_env.reset_idx

        def task_reset_idx(*args, **kwargs):
            if self._in_advance and self.interval >= 0:
                self._reset_calls_in_advance[self.interval] += 1
            return self._task_reset_idx(*args, **kwargs)

        def sim_reset_idx(*args, **kwargs):
            if self._in_advance and self.interval >= 0:
                self._reset_calls_in_advance[self.interval] += 1
            return self._sim_reset_idx(*args, **kwargs)

        self.task.reset_idx = task_reset_idx
        self.task.sim_env.reset_idx = sim_reset_idx

    def close(self):
        self.task.sim_env.set_physics_step_callback(self.original_callback)
        self.task.reset_idx = self._task_reset_idx
        self.task.sim_env.reset_idx = self._sim_reset_idx
        self._task_module.bounded_drone_target_step = self._original_bounded_step

    def _state(self):
        if not self.route_enabled or self.task._target_route_manager is None:
            return torch.full((self.envs,), STATE_OFF, dtype=torch.int16, device=self.device)
        return self.task._target_route_manager.recovery_state.to(torch.int16)

    def _age(self):
        if not self.route_enabled:
            return torch.full((self.envs,), -1, dtype=torch.int16, device=self.device)
        return self.task._target_route_manager.recovery_age_steps.to(torch.int16)

    def _phase_ages(self):
        if not self.route_enabled:
            missing = torch.full((self.envs,), -1, dtype=torch.int16, device=self.device)
            return missing, missing, missing
        manager = self.task._target_route_manager
        return (
            manager.recovery_brake_age_steps.to(torch.int16),
            manager.recovery_connect_age_steps.to(torch.int16),
            manager.recovery_connect_timeout_steps.to(torch.int16),
        )

    def _status(self):
        if self.task._target_route_manager is None:
            return torch.full((self.envs,), -1, dtype=torch.int16, device=self.device)
        return self.task._target_route_manager.status_code.to(torch.int16)

    def _counters(self):
        if not self.route_enabled:
            return tuple(torch.zeros((), dtype=torch.long, device=self.device) for _ in range(6))
        manager = self.task._target_route_manager
        return tuple(
            value.detach().clone()
            for value in (
                manager.recovery_entries, manager.recovery_route_resumes,
                manager.recovery_no_connector_count, manager.recovery_hard_breach_count,
                manager.recovery_brake_timeout_count, manager.recovery_connect_timeout_count,
            )
        )

    def _geometry(self, position_xy):
        bars = self.task.obs_dict["obstacle_position"][
            :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
        ]
        half = self.task.obs_dict["asset_collision_half_extents"][
            :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
        ]
        bounds_lo = self.task.obs_dict["env_bounds_min"][:, :2]
        bounds_hi = self.task.obs_dict["env_bounds_max"][:, :2]
        wall = float(self.task.cur.wall_margin)
        hard_lo = bounds_lo + wall + self.support_xy + self.hard_reserve_m
        hard_hi = bounds_hi - wall - self.support_xy - self.hard_reserve_m
        hard_half = half + self.support_xy.unsqueeze(1) + self.hard_reserve_m
        soft_wall = wall + float(self.task.tm.physical_boundary_margin)
        soft_lo = bounds_lo + soft_wall + self.support_xy
        soft_hi = bounds_hi - soft_wall - self.support_xy
        soft_half = (
            half + self.support_xy.unsqueeze(1)
            + float(self.task.tm.physical_tracking_margin) + self.hard_reserve_m
        )
        hard, hard_reason = _signed_aabb_margin_reason(
            position_xy, bars, hard_half, hard_lo, hard_hi
        )
        soft, soft_reason = _signed_aabb_margin_reason(
            position_xy, bars, soft_half, soft_lo, soft_hi
        )
        return hard, soft, hard_reason, soft_reason

    def begin_interval(self, interval: int):
        if interval != self.interval + 1 or not 0 <= interval < self.steps:
            raise RuntimeError("non-monotone observer interval")
        self.interval = interval
        self.substep = 0
        pos = self.ctrl.position[:, :2].detach()
        vel = self.ctrl.linvel[:, :2].detach()
        hard, soft, hard_reason, soft_reason = self._geometry(pos)
        self.i16["state_before"][interval] = self._state()
        self.i16["age_before"][interval] = self._age()
        brake_age, connect_age, connect_timeout = self._phase_ages()
        self.i16["brake_age_before"][interval] = brake_age
        self.i16["connect_age_before"][interval] = connect_age
        self.i16["connect_timeout_steps"][interval] = connect_timeout
        self.i16["hard_reason_before"][interval] = hard_reason
        self.i16["soft_reason_before"][interval] = soft_reason
        self.f32["hard_margin_before_m"][interval] = hard
        self.f32["soft_margin_before_m"][interval] = soft
        self.f32["speed_before_mps"][interval] = vel.norm(dim=1)
        self.xy["position_before_xy"][interval] = pos
        self.xy["velocity_before_xy"][interval] = vel
        if self.route_enabled:
            self.xy["anchor_before_xy"][interval] = (
                self.task._target_route_manager.recovery_anchor.detach()
            )
        self._before_position = pos.clone()
        self._before_counters = self._counters()
        self.ctrl.begin_control_interval()

    def advance_target(self):
        if self.interval < 0 or self._before_position is None:
            raise RuntimeError("begin_interval must precede advance_target")
        self._in_advance = True
        self._exact_calls = []
        try:
            self.task._advance_target()
        finally:
            self._in_advance = False
        row = self.interval
        after_direct = self.ctrl.position[:, :2].detach()
        changed = (after_direct != self._before_position).any(dim=1)
        self.i32["direct_position_write"][row] = changed.to(torch.int32)
        self.i32["reset_call_during_advance"][row] = self._reset_calls_in_advance[row]
        self.i16["state_after"][row] = self._state()
        self.i16["age_after"][row] = self._age()
        brake_age, connect_age, connect_timeout = self._phase_ages()
        self.i16["brake_age_after"][row] = brake_age
        self.i16["connect_age_after"][row] = connect_age
        self.i16["connect_timeout_steps"][row] = connect_timeout
        self.i16["status_after"][row] = self._status()
        after = self._counters()
        for name, before, final in zip(
            ("entry_delta", "resume_delta", "no_connector_delta", "hard_breach_delta",
             "brake_timeout_delta", "connect_timeout_delta"),
            self._before_counters, after,
        ):
            delta = final - before
            # Scalar event counts are retained in env slot zero; the summary denominator is explicit.
            self.i32[name][row, 0] = delta.to(torch.int32)
        # Count the transition event once.  STATUS_TIMEOUT remains latched while the manager is
        # in NO_CONNECTOR, so counting status alone would turn one timeout into every remaining
        # interval in the cell.
        self.i32["timeout_event"][row] = (
            (self.i16["state_after"][row] == STATE_NO_CONNECTOR)
            & (self.i16["state_before"][row] != STATE_NO_CONNECTOR)
            & (
                (self.i16["status_after"][row] == STATUS_BRAKE_TIMEOUT)
                | (self.i16["status_after"][row] == STATUS_CONNECT_TIMEOUT)
            )
        ).to(torch.int32)
        self.xy["command_xy"][row] = self.ctrl.velocity_command[:, :2]

        if self.route_enabled:
            manager = self.task._target_route_manager
            anchor = manager.recovery_anchor.detach()
            self.xy["anchor_xy"][row] = anchor
            distance = (anchor - self._before_position).norm(dim=1)
            active = self.i16["state_after"][row] == STATE_CONNECT
            self.f32["anchor_distance_m"][row] = torch.where(
                active, distance, torch.full_like(distance, float("nan"))
            )
            new_connector = active & (self.i16["state_before"][row] != STATE_CONNECT)
            if bool(new_connector.any()):
                connector_started = time.perf_counter()
                from aerial_gym.task.navrl_task.target_route_planner import _segment_aabb_distance

                bars = self.task.obs_dict["obstacle_position"][
                    :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
                ]
                raw_half = self.task.obs_dict["asset_collision_half_extents"][
                    :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
                ]
                _, _, _, _, hard_lo, hard_hi, hard_half = self.task._route_recovery_geometry(
                    self._before_position, bars, raw_half, self.task._target_route_support_xy
                )
                ids = new_connector.nonzero(as_tuple=False).squeeze(-1)
                selected = [
                    value[ids].detach().cpu().numpy()
                    for value in (
                        self._before_position, anchor, bars, hard_half, hard_lo, hard_hi,
                    )
                ]
                starts, anchors, bar_rows, half_rows, lows, highs = selected
                clearances = []
                for start, endpoint, centers, extents, low, high in zip(
                    starts, anchors, bar_rows, half_rows, lows, highs
                ):
                    obstacle = min(
                        (_segment_aabb_distance(start, endpoint, center - extent, center + extent)
                         for center, extent in zip(centers, extents)),
                        default=float("inf"),
                    )
                    boundary = min(
                        float(np.minimum(start - low, high - start).min()),
                        float(np.minimum(endpoint - low, high - endpoint).min()),
                    )
                    clearances.append(min(obstacle, boundary))
                self._connector_clearance_cache[ids] = torch.as_tensor(
                    clearances, dtype=torch.float32, device=self.device
                )
                self._connector_cpu_s += time.perf_counter() - connector_started
            self.f32["connector_clearance_m"][row] = torch.where(
                active, self._connector_clearance_cache,
                torch.full_like(self._connector_clearance_cache, float("nan")),
            )
            lo = self.task.obs_dict["env_bounds_min"][:, :2]
            resolution = float(self.task.tm.route_resolution_m)
            cell = torch.floor((anchor - lo) / resolution).to(torch.int16)
            self.i16["anchor_cell_i"][row] = torch.where(
                active, cell[:, 0], torch.full_like(cell[:, 0], -1)
            )
            self.i16["anchor_cell_j"][row] = torch.where(
                active, cell[:, 1], torch.full_like(cell[:, 1], -1)
            )
            # The local candidate set is frozen: 24 headings x 3 speed scales plus stop.
            self.i16["candidate_count"][row] = torch.where(
                active, torch.full_like(cell[:, 0], 73), torch.full_like(cell[:, 0], -1)
            )
            horizon = int(math.ceil(float(self.task.tm.avoidance_lookahead_s) / self.task.step_dt))
            self.i16["candidate_horizon_steps"][row] = torch.where(
                active, torch.full_like(cell[:, 0], horizon), torch.full_like(cell[:, 0], -1)
            )
            self.i16["candidate_safe_prefix_steps"][row] = -1
            self.i16["candidate_full_horizon_safe"][row] = -1
            self.i16["candidate_selected_index"][row] = -1
            for details in self._exact_calls:
                global_ids = details["row_ids"].to(device=self.device, dtype=torch.long)
                in_range = (global_ids >= 0) & (global_ids < self.envs)
                safe_ids = global_ids.clamp(min=0, max=self.envs - 1)
                counts = torch.bincount(safe_ids, minlength=self.envs)
                binding_error = (
                    ~in_range
                    | (counts[safe_ids] != 1)
                    | details["selected_binding_error"]
                )
                self.i32["candidate_binding_error"][row, safe_ids] = binding_error.to(torch.int32)
                self.i16["candidate_count"][row, safe_ids] = int(details["count"])
                self.i16["candidate_horizon_steps"][row, safe_ids] = int(details["horizon"])
                self.i16["candidate_selected_index"][row, safe_ids] = details["selected_index"].to(torch.int16)
                self.i16["candidate_safe_prefix_steps"][row, safe_ids] = details["safe_prefix"]
                self.i16["candidate_full_horizon_safe"][row, safe_ids] = (
                    details["full_horizon_safe"].to(torch.int16)
                )
                initial_distance = (anchor[safe_ids] - details["old_xy"]).norm(dim=1)
                self.f32["planned_first_progress_m"][row, safe_ids] = (
                    initial_distance
                    - (anchor[safe_ids] - details["selected_first_xy"]).norm(dim=1)
                )
                self.f32["planned_horizon_progress_m"][row, safe_ids] = (
                    initial_distance
                    - (anchor[safe_ids] - details["selected_final_xy"]).norm(dim=1)
                )

        speed = self.ctrl.linvel[:, :2].norm(dim=1)
        decel = float(getattr(self.task.tm, "recovery_brake_decel_p05", 0.0))
        formula = speed.square() / (2.0 * decel) if math.isfinite(decel) and decel > 0 else torch.full_like(speed, float("nan"))
        measured = torch.full_like(speed, self.measured_stop_distance_p95_m)
        self.f32["formula_stop_distance_m"][row] = formula
        self.f32["stop_distance_m"][row] = measured
        self.f32["stop_margin_m"][row] = self.f32["hard_margin_before_m"][row] - measured

    def record_substep(self):
        if self.interval < 0 or self.substep >= self.physics_substeps:
            raise RuntimeError("unexpected physics callback count")
        i, j = self.interval, self.substep
        pos = self.ctrl.position[:, :2]
        margin, _, _, _ = self._geometry(pos)
        bars = self.task.obs_dict["obstacle_position"][
            :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
        ]
        raw_half = self.task.obs_dict["asset_collision_half_extents"][
            :, self.task._bar_offset:self.task._bar_offset + self.task.n_bars_active, :2
        ]
        lo = (
            self.task.obs_dict["env_bounds_min"][:, :2] + float(self.task.cur.wall_margin)
            + self.support_xy + self.hard_reserve_m
        )
        hi = (
            self.task.obs_dict["env_bounds_max"][:, :2] - float(self.task.cur.wall_margin)
            - self.support_xy - self.hard_reserve_m
        )
        hard_half = raw_half + self.support_xy.unsqueeze(1) + self.hard_reserve_m
        valid = _geometry_valid(pos, bars, hard_half, lo, hi)
        geometry_invalid = getattr(self.ctrl, "watchdog_geometry_invalid", None)
        if geometry_invalid is not None and self.route_enabled:
            valid &= ~geometry_invalid
        self.sub_f32["hard_margin_m"][i, j] = margin
        self.sub_f32["support_xy_m"][i, j] = self.support_xy[:, 0]
        self.sub_f32["contact_force_n"][i, j] = self.ctrl.contact_force.norm(dim=1)
        self.sub_f32["velocity_error_mps"][i, j] = self.ctrl.last_velocity_error.norm(dim=1)
        self.sub_f32["tilt_deg"][i, j] = torch.rad2deg(self.ctrl.last_tilt_rad)
        orientation = self.ctrl.orientation
        obb_valid = torch.isfinite(self.ctrl.position).all(dim=1)
        obb_valid &= torch.isfinite(orientation).all(dim=1)
        obb_valid &= orientation.norm(dim=1) > 1e-6
        self.sub_i8["geometry_valid"][i, j] = valid.to(torch.int8)
        self.sub_i8["obb_valid"][i, j] = obb_valid.to(torch.int8)
        self.sub_i8["motor_saturated"][i, j] = self.ctrl.last_saturated.to(torch.int8)
        self.sub_i8["watchdog_breach"][i, j] = self.ctrl.watchdog_breach.to(torch.int8)
        self.substep += 1

    def finish_interval(self):
        if self.substep != self.physics_substeps:
            raise RuntimeError(
                "physics callback count drift: %d != %d" % (self.substep, self.physics_substeps)
            )
        i = self.interval
        pos = self.ctrl.position[:, :2].detach()
        vel = self.ctrl.linvel[:, :2].detach()
        hard, soft, hard_reason, soft_reason = self._geometry(pos)
        self.f32["hard_margin_after_m"][i] = hard
        self.f32["soft_margin_after_m"][i] = soft
        self.i16["hard_reason_after"][i] = hard_reason
        self.i16["soft_reason_after"][i] = soft_reason
        self.f32["speed_after_mps"][i] = vel.norm(dim=1)
        self.xy["position_after_xy"][i] = pos
        self.xy["velocity_after_xy"][i] = vel
        if self.route_enabled:
            anchor = self.task._target_route_manager.recovery_anchor.detach()
            active = self.i16["state_after"][i] == STATE_CONNECT
            distance = (anchor - pos).norm(dim=1)
            self.f32["anchor_distance_after_m"][i] = torch.where(
                active, distance, torch.full_like(distance, float("nan"))
            )

    def mark_runner_reset(self, env_ids):
        if self.interval < 0:
            raise RuntimeError("runner reset before interval")
        self.i32["runner_reset_after_interval"][self.interval, env_ids] += 1
        self._connector_clearance_cache[env_ids] = float("nan")

    def write(self, path: Path, metadata: Mapping[str, object]) -> Dict[str, object]:
        if self.interval + 1 != self.steps:
            raise RuntimeError("telemetry ended before fixed step count")
        arrays: Dict[str, np.ndarray] = {}
        for collection in (self.i16, self.i32, self.f32, self.xy, self.sub_f32, self.sub_i8):
            for name, tensor in collection.items():
                arrays[name] = tensor.detach().cpu().contiguous().numpy()
        meta = dict(metadata)
        candidate_ms = 1000.0 * self._candidate_cpu_s
        if self._candidate_event_pairs:
            torch.cuda.synchronize(self.device)
            candidate_ms += sum(start.elapsed_time(end) for start, end in self._candidate_event_pairs)
        meta.update({
            "schema": SCHEMA,
            "steps": self.steps,
            "envs": self.envs,
            "physics_substeps": self.physics_substeps,
            "interval_denominator": self.steps * self.envs,
            "substep_denominator": self.steps * self.physics_substeps * self.envs,
            "state_codes": {
                "off": STATE_OFF, "normal": STATE_NORMAL, "brake": STATE_BRAKE,
                "connect": STATE_CONNECT, "route": STATE_ROUTE,
                "no_connector": STATE_NO_CONNECTOR,
            },
            "status_codes": {
                "recovery_no_connector": 19, "recovery_hard_breach": 20,
                "recovery_local_infeasible_soft_free": 21,
                "recovery_brake_timeout": 22, "recovery_connect_timeout": 23,
            },
            "geometry_reason_codes": {
                "wall_x_min": 1, "wall_x_max": 2, "wall_y_min": 3, "wall_y_max": 4,
                "bar_index": "1000 + zero_based_active_bar_index",
            },
            "missing_int16": -1,
            "missing_float": "NaN",
            "measured_stop_distance_p95_m": self.measured_stop_distance_p95_m,
            "candidate_certificate_calls": self._certificate_calls,
            "candidate_observer_device_ms": candidate_ms,
            "connector_observer_cpu_ms": 1000.0 * self._connector_cpu_s,
        })
        arrays["metadata_json_u8"] = np.frombuffer(
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
        _atomic_npz(path, arrays)
        return {"path": str(path), "sha256": sha256_file(path), "metadata": meta}


def load_and_verify(path: Path, expected_sha256: str = "") -> Dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("telemetry artifact missing: %s" % path)
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError("telemetry artifact SHA256 mismatch")
    with np.load(str(path), allow_pickle=False) as payload:
        required = {
            "state_before", "state_after", "age_before", "age_after", "brake_age_before",
            "brake_age_after", "connect_age_before", "connect_age_after",
            "connect_timeout_steps", "status_after",
            "hard_reason_before", "soft_reason_before", "hard_reason_after", "soft_reason_after",
            "hard_margin_before_m", "soft_margin_before_m", "hard_margin_after_m",
            "soft_margin_after_m", "position_before_xy", "position_after_xy", "command_xy",
            "stop_distance_m", "formula_stop_distance_m", "stop_margin_m",
            "planned_first_progress_m", "planned_horizon_progress_m",
            "direct_position_write", "reset_call_during_advance", "runner_reset_after_interval",
            "candidate_count", "candidate_horizon_steps", "candidate_safe_prefix_steps",
            "candidate_selected_index", "candidate_full_horizon_safe", "timeout_event",
            "entry_delta", "resume_delta",
            "no_connector_delta", "hard_breach_delta", "brake_timeout_delta",
            "connect_timeout_delta", "connector_clearance_m",
            "candidate_binding_error", "anchor_xy", "anchor_before_xy", "anchor_distance_m",
            "anchor_distance_after_m", "anchor_cell_i", "anchor_cell_j",
            "hard_margin_m", "contact_force_n", "geometry_valid", "obb_valid",
            "motor_saturated", "watchdog_breach", "metadata_json_u8",
        }
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError("telemetry fields missing: %s" % sorted(missing))
        metadata = json.loads(bytes(payload["metadata_json_u8"].tolist()).decode("utf-8"))
        if metadata.get("schema") != SCHEMA:
            raise RuntimeError("wrong packed telemetry schema")
        steps, envs, substeps = (
            int(metadata["steps"]), int(metadata["envs"]), int(metadata["physics_substeps"])
        )
        if payload["state_before"].shape != (steps, envs):
            raise RuntimeError("interval telemetry shape mismatch")
        if payload["hard_margin_m"].shape != (steps, substeps, envs):
            raise RuntimeError("substep telemetry shape mismatch")
        if int(metadata["interval_denominator"]) != steps * envs:
            raise RuntimeError("interval denominator drift")
        if int(metadata["substep_denominator"]) != steps * substeps * envs:
            raise RuntimeError("substep denominator drift")

        before = payload["state_before"].astype(np.int16)
        after = payload["state_after"].astype(np.int16)
        allowed = (
            (before == STATE_OFF) & (after == STATE_OFF)
            | np.isin(before, [STATE_NORMAL, STATE_ROUTE])
              & np.isin(after, [STATE_NORMAL, STATE_BRAKE, STATE_CONNECT, STATE_ROUTE, STATE_NO_CONNECTOR])
            # A stopped actor can obtain its anchor, certify the route handoff and resume in one
            # task interval.  The boundary samples then compress BRAKE->CONNECT->ROUTE to
            # BRAKE->ROUTE; the exact resume-counter identity below proves the hidden transition.
            | (before == STATE_BRAKE) & np.isin(
                after, [STATE_BRAKE, STATE_CONNECT, STATE_ROUTE, STATE_NO_CONNECTOR]
              )
            | (before == STATE_CONNECT) & np.isin(after, [STATE_CONNECT, STATE_ROUTE, STATE_NO_CONNECTOR])
            | (before == STATE_NO_CONNECTOR) & (after == STATE_NO_CONNECTOR)
        )
        runner_reset = payload["runner_reset_after_interval"] > 0
        # A terminal runner reset occurs after the recorded state; the next interval legitimately
        # starts NORMAL and is therefore excluded from cross-interval transition accounting.
        if not bool(allowed.all()):
            raise RuntimeError("illegal within-interval recovery transition")
        if steps > 1:
            cross_allowed = runner_reset[:-1] | (after[:-1] == before[1:])
            if not bool(cross_allowed.all()):
                raise RuntimeError("illegal cross-interval recovery transition")
        direct_writes = int(payload["direct_position_write"].sum())
        reset_calls = int(payload["reset_call_during_advance"].sum())
        if direct_writes != 0 or reset_calls != 0:
            raise RuntimeError("recovery path wrote target position or invoked reset")
        connect_intervals = 0
        connect_actual_regressions = 0
        connect_actual_max_increase_m = 0.0
        active = after == STATE_CONNECT
        if active.any():
            if (payload["candidate_count"][active] != 73).any():
                raise RuntimeError("CONNECT candidate count drift")
            selected = payload["candidate_selected_index"][active]
            if (selected < 0).any() or (selected >= payload["candidate_count"][active]).any():
                raise RuntimeError("CONNECT selected candidate index is missing/out of range")
            if (payload["candidate_horizon_steps"][active] <= 0).any():
                raise RuntimeError("CONNECT horizon missing")
            if (payload["candidate_safe_prefix_steps"][active] < 0).any():
                raise RuntimeError("CONNECT safe-prefix hook missing")
            if (payload["candidate_full_horizon_safe"][active] != 1).any():
                raise RuntimeError("CONNECT selected command lacks full-horizon certificate")
            if (payload["candidate_safe_prefix_steps"][active]
                    != payload["candidate_horizon_steps"][active]).any():
                raise RuntimeError("CONNECT selected command safe-prefix is incomplete")
            if not np.isfinite(payload["connector_clearance_m"][active]).all():
                raise RuntimeError("CONNECT connector clearance missing/nonfinite")
            if (payload["connector_clearance_m"][active] <= 0.0).any():
                raise RuntimeError("CONNECT connector is not hard-safe")
            if (payload["connect_age_after"][active] < 0).any():
                raise RuntimeError("CONNECT age missing")
            if (payload["connect_timeout_steps"][active] <= 0).any():
                raise RuntimeError("CONNECT timeout budget missing")
            for field in ("anchor_distance_m", "anchor_xy"):
                if not np.isfinite(payload[field][active]).all():
                    raise RuntimeError("CONNECT %s is missing/nonfinite" % field)
            if (payload["anchor_cell_i"][active] < 0).any() or (payload["anchor_cell_j"][active] < 0).any():
                raise RuntimeError("CONNECT anchor cell is missing")
            if (payload["candidate_binding_error"][active] != 0).any():
                raise RuntimeError("CONNECT candidate/environment binding is ambiguous")
            # The first kinematic sample is descriptive: a physical actor can initially coast
            # away while its velocity reverses.  The certified 1 s horizon must still progress.
            # Actual PhysX interval regression is a scientific cell failure, not an integrity
            # VOID: the artifact remains interpretable, and the preregistered 32-cell gate must
            # finalize a FAIL rather than discard it.
            if not np.isfinite(payload["planned_first_progress_m"][active]).all():
                raise RuntimeError("CONNECT first-sample progress is missing/nonfinite")
            if (not np.isfinite(payload["planned_horizon_progress_m"][active]).all()
                    or (payload["planned_horizon_progress_m"][active] < -1e-6).any()):
                raise RuntimeError("CONNECT certificate has negative fixed-anchor progress: horizon")
            if not np.isfinite(payload["anchor_distance_after_m"][active]).all():
                raise RuntimeError("CONNECT post-physics anchor distance is missing")
            actual_increase = (
                payload["anchor_distance_after_m"][active] - payload["anchor_distance_m"][active]
            )
            connect_intervals = int(active.sum())
            connect_actual_regressions = int((actual_increase > 1e-5).sum())
            connect_actual_max_increase_m = float(np.max(actual_increase))
            continuing = active & (before == STATE_CONNECT)
            if continuing.any() and not np.allclose(
                payload["anchor_before_xy"][continuing], payload["anchor_xy"][continuing],
                rtol=0.0, atol=1e-7,
            ):
                raise RuntimeError("CONNECT anchor changed within an interval")
            if steps > 1:
                carried = active[1:] & active[:-1] & ~runner_reset[:-1]
                if carried.any() and not np.allclose(
                    payload["anchor_xy"][1:][carried], payload["anchor_xy"][:-1][carried],
                    rtol=0.0, atol=1e-7,
                ):
                    raise RuntimeError("CONNECT anchor changed across intervals")
            speed = float(metadata.get("speed_mps", float("nan")))
            expected_connect_timeout = int(math.ceil((
                math.sqrt(2.0) * 3.5 * 0.25 / max(speed, 0.10)
                + max(speed, 0.10) / 4.0
                + math.pi / math.radians(150.0) + 0.20
            ) / 0.1))
            if (payload["connect_timeout_steps"][active] != expected_connect_timeout).any():
                raise RuntimeError("CONNECT timeout budget differs from frozen derivation")
        brake = after == STATE_BRAKE
        if brake.any() and (payload["brake_age_after"][brake] < 0).any():
            raise RuntimeError("BRAKE age is missing")
        brake_timeout = int(metadata.get("brake_timeout_steps", 0))
        if brake.any() and (brake_timeout <= 0 or (payload["brake_age_after"][brake] > brake_timeout).any()):
            raise RuntimeError("BRAKE timeout budget/age drift")
        recovery_active = np.isin(after, [STATE_BRAKE, STATE_CONNECT])
        if recovery_active.any() and (payload["age_after"][recovery_active] <= 0).any():
            raise RuntimeError("recovery age is missing/nonpositive")
        for field in ("hard_reason_before", "soft_reason_before", "hard_reason_after", "soft_reason_after"):
            if (payload[field] <= 0).any():
                raise RuntimeError("hard/soft geometry reason is missing: %s" % field)
        recovery_arm = metadata.get("route_mode") == "global_astar_recovery_v2"
        if recovery_arm:
            registered_stop = float(metadata.get("measured_stop_distance_p95_m", float("nan")))
            if not math.isfinite(registered_stop) or registered_stop < 0.0:
                raise RuntimeError("measured p95 stop distance is missing")
            if not np.allclose(payload["stop_distance_m"], registered_stop, rtol=0.0, atol=1e-7):
                raise RuntimeError("per-cell measured p95 stop distance drift")
        if not np.isfinite(payload["hard_margin_m"]).all():
            raise RuntimeError("substep hard margin is nonfinite")
        if not (payload["geometry_valid"] == 1).all():
            raise RuntimeError("geometry validity gate failed")
        if not (payload["obb_valid"] == 1).all():
            raise RuntimeError("OBB validity gate failed")
        no_connector = after == STATE_NO_CONNECTOR
        nonzero_no_connector = np.linalg.norm(payload["command_xy"], axis=2) > 1e-7
        if bool((no_connector & nonzero_no_connector).any()):
            raise RuntimeError("NO_CONNECTOR emitted nonzero planar command")
        new_no_connector = (after == STATE_NO_CONNECTOR) & (before != STATE_NO_CONNECTOR)
        scalar_no_connector = int(payload["no_connector_delta"][:, 0].sum())
        if int(new_no_connector.sum()) != scalar_no_connector:
            raise RuntimeError("no-connector event/aggregate partition mismatch")
        reason_codes = payload["status_after"][new_no_connector]
        allowed_reasons = np.asarray([19, 20, 21, 22, 23], dtype=np.int16)
        if reason_codes.size and not np.isin(reason_codes, allowed_reasons).all():
            raise RuntimeError("unknown no-connector reason")
        expected_timeout = new_no_connector & np.isin(
            payload["status_after"], [STATUS_BRAKE_TIMEOUT, STATUS_CONNECT_TIMEOUT]
        )
        if not np.array_equal(payload["timeout_event"] > 0, expected_timeout):
            raise RuntimeError("timeout event is not the unique NO_CONNECTOR transition")
        brake_timeout_events = int((reason_codes == STATUS_BRAKE_TIMEOUT).sum())
        connect_timeout_events = int((reason_codes == STATUS_CONNECT_TIMEOUT).sum())
        if int(payload["brake_timeout_delta"][:, 0].sum()) != brake_timeout_events:
            raise RuntimeError("BRAKE timeout status/counter partition mismatch")
        if int(payload["connect_timeout_delta"][:, 0].sum()) != connect_timeout_events:
            raise RuntimeError("CONNECT timeout status/counter partition mismatch")
        entries = int(payload["entry_delta"][:, 0].sum())
        resumes = int(payload["resume_delta"][:, 0].sum())
        observed_resumes = int(
            ((after == STATE_ROUTE) & np.isin(before, [STATE_BRAKE, STATE_CONNECT])).sum()
        )
        if resumes != observed_resumes:
            raise RuntimeError(
                "route-resume transition/counter mismatch: transitions=%d resumes=%d"
                % (observed_resumes, resumes)
            )
        active_recovery = np.isin(after, [STATE_BRAKE, STATE_CONNECT])
        reset_active = int((runner_reset & active_recovery).sum())
        current_open = int((active_recovery[-1] & ~runner_reset[-1]).sum())
        open_entries = entries - resumes - scalar_no_connector - reset_active
        if open_entries != current_open:
            raise RuntimeError(
                "recovery entry outcome partition mismatch: entries=%d resumes=%d "
                "no_connector=%d reset_active=%d open=%d expected_open=%d"
                % (entries, resumes, scalar_no_connector, reset_active, open_entries, current_open)
            )
        return {
            "schema": SCHEMA,
            "sha256": actual,
            "steps": steps,
            "envs": envs,
            "physics_substeps": substeps,
            "interval_denominator": steps * envs,
            "substep_denominator": steps * substeps * envs,
            "recovery_entries": entries,
            "route_resumes": resumes,
            "no_connector_events": scalar_no_connector,
            "runner_reset_during_recovery": reset_active,
            "open_recoveries_at_cell_end": open_entries,
            "no_connector_reason_counts": {
                str(code): int((reason_codes == code).sum()) for code in allowed_reasons.tolist()
            },
            "hard_breach_events": int(payload["hard_breach_delta"][:, 0].sum()),
            "timeout_env_intervals": int(payload["timeout_event"].sum()),
            "watchdog_breach_substeps": int(payload["watchdog_breach"].sum()),
            "contact_substeps": int((payload["contact_force_n"] > 0.05).sum()),
            "runner_reset_envs": int(runner_reset.sum()),
            "candidate_certificate_calls": int(metadata.get("candidate_certificate_calls", 0)),
            "candidate_observer_device_ms": float(metadata.get("candidate_observer_device_ms", 0.0)),
            "connector_observer_cpu_ms": float(metadata.get("connector_observer_cpu_ms", 0.0)),
            "direct_position_writes": direct_writes,
            "reset_calls_during_advance": reset_calls,
            "connect_intervals": connect_intervals,
            "connect_actual_regressions": connect_actual_regressions,
            "connect_actual_max_increase_m": connect_actual_max_increase_m,
        }
