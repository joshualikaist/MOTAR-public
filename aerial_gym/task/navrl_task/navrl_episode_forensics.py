"""Evaluation-only episode forensics: acquisition (TD-T1) and terminal approach (TD-T2).

This recorder lives outside the policy, reward, perception, safety-filter and termination paths. It
reads detached tensors, draws no randomness, writes into no task buffer, and its labels are never an
input to anything. If any label here ever reaches an observation, that is a new observation lineage
and needs its own preregistration.

Preregistrations, fixed before this file existed:
  docs/preregistration_task_diagnostics_t1_2026-09-14.md
  docs/preregistration_task_diagnostics_t2_2026-09-14.md

Every threshold below is quoted from those documents. They are labelling and attribution rules, not
gates, and each record also stores the raw quantities so another rule can be applied afterwards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

# --- TD-T1 ---------------------------------------------------------------------------------------
LABELS = ("SEARCH", "ACQUIRED", "TRACKING", "LOST_SHORT", "LOST_LONG", "REACQUIRED", "TERMINATED")
LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
LOSS_RUN_SHORT_STEPS = 10          # 1.0 s at the 0.1 s control interval; one tenth of tracker memory
EPISODE_LABEL_NEVER_SEEN = "NEVER_SEEN"

# --- TD-T2 ---------------------------------------------------------------------------------------
TERMINAL_WINDOW_STEPS = 20         # 2.0 s
NEAR_RANGE_M = 1.0
FAST_RADIAL_SPEED_MPS = 1.5
SLOW_CLOSING_SPEED_MPS = 0.3
GOVERNOR_SHARE_MIN = 0.5
COMMAND_ERROR_MIN_MPS = 0.5
GOVERNOR_INTERVENTION_EPS_MPS = 1e-3
ACTION_SATURATION_EPS = 1e-3
RANGE_INCREASE_EPS_M = 1e-3
FAILURE_CLASSES = ("NEVER_ACQUIRED", "OBSTACLE_CONTACT", "LOST_TRACK", "TERMINAL_OVERSHOOT",
                   "FLY_BY", "FILTER_LIMITED", "CONTROL_TRACKING_ERROR", "SLOW_APPROACH",
                   "TIMEOUT_WITH_TRACK", "UNKNOWN")
OUTCOMES = ("capture", "crash", "timeout")
# The live crash diagnostics encode the cause as 0=contact, 1=below, 2=above, 3=oob, -1=none. That
# numbering is one of three in this repository; it is mapped here rather than compared as an integer.
CRASH_CAUSE_NAMES = {0: "contact", 1: "below", 2: "above", 3: "oob"}


def yaw_from_xyzw(quaternion):
    """World yaw of an xyzw quaternion batch, written out so no convention is assumed elsewhere."""
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def step_label(visible, ever_acquired, was_visible, loss_run_steps, terminated):
    """The TD-T1 per-step label. Pure: this is the whole state machine, and it is testable alone.

    `loss_run_steps` is the length of the current not-visible run **including this step**.
    """
    if terminated:
        return "TERMINATED"
    if not ever_acquired:
        return "ACQUIRED" if visible else "SEARCH"
    if visible:
        return "REACQUIRED" if not was_visible else "TRACKING"
    return "LOST_SHORT" if loss_run_steps <= LOSS_RUN_SHORT_STEPS else "LOST_LONG"


def classify_terminal_failure(record):
    """The TD-T2 taxonomy, in its preregistered priority order. First match wins.

    Pure, and deliberately readable as the table it came from: an attribution rule, not a model. A
    record that matches nothing is UNKNOWN, which is a result rather than a gap to be filled.
    """
    if record.get("legacy_capture_0p5m"):
        return "CAPTURED"
    if not record.get("ever_acquired"):
        return "NEVER_ACQUIRED"
    if record.get("outcome") == "crash" and record.get("crash_cause") in ("contact", "target"):
        return "OBSTACLE_CONTACT"
    if (not record.get("visible_at_end")) and (record.get("final_loss_run_steps") or 0) > LOSS_RUN_SHORT_STEPS:
        return "LOST_TRACK"
    minimum = record.get("minimum_center_distance_m")
    radial = record.get("radial_speed_at_closest_approach_mps")
    tangential = record.get("tangential_speed_at_closest_approach_mps")
    if minimum is not None and minimum <= NEAR_RANGE_M:
        final_range = record.get("final_range_m")
        increased = (final_range is not None
                     and final_range > minimum + RANGE_INCREASE_EPS_M)
        if radial is not None and abs(radial) >= FAST_RADIAL_SPEED_MPS and increased:
            return "TERMINAL_OVERSHOOT"
        if (tangential is not None and radial is not None
                and tangential >= abs(radial)):
            return "FLY_BY"
    governor_share = record.get("governor_intervened_share_final_window")
    if governor_share is not None and governor_share >= GOVERNOR_SHARE_MIN:
        return "FILTER_LIMITED"
    command_error = record.get("mean_abs_command_error_final_window_mps")
    if (command_error is not None and command_error >= COMMAND_ERROR_MIN_MPS
            and (governor_share or 0.0) <= 0.0):
        return "CONTROL_TRACKING_ERROR"
    if record.get("outcome") == "timeout" and record.get("visible_at_end"):
        closing = record.get("mean_closing_speed_final_window_mps")
        final_range = record.get("final_range_m")
        if (closing is not None and closing < SLOW_CLOSING_SPEED_MPS
                and final_range is not None and final_range > record.get("success_radius_m", 0.5)):
            return "SLOW_APPROACH"
        return "TIMEOUT_WITH_TRACK"
    return "UNKNOWN"


def _finite(value):
    """JSON cannot hold a NaN, and a NaN silently read as a number is worse than a null."""
    number = float(value)
    return number if math.isfinite(number) else None


class EpisodeForensics:
    """Per-episode acquisition and terminal records for one vectorized task."""

    def __init__(self, num_envs, device, *, step_dt, success_radius_m, target_radius_m=None,
                 density_bars=None, checkpoint_sha256=None, seed=None):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = device
        self.step_dt = float(step_dt)
        self.success_radius_m = float(success_radius_m)
        self.target_radius_m = None if target_radius_m is None else float(target_radius_m)
        self.context = {"density_bars": density_bars, "checkpoint_sha256": checkpoint_sha256,
                        "seed": seed, "step_dt_s": self.step_dt,
                        "success_radius_m": self.success_radius_m,
                        "target_radius_m": self.target_radius_m}
        self.records = []
        self._episode_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        long_zeros = lambda: torch.zeros(num_envs, dtype=torch.long, device=device)
        bool_zeros = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)
        float_zeros = lambda: torch.zeros(num_envs, dtype=torch.float64, device=device)
        self._obs_steps = long_zeros()
        self._visible_steps = long_zeros()
        self._ever_acquired = bool_zeros()
        self._first_acq = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._first_cam_acq = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._was_visible = bool_zeros()
        self._loss_run = long_zeros()
        self._invisible_run = long_zeros()
        self._longest_invisible = long_zeros()
        self._loss_count = long_zeros()
        self._total_lost = long_zeros()
        self._longest_lost = long_zeros()
        self._reacq_count = long_zeros()
        self._first_reacq_latency = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._sum_reacq_latency = long_zeros()
        self._label_counts = torch.zeros((num_envs, len(LABELS)), dtype=torch.long, device=device)
        self._track_age_at_acq = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._last_track_age = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._dist_at_acq = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._dist_at_first_loss = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._dist_at_first_reacq = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._min_center = torch.full((num_envs,), float("inf"), dtype=torch.float64, device=device)
        self._final_range = torch.full((num_envs,), float("nan"), dtype=torch.float64, device=device)
        self._inside_1m = long_zeros()
        self._inside_half = long_zeros()
        self._legacy_capture = bool_zeros()
        # Closest-approach snapshot, written only when the swept minimum improves.
        self._snapshot = {name: torch.full((num_envs,), float("nan"), dtype=torch.float64,
                                           device=device)
                          for name in ("radial_mps", "tangential_mps", "bearing_rad", "yaw_rad",
                                       "track_age_s", "range_sigma_m", "clearance_m",
                                       "commanded_mps", "actual_mps", "governor_scale")}
        self._snapshot_visible = bool_zeros()
        # Final-window ring buffers (TD-T2 window = 20 observation steps).
        self._ring_position = long_zeros()
        window = TERMINAL_WINDOW_STEPS
        self._ring = {name: torch.zeros((num_envs, window), dtype=torch.float64, device=device)
                      for name in ("closing_mps", "command_error_mps", "governor", "visible",
                                   "range_m", "clearance_m")}
        self._perception = {"visible": bool_zeros(), "camera_visible": bool_zeros(),
                            "track_age_s": float_zeros(), "range_sigma_m": float_zeros()}
        self._swept_distance = torch.full((num_envs,), float("nan"), dtype=torch.float64,
                                          device=device)

    # -- inputs ------------------------------------------------------------------------------
    def observe_perception(self, diagnostics):
        """Cache the evaluator-facing perception diagnostics for this step. Read-only."""
        visible = diagnostics["visible"].detach().to(torch.bool)
        camera = diagnostics.get("camera_visible", visible).detach().to(torch.bool)
        self._perception["visible"] = visible.clone()
        self._perception["camera_visible"] = camera.clone()
        age = diagnostics.get("track_age")
        self._perception["track_age_s"] = (
            torch.zeros_like(self._perception["track_age_s"]) if age is None
            else age.detach().to(torch.float64).clone())
        covariance = diagnostics.get("track_covariance")
        if covariance is None:
            sigma = torch.zeros_like(self._perception["range_sigma_m"])
        else:
            position_trace = covariance.detach().to(torch.float64)[:, 0:3, 0:3].diagonal(
                dim1=-2, dim2=-1).sum(dim=-1)
            sigma = position_trace.clamp_min(0.0).sqrt()
        self._perception["range_sigma_m"] = sigma

    def observe_capture(self, swept_distance_m, captured=None):
        """Cache the swept centre distance the capture test itself used, so the two cannot drift."""
        self._swept_distance = swept_distance_m.detach().to(torch.float64).clone()
        if captured is not None:
            self._legacy_capture |= captured.detach().to(torch.bool)

    def record_step(self, *, valid, robot_position, robot_velocity, robot_orientation_xyzw,
                    target_position, target_velocity, requested_speed_mps, executed_speed_mps,
                    governor_scale, action_xy, clearance_m):
        """One observation step. Everything here is detached, and nothing is written back."""
        valid = valid.detach().to(torch.bool)
        if not bool(valid.any()):
            return
        position = robot_position.detach().to(torch.float64)
        velocity = robot_velocity.detach().to(torch.float64)
        target = target_position.detach().to(torch.float64)
        target_speed = target_velocity.detach().to(torch.float64)
        offset = target - position
        distance = offset.norm(dim=1)
        direction = offset / distance.clamp_min(1e-9).unsqueeze(1)
        relative = velocity - target_speed
        radial = (relative * direction).sum(dim=1)              # >0 means closing
        tangential = (relative - radial.unsqueeze(1) * direction).norm(dim=1)
        bearing = torch.atan2(offset[:, 1], offset[:, 0])
        yaw = yaw_from_xyzw(robot_orientation_xyzw.detach().to(torch.float64))
        yaw_error = torch.atan2(torch.sin(bearing - yaw), torch.cos(bearing - yaw))
        swept = torch.where(torch.isfinite(self._swept_distance), self._swept_distance, distance)
        visible = self._perception["visible"] & valid
        camera_visible = self._perception["camera_visible"] & valid
        requested = requested_speed_mps.detach().to(torch.float64)
        executed = executed_speed_mps.detach().to(torch.float64)
        actual = velocity[:, 0:2].norm(dim=1)
        governor = (requested - executed) > GOVERNOR_INTERVENTION_EPS_MPS
        saturated = (action_xy.detach().abs().to(torch.float64) >= 1.0 - ACTION_SATURATION_EPS).any(dim=1)
        clearance = clearance_m.detach().to(torch.float64)

        one = torch.ones_like(self._obs_steps)
        self._obs_steps = torch.where(valid, self._obs_steps + one, self._obs_steps)
        self._visible_steps = torch.where(visible, self._visible_steps + one, self._visible_steps)
        fresh = visible & ~self._ever_acquired
        self._first_acq = torch.where(fresh, self._obs_steps, self._first_acq)
        self._dist_at_acq = torch.where(fresh, distance, self._dist_at_acq)
        self._track_age_at_acq = torch.where(fresh, self._perception["track_age_s"],
                                             self._track_age_at_acq)
        fresh_cam = camera_visible & (self._first_cam_acq < 0)
        self._first_cam_acq = torch.where(fresh_cam, self._obs_steps, self._first_cam_acq)
        hidden = valid & ~visible
        self._invisible_run = torch.where(hidden, self._invisible_run + one,
                                          torch.where(valid, torch.zeros_like(self._invisible_run),
                                                      self._invisible_run))
        self._longest_invisible = torch.maximum(self._longest_invisible, self._invisible_run)
        lost_now = hidden & self._ever_acquired
        starts_loss = lost_now & self._was_visible
        self._loss_count = torch.where(starts_loss, self._loss_count + one, self._loss_count)
        self._dist_at_first_loss = torch.where(starts_loss & (self._loss_count == 1), distance,
                                               self._dist_at_first_loss)
        self._loss_run = torch.where(lost_now, self._loss_run + one, self._loss_run)
        self._total_lost = torch.where(lost_now, self._total_lost + one, self._total_lost)
        self._longest_lost = torch.maximum(self._longest_lost, self._loss_run)
        reacquired = visible & self._ever_acquired & ~self._was_visible & (self._loss_run > 0)
        self._reacq_count = torch.where(reacquired, self._reacq_count + one, self._reacq_count)
        self._sum_reacq_latency = torch.where(reacquired, self._sum_reacq_latency + self._loss_run,
                                              self._sum_reacq_latency)
        first_reacq = reacquired & (self._first_reacq_latency < 0)
        self._first_reacq_latency = torch.where(first_reacq, self._loss_run,
                                                self._first_reacq_latency)
        self._dist_at_first_reacq = torch.where(first_reacq, distance, self._dist_at_first_reacq)
        self._loss_run = torch.where(visible & valid, torch.zeros_like(self._loss_run),
                                     self._loss_run)

        labels = self._label_indices(visible, hidden, valid, reacquired, fresh)
        self._label_counts.scatter_add_(
            1, labels.unsqueeze(1),
            valid.to(torch.long).unsqueeze(1) * torch.ones_like(labels).unsqueeze(1))
        self._ever_acquired |= fresh
        self._was_visible = torch.where(valid, visible, self._was_visible)
        self._last_track_age = torch.where(valid, self._perception["track_age_s"],
                                           self._last_track_age)

        improved = valid & (swept < self._min_center)
        self._min_center = torch.where(improved, swept, self._min_center)
        for name, value in (("radial_mps", radial), ("tangential_mps", tangential),
                            ("bearing_rad", bearing), ("yaw_rad", yaw_error),
                            ("track_age_s", self._perception["track_age_s"]),
                            ("range_sigma_m", self._perception["range_sigma_m"]),
                            ("clearance_m", clearance), ("commanded_mps", executed),
                            ("actual_mps", actual), ("governor_scale",
                                                     governor_scale.detach().to(torch.float64))):
            self._snapshot[name] = torch.where(improved, value, self._snapshot[name])
        self._snapshot_visible = torch.where(improved, visible, self._snapshot_visible)
        self._final_range = torch.where(valid, distance, self._final_range)
        self._inside_1m = torch.where(valid & (distance <= NEAR_RANGE_M),
                                      self._inside_1m + one, self._inside_1m)
        self._inside_half = torch.where(valid & (distance <= self.success_radius_m),
                                        self._inside_half + one, self._inside_half)

        slot = (self._ring_position % TERMINAL_WINDOW_STEPS).unsqueeze(1)
        valid_f = valid.to(torch.float64).unsqueeze(1)
        for name, value in (("closing_mps", radial), ("command_error_mps", (executed - actual).abs()),
                            ("governor", governor.to(torch.float64)),
                            ("visible", visible.to(torch.float64)), ("range_m", distance),
                            ("clearance_m", clearance)):
            current = self._ring[name].gather(1, slot)
            self._ring[name].scatter_(1, slot, torch.where(valid_f > 0, value.unsqueeze(1), current))
        self._ring_position = torch.where(valid, self._ring_position + one, self._ring_position)

    def _label_indices(self, visible, hidden, valid, reacquired, fresh):
        labels = torch.full_like(self._obs_steps, LABEL_INDEX["SEARCH"])
        labels = torch.where(fresh, torch.full_like(labels, LABEL_INDEX["ACQUIRED"]), labels)
        tracking = visible & self._ever_acquired & ~reacquired
        labels = torch.where(tracking, torch.full_like(labels, LABEL_INDEX["TRACKING"]), labels)
        labels = torch.where(reacquired, torch.full_like(labels, LABEL_INDEX["REACQUIRED"]), labels)
        lost = hidden & self._ever_acquired
        short = lost & (self._loss_run <= LOSS_RUN_SHORT_STEPS)
        long_loss = lost & (self._loss_run > LOSS_RUN_SHORT_STEPS)
        labels = torch.where(short, torch.full_like(labels, LABEL_INDEX["LOST_SHORT"]), labels)
        labels = torch.where(long_loss, torch.full_like(labels, LABEL_INDEX["LOST_LONG"]), labels)
        return torch.where(valid, labels, torch.zeros_like(labels))

    # -- episode boundary --------------------------------------------------------------------
    def finish(self, finished, successes, crashes, timeouts, crash_cause=None):
        """Emit one record per finished episode. Called after the task has resolved outcomes."""
        finished = finished.detach().to(torch.bool)
        if not bool(finished.any()):
            return
        outcome = torch.where(successes.detach().to(torch.bool), 0,
                              torch.where(crashes.detach().to(torch.bool) if crashes.dtype == torch.bool
                                          else crashes.detach() > 0, 1, 2))
        self._legacy_capture |= successes.detach().to(torch.bool)
        causes = None if crash_cause is None else crash_cause.detach().cpu().tolist()
        indices = torch.nonzero(finished, as_tuple=False).flatten().tolist()
        window = TERMINAL_WINDOW_STEPS
        for index in indices:
            filled = int(min(int(self._obs_steps[index].item()), window))
            ring = {}
            if filled:
                order = [(int(self._ring_position[index].item()) - offset - 1) % window
                         for offset in range(filled)]
                for name, buffer in self._ring.items():
                    ring[name] = [float(buffer[index, slot].item()) for slot in order]
            cause = None if causes is None else CRASH_CAUSE_NAMES.get(int(causes[index]))
            self.records.append(self._record(index, int(outcome[index].item()), ring, cause))
        self._episode_index = torch.where(finished, self._episode_index + 1, self._episode_index)

    def _record(self, index, outcome_code, ring, crash_cause_code):
        acquired = bool(self._ever_acquired[index].item())
        obs_steps = int(self._obs_steps[index].item())
        visible_steps = int(self._visible_steps[index].item())
        loss_count = int(self._loss_count[index].item())
        reacq_count = int(self._reacq_count[index].item())
        governor_share = (sum(ring["governor"]) / len(ring["governor"])) if ring else None
        command_error = (sum(ring["command_error_mps"]) / len(ring["command_error_mps"])
                         if ring else None)
        closing = (sum(ring["closing_mps"]) / len(ring["closing_mps"])) if ring else None
        minimum = _finite(self._min_center[index].item())
        surface = (None if minimum is None or self.target_radius_m is None
                   else minimum - self.target_radius_m)
        radial = _finite(self._snapshot["radial_mps"][index].item())
        tangential = _finite(self._snapshot["tangential_mps"][index].item())
        record = {
            "env_index": int(index),
            "episode_index": int(self._episode_index[index].item()),
            "outcome": OUTCOMES[outcome_code],
            "crash_cause": crash_cause_code,
            "observation_steps": obs_steps,
            "visible_steps": visible_steps,
            "visibility_fraction": (visible_steps / obs_steps) if obs_steps else None,
            "ever_acquired": acquired,
            "first_acquisition_step": (int(self._first_acq[index].item()) if acquired else None),
            "first_camera_acquisition_step": (int(self._first_cam_acq[index].item())
                                              if int(self._first_cam_acq[index].item()) >= 0
                                              else None),
            "loss_count": loss_count,
            "total_lost_steps": int(self._total_lost[index].item()),
            "longest_lost_steps": int(self._longest_lost[index].item()),
            "longest_invisible_steps": int(self._longest_invisible[index].item()),
            "final_loss_run_steps": int(self._loss_run[index].item()),
            "reacquired": reacq_count > 0,
            "reacquisition_count": reacq_count,
            "first_reacquisition_latency_steps": (int(self._first_reacq_latency[index].item())
                                                  if int(self._first_reacq_latency[index].item()) >= 0
                                                  else None),
            "mean_reacquisition_latency_steps": (
                float(self._sum_reacq_latency[index].item()) / reacq_count if reacq_count else None),
            "track_age_at_first_acquisition_s": (_finite(self._track_age_at_acq[index].item())
                                                 if acquired else None),
            "final_track_age_s": _finite(self._last_track_age[index].item()),
            "distance_at_acquisition_m": (_finite(self._dist_at_acq[index].item())
                                          if acquired else None),
            "distance_at_first_loss_m": _finite(self._dist_at_first_loss[index].item()),
            "distance_at_first_reacquisition_m": _finite(self._dist_at_first_reacq[index].item()),
            "final_range_m": _finite(self._final_range[index].item()),
            "label_step_counts": {name: int(self._label_counts[index, position].item())
                                  for position, name in enumerate(LABELS)},
            "episode_label": (EPISODE_LABEL_NEVER_SEEN if not acquired else "TERMINATED"),
            "visible_at_end": bool(self._was_visible[index].item()),
            # TD-T2 evaluation-only metrics. The historical criterion is recorded unchanged.
            "legacy_capture_0p5m": bool(self._legacy_capture[index].item()),
            "success_radius_m": self.success_radius_m,
            "minimum_center_distance_m": minimum,
            "minimum_surface_distance_m": surface,
            "radial_speed_at_closest_approach_mps": radial,
            "tangential_speed_at_closest_approach_mps": tangential,
            "relative_speed_at_closest_approach_mps": (
                None if radial is None or tangential is None
                else math.hypot(radial, tangential)),
            "bearing_at_closest_approach_rad": _finite(self._snapshot["bearing_rad"][index].item()),
            "yaw_error_at_closest_approach_rad": _finite(self._snapshot["yaw_rad"][index].item()),
            "visible_at_closest_approach": bool(self._snapshot_visible[index].item()),
            "track_age_at_closest_approach_s": _finite(self._snapshot["track_age_s"][index].item()),
            "range_sigma_at_closest_approach_m": _finite(
                self._snapshot["range_sigma_m"][index].item()),
            "clearance_at_closest_approach_m": _finite(self._snapshot["clearance_m"][index].item()),
            "commanded_speed_at_closest_approach_mps": _finite(
                self._snapshot["commanded_mps"][index].item()),
            "actual_speed_at_closest_approach_mps": _finite(
                self._snapshot["actual_mps"][index].item()),
            "governor_scale_at_closest_approach": _finite(
                self._snapshot["governor_scale"][index].item()),
            "time_inside_1m_steps": int(self._inside_1m[index].item()),
            "time_inside_0p5m_steps": int(self._inside_half[index].item()),
            "final_window_steps": len(ring.get("governor", [])),
            "governor_intervened_share_final_window": governor_share,
            "mean_abs_command_error_final_window_mps": command_error,
            "mean_closing_speed_final_window_mps": closing,
            "min_clearance_final_window_m": (min(ring["clearance_m"]) if ring else None),
        }
        record["failure_class"] = classify_terminal_failure(record)
        record.update(self.context)
        return record

    def reset_idx(self, env_ids):
        """Clear per-episode state. Called by the task where it resets its own buffers."""
        if env_ids is None or len(env_ids) == 0:
            return
        for tensor, value in ((self._obs_steps, 0), (self._visible_steps, 0), (self._loss_run, 0),
                              (self._invisible_run, 0), (self._longest_invisible, 0),
                              (self._loss_count, 0), (self._total_lost, 0), (self._longest_lost, 0),
                              (self._reacq_count, 0), (self._sum_reacq_latency, 0),
                              (self._ring_position, 0), (self._inside_1m, 0),
                              (self._inside_half, 0)):
            tensor[env_ids] = value
        for tensor in (self._first_acq, self._first_cam_acq, self._first_reacq_latency):
            tensor[env_ids] = -1
        for tensor in (self._ever_acquired, self._was_visible, self._legacy_capture,
                       self._snapshot_visible):
            tensor[env_ids] = False
        for tensor in (self._track_age_at_acq, self._last_track_age, self._dist_at_acq,
                       self._dist_at_first_loss, self._dist_at_first_reacq, self._final_range,
                       self._swept_distance):
            tensor[env_ids] = float("nan")
        self._min_center[env_ids] = float("inf")
        self._label_counts[env_ids] = 0
        for tensor in self._snapshot.values():
            tensor[env_ids] = float("nan")
        for tensor in self._ring.values():
            tensor[env_ids] = 0.0

    # -- output ------------------------------------------------------------------------------
    def summary(self):
        """Counts only. Every question in the preregistration is answered from the rows."""
        rows = self.records
        acquired = [row for row in rows if row["ever_acquired"]]
        lost = [row for row in acquired if row["loss_count"] > 0]
        classes = {name: 0 for name in ("CAPTURED",) + FAILURE_CLASSES}
        for row in rows:
            classes[row["failure_class"]] = classes.get(row["failure_class"], 0) + 1
        return {
            "episodes": len(rows),
            "never_acquired": len(rows) - len(acquired),
            "acquired": len(acquired),
            "lost_at_least_once": len(lost),
            "reacquired_at_least_once": len([row for row in lost if row["reacquired"]]),
            "outcomes": {name: len([row for row in rows if row["outcome"] == name])
                         for name in OUTCOMES},
            "failure_classes": classes,
            "thresholds": {"loss_run_short_steps": LOSS_RUN_SHORT_STEPS,
                           "terminal_window_steps": TERMINAL_WINDOW_STEPS,
                           "near_range_m": NEAR_RANGE_M,
                           "fast_radial_speed_mps": FAST_RADIAL_SPEED_MPS,
                           "slow_closing_speed_mps": SLOW_CLOSING_SPEED_MPS,
                           "governor_share_min": GOVERNOR_SHARE_MIN,
                           "command_error_min_mps": COMMAND_ERROR_MIN_MPS},
            "scope": "Evaluation-only instrumentation of existing checkpoints; no policy, reward, "
                     "detector, controller or termination change",
            "causality_vs_d8b": "NOT_TESTED",
        }

    def export(self, path):
        """Write rows plus a count summary. Never overwrites an existing file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "navrl_episode_forensics_v1", "summary": self.summary(),
                   "context": self.context, "labels": list(LABELS),
                   "failure_classes": list(FAILURE_CLASSES), "records": self.records}
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return target
