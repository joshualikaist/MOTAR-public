"""Evaluation-only joint telemetry for NavRL speed-allocation diagnosis.

The recorder deliberately lives outside the policy/reward path.  It attributes decision-time
kinematics to the episode's eventual outcome, and keeps the final 1 s before a bar contact.  The
heading quantities are finite-difference *proxies*, not planned-trajectory curvature and not
causal estimates.
"""

from __future__ import annotations

import math

import torch


OUTCOME_LABELS = ("capture", "crash", "timeout")
RISK_LABELS = ("negative", "low_0_0p5", "medium_0p5_1p5", "high_ge_1p5")
RISK_EDGES_M = (0.0, 0.5, 1.5)
METRIC_LABELS = (
    "actual_speed_mps",
    "requested_command_speed_mps",
    "executed_command_speed_mps",
    "actual_direction_min_clearance_m",
    "requested_direction_min_clearance_m",
    "executed_direction_min_clearance_m",
    "actual_stopping_distance_m",
    "requested_stopping_distance_m",
    "executed_stopping_distance_m",
    "actual_stopping_margin_m",
    "requested_stopping_margin_m",
    "executed_stopping_margin_m",
    "policy_action_delta_xy_norm",
    "requested_heading_rate_proxy_radps",
    "realized_heading_rate_proxy_radps",
    "requested_curvature_proxy_radpm",
    "realized_curvature_proxy_radpm",
)


def _wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def risk_bin_index(actual_stopping_margin_m: torch.Tensor) -> torch.Tensor:
    """Map margin to fixed, preregistered bins; lower margin means higher physical risk."""

    return torch.bucketize(
        actual_stopping_margin_m,
        torch.tensor(
            RISK_EDGES_M,
            device=actual_stopping_margin_m.device,
            dtype=actual_stopping_margin_m.dtype,
        ),
        right=True,
    )


class JointSpeedTelemetry:
    """GPU-friendly per-step accumulator with CPU-testable export semantics."""

    schema_version = 1

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        step_dt: float,
        brake_mps2: float,
        reaction_s: float,
        hard_margin_m: float,
        precontact_window_s: float = 1.0,
        heading_min_speed_mps: float = 0.25,
    ):
        if num_envs <= 0 or step_dt <= 0 or brake_mps2 <= 0:
            raise ValueError("invalid joint telemetry dimensions or braking contract")
        self.num_envs = int(num_envs)
        self.device = device
        self.step_dt = float(step_dt)
        self.brake_mps2 = float(brake_mps2)
        self.reaction_s = float(reaction_s)
        self.hard_margin_m = float(hard_margin_m)
        self.heading_min_speed_mps = float(heading_min_speed_mps)
        self.window_steps = max(1, int(round(float(precontact_window_s) / self.step_dt)))
        self.metric_count = len(METRIC_LABELS)
        self.risk_count = len(RISK_LABELS)

        shape = (self.num_envs, self.risk_count, self.metric_count)
        self._episode_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self._episode_count = torch.zeros(shape, dtype=torch.long, device=device)
        self._outcome_sum = torch.zeros(
            (3, self.risk_count, self.metric_count), dtype=torch.float64, device=device
        )
        self._outcome_count = torch.zeros(
            (3, self.risk_count, self.metric_count), dtype=torch.long, device=device
        )
        self._outcome_episodes = torch.zeros(3, dtype=torch.long, device=device)

        self._prev_actual_xy = torch.zeros((self.num_envs, 2), device=device)
        self._prev_requested_xy = torch.zeros((self.num_envs, 2), device=device)
        self._prev_policy_xy = torch.zeros((self.num_envs, 2), device=device)
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self._ring_values = torch.zeros(
            (self.num_envs, self.window_steps, self.metric_count),
            dtype=torch.float64,
            device=device,
        )
        self._ring_valid = torch.zeros(
            (self.num_envs, self.window_steps, self.metric_count),
            dtype=torch.bool,
            device=device,
        )
        self._ring_risk = torch.zeros(
            (self.num_envs, self.window_steps), dtype=torch.long, device=device
        )
        self._ring_position = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._ring_filled = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._contact_sum = torch.zeros(self.metric_count, dtype=torch.float64, device=device)
        self._contact_count = torch.zeros(self.metric_count, dtype=torch.long, device=device)
        self._contact_risk_steps = torch.zeros(self.risk_count, dtype=torch.long, device=device)
        self._contact_episodes = 0

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._episode_sum[env_ids] = 0
        self._episode_count[env_ids] = 0
        self._prev_actual_xy[env_ids] = 0
        self._prev_requested_xy[env_ids] = 0
        self._prev_policy_xy[env_ids] = 0
        self._prev_valid[env_ids] = False
        self._ring_values[env_ids] = 0
        self._ring_valid[env_ids] = False
        self._ring_risk[env_ids] = 0
        self._ring_position[env_ids] = 0
        self._ring_filled[env_ids] = 0

    def record_step(
        self,
        *,
        actual_velocity_xy: torch.Tensor,
        requested_command_xy: torch.Tensor,
        executed_command_xy: torch.Tensor,
        policy_action_xy: torch.Tensor,
        actual_direction_clearance_m: torch.Tensor,
        requested_direction_clearance_m: torch.Tensor,
        executed_direction_clearance_m: torch.Tensor,
    ) -> None:
        """Record the state at action selection, before physics advances by one control step."""

        vectors = (actual_velocity_xy, requested_command_xy, executed_command_xy, policy_action_xy)
        if any(v.shape != (self.num_envs, 2) for v in vectors):
            raise ValueError("joint telemetry vectors must be [num_envs, 2]")
        clearances = (
            actual_direction_clearance_m,
            requested_direction_clearance_m,
            executed_direction_clearance_m,
        )
        if any(value.shape != (self.num_envs,) for value in clearances):
            raise ValueError("joint telemetry clearances must be [num_envs]")
        finite = torch.isfinite(torch.cat(vectors, dim=1)).all(dim=1)
        finite &= torch.stack(clearances, dim=1).isfinite().all(dim=1)
        if not bool(finite.all()):
            raise RuntimeError("non-finite joint speed telemetry input")

        actual_speed = actual_velocity_xy.norm(dim=1)
        requested_speed = requested_command_xy.norm(dim=1)
        executed_speed = executed_command_xy.norm(dim=1)
        actual_usable = (actual_direction_clearance_m - self.hard_margin_m).clamp(min=0.0)
        requested_usable = (
            requested_direction_clearance_m - self.hard_margin_m
        ).clamp(min=0.0)
        executed_usable = (
            executed_direction_clearance_m - self.hard_margin_m
        ).clamp(min=0.0)

        def stopping_distance(speed: torch.Tensor) -> torch.Tensor:
            return speed * self.reaction_s + speed.square() / (2.0 * self.brake_mps2)

        actual_stop = stopping_distance(actual_speed)
        requested_stop = stopping_distance(requested_speed)
        executed_stop = stopping_distance(executed_speed)
        actual_margin = actual_usable - actual_stop
        requested_margin = requested_usable - requested_stop
        executed_margin = executed_usable - executed_stop

        prev = self._prev_valid
        actual_turn_valid = (
            prev
            & (actual_speed >= self.heading_min_speed_mps)
            & (self._prev_actual_xy.norm(dim=1) >= self.heading_min_speed_mps)
        )
        requested_turn_valid = (
            prev
            & (requested_speed >= self.heading_min_speed_mps)
            & (self._prev_requested_xy.norm(dim=1) >= self.heading_min_speed_mps)
        )
        actual_heading = torch.atan2(actual_velocity_xy[:, 1], actual_velocity_xy[:, 0])
        prev_actual_heading = torch.atan2(self._prev_actual_xy[:, 1], self._prev_actual_xy[:, 0])
        requested_heading = torch.atan2(requested_command_xy[:, 1], requested_command_xy[:, 0])
        prev_requested_heading = torch.atan2(
            self._prev_requested_xy[:, 1], self._prev_requested_xy[:, 0]
        )
        actual_rate = _wrap_angle(actual_heading - prev_actual_heading).abs() / self.step_dt
        requested_rate = (
            _wrap_angle(requested_heading - prev_requested_heading).abs() / self.step_dt
        )
        action_delta = (policy_action_xy - self._prev_policy_xy).norm(dim=1)

        values = torch.stack(
            (
                actual_speed,
                requested_speed,
                executed_speed,
                actual_direction_clearance_m,
                requested_direction_clearance_m,
                executed_direction_clearance_m,
                actual_stop,
                requested_stop,
                executed_stop,
                actual_margin,
                requested_margin,
                executed_margin,
                action_delta,
                requested_rate,
                actual_rate,
                requested_rate / requested_speed.clamp(min=self.heading_min_speed_mps),
                actual_rate / actual_speed.clamp(min=self.heading_min_speed_mps),
            ),
            dim=1,
        ).to(torch.float64)
        valid = torch.ones_like(values, dtype=torch.bool)
        valid[:, 12] = prev
        valid[:, 13] = requested_turn_valid
        valid[:, 14] = actual_turn_valid
        valid[:, 15] = requested_turn_valid
        valid[:, 16] = actual_turn_valid

        risk = risk_bin_index(actual_margin)
        env = torch.arange(self.num_envs, device=self.device)
        self._episode_sum[env, risk] += torch.where(valid, values, torch.zeros_like(values))
        self._episode_count[env, risk] += valid.to(torch.long)

        slot = self._ring_position
        self._ring_values[env, slot] = values
        self._ring_valid[env, slot] = valid
        self._ring_risk[env, slot] = risk
        self._ring_position = (slot + 1) % self.window_steps
        self._ring_filled.clamp_max_(self.window_steps - 1).add_(1)

        self._prev_actual_xy.copy_(actual_velocity_xy)
        self._prev_requested_xy.copy_(requested_command_xy)
        self._prev_policy_xy.copy_(policy_action_xy)
        self._prev_valid.fill_(True)

    def finish(
        self,
        finished: torch.Tensor,
        successes: torch.Tensor,
        crashes: torch.Tensor,
        timeouts: torch.Tensor,
        crash_cause_code: torch.Tensor,
    ) -> None:
        idx = finished.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        captured = successes[idx].bool()
        crashed = crashes[idx] > 0
        timed_out = timeouts[idx].bool()
        exclusive = captured.long() + crashed.long() + timed_out.long()
        if not bool((exclusive == 1).all()):
            raise RuntimeError("joint telemetry received non-exclusive outcomes")
        outcome = torch.where(
            captured,
            torch.zeros_like(idx),
            torch.where(crashed, torch.ones_like(idx), torch.full_like(idx, 2)),
        )
        self._outcome_episodes += torch.bincount(outcome, minlength=3)
        flat_out = outcome[:, None] * self.risk_count + torch.arange(
            self.risk_count, device=self.device
        )[None, :]
        self._outcome_sum.view(-1, self.metric_count).index_add_(
            0, flat_out.reshape(-1), self._episode_sum[idx].reshape(-1, self.metric_count)
        )
        self._outcome_count.view(-1, self.metric_count).index_add_(
            0, flat_out.reshape(-1), self._episode_count[idx].reshape(-1, self.metric_count)
        )

        contact = crash_cause_code[idx] == 0
        for env_id in idx[contact].tolist():
            filled = int(self._ring_filled[env_id].item())
            if filled <= 0:
                raise RuntimeError("bar contact has no preceding joint telemetry")
            valid = self._ring_valid[env_id, :filled]
            values = self._ring_values[env_id, :filled]
            self._contact_sum += torch.where(valid, values, torch.zeros_like(values)).sum(dim=0)
            self._contact_count += valid.sum(dim=0)
            self._contact_risk_steps += torch.bincount(
                self._ring_risk[env_id, :filled], minlength=self.risk_count
            )
            self._contact_episodes += 1

    @staticmethod
    def _cell_payload(sums: torch.Tensor, counts: torch.Tensor) -> dict:
        result = {"step_samples": int(counts[0].item())}
        for i, label in enumerate(METRIC_LABELS):
            n = int(counts[i].item())
            result[label] = float(sums[i].item() / n) if n else None
            result[label + "_samples"] = n
        return result

    def payload(self, expected_outcomes, *, expected_bar_contacts: int) -> dict:
        expected = tuple(int(v) for v in expected_outcomes)
        observed = tuple(int(v) for v in self._outcome_episodes.tolist())
        if observed != expected:
            raise RuntimeError(
                "NavRL joint telemetry outcome mismatch: %s != %s" % (observed, expected)
            )
        if int(self._contact_episodes) != int(expected_bar_contacts):
            raise RuntimeError(
                "NavRL joint telemetry bar-contact mismatch: %d != %d"
                % (int(self._contact_episodes), int(expected_bar_contacts))
            )

        outcome_payload = {}
        for out_i, out_label in enumerate(OUTCOME_LABELS):
            cells = {}
            for risk_i, risk_label in enumerate(RISK_LABELS):
                cells[risk_label] = self._cell_payload(
                    self._outcome_sum[out_i, risk_i], self._outcome_count[out_i, risk_i]
                )
            outcome_payload[out_label] = {
                "episodes": observed[out_i],
                "risk_bins": cells,
                "negative_margin_step_rate": (
                    int(self._outcome_count[out_i, 0, 0].item())
                    / max(1, int(self._outcome_count[out_i, :, 0].sum().item()))
                ),
            }

        contact = self._cell_payload(self._contact_sum, self._contact_count)
        contact["episodes"] = int(self._contact_episodes)
        total_contact_steps = int(self._contact_risk_steps.sum().item())
        contact["risk_step_counts"] = {
            label: int(self._contact_risk_steps[i].item())
            for i, label in enumerate(RISK_LABELS)
        }
        contact["negative_margin_step_rate"] = (
            int(self._contact_risk_steps[0].item()) / total_contact_steps
            if total_contact_steps
            else None
        )
        return {
            "schema_version": self.schema_version,
            "evaluation_only": True,
            "risk_variable": "decision_time_actual_direction_stopping_margin_m",
            "risk_bin_edges_m": list(RISK_EDGES_M),
            "risk_bin_labels": list(RISK_LABELS),
            "step_dt_s": self.step_dt,
            "precontact_window_s": self.window_steps * self.step_dt,
            "braking_contract": {
                "brake_mps2": self.brake_mps2,
                "reaction_s": self.reaction_s,
                "hard_margin_m": self.hard_margin_m,
                "formula": "same_direction_usable_clearance - (v*reaction + v^2/(2*brake))",
            },
            "proxy_notice": (
                "heading-rate and curvature are finite-difference velocity/command proxies; "
                "outcome strata are descriptive associations, not causal effects"
            ),
            "outcomes": outcome_payload,
            "bar_contact_preceding": contact,
        }


def assess_preregistered_speed_gate(joint: dict) -> dict:
    """Apply fixed descriptive gates without turning association into a causal claim."""

    if int(joint.get("schema_version", -1)) != 1:
        raise ValueError("unsupported joint telemetry schema")
    outcomes = joint.get("outcomes") or {}
    capture = outcomes.get("capture") or {}
    contact = joint.get("bar_contact_preceding") or {}
    contact_episodes = int(contact.get("episodes", 0))
    contact_steps = int(contact.get("step_samples", 0))
    capture_steps = sum(
        int((cell or {}).get("step_samples", 0))
        for cell in (capture.get("risk_bins") or {}).values()
    )
    quality_pass = contact_episodes >= 100 and contact_steps >= 500 and capture_steps >= 1000
    contact_negative = contact.get("negative_margin_step_rate")
    capture_negative = capture.get("negative_margin_step_rate")
    delta = (
        float(contact_negative) - float(capture_negative)
        if contact_negative is not None and capture_negative is not None
        else None
    )
    association_pass = bool(
        quality_pass
        and float(contact_negative) >= 0.50
        and delta is not None
        and delta >= 0.10
    )
    verdict = (
        "supports_descriptive_speed_risk_association"
        if association_pass
        else ("does_not_meet_preregistered_association_gate" if quality_pass else "insufficient_quality")
    )
    return {
        "schema_version": 1,
        "quality": {
            "bar_contact_episodes_min": 100,
            "bar_contact_preceding_steps_min": 500,
            "capture_steps_min": 1000,
            "passed": quality_pass,
        },
        "association_gate": {
            "contact_negative_margin_rate_min": 0.50,
            "contact_minus_capture_negative_margin_rate_min_pp": 10.0,
            "contact_negative_margin_rate": contact_negative,
            "capture_negative_margin_rate": capture_negative,
            "delta_pp": None if delta is None else 100.0 * delta,
            "passed": association_pass,
        },
        "verdict": verdict,
        "causal_claim_allowed": False,
        "interpretation": (
            "PASS only supports an association between unsafe speed margin and bar contact. "
            "It does not identify speed as the cause or justify governor tuning."
        ),
    }
