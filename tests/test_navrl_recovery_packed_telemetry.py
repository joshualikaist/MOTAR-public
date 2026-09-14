"""CPU-only adversarial tests for the recovery-v2 packed telemetry artifact."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_packed_test_module", ROOT / "tools/navrl_recovery_packed_telemetry.py"
)
PACKED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKED
SPEC.loader.exec_module(PACKED)


def fixture(steps=3, substeps=2, envs=2, connect=True):
    shape = (steps, envs)
    subshape = (steps, substeps, envs)
    state_before = np.zeros(shape, dtype=np.int16)
    state_after = np.zeros(shape, dtype=np.int16)
    candidate = np.full(shape, -1, dtype=np.int16)
    horizon = np.full(shape, -1, dtype=np.int16)
    prefix = np.full(shape, -1, dtype=np.int16)
    full = np.full(shape, -1, dtype=np.int16)
    if connect:
        # One NORMAL->BRAKE entry remains open in CONNECT at cell end.
        state_after[0, 0] = PACKED.STATE_BRAKE
        state_before[1, 0] = PACKED.STATE_BRAKE
        state_after[1, 0] = PACKED.STATE_CONNECT
        state_before[2, 0] = PACKED.STATE_CONNECT
        state_after[2, 0] = PACKED.STATE_CONNECT
        candidate[2, 0] = 73
        horizon[2, 0] = 10
        prefix[2, 0] = 10
        full[2, 0] = 1
        candidate[1, 0] = 73
        horizon[1, 0] = 10
        prefix[1, 0] = 10
        full[1, 0] = 1
    metadata = {
        "schema": PACKED.SCHEMA,
        "route_mode": "global_astar_recovery_v2",
        "steps": steps,
        "envs": envs,
        "physics_substeps": substeps,
        "interval_denominator": steps * envs,
        "substep_denominator": steps * substeps * envs,
        "measured_stop_distance_p95_m": 0.2,
        "speed_mps": 0.6,
        "brake_timeout_steps": 20,
    }
    arrays = {
        "state_before": state_before,
        "state_after": state_after,
        "age_before": np.zeros(shape, dtype=np.int16),
        "age_after": np.zeros(shape, dtype=np.int16),
        "brake_age_before": np.zeros(shape, dtype=np.int16),
        "brake_age_after": np.zeros(shape, dtype=np.int16),
        "connect_age_before": np.zeros(shape, dtype=np.int16),
        "connect_age_after": np.zeros(shape, dtype=np.int16),
        "connect_timeout_steps": np.ones(shape, dtype=np.int16) * 37,
        "status_after": np.zeros(shape, dtype=np.int16),
        "hard_reason_before": np.ones(shape, dtype=np.int16),
        "soft_reason_before": np.ones(shape, dtype=np.int16),
        "hard_reason_after": np.ones(shape, dtype=np.int16),
        "soft_reason_after": np.ones(shape, dtype=np.int16),
        "hard_margin_before_m": np.ones(shape, dtype=np.float32),
        "soft_margin_before_m": np.ones(shape, dtype=np.float32),
        "hard_margin_after_m": np.ones(shape, dtype=np.float32),
        "soft_margin_after_m": np.ones(shape, dtype=np.float32),
        "connector_clearance_m": np.ones(shape, dtype=np.float32),
        "anchor_distance_m": np.ones(shape, dtype=np.float32),
        "anchor_distance_after_m": np.ones(shape, dtype=np.float32),
        "anchor_cell_i": np.ones(shape, dtype=np.int16),
        "anchor_cell_j": np.ones(shape, dtype=np.int16),
        "stop_distance_m": np.ones(shape, dtype=np.float32) * 0.2,
        "formula_stop_distance_m": np.ones(shape, dtype=np.float32) * 0.3,
        "stop_margin_m": np.ones(shape, dtype=np.float32) * 0.8,
        "planned_first_progress_m": np.zeros(shape, dtype=np.float32),
        "planned_horizon_progress_m": np.zeros(shape, dtype=np.float32),
        "position_before_xy": np.zeros(shape + (2,), dtype=np.float32),
        "position_after_xy": np.zeros(shape + (2,), dtype=np.float32),
        "velocity_before_xy": np.zeros(shape + (2,), dtype=np.float32),
        "velocity_after_xy": np.zeros(shape + (2,), dtype=np.float32),
        "command_xy": np.zeros(shape + (2,), dtype=np.float32),
        "anchor_xy": np.zeros(shape + (2,), dtype=np.float32),
        "anchor_before_xy": np.zeros(shape + (2,), dtype=np.float32),
        "direct_position_write": np.zeros(shape, dtype=np.int32),
        "reset_call_during_advance": np.zeros(shape, dtype=np.int32),
        "runner_reset_after_interval": np.zeros(shape, dtype=np.int32),
        "entry_delta": np.zeros(shape, dtype=np.int32),
        "resume_delta": np.zeros(shape, dtype=np.int32),
        "no_connector_delta": np.zeros(shape, dtype=np.int32),
        "hard_breach_delta": np.zeros(shape, dtype=np.int32),
        "brake_timeout_delta": np.zeros(shape, dtype=np.int32),
        "connect_timeout_delta": np.zeros(shape, dtype=np.int32),
        "candidate_binding_error": np.zeros(shape, dtype=np.int32),
        "timeout_event": np.zeros(shape, dtype=np.int32),
        "candidate_count": candidate,
        "candidate_horizon_steps": horizon,
        "candidate_selected_index": np.where(candidate > 0, 0, -1).astype(np.int16),
        "candidate_safe_prefix_steps": prefix,
        "candidate_full_horizon_safe": full,
        "hard_margin_m": np.ones(subshape, dtype=np.float32),
        "support_xy_m": np.ones(subshape, dtype=np.float32),
        "contact_force_n": np.zeros(subshape, dtype=np.float32),
        "velocity_error_mps": np.zeros(subshape, dtype=np.float32),
        "tilt_deg": np.zeros(subshape, dtype=np.float32),
        "geometry_valid": np.ones(subshape, dtype=np.int8),
        "obb_valid": np.ones(subshape, dtype=np.int8),
        "motor_saturated": np.zeros(subshape, dtype=np.int8),
        "watchdog_breach": np.zeros(subshape, dtype=np.int8),
        "metadata_json_u8": np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
        ),
    }
    if connect:
        arrays["entry_delta"][0, 0] = 1
        arrays["age_after"][0:, 0] = np.asarray([1, 2, 3], dtype=np.int16)
    return arrays


class PackedTelemetryTest(unittest.TestCase):
    def write(self, arrays):
        directory = Path(tempfile.mkdtemp(prefix="recovery-packed-test-"))
        path = directory / "cell.npz"
        PACKED._atomic_npz(path, arrays)
        return path

    def test_valid_fixture_round_trips_with_fixed_denominators(self):
        path = self.write(fixture())
        result = PACKED.load_and_verify(path)
        self.assertEqual(result["interval_denominator"], 6)
        self.assertEqual(result["substep_denominator"], 12)

    def test_atomic_writer_refuses_overwrite(self):
        arrays = fixture()
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            PACKED._atomic_npz(path, arrays)

    def test_segment_certificate_uses_passed_reachable_tube_once(self):
        p0 = torch.tensor([[[-1.0, 0.61]]])
        p1 = torch.tensor([[[1.0, 0.61]]])
        bars = torch.tensor([[[0.0, 0.0]]])
        half = torch.tensor([[[0.5, 0.5]]])
        self.assertFalse(bool(PACKED._segments_hit_closed_aabb(p0, p1, bars, half, 0.10).item()))
        self.assertTrue(bool(PACKED._segments_hit_closed_aabb(p0, p1, bars, half, 0.12).item()))

    def test_connect_without_safe_prefix_is_rejected(self):
        arrays = fixture()
        arrays["candidate_safe_prefix_steps"][1, 0] = -1
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "safe-prefix"):
            PACKED.load_and_verify(path)

    def test_connect_negative_planned_anchor_progress_is_rejected(self):
        arrays = fixture()
        arrays["planned_horizon_progress_m"][1, 0] = -0.01
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "negative fixed-anchor progress"):
            PACKED.load_and_verify(path)

    def test_initial_coast_is_descriptive_when_horizon_and_actual_interval_progress(self):
        arrays = fixture()
        arrays["planned_first_progress_m"][1, 0] = -0.04
        arrays["planned_horizon_progress_m"][1, 0] = 0.01
        arrays["anchor_distance_m"][1, 0] = 0.30
        arrays["anchor_distance_after_m"][1, 0] = 0.29
        result = PACKED.load_and_verify(self.write(arrays))
        self.assertEqual(result["schema"], PACKED.SCHEMA)
        self.assertEqual(result["connect_actual_regressions"], 0)

    def test_actual_interval_regression_is_counted_not_an_integrity_refuse(self):
        arrays = fixture()
        arrays["planned_first_progress_m"][1, 0] = -0.04
        arrays["planned_horizon_progress_m"][1, 0] = 0.01
        arrays["anchor_distance_m"][1, 0] = 0.30
        arrays["anchor_distance_after_m"][1, 0] = 0.35
        result = PACKED.load_and_verify(self.write(arrays))
        self.assertGreaterEqual(result["connect_intervals"], 1)
        self.assertEqual(result["connect_actual_regressions"], 1)
        self.assertGreater(result["connect_actual_max_increase_m"], 0.04)

    def test_compressed_brake_connect_route_requires_exact_resume_counter(self):
        arrays = fixture(connect=False)
        arrays["state_after"][0, 0] = PACKED.STATE_BRAKE
        arrays["state_before"][1, 0] = PACKED.STATE_BRAKE
        arrays["state_after"][1, 0] = PACKED.STATE_ROUTE
        arrays["state_before"][2, 0] = PACKED.STATE_ROUTE
        arrays["state_after"][2, 0] = PACKED.STATE_ROUTE
        arrays["entry_delta"][0, 0] = 1
        arrays["age_after"][0, 0] = 1
        arrays["brake_age_after"][0, 0] = 1
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "route-resume transition"):
            PACKED.load_and_verify(path)
        arrays["resume_delta"][1, 0] = 1
        path = self.write(arrays)
        result = PACKED.load_and_verify(path)
        self.assertEqual(result["route_resumes"], 1)

    def test_brake_to_normal_transition_remains_illegal(self):
        arrays = fixture(connect=False)
        arrays["state_before"][1, 0] = PACKED.STATE_BRAKE
        arrays["state_after"][1, 0] = PACKED.STATE_NORMAL
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "illegal"):
            PACKED.load_and_verify(path)

    def test_position_write_and_reset_are_independently_rejected(self):
        for field in ("direct_position_write", "reset_call_during_advance"):
            arrays = fixture()
            arrays[field][0, 0] = 1
            path = self.write(arrays)
            with self.assertRaisesRegex(RuntimeError, "wrote target position or invoked reset"):
                PACKED.load_and_verify(path)

    def test_no_connector_requires_zero_command(self):
        arrays = fixture(connect=False)
        arrays["state_after"][0, 0] = PACKED.STATE_CONNECT
        arrays["candidate_count"][0, 0] = 73
        arrays["candidate_horizon_steps"][0, 0] = 10
        arrays["candidate_selected_index"][0, 0] = 0
        arrays["candidate_safe_prefix_steps"][0, 0] = 10
        arrays["candidate_full_horizon_safe"][0, 0] = 1
        arrays["age_after"][0, 0] = 1
        arrays["state_before"][1, 0] = PACKED.STATE_CONNECT
        arrays["state_after"][1, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["state_before"][2, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["state_after"][2, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["status_after"][1:, 0] = 19
        arrays["no_connector_delta"][1, 0] = 1
        arrays["command_xy"][1, 0, 0] = 0.01
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            PACKED.load_and_verify(path)

    def test_nonfinite_substep_margin_is_rejected(self):
        arrays = fixture()
        arrays["hard_margin_m"][0, 0, 0] = np.nan
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            PACKED.load_and_verify(path)

    def test_sha_mismatch_is_rejected(self):
        path = self.write(fixture())
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            PACKED.load_and_verify(path, "0" * 64)

    def test_latched_timeout_is_not_recounted(self):
        arrays = fixture(connect=False)
        arrays["state_after"][0, 0] = PACKED.STATE_CONNECT
        arrays["candidate_count"][0, 0] = 73
        arrays["candidate_horizon_steps"][0, 0] = 10
        arrays["candidate_selected_index"][0, 0] = 0
        arrays["candidate_safe_prefix_steps"][0, 0] = 10
        arrays["candidate_full_horizon_safe"][0, 0] = 1
        arrays["age_after"][0, 0] = 1
        arrays["state_before"][1, 0] = PACKED.STATE_CONNECT
        arrays["state_after"][1, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["state_before"][2, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["state_after"][2, 0] = PACKED.STATE_NO_CONNECTOR
        arrays["status_after"][1:, 0] = PACKED.STATUS_TIMEOUT
        arrays["timeout_event"][1, 0] = 1
        arrays["brake_timeout_delta"][1, 0] = 1
        arrays["entry_delta"][0, 0] = 1
        arrays["no_connector_delta"][1, 0] = 1
        path = self.write(arrays)
        result = PACKED.load_and_verify(path)
        self.assertEqual(result["timeout_env_intervals"], 1)

    def test_cross_interval_state_discontinuity_is_rejected(self):
        arrays = fixture(connect=False)
        arrays["state_after"][0, 0] = PACKED.STATE_BRAKE
        path = self.write(arrays)
        with self.assertRaisesRegex(RuntimeError, "cross-interval"):
            PACKED.load_and_verify(path)


if __name__ == "__main__":
    unittest.main()
