"""CPU contracts for recovery-v2 packed-telemetry forensics."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/diagnose_navrl_physical_target_recovery_v2_packed.py"
SPEC = importlib.util.spec_from_file_location("recovery_v2_packed_forensics", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class PackedForensicsTest(unittest.TestCase):
    def test_generic_no_connector_classes_follow_start_state_and_soft_margin(self):
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_BRAKE, 19, -0.04),
            "brake_no_anchor_likely",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_CONNECT, 19, 0.16),
            "connect_failed_resume_likely",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_CONNECT, 19, -0.05),
            "connect_failed_certificate_likely",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_NORMAL, 19, -0.05),
            "same_interval_brake_no_anchor_likely",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_BRAKE, 22, 0.02),
            "brake_timeout",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_CONNECT, 23, 0.10),
            "connect_timeout",
        )
        self.assertEqual(
            MOD.classify_no_connector_entry(MOD.STATE_NORMAL, 20, 0.40),
            "hard_breach",
        )

    def test_connect_rest_command_can_point_at_anchor_while_realized_speed_lags_envelope(self):
        payload = {
            "state_after": np.array([[MOD.STATE_CONNECT]], dtype=np.int16),
            "command_xy": np.array([[[0.40, 0.0]]], dtype=np.float32),
            "velocity_before_xy": np.array([[[0.02, 0.0]]], dtype=np.float32),
            "velocity_after_xy": np.array([[[0.055, 0.0]]], dtype=np.float32),
            "position_before_xy": np.array([[[0.0, 0.0]]], dtype=np.float32),
            "anchor_xy": np.array([[[1.0, 0.0]]], dtype=np.float32),
            "anchor_distance_m": np.array([[1.0]], dtype=np.float32),
            "anchor_distance_after_m": np.array([[0.99]], dtype=np.float32),
            "candidate_selected_index": np.array([[0]], dtype=np.int16),
            "planned_horizon_progress_m": np.array([[0.24]], dtype=np.float32),
        }
        tracking = MOD.connect_tracking(payload)
        self.assertEqual(tracking["connect_intervals"], 1)
        self.assertEqual(tracking["rest_command_toward_anchor_fraction"], 1.0)
        self.assertLess(tracking["rest_mean_realized_mps"], 0.10)
        self.assertLess(tracking["rest_realized_over_envelope_dv"], 0.20)
        self.assertEqual(tracking["actual_regression_count"], 0)

    def test_diagnose_gate_pools_70bar_mechanism_and_does_not_write_into_receipt(self):
        directory = Path(tempfile.mkdtemp(prefix="recovery-v2-packed-"))
        raw = directory / "raw"
        raw.mkdir()
        shape = (2, 2)
        arrays = {
            "state_before": np.array([[0, 1], [2, 4]], dtype=np.int16),
            "state_after": np.array([[4, 4], [4, 4]], dtype=np.int16),
            "status_after": np.array([[19, 22], [19, 19]], dtype=np.int16),
            "soft_margin_before_m": np.array([[-0.05, 0.02], [0.16, -0.01]], dtype=np.float32),
            "command_xy": np.zeros(shape + (2,), dtype=np.float32),
            "velocity_before_xy": np.zeros(shape + (2,), dtype=np.float32),
            "velocity_after_xy": np.zeros(shape + (2,), dtype=np.float32),
            "position_before_xy": np.zeros(shape + (2,), dtype=np.float32),
            "anchor_xy": np.ones(shape + (2,), dtype=np.float32),
            "anchor_distance_m": np.ones(shape, dtype=np.float32),
            "anchor_distance_after_m": np.ones(shape, dtype=np.float32),
            "candidate_selected_index": np.full(shape, -1, dtype=np.int16),
            "planned_horizon_progress_m": np.zeros(shape, dtype=np.float32),
        }
        np.savez(raw / "route_global_astar_recovery_v2__speed_0p6__bars_70.npz", **arrays)
        summary = {
            "cells": [
                {
                    "record_id": "route_off__speed_0.6__bars_70",
                    "route_mode": "off",
                    "speed_mps": 0.6,
                    "bars": 70,
                    "pass": True,
                    "gates": {"speed": True},
                    "mean_speed_ratio": 0.9,
                    "route": {
                        "plan_success_fraction": None,
                        "fallback_interval_fraction": None,
                        "goal_completions_per_env": None,
                    },
                },
                {
                    "record_id": "route_global_astar_recovery_v2__speed_0.6__bars_70",
                    "route_mode": "global_astar_recovery_v2",
                    "speed_mps": 0.6,
                    "bars": 70,
                    "pass": False,
                    "gates": {"speed": False, "connect_actual_progress": False},
                    "mean_speed_ratio": 0.63,
                    "telemetry": {
                        "path": "route_global_astar_recovery_v2__speed_0p6__bars_70.npz"
                    },
                    "route": {
                        "plan_success_fraction": 0.91,
                        "fallback_interval_fraction": 0.37,
                        "goal_completions_per_env": 0.21875,
                        "counter_delta": {
                            "plan_attempts": 45,
                            "plan_successes": 41,
                            "fallback_intervals": 3537,
                            "goal_completions": 7,
                        },
                    },
                },
            ]
        }
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (directory / "receipt.json").write_text(
            json.dumps({
                "verdict": {
                    "execution_integrity": "PASS_32_CELL_INTEGRITY",
                    "route_mechanism": "FAIL_ROUTE_MECHANISM",
                    "long_training_authorized": False,
                }
            }),
            encoding="utf-8",
        )
        report = MOD.diagnose_gate(directory)
        self.assertEqual(report["cells_pass"], 1)
        self.assertEqual(report["recovery_cells_pass"], 0)
        self.assertEqual(report["mechanism_pool_70bar"]["plan_attempts"], 45)
        self.assertEqual(
            report["pooled_no_connector_classes"]["same_interval_brake_no_anchor_likely"],
            1,
        )
        self.assertEqual(report["pooled_no_connector_classes"]["brake_timeout"], 1)
        self.assertEqual(
            report["pooled_no_connector_classes"]["connect_failed_resume_likely"],
            1,
        )
        self.assertFalse((directory / "receipt.json").read_text(encoding="utf-8") == "")
        self.assertEqual(
            json.loads((directory / "receipt.json").read_text(encoding="utf-8"))["verdict"][
                "route_mechanism"
            ],
            "FAIL_ROUTE_MECHANISM",
        )


if __name__ == "__main__":
    unittest.main()
