"""CPU-only contract tests for the braking-route-v3 verifier and frozen grid."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

# The verifier binds the speed grid at import time.  Canonical-default tests must not inherit a
# leftover GPU-shell export of NAVRL_TARGET_BRAKING_CONTRACT_VARIANT.
os.environ.pop("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", None)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "braking_v3_gate_standalone",
    ROOT / "tools/verify_navrl_braking_route_v3_gate.py",
)
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _identity():
    hashes = {name: "a" * 64 for name in GATE.IDENTITY_HASH_FIELDS}
    hashes.update({
        "runtime_clean": True,
        "physical_geometry_version": "v2",
        "placement_mode": "footprint_clearance",
        "preregistration_sha256": GATE.PREREG_SHA256,
        "robot_name": "navrl_ref5in_v2_quad",
        "target_dynamics": "physical",
        "target_pattern": "waypoint",
    })
    return hashes


def _diagnostics(**overrides):
    payload = {
        "runtime_replan_unsafe_start_count": 0,
        "soft_envelope_exit_count": 0,
        "accepted_terminal_stop_certificate_count": 100,
        "accepted_command_count": 100,
        "plan_attempt_count": 100,
        "plan_success_count": 100,
        "fallback_interval_count": 0,
        "goal_completion_count": 32,
    }
    payload.update(overrides)
    return payload


def _cell(route, speed, bars, seed, identity, **overrides):
    local_name = (
        "off_bounded_local_step_infeasible_fraction"
        if route == "off"
        else "routed_local_step_invalidation_fraction"
    )
    absent = (
        "routed_local_step_invalidation_fraction"
        if route == "off"
        else "off_bounded_local_step_infeasible_fraction"
    )
    row = {
        "record_id": GATE.record_id(route, speed, bars),
        "seed": seed, "envs": GATE.ENVS, "steps": GATE.STEPS,
        "warmup_steps": GATE.WARMUP_STEPS,
        "route_mode": route, "speed_mps": speed, "bars": bars,
        "tracking_rmse_mps": 0.1, "mean_speed_ratio": 0.9,
        "contact_step_fraction": 0.0, "motor_saturation_fraction": 0.0,
        "max_tilt_deg": 1.0, "invalid_state_fraction": 0.0,
        local_name: 0.0, absent: None,
        "identity": identity,
        "initial_layout_sha256": "b" * 64,
        "initial_robot_pose_sha256": "c" * 64,
        "initial_target_pose_sha256": "d" * 64,
        "v3_diagnostics": None if route == "off" else _diagnostics(),
    }
    row.update(overrides)
    row["physical_gates"] = GATE.physical_gate_metrics(row)
    row["physical_pass"] = all(row["physical_gates"].values())
    return row


def _summary(stage, identity=None, cells=None, **overrides):
    config = GATE.STAGES[stage]
    identity = identity or _identity()
    if cells is None:
        cells = [
            _cell(route, speed, bars, config["seed"], identity)
            for route in GATE.ROUTE_ARMS
            for speed in GATE.SPEEDS
            for bars in config["densities"]
        ]
    payload = {
        "schema": GATE.SCHEMA, "stage": stage, "seed": config["seed"],
        "route_arms": list(GATE.ROUTE_ARMS), "speeds_mps": list(GATE.SPEEDS),
        "densities": list(config["densities"]),
        "envs": GATE.ENVS, "steps": GATE.STEPS, "warmup_steps": GATE.WARMUP_STEPS,
        "physical_gates_preregistered": GATE.PHYSICAL_GATES,
        "identity": identity, "pilot_authorization": None, "cells": cells,
    }
    payload.update(overrides)
    return payload


class BrakingRouteV3GateContractTest(unittest.TestCase):
    def test_preregistration_bytes_and_grid_are_frozen(self):
        self.assertEqual(GATE.sha256_file(GATE.PREREG), GATE.PREREG_SHA256)
        self.assertEqual(GATE.STAGES["pilot"]["seed"], 829)
        self.assertEqual(GATE.STAGES["confirmatory"]["seed"], 839)
        self.assertEqual(list(GATE.STAGES["pilot"]["densities"]), [70])
        self.assertEqual(list(GATE.STAGES["confirmatory"]["densities"]), [70, 115, 160, 205])
        self.assertNotIn(300, GATE.STAGES["confirmatory"]["densities"])
        self.assertEqual(GATE.ROUTE_ARMS, ("off", "global_astar_braking_v3"))
        self.assertEqual(GATE.ENVS, 32)
        self.assertEqual(GATE.STEPS, 300)

    def test_pilot_pass_does_not_authorize_long_training(self):
        payload = _summary("pilot")
        verdict = GATE.derive_verdict(payload)
        self.assertEqual(verdict["execution_integrity"], "PASS_8_CELL_INTEGRITY")
        self.assertEqual(verdict["gate"], "PASS_PILOT_AUTHORIZES_CONFIRMATORY")
        self.assertFalse(verdict["long_training_authorized"])

    def test_pilot_mechanism_fail_blocks_confirmatory(self):
        identity = _identity()
        cells = []
        for route in GATE.ROUTE_ARMS:
            for speed in GATE.SPEEDS:
                extras = {}
                if route != "off":
                    extras["v3_diagnostics"] = _diagnostics(runtime_replan_unsafe_start_count=1)
                cells.append(_cell(route, speed, 70, 829, identity, **extras))
        verdict = GATE.derive_verdict(_summary("pilot", identity=identity, cells=cells))
        self.assertEqual(verdict["gate"], "FAIL_BLOCKS_CONFIRMATORY")
        self.assertFalse(verdict["mechanism_pass"])

    def test_integrity_clean_fail_verifies_as_fail_not_void(self):
        identity = _identity()
        cells = []
        for route in GATE.ROUTE_ARMS:
            for speed in GATE.SPEEDS:
                extras = {}
                if route != "off":
                    extras["v3_diagnostics"] = _diagnostics(
                        soft_envelope_exit_count=1
                    )
                cells.append(_cell(route, speed, 70, 829, identity, **extras))
        payload = _summary("pilot", identity=identity, cells=cells)
        payload["verdict"] = GATE.derive_verdict(payload)
        self.assertEqual(payload["verdict"]["execution_integrity"], "PASS_8_CELL_INTEGRITY")
        self.assertEqual(payload["verdict"]["gate"], "FAIL_BLOCKS_CONFIRMATORY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(GATE.verify_summary(path), 0)

    def test_confirmatory_without_pilot_pass_is_void(self):
        verdict = GATE.derive_verdict(_summary("confirmatory"))
        self.assertEqual(verdict["execution_integrity"], "VOID_EXECUTION")
        self.assertEqual(verdict["gate"], "NOT_INTERPRETED")

    def test_prereg_sha_drift_is_void(self):
        identity = _identity()
        identity["preregistration_sha256"] = "0" * 64
        verdict = GATE.derive_verdict(_summary("pilot", identity=identity))
        self.assertEqual(verdict["execution_integrity"], "VOID_EXECUTION")

    def test_threshold_override_is_void(self):
        gates = dict(GATE.PHYSICAL_GATES)
        gates["mean_speed_ratio_min"] = 0.1
        verdict = GATE.derive_verdict(_summary("pilot", physical_gates_preregistered=gates))
        self.assertEqual(verdict["execution_integrity"], "VOID_EXECUTION")

    def test_default_grid_stays_canonical_and_does_not_alias_1p25(self):
        self.assertEqual(GATE.CONTRACT_VARIANT, "canonical_1p5")
        self.assertEqual(list(GATE.SPEEDS), [0.6, 0.9, 1.2, 1.5])
        self.assertNotIn(1.25, GATE.SPEEDS)
        self.assertEqual(GATE.record_id("off", 1.2, 70), "route_off__speed_1.2__bars_70")
        self.assertEqual(GATE.record_id("off", 1.5, 70), "route_off__speed_1.5__bars_70")

    def test_lower_contract_grid_is_isolated(self):
        import os
        previous = os.environ.get("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT")
        os.environ["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"] = "baseline_1p25"
        try:
            spec = importlib.util.spec_from_file_location(
                "braking_v3_gate_lower_standalone",
                ROOT / "tools/verify_navrl_braking_route_v3_gate.py",
            )
            lower = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lower)
            self.assertEqual(lower.CONTRACT_VARIANT, "baseline_1p25")
            self.assertEqual(list(lower.SPEEDS), [0.6, 0.9, 1.2, 1.25])
            self.assertNotIn(1.5, lower.SPEEDS)
            self.assertEqual(
                lower.sha256_file(lower.PREREG),
                "17e7f35e350087bf2733aca70dfca7210efe667ad1845dd053f7334ddf1645b4",
            )
            self.assertTrue(lower.PREREG.name.endswith("matched_spawn_2026-09-01.md"))
            self.assertEqual(
                lower.sha256_file(lower.EXECUTION_AUTHORITY),
                "70b8a08f0c95040a86c43e1be5ac11d0b688b9a6874f3cfc4548f914882f085f",
            )
            historical = ROOT / "docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md"
            self.assertEqual(
                lower.sha256_file(historical),
                "cd1347121c24ecd10273189360bed9ca76ffa80673aa89addf3ff0eaebc16252",
            )
            self.assertEqual(
                lower.record_id("off", 1.25, 70),
                "route_off__speed_1.25__bars_70",
            )
            self.assertNotEqual(
                lower.record_id("off", 1.25, 70),
                lower.record_id("off", 1.2, 70),
            )
        finally:
            if previous is None:
                os.environ.pop("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", None)
            else:
                os.environ["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"] = previous
        self.assertEqual(list(GATE.SPEEDS), [0.6, 0.9, 1.2, 1.5])


if __name__ == "__main__":
    unittest.main()
