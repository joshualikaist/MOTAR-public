"""CPU contracts for the recovery-v2 NO_CONNECTOR geometry preregistration."""

import importlib.util
import json
import math
import sys
import tempfile
import unittest

from history_helpers import require_local_evidence
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/diagnose_navrl_physical_target_recovery_v2_no_connector.py"
SPEC = importlib.util.spec_from_file_location("recovery_v2_no_connector_forensics", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class NoConnectorForensicsContractTest(unittest.TestCase):
    def test_frozen_probe_is_70bar_recovery_v2_lower_speeds_only(self):
        self.assertEqual(MOD.DENSITIES, (70,))
        self.assertEqual(MOD.SPEEDS, (0.6, 0.9, 1.2, 1.25))
        self.assertEqual(MOD.ENVS, 32)
        self.assertEqual(MOD.SEED, 827)
        self.assertEqual(MOD.ROUTE_MODE, "global_astar_recovery_v2")
        self.assertEqual(MOD.CONTRACT_VARIANT, "baseline_1p25")
        self.assertEqual(MOD.VEL_KP, 2.5)
        self.assertEqual(MOD.TRACKING_MARGIN_M, 0.45)
        self.assertEqual(MOD.ANCHOR_RADIUS_CELLS, 3)
        self.assertEqual(MOD.SOFT_HYSTERESIS_M, 0.25)
        self.assertAlmostEqual(MOD.RECOVERY_HARD_EPSILON_M, 1e-4 + 0.0123)
        contract = MOD.probe_contract()
        self.assertEqual(contract["runtime_wall_margin_m"], 0.50)
        self.assertEqual(contract["route_boundary_margin_m"], 1.25)
        self.assertEqual(contract["boundary_soft_minus_hard_m"], 0.75)
        self.assertTrue(contract["gate_artifacts_read_only"])
        self.assertFalse(contract.get("passes_32_cell_mechanism", False))
        self.assertEqual(
            contract["output_root"],
            "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827",
        )
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"NAVRL_NUM_BARS": "70"', source)
        self.assertIn('"NAVRL_MAX_BARS": "300"', source)
        self.assertIn('"NAVRL_TARGET_ROUTE_MODE": ROUTE_MODE', source)
        self.assertNotIn("1.5", "".join(str(s) for s in MOD.SPEEDS))

    def test_recovery_v2_anchor_kwargs_are_not_the_v1_forensic_search(self):
        recovery = MOD.recovery_v2_anchor_kwargs()
        legacy = MOD.v1_forensic_anchor_kwargs()
        self.assertEqual(recovery["soft_hysteresis_m"], 0.25)
        self.assertEqual(legacy["soft_hysteresis_m"], 0.0)
        self.assertGreater(recovery["hard_epsilon_m"], legacy["hard_epsilon_m"])
        self.assertAlmostEqual(recovery["hard_epsilon_m"], 0.0124)

    def test_replica_matches_planner_recovery_anchor_and_can_disagree_with_v1(self):
        point = [20.0, 20.0]
        bars = np.zeros((0, 2), dtype=np.float64)
        half = np.zeros((0, 2), dtype=np.float64)
        lo, hi = [0.0, 0.0], [40.0, 40.0]
        support = [0.2068816086567407, 0.2068816086567407]
        replica = MOD.recovery_anchor_query(point, bars, half, lo, hi, support)
        planner = MOD._load_planner()
        direct = planner.nearest_soft_free_anchor(
            point, bars, half, lo, hi, support, 0.50, 1.25,
            **MOD.recovery_v2_anchor_kwargs(),
        )
        self.assertEqual(replica["exists"], True)
        self.assertEqual(replica["hard_connector_safe"], True)
        self.assertEqual(replica["cell_ij"], direct["cell_ij"])
        self.assertEqual(MOD.recovery_anchor_query(
            point, bars, half, lo, hi, support, variant="v1"
        )["exists"], True)

        boxed_point = [1.0, 1.0]
        recovery = MOD.recovery_anchor_query(
            boxed_point, bars, half, lo, hi, [0.20, 0.20]
        )
        legacy = MOD.recovery_anchor_query(
            boxed_point, bars, half, lo, hi, [0.20, 0.20], variant="v1"
        )
        self.assertEqual(legacy.get("exists"), True)
        self.assertEqual(legacy.get("hard_connector_safe"), True)
        self.assertEqual(recovery.get("exists"), False)
        self.assertEqual(recovery.get("hard_connector_safe"), False)

    def test_descriptive_verdict_cannot_pass_the_32cell_mechanism(self):
        present = MOD.descriptive_verdict(18, 20)
        self.assertEqual(present["label"], "ANCHOR_PRESENT_LATCH")
        self.assertFalse(present["passes_32_cell_mechanism"])
        self.assertFalse(present["authorizes_retune_or_ppo"])
        absent = MOD.descriptive_verdict(2, 20)
        self.assertEqual(absent["label"], "ANCHOR_ABSENT_AT_LATCH")
        small = MOD.descriptive_verdict(19, 19)
        self.assertEqual(small["label"], "INCONCLUSIVE")
        mixed = MOD.descriptive_verdict(10, 20)
        self.assertEqual(mixed["label"], "INCONCLUSIVE")

    def test_wilson_interval_is_the_v1_formula(self):
        lower, upper = MOD.wilson_interval(18, 20)
        z = 1.96
        p = 18.0 / 20.0
        denominator = 1.0 + z * z / 20.0
        center = p + z * z / 40.0
        spread = z * math.sqrt(p * (1.0 - p) / 20.0 + z * z / (4.0 * 20.0 * 20.0))
        self.assertAlmostEqual(lower, (center - spread) / denominator)
        self.assertAlmostEqual(upper, (center + spread) / denominator)
        self.assertGreater(lower, 0.5)
        self.assertIsNone(MOD.wilson_interval(0, 0))

    def test_packed_classify_is_delegated_not_redefined(self):
        packed = MOD._load_packed_diag()
        cases = (
            (1, 19, -0.04, "brake_no_anchor_likely"),
            (0, 19, -0.05, "same_interval_brake_no_anchor_likely"),
            (2, 19, 0.16, "connect_failed_resume_likely"),
            (2, 19, -0.05, "connect_failed_certificate_likely"),
            (1, 22, 0.02, "brake_timeout"),
        )
        for state, status, soft, label in cases:
            self.assertEqual(MOD.classify_no_connector_entry(state, status, soft), label)
            self.assertEqual(
                MOD.classify_no_connector_entry(state, status, soft),
                packed.classify_no_connector_entry(state, status, soft),
            )

    def test_runtime_replica_agree_is_false_when_runtime_bool_is_missing(self):
        present = {"exists": True, "hard_connector_safe": True}
        absent = {"exists": False, "hard_connector_safe": False}
        self.assertTrue(MOD.runtime_replica_agree(True, present))
        self.assertTrue(MOD.runtime_replica_agree(False, absent))
        self.assertFalse(MOD.runtime_replica_agree(True, absent))
        self.assertFalse(MOD.runtime_replica_agree(None, present))
        self.assertFalse(MOD.replica_present({"exists": True, "hard_connector_safe": False}))

    def test_analyze_events_pools_primary_only_and_voids_identity_disagreement(self):
        replica = {"exists": True, "hard_connector_safe": True}
        primary = {
            "event": "no_connector",
            "packed_class": "brake_no_anchor_likely",
            "runtime_replica_agree": True,
            "runtime_anchor_ok": True,
            "nearest_soft_free_anchor": replica,
            "hard_free": True,
            "soft_free": False,
        }
        timeout = {
            "event": "no_connector",
            "packed_class": "brake_timeout",
            "runtime_replica_agree": False,
            "nearest_soft_free_anchor": replica,
            "hard_free": True,
            "soft_free": False,
        }
        resume = {
            "event": "no_connector",
            "packed_class": "connect_failed_resume_likely",
            "runtime_replica_agree": False,
            "resume_replan_status": "unsafe_start",
            "nearest_soft_free_anchor": replica,
        }
        pooled = MOD.analyze_events([primary, dict(primary), timeout, resume])
        self.assertEqual(pooled["primary_n"], 2)
        self.assertEqual(pooled["anchor_present"], 2)
        self.assertEqual(pooled["hard_free_soft_unsafe"], 2)
        self.assertFalse(pooled["identity_void"])
        self.assertEqual(pooled["class_counts"]["brake_timeout"], 1)
        self.assertEqual(pooled["resume_replan_status_counts"]["unsafe_start"], 1)
        self.assertEqual(pooled["decision_rule"]["label"], "INCONCLUSIVE")
        self.assertFalse(pooled["decision_rule"]["passes_32_cell_mechanism"])

        disagreed = dict(primary)
        disagreed["runtime_replica_agree"] = False
        voided = MOD.analyze_events([disagreed, timeout])
        self.assertTrue(voided["identity_void"])
        self.assertEqual(voided["decision_rule"]["label"], "VOID_OBSERVER_IDENTITY")
        self.assertEqual(voided["primary_n"], 1)

        missing = dict(primary)
        missing["runtime_replica_agree"] = None
        missing["runtime_anchor_ok"] = None
        self.assertTrue(MOD.analyze_events([missing])["identity_void"])
        malformed = dict(primary)
        malformed["runtime_replica_agree"] = True
        malformed["runtime_anchor_ok"] = False
        malformed["nearest_soft_free_anchor"] = {"exists": None, "hard_connector_safe": None}
        self.assertTrue(MOD.analyze_events([malformed])["identity_void"])

    def test_attach_observer_does_not_change_returns_and_records_packed_replica(self):
        class FakeManager:
            STATUS_CODES = {"ok": 1, "recovery_no_connector": 19, "unsafe_start": 3}

            def __init__(self):
                self.recovery_state = np.array([1, 0], dtype=np.int16)
                self.status_code = np.array([0, 0], dtype=np.int32)
                self._anchor = np.array([True, False])
                self._brake = np.array([True, True])

            def mark_no_connector(self, mask, hard_breach=False, timeout_kind=None):
                flags = np.asarray(mask, dtype=bool)
                self.recovery_state = np.where(flags, np.int16(4), self.recovery_state)
                self.status_code = np.where(flags, np.int32(19), self.status_code)
                return "mark-ok"

            def mark_local_infeasible_soft_free(self, mask):
                return "soft-ok"

            def recovery_anchor_idx(self, env_ids, *args, **kwargs):
                return self._anchor.copy()

            def brake_connector_idx(self, env_ids, *args, **kwargs):
                return self._brake.copy()

            def plan_idx(self, env_ids, *args, **kwargs):
                if kwargs.get("is_replan"):
                    for env in np.atleast_1d(np.asarray(env_ids)).astype(int).tolist():
                        self.status_code[int(env)] = 3
                return "plan-ok"

        class FakeTask:
            def __init__(self):
                self._target_route_mode = "global_astar_recovery_v2"
                self._target_route_manager = FakeManager()
                self.num_envs = 2
                self.n_bars_active = 0
                self._bar_offset = 0
                self.num_task_steps = 7
                self.target_position = np.array([[20.0, 20.0, 1.5], [1.0, 1.0, 1.5]])
                self.target_vel_w = np.array([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
                self._target_route_support_xy = np.array([
                    [0.2068816086567407, 0.2068816086567407],
                    [0.20, 0.20],
                ])
                self.obs_dict = {
                    "obstacle_position": np.zeros((2, 1, 3)),
                    "asset_collision_half_extents": np.zeros((2, 1, 3)),
                    "env_bounds_min": np.zeros((2, 3)),
                    "env_bounds_max": np.array([[40.0, 40.0, 3.0], [40.0, 40.0, 3.0]]),
                }

        task = FakeTask()
        recorder = MOD.attach_observer(task)
        recorder.begin_interval()
        anchor = task._target_route_manager.recovery_anchor_idx([0, 1])
        brake = task._target_route_manager.brake_connector_idx([0])
        plan = task._target_route_manager.plan_idx([1], is_replan=True)
        soft = task._target_route_manager.mark_local_infeasible_soft_free(np.array([False, False]))
        marked = task._target_route_manager.mark_no_connector(np.array([True, False]))
        self.assertTrue(np.array_equal(anchor, np.array([True, False])))
        self.assertTrue(np.array_equal(brake, np.array([True, True])))
        self.assertEqual(plan, "plan-ok")
        self.assertEqual(soft, "soft-ok")
        self.assertEqual(marked, "mark-ok")
        self.assertEqual(len(recorder.events), 1)
        event = recorder.events[0]
        json.dumps(event, allow_nan=False)
        self.assertEqual(event["packed_class"], "brake_no_anchor_likely")
        self.assertTrue(event["primary"])
        self.assertTrue(event["nearest_soft_free_anchor"]["exists"])
        self.assertTrue(event["nearest_soft_free_anchor"]["hard_connector_safe"])
        self.assertTrue(event["runtime_anchor_ok"])
        self.assertTrue(event["runtime_replica_agree"])
        self.assertEqual(event["resume_replan_status"], None)
        self.assertEqual(recorder.last_replan_status[1], "unsafe_start")

    def test_child_env_is_recovery_v2_lower_1p25_without_packed_telemetry_or_1p5(self):

        require_local_evidence(ROOT / "results/navrl_physical_target_braking_lower1p25_headingrest_seed827/receipt.json")
        frozen = MOD.FROZEN_CHILD_ENV
        self.assertEqual(frozen["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"], "baseline_1p25")
        self.assertEqual(frozen["NAVRL_NUM_BARS"], "70")
        self.assertEqual(frozen["NAVRL_MAX_BARS"], "300")
        self.assertEqual(frozen["NAVRL_TARGET_ROUTE_MODE"], "global_astar_recovery_v2")
        self.assertNotIn("NAVRL_TARGET_RECOVERY_EVAL_TELEMETRY", frozen)
        self.assertNotIn("1.5", "".join(frozen.values()))
        child = MOD.build_child_environment(0.6)
        self.assertEqual(child["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"], "baseline_1p25")
        self.assertEqual(child["NAVRL_NUM_BARS"], "70")
        self.assertEqual(child["NAVRL_TARGET_SPEED"], "0.6")
        self.assertEqual(child["NAVRL_TARGET_RECOVERY_PROBE_VALIDATED"], "1")
        self.assertEqual(child["NAVRL_TRAINING_SOURCE_MANIFEST_SHA256"], MOD.TRAINING_SOURCE_MANIFEST_SHA256)
        self.assertNotIn("NAVRL_TARGET_RECOVERY_EVAL_TELEMETRY", child)
        self.assertNotIn("1.5", child["NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS"])
        self.assertEqual(
            child["NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS"],
            "0.6,0.9,1.2,1.25",
        )
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn('"NAVRL_TARGET_RECOVERY_EVAL_TELEMETRY"', source)
        self.assertIn("packed 32-cell telemetry observer is forbidden", source)
        self.assertIn("import isaacgym", source)
        self.assertLess(source.index("import isaacgym"), source.index("import aerial_gym"))

    def test_run_refuses_gate_path_and_existing_output_and_summary_is_immutable(self):
        with mock.patch.object(sys, "argv", [str(PATH), "--run", "--output", str(MOD.GATE_DIR)]):
            with self.assertRaisesRegex(RuntimeError, "32-cell gate"):
                MOD.main()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / MOD.OUTPUT_ROOT.name
            fake.mkdir()
            with mock.patch.object(MOD, "OUTPUT_ROOT", fake):
                with mock.patch.object(sys, "argv", [str(PATH), "--run", "--output", str(fake)]):
                    with self.assertRaisesRegex(RuntimeError, "fresh canonical OUTPUT_ROOT"):
                        MOD.main()
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            MOD.summarize(MOD.OUTPUT_ROOT)
        with mock.patch.object(sys, "argv", [str(PATH), "--summarize"]):
            with self.assertRaisesRegex(RuntimeError, "internal-only"):
                MOD.main()


if __name__ == "__main__":
    unittest.main()
