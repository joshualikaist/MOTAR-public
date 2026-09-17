"""The retraining-readiness audit must stay bound to source and to its own evidence.

These guard the two ways this audit could quietly become wrong: the source facts it
rests on could change, or the documents could drift from the JSON they summarise.
"""
from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "docs/audits"
STATUS = AUDITS / "retraining_readiness_status_2026-09-17.json"
DIFF = AUDITS / "training_contract_vs_current_runtime_2026-09-17.json"
SNAPSHOT = AUDITS / "current_runtime_contract_2026-09-17.json"
AUDIT_DOC = AUDITS / "retraining_readiness_audit_2026-09-17.md"
CONTRACT_DOC = AUDITS / "frozen_policy_training_contract_2026-09-17.md"
PREREG = ROOT / "docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md"

TARGET_MOTION = ROOT / "aerial_gym/task/navrl_task/target_motion.py"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
V2_LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"

CHECKPOINT_SHA = "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"


def read(path):
    return path.read_text(encoding="utf-8")


class SourceFactsTheAuditRestsOn(unittest.TestCase):
    longMessage = False

    def test_capture_still_ends_the_episode_at_the_success_radius(self):
        # The whole termination verdict (A_CLOSE_APPROACH_TASK) and the removal of
        # time_within_approach_band_frac depend on this being true.
        task = read(NAVRL_TASK)
        self.assertIn("capture ends the episode", task,
                      "the capture-terminates-episode contract is no longer stated in source")
        self.assertIn("captured = seg_dist < self.task_config.success_radius", task,
                      "the swept capture test changed shape")
        self.assertRegex(read(TASK_CONFIG), r"success_radius\s*=\s*0\.5",
                         "success_radius is no longer 0.5 m; the termination audit is stale")

    def test_frozen_policy_trained_on_the_legacy_lineage(self):
        # The headline finding, and the reason arm H exists.
        self.assertIn('TARGET_MOTION_MODEL = "symmetric_local_steer_v2_heading_continuity90"',
                      read(TARGET_MOTION),
                      "the legacy target motion model constant changed")
        self.assertIn('NAVRL_TARGET_DYNAMICS:-legacy', read(V2_LAUNCHER),
                      "the v2 launcher no longer defaults target dynamics to legacy")

    def test_heading_valid_key_absence_is_still_classified_as_assumed(self):
        source = read(TARGET_MOTION)
        self.assertIn("HEADING_VALID_SPEED_ASSUMED", source)
        self.assertIn("1e-05", source,
                      "the pre-key inline epsilon note vanished; the 1e4 gap claim is unsupported")
        self.assertRegex(source, r"HEADING_VALID_SPEED_MPS\s*=\s*0\.10",
                         "the running heading-valid threshold changed")

    def test_config_drift_guard_still_only_warns(self):
        # If this ever becomes fail-closed, the strict gate this audit adds is
        # redundant and the audit text must be corrected rather than left stale.
        task = read(NAVRL_TASK)
        self.assertIn("warn, never override", task,
                      "the drift guard's documented behaviour changed; re-audit section 3")
        self.assertFalse(json.loads(read(DIFF))["guard_fails_closed"])

    def test_action_z_is_still_overwritten_but_observed(self):
        docstring = read(TASK_CONFIG)
        self.assertIn("retained for checkpoint compatibility", docstring)
        self.assertIn("overwritten by the task-level altitude PI loop", docstring)


class ArtifactsAgreeWithEachOther(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.status = json.loads(read(STATUS))
        self.diff = json.loads(read(DIFF))

    def test_status_counts_match_the_diff_artifact(self):
        changed = sorted(k for k, v in self.diff["rows"].items()
                         if v["status"].startswith("CHANGED"))
        self.assertEqual(self.status["contract_diff"]["changed_keys"], changed,
                         "status changed_keys drifted from the diff artifact")
        guarded = set(self.diff["guarded_keys"])
        self.assertEqual(
            self.status["contract_diff"]["changed_and_unguarded"],
            sorted(k for k in changed if k not in guarded),
            "status unguarded list drifted from the diff artifact")

    def test_status_records_no_training_and_no_gpu_work(self):
        self.assertFalse(self.status["new_ppo_training_run"])
        self.assertFalse(self.status["gpu_work_performed"])
        self.assertEqual(self.status["retraining_decision"]["decision"], "PENDING_EVALUATION")
        self.assertEqual(self.status["evaluation"]["state"], "READY_NOT_RUN")

    def test_status_records_no_silent_fixes(self):
        self.assertFalse(self.status["silent_fixes_applied"])
        self.assertFalse(self.status["historical_results_overwritten"])
        for defect in self.status["code_defects"]:
            self.assertFalse(defect["fixed"],
                             f"{defect['id']} is marked fixed; this audit does not apply fixes")

    def test_every_defect_names_a_real_source_location(self):
        for defect in self.status["code_defects"]:
            path, _, lines = defect["location"].partition(":")
            self.assertTrue((ROOT / path).is_file(), f"{defect['id']}: {path} missing")
            first = int(re.split(r"[-,]", lines)[0])
            total = len((ROOT / path).read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(first, total,
                                 f"{defect['id']}: line {first} beyond {path} ({total} lines)")

    def test_checkpoint_sha_is_consistent_everywhere(self):
        self.assertEqual(self.status["frozen_policy"]["checkpoint_sha256"], CHECKPOINT_SHA)
        for doc in (AUDIT_DOC, CONTRACT_DOC):
            self.assertIn(CHECKPOINT_SHA, read(doc), f"{doc.name} lost the checkpoint SHA")

    def test_audit_document_reports_the_same_defect_count(self):
        text = read(AUDIT_DOC)
        ids = set(re.findall(r"\| (D[1-9]) \|", text))
        self.assertEqual(ids, {d["id"] for d in self.status["code_defects"]},
                         "the audit document's defect table drifted from the status JSON")

    def test_termination_verdict_is_consistent(self):
        self.assertEqual(self.status["termination_audit"]["verdict"], "A_CLOSE_APPROACH_TASK")
        self.assertTrue(self.status["termination_audit"]["capture_ends_episode"])
        self.assertFalse(self.status["termination_audit"]["termination_modified_by_this_work"])
        self.assertIn("A. CLOSE_APPROACH_TASK", read(AUDIT_DOC))


class PreregistrationWasFrozenBeforeMeasurement(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.prereg = read(PREREG)
        self.status = json.loads(read(STATUS))

    def test_amendment_is_recorded_as_pre_measurement(self):
        self.assertIn("AMENDMENT 1", self.prereg)
        self.assertIn("측정 개시 전", self.prereg)
        self.assertTrue(self.status["preregistration"]["amended_before_any_measurement"])
        self.assertIn("NOT_RUN", self.prereg)

    def test_in_distribution_arm_exists(self):
        # Without arm H an E2 drop cannot be separated from "all arms are off-distribution".
        self.assertIn("H_historical", self.status["preregistration"]["arms"])
        self.assertIn("arm H", self.prereg)

    def test_ill_defined_metric_was_removed_not_thresholded(self):
        removed = self.status["preregistration"]["removed_outcomes"]
        self.assertIn("time_within_approach_band_frac", removed)
        self.assertIn("제거한다", self.prereg)

    def test_every_previously_unfrozen_number_is_now_fixed(self):
        prereg_status = self.status["preregistration"]
        self.assertEqual(prereg_status["evaluation_seeds"], [4101, 4102, 4103])
        self.assertEqual(prereg_status["episodes_per_cell"], 2048)
        self.assertEqual(prereg_status["cells"], 48)
        self.assertEqual(
            prereg_status["cells"],
            len(prereg_status["arms"]) * len(prereg_status["densities"])
            * len(prereg_status["evaluation_seeds"]),
            "cell count is inconsistent with arms x densities x seeds")
        for token in ("4101", "2048", "Wilson", "Holm-Bonferroni", "BCa"):
            self.assertIn(token, self.prereg, f"preregistration never fixes {token}")

    def test_condition_gate_is_mandated_and_fails_closed(self):
        gate = ROOT / self.status["preregistration"]["condition_gate"]
        self.assertTrue(gate.is_file(), "the mandated condition gate does not exist")
        self.assertTrue(self.status["preregistration"]["condition_gate_fails_closed"])
        self.assertIn("check_eval_condition_contract.py", self.prereg)


class ConditionGateBehaviour(unittest.TestCase):
    longMessage = False

    def test_gate_declares_the_observation_critical_keys(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gate", ROOT / "tools/audits/check_eval_condition_contract.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        for key in ("cfg_max_velocity", "cfg_lidar_hbeams", "cfg_lidar_max_range",
                    "cfg_episode_len_steps", "cfg_yaw_rate_max"):
            self.assertIn(key, gate.CRITICAL,
                          f"{key} drifted out of the gate's critical set")
        # The independent variable must be exactly the target-behaviour axis.
        for key in gate.DECLARED_INDEPENDENT:
            self.assertTrue(key.startswith("cfg_target_"),
                            f"{key} is declared independent but is not a target-behaviour key")

    def test_gate_flags_a_confound_and_passes_an_identical_environment(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gate", ROOT / "tools/audits/check_eval_condition_contract.py")
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        base = {"cfg_max_velocity": 2.5, "cfg_lidar_hbeams": 72, "cfg_target_pattern": "mixed"}
        verdict, _ = gate.compare(base, dict(base), set())
        self.assertEqual(verdict, "MATCHED")
        drifted = dict(base, cfg_max_velocity=2.0)
        verdict, rows = gate.compare(base, drifted, set())
        self.assertEqual(verdict, "CONFOUNDED")
        self.assertTrue(any(r["state"] == "CONFOUNDED_CRITICAL" for r in rows))
        # the declared independent variable must NOT trip the gate
        independent = dict(base, cfg_target_pattern="waypoint")
        verdict, _ = gate.compare(base, independent, set())
        self.assertEqual(verdict, "MATCHED")


if __name__ == "__main__":
    unittest.main()
