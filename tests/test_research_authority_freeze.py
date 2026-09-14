import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from history_helpers import require_local_evidence


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_research_authority", ROOT / "tools" / "check_research_authority.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestResearchAuthorityFreeze(unittest.TestCase):
    def test_frozen_authority_matches_result_summaries(self):
        require_local_evidence(ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json")
        receipt = MODULE.verify_authority()
        self.assertFalse(receipt["track_a"]["stage2_authorised"])
        self.assertFalse(receipt["track_b"]["long_training_authorized"])
        self.assertEqual(receipt["hardware_state"]["real_flights"], 0)
        geometry = receipt["corrected_environment_v2_2026_08_27"]["density_geometry_audit"]
        self.assertTrue(geometry["training_cap_passes"])
        self.assertEqual(geometry["training_cap_bars"], 205)
        self.assertEqual(geometry["disconnected_at_bars"], 300)
        self.assertFalse(geometry["authorizes_ppo"])
        gate = receipt["corrected_environment_v2_2026_08_27"]["route_physical_gate_r2"]
        self.assertEqual(gate["execution_integrity"], "PASS_32_CELL_INTEGRITY")
        self.assertEqual(gate["route_mechanism"], "FAIL_ROUTE_MECHANISM")
        self.assertFalse(gate["authorizes_ppo"])
        self.assertEqual(gate["fresh_ppo_epochs_run"], 0)
        lower_v3 = receipt["corrected_environment_v2_2026_08_27"][
            "braking_aware_route_v3_lower1p25"
        ]
        self.assertFalse(lower_v3["gpu_authority"])
        self.assertTrue(lower_v3["gpu_authority_consumed"])
        self.assertFalse(lower_v3["authorizes_ppo"])
        self.assertFalse(lower_v3["confirmatory_authorized"])
        self.assertEqual(lower_v3["pilot_execution_integrity"], "PASS_8_CELL_INTEGRITY")
        self.assertEqual(lower_v3["pilot_gate"], "FAIL_BLOCKS_CONFIRMATORY")
        self.assertEqual(
            lower_v3["gpu_authority_scope"],
            "one fresh baseline_1p25 receipt and one seed-829 70-bar 8-cell pilot only",
        )

    def test_track_b_prohibits_training_and_retuning(self):

        require_local_evidence(ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json")
        receipt = MODULE.verify_authority()
        prohibited = set(receipt["track_b"]["prohibited"])
        self.assertTrue({"ppo_training", "gain_or_margin_retune", "cell_or_grid_rerun"} <= prohibited)

    def test_route_off_smoke_is_narrow_and_does_not_reverse_routed_fail(self):

        require_local_evidence(ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json")
        receipt = MODULE.verify_authority()
        corrected = receipt["corrected_environment_v2_2026_08_27"]
        self.assertFalse(corrected["route_physical_gate_r2"]["authorizes_ppo"])
        smoke = corrected["route_off_learning_viability_smoke"]
        self.assertEqual(smoke["status"], "PASS_LEARNING_VIABILITY")
        self.assertEqual(smoke["fixed_bars"], 70)
        self.assertEqual(smoke["max_epochs"], 500)
        self.assertTrue(smoke["fresh_only"])
        self.assertFalse(smoke["gpu_authority"])
        self.assertTrue(smoke["gpu_authority_consumed"])
        self.assertFalse(smoke["authorizes_routed_ppo"])
        self.assertFalse(smoke["authorizes_long_training"])
        curriculum = corrected["route_off_curriculum"]
        self.assertEqual(curriculum["status"], "OPERATOR_STOPPED_INCOMPLETE")
        self.assertTrue(curriculum["resume_forbidden"])
        self.assertFalse(curriculum["gpu_authority"])
        self.assertTrue(curriculum["gpu_authority_consumed"])
        self.assertTrue(curriculum["fresh_only"])
        self.assertEqual(curriculum["density_bars"], [70, 205, 15])
        self.assertFalse(curriculum["authorizes_routed_ppo"])
        heldout = corrected["route_off_heldout_eval"]
        self.assertEqual(heldout["eval_seed"], 313)
        self.assertEqual(heldout["densities"], [70, 85, 100, 115, 130, 145])
        self.assertFalse(heldout["ood_205_included"])
        self.assertTrue(heldout["gen_ppo_forbidden"])
        self.assertFalse(heldout["authorizes_routed_ppo"])
        self.assertFalse(heldout["authorizes_resume"])
        self.assertFalse(heldout["authorizes_second_curriculum"])
        self.assertFalse(heldout["gpu_authority"])
        self.assertTrue(heldout["gpu_authority_consumed"])
        self.assertEqual(heldout["status"], "COMPLETE_VALID_WITH_METADATA_ERRATUM")
        self.assertEqual(
            heldout["result_summary_sha256"],
            "fd52ad6c4d4a9ba564510fd556cfd561b48ab9771a22799ecdc9956f84249559",
        )
        self.assertEqual(heldout["checkpoint_sha256"], (
            "541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132"
        ))
        self.assertEqual(
            heldout["preregistration_sha256"],
            "072060a82421ea67c6b1abfbb541d67ca89b7a26dd091a849f893edc520708c5",
        )

    def test_evidence_sha_drift_fails_closed(self):
        receipt = json.loads(MODULE.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
        receipt["evidence"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA drift"):
                MODULE.verify_authority(path)


if __name__ == "__main__":
    unittest.main()
