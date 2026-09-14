import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "tools/run_navrl_ref5in_detection_range_stage1_campaign.py"


def load_campaign():
    spec = importlib.util.spec_from_file_location("detrange_campaign_test", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DetectionRangeCampaignContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = load_campaign()
        cls.stage1 = cls.campaign.load_stage1()

    def test_status_names_the_preregistered_contract(self):
        payload = self.campaign.initial_state(self.stage1)
        contract = payload["contract"]
        self.assertEqual(contract["arms"], {"clip20": 20.0, "clip28": 28.0})
        self.assertEqual(contract["goal_distance_m"], [22.5, 28.0])
        self.assertEqual(contract["terminal_epoch"], 2900)
        self.assertEqual(contract["adaptation_epochs_per_arm"], 1000)
        self.assertEqual(contract["episodes_per_arm"], 2049)
        self.assertEqual(contract["primary_gate"]["range_helps_at_or_below"], -15.0)

    def test_gate_zero_failure_is_an_exception_before_evaluation(self):
        module = self.stage1
        original_verify = module.verify_training
        original_passed = module.training_gates_passed
        module.verify_training = lambda arm: {
            key: {"checked_by_launcher": True, "passed": key != "budget"}
            for key in module.TRAINING_GATES.values()
        }
        module.training_gates_passed = lambda evidence: False
        try:
            with self.assertRaises(module.ContractError):
                self.campaign.require_gate0(module, "clip20")
        finally:
            module.verify_training = original_verify
            module.training_gates_passed = original_passed

    def test_phase_order_is_single_owner_and_fail_fast(self):
        source = CAMPAIGN.read_text(encoding="utf-8")
        names = (
            "preflight",
            "train_clip20",
            "gate0_clip20",
            "train_clip28",
            "gate0_clip28",
            "evaluate_clip20",
            "evaluate_clip28",
            "finalize",
            "verify",
        )
        positions = [source.index(f'"{name}"') for name in names]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('status="failed"', source)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", source)


if __name__ == "__main__":
    unittest.main()
