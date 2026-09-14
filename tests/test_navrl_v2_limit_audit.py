import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/summarize_navrl_v2_limit_audit.py"
SPEC = importlib.util.spec_from_file_location("summarize_navrl_v2_limit_audit", MODULE_PATH)
AUDIT_TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_TOOL)


class NavrlV2LimitAuditTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = AUDIT_TOOL.wilson(1485, 2050)
        rate = 1485 / 2050
        self.assertLess(low, rate)
        self.assertGreater(high, rate)
        self.assertAlmostEqual(low, 0.7046419392)
        self.assertAlmostEqual(high, 0.7432991596)

    def test_action_difference_matches_frozen_counts(self):
        low, high = AUDIT_TOOL.rate_difference_ci(1485, 2050, 1380, 2049)
        self.assertAlmostEqual(low, 0.0228484571)
        self.assertAlmostEqual(high, 0.0789334949)
        self.assertGreater(low, 0.0)

    def test_frozen_audit_contract(self):
        payload = json.loads(
            (ROOT / "results/navrl_v2_ep24000_limit_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["status"], "training-stopped-core-audit-complete")
        self.assertIn("mirror-paired", payload["pending_causal_checks"][0])
        self.assertEqual(payload["checkpoint"]["epoch"], 24000)
        self.assertEqual(
            payload["checkpoint"]["sha256"],
            "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f",
        )
        self.assertEqual(payload["training"]["canonical_epochs"], 14510)
        self.assertEqual(payload["training"]["epochs_at_205"], 4910)

    def test_next_run_is_single_variable_and_not_a_curriculum(self):
        payload = json.loads(
            (ROOT / "results/navrl_v2_ep24000_limit_audit.json").read_text(
                encoding="utf-8"
            )
        )
        experiment = payload["next_experiment"]
        self.assertEqual(experiment["single_changed_variable"], "NAVRL_OBSTACLE_SELECTOR")
        self.assertEqual(experiment["samples_per_arm"], 4_096_000)
        self.assertIn("fixed-205", experiment["name"])


if __name__ == "__main__":
    unittest.main()
