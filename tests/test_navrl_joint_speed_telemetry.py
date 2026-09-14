import unittest
import ast
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym/task/navrl_task/joint_speed_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("joint_speed_telemetry_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
JointSpeedTelemetry = MODULE.JointSpeedTelemetry
assess_preregistered_speed_gate = MODULE.assess_preregistered_speed_gate
risk_bin_index = MODULE.risk_bin_index


def synthetic_joint(contact_rate, capture_rate, *, contact_episodes=120):
    return {
        "schema_version": 1,
        "outcomes": {
            "capture": {
                "negative_margin_step_rate": capture_rate,
                "risk_bins": {
                    "negative": {"step_samples": int(2000 * capture_rate)},
                    "high_ge_1p5": {"step_samples": int(2000 * (1 - capture_rate))},
                },
            }
        },
        "bar_contact_preceding": {
            "episodes": contact_episodes,
            "step_samples": 1000,
            "negative_margin_step_rate": contact_rate,
        },
    }


class JointSpeedTelemetryTests(unittest.TestCase):
    def test_task_wiring_is_bulk_only_and_covers_lifecycle(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "aerial_gym/task/navrl_task/navrl_task.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        task = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
        )
        methods = {
            node.name: node for node in task.body if isinstance(node, ast.FunctionDef)
        }
        init_text = ast.get_source_segment(source, methods["__init__"])
        self.assertIn("if self._joint_speed_telemetry_enabled", init_text)
        self.assertIn("JointSpeedTelemetry", init_text)
        self.assertIn("_joint_speed_telemetry.reset_idx", ast.get_source_segment(source, methods["reset_idx"]))
        step_text = ast.get_source_segment(source, methods["step"])
        self.assertIn("_joint_speed_telemetry.record_step", step_text)
        self.assertIn("_joint_speed_telemetry.finish", step_text)
        export_text = ast.get_source_segment(source, methods["_export_bulk_eval_result"])
        self.assertIn('"joint_speed_allocation"', export_text)
        self.assertIn("if self._joint_speed_telemetry is not None", export_text)

        root = Path(__file__).resolve().parents[1]
        launcher = (
            root / "aerial_gym/rl_training/rl_games/eval_navrl_v2_joint_speed_telemetry.sh"
        ).read_text(encoding="utf-8")
        evaluator = (
            root / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("export NAVRL_JOINT_SPEED_TELEMETRY=1", launcher)
        self.assertIn("rev-parse --git-common-dir", launcher)
        self.assertIn('RESULT_ROOT="${REPO_ROOT}/results/', launcher)
        self.assertIn('elif "joint_speed_allocation" in payload', evaluator)

    def test_risk_bin_boundaries_are_fixed(self):
        values = torch.tensor([-0.01, 0.0, 0.49, 0.5, 1.49, 1.5, 9.0])
        self.assertEqual(risk_bin_index(values).tolist(), [0, 1, 1, 2, 2, 3, 3])

    def test_outcome_attribution_contact_window_and_proxy_counts(self):
        recorder = JointSpeedTelemetry(
            2, "cpu", step_dt=0.1, brake_mps2=2.0, reaction_s=0.1, hard_margin_m=0.45
        )
        for step in range(6):
            recorder.record_step(
                actual_velocity_xy=torch.tensor([[2.0, 0.1 * step], [0.3, 0.0]]),
                requested_command_xy=torch.tensor([[2.5, 0.2 * step], [0.5, 0.0]]),
                executed_command_xy=torch.tensor([[2.0, 0.16 * step], [0.5, 0.0]]),
                policy_action_xy=torch.tensor([[0.8, 0.05 * step], [0.2, 0.0]]),
                actual_direction_clearance_m=torch.tensor([0.6, 5.0]),
                requested_direction_clearance_m=torch.tensor([0.6, 5.0]),
                executed_direction_clearance_m=torch.tensor([0.6, 5.0]),
            )
        recorder.finish(
            torch.tensor([True, True]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            torch.tensor([0, -1]),
        )
        payload = recorder.payload((1, 1, 0), expected_bar_contacts=1)
        self.assertEqual(payload["bar_contact_preceding"]["episodes"], 1)
        self.assertEqual(payload["bar_contact_preceding"]["step_samples"], 6)
        self.assertEqual(payload["bar_contact_preceding"]["negative_margin_step_rate"], 1.0)
        self.assertEqual(
            payload["bar_contact_preceding"]["realized_heading_rate_proxy_radps_samples"], 5
        )
        self.assertEqual(payload["outcomes"]["capture"]["episodes"], 1)
        self.assertEqual(payload["outcomes"]["crash"]["episodes"], 1)

    def test_fail_closed_on_outcome_or_contact_disagreement(self):
        recorder = JointSpeedTelemetry(
            1, "cpu", step_dt=0.1, brake_mps2=2.0, reaction_s=0.1, hard_margin_m=0.45
        )
        recorder.record_step(
            actual_velocity_xy=torch.tensor([[1.0, 0.0]]),
            requested_command_xy=torch.tensor([[1.0, 0.0]]),
            executed_command_xy=torch.tensor([[1.0, 0.0]]),
            policy_action_xy=torch.tensor([[0.4, 0.0]]),
            actual_direction_clearance_m=torch.tensor([2.0]),
            requested_direction_clearance_m=torch.tensor([2.0]),
            executed_direction_clearance_m=torch.tensor([2.0]),
        )
        recorder.finish(
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([-1]),
        )
        with self.assertRaisesRegex(RuntimeError, "outcome mismatch"):
            recorder.payload((0, 0, 1), expected_bar_contacts=0)
        with self.assertRaisesRegex(RuntimeError, "bar-contact mismatch"):
            recorder.payload((1, 0, 0), expected_bar_contacts=1)

    def test_preregistered_gate_pass_and_noncausal_label(self):
        result = assess_preregistered_speed_gate(synthetic_joint(0.62, 0.30))
        self.assertTrue(result["quality"]["passed"])
        self.assertTrue(result["association_gate"]["passed"])
        self.assertFalse(result["causal_claim_allowed"])

    def test_actual_risk_uses_actual_direction_not_requested_direction(self):
        recorder = JointSpeedTelemetry(
            1, "cpu", step_dt=0.1, brake_mps2=2.0, reaction_s=0.1, hard_margin_m=0.45
        )
        recorder.record_step(
            actual_velocity_xy=torch.tensor([[2.0, 0.0]]),
            requested_command_xy=torch.tensor([[0.0, 2.0]]),
            executed_command_xy=torch.tensor([[0.0, 1.5]]),
            policy_action_xy=torch.tensor([[0.0, 0.8]]),
            # Actual path is blocked, requested/executed command direction is open.
            actual_direction_clearance_m=torch.tensor([0.6]),
            requested_direction_clearance_m=torch.tensor([5.0]),
            executed_direction_clearance_m=torch.tensor([5.0]),
        )
        recorder.finish(
            torch.tensor([True]), torch.tensor([True]), torch.tensor([False]),
            torch.tensor([False]), torch.tensor([-1]),
        )
        payload = recorder.payload((1, 0, 0), expected_bar_contacts=0)
        negative = payload["outcomes"]["capture"]["risk_bins"]["negative"]
        self.assertEqual(negative["step_samples"], 1)
        self.assertLess(negative["actual_stopping_margin_m"], 0.0)
        self.assertGreater(negative["requested_stopping_margin_m"], 0.0)
        self.assertGreater(negative["executed_stopping_margin_m"], 0.0)

    def test_preregistered_gate_does_not_move_for_weak_or_small_samples(self):
        weak = assess_preregistered_speed_gate(synthetic_joint(0.45, 0.30))
        self.assertEqual(weak["verdict"], "does_not_meet_preregistered_association_gate")
        small = assess_preregistered_speed_gate(
            synthetic_joint(0.9, 0.1, contact_episodes=20)
        )
        self.assertEqual(small["verdict"], "insufficient_quality")


if __name__ == "__main__":
    unittest.main()
