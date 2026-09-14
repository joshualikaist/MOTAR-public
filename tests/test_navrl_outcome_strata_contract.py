"""CPU-only wiring checks for outcome-aware held-out strata."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
TASK = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "NavRLTask")


def method(name):
    return next(node for node in TASK.body if isinstance(node, ast.FunctionDef) and node.name == name)


class TestOutcomeStrataWiring(unittest.TestCase):
    def test_bulk_progress_uses_eval_not_training_counters(self):
        progress = method("_log_progress")
        calls = {
            node.func.attr
            for node in ast.walk(progress)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_record_eval_outcome_strata", calls)
        self.assertNotIn("_record_density_strata", calls)

    def test_export_contains_all_outcomes_and_crash_causes(self):
        export = method("_export_bulk_eval_result")
        strings = {
            node.value for node in ast.walk(export)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {"successes", "crash", "timeout", "crash_causes", "capture_rate",
             "crash_rate", "timeout_rate", "bar_contact", "out_of_bounds",
             "distance_by_pattern"}.issubset(strings)
        )

    def test_export_fails_closed_on_accounting(self):
        export = method("_export_bulk_eval_result")
        calls = {
            node.func.attr
            for node in ast.walk(export)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_validate_eval_outcome_strata", calls)
        validation = method("_validate_eval_outcome_strata")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(validation)))

    def test_eval_counters_are_not_checkpointed_density_counters(self):
        init = method("__init__")
        attrs = {
            node.attr for node in ast.walk(init)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
        }
        self.assertTrue(
            {"_eval_speed_crash", "_eval_speed_timeout", "_eval_dist_crash",
             "_eval_dist_timeout", "_eval_pattern_crash_cause"}.issubset(attrs)
        )

    def test_eval_bins_use_applied_speed_min_but_training_bins_stay_historical(self):
        eval_bins = method("_episode_eval_strata_bins")
        eval_strings = {
            node.value for node in ast.walk(eval_bins)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("speed_min", eval_strings)
        training_bins = method("_episode_strata_bins")
        training_strings = {
            node.value for node in ast.walk(training_bins)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("speed_min", training_strings)


if __name__ == "__main__":
    unittest.main()
