"""The forensics hooks must be unable to change the run they measure.

Source-level checks, not behaviour checks: the GPU evidence that a real trajectory is unchanged
lives in its own result directory, and these tests assert the properties that make that evidence
mean something, so a later edit cannot quietly wire a diagnostic label into an observation, a
reward or a termination.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
FORENSICS = ROOT / "aerial_gym/task/navrl_task/navrl_episode_forensics.py"
DIGEST = ROOT / "aerial_gym/task/navrl_task/navrl_trajectory_digest.py"
ALLOWED_METHODS = {"observe_perception", "observe_capture", "record_step", "finish", "reset_idx",
                   "export", "summary", "record", "as_dict", "hexdigest"}
BEHAVIOUR_FUNCTIONS = ("compute_state_reward_and_terminations", "_build_structured_observation",
                       "transform_action_to_command", "reset_idx", "step")


def parsed(path):
    return ast.parse(path.read_text())


class ModuleIndependenceTest(unittest.TestCase):
    def test_the_recorders_import_no_simulator_or_policy_module(self):
        for path in (FORENSICS, DIGEST):
            names = []
            for node in ast.walk(parsed(path)):
                if isinstance(node, ast.Import):
                    names += [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
            for name in names:
                self.assertFalse(name.startswith(("aerial_gym", "isaacgym", "rl_games")),
                                 "%s imports %s" % (path.name, name))

    def test_the_recorders_draw_no_randomness(self):
        for path in (FORENSICS, DIGEST):
            source = path.read_text()
            for forbidden in ("torch.rand", "torch.randn", "torch.randint", "random.",
                              "np.random", "numpy.random", "manual_seed", "default_rng"):
                self.assertNotIn(forbidden, source, "%s: %s" % (path.name, forbidden))

    def test_the_recorders_never_write_through_their_inputs(self):
        """No in-place op may touch a caller's tensor; every read goes through detach()."""
        source = FORENSICS.read_text()
        for forbidden in ("robot_position[", "target_position[", "diagnostics[\"visible\"]["):
            self.assertNotIn(forbidden, source)
        tree = parsed(FORENSICS)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("copy_", "add_", "mul_", "fill_") and isinstance(
                        node.func.value, ast.Name):
                    self.fail("in-place write on a bare name: " + node.func.attr)


class TaskHookTest(unittest.TestCase):
    def setUp(self):
        self.source = TASK.read_text()
        self.tree = parsed(TASK)
        self.functions = {node.name: node for node in ast.walk(self.tree)
                          if isinstance(node, ast.FunctionDef)}

    def attribute_uses(self, tree, name):
        found = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == name
                    and isinstance(node.value, ast.Name) and node.value.id == "self"):
                found.append(node)
        return found

    def test_both_recorders_default_to_off(self):
        for flag in ("NAVRL_EPISODE_FORENSICS", "NAVRL_TRAJECTORY_DIGEST"):
            self.assertIn('os.environ.get(\n            "%s", "0"\n        )' % flag, self.source)

    def test_forensics_is_refused_outside_a_checkpointed_evaluation(self):
        self.assertIn("NAVRL_EPISODE_FORENSICS is evaluation-only and requires", self.source)

    def test_every_call_is_guarded_and_uses_only_the_allowed_methods(self):
        calls = 0
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Attribute)
                    and owner.attr in ("_episode_forensics", "_trajectory_digest")):
                continue
            calls += 1
            self.assertIn(node.func.attr, ALLOWED_METHODS, node.func.attr)
        self.assertGreaterEqual(calls, 6)
        # Each recorder is only ever reached behind an `is not None` guard.
        for name in ("_episode_forensics", "_trajectory_digest"):
            guards = self.source.count("if self.%s is not None:" % name)
            self.assertGreaterEqual(guards, 2, name)

    def test_no_recorder_value_is_assigned_into_the_task(self):
        """A diagnostic that is read back into a task attribute could reach a decision."""
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            value = getattr(node, "value", None)
            if value is None:
                continue
            for inner in ast.walk(value):
                if (isinstance(inner, ast.Attribute)
                        and isinstance(inner.value, ast.Attribute)
                        and inner.value.attr in ("_episode_forensics", "_trajectory_digest")):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                            self.fail("recorder output assigned to self." + target.attr)

    def test_the_decision_paths_touch_only_the_documented_hooks(self):
        expected = {"compute_state_reward_and_terminations": {"observe_capture"},
                    "_build_structured_observation": {"observe_perception"},
                    "transform_action_to_command": set()}
        for name, allowed in expected.items():
            function = self.functions.get(name)
            if function is None:
                continue
            used = set()
            for node in ast.walk(function):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr in ("_episode_forensics", "_trajectory_digest")):
                    used.add(node.func.attr)
            self.assertEqual(used, allowed, name)

    def test_the_labels_never_appear_in_observation_or_reward_code(self):
        labels = ("LOST_SHORT", "LOST_LONG", "REACQUIRED", "NEVER_ACQUIRED", "FLY_BY",
                  "TERMINAL_OVERSHOOT", "failure_class")
        for name in BEHAVIOUR_FUNCTIONS:
            function = self.functions.get(name)
            if function is None:
                continue
            body = ast.get_source_segment(self.source, function) or ""
            for label in labels:
                self.assertNotIn(label, body, "%s names %s" % (name, label))

    def test_the_forensic_export_is_a_separate_file_from_the_bulk_result(self):
        self.assertIn("NAVRL_EPISODE_FORENSICS_JSON", self.source)
        self.assertIn("_export_episode_forensics", self.source)
        export = self.functions["_export_episode_forensics"]
        body = ast.get_source_segment(self.source, export) or ""
        self.assertNotIn("_bulk_eval_output", body)


if __name__ == "__main__":
    unittest.main()
