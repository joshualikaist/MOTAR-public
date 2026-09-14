"""Source-level guards for bugs found while independently auditing verifications 2 and 4."""

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VerificationGuardTest(unittest.TestCase):
    def test_episode_dump_does_not_gate_normal_perception_resets(self):
        path = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        reset = next(
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset_idx"
        )
        vision_if = next(
            node for node in ast.walk(reset)
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute)
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "self"
                and node.test.attr == "vision_mode"
            )
        )
        source = path.read_text(encoding="utf-8")
        body = "\n".join(ast.get_source_segment(source, node) or "" for node in vision_if.body)
        self.assertIn("self.perception.reset_idx(env_ids)", body)
        self.assertIn("self.prev_action[env_ids] = 0.0", body)
        self.assertNotIn("self._episode_dump_path", body)

    def test_appearance_ab_uses_only_the_narrow_threshold_override(self):
        path = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_appearance_navigation_ab.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn("NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH=1", text)
        self.assertNotIn("NAVRL_V2_FORCE=1", text)

    def test_evaluator_narrow_override_only_skips_detector_threshold(self):
        path = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn('key == "cfg_detector_threshold" and allow_detector_threshold', text)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, re.S)
        guarded = [block for block in blocks if "and allow_detector_threshold" in block]
        self.assertGreaterEqual(len(guarded), 1)
        for block in guarded:
            # This catches the 2026-08-12 failure where the top-level preflight block referenced
            # the narrow override without defining it in that Python process.
            self.assertIn("allow_detector_threshold =", block)

    def test_corrective_launchers_pin_fresh_seeds_and_no_result_selection(self):
        games = ROOT / "aerial_gym/rl_training/rl_games"
        threshold = (games / "eval_navrl_v2_detector_threshold_diagnostic.sh").read_text()
        pose = (games / "eval_navrl_v2_pose_noise_rng_audit.sh").read_text()
        self.assertIn("SEEDS=(191 193)", threshold)
        self.assertIn("diagnostic only", threshold)
        self.assertIn("POSE_NOISE_SEED=9181", pose)
        self.assertIn("NAVRL_POSE_NOISE_SEED", pose)


if __name__ == "__main__":
    unittest.main()
