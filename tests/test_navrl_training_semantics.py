import ast
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
PPO_PATH = ROOT / "aerial_gym/rl_training/rl_games/ppo_navrl_perception_transformer.yaml"
TREE = ast.parse(TASK_PATH.read_text(encoding="utf-8"), filename=str(TASK_PATH))


def load_function(name):
    node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(TASK_PATH), "exec"), namespace)
    return namespace[name]


episode_limit_reached = load_function("_episode_limit_reached")
episode_outcome_info = load_function("_episode_outcome_info")


class NavRLTrainingSemanticsTest(unittest.TestCase):
    def test_configured_horizon_is_exact_not_plus_one(self):
        steps = torch.tensor([599, 600, 601])
        self.assertEqual(
            episode_limit_reached(steps, 600).tolist(), [False, True, True]
        )

    def test_timeout_bootstrap_key_excludes_true_terminals(self):
        successes = torch.tensor([False, True, False, False])
        crashes = torch.tensor([False, False, True, False])
        truncations = torch.tensor([True, True, True, False])
        timeouts, infos = episode_outcome_info(successes, crashes, truncations)
        self.assertEqual(timeouts.tolist(), [True, False, False, False])
        self.assertIs(infos["timeouts"], infos["time_outs"])
        self.assertTrue(torch.equal(infos["time_outs"], timeouts))

    def test_rlgames_bootstrap_is_enabled_for_the_published_training_config(self):
        text = PPO_PATH.read_text(encoding="utf-8")
        self.assertIn("value_bootstrap: True", text)


if __name__ == "__main__":
    unittest.main()
