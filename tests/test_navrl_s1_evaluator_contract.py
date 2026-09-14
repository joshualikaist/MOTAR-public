import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_s1_search_state_heldout.sh"
PYTHON = "/home/fair/miniconda3/envs/aerialgym/bin/python"


class S1EvaluatorContractTest(unittest.TestCase):
    def checkpoint(self, directory, arm, *, masked=False, epoch=3000):
        path = Path(directory) / f"last_gen_ppo_ep_{epoch}_rew_1.0.pth"
        torch.save(
            {
                "env_state": {
                    "cfg_search_state": arm,
                    "cfg_search_state_force_invalid": masked,
                    "cfg_geofence_actor": arm != "off",
                }
            },
            path,
        )
        return path

    def invoke(self, arm, checkpoint, *extra):
        env = dict(os.environ)
        env.update(PYTHON=PYTHON, NAVRL_S1_EVAL_PREFLIGHT_ONLY="1")
        return subprocess.run(
            ["bash", str(EVALUATOR), arm, str(checkpoint), *extra],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )

    def test_seed_density_arm_and_mask_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, "belief")
            completed = self.invoke("belief", checkpoint, "--masked")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("arm=belief masked=1 seed=331 densities=70 145 episodes=2049", completed.stdout)
            self.assertIn("gen_ppo=forbidden", completed.stdout)
            self.assertIn("belief_masked", completed.stdout)

    def test_checkpoint_arm_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, "coverage")
            completed = self.invoke("belief", checkpoint)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("checkpoint search-state mismatch", completed.stdout)

    def test_nonterminal_and_masked_training_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            nonterminal = self.checkpoint(directory, "off", epoch=2999)
            self.assertEqual(self.invoke("off", nonterminal).returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            masked = self.checkpoint(directory, "coverage", masked=True)
            completed = self.invoke("coverage", masked)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("evaluation mask enabled", completed.stdout)

    def test_off_mask_is_meaningless_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, "off")
            completed = self.invoke("off", checkpoint, "--masked")
            self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
