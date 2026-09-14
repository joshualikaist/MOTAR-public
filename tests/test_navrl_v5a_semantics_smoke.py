import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "aerial_gym/rl_training/rl_games/train_navrl_v2_v5a_semantics_smoke.sh"
)


class NavRLV5ASemanticsSmokeTest(unittest.TestCase):
    def _preflight(self, **hostile):
        env = os.environ.copy()
        env.pop("CKPT", None)
        env.update(
            {
                "V5A_PREFLIGHT_ONLY": "1",
                # These must all be discarded by the closed wrapper.
                "MAX_EPOCHS": "99999",
                "SEED": "999",
                "NAVRL_DETECTOR_CHECKPOINT": "/tmp/hostile-detector.pth",
                "NAVRL_OBSTACLE_SELECTOR": "ttc",
                "NAVRL_SPEED_GOVERNOR": "riskcap",
                "NAVRL_POSE_NOISE_POS_STD_M": "10",
                "NAVRL_ACTION_POLICY": "truncated_gaussian",
                "NAVRL_DENSITY_START": "300",
            }
        )
        env.update(hostile)
        return subprocess.run(
            [str(LAUNCHER)],
            cwd=LAUNCHER.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_hostile_environment_is_sanitized_and_contract_is_pinned(self):
        completed = self._preflight()
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("seed=197", completed.stdout)
        self.assertIn("episode=600 steps", completed.stdout)
        self.assertIn("density 70->300", completed.stdout)
        self.assertIn("PREFLIGHT PASS", completed.stdout)
        self.assertNotIn("seed=999", completed.stdout)
        self.assertNotIn("300->300", completed.stdout)

    def test_cli_and_checkpoint_resume_are_rejected(self):
        completed = subprocess.run(
            [str(LAUNCHER), "--checkpoint", "/tmp/hostile.pth"],
            cwd=LAUNCHER.parent,
            env={**os.environ, "V5A_PREFLIGHT_ONLY": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("no CLI arguments", completed.stdout)

        completed = self._preflight(CKPT="/tmp/hostile.pth")
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("refusing inherited CKPT", completed.stdout)

    def test_source_declares_exact_horizon_and_bootstrap_contract(self):
        task = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(
            encoding="utf-8"
        )
        yaml = (
            ROOT
            / "aerial_gym/rl_training/rl_games/ppo_navrl_perception_transformer.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("return sim_steps >= int(episode_len_steps)", task)
        self.assertIn('"time_outs": timeouts', task)
        self.assertIn("value_bootstrap: True", yaml)

    def test_max_epoch_does_not_duplicate_a_periodic_checkpoint(self):
        agent = (
            ROOT
            / "aerial_gym/rl_training/rl_games/early_stop_a2c_agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn("periodic_checkpoint_saved = (", agent)
        self.assertIn("epoch_num % self.save_freq == 0", agent)
        self.assertIn('self.save(os.path.join(self.nn_dir, "last_" + checkpoint_name))', agent)
        self.assertNotIn(
            '"_rew_" + str(mean_rewards).replace("[", "_").replace("]", "_")',
            agent[agent.index("and epoch_num >= self.max_epochs") : agent.index("MAX EPOCHS NUM!")],
        )


if __name__ == "__main__":
    unittest.main()
