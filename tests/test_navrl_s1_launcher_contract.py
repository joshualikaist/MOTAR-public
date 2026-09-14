import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_s1_search_state.sh"
PYTHON = "/home/fair/miniconda3/envs/aerialgym/bin/python"


class S1LauncherContractTest(unittest.TestCase):
    def invoke(self, *args, extra=None):
        env = dict(os.environ)
        env.update(PYTHON=PYTHON, NAVRL_S1_SEARCH_PREFLIGHT_ONLY="1")
        env.pop("CKPT", None)
        env.pop("CHECKPOINT", None)
        env.update(extra or {})
        return subprocess.run(
            ["bash", str(LAUNCHER), *args], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )

    def test_exactly_one_valid_arm_is_required(self):
        self.assertEqual(self.invoke().returncode, 2)
        self.assertEqual(self.invoke("off", "belief").returncode, 2)
        self.assertEqual(self.invoke("unknown").returncode, 2)

    def test_checkpoint_inputs_are_rejected(self):
        completed = self.invoke("off", extra={"CKPT": "/tmp/not-allowed.pth"})
        self.assertEqual(completed.returncode, 2)
        self.assertIn("fresh-only", completed.stdout)

    def test_frozen_tuple_and_derived_geofence(self):
        for arm in ("off", "geofence", "coverage", "belief"):
            with self.subTest(arm=arm):
                completed = self.invoke(arm)
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertIn(f"arm={arm} search={arm}", completed.stdout)
                self.assertIn("fresh=1", completed.stdout)
                self.assertIn("seed=919 epochs=3000 bars=70 density_curriculum=0", completed.stdout)
                self.assertIn("route=off speed=U[0.3,1.25]@1 governor=off save=250", completed.stdout)
                self.assertIn(f"run_tag=s1-search-{arm}-s919", completed.stdout)
                expected_geo = "0" if arm == "off" else "1"
                self.assertIn(f"geofence={expected_geo}", completed.stdout)


if __name__ == "__main__":
    unittest.main()
