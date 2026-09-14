import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_corrected_nonoverlap_physical_smoke.sh"


class CorrectedNonoverlapSmokeContractTest(unittest.TestCase):
    def test_preflight_is_fresh_fixed_70_and_nonoverlap(self):
        env = dict(os.environ)
        env.update(
            CORRECTED_NONOVERLAP_SMOKE_PREFLIGHT_ONLY="1",
            CKPT="",
            CHECKPOINT="",
        )
        result = subprocess.run(
            [str(LAUNCHER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("fresh=1 seed=907 epochs=500 bars=70 fixed", result.stdout)
        self.assertIn("speed=U[0.3,1.25]@1", result.stdout)
        self.assertIn("route=off", result.stdout)
        self.assertIn("placement=footprint_clearance", result.stdout)
        self.assertIn("surface_clearance=0.45", result.stdout)
        self.assertIn("density=70->70 fixed_bars=70", result.stdout)
        self.assertIn("speed_final=1.25 ramp=1", result.stdout)

    def test_checkpoint_and_cli_are_rejected(self):
        for extra, env_update in (([], {"CKPT": "/tmp/forbidden.pth"}), (["--bad"], {})):
            env = dict(os.environ)
            env.update(CORRECTED_NONOVERLAP_SMOKE_PREFLIGHT_ONLY="1", **env_update)
            result = subprocess.run(
                [str(LAUNCHER), *extra], cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_launcher_pins_source_origin_and_receipt(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("NAVRL_REQUIRE_SOURCE_ROOT", source)
        self.assertIn("create_navrl_source_bundle.py", source)
        self.assertIn("--require-clean", source)
        self.assertIn("NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1", source)


if __name__ == "__main__":
    unittest.main()
