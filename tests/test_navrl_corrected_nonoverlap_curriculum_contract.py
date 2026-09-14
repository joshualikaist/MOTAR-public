import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_corrected_nonoverlap_physical_curriculum.sh"


class CorrectedNonoverlapCurriculumContractTest(unittest.TestCase):
    def test_preflight_pins_fresh_route_off_nonoverlap_curriculum(self):
        env = dict(os.environ)
        env.update(CORRECTED_NONOVERLAP_CURRICULUM_PREFLIGHT_ONLY="1", CKPT="", CHECKPOINT="")
        result = subprocess.run(
            [str(LAUNCHER)], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("fresh=1 seed=911 epochs=30000 density=70:15:205", result.stdout)
        self.assertIn("route=off", result.stdout)
        self.assertIn("placement=footprint_clearance", result.stdout)
        self.assertIn("density=70->205 fixed_bars=curriculum", result.stdout)
        self.assertIn("speed_final=1.25 ramp=1", result.stdout)

    def test_checkpoint_and_cli_are_rejected(self):
        for extra, update in (([], {"CKPT": "/tmp/x.pth"}), (["--bad"], {})):
            env = dict(os.environ)
            env.update(CORRECTED_NONOVERLAP_CURRICULUM_PREFLIGHT_ONLY="1", **update)
            result = subprocess.run(
                [str(LAUNCHER), *extra], cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_source_and_density_contract_are_closed(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for literal in (
            "NAVRL_REQUIRE_SOURCE_ROOT", "create_navrl_source_bundle.py", "--require-clean",
            "NAVRL_DENSITY_CHECK_EPS=16384", "NAVRL_DENSITY_MIN_EPOCHS=1000",
            "NAVRL_DENSITY_THRESHOLD_SCHEDULE=70:0.82,85:0.77,100:0.72,115:0.70",
        ):
            self.assertIn(literal, source)


if __name__ == "__main__":
    unittest.main()
