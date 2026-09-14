from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
LAUNCHER = RL_ROOT / "eval_navrl_v2_governor_adaptation_abc.sh"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"


class NavRLABCLauncherTest(unittest.TestCase):
    def test_publication_contract_is_pinned(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("SEED=53", text)
        self.assertIn("GAMES=2049", text)
        self.assertIn("DENSITIES=(130 160 190 205 220)", text)
        self.assertNotIn("NAVRL_ABC_SEED", text)
        self.assertNotIn("NAVRL_ABC_GAMES", text)
        self.assertIn(
            "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f",
            text,
        )
        self.assertIn(
            "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40",
            text,
        )

    def test_three_arms_share_one_bundle_and_run_sequentially(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}"', text)
        expected = (
            'for bars in "${DENSITIES[@]}"; do run_cell A_off',
            'for bars in "${DENSITIES[@]}"; do run_cell B_source_riskcap',
            'for bars in "${DENSITIES[@]}"; do run_cell C_trained_riskcap',
        )
        positions = [text.index(fragment) for fragment in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("SKIP complete", text)
        self.assertIn("partial cell directory requires manual inspection/move", text)

        evaluator = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn('SHARED_SOURCE_BUNDLE="${NAVRL_V2_SHARED_SOURCE_BUNDLE:-}"', evaluator)
        self.assertIn('ln -s "${SOURCE_MANIFEST}"', evaluator)

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(encoding="utf-8"), re.S)
        self.assertEqual(len(blocks), 4)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{LAUNCHER}:heredoc{index}", "exec")


if __name__ == "__main__":
    unittest.main()
