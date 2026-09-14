from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_detector_navigation_ab.sh"


class DetectorNavigationABLauncherTest(unittest.TestCase):
    def test_primary_contract_is_pinned(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("SEEDS=(83 89)", text)
        self.assertIn("GAMES=2049", text)
        self.assertIn("NONINFERIORITY_MARGIN_PP=2.0", text)
        self.assertIn("THRESHOLD=0.55", text)
        self.assertIn(
            "8da32d6f21bfbd3bdd5ec5de9ef9cb09e8deb4bd5ce511630e19afee33f26f10",
            text,
        )
        self.assertIn('"noninferiority_margin_pp": -2.0', text)
        self.assertNotIn("NAVRL_DETECTOR_AB_SEEDS", text)
        self.assertNotIn("NAVRL_DETECTOR_AB_MARGIN", text)

    def test_arms_share_source_and_are_resume_safe(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}"', text)
        self.assertIn('run_cell "${seed}" analytic_bootstrap ""', text)
        self.assertIn('run_cell "${seed}" learned_v2 "${DETECTOR}"', text)
        self.assertIn("SKIP complete", text)
        self.assertIn("partial cell requires manual inspection/move", text)
        self.assertIn("offline detector gate did not pass", text)

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(encoding="utf-8"), re.S)
        self.assertEqual(len(blocks), 3)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{LAUNCHER}:heredoc{index}", "exec")


if __name__ == "__main__":
    unittest.main()
