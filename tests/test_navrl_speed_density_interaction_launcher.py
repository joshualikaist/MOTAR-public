from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "aerial_gym/rl_training/rl_games/eval_navrl_v2_speed_density_interaction.sh"
)


class NavRLSpeedDensityInteractionLauncherTest(unittest.TestCase):
    def test_preregistered_grid_is_pinned(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("SEEDS=(59 61)", text)
        self.assertIn("SPEEDS=(0.3 1.5)", text)
        self.assertIn("DENSITIES=(130 160 190 205)", text)
        self.assertIn("GAMES=2049", text)
        self.assertIn(
            "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40",
            text,
        )
        self.assertNotIn("NAVRL_SPEED_DENSITY_SEEDS", text)
        self.assertNotIn("NAVRL_SPEED_DENSITY_SPEEDS", text)
        self.assertNotIn("NAVRL_SPEED_DENSITY_DENSITIES", text)

    def test_cells_share_source_and_primary_model_is_fixed(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}"', text)
        self.assertIn('NAVRL_SPEED_GOVERNOR=riskcap', text)
        self.assertIn("SKIP complete", text)
        self.assertIn("partial cell directory requires manual inspection/move", text)
        self.assertIn(
            "binomial_logit(capture) ~ seed + density + fast + density:fast",
            text,
        )
        self.assertIn("likelihood-ratio test of density:fast, 1 df", text)
        self.assertIn('for seed in "${SEEDS[@]}"', text)
        self.assertIn('for speed in "${SPEEDS[@]}"', text)
        self.assertIn('for bars in "${DENSITIES[@]}"', text)

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(encoding="utf-8"), re.S)
        self.assertEqual(len(blocks), 4)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{LAUNCHER}:heredoc{index}", "exec")


if __name__ == "__main__":
    unittest.main()
