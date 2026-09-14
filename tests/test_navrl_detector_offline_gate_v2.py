import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/train_navrl_target_detector_v2.py"
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/run_navrl_detector_offline_gate.sh"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("navrl_detector_gate_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = torch
    return module


class DetectorOfflineGateV2Test(unittest.TestCase):
    def test_campaign_contract_is_pinned_and_does_not_train_ppo(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--train-frames 8192", text)
        self.assertIn("--validation-frames 2048", text)
        self.assertIn("--test-frames 4096", text)
        self.assertIn("--epochs 10", text)
        self.assertIn("PYTHONNOUSERSITE=1", text)
        self.assertNotIn("train_navrl_v2", text)
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("TRAIN_SEED = 71", script)
        self.assertIn("VALIDATION_SEED = 73", script)
        self.assertIn("TEST_SEED = 79", script)
        self.assertIn('"train_seed": args.train_seed', script)
        self.assertIn('"validation_seed": args.validation_seed', script)
        self.assertIn('"test_seed": args.test_seed', script)
        self.assertIn('os.environ.setdefault("NAVRL_PLACEMENT_MODE", "navrl_band")', script)

    def test_preflight_does_not_initialize_isaac_gym(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--preflight"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("PREFLIGHT PASS", completed.stdout)
        self.assertNotIn("Isaac Gym", completed.stdout + completed.stderr)

    def test_perfect_scores_produce_perfect_detection_metrics(self):
        gate = load_gate_module()
        mask = torch.zeros(4, 4, 4, dtype=torch.bool)
        mask[0, 1, 1:3] = True  # visible, partial, small, far
        mask[3, 2, 1:3] = True  # another visible frame
        scores = torch.where(mask, torch.full(mask.shape, 0.9), torch.full(mask.shape, 0.1))
        dataset = {
            "mask": mask,
            "depth": torch.full(mask.shape, 8.0, dtype=torch.float16),
            "planned_present": torch.tensor([True, False, True, True]),
            "forced_occlusion": torch.tensor([True, False, True, False]),
            "center_range_m": torch.tensor([15.0, 8.0, 9.0, 10.0]),
        }
        metrics = gate.detector_metrics(scores, dataset, 0.55, 2, 10.0)
        self.assertEqual(metrics["frame_precision"], 1.0)
        self.assertEqual(metrics["frame_recall"], 1.0)
        self.assertEqual(metrics["pixel_iou"], 1.0)
        self.assertEqual(metrics["absent_false_positive_rate"], 0.0)
        self.assertEqual(metrics["fully_occluded_false_positive_rate"], 0.0)
        self.assertEqual(metrics["partial_occlusion_recall"], 1.0)
        self.assertEqual(metrics["small_target_recall"], 1.0)
        self.assertEqual(metrics["far_14_20m_recall"], 1.0)
        self.assertEqual(metrics["bearing_mae_deg"], 0.0)
        self.assertEqual(metrics["range_mae_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
