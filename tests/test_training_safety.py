import importlib.util
from pathlib import Path
import unittest

import torch
import yaml


_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _ROOT / "aerial_gym/rl_training/rl_games/training_safety.py"
_SPEC = importlib.util.spec_from_file_location("training_safety_standalone", _MODULE_PATH)
_SAFETY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAFETY)


class TrainingSafetyTest(unittest.TestCase):
    def test_transformer_actor_learning_rate_is_fixed(self):
        path = _ROOT / "aerial_gym/rl_training/rl_games/ppo_navrl_perception_transformer.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["params"]["config"]
        self.assertEqual(float(config["learning_rate"]), 1e-4)
        self.assertEqual(config["lr_schedule"], "None")

    def test_first_nonfinite_metric_is_reported(self):
        result = _SAFETY.first_nonfinite_training_value(
            (("ppo/a_loss", [torch.tensor(0.1), torch.tensor(float("nan"))]),),
            (),
        )
        self.assertEqual(result, "ppo/a_loss[1]")

    def test_first_nonfinite_parameter_is_reported(self):
        result = _SAFETY.first_nonfinite_training_value(
            (("ppo/a_loss", [torch.tensor(0.1)]),),
            (("actor.weight", torch.tensor([1.0, float("inf")])),),
        )
        self.assertEqual(result, "model.actor.weight")

    def test_mean_reward_fail_fast_is_independent_of_collapse_guard(self):
        self.assertTrue(_SAFETY.is_finite_training_value(torch.tensor(38.7)))
        self.assertFalse(_SAFETY.is_finite_training_value(float("nan")))
        self.assertFalse(_SAFETY.is_finite_training_value(torch.tensor(float("inf"))))

    def test_checkpoint_learning_rate_is_overridden(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-2)
        _SAFETY.reset_optimizer_learning_rate(optimizer, 1e-4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_density_capture_guard_environment_overrides(self):
        config = {
            "enable": True,
            "window_epochs": 50,
            "min_epochs_at_density": 100,
            "min_peak_capture": 0.5,
            "drop_absolute": 0.25,
            "patience_epochs": 25,
        }
        result = _SAFETY.apply_density_capture_guard_overrides(
            config,
            {
                "NAVRL_DENSITY_GUARD_WINDOW_EPOCHS": "20",
                "NAVRL_DENSITY_GUARD_MIN_EPOCHS": "40",
                "NAVRL_DENSITY_GUARD_MIN_PEAK": "0.55",
                "NAVRL_DENSITY_GUARD_DROP": "0.10",
                "NAVRL_DENSITY_GUARD_PATIENCE": "10",
            },
        )
        self.assertEqual(result["window_epochs"], 20)
        self.assertEqual(result["min_epochs_at_density"], 40)
        self.assertEqual(result["min_peak_capture"], 0.55)
        self.assertEqual(result["drop_absolute"], 0.10)
        self.assertEqual(result["patience_epochs"], 10)
        self.assertEqual(config["drop_absolute"], 0.25)

    def test_density_capture_guard_rejects_invalid_rate(self):
        with self.assertRaises(ValueError):
            _SAFETY.apply_density_capture_guard_overrides(
                {"enable": True},
                {"NAVRL_DENSITY_GUARD_DROP": "1.5"},
            )


if __name__ == "__main__":
    unittest.main()
