"""CPU-only tests for the frozen-policy symmetric-corridor diagnostic."""

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


os.environ.update(
    {
        "NAVRL_LIDAR_HBEAMS": "72",
        "NAVRL_LIDAR_VBEAMS": "4",
        "NAVRL_MAX_OBSTACLES": "8",
    }
)

ROOT = Path(__file__).resolve().parents[1]
# Load the two pure-torch files without importing aerial_gym.__init__, which eagerly imports Isaac
# Gym.  This keeps the test genuinely CPU-only and follows test_ppo_update_safety.py's convention.
for name in ("aerial_gym", "aerial_gym.rl_training", "aerial_gym.rl_training.rl_games"):
    sys.modules.setdefault(name, types.ModuleType(name))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load(
    "aerial_gym.rl_training.rl_games.ppo_update_safety",
    ROOT / "aerial_gym/rl_training/rl_games/ppo_update_safety.py",
)
PROBE = load(
    "aerial_gym.rl_training.rl_games.navrl_mode_probe",
    ROOT / "aerial_gym/rl_training/rl_games/navrl_mode_probe.py",
)
ModeProbeRecorder = PROBE.ModeProbeRecorder
build_probe_observations = PROBE.build_probe_observations


class ModeProbeTest(unittest.TestCase):
    CONTRACT = {
        "reflection_pair_max_abs": {
            "symmetric_lr_to_rl": 0.0,
            "left_lr_to_right_rl": 0.0,
            "left_rl_to_right_lr": 0.0,
        }
    }

    def test_fixture_is_physical_and_reflection_pairs_are_exact(self):
        reference = torch.zeros(3, 898)
        fixtures, contract = build_probe_observations(reference, offset_deg=5.0)
        self.assertEqual(
            set(fixtures),
            {"symmetric_lr", "symmetric_rl", "left_lr", "left_rl", "right_lr", "right_rl"},
        )
        self.assertEqual(tuple(fixtures["left_lr"].shape), (1, 898))
        self.assertEqual(contract, self.CONTRACT)
        # The two centre arms have an identical physically symmetric static scan.
        self.assertTrue(torch.equal(fixtures["symmetric_lr"][:, :288], fixtures["symmetric_rl"][:, :288]))
        scan = fixtures["symmetric_lr"][:, :288].view(1, 4, 72)
        reflect_index = (-torch.arange(72)) % 72
        self.assertTrue(torch.equal(scan, scan.index_select(2, reflect_index)))
        # Critical regression: centred obstacle tokens must remain two opposite non-zero bearings,
        # never the old arithmetic midpoint's duplicate y=0 obstacles.
        tokens_lr = fixtures["symmetric_lr"][:, 288 : 288 + 5 * 8 * 12].view(1, 5, 8, 12)
        y = tokens_lr[0, 0, :2, 1]
        self.assertGreater(float(y[0]), 0.0)
        self.assertLess(float(y[1]), 0.0)
        self.assertAlmostEqual(float(y.sum()), 0.0, places=6)
        self.assertFalse(bool((y == 0).any()))

    def _outputs(self, symmetric, left, right, sigma=0.4, *, slot_delta=0.0):
        payload = {}
        actions = {
            "symmetric_lr": symmetric,
            "symmetric_rl": [symmetric[0], symmetric[1] + slot_delta, symmetric[2], symmetric[3]],
            "left_lr": left,
            "left_rl": [left[0], left[1] + slot_delta, left[2], left[3]],
            "right_lr": [right[0], right[1] + slot_delta, right[2], right[3]],
            "right_rl": right,
        }
        for arm, action in actions.items():
            tensor = torch.tensor([action], dtype=torch.float32)
            payload[arm] = {
                "action": tensor,
                "mu": tensor.clone(),
                "sigma": torch.full_like(tensor, sigma),
            }
        return payload

    def test_preregistered_positive_pattern_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ModeProbeRecorder(Path(directory) / "probe.json", max_velocity_mps=2.5)
            recorder.record(
                self._outputs(
                    [0.02, 0.01, 0.0, 0.0],
                    [0.40, 0.40, 0.0, 0.20],
                    [0.40, -0.40, 0.0, -0.20],
                ),
                self.CONTRACT,
            )
            payload = recorder.write()
            self.assertEqual(
                payload["verdict"], "MODE_AVERAGING_SUPPORTED_IN_SYNTHETIC_POLICY_SCREEN"
            )
            self.assertTrue(all(payload["checks"].values()))
            self.assertEqual(json.loads((Path(directory) / "probe.json").read_text())["samples"], 1)

    def test_policy_chirality_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ModeProbeRecorder(Path(directory) / "probe.json", max_velocity_mps=2.5)
            recorder.record(
                self._outputs(
                    [0.02, 0.01, 0.0, 0.0],
                    [0.40, 0.40, 0.0, 0.20],
                    [0.40, 0.35, 0.0, 0.20],
                ),
                self.CONTRACT,
            )
            payload = recorder.payload()
            self.assertEqual(payload["verdict"], "INCONCLUSIVE_POLICY_CHIRALITY")
            self.assertFalse(payload["checks"]["policy_reflection_quality"])

    def test_slot_order_sensitivity_fails_closed_before_mechanism(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ModeProbeRecorder(Path(directory) / "probe.json", max_velocity_mps=2.5)
            recorder.record(
                self._outputs(
                    [0.02, 0.0, 0.0, 0.0],
                    [0.40, 0.40, 0.0, 0.20],
                    [0.40, -0.40, 0.0, -0.20],
                    slot_delta=0.30,
                ),
                self.CONTRACT,
            )
            payload = recorder.payload()
            self.assertEqual(payload["verdict"], "INCONCLUSIVE_SLOT_ORDER_SENSITIVITY")
            self.assertFalse(payload["checks"]["slot_permutation_quality"])

    def test_schema_drift_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "base structured schema"):
            build_probe_observations(torch.zeros(1, 946))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text("occupied", encoding="utf-8")
            recorder = ModeProbeRecorder(path, max_velocity_mps=2.5)
            recorder.record(
                self._outputs([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
                self.CONTRACT,
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                recorder.write()


if __name__ == "__main__":
    unittest.main()
