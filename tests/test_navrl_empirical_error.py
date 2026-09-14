import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aerial_gym/task/navrl_task/navrl_empirical_error.py"
SPEC = importlib.util.spec_from_file_location("navrl_empirical_error_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    rows = [
        {"probabilities": [.8, .1, .1], "reference_dt_s": .1},
        {"probabilities": [.2, .7, .1], "reference_dt_s": .1},
        {"probabilities": [.3, .2, .5], "reference_dt_s": .1},
    ]
    bins = []
    for target_bin in (1, 2, 3):
        by_destination = {str(dest): rows for dest in (1, 2, 3)}
        bins.append({
            "target_bin": target_bin,
            "initial_state_probabilities": [.6, .25, .15],
            "transition_rows": rows,
            "transition_rows_by_destination_bin": by_destination,
            "offsets": {
                "HIT": {"joint_du_dv_normalized_samples": [[.01, -.02], [0., 0.]]},
                "FALSE_LOCK": {"joint_du_dv_normalized_samples": [[.4, -.3]]},
            },
            "latency_ms_samples": [40., 60., 121.],
        })
    return {
        "schema_version": 1, "test_used": False,
        "states": ["HIT", "FALSE_LOCK", "NO_LOCK"],
        "size_edges_px": [8., 16., 32., 64.],
        "fallback_bin_by_raw_bin": [1, 1, 2, 3, 3],
        "latency_ms_samples_by_source_bin": {
            "1": [40., 60., 121.], "2": [40., 60.], "3": [40., 60.]
        },
        "bins": bins,
    }


class EmpiricalErrorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "model.json"
        self.path.write_text(json.dumps(fixture(), sort_keys=True), encoding="utf-8")
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def make(self, seed=7):
        return MODULE.EmpiricalPerceptionError(self.path, self.sha, 5, "cpu", .1, seed)

    def test_hash_is_mandatory_and_fail_closed(self):
        with self.assertRaises(RuntimeError):
            MODULE.EmpiricalPerceptionError(self.path, "", 1, "cpu", .1, 1)
        with self.assertRaises(RuntimeError):
            MODULE.EmpiricalPerceptionError(self.path, "0" * 64, 1, "cpu", .1, 1)

    def test_fixed_seed_is_identical_and_different_seed_changes(self):
        size = torch.tensor([7.99, 8., 16., 32., 64.])
        eligible = torch.ones(5, dtype=torch.bool)
        left, right, other = self.make(7), self.make(7), self.make(8)
        trace_left, trace_right, trace_other = [], [], []
        for _ in range(20):
            trace_left.append(tuple(t.clone() for t in left.sample(size, eligible)))
            trace_right.append(tuple(t.clone() for t in right.sample(size, eligible)))
            trace_other.append(tuple(t.clone() for t in other.sample(size, eligible)))
        for a, b in zip(trace_left, trace_right):
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))
        self.assertTrue(any(
            not torch.equal(x, y)
            for a, b in zip(trace_left, trace_other) for x, y in zip(a, b)
        ))
        # Boundary convention is [lower, upper), then unsupported end bins clamp explicitly.
        self.assertEqual(left.last_source_bin.tolist(), [1, 1, 2, 3, 3])

    def test_reset_reinitializes_only_requested_chains(self):
        sampler = self.make()
        sampler.sample(torch.full((5,), 20.), torch.ones(5, dtype=torch.bool))
        before = sampler.state.clone()
        sampler.reset_idx(torch.tensor([1, 3]))
        self.assertEqual(sampler.state[[1, 3]].tolist(), [-1, -1])
        self.assertTrue(torch.equal(sampler.state[[0, 2, 4]], before[[0, 2, 4]]))

    def test_latency_uses_half_up_whole_observation_steps(self):
        sampler = self.make()
        self.assertEqual(sampler.max_latency_steps, 1)
        _, _, delay = sampler.sample(
            torch.full((5,), 20.), torch.ones(5, dtype=torch.bool)
        )
        for milliseconds, steps in zip(sampler.last_latency_ms.tolist(), delay.tolist()):
            self.assertEqual(steps, int(milliseconds / 100. + .5))

    def test_cadence_conversion_preserves_reference_row(self):
        row = [.8, .1, .1]
        self.assertEqual(MODULE.cadence_adjusted_row(row, 0, .1, .1), row)
        doubled = MODULE.cadence_adjusted_row(row, 0, .1, .2)
        self.assertAlmostEqual(doubled[0], .64)
        self.assertAlmostEqual(sum(doubled), 1.)


class SourceContractTest(unittest.TestCase):
    def test_detector_channel_publishes_bbox_and_false_lock_does_not_carve(self):
        detector = (ROOT / "aerial_gym/task/navrl_task/navrl_detector.py").read_text()
        perception = (ROOT / "aerial_gym/task/navrl_task/navrl_perception.py").read_text()
        self.assertIn('"bbox": self.detect_bbox', detector)
        self.assertIn("map_visible = clean_visible & (empirical_state == 0)", perception)
        self.assertIn("empirical_offset[:, 0] * self.detect_width", perception)
        self.assertIn("latency_steps=empirical_latency_steps", perception)


if __name__ == "__main__":
    unittest.main()

