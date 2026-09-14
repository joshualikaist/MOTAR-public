"""CPU contract tests for the offline instance adapter. No Isaac, SAM, or PPO."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_MODULE_PATH = (
    Path(__file__).parents[1]
    / "aerial_gym/task/navrl_task/navrl_instance_adapter.py"
)
_SPEC = importlib.util.spec_from_file_location("navrl_instance_adapter_standalone", _MODULE_PATH)
ADAPTER = importlib.util.module_from_spec(_SPEC)
sys.modules["navrl_instance_adapter_standalone"] = ADAPTER
_SPEC.loader.exec_module(ADAPTER)


class InstanceAdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.fixture = ADAPTER.two_blob_fixture()
        self.rgb = self.fixture["rgb"]
        self.depth = self.fixture["depth"]

    def test_stub_keeps_two_instances(self):
        instances = ADAPTER.stub_detect(self.rgb, self.depth)
        self.assertEqual(len(instances), 2)
        depths = sorted(inst.depth_median for inst in instances)
        self.assertAlmostEqual(depths[0], 4.0, places=5)
        self.assertAlmostEqual(depths[1], 8.0, places=5)

    def test_union_centroid_is_a_ghost_between_blobs(self):
        score = ADAPTER.colour_score(self.rgb)
        mask = (score >= ADAPTER.DEFAULT_PIXEL_THRESHOLD) & (
            self.depth < ADAPTER.DEFAULT_MAX_DEPTH
        )
        collapsed = ADAPTER.union_collapse_centroid(mask, self.depth)
        ghost = (collapsed["u"], collapsed["v"])
        self.assertFalse(ADAPTER.point_in_bbox(ghost, self.fixture["left_bbox"]))
        self.assertFalse(ADAPTER.point_in_bbox(ghost, self.fixture["right_bbox"]))
        left_u = 0.5 * (self.fixture["left_bbox"][0] + self.fixture["left_bbox"][2])
        right_u = 0.5 * (self.fixture["right_bbox"][0] + self.fixture["right_bbox"][2])
        self.assertGreater(ghost[0], min(left_u, right_u))
        self.assertLess(ghost[0], max(left_u, right_u))
        self.assertAlmostEqual(collapsed["range"], 6.0, places=5)

    def test_close_scores_are_ambiguous_and_do_not_lock(self):
        instances = ADAPTER.stub_detect(self.rgb, self.depth)
        decision = ADAPTER.associate_and_decide(instances, ambiguous_margin=0.10)
        self.assertEqual(decision.status, ADAPTER.DECISION_AMBIGUOUS)
        self.assertIsNone(decision.selected_id)
        self.assertEqual(decision.n_instances, 2)

    def test_large_score_gap_selects_target(self):
        instances = list(ADAPTER.stub_detect(self.rgb, self.depth))
        ranked = sorted(instances, key=lambda inst: inst.uv[0])
        left, right = ranked
        left = ADAPTER.InstanceDetection(
            instance_id=left.instance_id,
            mask=left.mask,
            bbox=left.bbox,
            score=0.91,
            depth_median=left.depth_median,
            uv=left.uv,
        )
        right = ADAPTER.InstanceDetection(
            instance_id=right.instance_id,
            mask=right.mask,
            bbox=right.bbox,
            score=0.40,
            depth_median=right.depth_median,
            uv=right.uv,
        )
        decision = ADAPTER.associate_and_decide([left, right], ambiguous_margin=0.10)
        self.assertEqual(decision.status, ADAPTER.DECISION_TARGET)
        self.assertEqual(decision.selected_id, left.instance_id)

    def test_empty_input_rejects(self):
        rgb = np.full((16, 16, 3), 0.2)
        depth = np.full((16, 16), 20.0)
        decision = ADAPTER.associate_and_decide(ADAPTER.stub_detect(rgb, depth))
        self.assertEqual(decision.status, ADAPTER.DECISION_REJECT)
        self.assertIsNone(decision.selected_id)

    def test_union_collapse_guard_rejects_k_gt_1(self):
        instances = ADAPTER.stub_detect(self.rgb, self.depth)
        with self.assertRaises(ADAPTER.UnionCollapseError):
            ADAPTER.forbid_union_collapse(instances)
        with self.assertRaises(ADAPTER.UnionCollapseError):
            ADAPTER.union_from_instances(instances, self.depth)
        ADAPTER.forbid_union_collapse(instances[:1])
        single = ADAPTER.union_from_instances(instances[:1], self.depth)
        self.assertGreater(single["count"], 0.0)

    def test_sam_backend_is_fail_closed(self):
        with self.assertRaises(ADAPTER.SamBackendNotInstalled):
            ADAPTER.run_backend("sam", self.rgb, self.depth)
        spec = ADAPTER.sam_backend_spec()
        self.assertEqual(spec["transport"], "npz")
        self.assertIn("separate", spec["process"])

    def test_adapter_defaults_off(self):
        self.assertFalse(ADAPTER.adapter_enabled())

    def test_perception_module_does_not_import_adapter(self):
        source = (
            Path(__file__).parents[1]
            / "aerial_gym/task/navrl_task/navrl_perception.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("navrl_instance_adapter", source)


if __name__ == "__main__":
    unittest.main()
