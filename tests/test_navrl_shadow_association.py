import importlib.util
from pathlib import Path
import unittest

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym/task/navrl_task/shadow_association.py"
)
_SPEC = importlib.util.spec_from_file_location("navrl_shadow_association_for_test", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
connected_components = _MOD.connected_components
component_candidates = _MOD.component_candidates


class ShadowAssociationTests(unittest.TestCase):
    def test_two_separate_blobs_get_two_labels(self):
        mask = torch.zeros(1, 8, 10, dtype=torch.bool)
        mask[0, 1:3, 1:3] = True          # blob A: 4 px
        mask[0, 5:8, 6:9] = True          # blob B: 9 px
        labels = connected_components(mask)
        fg = labels[mask]
        self.assertEqual(len(torch.unique(fg)), 2)
        self.assertTrue((labels[~mask] == 0).all())

    def test_l_shaped_component_is_one_label(self):
        mask = torch.zeros(1, 12, 12, dtype=torch.bool)
        mask[0, 2, 2:9] = True
        mask[0, 2:9, 2] = True            # touching arm -> single 4-connected component
        labels = connected_components(mask)
        self.assertEqual(len(torch.unique(labels[mask])), 1)

    def test_diagonal_pixels_are_separate_under_4_connectivity(self):
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[0, 1, 1] = True
        mask[0, 2, 2] = True
        labels = connected_components(mask)
        self.assertEqual(len(torch.unique(labels[mask])), 2)

    def test_candidates_ranked_by_count_with_correct_centroids(self):
        mask = torch.zeros(2, 8, 10, dtype=torch.bool)
        depth = torch.full((2, 8, 10), 5.0)
        mask[0, 1:3, 1:3] = True                  # 4 px, centroid (u=1.5, v=1.5)
        mask[0, 5:8, 6:9] = True                  # 9 px, centroid (u=7, v=6)
        depth[0, 5:8, 6:9] = 11.0
        cand = component_candidates(mask, depth, top_k=3)
        self.assertEqual(int(cand["num_candidates"][0]), 2)
        self.assertEqual(float(cand["count"][0, 0]), 9.0)   # largest first
        self.assertAlmostEqual(float(cand["u"][0, 0]), 7.0, places=5)
        self.assertAlmostEqual(float(cand["v"][0, 0]), 6.0, places=5)
        self.assertAlmostEqual(float(cand["depth"][0, 0]), 11.0, places=5)
        self.assertEqual(float(cand["count"][0, 1]), 4.0)
        self.assertAlmostEqual(float(cand["u"][0, 1]), 1.5, places=5)
        # env 1 is empty: zero candidates, zero counts
        self.assertEqual(int(cand["num_candidates"][1]), 0)
        self.assertEqual(float(cand["count"][1].sum()), 0.0)

    def test_top_k_truncates_but_num_candidates_reports_all(self):
        mask = torch.zeros(1, 6, 20, dtype=torch.bool)
        for i in range(5):
            mask[0, 1:3, 4 * i : 4 * i + 2] = True  # five 4-px blobs, separated
        cand = component_candidates(mask, torch.ones(1, 6, 20), top_k=3)
        self.assertEqual(int(cand["num_candidates"][0]), 5)
        self.assertEqual(int((cand["count"][0] > 0).sum()), 3)

    def test_read_only_inputs(self):
        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[0, 2:4, 2:4] = True
        depth = torch.rand(1, 8, 8)
        mask_c, depth_c = mask.clone(), depth.clone()
        component_candidates(mask, depth, top_k=2)
        self.assertTrue(torch.equal(mask, mask_c))
        self.assertTrue(torch.equal(depth, depth_c))


if __name__ == "__main__":
    unittest.main()
