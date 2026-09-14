import importlib.util
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_search_state.py"
SPEC = importlib.util.spec_from_file_location("navrl_search_state_mask_test", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SearchMaskTest(unittest.TestCase):
    def test_force_invalid_exact_sentinels(self):
        grid = M.SearchGrid(
            2,
            ([-20.0, -20.0], [20.0, 20.0]),
            search_state="belief",
        )
        pos = torch.zeros(2, 3)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        features = grid.features(pos, quat, force_invalid=True)
        self.assertTrue(torch.equal(features[:, :28], torch.zeros(2, 28)))
        self.assertTrue(
            torch.equal(features[:, 28:40], torch.full((2, 12), 1.0 / 12.0))
        )
        self.assertTrue(torch.equal(features[:, 40:52], torch.zeros(2, 12)))
        self.assertTrue(torch.equal(features[:, 52], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
