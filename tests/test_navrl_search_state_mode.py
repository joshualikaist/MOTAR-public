import importlib.util
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_search_state.py"
SPEC = importlib.util.spec_from_file_location("navrl_search_state_mode_test", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SearchModeTest(unittest.TestCase):
    def setUp(self):
        self.grid = M.SearchGrid(
            1,
            ([-20.0, -20.0], [20.0, 20.0]),
            search_state="coverage",
            step_dt=0.1,
        )
        self.pos = torch.tensor([[0.0, 0.0, 1.0]])
        self.quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.depth = torch.full((1, 2, 9), 20.0)
        self.state = torch.zeros(1, 6)
        self.cov = torch.eye(6).unsqueeze(0)

    def update(self, active):
        self.grid.update(
            self.pos, self.quat, self.depth, self.state, self.cov,
            torch.tensor([active]),
        )
        return self.grid.features(self.pos, self.quat)[0, :4]

    def test_never_tracked_stale_reset_sequence(self):
        never = self.update(False)
        self.assertTrue(torch.equal(never[:3], torch.tensor([1.0, 0.0, 0.0])))
        self.assertGreater(float(never[3]), 0.0)
        tracked = self.update(True)
        self.assertTrue(torch.equal(tracked[:3], torch.tensor([0.0, 1.0, 0.0])))
        self.assertEqual(float(tracked[3]), 0.0)
        stale = self.update(False)
        self.assertTrue(torch.equal(stale[:3], torch.tensor([0.0, 0.0, 1.0])))
        self.grid.blind_steps[:] = 100000
        self.assertEqual(float(self.grid.features(self.pos, self.quat)[0, 3]), 1.0)
        self.grid.reset(torch.tensor([0]))
        reset = self.grid.features(self.pos, self.quat)[0, :4]
        self.assertTrue(torch.equal(reset, torch.tensor([1.0, 0.0, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
