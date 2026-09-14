import importlib.util
import inspect
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_search_state.py"
SPEC = importlib.util.spec_from_file_location("navrl_search_state_leak_test", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SearchStateLeakTest(unittest.TestCase):
    def test_update_has_no_target_or_semantic_input(self):
        parameters = tuple(inspect.signature(M.SearchGrid.update).parameters)
        self.assertEqual(
            parameters,
            ("self", "drone_pos_w", "quat", "depth", "tracker_state", "tracker_cov", "tracker_active"),
        )
        self.assertFalse(any("target" in name or "semantic" in name for name in parameters))

    def test_identical_sensor_track_inputs_are_bit_identical(self):
        # Two arbitrary simulator target locations are intentionally unused: the public API has no
        # slot through which either value could affect the grid.
        target_a = torch.tensor([18.0, 4.0, 1.0])
        target_b = torch.tensor([-17.0, -9.0, 1.0])
        self.assertFalse(torch.equal(target_a, target_b))
        grids = [
            M.SearchGrid(1, ([-20.0, -20.0], [20.0, 20.0]), search_state="belief")
            for _ in range(2)
        ]
        pos = torch.tensor([[1.0, -2.0, 1.0]])
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        depth = torch.full((1, 4, 17), 20.0)
        state = torch.tensor([[4.0, 3.0, 1.0, 0.2, 0.0, 0.0]])
        cov = torch.eye(6).unsqueeze(0)
        active = torch.ones(1, dtype=torch.bool)
        outputs = []
        for grid in grids:
            grid.update(pos, quat, depth, state, cov, active)
            outputs.append(grid.features(pos, quat))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


if __name__ == "__main__":
    unittest.main()
