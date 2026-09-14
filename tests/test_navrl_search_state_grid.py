import importlib.util
import math
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_search_state.py"
SPEC = importlib.util.spec_from_file_location("navrl_search_state_grid_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SearchGrid = MODULE.SearchGrid


def fixture(state="belief"):
    grid = SearchGrid(
        1,
        (torch.tensor([-20.0, -20.0]), torch.tensor([20.0, 20.0])),
        device="cpu",
        search_state=state,
        camera_hfov_rad=math.radians(87.0),
        camera_range_m=20.0,
        depth_far_m=10.0,
    )
    pose = torch.tensor([[0.0, 0.0, 1.0]])
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    depth = torch.full((1, 6, 81), 10.0)
    tracker_state = torch.zeros(1, 6)
    tracker_cov = torch.eye(6).unsqueeze(0)
    inactive = torch.zeros(1, dtype=torch.bool)
    return grid, pose, quat, depth, tracker_state, tracker_cov, inactive


class SearchGridCoverageTest(unittest.TestCase):
    def test_only_unoccluded_frustum_cells_are_marked_and_mass_accounts(self):
        grid, pose, quat, depth, state, cov, inactive = fixture("coverage")
        grid.update(pose, quat, depth, state, cov, inactive)
        centres = grid.cell_centres[0]
        viewed = grid.viewed[0].reshape(-1)
        angles = torch.atan2(centres[:, 1], centres[:, 0])
        ranges = centres.norm(dim=1)
        self.assertTrue(bool(viewed.any()))
        self.assertFalse(bool(viewed[angles.abs() > math.radians(43.5)].any()))
        self.assertFalse(bool(viewed[ranges > 20.0].any()))

        features = grid.features(pose, quat)
        sector_unviewed_mass = features[0, 4::2]
        self.assertAlmostEqual(
            float(sector_unviewed_mass.sum()),
            float((~grid.viewed).float().mean()),
            places=6,
        )

        # Put a 2 m obstacle into the exact column of a previously visible 5,1 m cell.
        blocked_grid, pose, quat, depth, state, cov, inactive = fixture("coverage")
        cell = torch.tensor([5.0, 1.0])
        angle = math.atan2(float(cell[1]), float(cell[0]))
        column = round((math.radians(87.0) * 0.5 - angle) / math.radians(87.0) * 80)
        depth[:, :, column] = 2.0
        blocked_grid.update(pose, quat, depth, state, cov, inactive)
        index = int(((blocked_grid.cell_centres[0] - cell).abs().sum(dim=1)).argmin())
        self.assertFalse(bool(blocked_grid.viewed[0].reshape(-1)[index]))

    def test_belief_normalizes_penalizes_viewed_cells_and_diffuses_symmetrically(self):
        grid, pose, quat, depth, state, cov, inactive = fixture("belief")
        grid.update(pose, quat, depth, state, cov, inactive)
        self.assertAlmostEqual(float(grid.belief.sum()), 1.0, places=6)
        viewed = grid.viewed[0]
        self.assertLess(
            float(grid.belief[0][viewed].mean()),
            float(grid.belief[0][~viewed].mean()),
        )
        self.assertGreaterEqual(float(grid.normalized_entropy()), 0.0)
        self.assertLessEqual(float(grid.normalized_entropy()), 1.0)

        impulse = torch.zeros_like(grid.belief)
        impulse[0, 10, 10] = 1.0
        diffused = grid._diffuse(impulse)[0]
        self.assertAlmostEqual(float(diffused[9, 10]), float(diffused[11, 10]), places=8)
        self.assertAlmostEqual(float(diffused[10, 9]), float(diffused[10, 11]), places=8)
        self.assertAlmostEqual(float(diffused[9, 10]), float(diffused[10, 9]), places=8)

    def test_active_to_inactive_reinjects_last_kf_location(self):
        grid, pose, quat, depth, state, cov, inactive = fixture("belief")
        state[0, :2] = torch.tensor([7.0, -3.0])
        cov *= 0.05
        active = torch.ones(1, dtype=torch.bool)
        grid.update(pose, quat, depth, state, cov, active)
        grid.belief.fill_(1.0 / grid.num_cells)
        grid.update(pose, quat, depth, torch.zeros_like(state), torch.eye(6).unsqueeze(0), inactive)
        maximum = int(grid.belief.reshape(-1).argmax())
        centre = grid.cell_centres[0, maximum]
        self.assertLessEqual(float((centre - state[0, :2]).norm()), math.sqrt(2.0) + 1e-6)


if __name__ == "__main__":
    unittest.main()
