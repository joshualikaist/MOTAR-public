"""Contracts for the fresh physical-lineage non-overlap bar sampler."""

import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "aerial_gym/rl_training/rl_games"


def placement_worker():
    import aerial_gym  # Isaac Gym must load before torch, in a clean child interpreter.
    import torch
    from aerial_gym.env_manager.asset_manager import AssetManager

    def manager(envs, bars, arena=40.0, clearance=0.45, attempts=128):
        obj = AssetManager.__new__(AssetManager)
        obj.placement_surface_clearance = clearance
        obj.placement_candidate_batch_size = 32
        obj.placement_attempts_before_relax = attempts
        generator = torch.Generator().manual_seed(827 + bars)
        obj.asset_collision_half_extents = torch.empty(envs, bars, 3)
        obj.asset_collision_half_extents[:, :, :2] = 0.2 + 0.2 * torch.rand(
            envs, bars, 2, generator=generator
        )
        obj.asset_collision_half_extents[:, :, 2] = 1.5
        obj.env_bounds_min = torch.zeros(envs, bars, 3)
        obj.env_bounds_max = torch.zeros(envs, bars, 3)
        obj.env_bounds_max[:, :, :2] = arena
        obj.env_bounds_max[:, :, 2] = 3.0
        obj.asset_min_state_ratio = torch.zeros(envs, bars, 13)
        obj.asset_max_state_ratio = torch.ones(envs, bars, 13)
        return obj

    envs, bars, clearance = 16, 300, 0.45
    obj = manager(envs, bars, clearance=clearance)
    torch.manual_seed(20260827)
    placed = obj._footprint_clearance_xy_spacing(
        torch.zeros(envs, bars, 3), bars, torch.arange(envs)
    )
    xy = placed[:, :, :2]
    support = torch.linalg.vector_norm(obj.asset_collision_half_extents[:, :, :2], dim=2)
    required = support[:, :, None] + support[:, None, :] + clearance
    eye = torch.eye(bars, dtype=torch.bool)[None]
    delta = xy[:, :, None, :] - xy[:, None, :, :]
    squared_margin = (delta * delta).sum(dim=3) - required * required
    payload = {
        "min_squared_margin": float(squared_margin.masked_fill(eye, float("inf")).amin()),
        "inside_bounds": bool(
            (xy[:, :, 0] >= support).all() and (xy[:, :, 0] <= 40.0 - support).all()
            and (xy[:, :, 1] >= support).all() and (xy[:, :, 1] <= 40.0 - support).all()
        ),
    }
    invalid = manager(1, 2)
    invalid.asset_collision_half_extents[0, 1, 0] = float("nan")
    try:
        invalid._footprint_clearance_xy_spacing(torch.zeros(1, 2, 3), 2, torch.tensor([0]))
        payload["invalid_rejected"] = False
    except RuntimeError:
        payload["invalid_rejected"] = True
    impossible = manager(1, 3, arena=1.0, attempts=1)
    impossible.asset_collision_half_extents[:, :, :2] = 0.4
    try:
        impossible._footprint_clearance_xy_spacing(torch.zeros(1, 3, 3), 3, torch.tensor([0]))
        payload["impossible_rejected"] = False
    except RuntimeError:
        payload["impossible_rejected"] = True
    print("PLACEMENT_JSON=" + json.dumps(payload, sort_keys=True))


def run_worker():
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    # Isaac Gym builds gymtorch through ninja during import.  Bind the child to
    # the selected interpreter's toolchain instead of inheriting the caller's PATH.
    env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--placement-worker"],
        cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(result.stdout)
    line = next(line for line in result.stdout.splitlines() if line.startswith("PLACEMENT_JSON="))
    return json.loads(line.split("=", 1)[1])


class TestFootprintClearancePlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = run_worker()

    def test_300_bars_have_no_overlap_or_merge_fallback(self):
        self.assertGreaterEqual(self.payload["min_squared_margin"], -1e-5)
        self.assertTrue(self.payload["inside_bounds"])

    def test_invalid_collision_geometry_fails_closed(self):
        self.assertTrue(self.payload["invalid_rejected"])

    def test_impossible_arena_refuses_instead_of_merging(self):
        self.assertTrue(self.payload["impossible_rejected"])

    def test_physical_routed_launcher_pins_new_fresh_contract(self):
        env = dict(os.environ)
        env["NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY"] = "1"
        result = subprocess.run(
            [str(RL / "train_navrl_physical_routed_fresh.sh")],
            cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("robot=navrl_ref5in_v2_quad", result.stdout)
        self.assertIn("placement=footprint_clearance", result.stdout)
        self.assertIn("surface=0.45m overlap_fallback=off", result.stdout)


if __name__ == "__main__":
    if "--placement-worker" in sys.argv:
        placement_worker()
    else:
        unittest.main()
