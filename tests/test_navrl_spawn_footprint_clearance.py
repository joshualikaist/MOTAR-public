"""Footprint-aware spawn clearance for `NavRLTask._randomize_general_drone_spawn` (2026-08-27).

Why this exists.  The spawn sampler used to accept a candidate whose distance to the nearest bar
CENTRE was >= 0.65 m.  That test is blind to how big the bar is: the `bars_h3` pool spans XY
circumradius 0.3133..0.5465 m, so the same 0.65 m bought 0.34 m of real surface clearance at a
small bar and only 0.10 m at a large one, while the geometry audit inflates every bar by the
robot disk plus the tracking reserve (0.6498 m).  At 205 bars that mismatch put 27.45% of spawns
inside the inflated obstacle set
(results/navrl_v2_density_geometry_audit_2026-08-27/summary.md).

These cases therefore deliberately use SEVERAL distinct bar sizes: a scenario with one bar size
cannot distinguish the per-bar rule from a flat constant, and a regression to any flat centre
clearance must fail here.

CPU only: the task methods are extracted from the source with `ast` and executed against stub
objects.  Nothing imports Isaac Gym.

Run: PYTHONNOUSERSITE=1 python -m unittest tests.test_navrl_spawn_footprint_clearance
"""

import ast
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
import xml.etree.ElementTree as ET

import torch

REPO = Path(__file__).resolve().parents[1]
_TASK_PATH = REPO / "aerial_gym/task/navrl_task/navrl_task.py"
_TASK_TREE = ast.parse(_TASK_PATH.read_text(encoding="utf-8"), filename=str(_TASK_PATH))


def _exec_node(node, namespace):
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_TASK_PATH), "exec"), namespace)
    return namespace[node.name]


def _load_task_function(name, namespace=None):
    node = next(
        n for n in _TASK_TREE.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return _exec_node(node, {"math": math, "torch": torch, **(namespace or {})})


def _load_task_method(name, namespace=None):
    """Load one NavRLTask method as a free function taking an explicit stub `self`."""
    task_class = next(
        n for n in _TASK_TREE.body if isinstance(n, ast.ClassDef) and n.name == "NavRLTask"
    )
    node = next(
        n for n in task_class.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return _exec_node(
        node,
        {"math": math, "torch": torch, "Path": Path, "ET": ET, **(namespace or {})},
    )


def _load_task_constant(name):
    for node in _TASK_TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("navrl_task.py no longer defines %s" % name)


PROP_RADIUS_5IN_M = _load_task_constant("PROP_RADIUS_5IN_M")
_accepted = _load_task_function("_spawn_footprint_clearance_accepted")
_inflation_radius = _load_task_method(
    "_robot_spawn_inflation_radius_m", {"PROP_RADIUS_5IN_M": PROP_RADIUS_5IN_M}
)
_required_margin = _load_task_method("_spawn_required_surface_margin_m")
_randomize_spawn = _load_task_method(
    "_randomize_general_drone_spawn",
    {"_spawn_footprint_clearance_accepted": _accepted},
)

# The exact `bars_h3` pool extremes, recomputed from the URDFs the audit tool reads.
_POOL_DIR = REPO / "resources/models/environment_assets/bars_h3"

# The pre-fix acceptance rule, kept here so a regression to it is caught rather than described.
LEGACY_FLAT_CENTER_CLEARANCE_M = 0.65


def _pool_circumradii():
    radii = []
    for path in sorted(_POOL_DIR.glob("*.urdf")):
        box = ET.parse(str(path)).getroot().find(".//collision/geometry/box")
        w, d, _h = (float(v) for v in box.get("size").split())
        radii.append(0.5 * math.hypot(w, d))
    return radii


def _rectangle_surface_distance(points_xy, centers_xy, half_xy):
    """Exact axis-aligned rectangle surface distance, the quantity the geometry audit grids."""
    delta = (points_xy.unsqueeze(1) - centers_xy).abs() - half_xy
    return delta.clamp(min=0.0).norm(dim=2)


def _ref5in_stub(tracking_margin_m=0.45, urdf="quad_navrl_ref5in.urdf"):
    """Stub carrying exactly the live attributes the derivation reads."""
    stub = types.SimpleNamespace(
        sim_env=types.SimpleNamespace(
            robot_manager=types.SimpleNamespace(
                cfg=types.SimpleNamespace(
                    control_allocator_config=types.SimpleNamespace(
                        allocation_matrix=[
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0],
                            [-0.0777817, -0.0777817, 0.0777817, 0.0777817],
                            [-0.0777817, 0.0777817, 0.0777817, -0.0777817],
                            [-0.01, 0.01, -0.01, 0.01],
                        ]
                    ),
                    robot_asset=types.SimpleNamespace(
                        asset_folder=str(REPO / "resources/robots/quad"), file=urdf
                    ),
                )
            )
        ),
        tm=types.SimpleNamespace(physical_tracking_margin=tracking_margin_m),
    )
    # The extracted methods take `self` explicitly, so wire the one call between them by hand.
    stub._robot_spawn_inflation_radius_m = lambda: _inflation_radius(stub)
    return stub


class SpawnMarginDerivationTest(unittest.TestCase):
    def test_inflation_radius_comes_from_the_live_prop_tip_span(self):
        stub = _ref5in_stub()
        radius = _inflation_radius(stub)
        span = 2.0 * (0.0777817 + PROP_RADIUS_5IN_M)
        self.assertAlmostEqual(span, 0.2825634, places=7)
        self.assertAlmostEqual(radius, 0.5 * span * math.sqrt(2.0), places=12)
        self.assertAlmostEqual(radius, 0.199802, places=6)

    def test_v2_collision_box_wins_when_it_exceeds_the_tip_span(self):
        """quad_navrl_ref5in_v2.urdf's 0.283 m box is wider than the 0.2825634 m tip span."""
        v2 = REPO / "resources/robots/quad/quad_navrl_ref5in_v2.urdf"
        if not v2.is_file():
            self.skipTest("v2 robot asset is absent")
        radius = _inflation_radius(_ref5in_stub(urdf=v2.name))
        self.assertGreater(radius, _inflation_radius(_ref5in_stub()))
        self.assertAlmostEqual(radius, 0.5 * 0.283 * math.sqrt(2.0), places=12)

    def test_required_margin_is_robot_disk_plus_live_tracking_reserve(self):
        stub = _ref5in_stub(tracking_margin_m=0.45)
        self.assertAlmostEqual(_required_margin(stub), 0.199802 + 0.45, places=6)
        # The reserve is read live, not frozen: changing the config changes the margin.
        moved = _ref5in_stub(tracking_margin_m=0.60)
        self.assertAlmostEqual(_required_margin(moved), 0.199802 + 0.60, places=6)

    def test_margin_matches_the_geometry_audit_inflation(self):
        """The spawn predicate and the audit's free-space inflation must not drift apart."""
        spec = importlib.util.spec_from_file_location(
            "audit_navrl_v2_density_geometry_standalone",
            REPO / "tools/audit_navrl_v2_density_geometry.py",
        )
        audit = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = audit
        spec.loader.exec_module(audit)
        audit_inflation = 0.5 * audit.ROBOT_TIP_AABB * math.sqrt(2.0) + audit.TRACKING_RESERVE
        self.assertAlmostEqual(_required_margin(_ref5in_stub()), audit_inflation, places=9)
        self.assertAlmostEqual(
            audit.SPAWN_SURFACE_MARGIN_M, _required_margin(_ref5in_stub()), places=9
        )


class SpawnAcceptancePredicateTest(unittest.TestCase):
    def setUp(self):
        self.margin = 0.649802
        # Deliberately varied footprints: the pool's smallest and largest squares plus a
        # rectangular middle case.  One size alone could not tell a per-bar rule from a flat one.
        self.half = torch.tensor(
            [
                [[0.2215, 0.2215]],  # circumradius 0.3133 (pool minimum)
                [[0.3864, 0.3864]],  # circumradius 0.5465 (pool maximum)
                [[0.4000, 0.1500]],  # circumradius 0.4272 (non-square)
            ]
        )
        self.radius = self.half.norm(dim=2).squeeze(1)

    def test_pool_extremes_are_the_ones_being_tested(self):
        radii = _pool_circumradii()
        self.assertAlmostEqual(min(radii), 0.3133, places=3)
        self.assertAlmostEqual(max(radii), 0.5465, places=3)

    def test_acceptance_threshold_scales_with_each_bar(self):
        centers = torch.zeros((3, 1, 2))
        eps = 1e-4
        just_inside = (self.radius + self.margin - eps).unsqueeze(1) * torch.tensor([[1.0, 0.0]])
        just_outside = (self.radius + self.margin + eps).unsqueeze(1) * torch.tensor([[1.0, 0.0]])
        self.assertEqual(
            _accepted(just_inside, centers, self.half, self.margin).tolist(),
            [False, False, False],
        )
        self.assertEqual(
            _accepted(just_outside, centers, self.half, self.margin).tolist(),
            [True, True, True],
        )

    def test_flat_center_clearance_is_rejected_for_the_large_bars(self):
        """A regression to any flat centre rule must fail: 0.65 m is fine only for a tiny bar."""
        centers = torch.zeros((3, 1, 2))
        candidates = torch.tensor([[LEGACY_FLAT_CENTER_CLEARANCE_M, 0.0]]).repeat(3, 1)
        accepted = _accepted(candidates, centers, self.half, self.margin)
        # Under the old rule all three were accepted; only bars smaller than the required
        # surface margin difference may pass now, and none of these do.
        self.assertEqual(accepted.tolist(), [False, False, False])
        self.assertLess(
            float(LEGACY_FLAT_CENTER_CLEARANCE_M - self.radius[1]),
            0.15,
            "the largest pool bar left barely 0.10 m of surface clearance under the flat rule",
        )

    def test_empty_bar_set_accepts_everything(self):
        candidates = torch.zeros((4, 2))
        accepted = _accepted(candidates, torch.zeros((4, 0, 2)), torch.zeros((4, 0, 2)), self.margin)
        self.assertTrue(bool(accepted.all()))


class RandomizeGeneralSpawnTest(unittest.TestCase):
    """End-to-end over the real method, with a mixed-size synthetic arena."""

    def _scenario(self, num_envs, spacing=4.0, arena=40.0):
        torch.manual_seed(20260827)
        sizes = torch.tensor(
            [
                [0.4430, 0.4430],  # pool minimum footprint
                [0.7728, 0.7728],  # pool maximum footprint
                [0.8000, 0.3000],  # wide, thin
                [0.3000, 0.8000],  # thin, tall
            ]
        )
        coords = torch.arange(spacing, arena - spacing + 1e-6, spacing)
        grid = torch.stack(torch.meshgrid(coords, coords), dim=-1).reshape(-1, 2)
        n_bars = grid.shape[0]
        half = 0.5 * sizes[torch.arange(n_bars) % sizes.shape[0]]
        obs = {
            "env_bounds_min": torch.zeros((num_envs, 3)),
            "env_bounds_max": torch.tensor([arena, arena, 3.0]).repeat(num_envs, 1),
            "obstacle_position": torch.zeros((num_envs, n_bars, 3)),
            "asset_collision_half_extents": torch.zeros((num_envs, n_bars, 3)),
            "robot_position": torch.zeros((num_envs, 3)),
            "robot_orientation": torch.zeros((num_envs, 4)),
            "robot_linvel": torch.zeros((num_envs, 3)),
            "robot_angvel": torch.zeros((num_envs, 3)),
        }
        obs["obstacle_position"][:, :, 0:2] = grid
        obs["asset_collision_half_extents"][:, :, 0:2] = half
        margin = 0.649802
        stub = types.SimpleNamespace(
            obs_dict=obs,
            device="cpu",
            _bar_offset=0,
            # No distractors in this scenario, so the solid-obstacle slice the sampler reads
            # ([_solid_obstacle_offset : _bar_offset + n_bars_active]) is exactly these bars.
            _solid_obstacle_offset=0,
            n_bars_active=n_bars,
            task_config=types.SimpleNamespace(flight_altitude=1.5),
            _spawn_required_surface_margin_m=lambda: margin,
        )
        return stub, obs, grid, half, margin

    def test_every_spawn_clears_each_bar_actual_surface(self):
        num_envs = 256
        stub, obs, grid, half, margin = self._scenario(num_envs)
        _randomize_spawn(stub, torch.arange(num_envs))
        spawns = obs["robot_position"][:, 0:2]

        centers = grid.unsqueeze(0).expand(num_envs, -1, -1)
        halves = half.unsqueeze(0).expand(num_envs, -1, -1)
        surface = _rectangle_surface_distance(spawns, centers, halves).amin(dim=1)
        worst = float(surface.min())
        self.assertGreaterEqual(
            worst,
            margin - 1e-6,
            "a spawn cleared the nearest bar surface by only %.4f m (need %.4f m)"
            % (worst, margin),
        )

    def test_a_flat_center_rule_would_have_failed_this_scenario(self):
        """Guards the guard: the scenario must be able to expose size blindness."""
        num_envs = 256
        stub, obs, grid, half, margin = self._scenario(num_envs)
        centers = grid.unsqueeze(0).expand(num_envs, -1, -1)
        halves = half.unsqueeze(0).expand(num_envs, -1, -1)
        torch.manual_seed(7)
        lo, hi = 1.0, 39.0
        candidates = lo + (hi - lo) * torch.rand((num_envs, 2))
        flat_ok = (
            torch.cdist(candidates.unsqueeze(1), centers).squeeze(1).amin(dim=1)
            >= LEGACY_FLAT_CENTER_CLEARANCE_M
        )
        surface = _rectangle_surface_distance(candidates, centers, halves).amin(dim=1)
        violating = flat_ok & (surface < margin)
        self.assertGreater(
            int(violating.sum()),
            0,
            "scenario is too sparse to detect the flat-clearance bug",
        )

    def test_other_spawn_state_is_untouched(self):
        num_envs = 16
        stub, obs, _grid, _half, _margin = self._scenario(num_envs)
        _randomize_spawn(stub, torch.arange(num_envs))
        self.assertTrue(bool((obs["robot_position"][:, 2] == 1.5).all()))
        quat = obs["robot_orientation"]
        self.assertTrue(torch.allclose(quat.norm(dim=1), torch.ones(num_envs), atol=1e-6))
        self.assertTrue(bool((quat[:, 0:2] == 0.0).all()))  # yaw-only rotation
        self.assertTrue(bool((obs["robot_linvel"] == 0.0).all()))
        self.assertTrue(bool((obs["robot_angvel"] == 0.0).all()))

    def test_empty_env_ids_is_a_no_op(self):
        stub, obs, _grid, _half, _margin = self._scenario(4)
        before = obs["robot_position"].clone()
        _randomize_spawn(stub, torch.zeros(0, dtype=torch.long))
        self.assertTrue(torch.equal(obs["robot_position"], before))

    def test_exhaustion_fails_closed_instead_of_using_unchecked_candidate(self):
        obs = {
            "env_bounds_min": torch.zeros((1, 3)),
            "env_bounds_max": torch.tensor([[40.0, 40.0, 3.0]]),
            "obstacle_position": torch.tensor([[[0.0, 0.0], [20.0, 20.0]]]),
            "asset_collision_half_extents": torch.tensor(
                [[[0.1, 0.1], [20.0, 20.0]]]
            ),
            "robot_position": torch.zeros((1, 3)),
            "robot_orientation": torch.zeros((1, 4)),
            "robot_linvel": torch.zeros((1, 3)),
            "robot_angvel": torch.zeros((1, 3)),
        }
        stub = types.SimpleNamespace(
            device="cpu",
            obs_dict=obs,
            _bar_offset=1,
            _solid_obstacle_offset=1,
            n_bars_active=1,
            _spawn_required_surface_margin_m=lambda: 0.65,
            task_config=types.SimpleNamespace(flight_altitude=1.5),
        )
        with self.assertRaisesRegex(RuntimeError, "8192 attempts"):
            _randomize_spawn(stub, torch.tensor([0]))


_sample_general_target = _load_task_method(
    "_sample_general_target",
    {"_spawn_footprint_clearance_accepted": _accepted},
)


class SampleGeneralTargetTest(unittest.TestCase):
    """`_sample_general_target`'s static-goal fallback (2026-08-30).

    Same defect as the drone spawn fix, one call site over: the fallback used when the target
    is not `bounded`/`physical` compared a candidate goal to `goal_min_bar_clearance` (1.0 m)
    from each bar's CENTRE, blind to the bar's own footprint.  Measured impact was small
    (`goal_in_inflated_obstacle_frac` ~0.013% at 205 bars per the 2026-08-27 audit) because the
    real default is 1.0 m, not the function's 0.65 m getattr fallback -- but it is the identical
    bug shape, so it is fixed with the identical helper (`_spawn_footprint_clearance_accepted`).

    The bounded/physical branch (`_target_spawn_center_clearance`, navrl_task.py:5840) is left
    untouched: it already derives clearance from the pool's max bar radius and was not part of
    what the audit flagged. `_sample_general_target` is a plain module-level free function here
    (not a class attribute) -- storing it on the class would let attribute lookup auto-bind
    `self` as its first positional argument on top of the explicit stub `self` it already takes.
    """

    def _scenario(self, num_envs, spacing=4.0, arena=40.0):
        torch.manual_seed(20260830)
        sizes = torch.tensor(
            [
                [0.4430, 0.4430],  # pool minimum footprint
                [0.7728, 0.7728],  # pool maximum footprint
                [0.8000, 0.3000],
                [0.3000, 0.8000],
            ]
        )
        coords = torch.arange(spacing, arena - spacing + 1e-6, spacing)
        grid = torch.stack(torch.meshgrid(coords, coords), dim=-1).reshape(-1, 2)
        n_bars = grid.shape[0]
        half = 0.5 * sizes[torch.arange(n_bars) % sizes.shape[0]]
        margin = 0.649802
        bars_xy = grid.unsqueeze(0).expand(num_envs, -1, -1).clone()
        bar_half = half.unsqueeze(0).expand(num_envs, -1, -1).clone()
        b_min = torch.zeros((num_envs, 3))
        b_max = torch.tensor([arena, arena, 3.0]).repeat(num_envs, 1)
        start_pos = torch.full((num_envs, 3), arena / 2.0)
        stub = types.SimpleNamespace(
            device="cpu",
            n_bars_active=n_bars,
            _physical_target=False,
            _target_dynamics="static",
            _target_speed_max=lambda: 0.0,
            _general_goal_distance_bounds=lambda: (2.0, 15.0, None),
            _spawn_required_surface_margin_m=lambda: margin,
            task_config=types.SimpleNamespace(goal_min_bar_clearance=1.0, flight_altitude=1.5),
        )
        env_ids = torch.arange(num_envs)
        return stub, env_ids, start_pos, b_min, b_max, bars_xy, bar_half, grid, half, margin

    def test_every_goal_clears_each_bar_actual_surface(self):
        num_envs = 256
        stub, env_ids, start_pos, b_min, b_max, bars_xy, bar_half, grid, half, margin = (
            self._scenario(num_envs)
        )
        goal = _sample_general_target(
            stub, env_ids, start_pos, b_min, b_max, bars_xy, bar_half
        )
        centers = grid.unsqueeze(0).expand(num_envs, -1, -1)
        halves = half.unsqueeze(0).expand(num_envs, -1, -1)
        surface = _rectangle_surface_distance(goal[:, 0:2], centers, halves).amin(dim=1)
        worst = float(surface.min())
        self.assertGreaterEqual(
            worst,
            margin - 1e-6,
            "a static-fallback goal cleared the nearest bar surface by only %.4f m (need %.4f m)"
            % (worst, margin),
        )
        self.assertTrue(bool((goal[:, 2] == 1.5).all()) or bool(torch.isfinite(goal[:, 2]).all()))

    def test_flat_default_clearance_would_have_failed_this_scenario(self):
        """Guards the guard, using the REAL default (1.0 m), not the unreachable 0.65 m literal."""
        num_envs = 256
        stub, env_ids, start_pos, b_min, b_max, bars_xy, bar_half, grid, half, margin = (
            self._scenario(num_envs)
        )
        torch.manual_seed(11)
        lo, hi = 1.0, 39.0
        candidates = lo + (hi - lo) * torch.rand((num_envs, 2))
        centers = grid.unsqueeze(0).expand(num_envs, -1, -1)
        halves = half.unsqueeze(0).expand(num_envs, -1, -1)
        flat_ok = (
            torch.cdist(candidates.unsqueeze(1), centers).squeeze(1).amin(dim=1)
            >= stub.task_config.goal_min_bar_clearance
        )
        surface = _rectangle_surface_distance(candidates, centers, halves).amin(dim=1)
        violating = flat_ok & (surface < margin)
        self.assertGreater(
            int(violating.sum()),
            0,
            "scenario is too sparse to detect the flat-clearance bug at the real 1.0 m default",
        )

    def test_bounded_physical_branch_keeps_its_own_clearance_source(self):
        """Structural guard: the untouched branch must still call _target_spawn_center_clearance
        and must NOT route through the footprint-aware helper."""
        method_src = ast.get_source_segment(
            _TASK_PATH.read_text(encoding="utf-8"),
            next(
                n
                for n in next(
                    c
                    for c in _TASK_TREE.body
                    if isinstance(c, ast.ClassDef) and c.name == "NavRLTask"
                ).body
                if isinstance(n, ast.FunctionDef) and n.name == "_sample_general_target"
            ),
        )
        self.assertIn("_target_spawn_center_clearance", method_src)
        self.assertIn("use_center_clearance", method_src)
        self.assertIn("goal_required_margin is None", method_src)

    def test_physical_exhaustion_reports_full_batched_budget(self):
        stub = types.SimpleNamespace(
            device="cpu",
            n_bars_active=1,
            _physical_target=True,
            _target_dynamics="physical",
            _target_speed_max=lambda: 1.5,
            _target_spawn_center_clearance=lambda: 100.0,
            _general_goal_distance_bounds=lambda: (6.0, 28.0, None),
            cur=types.SimpleNamespace(wall_margin=0.5),
            tm=types.SimpleNamespace(physical_boundary_margin=0.75),
            task_config=types.SimpleNamespace(goal_min_bar_clearance=1.0, flight_altitude=1.5),
        )
        env_ids = torch.tensor([0])
        start = torch.tensor([[20.0, 20.0, 1.5]])
        b_min = torch.zeros((1, 3))
        b_max = torch.tensor([[40.0, 40.0, 3.0]])
        bars = torch.tensor([[[20.0, 20.0]]])
        half = torch.tensor([[[20.0, 20.0]]])
        with self.assertRaisesRegex(RuntimeError, "16384 attempts"):
            _sample_general_target(stub, env_ids, start, b_min, b_max, bars, half)


if __name__ == "__main__":
    unittest.main()
