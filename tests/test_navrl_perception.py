import ast
import importlib.util
import inspect
import math
import os
from pathlib import Path
import sys
import types
import unittest

import torch

# Load the one sibling dependency under its production module name without importing the
# aerial_gym package. Importing that package after torch violates Isaac Gym's import-order rule,
# which made this supposedly CPU-only test fail before collecting a single case.
_TASK_DIR = Path(__file__).parents[1] / "aerial_gym/task/navrl_task"
for _package in ("aerial_gym", "aerial_gym.task", "aerial_gym.task.navrl_task"):
    sys.modules.setdefault(_package, types.ModuleType(_package))
_CORRIDOR_NAME = "aerial_gym.task.navrl_task.navrl_corridor"
_CORRIDOR_SPEC = importlib.util.spec_from_file_location(
    _CORRIDOR_NAME, _TASK_DIR / "navrl_corridor.py"
)
_CORRIDOR = importlib.util.module_from_spec(_CORRIDOR_SPEC)
sys.modules[_CORRIDOR_NAME] = _CORRIDOR
_CORRIDOR_SPEC.loader.exec_module(_CORRIDOR)
_SEARCH_NAME = "aerial_gym.task.navrl_task.navrl_search_state"
_SEARCH_SPEC = importlib.util.spec_from_file_location(
    _SEARCH_NAME, _TASK_DIR / "navrl_search_state.py"
)
_SEARCH = importlib.util.module_from_spec(_SEARCH_SPEC)
sys.modules[_SEARCH_NAME] = _SEARCH
_SEARCH_SPEC.loader.exec_module(_SEARCH)

_MODULE_PATH = Path(__file__).parents[1] / "aerial_gym/task/navrl_task/navrl_perception.py"
_SPEC = importlib.util.spec_from_file_location("navrl_perception_standalone", _MODULE_PATH)
_PERCEPTION = importlib.util.module_from_spec(_SPEC)
# VBEAMS/HBEAMS are module constants frozen from NAVRL_LIDAR_* at exec time, and these cases are
# written against the 4x36 geometry (they index specific bins).  Under a full `unittest discover`
# an earlier module -- tests/test_navrl_mode_probe.py -- sets HBEAMS=72 at import scope, which used
# to leak in here.  This module object is created by spec_from_file_location and is NOT the shared
# aerial_gym.task.navrl_task.navrl_perception entry in sys.modules, so pinning the environment for
# the duration of exec_module isolates it without disturbing anyone else.  Reloading the shared
# module would break every other test module already holding a reference to it.
_SAVED_LIDAR_ENV = {k: os.environ.get(k) for k in ("NAVRL_LIDAR_VBEAMS", "NAVRL_LIDAR_HBEAMS")}
os.environ["NAVRL_LIDAR_VBEAMS"] = "4"
os.environ["NAVRL_LIDAR_HBEAMS"] = "36"
try:
    _SPEC.loader.exec_module(_PERCEPTION)
finally:
    for _k, _v in _SAVED_LIDAR_ENV.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
NavRLPerceptionModule = _PERCEPTION.NavRLPerceptionModule
STRUCTURED_OBS_DIM = _PERCEPTION.STRUCTURED_OBS_DIM
select_cluster_sector_obstacles = _PERCEPTION.select_cluster_sector_obstacles


_TASK_PATH = _TASK_DIR / "navrl_task.py"
_TASK_TREE = ast.parse(_TASK_PATH.read_text(encoding="utf-8"), filename=str(_TASK_PATH))


def _load_task_function(name, namespace=None):
    """Load one dependency-free task helper without importing Isaac Gym task modules."""
    node = next(
        node
        for node in _TASK_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"math": math, **(namespace or {})}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_TASK_PATH), "exec"), namespace)
    return namespace[name]


def _load_task_method(name, namespace):
    """Load one task method as a free function for a lightweight contract test."""
    task_class = next(
        node
        for node in _TASK_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
    )
    node = next(
        node
        for node in task_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_TASK_PATH), "exec"), namespace)
    return namespace[name]


_full_eval_distribution_enabled = _load_task_function(
    "_full_eval_distribution_enabled"
)
_goal_distance_bounds = _load_task_function("_goal_distance_bounds")
_goal_front_centered = _load_task_function(
    "_goal_front_centered", {"torch": torch}
)
_fov_curriculum_saturated = _load_task_function("_fov_curriculum_saturated")
_fov_curriculum_bearing_limit_rad = _load_task_function(
    "_fov_curriculum_bearing_limit_rad",
    {"_fov_curriculum_saturated": _fov_curriculum_saturated},
)
_general_goal_distance_bounds_method = _load_task_method(
    "_general_goal_distance_bounds",
    {"_goal_distance_bounds": _goal_distance_bounds},
)
_runtime_physics_contract_method = _load_task_method(
    "_runtime_physics_contract", {}
)
_align_general_spawn_yaw_method = _load_task_method(
    "_align_general_spawn_yaw_to_target", {"torch": torch}
)


def _configs():
    camera = types.SimpleNamespace(
        detector_max_range=20.0,
        detector_hfov_deg=87.0,
        detector_vfov_deg=58.0,
        camera_width=80,
        camera_height=45,
        camera_translation=[0.1, 0.0, 0.03],
        camera_target_radius=0.15,
        tracker_memory_s=5.0,
    )
    perception = types.SimpleNamespace(
        lidar_max_range=4.0,
        min_target_pixels=2,
        pixel_threshold=0.55,
        detection_dropout_prob=0.0,
        detection_latency_s=0.0,
        range_error_m=0.0,
        rgb_noise_std=0.0,
        depth_noise_std=0.0,
        history_interval_s=0.5,
        detector_checkpoint="",
    )
    return camera, perception


class NavRLPerceptionTest(unittest.TestCase):
    def setUp(self):
        camera, perception = _configs()
        self.module = NavRLPerceptionModule(2, "cpu", perception, 0.1, camera)
        self.rgb = torch.full((2, 3, 45, 80), 0.15)
        self.depth = torch.full((2, 45, 80), 10.0)
        self.rgb[0, 0, 20:24, 38:43] = 0.88
        self.rgb[0, 1, 20:24, 38:43] = 0.08
        self.rgb[0, 2, 20:24, 38:43] = 0.045
        self.depth[0, 20:24, 38:43] = 6.0
        self.lidar = torch.full((2, 144), 4.0)
        self.lidar[0, 18] = 2.0
        self.pos = torch.zeros(2, 3)
        self.vel = torch.zeros(2, 3)
        self.quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        self.bounds_min = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.bounds_max = torch.tensor([[40.0, 20.0, 3.0], [40.0, 20.0, 3.0]])

    def test_goal_centered_diagnostic_is_forward_only(self):
        goals = torch.tensor(
            [
                [3.0, 0.1, 0.0],
                [-3.0, 0.1, 0.0],  # old abs-sine-only label falsely accepted the rear cone
                [3.0, 2.0, 0.0],
            ]
        )
        self.assertEqual(_goal_front_centered(goals).tolist(), [True, False, False])

    def _observe(self):
        return self.module.observe(
            self.rgb,
            self.depth,
            self.lidar,
            self.pos,
            self.vel,
            self.quat,
            torch.zeros(2),
            torch.zeros(2, 4),
            2.0,
            1.0,
            training=False,
            env_bounds_min=self.bounds_min,
            env_bounds_max=self.bounds_max,
        )

    def test_body_geofence_ranges_rotate_with_vehicle(self):
        pos = torch.tensor([[10.0, 5.0, 1.0], [10.0, 5.0, 1.0]])
        half = math.sqrt(0.5)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half, half]])
        feat = _PERCEPTION.body_geofence_features(
            pos, quat, self.bounds_min, self.bounds_max
        )
        diagonal = math.sqrt(40.0**2 + 20.0**2)
        expected = torch.tensor(
            [
                [30.0, 15.0, 10.0, 5.0],
                [15.0, 10.0, 5.0, 30.0],
            ]
        ) / diagonal
        self.assertTrue(torch.allclose(feat[:, :4], expected, atol=1e-6))
        self.assertTrue(torch.equal(feat[:, 4:], torch.ones(2, 4)))

    def test_body_geofence_dropout_marks_missing_ranges(self):
        feat = _PERCEPTION.body_geofence_features(
            self.pos,
            self.quat,
            self.bounds_min,
            self.bounds_max,
            dropout=1.0,
        )
        self.assertTrue(torch.equal(feat[:, :4], torch.ones(2, 4)))
        self.assertTrue(torch.equal(feat[:, 4:], torch.zeros(2, 4)))

    def test_force_invalid_ablation_preserves_schema_and_masks_token(self):
        if not (_PERCEPTION.GEOFENCE_ACTOR and _PERCEPTION.GEOFENCE_FORCE_INVALID):
            self.skipTest("requires geofence force-invalid evaluation schema")
        obs, _ = self._observe()
        self.assertEqual(tuple(obs.shape), (2, STRUCTURED_OBS_DIM))
        self.assertTrue(torch.equal(obs[:, -8:-4], torch.ones(2, 4)))
        self.assertTrue(torch.equal(obs[:, -4:], torch.zeros(2, 4)))

    def test_raw_rgbd_produces_track_without_semantic_input(self):
        obs, diag = self._observe()
        self.assertEqual(tuple(obs.shape), (2, STRUCTURED_OBS_DIM))
        self.assertTrue(bool(diag["visible"][0]))
        self.assertFalse(bool(diag["visible"][1]))
        self.assertGreater(float(diag["confidence"][0]), 0.5)
        self.assertTrue(torch.isfinite(obs).all())

    def test_target_return_mask_is_sensor_associated_and_resettable(self):
        self.depth[0, 20:24, 38:43] = 3.0
        self.lidar[0] = 4.0
        # Four vertical returns at physical bearing zero and the detected surface range.
        self.lidar[0, 18::36] = 3.0
        self._observe()
        self.assertTrue(bool(self.module.last_target_like[0, :, 18].all()))
        self.assertFalse(bool(self.module.last_target_like[1].any()))
        self.module.reset_idx(torch.tensor([0]))
        self.assertFalse(bool(self.module.last_target_like[0].any()))

    def test_actor_perception_api_has_no_oracle_argument(self):
        names = set(inspect.signature(self.module.observe).parameters)
        forbidden = {"target_position", "target_velocity", "target_mask", "semantic_id"}
        self.assertFalse(names & forbidden)

    def test_speed_governor_does_not_read_semantic_target_ids(self):
        task_class = next(
            node
            for node in _TASK_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
        )
        method = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_sensor_speed_governor"
        )
        names = {
            node.id for node in ast.walk(method) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
        }
        constants = {
            node.value for node in ast.walk(method) if isinstance(node, ast.Constant)
        }
        self.assertNotIn("segmentation_pixels", names)
        self.assertNotIn(50, constants)
        self.assertIn("last_target_like", names)

    def test_uncertainty_grows_during_occlusion(self):
        self._observe()
        p_seen = torch.diagonal(self.module.tracker.cov[0]).sum().item()
        self.rgb[:] = 0.15
        for _ in range(4):
            _, diag = self._observe()
        p_hidden = torch.diagonal(self.module.tracker.cov[0]).sum().item()
        self.assertFalse(bool(diag["visible"][0]))
        self.assertGreater(p_hidden, p_seen)
        self.assertGreater(float(diag["track_age"][0]), 0.0)

    def test_lidar_range_can_continue_camera_initialized_track(self):
        self.depth[0, 20:24, 38:43] = 3.0
        self.lidar[0] = 4.0
        self._observe()
        self.rgb[:] = 0.15
        # Target sits dead ahead (bearing 0). Under the PHYSICAL bin convention (bearing
        # decreases with the index: 180 - 10*j at 36 beams) that is bin 18, not bin 17 --
        # the old fixture value 17 encoded the mirrored table this test must now reject.
        self.lidar[0, 18] = 2.85
        _, diag = self._observe()
        self.assertFalse(bool(diag["camera_visible"][0]))
        self.assertTrue(bool(diag["lidar_visible"][0]))
        self.assertTrue(bool(diag["visible"][0]))

    def test_range_error_shifts_reported_surface_range(self):
        camera, perception = _configs()
        perception.range_error_m = 0.5
        module = NavRLPerceptionModule(1, "cpu", perception, 0.1, camera)
        rgb = self.rgb[:1].clone()
        depth = self.depth[:1].clone()
        meas, surface, _, visible, _, _ = module._detect_rgbd(rgb, depth, training=True)
        self.assertTrue(bool(visible[0]))
        self.assertAlmostEqual(float(surface[0]), 6.5, places=3)
        self.assertGreater(float(meas[0, 0]), 6.0)

    def test_detection_latency_defers_visible_measurements(self):
        camera, perception = _configs()
        perception.detection_latency_s = 0.1
        module = NavRLPerceptionModule(1, "cpu", perception, 0.1, camera)
        rgb = self.rgb[:1].clone()
        depth = self.depth[:1].clone()
        _, _, _, visible0, _, _ = module._detect_rgbd(rgb, depth, training=True)
        self.assertFalse(bool(visible0[0]))
        _, _, _, visible1, _, _ = module._detect_rgbd(rgb, depth, training=True)
        self.assertTrue(bool(visible1[0]))


class ClusterSectorSelectorTest(unittest.TestCase):
    def setUp(self):
        self.bearings = torch.deg2rad(torch.arange(-120.0, 125.0, 5.0))
        self.max_range = 12.0

    def _select(self, scan, *, slots=4, sectors=4):
        return select_cluster_sector_obstacles(
            torch.tensor([scan], dtype=torch.float32),
            self.bearings,
            max_range=self.max_range,
            max_obstacles=slots,
            token_fov_deg=240.0,
            cluster_gap_m=0.45,
            num_sectors=sectors,
        )

    def test_adjacent_surface_returns_consume_one_slot(self):
        scan = [self.max_range] * len(self.bearings)
        # Three adjacent 5-degree returns at ~2 m are one physical surface cluster.
        scan[10:13] = [2.10, 2.00, 2.08]
        scan[20] = 3.0
        scan[30] = 4.0
        ranges, indices, valid = self._select(scan)
        picked = indices[0, valid[0]].tolist()
        self.assertEqual(sum(index in (10, 11, 12) for index in picked), 1)
        self.assertEqual(valid.sum().item(), 3)
        self.assertTrue(torch.all(ranges[0, 1:] >= ranges[0, :-1]))

    def test_each_nonempty_sector_keeps_its_nearest_cluster(self):
        scan = [self.max_range] * len(self.bearings)
        # One representative in every 60-degree sector. Index 7 is a second, farther cluster in
        # the first sector and must not displace coverage of the far fourth-sector obstacle.
        for index, distance in ((2, 2.0), (7, 2.5), (16, 3.0), (28, 4.0), (43, 8.0)):
            scan[index] = distance
        _, indices, valid = self._select(scan)
        picked = set(indices[0, valid[0]].tolist())
        self.assertEqual(picked, {2, 16, 28, 43})

    def test_empty_sectors_are_filled_by_remaining_clusters(self):
        scan = [self.max_range] * len(self.bearings)
        # All returns lie in one sector but are separated by no-return beams, hence three clusters.
        for index, distance in ((3, 2.0), (6, 3.0), (9, 4.0)):
            scan[index] = distance
        ranges, indices, valid = self._select(scan)
        self.assertEqual(valid.sum().item(), 3)
        self.assertEqual(set(indices[0, valid[0]].tolist()), {3, 6, 9})
        self.assertEqual(ranges[0, valid[0]].tolist(), [2.0, 3.0, 4.0])


class HeldOutDistributionContractTest(unittest.TestCase):
    def test_bulk_eval_ignores_checkpoint_distance_and_fov_clocks(self):
        full_distribution = _full_eval_distribution_enabled(True, "0")
        self.assertTrue(full_distribution)
        self.assertEqual(
            _goal_distance_bounds(4.0, 18.0, 7.0, full_distribution),
            (4.0, 18.0),
        )
        self.assertTrue(
            _fov_curriculum_saturated(full_distribution, 0, 32, 3000)
        )

        def fail_if_checkpoint_clock_is_read():
            self.fail("full held-out evaluation read checkpoint k_max_cur")

        task = types.SimpleNamespace(
            general_eval_mode=False,
            _eval_full_distribution=True,
            general_goal_dist_min=4.0,
            general_goal_dist_max=18.0,
            _goal_x_max=fail_if_checkpoint_clock_is_read,
        )
        self.assertEqual(
            _general_goal_distance_bounds_method(task),
            (4.0, 18.0, True),
        )

    def test_explicit_full_distribution_flag_has_the_same_contract(self):
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value):
                self.assertTrue(_full_eval_distribution_enabled(False, value))

    def test_training_uses_general_min_and_only_curriculum_gates_the_max(self):
        full_distribution = _full_eval_distribution_enabled(False, "0")
        self.assertFalse(full_distribution)
        self.assertEqual(
            _goal_distance_bounds(4.0, 18.0, 7.0, full_distribution),
            (4.0, 7.0),
        )
        self.assertFalse(_fov_curriculum_saturated(False, 3200, 32, 3000))
        self.assertTrue(_fov_curriculum_saturated(False, 96000, 32, 3000))

    def test_fov_bearing_limit_starts_inside_camera_and_ends_unrestricted(self):
        initial = _fov_curriculum_bearing_limit_rad(
            False, 0, 32, 3000, 87.0
        )
        midpoint = _fov_curriculum_bearing_limit_rad(
            False, 48000, 32, 3000, 87.0
        )
        final = _fov_curriculum_bearing_limit_rad(
            False, 96000, 32, 3000, 87.0
        )
        self.assertAlmostEqual(initial, math.radians(87.0 * 0.5 * 0.85))
        self.assertGreater(midpoint, initial)
        self.assertLess(midpoint, math.pi)
        self.assertAlmostEqual(final, math.pi)
        self.assertAlmostEqual(
            _fov_curriculum_bearing_limit_rad(True, 0, 32, 3000, 87.0),
            math.pi,
        )

    def test_general_reset_applies_yaw_curriculum_after_target_sampling(self):
        task_class = next(
            node
            for node in _TASK_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
        )
        reset_method = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "reset_idx"
        )
        called_methods = {
            node.func.attr
            for node in ast.walk(reset_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_sample_general_target", called_methods)
        self.assertIn("_align_general_spawn_yaw_to_target", called_methods)

    def test_general_spawn_yaw_stays_within_current_relative_bearing_limit(self):
        starts = torch.zeros((4, 3))
        goals = torch.tensor(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [-4.0, 0.0, 0.0], [0.0, -4.0, 0.0]]
        )
        limit = math.radians(87.0 * 0.5 * 0.85)
        task = types.SimpleNamespace(
            general_spawn_mode=True,
            vision_mode=True,
            device="cpu",
            obs_dict={"robot_orientation": torch.zeros((4, 4))},
            _fov_curriculum_is_saturated=lambda: False,
            _fov_curriculum_bearing_limit_rad=lambda: limit,
        )
        env_ids = torch.arange(4)
        _align_general_spawn_yaw_method(task, env_ids, starts, goals)
        quat = task.obs_dict["robot_orientation"]
        yaw = 2.0 * torch.atan2(quat[:, 2], quat[:, 3])
        target_bearing = torch.atan2(goals[:, 1], goals[:, 0])
        relative = torch.atan2(
            torch.sin(target_bearing - yaw), torch.cos(target_bearing - yaw)
        )
        self.assertTrue(bool((relative.abs() <= limit + 1e-6).all()))
        self.assertTrue(bool(torch.isfinite(quat).all()))

        # A saturated or held-out evaluation keeps the independently sampled random yaw intact.
        before = quat.clone()
        task._fov_curriculum_is_saturated = lambda: True
        _align_general_spawn_yaw_method(task, env_ids, starts, goals)
        self.assertTrue(torch.equal(task.obs_dict["robot_orientation"], before))

    def test_runtime_physics_contract_reads_simulator_objects(self):
        class FakeSimConfig:
            class sim:
                substeps = 2

        task = types.SimpleNamespace(
            sim_env=types.SimpleNamespace(
                sim_config=FakeSimConfig,
                cfg=types.SimpleNamespace(
                    env=types.SimpleNamespace(
                        num_physics_steps_per_env_step_mean=10
                    )
                ),
            ),
            obs_dict={"dt": 0.01},
            step_dt=0.1,
        )
        self.assertEqual(
            _runtime_physics_contract_method(task),
            {
                "runtime_sim_config_class": "FakeSimConfig",
                "physics_dt_s": 0.01,
                "physics_substeps": 2,
                "physics_steps_per_rl_step": 10,
                "rl_step_dt_s": 0.1,
            },
        )

    def test_bulk_json_exports_the_applied_distribution_contract(self):
        task_class = next(
            node
            for node in _TASK_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
        )
        export_method = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_export_bulk_eval_result"
        )
        strings = {
            node.value
            for node in ast.walk(export_method)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {
                "goal_dist_min_m",
                "goal_dist_max_m",
                "full_goal_distribution",
                "fov_curriculum_saturated",
                "evaluation_nonce",
                "action_selection",
                "strata",
                "speed_bin_edges_mps",
                "distance_bin_edges_m",
            }.issubset(strings)
        )
        physics_method = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_runtime_physics_contract"
        )
        physics_strings = {
            node.value
            for node in ast.walk(physics_method)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {
                "runtime_sim_config_class",
                "physics_dt_s",
                "physics_substeps",
                "physics_steps_per_rl_step",
                "rl_step_dt_s",
            }.issubset(physics_strings)
        )
        called_methods = {
            node.func.attr
            for node in ast.walk(export_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_general_goal_distance_bounds", called_methods)
        self.assertIn("_fov_curriculum_is_saturated", called_methods)

        get_state_method = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_env_state"
        )
        checkpoint_strings = {
            node.value
            for node in ast.walk(get_state_method)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {
                "cfg_runtime_sim_config_class",
                "cfg_physics_dt_s",
                "cfg_physics_substeps",
                "cfg_physics_steps_per_rl_step",
                "cfg_rl_step_dt_s",
                "cfg_detector_max_range",
                "cfg_detect_width",
                "cfg_detect_height",
            }.issubset(checkpoint_strings)
        )


class DetectorCheckpointIntegrityTest(unittest.TestCase):
    """The harness exports NAVRL_EXPECTED_DETECTOR_SHA256; the loader must actually check it.

    It did not until 2026-08-06, so every learned-detector result recorded a detector SHA that
    had never been compared against the bytes loaded. These tests keep the guard alive.
    """

    def setUp(self):
        import hashlib
        import tempfile

        camera, perception = _configs()
        self.camera, self.perception = camera, perception
        module = NavRLPerceptionModule(1, "cpu", perception, 0.1, camera)
        handle = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        torch.save({"model": module.segmenter.state_dict()}, handle.name)
        self.checkpoint = handle.name
        self.sha = hashlib.sha256(open(handle.name, "rb").read()).hexdigest()
        self.addCleanup(os.environ.pop, "NAVRL_EXPECTED_DETECTOR_SHA256", None)

    def _build(self):
        self.perception.detector_checkpoint = self.checkpoint
        return NavRLPerceptionModule(1, "cpu", self.perception, 0.1, self.camera)

    def test_matching_sha_loads(self):
        os.environ["NAVRL_EXPECTED_DETECTOR_SHA256"] = self.sha
        self._build()

    def test_mismatched_sha_is_fatal(self):
        os.environ["NAVRL_EXPECTED_DETECTOR_SHA256"] = "0" * 64
        with self.assertRaises(RuntimeError) as ctx:
            self._build()
        self.assertIn("SHA mismatch", str(ctx.exception))

    def test_absent_expectation_still_loads(self):
        """Interactive/legacy runs that never set the variable must keep working."""
        os.environ.pop("NAVRL_EXPECTED_DETECTOR_SHA256", None)
        self._build()


class SegmenterArchitectureDispatch(unittest.TestCase):
    """An artifact's meta.architecture selects the module class; legacy payloads keep the 1x1."""

    def _payload_roundtrip(self, model, architecture):
        import tempfile

        camera, perception = _configs()
        handle = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        torch.save({"model": model.state_dict(), "meta": {"architecture": architecture}},
                   handle.name)
        perception.detector_checkpoint = handle.name
        return NavRLPerceptionModule(1, "cpu", perception, 0.1, camera)

    def test_spatial_architecture_constructs_the_cnn(self):
        module = self._payload_roundtrip(
            _PERCEPTION.SpatialTargetSegmenter(), "SpatialTargetSegmenter/cnn7x7-RGBD"
        )
        self.assertIsInstance(module.segmenter, _PERCEPTION.SpatialTargetSegmenter)

    def test_pixel_architecture_and_legacy_default_keep_the_1x1(self):
        module = self._payload_roundtrip(
            _PERCEPTION.AppearanceTargetSegmenter(), "AppearanceTargetSegmenter/1x1-RGBD"
        )
        self.assertIsInstance(module.segmenter, _PERCEPTION.AppearanceTargetSegmenter)
        self.assertNotIsInstance(module.segmenter, _PERCEPTION.SpatialTargetSegmenter)
        # v1-era payloads carry no meta at all
        import tempfile

        camera, perception = _configs()
        handle = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        torch.save({"model": _PERCEPTION.AppearanceTargetSegmenter().state_dict()}, handle.name)
        perception.detector_checkpoint = handle.name
        legacy = NavRLPerceptionModule(1, "cpu", perception, 0.1, camera)
        self.assertIsInstance(legacy.segmenter, _PERCEPTION.AppearanceTargetSegmenter)

    def test_spatial_forward_contract(self):
        model = _PERCEPTION.SpatialTargetSegmenter()
        rgb = torch.rand(2, 3, 45, 80)
        depth = torch.rand(2, 45, 80) * 20.0
        out = model(rgb, depth, 20.0)
        self.assertEqual(tuple(out.shape), (2, 45, 80))
        self.assertTrue(bool((out >= 0).all() and (out <= 1).all()))

    def test_spatial_head_stays_small(self):
        """The escalation budget: spatial context, not capacity -- keep it a few k params."""
        n = sum(p.numel() for p in _PERCEPTION.SpatialTargetSegmenter().parameters())
        self.assertLess(n, 5000, n)

    def test_wide_head_dispatch_and_budget(self):
        """The wide tag must win prefix dispatch (it startswith the narrow tag) and stay cheap."""
        wide = _PERCEPTION.build_target_segmenter("SpatialTargetSegmenterWide/cnn9x9x24-RGBD")
        self.assertIsInstance(wide, _PERCEPTION.SpatialTargetSegmenterWide)
        narrow = _PERCEPTION.build_target_segmenter("SpatialTargetSegmenter/cnn7x7-RGBD")
        self.assertIsInstance(narrow, _PERCEPTION.SpatialTargetSegmenter)
        self.assertNotIsInstance(narrow, _PERCEPTION.SpatialTargetSegmenterWide)
        n = sum(p.numel() for p in wide.parameters())
        self.assertLess(n, 15000, n)
        rgb = torch.rand(2, 3, 45, 80)
        depth = torch.rand(2, 45, 80) * 20.0
        out = wide(rgb, depth, 20.0)
        self.assertEqual(tuple(out.shape), (2, 45, 80))


if __name__ == "__main__":
    unittest.main()
