"""CPU-only tests for the DETECT-resolution decoupling (WORKLOG 2026-08-22).

The detection camera can now run at a higher resolution than the RGB/perception camera:
NAVRL_DETECT_WIDTH/HEIGHT default to NAVRL_CAMERA_WIDTH/HEIGHT, and when they differ the target
count/centroid/range come from a detect-resolution ray-cast while the RGB image, the obstacle
depth map and everything built from them stay at the camera resolution.

Loaded the same way tests/test_navrl_detector_appearance.py and tests/test_navrl_perception.py
load their subjects: navrl_detector by file path with `warp` stubbed (so the @wp.kernel
decorators evaluate and wp.launch becomes a no-op), navrl_perception by file path with the
aerial_gym package stubbed. No Isaac Gym, no GPU, no torch CUDA.

What this file guards:
  - the two knobs exist and default to the camera resolution, so the default configuration is
    the historical one;
  - the renderer -> perception channel is a strict one-frame hand-off (a stale read raises);
  - the fail-closed guards fire on EVERY knob that breaks the "segmenting a high-resolution
    render == reading the high-resolution mask" identity, and do NOT fire on the knobs that
    perturb both resolutions identically;
  - the detect-resolution reduction arithmetic (count, integer centroid sums, mean range,
    bounding box, empty-mask convention);
  - the intrinsics split: a detect-resolution centroid is converted with detect-resolution
    intrinsics, and detect_fx/fy/cx/cy are the SAME floats as fx/fy/cx/cy when the resolutions
    are equal (this is what makes the default path bit-identical);
  - the camera-resolution branch still contains the historical statements verbatim.

What this does NOT prove: that the warp kernel writes what the reduction expects, or the
end-to-end bit-identity of the default path -- both need a GPU and were measured separately
(see the WORKLOG entry / the report accompanying this change).

Run: PYTHONNOUSERSITE=1 python -m unittest discover -s tests -p "test_navrl_detect_resolution.py"
"""

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]

# ---- stub warp exactly as tests/test_navrl_detector_appearance.py does -------------------
_warp = types.ModuleType("warp")


def _passthrough(fn=None, **_kwargs):
    if fn is None:
        return lambda g: g
    return fn


def _any_callable(*_args, **_kwargs):
    return None


class _WarpStub(types.ModuleType):
    def __getattr__(self, name):
        if name == "kernel":
            return _passthrough
        return _any_callable


_warp.kernel = _passthrough
_warp.__class__ = _WarpStub
sys.modules.setdefault("warp", _warp)

for _pkg in ("aerial_gym", "aerial_gym.utils", "aerial_gym.task", "aerial_gym.task.navrl_task"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_math_stub = types.ModuleType("aerial_gym.utils.math")
_math_stub.quat_rotate = _any_callable
_math_stub.quat_mul = _any_callable
sys.modules.setdefault("aerial_gym.utils.math", _math_stub)


def _load(name, relative_path):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DET = _load("navrl_detector_detectres", "aerial_gym/task/navrl_task/navrl_detector.py")
DET_SOURCE = (REPO / "aerial_gym/task/navrl_task/navrl_detector.py").read_text(encoding="utf-8")

# navrl_perception imports its corridor sibling under the production module name.
_load(
    "aerial_gym.task.navrl_task.navrl_corridor",
    "aerial_gym/task/navrl_task/navrl_corridor.py",
)
# The perception module resolves the channel through the PRODUCTION module name, so bind the
# already-loaded detector there: that exercises the real import line instead of stubbing past it.
sys.modules["aerial_gym.task.navrl_task"].__path__ = [str(REPO / "aerial_gym/task/navrl_task")]
sys.modules["aerial_gym.task.navrl_task.navrl_detector"] = _DET
_PERC = _load("navrl_perception_detectres", "aerial_gym/task/navrl_task/navrl_perception.py")

# The warp stub is needed only while the subjects above are executed: their @wp.kernel decorators
# and annotations evaluate at import. Leaving it in sys.modules made every LATER test module in a
# full-suite run import a fake warp whose Mesh returns None, which is how the dynamic-mesh gates
# came to error on a None mesh. Put the name back now that the loading is done.
if sys.modules.get("warp") is _warp:
    del sys.modules["warp"]
PERC_SOURCE = (REPO / "aerial_gym/task/navrl_task/navrl_perception.py").read_text(
    encoding="utf-8"
)


def _load_task_config(env=None):
    """Exec the task config with a controlled environment (it reads os.environ at class-body
    execution time, so a fresh exec per case is the only way to vary the knobs)."""
    path = REPO / "aerial_gym/config/task_config/navrl_task_config.py"
    saved = dict(os.environ)
    try:
        for key in [k for k in os.environ if k.startswith("NAVRL_")]:
            del os.environ[key]
        os.environ.update(env or {})
        namespace = {"__name__": "navrl_task_config_probe", "__file__": str(path)}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        return namespace["task_config"]
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---------------------------------------------------------------------------- config knobs
class ConfigKnobs(unittest.TestCase):
    def test_defaults_equal_the_camera_resolution(self):
        vision = _load_task_config().vision
        self.assertEqual(vision.detect_width, vision.camera_width)
        self.assertEqual(vision.detect_height, vision.camera_height)
        self.assertEqual((vision.detect_width, vision.detect_height), (160, 90))

    def test_defaults_follow_a_camera_resolution_override(self):
        """The default must track the camera knob, not the 160x90 literal -- otherwise raising
        only the camera resolution would silently DECOUPLE the two and take the new path."""
        vision = _load_task_config(
            {"NAVRL_CAMERA_WIDTH": "480", "NAVRL_CAMERA_HEIGHT": "300"}
        ).vision
        self.assertEqual((vision.detect_width, vision.detect_height), (480, 300))

    def test_env_override(self):
        vision = _load_task_config(
            {"NAVRL_DETECT_WIDTH": "1920", "NAVRL_DETECT_HEIGHT": "1200"}
        ).vision
        self.assertEqual((vision.camera_width, vision.camera_height), (160, 90))
        self.assertEqual((vision.detect_width, vision.detect_height), (1920, 1200))


# ------------------------------------------------------------------------- the hand-off channel
class DetectChannel(unittest.TestCase):
    def setUp(self):
        self.channel = _DET._DetectResolutionChannel()

    def test_consume_returns_the_published_frame_once(self):
        self.channel.publish({"count": 7})
        self.assertEqual(self.channel.consume()["count"], 7)

    def test_second_consume_raises_instead_of_serving_a_stale_frame(self):
        self.channel.publish({"count": 7})
        self.channel.consume()
        with self.assertRaises(RuntimeError):
            self.channel.consume()

    def test_consume_before_any_publish_raises(self):
        with self.assertRaises(RuntimeError):
            self.channel.consume()

    def test_module_singleton_exists(self):
        self.assertIsInstance(_DET.DETECT_CHANNEL, _DET._DetectResolutionChannel)


# ----------------------------------------------------------- detector-side fail-closed guard
def _detector_guard_stub(**knobs):
    """A stand-in `self` carrying only what the guard reads."""
    stub = types.SimpleNamespace(
        detect_width=1920, detect_height=1200, width=160, height=90,
        app_hue_deg=0.0, app_light_gain=0.0, app_albedo_jitter=0.0,
        app_texture_std=0.0, app_motion_blur=0.0,
        # Appearance distractors break the identity too (they are painted the target colour, so
        # perception's segmenter fires on pixels the target ray-cast never produced). The
        # distractor arm of the guard lives in tests/test_navrl_distractors.py.
        num_distractors=0,
    )
    for key, value in knobs.items():
        setattr(stub, key, value)
    return stub


class DetectorFailsClosed(unittest.TestCase):
    GUARD = staticmethod(_DET.NavRLTargetDetector._assert_detect_decoupling_is_equivalent)

    def test_all_knobs_zero_is_accepted(self):
        self.GUARD(_detector_guard_stub())  # must not raise

    def test_each_appearance_knob_raises(self):
        for knob in (
            "app_hue_deg",
            "app_light_gain",
            "app_albedo_jitter",
            "app_texture_std",
            "app_motion_blur",
        ):
            with self.subTest(knob=knob):
                with self.assertRaises(RuntimeError) as caught:
                    self.GUARD(_detector_guard_stub(**{knob: 0.25}))
                message = str(caught.exception)
                self.assertIn("not", message.lower())
                self.assertIn("1920x1200", message)

    def test_the_message_names_every_offending_knob(self):
        with self.assertRaises(RuntimeError) as caught:
            self.GUARD(_detector_guard_stub(app_hue_deg=30.0, app_motion_blur=0.4))
        message = str(caught.exception)
        self.assertIn("appearance_hue_deg", message)
        self.assertIn("appearance_motion_blur", message)


# ------------------------------------------------------------- detect-resolution reduction
class DetectReduction(unittest.TestCase):
    """Exercise _render_detect's arithmetic with the warp launch stubbed to a no-op.

    The mask/depth block buffers are pre-filled, so this measures exactly the reduction: count,
    the two INTEGER centroid sums, the masked depth sum, and the bounding box.
    """

    W, H, N = 8, 6, 3

    def _stub(self):
        stub = types.SimpleNamespace(
            num_envs=self.N, detect_width=self.W, detect_height=self.H,
            device="cpu", max_range=20.0, target_radius=0.15,
            target_half_extents=None, target_use_oriented_box=False,
            mesh_ids=None, _origins_wp=None, _orientations_wp=None,
            _targets_wp=None, _target_orientations_wp=None,
            _detect_mask_block_wp=None, _detect_depth_block_wp=None,
            _detect_ray_blocks=[(0, self.H, None)],
            _detect_mask_block=torch.zeros(self.N, self.H, self.W, dtype=torch.int32),
            _detect_depth_block=torch.full((self.N, self.H, self.W), 20.0),
            _detect_row_count=torch.zeros(self.N, self.H, dtype=torch.int64),
            _detect_row_depth=torch.zeros(self.N, self.H),
            _detect_col_count=torch.zeros(self.N, self.W, dtype=torch.int64),
            _detect_u_index=torch.arange(self.W, dtype=torch.int64).view(1, -1),
            _detect_v_index=torch.arange(self.H, dtype=torch.int64).view(1, -1),
            target_color=torch.tensor([[0.88, 0.08, 0.045]]).repeat(self.N, 1),
        )
        return stub

    def _run(self, stub):
        _DET.DETECT_CHANNEL.clear()
        _DET.NavRLTargetDetector._render_detect(stub)
        return _DET.DETECT_CHANNEL.consume()

    def test_count_centroid_and_range(self):
        stub = self._stub()
        # env 0: a 2x3 patch at rows 1-2, cols 4-6, at 7.0 m.
        stub._detect_mask_block[0, 1:3, 4:7] = 1
        stub._detect_depth_block[0, 1:3, 4:7] = 7.0
        # env 1: a single pixel at (row 5, col 0) at 3.5 m.
        stub._detect_mask_block[1, 5, 0] = 1
        stub._detect_depth_block[1, 5, 0] = 3.5
        # env 2: nothing.
        frame = self._run(stub)
        self.assertEqual(frame["count"].tolist(), [6, 1, 0])
        # u centroid: env0 = mean(4,5,6) = 5, env1 = 0.
        denom = frame["count"].clamp(min=1).float()
        self.assertAlmostEqual(float(frame["u_sum"][0] / denom[0]), 5.0, places=6)
        self.assertAlmostEqual(float(frame["v_sum"][0] / denom[0]), 1.5, places=6)
        self.assertAlmostEqual(float(frame["u_sum"][1] / denom[1]), 0.0, places=6)
        self.assertAlmostEqual(float(frame["v_sum"][1] / denom[1]), 5.0, places=6)
        self.assertAlmostEqual(float(frame["depth_sum"][0] / denom[0]), 7.0, places=5)
        self.assertAlmostEqual(float(frame["depth_sum"][1] / denom[1]), 3.5, places=5)
        # An empty env contributes nothing at all -- no far-plane leakage into the range.
        self.assertEqual(float(frame["depth_sum"][2]), 0.0)
        self.assertEqual(float(frame["u_sum"][2]), 0.0)

    def test_bounding_box_and_empty_convention(self):
        stub = self._stub()
        stub._detect_mask_block[0, 1:3, 4:7] = 1
        self._run(stub)
        self.assertEqual(stub.detect_bbox[0].tolist(), [4.0, 1.0, 6.0, 2.0])
        # Empty mask keeps the camera-resolution convention: min = size, max = -1.
        self.assertEqual(stub.detect_bbox[2].tolist(), [self.W, self.H, -1.0, -1.0])

    def test_frame_carries_the_rendered_target_colour_and_far_plane(self):
        stub = self._stub()
        stub._detect_mask_block[0, 0, 0] = 1
        frame = self._run(stub)
        self.assertEqual(frame["width"], self.W)
        self.assertEqual(frame["height"], self.H)
        self.assertEqual(frame["far_plane"], stub.max_range)
        self.assertTrue(
            torch.allclose(frame["rgb"][0], torch.tensor([0.88, 0.08, 0.045]), atol=1e-6)
        )

    def test_the_frame_carries_no_image_sized_tensor(self):
        """The hand-off must stay a per-env summary: an image or a mask would be a semantic
        leak into the actor-safe module, and would defeat the whole VRAM argument."""
        stub = self._stub()
        stub._detect_mask_block[0, 0, 0] = 1
        frame = self._run(stub)
        for key, value in frame.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.shape[0], self.N, key)
                self.assertLessEqual(
                    value.numel(), self.N * 4, "%s is image-sized (%s)" % (key, value.shape)
                )


# ---------------------------------------------------------------------- perception side
def _perception_configs(**overrides):
    camera = types.SimpleNamespace(
        detector_max_range=20.0,
        detector_hfov_deg=87.0,
        detector_vfov_deg=58.0,
        camera_width=160,
        camera_height=90,
        detect_width=160,
        detect_height=90,
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
        enable_perturbations=False,
    )
    for key, value in overrides.items():
        target = camera if hasattr(camera, key) else perception
        setattr(target, key, value)
    return camera, perception


def _module(**overrides):
    camera, perception = _perception_configs(**overrides)
    return _PERC.NavRLPerceptionModule(2, "cpu", perception, 0.1, camera)


class PerceptionIntrinsics(unittest.TestCase):
    def test_equal_resolutions_give_the_identical_floats(self):
        module = _module()
        self.assertFalse(module.detect_decoupled)
        # Bit-identical, not merely close: this is what makes the default path unchanged.
        self.assertEqual(module.detect_fx, module.fx)
        self.assertEqual(module.detect_fy, module.fy)
        self.assertEqual(module.detect_cx, module.cx)
        self.assertEqual(module.detect_cy, module.cy)

    def test_detect_intrinsics_scale_with_the_detect_resolution(self):
        module = _module(detect_width=1920, detect_height=1200)
        self.assertTrue(module.detect_decoupled)
        self.assertAlmostEqual(module.detect_fx / module.fx, 1920.0 / 160.0, places=6)
        self.assertAlmostEqual(module.detect_fy / module.fy, 1200.0 / 90.0, places=6)
        self.assertAlmostEqual(module.detect_cx, (1920 - 1) * 0.5, places=6)
        self.assertAlmostEqual(module.detect_cy, (1200 - 1) * 0.5, places=6)
        # The camera-resolution intrinsics must NOT move: the obstacle map still uses them.
        self.assertAlmostEqual(module.fx, 160.0 / (2.0 * math.tan(math.radians(87.0) / 2)), 6)

    def test_a_detect_resolution_centroid_with_camera_fx_would_bias_the_bearing(self):
        """Motivation check: the two intrinsics differ by the resolution ratio, so mixing them
        is a systematic 12x bearing error here, not a rounding difference."""
        module = _module(detect_width=1920, detect_height=1200)
        u = module.detect_cx - 100.0  # 100 px left of centre on the DETECT image
        right = math.atan(100.0 / module.detect_fx)
        wrong = math.atan(100.0 / module.fx)
        self.assertGreater(abs(wrong - right), math.radians(20.0))
        self.assertAlmostEqual(math.degrees(right), 5.65, places=1)
        self.assertAlmostEqual(math.degrees(wrong), 49.9, places=0)
        del u


class PerceptionFailsClosed(unittest.TestCase):
    HI = {"detect_width": 640, "detect_height": 360}

    def test_clean_decoupled_config_is_accepted(self):
        module = _module(**self.HI)
        self.assertTrue(module.detect_decoupled)

    def test_equal_resolutions_never_trip_the_guard(self):
        """Every knob the guard refuses is legal at equal resolutions -- the guard must not
        become a new restriction on the historical configuration."""
        module = _module(
            detection_latency_s=0.3,
            rgb_noise_std=0.015,
            depth_noise_std=0.02,
            enable_perturbations=True,
        )
        self.assertFalse(module.detect_decoupled)

    def test_latency_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _module(detection_latency_s=0.3, **self.HI)
        self.assertIn("detection_latency_s", str(caught.exception))

    def test_image_noise_with_perturbations_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            _module(rgb_noise_std=0.015, enable_perturbations=True, **self.HI)
        self.assertIn("rgb_noise_std", str(caught.exception))
        with self.assertRaises(RuntimeError) as caught:
            _module(depth_noise_std=0.02, enable_perturbations=True, **self.HI)
        self.assertIn("depth_noise_std", str(caught.exception))

    def test_image_noise_with_perturbations_off_is_accepted(self):
        module = _module(rgb_noise_std=0.015, depth_noise_std=0.02, **self.HI)
        self.assertTrue(module.detect_decoupled)

    def test_a_learned_segmenter_raises(self):
        module = _module(**self.HI)
        module.segmenter = _PERC.SpatialTargetSegmenter()
        with self.assertRaises(RuntimeError) as caught:
            module._assert_detect_decoupling_is_equivalent(
                _perception_configs(**self.HI)[1]
            )
        self.assertIn("SpatialTargetSegmenter", str(caught.exception))

    def test_a_depth_weighted_1x1_head_raises(self):
        module = _module(**self.HI)
        with torch.no_grad():
            module.segmenter.classifier.weight[0, 3, 0, 0] = 0.5
        with self.assertRaises(RuntimeError) as caught:
            module._assert_detect_decoupling_is_equivalent(
                _perception_configs(**self.HI)[1]
            )
        self.assertIn("depth-channel weight", str(caught.exception))

    def test_a_detector_checkpoint_path_raises(self):
        cfg = _perception_configs(**self.HI)[1]
        cfg.detector_checkpoint = "/nonexistent/head.pth"
        module = _module(**self.HI)
        with self.assertRaises(RuntimeError) as caught:
            module._assert_detect_decoupling_is_equivalent(cfg)
        self.assertIn("detector_checkpoint", str(caught.exception))

    def test_a_profile_checkpoint_raises(self):
        cfg = _perception_configs(**self.HI)[1]
        cfg.detector_profile_checkpoint = "/nonexistent/profile.pth"
        module = _module(**self.HI)
        with self.assertRaises(RuntimeError) as caught:
            module._assert_detect_decoupling_is_equivalent(cfg)
        self.assertIn("detector_profile_checkpoint", str(caught.exception))

    def test_runtime_backstop_raises_when_training_is_forced(self):
        """cfg.enable_perturbations is False here, so __init__ cannot see it coming."""
        module = _module(rgb_noise_std=0.015, **self.HI)
        with self.assertRaises(RuntimeError):
            module._detect_rgbd(
                torch.zeros(2, 3, 90, 160), torch.full((2, 90, 160), 5.0), True
            )


class PerceptionUsesTheDetectFrame(unittest.TestCase):
    """Feed a synthetic detect frame straight into _detect_rgbd (the channel itself is the
    detector's job and is covered above)."""

    W, H = 640, 360

    def _module_with_frame(self, count, u_sum, v_sum, depth_sum, rgb=(0.88, 0.08, 0.045)):
        module = _module(detect_width=self.W, detect_height=self.H)
        frame = {
            "num_envs": 2,
            "width": self.W,
            "height": self.H,
            "far_plane": 20.0,
            "count": torch.tensor(count, dtype=torch.int64),
            "u_sum": torch.tensor(u_sum),
            "v_sum": torch.tensor(v_sum),
            "depth_sum": torch.tensor(depth_sum),
            "rgb": torch.tensor([list(rgb), list(rgb)]),
            "depth_probe": torch.tensor(depth_sum) / torch.tensor(count).clamp(min=1).float(),
        }
        module._consume_detect_frame = lambda: frame
        return module

    def _rgbd(self):
        # A camera-resolution image with NO target in it at all: the detection must come
        # entirely from the frame.
        return torch.full((2, 3, 90, 160), 0.15), torch.full((2, 90, 160), 8.0)

    def test_detection_comes_from_the_frame_not_the_image(self):
        u_centre = 0.25 * (self.W - 1)
        module = self._module_with_frame(
            count=[400, 0],
            u_sum=[400.0 * u_centre, 0.0],
            v_sum=[400.0 * ((self.H - 1) * 0.5), 0.0],
            depth_sum=[400.0 * 6.0, 0.0],
        )
        rgb, depth = self._rgbd()
        meas, surface_range, bearing, visible, confidence, pixels = module._detect_rgbd(
            rgb, depth, False
        )
        self.assertEqual(visible.tolist(), [True, False])
        self.assertAlmostEqual(float(surface_range[0]), 6.0, places=4)
        # The reported bearing is atan2 of the measurement vector, i.e. the back-projected ray
        # scaled to the CENTRE range and shifted by the camera mount offset -- rebuild it here
        # rather than approximating it, so this pins the intrinsics AND the geometry.
        ray = [1.0, -(u_centre - module.detect_cx) / module.detect_fx,
               -(((self.H - 1) * 0.5) - module.detect_cy) / module.detect_fy]
        norm = math.sqrt(sum(c * c for c in ray))
        centre_range = 6.0 + module.target_radius
        expected = math.atan2(
            0.0 + ray[1] / norm * centre_range, 0.1 + ray[0] / norm * centre_range
        )
        self.assertAlmostEqual(float(bearing[0]), expected, places=5)
        self.assertGreater(float(confidence[0]), 0.5)
        # Not-visible env: every pixel-derived quantity is zeroed, exactly like an empty mask.
        self.assertEqual(float(surface_range[1]), 0.0)
        self.assertEqual(float(confidence[1]), 0.0)
        # The camera-resolution mask (used by the obstacle map) is empty here and stays empty.
        self.assertEqual(int(pixels.sum()), 0)
        self.assertEqual(module._last_detect_count.tolist(), [400, 0])

    def test_min_pixels_is_applied_at_the_detect_resolution(self):
        module = self._module_with_frame(
            count=[2, 1], u_sum=[0.0, 0.0], v_sum=[0.0, 0.0], depth_sum=[10.0, 5.0]
        )
        rgb, depth = self._rgbd()
        _, _, _, visible, _, _ = module._detect_rgbd(rgb, depth, False)
        self.assertEqual(visible.tolist(), [True, False])  # min_target_pixels = 2

    def test_a_frame_whose_colour_fails_the_threshold_is_not_a_detection(self):
        """Perception decides target-ness itself: a frame carrying a colour its own segmenter
        rejects yields no detection, however many pixels it claims."""
        module = self._module_with_frame(
            count=[10000, 0], u_sum=[0.0, 0.0], v_sum=[0.0, 0.0],
            depth_sum=[50000.0, 0.0], rgb=(0.15, 0.15, 0.15),
        )
        rgb, depth = self._rgbd()
        _, _, _, visible, _, _ = module._detect_rgbd(rgb, depth, False)
        self.assertEqual(visible.tolist(), [False, False])

    def test_frame_shape_mismatch_raises(self):
        module = self._module_with_frame(
            count=[1, 1], u_sum=[0.0, 0.0], v_sum=[0.0, 0.0], depth_sum=[5.0, 5.0]
        )
        del module._consume_detect_frame
        _DET.DETECT_CHANNEL.clear()
        _DET.DETECT_CHANNEL.publish(
            {"num_envs": 2, "width": 99, "height": self.H, "far_plane": 20.0}
        )
        with self.assertRaises(RuntimeError):
            module._consume_detect_frame()


# ------------------------------------------------------------------- source-level invariants
class SourceInvariants(unittest.TestCase):
    def test_camera_resolution_branch_keeps_the_historical_statements(self):
        """The equal-resolution path must remain the ORIGINAL expressions. Every frozen
        checkpoint and result depends on these exact lines."""
        for statement in (
            "u = (mf * self._u).sum(dim=(1, 2)) / denom",
            "v = (mf * self._v).sum(dim=(1, 2)) / denom",
            "surface_range = (depth * mf).sum(dim=(1, 2)) / denom",
            "confidence = (score * mf).sum(dim=(1, 2)) / denom",
        ):
            self.assertIn(statement, PERC_SOURCE)
        for statement in (
            "u_center = (mask_f * self._u_grid).sum(dim=(1, 2)) / denom",
            "bearing = torch.atan((self.cx - u_center) / self.fx)",
            "surface_range = (self.target_depth * mask_f).sum(dim=(1, 2)) / denom",
        ):
            self.assertIn(statement, DET_SOURCE)

    def test_the_detect_render_is_gated(self):
        """No extra kernel launch, no extra buffer, when the resolutions are equal."""
        self.assertIn("if self.detect_decoupled:\n            self._render_detect()", DET_SOURCE)
        init = DET_SOURCE[
            DET_SOURCE.index("def __init__(self, warp_env"):DET_SOURCE.index("def _assert_detect")
        ]
        gate = init.index("if self.detect_decoupled:\n            detect_rows")
        for buffer_name in ("_detect_mask_block", "_detect_row_count", "_detect_ray_blocks"):
            self.assertGreater(
                init.index(buffer_name), gate, "%s is allocated unconditionally" % buffer_name
            )

    def test_the_perception_module_never_imports_the_renderer_at_module_level(self):
        """navrl_perception must stay importable without warp; the channel import is local."""
        tree = ast.parse(PERC_SOURCE)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.dump(node)
                self.assertNotIn("navrl_detector", text)
                self.assertNotIn("warp", text)


if __name__ == "__main__":
    unittest.main()
