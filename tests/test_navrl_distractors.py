"""CPU-only contracts for the appearance-distractor axis (D9).

The bootstrap detector is a 1x1 conv computing 3.0*R - 2.0*G - 2.0*B - 0.9 -- a red detector --
and the renderer paints the target a flat [0.88, 0.08, 0.045] while nothing else in the scene is
red.  "red pixel => target" is therefore true by construction, which is why detector v7's ~99.77%
frame precision was measured in a world containing ZERO objects that resemble the target.  This
change adds three static, collidable, identically-painted distractor bodies so that claim can be
falsified, and one fail-closed guard so it cannot be falsified silently.

Loaded the way the rest of this directory loads Isaac-Gym-adjacent subjects: modules by file path
with `warp` and the `aerial_gym` package stubbed, and -- where the real module cannot be imported
at all (asset_loader.py imports isaacgym at module scope) -- the exact function under test lifted
out with `ast` and compiled on its own.  No Isaac Gym, no GPU, no torch CUDA.

What this file guards:
  - DEFAULT OFF IS BIT-IDENTICAL: with NAVRL_DISTRACTOR_COUNT unset every distractor class reports
    num_assets == 0, AssetLoader skips a zero-count asset type before it reads the folder or draws
    from the RNG, the built asset order is exactly the historical one, and the renderer allocates
    no buffer and launches no kernel;
  - the fail-closed guard fires for distractors + a decoupled detect resolution, and does NOT fire
    for distractors alone or for decoupling alone;
  - distractors go through the SAME footprint_clearance sampler as the bars and come out obeying
    the same non-overlap / surface-clearance contract, including against the bars;
  - the URDFs parse, and each one's collision primitive is identical to its visual primitive;
  - the asset ordering keeps the physical target at obstacle index 0 and puts the distractors
    between it and the bars, which is what makes NavRLTask's widened _bar_offset correct.

What this does NOT prove: that the Warp distractor kernel writes what the paint expects (that
needs a GPU), or the end-to-end pixel behaviour of a distractor-populated scene.

Run: PYTHONNOUSERSITE=1 python -m unittest discover -s tests -p "test_navrl_distractors.py"
"""

import ast
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import types
import unittest
import xml.etree.ElementTree as ET

import torch

REPO = Path(__file__).resolve().parents[1]
OBJECTS = REPO / "resources/models/environment_assets/objects"

DETECTOR_PATH = REPO / "aerial_gym/task/navrl_task/navrl_detector.py"
ASSET_LOADER_PATH = REPO / "aerial_gym/env_manager/asset_loader.py"
ASSET_MANAGER_PATH = REPO / "aerial_gym/env_manager/asset_manager.py"
ENV_OBJECT_CONFIG_PATH = REPO / "aerial_gym/config/asset_config/env_object_config.py"
BARS_ENV_PATH = REPO / "aerial_gym/config/env_config/navrl_bars_env.py"
TASK_PATH = REPO / "aerial_gym/task/navrl_task/navrl_task.py"

DETECTOR_SOURCE = DETECTOR_PATH.read_text(encoding="utf-8")
ASSET_LOADER_SOURCE = ASSET_LOADER_PATH.read_text(encoding="utf-8")
TASK_SOURCE = TASK_PATH.read_text(encoding="utf-8")


# ---- stub warp exactly as tests/test_navrl_detect_resolution.py does -------------------------
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


def _aerial_gym_stub():
    """A stand-in for the `aerial_gym` package carrying only AERIAL_GYM_DIRECTORY.

    The real package does `import isaacgym` at module scope, which refuses to load once torch is
    imported. Under `unittest discover` some earlier test file may have replaced or dropped the
    module-level stub installed above, so every exec/load below re-asserts its own for the
    duration of the call instead of trusting global state.
    """
    stub = types.ModuleType("aerial_gym")
    stub.AERIAL_GYM_DIRECTORY = str(REPO)
    return stub


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DET = _load("navrl_detector_distractors", DETECTOR_PATH)


# --------------------------------------------------------------- config exec with a clean env
def _exec_config(path, env=None, extra_modules=None):
    """Exec a config module with a controlled NAVRL_* environment.

    The config classes read os.environ in their class bodies, so a fresh exec per case is the
    only way to vary the knobs (same technique as tests/test_navrl_detect_resolution.py).
    """
    saved = dict(os.environ)
    saved_modules = {}
    try:
        for key in [k for k in os.environ if k.startswith("NAVRL_")]:
            del os.environ[key]
        os.environ.update(env or {})
        for name, module in (extra_modules or {}).items():
            saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = module
        namespace = {"__name__": path.stem + "_probe", "__file__": str(path)}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        module = types.ModuleType(path.stem + "_probe")
        module.__dict__.update(namespace)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        os.environ.clear()
        os.environ.update(saved)


def _object_config(env=None):
    return _exec_config(
        ENV_OBJECT_CONFIG_PATH, env, extra_modules={"aerial_gym": _aerial_gym_stub()}
    )


def _bars_env_config(env=None):
    objects = _object_config(env)
    return _exec_config(
        BARS_ENV_PATH,
        env,
        extra_modules={
            "aerial_gym": _aerial_gym_stub(),
            "aerial_gym.config.asset_config.env_object_config": objects,
        },
    )


# ------------------------------------------------------- ast-lifted helpers from asset_loader
def _lift_function(source, path, name, globals_):
    """Compile one top-level def out of a module whose imports we cannot satisfy.

    asset_loader.py does `from isaacgym import gymapi` at module scope, so the module itself is
    unimportable here; the functions under test are pure and are lifted individually.
    """
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(path), "exec"), globals_)
            return globals_[name]
    raise AssertionError("%s does not define a top-level %s()" % (path.name, name))


_URDF_HALF_EXTENTS = _lift_function(
    ASSET_LOADER_SOURCE, ASSET_LOADER_PATH, "_urdf_collision_half_extents", {"ET": ET}
)


def _lift_method(source, path, class_name, method_name, globals_):
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    module = ast.Module(body=[child], type_ignores=[])
                    exec(compile(module, str(path), "exec"), globals_)
                    return globals_[method_name]
    raise AssertionError(
        "%s does not define %s.%s()" % (path.name, class_name, method_name)
    )


# ============================================================================ the URDF assets
DISTRACTOR_URDFS = {
    "sphere": "navrl_distractor_sphere.urdf",
    "box": "navrl_distractor_box.urdf",
    "pole": "navrl_distractor_pole.urdf",
}


def _primitive(element):
    """(kind, sorted dimension dict) of the single geometry primitive under `element`."""
    geometry = element.find("geometry")
    assert geometry is not None, "no <geometry> under %s" % element.tag
    children = list(geometry)
    assert len(children) == 1, "expected exactly one primitive, got %d" % len(children)
    child = children[0]
    return child.tag, {k: v for k, v in sorted(child.attrib.items())}


class DistractorUrdfs(unittest.TestCase):
    def test_every_urdf_exists_and_parses(self):
        for shape, name in DISTRACTOR_URDFS.items():
            with self.subTest(shape=shape):
                path = OBJECTS / name
                self.assertTrue(path.is_file(), "%s is missing" % path)
                root = ET.parse(path).getroot()
                self.assertEqual(root.tag, "robot")
                links = root.findall("link")
                self.assertEqual(len(links), 1, "distractors are single-body assets")
                self.assertEqual(links[0].get("name"), "base_link")

    def test_collision_geometry_equals_visual_geometry(self):
        """Not a visual-only decal: what the camera sees is what the LiDAR and PhysX see."""
        for shape, name in DISTRACTOR_URDFS.items():
            with self.subTest(shape=shape):
                link = ET.parse(OBJECTS / name).getroot().find("link")
                visual = _primitive(link.find("visual"))
                collision = _primitive(link.find("collision"))
                self.assertEqual(visual, collision)
                for tag in ("visual", "collision"):
                    origin = link.find(tag).find("origin")
                    self.assertEqual(
                        [float(v) for v in origin.get("xyz").split()], [0.0, 0.0, 0.0]
                    )

    def test_declared_geometry(self):
        link = ET.parse(OBJECTS / DISTRACTOR_URDFS["sphere"]).getroot().find("link")
        kind, dims = _primitive(link.find("collision"))
        self.assertEqual(kind, "sphere")
        self.assertAlmostEqual(float(dims["radius"]), 0.15, places=9)

        link = ET.parse(OBJECTS / DISTRACTOR_URDFS["box"]).getroot().find("link")
        kind, dims = _primitive(link.find("collision"))
        self.assertEqual(kind, "box")
        self.assertEqual([float(v) for v in dims["size"].split()], [0.30, 0.30, 0.30])

        link = ET.parse(OBJECTS / DISTRACTOR_URDFS["pole"]).getroot().find("link")
        kind, dims = _primitive(link.find("collision"))
        self.assertEqual(kind, "cylinder")
        self.assertAlmostEqual(float(dims["radius"]), 0.06, places=9)
        self.assertAlmostEqual(float(dims["length"]), 1.60, places=9)

    def test_sphere_radius_equals_the_detector_target_radius(self):
        """The whole point of the sphere: size-vs-range consistency cannot separate it."""
        vision = _exec_config(
            REPO / "aerial_gym/config/task_config/navrl_task_config.py"
        ).task_config.vision
        link = ET.parse(OBJECTS / DISTRACTOR_URDFS["sphere"]).getroot().find("link")
        _, dims = _primitive(link.find("collision"))
        self.assertAlmostEqual(
            float(dims["radius"]), float(vision.camera_target_radius), places=9
        )

    def test_pole_is_taller_and_thinner_than_the_target(self):
        target = ET.parse(OBJECTS / "navrl_target_drone.urdf").getroot().find("link")
        _, target_dims = _primitive(target.find("collision"))
        tx, ty, tz = [float(v) for v in target_dims["size"].split()]
        pole = ET.parse(OBJECTS / DISTRACTOR_URDFS["pole"]).getroot().find("link")
        _, pole_dims = _primitive(pole.find("collision"))
        self.assertGreater(float(pole_dims["length"]), tz)
        self.assertLess(2.0 * float(pole_dims["radius"]), min(tx, ty))

    def test_inertials_are_physical(self):
        for shape, name in DISTRACTOR_URDFS.items():
            with self.subTest(shape=shape):
                inertial = ET.parse(OBJECTS / name).getroot().find("link").find("inertial")
                self.assertGreater(float(inertial.find("mass").get("value")), 0.0)
                inertia = inertial.find("inertia")
                for axis in ("ixx", "iyy", "izz"):
                    self.assertGreater(float(inertia.get(axis)), 0.0)
                for axis in ("ixy", "ixz", "iyz"):
                    self.assertEqual(float(inertia.get(axis)), 0.0)

    def test_analytic_inertia_matches_the_declared_primitive(self):
        sphere = ET.parse(OBJECTS / DISTRACTOR_URDFS["sphere"]).getroot().find("link")
        m = float(sphere.find("inertial/mass").get("value"))
        r = float(_primitive(sphere.find("collision"))[1]["radius"])
        self.assertAlmostEqual(
            float(sphere.find("inertial/inertia").get("ixx")), 0.4 * m * r * r, places=9
        )

        box = ET.parse(OBJECTS / DISTRACTOR_URDFS["box"]).getroot().find("link")
        m = float(box.find("inertial/mass").get("value"))
        a, b, _ = [float(v) for v in _primitive(box.find("collision"))[1]["size"].split()]
        self.assertAlmostEqual(
            float(box.find("inertial/inertia").get("ixx")),
            m * (a * a + b * b) / 12.0,
            places=9,
        )

        pole = ET.parse(OBJECTS / DISTRACTOR_URDFS["pole"]).getroot().find("link")
        m = float(pole.find("inertial/mass").get("value"))
        dims = _primitive(pole.find("collision"))[1]
        r, h = float(dims["radius"]), float(dims["length"])
        inertia = pole.find("inertial/inertia")
        self.assertAlmostEqual(
            float(inertia.get("ixx")), m * (3.0 * r * r + h * h) / 12.0, places=6
        )
        self.assertAlmostEqual(float(inertia.get("izz")), 0.5 * m * r * r, places=9)


# ================================================ the collision-footprint extraction extension
class CollisionHalfExtents(unittest.TestCase):
    """The footprint_clearance sampler refuses a zero footprint, so a sphere/cylinder distractor
    is only placeable because asset_loader now understands those primitives."""

    def test_box_extraction_is_unchanged(self):
        self.assertEqual(
            _URDF_HALF_EXTENTS(str(OBJECTS / "navrl_target_drone.urdf")),
            [0.14, 0.14, 0.06],
        )
        self.assertEqual(
            _URDF_HALF_EXTENTS(str(OBJECTS / "cuboidal_rod.urdf")), [0.05, 0.05, 1.0]
        )

    def test_box_still_wins_over_the_new_branches(self):
        """The historical whole-tree box lookup runs FIRST, so no existing asset can change."""
        tree = ast.parse(ASSET_LOADER_SOURCE, filename=str(ASSET_LOADER_PATH))
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_urdf_collision_half_extents"
        )
        found = [
            node.args[0].value
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "find"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        self.assertEqual(
            found,
            [
                ".//collision/geometry/box",
                ".//collision/geometry/sphere",
                ".//collision/geometry/cylinder",
            ],
        )

    def test_sphere_and_cylinder_extraction(self):
        self.assertEqual(
            _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["sphere"])),
            [0.15, 0.15, 0.15],
        )
        self.assertEqual(
            _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["box"])), [0.15, 0.15, 0.15]
        )
        self.assertEqual(
            _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["pole"])), [0.06, 0.06, 0.80]
        )

    def test_every_distractor_has_a_placeable_footprint(self):
        """[0, 0, 0] is what the sampler rejects; none of the three may fall back to it."""
        for shape, name in DISTRACTOR_URDFS.items():
            with self.subTest(shape=shape):
                half = _URDF_HALF_EXTENTS(str(OBJECTS / name))
                self.assertIsNotNone(half)
                self.assertTrue(all(v > 0.0 for v in half[:2]))

    def test_unparseable_and_meshonly_return_none(self):
        self.assertIsNone(_URDF_HALF_EXTENTS(str(OBJECTS / "no_such_file.urdf")))


# ================================================================== the knobs, and default-off
class DistractorKnobs(unittest.TestCase):
    def test_default_is_zero_and_every_class_reports_no_assets(self):
        cfg = _object_config()
        self.assertEqual(cfg.NAVRL_DISTRACTOR_COUNT, 0)
        for name in (
            "navrl_distractor_sphere_params",
            "navrl_distractor_box_params",
            "navrl_distractor_pole_params",
        ):
            with self.subTest(cls=name):
                self.assertEqual(getattr(cfg, name).num_assets, 0)

    def test_count_is_dealt_round_robin_over_the_default_mix(self):
        cfg = _object_config({"NAVRL_DISTRACTOR_COUNT": "7"})
        counts = (
            cfg.navrl_distractor_sphere_params.num_assets,
            cfg.navrl_distractor_box_params.num_assets,
            cfg.navrl_distractor_pole_params.num_assets,
        )
        self.assertEqual(counts, (3, 2, 2))
        self.assertEqual(sum(counts), 7)

    def test_shape_mix_knob_selects_the_shapes(self):
        cfg = _object_config(
            {"NAVRL_DISTRACTOR_COUNT": "5", "NAVRL_DISTRACTOR_SHAPES": "sphere"}
        )
        self.assertEqual(cfg.navrl_distractor_sphere_params.num_assets, 5)
        self.assertEqual(cfg.navrl_distractor_box_params.num_assets, 0)
        self.assertEqual(cfg.navrl_distractor_pole_params.num_assets, 0)

        cfg = _object_config(
            {"NAVRL_DISTRACTOR_COUNT": "4", "NAVRL_DISTRACTOR_SHAPES": "pole,box"}
        )
        self.assertEqual(cfg.navrl_distractor_pole_params.num_assets, 2)
        self.assertEqual(cfg.navrl_distractor_box_params.num_assets, 2)
        self.assertEqual(cfg.navrl_distractor_sphere_params.num_assets, 0)

    def test_unknown_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            _object_config(
                {"NAVRL_DISTRACTOR_COUNT": "3", "NAVRL_DISTRACTOR_SHAPES": "cone"}
            )

    def test_negative_count_clamps_to_zero(self):
        cfg = _object_config({"NAVRL_DISTRACTOR_COUNT": "-4"})
        self.assertEqual(cfg.NAVRL_DISTRACTOR_COUNT, 0)

    def test_semantic_id_is_distinct_and_agrees_with_the_renderer(self):
        cfg = _object_config()
        self.assertEqual(cfg.DISTRACTOR_SEMANTIC_ID, _DET.DISTRACTOR_SEMANTIC_ID)
        self.assertNotIn(
            cfg.DISTRACTOR_SEMANTIC_ID,
            {cfg.INTERCEPT_TARGET_SEMANTIC_ID, cfg.OBJECT_SEMANTIC_ID},
        )

    def test_distractor_file_names_point_at_the_real_urdfs(self):
        cfg = _object_config({"NAVRL_DISTRACTOR_COUNT": "3"})
        for name, shape in (
            ("navrl_distractor_sphere_params", "sphere"),
            ("navrl_distractor_box_params", "box"),
            ("navrl_distractor_pole_params", "pole"),
        ):
            with self.subTest(shape=shape):
                params = getattr(cfg, name)
                self.assertEqual(params.file, DISTRACTOR_URDFS[shape])
                self.assertTrue((Path(params.asset_folder) / params.file).is_file())

    def test_distractors_are_real_scene_geometry(self):
        """Excluded from Warp they would be solid but invisible to the camera -- the one
        combination that would make the painted frame a lie."""
        cfg = _object_config({"NAVRL_DISTRACTOR_COUNT": "3"})
        for name in (
            "navrl_distractor_sphere_params",
            "navrl_distractor_box_params",
            "navrl_distractor_pole_params",
        ):
            with self.subTest(cls=name):
                params = getattr(cfg, name)
                self.assertTrue(params.include_in_warp)
                self.assertTrue(params.keep_in_env)
                self.assertFalse(params.per_link_semantic)
                self.assertEqual(params.semantic_id, cfg.DISTRACTOR_SEMANTIC_ID)

    def test_distractors_share_the_bar_placement_band(self):
        """AssetManager._placement_band reads asset index 0's ratio for ALL obstacles, and in a
        legacy (non-physical-target) lineage index 0 is a distractor."""
        cfg = _object_config({"NAVRL_DISTRACTOR_COUNT": "3"})
        bar_min = cfg.bar_asset_params.min_state_ratio
        bar_max = cfg.bar_asset_params.max_state_ratio
        for name in (
            "navrl_distractor_sphere_params",
            "navrl_distractor_box_params",
            "navrl_distractor_pole_params",
        ):
            with self.subTest(cls=name):
                params = getattr(cfg, name)
                self.assertEqual(params.min_state_ratio[0], bar_min[0])
                self.assertEqual(params.max_state_ratio[0], bar_max[0])
                self.assertEqual(params.min_state_ratio[1], bar_min[1])
                self.assertEqual(params.max_state_ratio[1], bar_max[1])

    def test_env_config_exposes_the_count_and_defaults_to_zero(self):
        self.assertEqual(_bars_env_config().NavRLBarsEnvCfg.env_config.num_distractors, 0)
        self.assertEqual(
            _bars_env_config({"NAVRL_DISTRACTOR_COUNT": "6"})
            .NavRLBarsEnvCfg.env_config.num_distractors,
            6,
        )


# ================================================================ default-off is bit-identical
class DefaultOffIsUnchanged(unittest.TestCase):
    def test_asset_map_order_of_the_historical_entries_is_preserved(self):
        env_config = _bars_env_config().NavRLBarsEnvCfg.env_config
        keys = list(env_config.asset_type_to_dict_map.keys())
        self.assertLess(keys.index("physical_target"), keys.index("bars"))
        # Distractors are declared BEFORE the target: AssetLoader appendleft()s every keep_in_env
        # asset, so the LAST one loaded is the one that ends up at obstacle index 0.
        for shape in ("distractor_sphere", "distractor_box", "distractor_pole"):
            self.assertLess(keys.index(shape), keys.index("physical_target"))

    def test_zero_count_asset_types_are_skipped_before_any_side_effect(self):
        """`if num_assets > 0` sits before randomly_pick_assets_from_folder and before
        load_selected_file_from_config, so a zero-count type touches neither the folder listing
        nor the RNG -- which is what keeps the historical asset order and placement stream."""
        tree = ast.parse(ASSET_LOADER_SOURCE, filename=str(ASSET_LOADER_PATH))
        loader = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "select_and_order_assets"
        )
        guards = [
            n for n in ast.walk(loader)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "num_assets"
        ]
        self.assertEqual(len(guards), 1)
        called = {
            node.func.attr
            for node in ast.walk(guards[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("randomly_pick_assets_from_folder", called)
        self.assertIn("load_selected_file_from_config", called)

    def test_default_asset_order_is_target_then_bars(self):
        order = _ordered_asset_types(distractor_count=0, num_bars=4)
        self.assertEqual(order, ["physical_target"] + ["bars"] * 4)

    def test_renderer_allocates_nothing_and_launches_nothing_at_zero(self):
        self.assertEqual(_DET._distractor_count(), 0)
        init = _method_source(DETECTOR_SOURCE, "NavRLTargetDetector", "__init__")
        self.assertIn("if self.num_distractors > 0:", init)
        self.assertIn("self._distractor_mask_wp", init)
        render = _method_source(DETECTOR_SOURCE, "NavRLTargetDetector", "_render")
        self.assertIn("if self.num_distractors > 0:", render)
        self.assertIn("_render_distractor_camera_kernel", render)

    def test_every_distractor_statement_in_the_paint_is_guarded(self):
        """With the knob unset, render_raw_rgbd must execute exactly the historical statements."""
        paint = _method_source(DETECTOR_SOURCE, "NavRLTargetDetector", "render_raw_rgbd")
        self.assertIn("if self.num_distractors > 0:", paint)
        for line in paint.splitlines():
            if "distractor" in line and not line.strip().startswith("#"):
                self.assertTrue(
                    line.strip().startswith("if self.num_distractors > 0:")
                    or line.startswith(" " * 12),
                    "unguarded distractor statement in render_raw_rgbd: %r" % line,
                )
        # The historical target paint is still the last word on a target pixel.
        self.assertIn("visible_target_pixels = self.target_mask > 0", paint)
        self.assertIn(
            "depth = torch.where(visible_target_pixels, self.target_depth, depth)", paint
        )
        self.assertLess(
            paint.index("if self.num_distractors > 0:"),
            paint.index("visible_target_pixels = self.target_mask > 0"),
        )

    def test_count_parser_ignores_junk_and_blanks(self):
        saved = os.environ.get("NAVRL_DISTRACTOR_COUNT")
        try:
            for raw, expected in (("", 0), ("   ", 0), ("nonsense", 0), ("-3", 0), ("5", 5)):
                os.environ["NAVRL_DISTRACTOR_COUNT"] = raw
                self.assertEqual(_DET._distractor_count(), expected, raw)
            os.environ.pop("NAVRL_DISTRACTOR_COUNT")
            self.assertEqual(_DET._distractor_count(), 0)
        finally:
            os.environ.pop("NAVRL_DISTRACTOR_COUNT", None)
            if saved is not None:
                os.environ["NAVRL_DISTRACTOR_COUNT"] = saved


def _method_source(source, class_name, method_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError("no %s.%s" % (class_name, method_name))


# ============================================================== the asset ordering / bar offset
def _ordered_asset_types(distractor_count, num_bars):
    """Run the REAL AssetLoader.select_and_order_assets over the real bars env config.

    Lifted out of asset_loader.py with ast because that module imports isaacgym at module scope.
    The only stubs are the two loader helpers it calls, which return the asset type name so the
    resulting list is directly readable as an ordering.
    """
    env_config = _bars_env_config(
        {
            "NAVRL_DISTRACTOR_COUNT": str(distractor_count),
            "NAVRL_MAX_BARS": str(num_bars),
            "NAVRL_TARGET_DYNAMICS": "physical",
        }
    ).NavRLBarsEnvCfg.env_config
    globals_ = {"deque": __import__("collections").deque, "random": random,
                "logger": types.SimpleNamespace(debug=lambda *a, **k: None)}
    fn = _lift_method(
        ASSET_LOADER_SOURCE, ASSET_LOADER_PATH, "AssetLoader", "select_and_order_assets", globals_
    )
    loader = types.SimpleNamespace(
        env_config=env_config,
        randomly_pick_assets_from_folder=lambda folder, num_assets=0: [None] * num_assets,
        load_selected_file_from_config=lambda asset_type, cfg, selected_file: {
            "asset_type": asset_type,
            "keep_in_env": cfg.keep_in_env,
        },
    )
    ordered, keep_in_env_num = fn(loader)
    types_in_order = [entry["asset_type"] for entry in ordered]
    assert keep_in_env_num == 1 + distractor_count, keep_in_env_num
    return types_in_order


class AssetOrdering(unittest.TestCase):
    def test_target_stays_at_index_zero_and_distractors_sit_between(self):
        order = _ordered_asset_types(distractor_count=3, num_bars=5)
        self.assertEqual(order[0], "physical_target")
        self.assertEqual(
            sorted(order[1:4]),
            sorted(["distractor_sphere", "distractor_box", "distractor_pole"]),
        )
        self.assertEqual(order[4:], ["bars"] * 5)

    def test_layout_holds_for_an_uneven_mix(self):
        order = _ordered_asset_types(distractor_count=7, num_bars=2)
        self.assertEqual(order[0], "physical_target")
        self.assertTrue(all(t.startswith("distractor_") for t in order[1:8]))
        self.assertEqual(order[8:], ["bars"] * 2)

    def test_bar_offset_widens_by_the_distractor_count(self):
        """Every bar slice in navrl_task.py starts at _bar_offset; if it did not move, the first
        `num_distractors` "bars" the task reasons about would actually be distractors."""
        self.assertIn(
            "self._bar_offset = (1 if self._physical_target else 0) + self._num_distractors",
            TASK_SOURCE,
        )
        self.assertIn('"num_distractors", 0', TASK_SOURCE)

    def test_active_obstacle_window_covers_the_distractors(self):
        """AssetManager parks assets beyond num_obstacles_in_env at -1000. Because that count is
        n_bars_active + _bar_offset and _bar_offset now includes the distractors, they stay
        placed at every density."""
        self.assertIn(
            'self.obs_dict["num_obstacles_in_env"] = clamped + int(self._bar_offset)',
            TASK_SOURCE,
        )


# ======================================================================= the fail-closed guard
GUARD = staticmethod(_DET.NavRLTargetDetector._assert_detect_decoupling_is_equivalent)


def _guard_stub(**knobs):
    stub = types.SimpleNamespace(
        detect_width=1920, detect_height=1200, width=160, height=90,
        app_hue_deg=0.0, app_light_gain=0.0, app_albedo_jitter=0.0,
        app_texture_std=0.0, app_motion_blur=0.0, num_distractors=0,
    )
    for key, value in knobs.items():
        setattr(stub, key, value)
    return stub


class FailsClosedOnDistractorsPlusDecoupling(unittest.TestCase):
    GUARD = GUARD

    def test_decoupling_alone_does_not_fire(self):
        self.GUARD(_guard_stub())  # decoupled 1920x1200 vs 160x90, no distractors

    def test_distractors_under_decoupling_raise(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.GUARD(_guard_stub(num_distractors=1))
        message = str(ctx.exception)
        # The message must name the broken identity, not just the knob.
        self.assertIn("1920x1200 detect vs 160x90 camera", message)
        self.assertIn("appearance distractor", message)
        self.assertIn("same nominal target colour", message)
        self.assertIn("target ray-cast never produced", message)
        self.assertIn("NAVRL_DETECT_WIDTH/HEIGHT", message)
        self.assertIn("NAVRL_DISTRACTOR_COUNT=0", message)

    def test_the_distractor_raise_precedes_the_appearance_offenders(self):
        """A distractor scene must report the distractor reason even when an appearance knob is
        also on, because the distractor break is the one that is silent."""
        with self.assertRaises(RuntimeError) as ctx:
            self.GUARD(_guard_stub(num_distractors=2, app_hue_deg=30.0))
        self.assertIn("appearance distractor", str(ctx.exception))

    def test_distractors_alone_never_reach_the_guard(self):
        """Camera-resolution path: detect_decoupled is False, so __init__ never calls the guard.
        This is the configuration the distractor axis is meant to be measured on."""
        init = _method_source(DETECTOR_SOURCE, "NavRLTargetDetector", "__init__")
        tree = ast.parse("class _C:\n" + "\n".join("    " + l for l in init.splitlines()))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.dump(node.test).find("detect_decoupled") >= 0:
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "_assert_detect_decoupling_is_equivalent"
                    ):
                        calls.append(inner)
        self.assertEqual(len(calls), 1, "the guard must be called only under detect_decoupled")
        # and nowhere else in __init__
        total = init.count("_assert_detect_decoupling_is_equivalent")
        self.assertEqual(total, 1)

    def test_num_distractors_is_set_before_the_guard_runs(self):
        init = _method_source(DETECTOR_SOURCE, "NavRLTargetDetector", "__init__")
        self.assertLess(
            init.index("self.num_distractors = _distractor_count()"),
            init.index("_assert_detect_decoupling_is_equivalent"),
        )


# ================================================================= footprint-clearance placement
def _asset_manager_class():
    """Load AssetManager with its two aerial_gym imports stubbed (no Isaac Gym, no GPU)."""
    math_stub = types.ModuleType("aerial_gym.utils.math")
    math_stub.torch = torch
    math_stub.torch_rand_float_tensor = _any_callable
    math_stub.torch_interpolate_ratio = _any_callable
    math_stub.quat_from_euler_xyz_tensor = _any_callable
    logging_stub = types.ModuleType("aerial_gym.utils.logging")

    class _Logger:
        def __init__(self, *_a, **_k):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    logging_stub.CustomLogger = _Logger
    saved = {
        "aerial_gym": sys.modules.get("aerial_gym"),
        "aerial_gym.utils": sys.modules.get("aerial_gym.utils"),
        "aerial_gym.utils.math": sys.modules.get("aerial_gym.utils.math"),
        "aerial_gym.utils.logging": sys.modules.get("aerial_gym.utils.logging"),
    }
    sys.modules["aerial_gym"] = _aerial_gym_stub()
    sys.modules["aerial_gym.utils"] = types.ModuleType("aerial_gym.utils")
    sys.modules["aerial_gym.utils.math"] = math_stub
    sys.modules["aerial_gym.utils.logging"] = logging_stub
    try:
        module = _load("navrl_asset_manager_distractors", ASSET_MANAGER_PATH)
        return module.AssetManager
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class DistractorPlacement(unittest.TestCase):
    """Distractors go through the SAME sampler as the bars, so they inherit the same contract:
    every pair of circumcircles is at least `surface_clearance` apart and every circle stays
    inside the placement band. Synthetic geometry -- no Isaac Gym, no env build."""

    ARENA = 40.0
    CLEARANCE = 0.45

    @classmethod
    def setUpClass(cls):
        cls.AssetManager = _asset_manager_class()

    def _manager(self, half_extents):
        envs, n = half_extents.shape[0], half_extents.shape[1]
        obj = self.AssetManager.__new__(self.AssetManager)
        obj.placement_surface_clearance = self.CLEARANCE
        obj.placement_candidate_batch_size = 32
        obj.placement_attempts_before_relax = 128
        obj.asset_collision_half_extents = half_extents
        obj.env_bounds_min = torch.zeros(envs, n, 3)
        obj.env_bounds_max = torch.zeros(envs, n, 3)
        obj.env_bounds_max[:, :, :2] = self.ARENA
        obj.env_bounds_max[:, :, 2] = 3.0
        obj.asset_min_state_ratio = torch.zeros(envs, n, 13)
        obj.asset_max_state_ratio = torch.ones(envs, n, 13)
        return obj

    def _mixed_half_extents(self, envs, num_bars, generator):
        """The real layout: three distractor footprints first (as keep_in_env puts them), bars
        after, all in one tensor -- exactly what AssetManager receives."""
        distractors = torch.tensor(
            [
                _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["sphere"])),
                _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["box"])),
                _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS["pole"])),
            ],
            dtype=torch.float32,
        ).unsqueeze(0).repeat(envs, 1, 1)
        bars = torch.empty(envs, num_bars, 3)
        bars[:, :, :2] = 0.2 + 0.2 * torch.rand(envs, num_bars, 2, generator=generator)
        bars[:, :, 2] = 1.5
        return torch.cat([distractors, bars], dim=1)

    def test_mixed_field_obeys_the_surface_clearance_contract(self):
        envs, num_bars = 8, 60
        generator = torch.Generator().manual_seed(20260901)
        half = self._mixed_half_extents(envs, num_bars, generator)
        n = half.shape[1]
        obj = self._manager(half)
        torch.manual_seed(20260901)
        placed = obj._footprint_clearance_xy_spacing(
            torch.zeros(envs, n, 3), n, torch.arange(envs)
        )
        xy = placed[:, :, :2]
        support = torch.linalg.vector_norm(half[:, :, :2], dim=2)
        required = support[:, :, None] + support[:, None, :] + self.CLEARANCE
        eye = torch.eye(n, dtype=torch.bool)[None]
        distance = (xy[:, :, None, :] - xy[:, None, :, :]).norm(dim=3)
        margin = (distance - required).masked_fill(eye, float("inf"))
        self.assertGreaterEqual(float(margin.amin()), -1e-5)
        # ... and specifically for the distractor rows against everything else.
        self.assertGreaterEqual(float(margin[:, :3, :].amin()), -1e-5)
        # every circumcircle inside the band
        self.assertTrue(bool((xy[:, :, 0] >= support - 1e-5).all()))
        self.assertTrue(bool((xy[:, :, 1] >= support - 1e-5).all()))
        self.assertTrue(bool((xy[:, :, 0] <= self.ARENA - support + 1e-5).all()))
        self.assertTrue(bool((xy[:, :, 1] <= self.ARENA - support + 1e-5).all()))

    def test_distractors_are_actually_moved_by_the_sampler(self):
        """A guard against wiring them in as parked/never-placed assets: their XY must be written,
        not left at the zeros the sampler was handed."""
        envs, num_bars = 4, 10
        generator = torch.Generator().manual_seed(4242)
        half = self._mixed_half_extents(envs, num_bars, generator)
        n = half.shape[1]
        obj = self._manager(half)
        torch.manual_seed(4242)
        placed = obj._footprint_clearance_xy_spacing(
            torch.zeros(envs, n, 3), n, torch.arange(envs)
        )
        self.assertTrue(bool((placed[:, :3, :2] != 0.0).all()))
        # z is untouched by the XY sampler, exactly as for bars.
        self.assertTrue(bool((placed[:, :, 2] == 0.0).all()))

    def test_a_zero_footprint_distractor_would_be_refused(self):
        """This is why asset_loader had to learn sphere/cylinder: an unparsed primitive yields
        [0, 0, 0] and the sampler fails closed rather than placing it on top of a bar."""
        half = torch.full((1, 3, 3), 0.3)
        half[0, 0, :2] = 0.0
        obj = self._manager(half)
        with self.assertRaises(RuntimeError):
            obj._footprint_clearance_xy_spacing(torch.zeros(1, 3, 3), 3, torch.tensor([0]))

    def test_placement_mode_routes_distractors_through_the_same_path(self):
        source = ASSET_MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'if self.placement_mode in ("footprint_clearance", "nonoverlap", '
            '"surface_clearance"):',
            source,
        )
        self.assertIn(
            "return self._footprint_clearance_xy_spacing(positions, num_used, env_ids)", source
        )


# ==================================================================================================
# Distractors as SOLID geometry at the two sites the measurement depends on (2026-09-01)
#
# _bar_offset kept every bar slice bars-only, which is right for bar logic but made the painted
# distractors read as FREE SPACE everywhere. Two of those places would corrupt the distractor
# measurement itself and are therefore widened to _solid_obstacle_offset:
#
#   NavRLTask._randomize_general_drone_spawn -- the drone could spawn INSIDE a distractor, and a
#       detection frame taken from inside a solid body is physically meaningless;
#   NavRLTask._plan_target_routes            -- a planned target route could pass straight THROUGH
#       a distractor, which corrupts the occlusion statistics.
#
# The other five sites (static goal placement, recovery clearance, the two bar-contact probes)
# are deliberately NOT changed and are recorded as a preregistration limitation.
#
# The index arithmetic under test:
#     _bar_offset            = (1 if physical target else 0) + num_distractors
#     _solid_obstacle_offset = _bar_offset - num_distractors
#     solid slice            = [_solid_obstacle_offset : _bar_offset + n_bars_active]
# i.e. the distractors are exactly the num_distractors rows immediately BEFORE _bar_offset, so the
# widened slice is contiguous: [distractors...][active bars...], never the target at index 0 and
# never a parked bar past the active window.
# ==================================================================================================

ROUTE_PLANNER_PATH = REPO / "aerial_gym/task/navrl_task/target_route_planner.py"
_ROUTES = _load("navrl_target_route_planner_distractors", ROUTE_PLANNER_PATH)

# The warp stub is needed only while the subjects above are executed: their @wp.kernel decorators
# and annotations evaluate at import. Leaving it in sys.modules made every LATER test module in a
# full-suite run import a fake warp whose Mesh returns None, which is how the dynamic-mesh gates
# came to error on a None mesh. Put the name back now that the loading is done.
if sys.modules.get("warp") is _warp:
    del sys.modules["warp"]

# The live _spawn_required_surface_margin_m() value, identical to the constant
# tests/test_navrl_spawn_footprint_clearance.py derives from the ref5in URDF + tracking margin.
SPAWN_MARGIN_M = 0.649802


def _lift_task_method(name, extra_globals=None, historical=False):
    """Lift one NavRLTask method as a free function taking an explicit stub `self`.

    ``historical=True`` renames every ``self._solid_obstacle_offset`` back to ``self._bar_offset``
    in the AST before compiling, which reproduces the EXACT pre-distractor bars-only slice these
    two sites used. Running both against the same inputs is what turns "default off is unchanged"
    into a measurement instead of an assertion.
    """
    tree = ast.parse(TASK_SOURCE, filename=str(TASK_PATH))
    node = None
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "NavRLTask":
            for child in cls.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    node = child
    if node is None:
        raise AssertionError("navrl_task.py no longer defines NavRLTask.%s" % name)
    if historical:
        renamed = 0
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "_solid_obstacle_offset":
                sub.attr = "_bar_offset"
                renamed += 1
        if renamed == 0:
            raise AssertionError(
                "%s no longer reads _solid_obstacle_offset, so the default-off comparison would "
                "compare the method against itself" % name
            )
    namespace = {"math": math, "torch": torch, "Path": Path, "ET": ET}
    namespace.update(extra_globals or {})
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(TASK_PATH), "exec"), namespace)
    return namespace[name]


_SPAWN_ACCEPTED = _lift_function(
    TASK_SOURCE, TASK_PATH, "_spawn_footprint_clearance_accepted", {"torch": torch}
)
_SPAWN = _lift_task_method(
    "_randomize_general_drone_spawn",
    {"_spawn_footprint_clearance_accepted": _SPAWN_ACCEPTED},
)
_SPAWN_HISTORICAL = _lift_task_method(
    "_randomize_general_drone_spawn",
    {"_spawn_footprint_clearance_accepted": _SPAWN_ACCEPTED},
    historical=True,
)
_PLAN_ROUTES = _lift_task_method("_plan_target_routes")
_PLAN_ROUTES_HISTORICAL = _lift_task_method("_plan_target_routes", historical=True)


def _distractor_half_xy(shape):
    """The REAL XY footprint the loader extracts from the shipped URDF (not a copied literal)."""
    return _URDF_HALF_EXTENTS(str(OBJECTS / DISTRACTOR_URDFS[shape]))[0:2]


def _rectangle_surface_distance(points_xy, centers_xy, half_xy):
    """Exact axis-aligned rectangle surface distance -- the quantity the geometry audit grids."""
    delta = (points_xy.unsqueeze(1) - centers_xy).abs() - half_xy
    return delta.clamp(min=0.0).norm(dim=2)


def _spawn_stub(obs, bar_offset, solid_offset, n_bars_active, margin=SPAWN_MARGIN_M):
    return types.SimpleNamespace(
        obs_dict=obs,
        device="cpu",
        _bar_offset=bar_offset,
        _solid_obstacle_offset=solid_offset,
        n_bars_active=n_bars_active,
        task_config=types.SimpleNamespace(flight_altitude=1.5),
        _spawn_required_surface_margin_m=lambda: margin,
    )


class _RecordingRouteManager:
    """Captures exactly the obstacle geometry `_plan_target_routes` hands to the planner."""

    def __init__(self, num_envs):
        self.captured = []
        self.valid = torch.zeros(num_envs, dtype=torch.bool)
        self.goal = torch.zeros((num_envs, 2))

    def plan_idx(self, env_ids, start_xy, goal_xy, obstacles, obstacle_half, *args, **kwargs):
        self.captured.append((obstacles.clone(), obstacle_half.clone()))
        return {"ok": int(len(env_ids))}


def _route_stub(obs, num_envs, bar_offset, solid_offset, n_bars_active, support=0.2068816086567407):
    manager = _RecordingRouteManager(num_envs)
    stub = types.SimpleNamespace(
        _target_route_enabled=True,
        obs_dict=obs,
        _bar_offset=bar_offset,
        _solid_obstacle_offset=solid_offset,
        n_bars_active=n_bars_active,
        _target_route_selector=torch.zeros((num_envs, 2)),
        _target_route_manager=manager,
        target_position=torch.zeros((num_envs, 3)),
        _tm_waypoint=torch.zeros((num_envs, 2)),
        _target_route_support_xy=torch.full((num_envs, 2), float(support)),
        num_task_steps=0,
        tm=types.SimpleNamespace(
            route_min_goal_distance_m=1.0, route_goal_exclusion_radius_m=1.0
        ),
    )
    return stub, manager


def _tagged_world(num_envs, num_distractors, bars_active, bars_parked, physical_target=True):
    """A synthetic obstacle tensor whose every row carries its own asset index as a tag.

    Row k gets centre x == k and half-extent x == k, so an off-by-one in EITHER slice -- or a
    slice that starts/ends on the wrong asset -- is directly visible in the recorded arrays.
    """
    target_rows = 1 if physical_target else 0
    total = target_rows + num_distractors + bars_active + bars_parked
    position = torch.zeros((num_envs, total, 3))
    half = torch.zeros((num_envs, total, 3))
    tags = torch.arange(total, dtype=torch.float32)
    position[:, :, 0] = tags
    half[:, :, 0] = tags
    obs = {
        "env_bounds_min": torch.zeros((num_envs, 3)),
        "env_bounds_max": torch.tensor([40.0, 40.0, 3.0]).repeat(num_envs, 1),
        "obstacle_position": position,
        "asset_collision_half_extents": half,
        "robot_position": torch.zeros((num_envs, 3)),
        "robot_orientation": torch.zeros((num_envs, 4)),
        "robot_linvel": torch.zeros((num_envs, 3)),
        "robot_angvel": torch.zeros((num_envs, 3)),
    }
    bar_offset = target_rows + num_distractors
    return obs, bar_offset, bar_offset - num_distractors


class SolidObstacleOffsetArithmetic(unittest.TestCase):
    """The derivation itself, pinned in source the way _bar_offset already is."""

    def test_offset_is_bar_offset_minus_the_distractor_count(self):
        """Exact-line, not `in`: `... - self._num_distractors + 1` contains the same substring
        and would be an off-by-one that silently drops the first distractor and pulls in a
        parked bar."""
        statements = [
            line.strip() for line in TASK_SOURCE.splitlines()
            if line.strip().startswith("self._solid_obstacle_offset =")
        ]
        self.assertEqual(
            statements,
            ["self._solid_obstacle_offset = self._bar_offset - self._num_distractors"],
        )

    def test_offset_derivation_is_the_only_assignment_and_is_never_mutated(self):
        """A second assignment anywhere would make the two widened slices non-contiguous."""
        tree = ast.parse(TASK_SOURCE, filename=str(TASK_PATH))
        writes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "_solid_obstacle_offset"
            and isinstance(node.ctx, ast.Store)
        ]
        self.assertEqual(len(writes), 1)

    def test_offset_equals_bar_offset_whenever_no_distractors_are_configured(self):
        """The whole default-off argument in one line of arithmetic."""
        for physical in (0, 1):
            for num_distractors in (0, 1, 3, 7):
                bar_offset = physical + num_distractors
                solid_offset = bar_offset - num_distractors
                self.assertEqual(solid_offset, physical)
                if num_distractors == 0:
                    self.assertEqual(solid_offset, bar_offset)

    def test_both_widened_sites_use_the_solid_offset_and_no_other_site_does(self):
        """Exactly two call sites moved; every other bar slice still starts at _bar_offset."""
        spawn = _method_source(TASK_SOURCE, "NavRLTask", "_randomize_general_drone_spawn")
        route = _method_source(TASK_SOURCE, "NavRLTask", "_plan_target_routes")
        for source in (spawn, route):
            self.assertEqual(
                source.count(
                    "self._solid_obstacle_offset : self._bar_offset + self.n_bars_active"
                ),
                2,
                "both the centre and the half-extent slice must be widened together",
            )
            self.assertNotIn(
                "self._bar_offset : self._bar_offset + self.n_bars_active", source
            )
        # Two methods x two slices.  The WIDENED slice -- distractors AND active bars -- may appear
        # nowhere else, because every other consumer of an obstacle slice wants bars only.
        self.assertEqual(
            TASK_SOURCE.count(
                "self._solid_obstacle_offset : self._bar_offset + self.n_bars_active"
            ),
            4,
        )
        # The one other legal reader of the offset is the evaluation-only distractor-lock
        # telemetry, and it takes the DISTRACTOR-ONLY slice, which STOPS at _bar_offset instead of
        # running past it into the bars.  The trailing comma is the whole point of matching on it:
        # it proves the slice ends there rather than being the widened one with a line break.  A
        # site that wanted distractors and silently got bars too would classify the detector's
        # reported measurement against a bar centre and produce a plausible FTLR that means
        # nothing.
        # 2026-09-03: the S1 shadow recorder (_record_s1_shadow_frame,
        # prereg_2026-09-03_s1_structure_fix_shadow) is a second legal reader.  It classifies the
        # SHADOW pipeline's counterfactual measurement with the identical distractor-only slice
        # and radius, so the online and shadow FTLR are comparable by construction.  It is
        # evaluation-only (NAVRL_S1_SHADOW=1 in bulk-eval mode) and never touches observations.
        distractor_only = "self._solid_obstacle_offset : self._bar_offset,"
        self.assertEqual(TASK_SOURCE.count(distractor_only), 2)
        for method in ("_record_distractor_lock_frame", "_record_s1_shadow_frame"):
            self.assertIn(
                distractor_only,
                _method_source(TASK_SOURCE, "NavRLTask", method),
            )
        self.assertEqual(
            TASK_SOURCE.count("self._solid_obstacle_offset :"),
            6,
            "a new reader of the solid-obstacle offset appeared; decide explicitly whether it "
            "wants the widened [distractors + active bars] slice or the distractor-only one, and "
            "pin it here",
        )
        # The five deliberately untouched sites still read the bars-only slice.
        self.assertGreaterEqual(
            TASK_SOURCE.count("self._bar_offset : self._bar_offset + self.n_bars_active"), 5
        )


class SolidObstacleSliceAlignment(unittest.TestCase):
    """Row j of the centres array must be the SAME asset as row j of the half-extents array,
    across the distractor->bar boundary. Pairing a distractor centre with a bar footprint is
    exactly the class of bug that produced the original flat-clearance defect."""

    DISTRACTORS = 5
    BARS_ACTIVE = 9
    BARS_PARKED = 4

    def _expected_tags(self, bar_offset, solid_offset):
        return list(range(solid_offset, bar_offset + self.BARS_ACTIVE))

    def _spawn_capture(self, obs, bar_offset, solid_offset, num_envs):
        captured = []

        def _recorder(candidate_xy, centers, half, margin):
            captured.append((centers.clone(), half.clone()))
            return _SPAWN_ACCEPTED(candidate_xy, centers, half, 0.0)

        spawn = _lift_task_method(
            "_randomize_general_drone_spawn",
            {"_spawn_footprint_clearance_accepted": _recorder},
        )
        stub = _spawn_stub(obs, bar_offset, solid_offset, self.BARS_ACTIVE)
        torch.manual_seed(20260901)
        spawn(stub, torch.arange(num_envs))
        self.assertTrue(captured, "the sampler never evaluated a candidate")
        return captured[0]

    def test_spawn_sees_distractors_then_active_bars_paired_row_for_row(self):
        num_envs = 4
        obs, bar_offset, solid_offset = _tagged_world(
            num_envs, self.DISTRACTORS, self.BARS_ACTIVE, self.BARS_PARKED
        )
        self.assertEqual((bar_offset, solid_offset), (1 + self.DISTRACTORS, 1))
        centers, half = self._spawn_capture(obs, bar_offset, solid_offset, num_envs)
        expected = self._expected_tags(bar_offset, solid_offset)
        self.assertEqual(centers.shape[1], len(expected))
        self.assertEqual(half.shape[1], len(expected))
        self.assertEqual(centers[0, :, 0].tolist(), [float(t) for t in expected])
        # The pairing itself: same tag on both axes, every row, boundary included.
        self.assertTrue(torch.equal(centers[:, :, 0], half[:, :, 0]))
        # Absolute anchors, so a uniform shift of BOTH slices cannot pass.
        self.assertEqual(float(centers[0, 0, 0]), float(solid_offset))
        self.assertEqual(float(centers[0, -1, 0]), float(bar_offset + self.BARS_ACTIVE - 1))
        # The target row and the parked bars are excluded.
        self.assertNotIn(0.0, centers[0, :, 0].tolist())
        self.assertNotIn(float(bar_offset + self.BARS_ACTIVE), centers[0, :, 0].tolist())

    def test_route_planner_sees_distractors_then_active_bars_paired_row_for_row(self):
        num_envs = 4
        obs, bar_offset, solid_offset = _tagged_world(
            num_envs, self.DISTRACTORS, self.BARS_ACTIVE, self.BARS_PARKED
        )
        stub, manager = _route_stub(
            obs, num_envs, bar_offset, solid_offset, self.BARS_ACTIVE
        )
        _PLAN_ROUTES(stub, torch.arange(num_envs), connected_goal=True)
        self.assertEqual(len(manager.captured), 1)
        centers, half = manager.captured[0]
        expected = self._expected_tags(bar_offset, solid_offset)
        self.assertEqual(centers.shape, (num_envs, len(expected), 2))
        self.assertEqual(half.shape, centers.shape)
        self.assertEqual(centers[0, :, 0].tolist(), [float(t) for t in expected])
        self.assertTrue(torch.equal(centers[:, :, 0], half[:, :, 0]))
        self.assertEqual(float(centers[0, 0, 0]), float(solid_offset))
        self.assertEqual(float(centers[0, -1, 0]), float(bar_offset + self.BARS_ACTIVE - 1))

    def test_a_one_row_misalignment_would_be_caught(self):
        """Guards the guard: the tagged world must be able to expose an off-by-one."""
        num_envs = 2
        obs, bar_offset, solid_offset = _tagged_world(
            num_envs, self.DISTRACTORS, self.BARS_ACTIVE, self.BARS_PARKED
        )
        centers = obs["obstacle_position"][
            :, solid_offset : bar_offset + self.BARS_ACTIVE, 0:2
        ]
        shifted_half = obs["asset_collision_half_extents"][
            :, solid_offset + 1 : bar_offset + self.BARS_ACTIVE + 1, 0:2
        ]
        self.assertEqual(shifted_half.shape, centers.shape)
        self.assertFalse(torch.equal(centers[:, :, 0], shifted_half[:, :, 0]))

    def test_legacy_lineage_without_a_physical_target_keeps_index_zero_as_a_distractor(self):
        """With NAVRL_TARGET_DYNAMICS unset there is no target actor, so _bar_offset is just the
        distractor count and the solid slice starts at row 0."""
        num_envs = 2
        obs, bar_offset, solid_offset = _tagged_world(
            num_envs, self.DISTRACTORS, self.BARS_ACTIVE, self.BARS_PARKED,
            physical_target=False,
        )
        self.assertEqual((bar_offset, solid_offset), (self.DISTRACTORS, 0))
        stub, manager = _route_stub(
            obs, num_envs, bar_offset, solid_offset, self.BARS_ACTIVE
        )
        _PLAN_ROUTES(stub, torch.arange(num_envs), connected_goal=True)
        centers, half = manager.captured[0]
        self.assertEqual(
            centers[0, :, 0].tolist(),
            [float(t) for t in range(0, bar_offset + self.BARS_ACTIVE)],
        )
        self.assertTrue(torch.equal(centers[:, :, 0], half[:, :, 0]))


class DefaultOffIsBitIdentical(unittest.TestCase):
    """With NAVRL_DISTRACTOR_COUNT unset _num_distractors == 0, so _solid_obstacle_offset ==
    _bar_offset and BOTH changed sites must be byte-for-byte the historical code. Proved by
    running the real method and an AST copy rewritten back to the pre-change bars-only slice
    against identical inputs and identical RNG, then comparing the tensors."""

    ARENA = 40.0

    def _bars_only_world(self, num_envs, bar_offset, n_bars, seed=20260901):
        generator = torch.Generator().manual_seed(seed)
        coords = torch.arange(4.0, self.ARENA - 4.0 + 1e-6, 4.0)
        grid = torch.stack(torch.meshgrid(coords, coords), dim=-1).reshape(-1, 2)[:n_bars]
        self.assertEqual(grid.shape[0], n_bars)
        sizes = torch.tensor(
            [[0.4430, 0.4430], [0.7728, 0.7728], [0.8000, 0.3000], [0.3000, 0.8000]]
        )
        half = 0.5 * sizes[torch.arange(n_bars) % sizes.shape[0]]
        total = bar_offset + n_bars
        position = torch.zeros((num_envs, total, 3))
        extents = torch.zeros((num_envs, total, 3))
        position[:, bar_offset:, 0:2] = grid
        extents[:, bar_offset:, 0:2] = half
        if bar_offset:  # the physical target actor sits at row 0 and is not an obstacle
            position[:, 0, 0:2] = torch.rand((num_envs, 2), generator=generator) * self.ARENA
            extents[:, 0, 0:2] = 0.14
        return {
            "env_bounds_min": torch.zeros((num_envs, 3)),
            "env_bounds_max": torch.tensor([self.ARENA, self.ARENA, 3.0]).repeat(num_envs, 1),
            "obstacle_position": position,
            "asset_collision_half_extents": extents,
            "robot_position": torch.zeros((num_envs, 3)),
            "robot_orientation": torch.zeros((num_envs, 4)),
            "robot_linvel": torch.zeros((num_envs, 3)),
            "robot_angvel": torch.zeros((num_envs, 3)),
        }

    def test_the_historical_copy_really_differs_from_the_live_one(self):
        """Guards the guard: if the rewrite were a no-op both comparisons below would be vacuous."""
        for name in ("_randomize_general_drone_spawn", "_plan_target_routes"):
            with self.subTest(method=name):
                with self.assertRaises(AssertionError):
                    # No _solid_obstacle_offset to rename => the helper refuses.
                    _lift_task_method("_sample_target_motion", historical=True)
                source = _method_source(TASK_SOURCE, "NavRLTask", name)
                self.assertIn("self._solid_obstacle_offset", source)

    def test_spawn_is_bit_identical_with_no_distractors(self):
        num_envs, n_bars = 128, 49
        for bar_offset in (0, 1):
            with self.subTest(bar_offset=bar_offset):
                results = []
                for fn in (_SPAWN, _SPAWN_HISTORICAL):
                    obs = self._bars_only_world(num_envs, bar_offset, n_bars)
                    stub = _spawn_stub(obs, bar_offset, bar_offset, n_bars)
                    torch.manual_seed(20260901)
                    fn(stub, torch.arange(num_envs))
                    results.append(obs)
                live, historical = results
                for key in (
                    "robot_position", "robot_orientation", "robot_linvel", "robot_angvel"
                ):
                    self.assertTrue(
                        torch.equal(live[key], historical[key]),
                        "%s differs from the pre-distractor sampler at bar_offset=%d"
                        % (key, bar_offset),
                    )
                # ...and the spawns are real, not an all-zero coincidence.
                self.assertTrue(bool((live["robot_position"][:, 0:2] != 0.0).all()))

    def test_route_planner_geometry_is_bit_identical_with_no_distractors(self):
        num_envs, n_bars = 8, 25
        for bar_offset in (0, 1):
            with self.subTest(bar_offset=bar_offset):
                captured = []
                for fn in (_PLAN_ROUTES, _PLAN_ROUTES_HISTORICAL):
                    obs = self._bars_only_world(num_envs, bar_offset, n_bars)
                    stub, manager = _route_stub(
                        obs, num_envs, bar_offset, bar_offset, n_bars
                    )
                    status = fn(stub, torch.arange(num_envs), connected_goal=True)
                    self.assertEqual(status, {"ok": num_envs})
                    captured.append(manager.captured[0])
                (live_c, live_h), (old_c, old_h) = captured
                self.assertEqual(live_c.shape, (num_envs, n_bars, 2))
                self.assertTrue(torch.equal(live_c, old_c))
                self.assertTrue(torch.equal(live_h, old_h))

    def test_default_off_slice_never_reaches_the_target_actor_row(self):
        """At bar_offset == 1 and zero distractors the slice must still start at row 1."""
        num_envs, n_bars = 4, 9
        obs = self._bars_only_world(num_envs, 1, n_bars)
        stub, manager = _route_stub(obs, num_envs, 1, 1, n_bars)
        _PLAN_ROUTES(stub, torch.arange(num_envs), connected_goal=True)
        centers, _half = manager.captured[0]
        self.assertEqual(centers.shape[1], n_bars)
        self.assertTrue(
            torch.equal(centers, obs["obstacle_position"][:, 1 : 1 + n_bars, 0:2])
        )


class SpawnTreatsDistractorsAsSolid(unittest.TestCase):
    """A drone that spawns inside a distractor produces a detection frame from inside a solid
    body. Uses the REAL URDF footprints of all three shapes: the pole's XY half-extents (0.06)
    are less than half the sphere's (0.15), so a size-blind rule must fail here."""

    ARENA = 24.0
    NUM_ENVS = 256

    def _world(self, num_envs=None):
        """[target][sphere,box,pole x8][bars...] -- the real layout, real footprints."""
        num_envs = num_envs or self.NUM_ENVS
        shapes = ["sphere", "box", "pole"] * 8
        distractor_half = torch.tensor(
            [_distractor_half_xy(s) for s in shapes], dtype=torch.float32
        )
        step = 2.5
        coords = torch.arange(3.0, self.ARENA - 3.0 + 1e-6, step)
        grid = torch.stack(torch.meshgrid(coords, coords), dim=-1).reshape(-1, 2)
        num_distractors = distractor_half.shape[0]
        bar_sizes = torch.tensor(
            [[0.4430, 0.4430], [0.7728, 0.7728], [0.8000, 0.3000], [0.3000, 0.8000]]
        )
        num_bars = 16
        self.assertGreaterEqual(grid.shape[0], num_distractors + num_bars)
        bar_half = 0.5 * bar_sizes[torch.arange(num_bars) % bar_sizes.shape[0]]
        total = 1 + num_distractors + num_bars
        position = torch.zeros((num_envs, total, 3))
        extents = torch.zeros((num_envs, total, 3))
        position[:, 1:, 0:2] = grid[: num_distractors + num_bars]
        extents[:, 1 : 1 + num_distractors, 0:2] = distractor_half
        extents[:, 1 + num_distractors :, 0:2] = bar_half
        obs = {
            "env_bounds_min": torch.zeros((num_envs, 3)),
            "env_bounds_max": torch.tensor([self.ARENA, self.ARENA, 3.0]).repeat(num_envs, 1),
            "obstacle_position": position,
            "asset_collision_half_extents": extents,
            "robot_position": torch.zeros((num_envs, 3)),
            "robot_orientation": torch.zeros((num_envs, 4)),
            "robot_linvel": torch.zeros((num_envs, 3)),
            "robot_angvel": torch.zeros((num_envs, 3)),
        }
        return obs, 1 + num_distractors, 1, num_distractors, num_bars

    def test_the_three_shapes_really_have_different_footprints(self):
        sphere = _distractor_half_xy("sphere")
        box = _distractor_half_xy("box")
        pole = _distractor_half_xy("pole")
        self.assertEqual(sphere, [0.15, 0.15])
        self.assertEqual(box, [0.15, 0.15])
        self.assertEqual(pole, [0.06, 0.06])
        self.assertLess(2.0 * pole[0], sphere[0])

    def test_no_spawn_lands_inside_any_distractor_or_bar(self):
        obs, bar_offset, solid_offset, num_distractors, num_bars = self._world()
        stub = _spawn_stub(obs, bar_offset, solid_offset, num_bars)
        torch.manual_seed(20260901)
        _SPAWN(stub, torch.arange(self.NUM_ENVS))
        spawns = obs["robot_position"][:, 0:2]
        centers = obs["obstacle_position"][:, solid_offset : bar_offset + num_bars, 0:2]
        halves = obs["asset_collision_half_extents"][
            :, solid_offset : bar_offset + num_bars, 0:2
        ]
        surface = _rectangle_surface_distance(spawns, centers[0], halves[0])
        self.assertGreaterEqual(float(surface.amin()), SPAWN_MARGIN_M - 1e-6)
        # ...and specifically for the distractor rows.
        distractor_surface = surface[:, :num_distractors]
        self.assertGreaterEqual(float(distractor_surface.amin()), SPAWN_MARGIN_M - 1e-6)

    def test_the_pre_change_sampler_did_spawn_inside_distractors(self):
        """Guards the guard: the same geometry and seed under the bars-only slice must put
        spawns inside distractors, otherwise the test above proves nothing."""
        obs, bar_offset, _solid_offset, num_distractors, num_bars = self._world()
        stub = _spawn_stub(obs, bar_offset, bar_offset, num_bars)
        torch.manual_seed(20260901)
        _SPAWN_HISTORICAL(stub, torch.arange(self.NUM_ENVS))
        spawns = obs["robot_position"][:, 0:2]
        distractor_centers = obs["obstacle_position"][
            0, 1 : 1 + num_distractors, 0:2
        ]
        distractor_half = obs["asset_collision_half_extents"][
            0, 1 : 1 + num_distractors, 0:2
        ]
        surface = _rectangle_surface_distance(spawns, distractor_centers, distractor_half)
        violations = int((surface.amin(dim=1) < SPAWN_MARGIN_M).sum())
        self.assertGreater(
            violations,
            0,
            "scenario is too sparse to expose distractors-as-free-space",
        )
        # And the geometric core of it: some spawns are inside the solid body itself.
        inside = int((surface.amin(dim=1) <= 0.0).sum())
        self.assertGreater(inside, 0, "no spawn landed inside a distractor's own footprint")

    def test_acceptance_boundary_follows_each_shape_own_size(self):
        """The per-shape boundary is |half| + margin. Substituting ANY single radius for all
        three shapes flips at least one of these four assertions, which is what makes a
        size-blind regression fail."""
        margin = SPAWN_MARGIN_M
        for shape in ("sphere", "box", "pole"):
            with self.subTest(shape=shape):
                half = torch.tensor([_distractor_half_xy(shape)], dtype=torch.float32)
                radius = float(half.norm(dim=1))
                center = torch.zeros((1, 1, 2))
                for offset, expected in ((1e-3, True), (-1e-3, False)):
                    candidate = torch.tensor([[radius + margin + offset, 0.0]])
                    accepted = bool(
                        _SPAWN_ACCEPTED(candidate, center, half.unsqueeze(0), margin)[0]
                    )
                    self.assertEqual(accepted, expected)
        sphere_radius = float(torch.tensor(_distractor_half_xy("sphere")).norm())
        pole_radius = float(torch.tensor(_distractor_half_xy("pole")).norm())
        self.assertGreater(sphere_radius - pole_radius, 0.12)
        # A rule that used the sphere's radius for the pole would reject this point;
        # a rule that used the pole's radius for the sphere would accept it.
        between = torch.tensor([[pole_radius + margin + 1e-3, 0.0]])
        pole_half = torch.tensor([[_distractor_half_xy("pole")]], dtype=torch.float32)
        sphere_half = torch.tensor([[_distractor_half_xy("sphere")]], dtype=torch.float32)
        center = torch.zeros((1, 1, 2))
        self.assertTrue(bool(_SPAWN_ACCEPTED(between, center, pole_half, margin)[0]))
        self.assertFalse(bool(_SPAWN_ACCEPTED(between, center, sphere_half, margin)[0]))


class RouteTreatsDistractorsAsSolid(unittest.TestCase):
    """A route planned through a distractor would corrupt the occlusion statistics. Run through
    the REAL DeterministicAStarRoutePlanner on the geometry `_plan_target_routes` actually hands
    over, so this is the planner's own safety predicate, not a restatement of it."""

    ARENA = 20.0
    SUPPORT = 0.2068816086567407

    def _obs(self, num_envs, distractor_xy, distractor_shape, num_bars=0):
        total = 1 + 1 + num_bars
        position = torch.zeros((num_envs, total, 3))
        extents = torch.zeros((num_envs, total, 3))
        position[:, 1, 0:2] = torch.tensor(distractor_xy)
        extents[:, 1, 0:2] = torch.tensor(_distractor_half_xy(distractor_shape))
        if num_bars:
            position[:, 2:, 0:2] = torch.tensor([2.0, 2.0])
            extents[:, 2:, 0:2] = 0.35
        return {
            "env_bounds_min": torch.zeros((num_envs, 3)),
            "env_bounds_max": torch.tensor([self.ARENA, self.ARENA, 3.0]).repeat(num_envs, 1),
            "obstacle_position": position,
            "asset_collision_half_extents": extents,
        }

    def _capture(self, fn, obs, num_envs, num_bars):
        stub, manager = _route_stub(
            obs, num_envs, bar_offset=2, solid_offset=1, n_bars_active=num_bars,
            support=self.SUPPORT,
        )
        fn(stub, torch.arange(num_envs), connected_goal=True)
        centers, half = manager.captured[0]
        return centers[0].numpy().astype("float64"), half[0].numpy().astype("float64")

    def _plan(self, centers, half, start, goal):
        planner = _ROUTES.DeterministicAStarRoutePlanner(_ROUTES.RoutePlannerConfig())
        return planner.plan(
            start, goal, centers, half,
            [0.0, 0.0], [self.ARENA, self.ARENA],
            [self.SUPPORT, self.SUPPORT],
        )

    def _inflated(self, half):
        planner_config = _ROUTES.RoutePlannerConfig()
        return half + self.SUPPORT + planner_config.tracking_margin_m

    def test_route_through_a_distractor_is_blocked_and_was_not_before(self):
        start, goal = [3.0, 10.0], [17.0, 10.0]
        for shape in ("sphere", "box", "pole"):
            with self.subTest(shape=shape):
                obs = self._obs(2, [10.0, 10.0], shape)
                old_c, old_h = self._capture(_PLAN_ROUTES_HISTORICAL, obs, 2, 0)
                new_c, new_h = self._capture(_PLAN_ROUTES, obs, 2, 0)
                # Pre-change the planner saw NO obstacles at all: the distractor was free space.
                self.assertEqual(old_c.shape, (0, 2))
                self.assertEqual(new_c.shape, (1, 2))

                old_plan = self._plan(old_c, old_h, start, goal)
                self.assertEqual(old_plan.status, "ok")
                self.assertEqual(old_plan.waypoints_xy.shape, (2, 2))
                # That straight route runs straight through the distractor.
                self.assertFalse(
                    _ROUTES.segment_is_safe(
                        old_plan.waypoints_xy[0], old_plan.waypoints_xy[1],
                        [1.25 + self.SUPPORT] * 2,
                        [self.ARENA - 1.25 - self.SUPPORT] * 2,
                        new_c, self._inflated(new_h),
                    )
                )

                new_plan = self._plan(new_c, new_h, start, goal)
                self.assertEqual(new_plan.status, "ok")
                self.assertGreater(new_plan.waypoints_xy.shape[0], 2)
                for i in range(new_plan.waypoints_xy.shape[0] - 1):
                    self.assertTrue(
                        _ROUTES.segment_is_safe(
                            new_plan.waypoints_xy[i], new_plan.waypoints_xy[i + 1],
                            [1.25 + self.SUPPORT] * 2,
                            [self.ARENA - 1.25 - self.SUPPORT] * 2,
                            new_c, self._inflated(new_h),
                        ),
                        "segment %d of the re-planned route still crosses the %s" % (i, shape),
                    )

    def test_distractor_and_bar_geometry_stay_paired_through_the_planner(self):
        """One distractor plus active bars: the planner must receive the distractor's own small
        footprint on the distractor row and the bar's larger one on the bar rows."""
        obs = self._obs(2, [10.0, 10.0], "pole", num_bars=3)
        centers, half = self._capture(_PLAN_ROUTES, obs, 2, 3)
        self.assertEqual(centers.shape, (4, 2))
        self.assertEqual(half.shape, centers.shape)
        self.assertAlmostEqual(float(half[0][0]), 0.06, places=6)
        self.assertAlmostEqual(float(centers[0][0]), 10.0, places=6)
        for row in range(1, 4):
            self.assertAlmostEqual(float(half[row][0]), 0.35, places=6)
            self.assertAlmostEqual(float(centers[row][0]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
