"""V1: the target's silhouette changes and nothing else does.

Contract: docs/v1_shared_airframe_contract.md. Every invariant here is checked by calling the
simulator's real extractors or by measuring the generated file, never by restating a number.
"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from renderer_validation.urdf_asset import (load_urdf_asset, orientation_report,
                                            compare_backends, rotation_from_rpy)
import generate_shared_airframe as generator

OBJECTS = ROOT / "resources/models/environment_assets/objects"
V2 = OBJECTS / "navrl_target_drone_v2.urdf"
V3 = OBJECTS / "navrl_target_drone_v3.urdf"
ASSET_LOADER = ROOT / "aerial_gym/env_manager/asset_loader.py"
TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"


def lift(path, name, namespace):
    """Compile one top-level def out of a module whose imports cannot be satisfied here.

    asset_loader.py does `from isaacgym import gymapi` at module scope, so the module itself is
    unimportable; the function under test is pure. This mirrors the pattern the distractor tests
    already use rather than inventing a second one.
    """
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{path.name} no longer defines a top-level {name}()")


HALF_EXTENTS = lift(ASSET_LOADER, "_urdf_collision_half_extents", {"ET": ElementTree})


def tokens(path, tag):
    """Whitespace-insensitive comparison of one element, so reformatting is not a difference."""
    element = ElementTree.parse(str(path)).getroot().find(f"link[@name='base_link']/{tag}")
    return None if element is None else ElementTree.tostring(element, encoding="unicode").split()


class PhysicsIsUntouchedTest(unittest.TestCase):
    """The simulator reads collision only. These call the readers rather than describing them."""

    def test_asset_loader_returns_the_same_half_extents(self):
        self.assertEqual(HALF_EXTENTS(str(V3)), HALF_EXTENTS(str(V2)))
        self.assertEqual(HALF_EXTENTS(str(V3)), [0.1415, 0.1415, 0.06])

    def test_the_task_finds_the_same_base_link_collision_box(self):
        query = "link[@name='base_link']/collision/geometry/box"
        source = TASK.read_text()
        self.assertIn(query, source, "NavRLTask no longer uses this lookup; update this test")
        sizes = [ElementTree.parse(str(p)).getroot().find(query).get("size") for p in (V2, V3)]
        self.assertEqual(sizes[0], sizes[1])

    def test_inertial_and_collision_are_carried_over_unchanged(self):
        for tag in ("inertial", "collision"):
            self.assertEqual(tokens(V3, tag), tokens(V2, tag), tag)

    def test_physics_lives_only_on_base_link(self):
        """The asset is thirteen links now; what must not spread is mass and collision.

        It was one link until warp_asset.py refused it: that loader indexes a per-link name list
        with a per-mesh counter, so it assumes one mesh per link and raised IndexError on thirteen
        visuals in one link. The interceptor URDF already uses one visual per link, so the asset
        follows it. The invariant this test was written for is unchanged and is asserted directly
        rather than through a link count.
        """
        root = ElementTree.parse(str(V3)).getroot()
        links = root.findall("link")
        self.assertEqual(links[0].get("name"), "base_link")
        with_collision = [l.get("name") for l in links if l.find("collision") is not None]
        self.assertEqual(with_collision, ["base_link"])
        for link in links[1:]:
            mass = link.find("inertial/mass")
            self.assertIsNotNone(mass, link.get("name"))
            self.assertEqual(float(mass.get("value")), 0.0, link.get("name"))
        joints = root.findall("joint")
        self.assertEqual(len(joints), len(links) - 1)
        self.assertTrue(all(j.get("type") == "fixed" for j in joints))
        self.assertTrue(all(j.find("parent").get("link") == "base_link" for j in joints))

    def test_no_visual_geometry_escapes_the_collision_box(self):
        """A silhouette larger than the collision proxy would be visible where nothing can be hit.

        Measured from the DECLARED shapes, not from tessellated vertices. A tessellated cylinder
        is inscribed, so its vertices under-measure the shape whenever the true extreme falls
        between sampled azimuths: the shortfall reaches 1.22 mm at 16 segments while the real
        margin here is 0.218 mm, so a vertex-based test can pass on geometry that escapes.
        """
        half = np.array(HALF_EXTENTS(str(V3)))
        root = ElementTree.parse(str(V3)).getroot()
        # Each part's placement moved from its <visual> origin to its fixed joint when the asset
        # became multi-link. Reading only the visual origin would put every part at the centre and
        # pass this test having measured nothing, so compose the joint origin with it.
        joint_of = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            joint_of[joint.find("child").get("link")] = (
                np.array([float(v) for v in origin.get("xyz").split()]),
                [float(v) for v in origin.get("rpy").split()])
        worst = np.zeros(3)
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                shape = list(visual.find("geometry"))[0]
                origin = visual.find("origin")
                centre = np.array([float(v) for v in origin.get("xyz").split()])
                roll, pitch, yaw = (float(v) for v in origin.get("rpy").split())
                offset, joint_rpy = joint_of.get(link.get("name"), (np.zeros(3), [0.0, 0.0, 0.0]))
                centre = offset + rotation_from_rpy(*joint_rpy) @ centre
                roll, pitch, yaw = (a + b for a, b in zip((roll, pitch, yaw), joint_rpy))
                if shape.tag == "box":
                    extent = np.array([float(v) for v in shape.get("size").split()]) / 2.0
                elif shape.tag == "sphere":
                    extent = np.full(3, float(shape.get("radius")))
                else:
                    radius, length = float(shape.get("radius")), float(shape.get("length"))
                    # Analytic support of a cylinder: exact half-extent along each world axis.
                    axis = rotation_from_rpy(roll, pitch, yaw) @ np.array([0.0, 0.0, 1.0])
                    extent = (np.abs(axis) * length / 2.0
                              + np.sqrt(np.maximum(1.0 - axis ** 2, 0.0)) * radius)
                worst = np.maximum(worst, np.abs(centre) + extent)
        self.assertTrue(np.all(worst <= half),
                        f"declared visual extent {worst} exceeds collision half extents {half}")
        # The margin is under a millimetre, which is why the measurement method matters here.
        self.assertLess(float((half - worst).min()), 0.01)


class SilhouetteTest(unittest.TestCase):
    def test_the_target_is_no_longer_a_single_box(self):
        visuals = ElementTree.parse(str(V3)).getroot().findall("link/visual")
        kinds = [child.tag for visual in visuals for child in visual.find("geometry")]
        self.assertEqual(len(visuals), 13)                 # body, 4 arms, 4 motors, 4 propellers
        self.assertEqual(kinds.count("box"), 1)
        self.assertEqual(kinds.count("cylinder"), 12)

    def test_geometry_is_closed_wound_outward_and_agreed_by_both_backends(self):
        asset = load_urdf_asset(V3)
        self.assertTrue(orientation_report(asset.mesh)["closed_and_consistently_wound_outward"])
        self.assertEqual(compare_backends(V3), [])

    def test_width_is_the_propeller_span_the_urdf_documents(self):
        """0.2825634 m, from motor radius plus propeller radius; not a number chosen here."""
        motors, arm, hub = generator.read_airframe_layout()
        expected = 2.0 * (abs(motors[0][0]) + generator.read_prop_radius())
        vertices = load_urdf_asset(V3).mesh.vertices.astype(np.float64)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        # MeshScene stores float32 for Warp, so compare relatively at that precision rather
        # than to nine decimals, where storage error reads as a geometry error.
        epsilon = float(np.finfo(np.float32).eps)
        for axis in (0, 1):
            self.assertLess(abs(float(extent[axis]) / expected - 1.0), epsilon)
        self.assertLess(abs(float(extent[2]) / 0.12 - 1.0), epsilon)
        self.assertLess(abs(float(extent[0]) / 0.283 - 1.0), 0.01)

    def test_arms_reach_from_the_body_to_the_motors(self):
        motors, arm, hub = generator.read_airframe_layout()
        self.assertAlmostEqual(arm[1], float(np.hypot(motors[0][0], motors[0][1])), places=6)

    def test_no_material_is_the_old_target_red(self):
        """The point of the change is that colour alone should not find the target."""
        asset = load_urdf_asset(V3)
        for name, rgba in zip(asset.material_names, asset.material_rgba):
            red, green, blue = rgba[:3]
            self.assertLess(red - max(green, blue), 0.1, f"{name} is a dominant red")


class GeneratorTest(unittest.TestCase):
    def test_the_committed_file_is_exactly_what_the_generator_produces(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/generate_shared_airframe.py"),
                                 "--check"], capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_dimension_is_read_rather_than_written_here(self):
        motors, arm, hub = generator.read_airframe_layout()
        self.assertEqual(len(motors), 4)
        self.assertAlmostEqual(generator.read_prop_radius(), 0.0635, places=9)
        source = (ROOT / "tools/generate_shared_airframe.py").read_text()
        self.assertNotIn("0.0777817", source, "motor positions must be read from the URDF")
        self.assertNotIn("0.0635", source, "the propeller radius must be read from the task")


class ConfigurationTest(unittest.TestCase):
    def test_v3_is_opt_in_and_the_default_is_unchanged(self):
        source = (ROOT / "aerial_gym/config/env_config/navrl_bars_env.py").read_text()
        self.assertIn('os.environ.get(\n            "NAVRL_TARGET_APPEARANCE", "v2"\n        )', source)
        self.assertIn("navrl_physical_target_v3_params", source)

    def test_the_v3_params_inherit_v2_and_change_only_the_file(self):
        source = (ROOT / "aerial_gym/config/asset_config/env_object_config.py").read_text()
        tree = ast.parse(source)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("navrl_physical_target_v3_params", classes)
        node = classes["navrl_physical_target_v3_params"]
        self.assertEqual([base.id for base in node.bases], ["navrl_physical_target_v2_params"])
        assigned = [target.id for statement in node.body if isinstance(statement, ast.Assign)
                    for target in statement.targets if isinstance(target, ast.Name)]
        self.assertEqual(assigned, ["file"])


if __name__ == "__main__":
    unittest.main()
