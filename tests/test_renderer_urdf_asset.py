"""CPU tests for URDF asset loading. Contract: docs/renderer_urdf_loader_v1_contract.md."""
import math
from pathlib import Path
import sys
import textwrap
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from renderer_validation.urdf_asset import (UrdfError, load_urdf_asset, box_mesh, sphere_mesh,
                                            cylinder_mesh, rotation_from_rpy, origin_transform,
                                            convex_outward_violations, open_edges, signed_volume,
                                            compare_backends, BACKENDS, DEFAULT_BACKEND,
                                            orientation_report, winding_inconsistent_edges,
                                            SPHERE_SEGMENTS, SPHERE_RINGS, CYLINDER_SEGMENTS,
                                            DEFAULT_MATERIAL_NAME)
from renderer_validation.scene import MeshScene

INTERCEPTOR = ROOT / "resources/robots/quad/quad_navrl_ref5in_v2.urdf"
TARGET = ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v2.urdf"
DISTRACTORS = [ROOT / f"resources/models/environment_assets/objects/navrl_distractor_{n}.urdf"
               for n in ("sphere", "box", "pole")]


def single(points, faces):
    return MeshScene(points, faces, [0] * len(faces), [0] * len(faces))


def write(directory, body, name="robot.urdf"):
    path = Path(directory) / name
    path.write_text(textwrap.dedent(body).strip())
    return path


class RotationConventionTest(unittest.TestCase):
    def test_rpy_is_fixed_axis_z_then_y_then_x(self):
        roll, pitch, yaw = 0.3, -0.7, 1.1
        rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        np.testing.assert_allclose(rotation_from_rpy(roll, pitch, yaw), rz @ ry @ rx, atol=1e-14)

    def test_rotation_is_orthonormal_and_right_handed(self):
        matrix = rotation_from_rpy(0.4, 1.2, -2.0)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_missing_origin_is_the_identity(self):
        np.testing.assert_array_equal(origin_transform(None), np.eye(4))

    def test_non_finite_and_malformed_origins_are_refused(self):
        import xml.etree.ElementTree as ElementTree
        for attributes in ('xyz="1 2 nan"', 'xyz="1 2"', 'rpy="0 0 abc"'):
            with self.assertRaises(UrdfError):
                origin_transform(ElementTree.fromstring(f"<origin {attributes}/>"))


class TessellationTest(unittest.TestCase):
    """A closed, outward-wound surface whose volume converges is the evidence that this is a shape."""

    def test_every_primitive_is_closed_and_wound_outward(self):
        for name, (points, faces) in (("box", box_mesh([0.4, 0.3, 0.2])),
                                      ("sphere", sphere_mesh(0.05)),
                                      ("cylinder", cylinder_mesh(0.01, 0.11))):
            mesh = single(points, faces)
            self.assertEqual(convex_outward_violations(mesh), 0, name)
            self.assertTrue(orientation_report(mesh)["closed_and_consistently_wound_outward"], name)
            self.assertEqual(winding_inconsistent_edges(mesh), 0, name)

    def test_box_tessellation_is_exact_before_float32_storage(self):
        """Separate the two layers: the tessellation is exact, the stored vertices are float32.

        MeshScene stores float32 because Warp needs it, which costs about one part in 1e7. A
        volume test that ignores this looks like a tessellation error and is not one.
        """
        points, faces = box_mesh([0.283, 0.283, 0.12])
        corners = np.asarray(points, dtype=np.float64)[np.asarray(faces)]
        exact = 0.283 * 0.283 * 0.12
        in_float64 = float(np.sum(np.einsum("ij,ij->i", corners[:, 0],
                                            np.cross(corners[:, 1], corners[:, 2]))) / 6.0)
        self.assertEqual(in_float64, exact)
        stored = single(points, faces)
        self.assertEqual(stored.vertices.dtype, np.float32)
        self.assertLess(abs(signed_volume(stored) / exact - 1.0), float(np.finfo(np.float32).eps))

    def test_sphere_and_cylinder_volume_converge_from_below(self):
        exact = 4.0 / 3.0 * math.pi * 0.05 ** 3
        coarse = signed_volume(single(*sphere_mesh(0.05, 16, 8)))
        fine = signed_volume(single(*sphere_mesh(0.05, 64, 32)))
        self.assertLess(coarse, fine)           # an inscribed surface only grows with resolution
        self.assertLess(fine, exact)
        self.assertLess(abs(fine / exact - 1.0), 0.01)
        exact = math.pi * 0.01 ** 2 * 0.11
        coarse = signed_volume(single(*cylinder_mesh(0.01, 0.11, 16)))
        fine = signed_volume(single(*cylinder_mesh(0.01, 0.11, 128)))
        self.assertLess(coarse, fine)
        self.assertLess(fine, exact)
        self.assertLess(abs(fine / exact - 1.0), 0.01)

    def test_cylinder_axis_is_z_and_centred(self):
        points, _ = cylinder_mesh(0.06, 1.6)
        points = np.asarray(points)
        self.assertAlmostEqual(points[:, 2].min(), -0.8, places=12)
        self.assertAlmostEqual(points[:, 2].max(), 0.8, places=12)
        radius = np.linalg.norm(points[:, :2], axis=1)
        self.assertAlmostEqual(float(radius.max()), 0.06, places=12)

    def test_no_degenerate_triangle_at_the_sphere_poles(self):
        points, faces = sphere_mesh(0.05)
        single(points, faces)   # MeshScene refuses zero-area triangles
        self.assertEqual(len(faces), SPHERE_SEGMENTS * 2 + SPHERE_SEGMENTS * (SPHERE_RINGS - 2) * 2)

    def test_cylinder_triangle_count_follows_the_declared_segments(self):
        self.assertEqual(len(cylinder_mesh(1.0, 1.0)[1]), CYLINDER_SEGMENTS * 4)


class RealAssetTest(unittest.TestCase):
    def test_interceptor_has_nine_links_and_three_materials(self):
        asset = load_urdf_asset(INTERCEPTOR)
        self.assertEqual(asset.robot_name, "quadrotor_ref5in_v2")
        self.assertEqual(len(asset.link_names), 9)
        self.assertEqual(asset.link_names[0], "base_link")
        self.assertEqual(list(asset.material_names), ["White", "Orange", "Blue"])
        self.assertEqual(len(set(asset.mesh.face_instance.tolist())), 9)
        self.assertEqual(asset.default_material_visuals, 0)

    def test_every_shipped_asset_is_closed_and_wound_outward(self):
        for path in [INTERCEPTOR, TARGET] + DISTRACTORS:
            for backend in BACKENDS:
                asset = load_urdf_asset(path, backend)
                self.assertEqual(asset.backend, backend)
                self.assertEqual(convex_outward_violations(asset.mesh), 0, path.name)
                self.assertTrue(orientation_report(asset.mesh)["closed_and_consistently_wound_outward"],
                                f"{path.name} via {backend}")

    def test_target_is_the_declared_red_box(self):
        asset = load_urdf_asset(TARGET)
        self.assertEqual(list(asset.material_names), ["TargetRed"])
        np.testing.assert_allclose(asset.material_rgba[0], [0.9, 0.2, 0.2, 0.95])
        self.assertLess(abs(signed_volume(asset.mesh) / (0.283 * 0.283 * 0.12) - 1.0),
                        float(np.finfo(np.float32).eps))

    def test_joint_origins_place_the_motors_where_the_urdf_says(self):
        """The four motors sit at +-0.0777817 in x and y, per the URDF's own arithmetic.

        Read from the link transforms, not from a mean of vertices. A mean of vertices is not a
        kinematic quantity: two tessellations of one sphere average to slightly different points,
        so it would make the test depend on how finely a backend subdivides.
        """
        for backend in BACKENDS:
            asset = load_urdf_asset(INTERCEPTOR, backend)
            places = {name: asset.link_transforms[i][:3, 3]
                      for i, name in enumerate(asset.link_names)}
            for index in range(4):
                np.testing.assert_allclose(np.abs(places[f"motor_{index}"][:2]),
                                           [0.0777817, 0.0777817], atol=1e-12)
                self.assertAlmostEqual(float(places[f"motor_{index}"][2]), 0.0, places=12)
            np.testing.assert_allclose(places["base_link"], [0, 0, 0], atol=1e-15)

    def test_loading_is_deterministic(self):
        first, second = load_urdf_asset(INTERCEPTOR), load_urdf_asset(INTERCEPTOR)
        np.testing.assert_array_equal(first.mesh.vertices, second.mesh.vertices)
        np.testing.assert_array_equal(first.mesh.triangles, second.mesh.triangles)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.source_sha256, second.source_sha256)

    def test_receipt_records_the_backend_the_library_versions_and_the_colour_choice(self):
        record = load_urdf_asset(INTERCEPTOR).as_dict()
        self.assertEqual(record["backend"], "urdfpy")
        self.assertIn("urdfpy", record["library_versions"])
        self.assertIn("trimesh", record["library_versions"])
        self.assertIn("unconverted", record["colour_note"])
        self.assertEqual(len(record["link_transforms"]), 9)
        self.assertEqual(load_urdf_asset(INTERCEPTOR, "builtin").as_dict()["triangles"], 736)

    def test_the_library_subdivides_curves_more_finely_than_the_declared_counts(self):
        """Recorded because it changes ray-cast cost, and because the receipt must explain it."""
        library = load_urdf_asset(INTERCEPTOR, "urdfpy")
        builtin = load_urdf_asset(INTERCEPTOR, "builtin")
        self.assertGreater(len(library.mesh.triangles), len(builtin.mesh.triangles))
        for asset in (library, builtin):
            self.assertTrue(orientation_report(asset.mesh)["closed_and_consistently_wound_outward"])

    def test_the_two_backends_agree_on_every_asset_on_the_appearance_path(self):
        """Agreement is about what the URDF states, not about how finely it is subdivided."""
        for path in [INTERCEPTOR, TARGET] + DISTRACTORS:
            self.assertEqual(compare_backends(path), [], path.name)


class RefusalTest(unittest.TestCase):
    """Both backends must honour one contract.

    Each of these was a real difference. urdfpy accepted a revolute joint that declares limits and
    evaluated it at the zero configuration, so an articulated robot loaded as a pose nobody chose.
    It substituted the default colour for a material reference that was never declared. And a
    zero-size box reached the renderer and failed there as a degenerate triangle, naming the
    triangle rather than the shape. The library backend now refuses all three.
    """

    def setUp(self):
        import tempfile
        self.directory = tempfile.mkdtemp()

    def refused_by_both(self, body, name, pattern):
        path = write(self.directory, body, name)
        for backend in BACKENDS:
            with self.assertRaises(UrdfError, msg=backend) as caught:
                load_urdf_asset(path, backend)
            self.assertRegex(str(caught.exception), pattern, backend)

    def test_articulated_joints_are_refused_rather_than_posed_at_zero(self):
        self.refused_by_both("""
            <robot name="m">
              <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
              <link name="b"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
              <joint name="j" type="revolute"><parent link="a"/><child link="b"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
                <origin xyz="0 0 2"/></joint>
            </robot>""", "revolute_with_limits.urdf", "fixed joints only")

    def test_undeclared_material_reference_is_refused_rather_than_defaulted(self):
        self.refused_by_both("""
            <robot name="m"><link name="a"><visual><geometry><box size="1 1 1"/></geometry>
            <material name="Nowhere"/></visual></link></robot>""",
            "undeclared_material.urdf", "undeclared material")

    def test_non_positive_dimensions_are_refused_naming_the_shape(self):
        for index, geometry in enumerate(('<box size="1 0 1"/>', '<sphere radius="0"/>',
                                          '<cylinder radius="1" length="-1"/>')):
            self.refused_by_both(f"""
                <robot name="m"><link name="a"><visual><geometry>
                {geometry}</geometry></visual></link></robot>""",
                f"degenerate_{index}.urdf", "(non-positive|must be positive)")

    def test_broken_link_graphs_are_refused(self):
        self.refused_by_both("""
            <robot name="m">
              <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
              <link name="b"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
            </robot>""", "two_roots.urdf", "(one root link|not all connected)")
        self.refused_by_both("""
            <robot name="m">
              <link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
              <joint name="j" type="fixed"><parent link="a"/><child link="ghost"/></joint>
            </robot>""", "ghost_child.urdf", "(unknown link|ghost)")

    def test_malformed_xml_and_wrong_root_are_refused(self):
        self.refused_by_both("<robot name='m'><link", "malformed.urdf", "(well-formed|could not load)")
        self.refused_by_both("<sdf/>", "not_a_robot.urdf", "(expected <robot>|could not load)")

    def test_a_visual_without_a_material_uses_the_declared_default_and_is_counted(self):
        path = write(self.directory, """
            <robot name="m"><link name="a"><visual><geometry>
            <box size="1 1 1"/></geometry></visual></link></robot>""", "nomat.urdf")
        for backend in BACKENDS:
            asset = load_urdf_asset(path, backend)
            self.assertEqual(list(asset.material_names), [DEFAULT_MATERIAL_NAME], backend)
            self.assertEqual(asset.default_material_visuals, 1, backend)

    def test_a_robot_with_no_visual_geometry_is_refused(self):
        self.refused_by_both("""
            <robot name="m"><link name="a"><collision><geometry>
            <box size="1 1 1"/></geometry></collision></link></robot>""",
            "no_visual.urdf", "no visual geometry")

    def test_an_unknown_backend_name_is_refused(self):
        with self.assertRaisesRegex(UrdfError, "backend must be one of"):
            load_urdf_asset(TARGET, "guess")


class MeshSupportTest(unittest.TestCase):
    """Mesh files are the reason to use a library at all; the builtin parser cannot read them."""

    MESH_ROBOT = ROOT / "resources/robots/BlueROV/rov.urdf"   # the only mesh URDF with fixed joints only

    ARTICULATED_MESH_ROBOT = ROOT / "resources/robots/morphy/morphy.urdf"

    def test_the_library_backend_loads_a_mesh_robot_the_builtin_one_refuses(self):
        if not self.MESH_ROBOT.exists():
            self.skipTest("no mesh-bearing URDF is shipped in this checkout")
        asset = load_urdf_asset(self.MESH_ROBOT, "urdfpy")
        self.assertGreater(len(asset.mesh.triangles), 1000)
        self.assertTrue(any(kind == "mesh" for _, kind, _, _ in asset.shapes))
        report = orientation_report(asset.mesh)
        # Two separate facts, and conflating them is what made this test wrong at first.
        # The hull is concave, so the convex centroid test reports tens of thousands of
        # "violations" that mean nothing here. And the shipped STL is genuinely not watertight.
        # Neither is a loader defect; the loader's job is to report them.
        self.assertGreater(convex_outward_violations(asset.mesh), 0)
        self.assertGreater(report["signed_volume_m3"], 0.0)
        self.assertGreater(report["open_edges"], 0)
        self.assertFalse(report["closed_and_consistently_wound_outward"])
        with self.assertRaisesRegex(UrdfError, "out of scope"):
            load_urdf_asset(self.MESH_ROBOT, "builtin")

    def test_an_articulated_mesh_robot_is_still_refused_by_both(self):
        """Mesh support does not relax the joint rule; morphy has revolute joints."""
        if not self.ARTICULATED_MESH_ROBOT.exists():
            self.skipTest("morphy is not shipped in this checkout")
        for backend in BACKENDS:
            with self.assertRaisesRegex(UrdfError, "fixed joints only"):
                load_urdf_asset(self.ARTICULATED_MESH_ROBOT, backend)


class IsolationTest(unittest.TestCase):
    def test_loader_imports_no_simulator_task_detector_or_policy(self):
        """Isolation means no aerial_gym. A general-purpose URDF parser is not part of it."""
        import ast
        source = (ROOT / "tools/renderer_validation/urdf_asset.py").read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("aerial_gym", imported)
        self.assertEqual(imported, {"dataclasses", "hashlib", "math", "pathlib", "xml", "numpy",
                                    "trimesh", "urdfpy"})

    def test_importing_the_loader_does_not_import_the_simulator(self):
        """Check a fresh interpreter, not this one.

        Asserting on sys.modules here only says whether some other test in the same run imported
        the simulator, which is not a property of this loader at all. A subprocess answers the
        question that was meant.
        """
        import subprocess
        script = ("import sys; sys.path.insert(0, %r);"
                  "from renderer_validation.urdf_asset import load_urdf_asset;"
                  "print([n for n in sys.modules if n == 'aerial_gym' or n.startswith('aerial_gym.')])"
                  % str(ROOT / "tools"))
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
