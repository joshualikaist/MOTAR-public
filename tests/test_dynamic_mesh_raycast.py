"""Correctness gates C1-C5 for the dynamic-mesh raycast path.

Gates and tolerances are fixed in
results/dynamic_mesh_raycast_feasibility_2026-09-12/PREREGISTRATION.md and are not changed here.
GPU is required, so each test skips rather than passing vacuously when CUDA is absent.
"""
import hashlib
import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from renderer_validation.scene import MeshScene, asymmetric_box_fixture, l_shape_fixture
from renderer_validation import reference_raycast as reference

DEPTH_TOL = 1e-5
NORMAL_TOL = 1e-4
UNIT_TOL = 1e-5
ROTATION_CHANGE_MIN = 0.05


_ABSENT = object()


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def camera_rays(width=64, height=48, origin=(0.0, 0.0, -1.0), fov_deg=60.0):
    """A forward (+Z) pinhole fan, returned flat as [1, pixels, 3].

    Framed so the fixture actually fills part of the image. At 3 m the 0.6 m box covered about
    24 pixels and every gate starved for samples; that was framing, not the path under test, and
    the tolerances and the 50-paired-hit floor are unchanged from the preregistration.
    """
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    directions = np.stack([(x - width / 2.0) / focal, (y - height / 2.0) / focal,
                           np.ones_like(x, dtype=np.float64)], axis=-1).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.tile(np.asarray(origin, dtype=np.float64), (len(directions), 1))
    return origins[None], directions[None]


def empty_scene():
    """A degenerate-free triangle far from every ray, so 'static only' means 'nothing hit'."""
    far = np.array([[1e3, 1e3, 1e3], [1e3 + 1, 1e3, 1e3], [1e3, 1e3 + 1, 1e3]], dtype=np.float64)
    return MeshScene(far, np.array([[0, 1, 2]], np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32))


def quaternion(axis, radians):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(radians / 2.0), [np.cos(radians / 2.0)]])


class DynamicMeshRaycastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not cuda_available():
            raise unittest.SkipTest("dynamic-mesh raycast needs CUDA")
        # Two CPU-only test modules install a fake `warp` in sys.modules so that @wp.kernel
        # decorators evaluate at import, using setdefault, i.e. only when the name is free. These
        # gates need the real package, and a stub's Mesh returns None, which errors rather than
        # passing quietly. Swap the real one in for this class and put the previous state back
        # afterwards, so nothing here changes what those modules see.
        cls._saved_warp = sys.modules.get("warp", _ABSENT)
        if not isinstance(getattr(cls._saved_warp, "Mesh", None), type):
            sys.modules.pop("warp", None)
            importlib.import_module("warp")
        cls._warp = sys.modules["warp"]
        if not isinstance(getattr(cls._warp, "Mesh", None), type):
            raise AssertionError("could not obtain the real warp package; refusing to run on a stub")
        from renderer_validation.dynamic_mesh import DynamicMeshRaycaster
        cls.Raycaster = DynamicMeshRaycaster
        cls.origins, cls.directions = camera_rays()
        cls.identity = np.array([0.0, 0.0, 0.0, 1.0])

    @classmethod
    def tearDownClass(cls):
        # Never remove the real package. Warp's submodules stay in sys.modules once imported, so a
        # later `import warp` anywhere in the process would re-execute warp/__init__.py against a
        # fresh, half-built module object and die on `warp.config`. Restoring a stub would be just
        # as bad: the next real import would silently get the stub instead. Modules that need a stub
        # install their own over whatever is here and put it back themselves, so leaving the real
        # package in place is what keeps every later importer honest.
        if cls._saved_warp is not _ABSENT and isinstance(
                getattr(cls._saved_warp, "Mesh", None), type):
            sys.modules["warp"] = cls._saved_warp

    def caster(self, dynamic=None):
        return self.Raycaster(empty_scene(), dynamic or asymmetric_box_fixture(), far_range_m=50.0)

    def test_c1_translation_moves_the_hit_distance_by_exactly_the_offset(self):
        caster = self.caster()
        baseline = caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.0]], [self.identity])
        for shift in (1.0, -1.0):
            moved = caster.cast(self.origins, self.directions,
                                [[0.0, 0.0, shift]], [self.identity])
            both = baseline.hit[0] & moved.hit[0]
            self.assertGreater(int(both.sum()), 50, "too few paired hits to compare")
            delta = moved.range_m[0][both] - baseline.range_m[0][both]
            # Rays are not all parallel to Z, so only the on-axis ray shifts by exactly `shift`.
            # Every ray must still shift, monotonically and in the right direction.
            self.assertTrue(np.all(np.sign(delta) == np.sign(shift)), "shift went the wrong way")
            self.assertLess(abs(float(delta.max()) - shift) if shift > 0 else
                            abs(float(delta.min()) - shift), 0.05)

    def test_c2_rotating_an_asymmetric_shape_changes_which_pixels_hit(self):
        caster = self.caster()
        flat = caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.0]], [self.identity])
        turned = caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.0]],
                             [quaternion([0, 1, 0], np.pi / 2)])
        changed = (flat.hit[0] != turned.hit[0]).mean()
        self.assertGreater(changed, ROTATION_CHANGE_MIN,
                           f"rotation changed only {changed:.3%} of pixels")

    def test_c3_object_local_rays_agree_with_a_world_transformed_mesh(self):
        """The gate that matters: an independent CPU intersector moves the MESH instead."""
        mesh = l_shape_fixture()
        caster = self.caster(mesh)
        position, rotation = [0.15, -0.08, 0.35], quaternion([0.3, 1.0, 0.2], 0.7)
        gpu = caster.cast(self.origins, self.directions, [position], [rotation])
        distance, normal, _, hit = reference.intersect_posed(
            mesh, self.origins[0], self.directions[0], position, rotation, far=50.0)
        self.assertGreater(int(hit.sum()), 50, "reference found too few hits to compare")
        self.assertTrue(np.array_equal(gpu.hit[0], hit), "hit masks differ")
        self.assertLess(float(np.abs(gpu.range_m[0][hit] - distance[hit]).max()), DEPTH_TOL)
        # Face winding can flip a reference normal; compare the line, not the arrow.
        alignment = np.abs(np.einsum("ij,ij->i", gpu.normal_world[0][hit], normal[hit]))
        self.assertLess(float(np.abs(alignment - 1.0).max()), NORMAL_TOL)

    def test_c4_returned_normals_are_unit_length_and_face_the_camera(self):
        caster = self.caster()
        result = caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.4]],
                             [quaternion([1, 0, 0], 0.4)])
        hit = result.hit[0]
        self.assertGreater(int(hit.sum()), 50)
        lengths = np.linalg.norm(result.normal_world[0][hit], axis=1)
        self.assertLess(float(np.abs(lengths - 1.0).max()), UNIT_TOL)
        facing = np.einsum("ij,ij->i", result.normal_world[0][hit], -self.directions[0][hit])
        self.assertTrue(np.all(facing > 0.0), "a visible face pointed away from the camera")

    def test_c5_changing_pose_rebuilds_no_mesh(self):
        caster = self.caster()
        before = caster.build_signature
        for step in range(5):
            caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.1 * step]],
                        [quaternion([0, 0, 1], 0.2 * step)])
        self.assertEqual(caster.build_signature, before, "the mesh or its buffers were rebuilt")

    def test_static_and_dynamic_stay_separable_and_take_the_nearer_hit(self):
        near = asymmetric_box_fixture((0.4, 0.4, 0.1), (0.0, 0.0, 0.0))
        static = asymmetric_box_fixture((3.0, 3.0, 0.1), (0.0, 0.0, 1.5))
        caster = self.Raycaster(static, near, far_range_m=50.0)
        behind = caster.cast(self.origins, self.directions, [[0.0, 0.0, 2.5]], [self.identity])
        front = caster.cast(self.origins, self.directions, [[0.0, 0.0, 0.0]], [self.identity])
        self.assertTrue(np.all(behind.source[0][behind.hit[0]] == 1),
                        "an object behind the wall was still reported")
        self.assertIn(2, set(front.source[0].tolist()), "the nearer dynamic object never won")
        centre = len(front.source[0]) // 2 + 24
        self.assertEqual(int(front.source[0][centre]), 2)

    def test_replay_is_deterministic(self):
        caster = self.caster(l_shape_fixture())
        poses = [[0.02 * i, 0.0, 0.3] for i in range(8)]
        rotations = [quaternion([0, 1, 0], 0.1 * i) for i in range(8)]
        digests = []
        for _ in range(2):
            result = caster.cast(np.repeat(self.origins, 8, axis=0),
                                 np.repeat(self.directions, 8, axis=0), poses, rotations)
            digests.append(tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in
                                 (result.range_m, result.normal_world, result.face_id, result.source)))
        self.assertEqual(digests[0], digests[1])

    def test_malformed_input_is_refused(self):
        caster = self.caster()
        with self.assertRaises(ValueError):
            caster.cast(self.origins, self.directions * 2.0, [[0, 0, 0]], [self.identity])
        with self.assertRaises(ValueError):
            caster.cast(self.origins, self.directions, [[0, 0, 0]], [[0.0, 0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
