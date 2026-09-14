"""CPU tests of new generic background geometry and counterfactual contracts."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from renderer_validation.background_scene import background_fixture, cycle_materials
from renderer_validation.background_validation import background_appearance, appearance_variants, evaluate_background, true_fraction
from renderer_validation.gbuffer import GBuffer
from renderer_validation.scene import Camera
from renderer_validation.validation import tensor_hash
import run_renderer_background_validation as runner


def synthetic():
    fixture, camera = background_fixture(), Camera(width=16, height=16)
    mesh = fixture.mesh
    chosen = [np.flatnonzero((mesh.face_instance == n) & (mesh.face_material == n))[0] for n in range(4)]
    faces = torch.tensor(np.repeat(chosen, 64).reshape(1,16,16), dtype=torch.int32)
    normals = torch.tensor(np.repeat([[-1.,0,0], [0,-1.,0], [1.,0,0], [0,0,-1.]], 64, axis=0), dtype=torch.float32).reshape(1,16,16,3)
    ranges = torch.linspace(2, 9, 256).reshape(1,16,16)
    instances = torch.tensor(mesh.face_instance.copy(), dtype=torch.int32)[faces.long()]
    valid = torch.ones_like(faces, dtype=torch.bool)
    return fixture, camera, GBuffer(ranges, ranges*.9, normals, faces, instances, valid)


class BackgroundSceneTest(unittest.TestCase):
    def test_contains_all_parts_and_objects(self):
        f = background_fixture()
        self.assertEqual(set(f.face_part), set(range(5)))
        self.assertEqual(set(f.mesh.face_instance), set(range(4)))
        self.assertEqual(len(f.mesh.triangles), 144)
        self.assertEqual(len(set(f.face_patch)), 62)

    def test_all_instances_share_four_materials(self):
        m = background_fixture().mesh
        for n in range(4):
            self.assertEqual(set(m.face_material[m.face_instance == n]), set(range(4)))

    def test_surface_triangles_share_a_material(self):
        f = background_fixture()
        for patch_id in set(f.face_patch):
            self.assertEqual(len(set(f.mesh.face_material[f.face_patch == patch_id])), 1)

    def test_floor_and_wall_winding(self):
        f = background_fixture()
        p = f.mesh.vertices[f.mesh.triangles]
        normals = np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
        self.assertTrue((normals[f.face_part==0,1] < 0).all())
        self.assertTrue((normals[f.face_part==1,2] < 0).all())

    def test_column_side_normals_are_outward(self):
        f = background_fixture()
        p = f.mesh.vertices[f.mesh.triangles]
        n = np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
        sides = (f.face_part==2) & (np.abs(n[:,1]) < 1e-6)
        radial = p.mean(1)-[-1.6,0,4.5]
        radial[:,1] = 0
        self.assertTrue((np.sum(n[sides]*radial[sides],axis=1)>0).all())

    def test_hinge_changes_only_leaf_vertices(self):
        a,b = background_fixture(0),background_fixture(45)
        leaf = np.unique(a.mesh.triangles[a.face_part==4])
        other = np.unique(a.mesh.triangles[a.face_part!=4])
        self.assertFalse(np.array_equal(a.mesh.vertices[leaf],b.mesh.vertices[leaf]))
        self.assertTrue(np.array_equal(a.mesh.vertices[other],b.mesh.vertices[other]))
        self.assertTrue(np.array_equal(a.mesh.face_material,b.mesh.face_material))

    def test_bad_joint_angles_rejected(self):
        for angle in (-61,61,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                background_fixture(angle)

    def test_metadata_is_read_only(self):
        f = background_fixture()
        for a in (f.face_part,f.face_patch,f.mesh.vertices):
            with self.assertRaises(ValueError):
                a.flat[0] = 20

    def test_cycle_has_no_fixed_material_and_preserves_geometry(self):
        m = background_fixture().mesh
        c = cycle_materials(m)
        self.assertTrue((c.face_material!=m.face_material).all())
        for field in ('vertices','triangles','face_instance'):
            self.assertTrue(np.array_equal(getattr(m,field),getattr(c,field)))

    def test_cycle_closes_after_four_applications(self):
        m = background_fixture().mesh
        c = m
        for _ in range(4):
            c = cycle_materials(c)
        self.assertTrue(np.array_equal(c.face_material,m.face_material))

    def test_generation_is_deterministic(self):
        self.assertEqual(background_fixture().as_dict(),background_fixture().as_dict())
        self.assertEqual(background_appearance(4).as_dict(),background_appearance(4).as_dict())


class BackgroundValidationTest(unittest.TestCase):
    def test_all_changed_uses_integer_counts(self):
        self.assertEqual(true_fraction(torch.ones(25815,dtype=torch.bool)),1.)
        self.assertEqual(true_fraction(torch.tensor([True,False,True])),2/3)
        self.assertIsNone(true_fraction(torch.empty(0,dtype=torch.bool)))
        with self.assertRaises(ValueError):
            true_fraction(torch.ones(4))

    def test_albedo_edit_preserves_lighting_bits(self):
        a = background_appearance(4)
        material,light = appearance_variants(a)
        self.assertTrue(np.array_equal(a.light_direction,material.light_direction))
        self.assertTrue(np.array_equal(a.light_direction[:,1:],light.light_direction[:,1:]))
        self.assertTrue(np.array_equal(-a.light_direction[:,0],light.light_direction[:,0]))
        self.assertFalse(material.light_direction.flags.writeable)
        self.assertFalse(light.light_direction.flags.writeable)

    def test_synthetic_contract_passes_without_warp(self):
        f,c,g = synthetic()
        r,_ = evaluate_background(f,c,background_appearance(1),g,g)
        self.assertEqual(r['status'],'TECHNICAL_PASS',r['checks'])
        self.assertEqual(r['historical_R4_R4b_status'],'FAIL_UNCHANGED')

    def test_no_pixels_never_vacuously_passes(self):
        f,c,g = synthetic()
        g = replace(g,valid=torch.zeros_like(g.valid))
        r,_ = evaluate_background(f,c,background_appearance(1),g,g)
        self.assertEqual(r['status'],'INSUFFICIENT_VISIBILITY')

    def test_missing_object_never_passes(self):
        f,c,g = synthetic()
        g = replace(g,instance_id=torch.zeros_like(g.instance_id))
        r,_ = evaluate_background(f,c,background_appearance(1),g,g)
        self.assertEqual(r['status'],'INSUFFICIENT_VISIBILITY')

    def test_geometry_drift_fails(self):
        f,c,g = synthetic()
        r,_ = evaluate_background(f,c,background_appearance(1),g,replace(g,depth_m=g.depth_m+1))
        self.assertFalse(r['global_checks']['geometry_rerender_exact'])
        self.assertEqual(r['status'],'TECHNICAL_FAIL')

    def test_no_light_response_fails(self):
        f,c,g = synthetic()
        a = background_appearance(1)
        a = replace(a,directional=np.zeros(1))
        r,_ = evaluate_background(f,c,a,g,g)
        self.assertFalse(r['checks']['0']['lighting_sensitive'])

    def test_identical_material_colors_fail_cycle_rgb_check(self):
        f,c,g = synthetic()
        a = replace(background_appearance(1),base_color=np.full((1,4,3),.5))
        r,_ = evaluate_background(f,c,a,g,g)
        self.assertTrue(r['checks']['0']['cycle_all_visible_ids_changed'])
        self.assertFalse(r['checks']['0']['cycle_all_visible_rgb_changed'])

    def test_original_inputs_unchanged(self):
        f,c,g = synthetic()
        before = tensor_hash(g.depth_m),f.as_dict(),background_appearance(1).as_dict()
        evaluate_background(f,c,background_appearance(1),g,g)
        self.assertEqual(before,(tensor_hash(g.depth_m),f.as_dict(),background_appearance(1).as_dict()))

    def test_legacy_c_is_not_redefined(self):
        from renderer_validation.r4_metrics import LAMBERTIAN_R2_MAX
        self.assertEqual(LAMBERTIAN_R2_MAX,.5)

    def test_full_source_snapshot_includes_contract_and_new_modules(self):
        paths = runner.source_bytes()
        for path in ['docs/renderer_background_v1_contract.md', 'tools/run_renderer_background_validation.py',
                     'tools/renderer_validation/background_scene.py','tools/renderer_validation/background_validation.py']:
            self.assertEqual(hashlib.sha256(paths[path]).hexdigest(),hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

    def test_isolation_guard(self):
        with patch.dict(sys.modules,{'aerial_gym.task.example': object()}):
            with self.assertRaises(RuntimeError):
                runner.isolation_guard()


if __name__ == '__main__':
    unittest.main()


class ExactFractionHazardTest(unittest.TestCase):
    """The reduction hazard that turned two passing background checks into failures.

    These assertions run on CPU, where torch.mean happens to be exact, so they cannot reproduce
    the fault. They pin the integer contract instead, and the sizes named here are the CUDA
    counts that actually failed, kept so the reason is not lost.
    """
    CUDA_FLOAT32_SHORTFALL = 7600
    CUDA_FLOAT64_SHORTFALL = (7898, 25815)

    def test_integer_fraction_is_exactly_one_at_the_sizes_that_broke(self):
        for size in (self.CUDA_FLOAT32_SHORTFALL,) + self.CUDA_FLOAT64_SHORTFALL:
            mask = torch.ones(size, dtype=torch.bool)
            self.assertEqual(true_fraction(mask), 1.0)
            self.assertIs(true_fraction(mask) == 1.0, True)

    def test_r3_does_not_compare_a_float_mean_to_one(self):
        """R3 gates on an equality with 1.0, so it must count rather than average."""
        source = (ROOT / "tools/renderer_validation/validation.py").read_text()
        self.assertIn("material_changed_fraction_selected", source)
        gated = [line for line in source.splitlines() if "material_changed_fraction_selected" in line
                 and "float(" in line and ".mean()" in line]
        self.assertEqual(gated, [])
