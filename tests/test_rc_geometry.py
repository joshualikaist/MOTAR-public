"""RC-R1 geometry: view frames, closed forms, metrics and every gate's decision boundary.

Each gate is checked in both directions. A gate that cannot fail is not a gate, so the tests
perturb a passing record until it fails and confirm the verdict flips.
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from renderer_validation import geometry_metrics as gm
from renderer_validation import view_grid as vg
from renderer_validation.analytic_primitives import (analytic_sphere_gbuffer, camera_directions,
                                                     gbuffer_from_analytic, project_points,
                                                     sphere_hits, sphere_projected_radius_px,
                                                     world_rays)
from renderer_validation.gbuffer import GBuffer
from renderer_validation.scene import Camera
from renderer_validation.target_arms import build_arm

CAMERA = Camera(width=64, height=48, horizontal_fov_deg=60.0, far_range_m=20.0)


def synthetic_gbuffer(masks, depth=2.0, normal=(0.0, 0.0, -1.0), faces=2):
    """A G-buffer whose content is known exactly, so a metric's value can be predicted by hand.

    The silhouette is split between `faces` face indices across image columns, because one face
    index would be refused by the preregistered minimum visible-triangle count.
    """
    masks = np.asarray(masks, dtype=bool)
    count, height, width = masks.shape
    ranges = np.where(masks, depth, 0.0).astype(np.float32)
    normals = np.zeros((count, height, width, 3), dtype=np.float32)
    normals[masks] = np.asarray(normal, dtype=np.float32)
    columns = np.arange(width) % int(faces)
    labels = np.where(masks, np.broadcast_to(columns, masks.shape), -1).astype(np.int32)
    return GBuffer(torch.tensor(ranges), torch.tensor(ranges), torch.tensor(normals),
                   torch.tensor(labels), torch.tensor(labels), torch.tensor(masks))


class ViewGridTest(unittest.TestCase):
    def test_every_view_looks_at_the_origin_with_world_up_minus_y(self):
        grid = vg.primary_grid()
        self.assertEqual(len(grid), 24)
        for index, rotation in enumerate(grid.rotations()):
            position = grid.positions[index]
            forward = rotation[:, 2]
            np.testing.assert_allclose(forward, -position / np.linalg.norm(position), atol=1e-12)
            self.assertGreater(float(rotation[1, 1]), 0.0, "camera +Y must point world-down")
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_elevation_raises_the_camera_and_distance_scales_the_position(self):
        low = vg.spherical_position(0.0, 0.0, 2.0)
        high = vg.spherical_position(0.0, 35.0, 2.0)
        self.assertLess(high[1], low[1])                       # world up is -Y
        self.assertAlmostEqual(float(np.linalg.norm(high)), 2.0, places=12)
        far = vg.spherical_position(17.0, 11.0, 4.0)
        near = vg.spherical_position(17.0, 11.0, 2.0)
        np.testing.assert_allclose(far, 2.0 * near, atol=1e-12)

    def test_quaternion_round_trip_and_refusals(self):
        for view in ((0.0, 0.0, 2.0), (137.0, -20.0, 1.5), (271.0, 35.0, 4.0)):
            rotation = vg.look_at_rotation(vg.spherical_position(*view))
            quaternion = vg.rotation_to_quaternion(rotation)
            self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)
            np.testing.assert_allclose(vg.quaternion_to_rotation(quaternion), rotation, atol=1e-12)
        with self.assertRaises(ValueError):
            vg.spherical_position(0.0, 85.0, 2.0)
        with self.assertRaises(ValueError):
            vg.spherical_position(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            vg.ViewGrid([(0.0, 0.0, 2.0), (0.0, 0.0, 2.0)])
        with self.assertRaises(ValueError):
            vg.ViewGrid([])

    def test_preregistered_split_is_four_and_twenty_and_disjoint(self):
        primary, fit, validation = vg.fit_validation_split()
        self.assertEqual((len(fit), len(validation)), (4, 20))
        self.assertFalse(set(fit) & set(validation))
        self.assertEqual(sorted(fit + validation), list(range(len(primary))))
        for index in fit:
            azimuth, elevation, distance = primary.views[index]
            self.assertEqual((elevation, distance), (vg.FIT_ELEVATION, vg.PRIMARY_DISTANCE_M))
            self.assertIn(azimuth, vg.FIT_AZIMUTHS)


class AnalyticTest(unittest.TestCase):
    def test_ray_construction_and_projection_are_exact_inverses(self):
        grid = vg.primary_grid()
        rotation = grid.rotations()[5]
        position = grid.positions[5]
        origin, directions = world_rays(CAMERA, position, rotation)
        for y, x in ((0, 0), (7, 13), (47, 63), (24, 32)):
            point = origin + 3.3 * directions[y, x]
            uv, depth = project_points(CAMERA, [point], position, rotation)
            np.testing.assert_allclose(uv[0], [x, y], atol=1e-9)
            self.assertGreater(depth[0], 0.0)

    def test_camera_directions_match_the_kernel_expression(self):
        directions = camera_directions(CAMERA)
        focal = CAMERA.focal_px
        for y, x in ((0, 0), (11, 29), (47, 63)):
            expected = np.array([(x - CAMERA.width / 2.0) / focal,
                                 (y - CAMERA.height / 2.0) / focal, 1.0])
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(directions[y, x], expected, atol=1e-12)

    def test_sphere_silhouette_matches_the_closed_form_at_several_distances(self):
        camera = Camera(width=320, height=240, horizontal_fov_deg=60.0, far_range_m=20.0)
        for distance in (1.5, 2.0, 3.0):
            grid = vg.ViewGrid([(23.0, -7.0, distance)])
            buffer = analytic_sphere_gbuffer(camera, grid, 0.15)
            area = int(buffer.valid.sum())
            measured = math.sqrt(area / math.pi)
            expected = sphere_projected_radius_px(camera, distance, 0.15)
            self.assertLess(abs(measured - expected), gm.SPHERE_RADIUS_TOLERANCE_PX)
            depth = buffer.depth_m[0][buffer.valid[0]]
            self.assertLess(abs(float(depth.min()) - (distance - 0.15)),
                            gm.SPHERE_DEPTH_TOLERANCE_M)

    def test_closed_form_refuses_a_camera_inside_the_sphere(self):
        with self.assertRaises(ValueError):
            sphere_projected_radius_px(CAMERA, 0.1, 0.15)
        with self.assertRaises(ValueError):
            sphere_projected_radius_px(CAMERA, 2.0, 0.0)

    def test_analytic_packing_refuses_malformed_buffers(self):
        ranges = np.ones((1, CAMERA.height, CAMERA.width))
        normals = np.zeros((1, CAMERA.height, CAMERA.width, 3))
        normals[..., 2] = 0.5                                   # not unit length
        hits = np.ones(ranges.shape, dtype=bool)
        with self.assertRaises(ValueError):
            gbuffer_from_analytic(CAMERA, ranges, normals, hits, "cpu")
        normals[..., 2] = 1.0
        with self.assertRaises(ValueError):
            gbuffer_from_analytic(CAMERA, ranges[:, :-1], normals, hits, "cpu")
        bad = ranges.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            gbuffer_from_analytic(CAMERA, bad, normals, hits, "cpu")

    def test_sphere_misses_when_the_ray_points_away(self):
        grid = vg.ViewGrid([(0.0, 0.0, 2.0)])
        ranges, normals, hits = sphere_hits(CAMERA, grid.positions[0],
                                            -grid.rotations()[0], 0.15)
        self.assertFalse(hits.any())
        self.assertEqual(float(np.abs(normals).max()), 0.0)


class MetricsTest(unittest.TestCase):
    def rectangle(self, width=30, height=20, count=1):
        masks = np.zeros((count, CAMERA.height, CAMERA.width), dtype=bool)
        masks[:, 15:15 + height, 20:20 + width] = True
        return masks

    def test_metrics_of_a_known_rectangle(self):
        buffer = synthetic_gbuffer(self.rectangle())
        row = gm.view_metrics(buffer, 0, CAMERA, np.eye(3), True)
        self.assertEqual(row["view"], 0)
        self.assertEqual(row["silhouette_pixels"], 600)
        self.assertEqual((row["bbox_width_px"], row["bbox_height_px"]), (30, 20))
        self.assertEqual((row["bbox_u_min"], row["bbox_u_max"]), (20, 49))
        self.assertAlmostEqual(row["centroid_u_px"], 34.5)
        self.assertAlmostEqual(row["centroid_v_px"], 24.5)
        self.assertAlmostEqual(row["equivalent_diameter_px"], 2 * math.sqrt(600 / math.pi))
        self.assertEqual((row["depth_min_m"], row["depth_max_m"]), (2.0, 2.0))
        self.assertEqual(row["visible_triangles"], 2)
        self.assertAlmostEqual(row["mean_abs_normal_dot_view"], 1.0)

    def test_degenerate_views_are_refused_in_both_preregistered_ways(self):
        small = np.zeros((1, CAMERA.height, CAMERA.width), dtype=bool)
        small[0, 0:10, 0:10] = True                            # 100 pixels, below 300
        with self.assertRaises(RuntimeError):
            gm.view_metrics(synthetic_gbuffer(small), 0, CAMERA, np.eye(3), True)
        # Enough pixels, but fewer visible triangles than the minimum.
        masks = self.rectangle()
        buffer = synthetic_gbuffer(masks, faces=2)
        with self.assertRaises(RuntimeError):
            gm.view_metrics(buffer, 0, CAMERA, np.eye(3), True, min_visible_triangles=3)
        # The analytic arm has no triangles, so the triangle rule must not apply to it.
        self.assertEqual(gm.view_metrics(buffer, 0, CAMERA, np.eye(3), False,
                                         min_visible_triangles=99)["silhouette_pixels"], 600)

    def test_refusals_are_recorded_rather_than_fatal(self):
        masks = np.zeros((2, CAMERA.height, CAMERA.width), dtype=bool)
        masks[0, 8:38, 10:45] = True                           # 1050 pixels, two face indices
        masks[1, 0:5, 0:5] = True                              # 25 pixels, refused
        grid = vg.ViewGrid([(0.0, 0.0, 2.0), (45.0, 0.0, 2.0)])
        arm = build_arm("box_proxy")
        measured, refused = gm.measured_arm_metrics(arm, synthetic_gbuffer(masks), CAMERA, grid)
        self.assertEqual([row["view"] for row in measured], [0])
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]["view"], 1)
        self.assertEqual(refused[0]["silhouette_pixels"], 25)
        self.assertIn("300", refused[0]["reason"])
        masks[0] = False
        masks[0, 0:5, 0:5] = True
        with self.assertRaises(RuntimeError):
            gm.measured_arm_metrics(arm, synthetic_gbuffer(masks), CAMERA, grid)

    def test_normal_histogram_separates_one_orientation_from_many(self):
        single = gm.normal_histogram(np.tile([0.0, 0.0, 1.0], (500, 1)))
        self.assertEqual(single["occupied_bins"], 1)
        self.assertAlmostEqual(single["entropy_bits"], 0.0)
        self.assertAlmostEqual(single["dominant_bin_fraction"], 1.0)
        rng = np.random.default_rng(7)
        directions = rng.normal(size=(5000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        spread = gm.normal_histogram(directions)
        self.assertGreater(spread["occupied_bins"], 20)
        self.assertGreater(spread["entropy_bits"], 4.0)
        self.assertLess(spread["dominant_bin_fraction"], 0.2)
        with self.assertRaises(ValueError):
            gm.normal_histogram(np.zeros((0, 3)))


class GateBoundaryTest(unittest.TestCase):
    """Every gate must flip when its own quantity crosses the preregistered threshold."""

    def sweep(self):
        return vg.ViewGrid([(0.0, 0.0, d) for d in (1.5, 2.0, 3.0)])

    def metrics_with_diameters(self, diameters, depths=None):
        rows = []
        for index, diameter in enumerate(diameters):
            row = {"view": index, "equivalent_diameter_px": diameter,
                   "depth_min_m": 1.0, "depth_max_m": 1.0}
            if depths is not None:
                row["depth_min_m"], row["depth_max_m"] = depths[index]
            rows.append(row)
        return rows

    def test_g4_flips_at_its_tolerance(self):
        arm = build_arm("analytic_sphere")
        grid = self.sweep()
        exact = [60.0 / d for _, _, d in grid.views]
        self.assertTrue(gm.gate_inverse_distance(arm, self.metrics_with_diameters(exact), grid)["passed"])
        perturbed = list(exact)
        perturbed[1] *= 1.0 + gm.INVERSE_DISTANCE_TOLERANCE + 1e-6
        self.assertFalse(gm.gate_inverse_distance(arm, self.metrics_with_diameters(perturbed),
                                                  grid)["passed"])

    def test_g4_records_a_line_of_sight_that_kept_one_distance(self):
        arm = build_arm("analytic_sphere")
        grid = vg.ViewGrid([(0.0, 0.0, 1.5), (0.0, 0.0, 2.0), (90.0, 0.0, 2.0)])
        rows = self.metrics_with_diameters([40.0, 30.0, 30.0])
        record = gm.gate_inverse_distance(arm, rows, grid)
        self.assertTrue(record["passed"])
        self.assertEqual([line["azimuth_deg"] for line in record["lines_without_two_distances"]],
                         [90.0])
        with self.assertRaises(RuntimeError):
            gm.gate_inverse_distance(arm, rows[2:], grid)

    def test_g5_flips_when_a_surface_leaves_the_circumscribed_sphere(self):
        arm = build_arm("analytic_sphere")
        grid = vg.ViewGrid([(0.0, 0.0, 2.0)])
        inside = self.metrics_with_diameters([40.0], depths=[(2.0 - 0.15, 2.0 + 0.15)])
        self.assertTrue(gm.gate_depth_bracket(arm, inside, grid)["passed"])
        outside = self.metrics_with_diameters(
            [40.0], depths=[(2.0 - 0.15 - gm.DEPTH_BRACKET_SLACK_M - 1e-4, 2.0)])
        self.assertFalse(gm.gate_depth_bracket(arm, outside, grid)["passed"])

    def test_g7_flips_at_its_tolerance_and_lists_unpaired_views(self):
        arm = build_arm("box_proxy")
        low = Camera(width=320, height=240)
        high = Camera(width=640, height=480)
        ratio = high.focal_px / low.focal_px
        matched = gm.gate_resolution_consistency(
            arm, self.metrics_with_diameters([40.0, 30.0]),
            self.metrics_with_diameters([40.0 * ratio, 30.0 * ratio]), low, high)
        self.assertTrue(matched["passed"])
        self.assertEqual(matched["views_measured_at_one_resolution_only"], [])
        broken = gm.gate_resolution_consistency(
            arm, self.metrics_with_diameters([40.0]),
            self.metrics_with_diameters([40.0 * ratio * (1.0 + gm.RESOLUTION_CONSISTENCY_TOLERANCE
                                                         + 1e-6)]), low, high)
        self.assertFalse(broken["passed"])
        partial = gm.gate_resolution_consistency(
            arm, self.metrics_with_diameters([40.0, 30.0]),
            [dict(self.metrics_with_diameters([40.0 * ratio])[0], view=0)], low, high)
        self.assertTrue(partial["passed"])
        self.assertEqual(partial["views_measured_at_one_resolution_only"], [1])

    def test_g3_compares_the_measured_box_with_its_projected_corners(self):
        arm = build_arm("box_proxy")
        grid = vg.ViewGrid([(0.0, 0.0, 2.0)])
        camera = Camera(width=320, height=240)
        corners = np.unique(np.asarray(arm.mesh.vertices, dtype=np.float64), axis=0)
        uv, _ = project_points(camera, corners, grid.positions[0], grid.rotations()[0])
        exact = [{"view": 0, "bbox_u_min": int(round(uv[:, 0].min())),
                  "bbox_u_max": int(round(uv[:, 0].max())),
                  "bbox_v_min": int(round(uv[:, 1].min())),
                  "bbox_v_max": int(round(uv[:, 1].max()))}]
        self.assertTrue(gm.gate_box_closed_form(arm, exact, camera, grid)["passed"])
        shifted = [dict(exact[0], bbox_u_min=exact[0]["bbox_u_min"] + 3)]
        self.assertFalse(gm.gate_box_closed_form(arm, shifted, camera, grid)["passed"])
        with self.assertRaises(TypeError):
            gm.gate_box_closed_form(build_arm("analytic_sphere"), exact, camera, grid)

    def test_g1_agrees_with_an_independent_intersector_on_a_real_render(self):
        """The only test here that actually runs Warp: the box, one view, small frame."""
        arm = build_arm("box_proxy")
        camera = Camera(width=80, height=60, far_range_m=20.0)
        grid = vg.ViewGrid([(37.0, -12.0, 1.2)])
        buffer = arm.gbuffer(camera, grid, device="cpu")
        record = gm.gate_cross_implementation(arm, buffer, camera, grid)
        self.assertTrue(record["passed"], record)
        self.assertEqual(record["worst_disagreement_fraction"], 0.0)
        self.assertLess(record["worst_range_difference_m"], gm.RANGE_DIFFERENCE_MAX_M)
        with self.assertRaises(TypeError):
            gm.gate_cross_implementation(build_arm("analytic_sphere"), buffer, camera, grid)

    def test_thresholds_record_every_applied_value(self):
        recorded = gm.thresholds()
        self.assertEqual(recorded["min_silhouette_pixels"], gm.MIN_SILHOUETTE_PIXELS)
        self.assertEqual(recorded["min_visible_triangles"], gm.MIN_VISIBLE_TRIANGLES)
        self.assertEqual(recorded["inverse_distance_tolerance"], gm.INVERSE_DISTANCE_TOLERANCE)
        self.assertEqual(recorded["normal_bins"], [gm.NORMAL_AZIMUTH_BINS, gm.NORMAL_POLAR_BINS])


class WarpAvailabilityTest(unittest.TestCase):
    """Guard for a defect this track hit: a suite-mate removing the real warp from sys.modules.

    Several CPU-only test modules swap a fake `warp` into sys.modules so their subjects' kernel
    decorators can evaluate at import. If one of them leaves the name absent after the real package
    has been imported, warp's submodules stay behind and the next real `import warp` re-executes
    warp/__init__.py against a half-built module, failing on `warp.config`. Isolated runs never see
    it; the full suite does.
    """

    def test_a_real_warp_import_still_works_after_the_rest_of_the_suite(self):
        import warp
        self.assertIsInstance(getattr(warp, "Mesh", None), type,
                              "sys.modules['warp'] is not the real package")
        self.assertTrue(hasattr(warp, "config"))


class ArmTest(unittest.TestCase):
    def test_the_four_specimens_have_their_preregistered_shapes(self):
        sphere = build_arm("analytic_sphere")
        self.assertEqual((sphere.kind, sphere.triangles, sphere.radius_m), ("analytic_sphere", 0, 0.15))
        box = build_arm("box_proxy")
        self.assertEqual((box.triangles, box.material_count), (12, 1))
        extent = (np.asarray(box.mesh.vertices).max(axis=0)
                  - np.asarray(box.mesh.vertices).min(axis=0))
        np.testing.assert_allclose(extent, [0.28, 0.28, 0.12], atol=1e-7)
        mesh = build_arm("quadrotor_mesh")
        self.assertEqual(mesh.triangles, 1548)
        self.assertEqual(mesh.description["visual_links"], 13)
        np.testing.assert_allclose(mesh.description["bounding_box_centre_m"], [0, 0, 0], atol=1e-7)

    def test_only_the_area_matched_box_takes_a_scale_and_it_is_required(self):
        with self.assertRaises(ValueError):
            build_arm("area_matched_box")
        with self.assertRaises(ValueError):
            build_arm("box_proxy", scale=1.1)
        with self.assertRaises(ValueError):
            build_arm("quadrotor_mesh", scale=1.1)
        with self.assertRaises(ValueError):
            build_arm("nonexistent_arm")
        scaled = build_arm("area_matched_box", scale=0.5)
        base = build_arm("box_proxy")
        self.assertAlmostEqual(scaled.circumscribed_radius_m, base.circumscribed_radius_m / 2, 6)


if __name__ == "__main__":
    unittest.main()
