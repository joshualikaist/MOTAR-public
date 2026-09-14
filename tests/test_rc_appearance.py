"""RC-R2/R3/R4 appearance: image statistics, area matching, and one-field-at-a-time variation.

The statistics are checked against images whose answers can be worked out by hand, and the
appearance helpers are checked for the property the experiments depend on: changing lighting changes
only lighting, changing material changes only material, and neither touches geometry.
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from renderer_validation import image_statistics as ims
from renderer_validation.area_match import (AREA_MATCH_TOLERANCE, area_ratios, evaluate_area_match,
                                            fit_isotropic_scale)
from renderer_validation.characterization_pipeline import (CharacterizationCell, appearance_for,
                                                           camera_relative_light, geometry_hashes,
                                                           recoloured, relit)
from renderer_validation.scene import Camera
from renderer_validation.view_grid import ViewGrid, fit_grid


class ImageStatisticsTest(unittest.TestCase):
    def test_luminance_uses_bt709_weights(self):
        image = np.zeros((1, 1, 3))
        for channel, weight in enumerate(ims.BT709):
            image[:] = 0.0
            image[0, 0, channel] = 1.0
            self.assertAlmostEqual(float(ims.luminance(image)[0, 0]), weight)
        with self.assertRaises(ValueError):
            ims.luminance(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            ims.luminance(np.full((2, 2, 3), np.nan))

    def test_flat_image_has_no_spread_and_no_gradient(self):
        record = ims.image_statistics(np.full((12, 14, 3), 0.4))
        # Not exactly zero: luminance is a dot product, and a BLAS implementation may sum the three
        # channels in a different order for different blocks of the image, so pixels of one colour
        # can differ in the last bit. The RC-R3 flat-variance threshold is 1e-12, four orders of
        # magnitude above this, so the distinction never reaches a verdict.
        self.assertLess(record["luminance_std"], 1e-15)
        self.assertLess(record["interior_gradient_mean"], 1e-15)
        self.assertAlmostEqual(record["luminance_mean"], 0.4)
        self.assertAlmostEqual(record["histogram_entropy_bits"], 0.0)
        self.assertLess(record["rms_contrast"], 1e-15)
        self.assertEqual(record["region"], "full_frame")
        self.assertEqual(record["pixels"], 12 * 14)

    def test_sobel_magnitude_of_a_known_ramp(self):
        ramp = np.tile(np.arange(6, dtype=np.float64) / 10.0, (5, 1))
        gradient = ims.convolve3(ramp, ims.SOBEL_X)
        # A unit-slope ramp gives a Sobel-x response of 8 * slope everywhere in the valid region.
        np.testing.assert_allclose(gradient, np.full((3, 4), 0.8), atol=1e-12)
        np.testing.assert_allclose(ims.convolve3(ramp, ims.SOBEL_Y), np.zeros((3, 4)), atol=1e-12)
        with self.assertRaises(ValueError):
            ims.convolve3(np.zeros((2, 2)), ims.SOBEL_X)

    def test_erosion_keeps_only_fully_surrounded_pixels(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        eroded = ims.erode3(mask)
        self.assertEqual(eroded.shape, (5, 5))
        self.assertEqual(int(eroded.sum()), 1)
        self.assertTrue(eroded[2, 2])
        with self.assertRaises(ValueError):
            ims.erode3(np.zeros((2, 9), dtype=bool))

    def test_histogram_entropy_of_a_uniform_spread_is_full(self):
        values = (np.arange(ims.HISTOGRAM_BINS) + 0.5) / ims.HISTOGRAM_BINS
        self.assertAlmostEqual(ims.histogram_entropy(values), math.log2(ims.HISTOGRAM_BINS))
        self.assertAlmostEqual(ims.histogram_entropy(np.full(50, 0.5)), 0.0)
        with self.assertRaises(ValueError):
            ims.histogram_entropy(np.array([]))
        with self.assertRaises(ValueError):
            ims.histogram_entropy(np.full(4, 5.0))

    def test_quantile_spread_groups_by_rank_not_by_value(self):
        values = np.array([0.0, 0.0, 1.0, 1.0])
        keys = np.array([0.0, 0.1, 10.0, 10.1])
        spread, groups = ims.quantile_spread(values, keys, quantiles=2)
        self.assertEqual((spread, groups), (0.0, 2))
        spread, groups = ims.quantile_spread(np.array([0.0, 1.0, 0.0, 1.0]), keys, quantiles=2)
        self.assertAlmostEqual(spread, 0.5)
        with self.assertRaises(ValueError):
            ims.quantile_spread(values, keys[:2], quantiles=2)

    def test_masked_statistics_ignore_everything_outside_the_mask(self):
        image = np.zeros((10, 10, 3))
        image[2:6, 2:6] = 0.8
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        masked = ims.image_statistics(image, mask)
        self.assertEqual(masked["pixels"], 16)
        self.assertAlmostEqual(masked["luminance_mean"], 0.8, places=6)
        self.assertEqual(masked["luminance_std"], 0.0)
        self.assertEqual(masked["region"], "silhouette")
        with self.assertRaises(ValueError):
            ims.image_statistics(image, np.zeros((10, 10), dtype=bool))
        with self.assertRaises(ValueError):
            ims.image_statistics(np.full((4, 4, 3), 1.5))
        with self.assertRaises(ValueError):
            ims.image_statistics(image, mask[:-1])

    def test_depth_grouping_is_reported_only_when_depth_is_given(self):
        rng = np.random.default_rng(3)
        image = np.clip(rng.uniform(0.2, 0.7, (9, 9, 3)), 0, 1)
        self.assertNotIn("within_depth_quantile_luminance_std_median",
                         ims.image_statistics(image))
        depth = np.tile(np.linspace(1.0, 2.0, 9), (9, 1))
        with_depth = ims.image_statistics(image, depth=depth)
        self.assertIn("within_depth_quantile_luminance_std_median", with_depth)
        self.assertEqual(with_depth["depth_quantile_groups"], ims.DEPTH_QUANTILES)


class AreaMatchTest(unittest.TestCase):
    def test_scale_is_the_square_root_of_the_area_ratio(self):
        self.assertAlmostEqual(fit_isotropic_scale([400.0, 400.0], [100.0, 100.0]), 2.0)
        self.assertAlmostEqual(fit_isotropic_scale([100.0], [400.0]), 0.5)
        for reference, base in (([0.0], [10.0]), ([10.0], [0.0]), ([], []), ([1.0, 2.0], [1.0])):
            with self.assertRaises(ValueError):
                fit_isotropic_scale(reference, base)

    def test_match_verdict_flips_exactly_at_the_tolerance(self):
        reference = [100.0, 100.0]
        inside = [100.0 * (1.0 + AREA_MATCH_TOLERANCE - 1e-9)] * 2
        outside = [100.0 * (1.0 + AREA_MATCH_TOLERANCE + 1e-9)] * 2
        self.assertEqual(evaluate_area_match(inside, reference, 1.0)["verdict"], "AREA_MATCHED")
        self.assertEqual(evaluate_area_match(outside, reference, 1.0)["verdict"], "AREA_MATCH_FAILED")
        below = [100.0 * (1.0 - AREA_MATCH_TOLERANCE - 1e-9)] * 2
        self.assertEqual(evaluate_area_match(below, reference, 1.0)["verdict"], "AREA_MATCH_FAILED")
        with self.assertRaises(ValueError):
            evaluate_area_match([1.0], reference, 1.0)

    def test_ratios_are_taken_against_the_named_reference(self):
        ratios = area_ratios({"a": [2.0, 2.0], "quadrotor_mesh": [1.0, 1.0]})
        self.assertAlmostEqual(ratios["a"]["ratio_to_reference"], 2.0)
        self.assertEqual(ratios["quadrotor_mesh"]["ratio_to_reference"], 1.0)
        self.assertEqual(ratios["a"]["views"], 2)
        with self.assertRaises(KeyError):
            area_ratios({"a": [1.0]})
        with self.assertRaises(ValueError):
            area_ratios({"quadrotor_mesh": [0.0]})


class AppearanceTest(unittest.TestCase):
    def test_colour_accepts_grey_one_triple_or_one_triple_per_material(self):
        grey = appearance_for(2, 3, colour=0.5)
        np.testing.assert_allclose(grey.base_color, np.full((2, 3, 3), 0.5))
        triple = appearance_for(2, 3, colour=(0.1, 0.2, 0.3))
        np.testing.assert_allclose(triple.base_color[0, 1], [0.1, 0.2, 0.3])
        per_material = appearance_for(1, 2, colour=[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        np.testing.assert_allclose(per_material.base_color[0, 1], [0.9, 0.9, 0.9])
        with self.assertRaises(ValueError):
            appearance_for(1, 2, colour=[[0.1, 0.1, 0.1]])
        with self.assertRaises(ValueError):
            appearance_for(0, 2)
        with self.assertRaises(ValueError):
            appearance_for(1, 2, kd=[0.5])

    def test_light_direction_may_be_shared_or_per_view(self):
        shared = appearance_for(3, 1, light_direction=(0.0, 0.0, -1.0))
        np.testing.assert_allclose(shared.light_direction, np.tile([0, 0, -1], (3, 1)))
        per_view = appearance_for(2, 1, light_direction=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(per_view.light_direction[1], [0, 1, 0])
        with self.assertRaises(ValueError):
            appearance_for(2, 1, light_direction=[[1.0, 0.0, 0.0]])

    def test_camera_relative_light_is_unit_length_and_on_the_camera_side(self):
        grid = fit_grid()
        light = camera_relative_light(grid)
        self.assertEqual(light.shape, (len(grid), 3))
        np.testing.assert_allclose(np.linalg.norm(light, axis=1), 1.0, atol=1e-12)
        for index, rotation in enumerate(grid.rotations()):
            forward = rotation[:, 2]
            # A surface facing the camera has normal -forward; it must be lit, so the
            # surface-to-light direction has a positive component along -forward.
            self.assertGreater(float(light[index] @ (-forward)), 0.3)
        with self.assertRaises(ValueError):
            camera_relative_light(grid, (0.0, 0.0, 0.0))

    def test_recolour_and_relight_change_exactly_one_field(self):
        base = appearance_for(2, 3, colour=0.5, kd=0.7, ambient=0.1, directional=0.6)
        other_colour = recoloured(base, 0.9)
        np.testing.assert_allclose(other_colour.kd, base.kd)
        # Appearance normalizes its light direction on construction, so a replace() round trip can
        # move a float32 unit vector by about 1e-7. Every RC cell builds its Appearance fresh rather
        # than by replacement, so no experiment depends on the difference.
        np.testing.assert_allclose(other_colour.light_direction, base.light_direction, atol=1e-6)
        np.testing.assert_allclose(other_colour.ambient, base.ambient)
        np.testing.assert_allclose(other_colour.directional, base.directional)
        self.assertFalse(np.array_equal(other_colour.base_color, base.base_color))
        other_light = relit(base, light_direction=(1.0, 0.0, 0.0), ambient=0.3)
        np.testing.assert_allclose(other_light.base_color, base.base_color)
        np.testing.assert_allclose(other_light.kd, base.kd)
        self.assertGreater(float(np.abs(other_light.light_direction
                                        - base.light_direction).max()), 1e-3)
        self.assertFalse(np.array_equal(other_light.light_direction, base.light_direction))
        np.testing.assert_allclose(other_light.directional, base.directional)
        with self.assertRaises(ValueError):
            relit(base)
        with self.assertRaises(ValueError):
            recoloured(base, [[0.1, 0.1, 0.1]])


class CellTest(unittest.TestCase):
    """One small CPU render per property, so these are real renders rather than stubs."""

    def cell(self, arm="box_proxy", views=((40.0, -15.0, 0.8),), width=160, height=120, **kwargs):
        camera = Camera(width=width, height=height, far_range_m=20.0)
        return CharacterizationCell(arm, camera, ViewGrid(list(views)), device="cpu", **kwargs)

    def test_flat_shading_of_one_colour_has_exactly_zero_variance(self):
        cell = self.cell()
        statistics = cell.image_statistics(cell.appearance(colour=0.55), "flat")
        self.assertEqual(statistics[0]["silhouette"]["luminance_std"], 0.0)

    def test_lambertian_shading_varies_where_two_orientations_are_visible(self):
        cell = self.cell()
        rows = cell.metrics()
        self.assertGreaterEqual(rows[0]["normal_histogram"]["occupied_bins"], 2)
        statistics = cell.image_statistics(cell.lit_appearance(colour=0.55), "lambertian")
        self.assertGreater(statistics[0]["silhouette"]["luminance_std"] ** 2, 1e-4)

    def test_lambertian_shading_is_constant_when_one_orientation_is_visible(self):
        """A head-on box face: 'more triangles' is not the same as 'more orientations'."""
        cell = self.cell(views=((0.0, 0.0, 0.8),))
        rows = cell.metrics()
        self.assertEqual(rows[0]["normal_histogram"]["occupied_bins"], 1)
        statistics = cell.image_statistics(cell.lit_appearance(colour=0.55), "lambertian")
        self.assertEqual(statistics[0]["silhouette"]["luminance_std"], 0.0)

    def test_geometry_is_byte_identical_across_shading_modes(self):
        cell = self.cell()
        before = geometry_hashes(cell.gbuffer())
        cell.shade(cell.appearance(colour=0.2), "flat")
        cell.shade(cell.lit_appearance(colour=0.9), "lambertian")
        self.assertEqual(before, geometry_hashes(cell.gbuffer()))

    def test_lighting_and_material_leave_geometry_and_silhouette_untouched(self):
        cell = self.cell()
        base = cell.lit_appearance(colour=0.5)
        reference = cell.arrays(base, "lambertian")
        for changed in (recoloured(base, 0.2), relit(base, ambient=0.05, directional=0.4)):
            arrays = cell.arrays(changed, "lambertian")
            for name in ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid"):
                np.testing.assert_array_equal(arrays[name], reference[name])
            self.assertFalse(np.array_equal(arrays["rgb"], reference["rgb"]))

    def test_instance_renumbering_changes_labels_and_nothing_else(self):
        plain = self.cell()
        offset = self.cell(instance_offset=100)
        light = camera_relative_light(plain.grid)
        first = plain.arrays(plain.appearance(colour=0.5, light_direction=light), "lambertian")
        second = offset.arrays(offset.appearance(colour=0.5, light_direction=light), "lambertian")
        np.testing.assert_array_equal(first["rgb"], second["rgb"])
        for name in ("range_m", "depth_m", "normal_world", "face_id", "valid"):
            np.testing.assert_array_equal(first[name], second[name])
        valid = first["valid"]
        np.testing.assert_array_equal(second["instance_id"][valid], first["instance_id"][valid] + 100)
        np.testing.assert_array_equal(second["instance_id"][~valid], -1)
        with self.assertRaises(ValueError):
            self.cell(instance_offset=-1)

    def test_the_analytic_arm_is_renumbered_the_same_way(self):
        plain = self.cell(arm="analytic_sphere", views=((0.0, 0.0, 0.6),))
        offset = self.cell(arm="analytic_sphere", views=((0.0, 0.0, 0.6),), instance_offset=7)
        first, second = plain.arrays(mode="flat"), offset.arrays(mode="flat")
        valid = first["valid"]
        np.testing.assert_array_equal(second["instance_id"][valid], first["instance_id"][valid] + 7)
        np.testing.assert_array_equal(first["range_m"], second["range_m"])

    def test_cell_refuses_a_bad_camera_or_grid(self):
        with self.assertRaises(TypeError):
            CharacterizationCell("box_proxy", "not a camera", ViewGrid([(0.0, 0.0, 1.0)]))
        with self.assertRaises(TypeError):
            CharacterizationCell("box_proxy", Camera(width=8, height=8), [(0.0, 0.0, 1.0)])
        with self.assertRaises(ValueError):
            self.cell().shade(mode="phong")


if __name__ == "__main__":
    unittest.main()
