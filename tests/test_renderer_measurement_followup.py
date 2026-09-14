"""Measurement/fit contracts with synthetic data; no historical threshold edits."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import measure_renderer_contract as study
from renderer_validation.gbuffer import GBuffer
from renderer_validation.scene import Camera


class MeasurementContracts(unittest.TestCase):
    def rectangle(self, clipped=False):
        mask = torch.zeros((1, 40, 60), dtype=torch.bool)
        mask[:, 10:30, 20:40] = True
        if clipped:
            mask[:, 0, 20] = True
        depth = mask.float()*2
        return GBuffer(depth, depth, torch.zeros((1, 40, 60, 3)), mask.int(), mask.int(), mask)

    def test_inclusive_bbox_and_centroid(self):
        result = study.measure(self.rectangle(), Camera(60, 40))
        self.assertEqual(result['area_px'], 400)
        self.assertEqual(result['width_px'], 20)
        self.assertEqual(result['height_px'], 20)
        self.assertEqual(result['centroid_px'], [29.5, 19.5])
        self.assertEqual(result['depth_median_m'], 2.)

    def test_border_refused(self):
        self.assertEqual(study.measure(self.rectangle(True), Camera(60, 40))['status'], 'REFUSED')

    def test_small_mask_refused(self):
        buffer = self.rectangle()
        buffer.valid.zero_()
        self.assertEqual(study.measure(buffer, Camera(60, 40))['status'], 'REFUSED')

    def test_zero_reference_centroid_safe(self):
        row = study.measure(self.rectangle(), Camera(60, 40))
        row['centroid_over_f'] = [0., 0.]
        self.assertEqual(study.errors(row, row)['centroid_over_f']['relative'], 0.)

    def test_resolution_normalization(self):
        row = study.measure(self.rectangle(), Camera(60, 40))
        reference = copy.deepcopy(row)
        reference['focal_px'] *= 2
        for key in ('width_px', 'height_px'):
            reference[key] *= 2
        reference['area_px'] *= 4
        error = study.errors(row, reference)
        for name in ('width_over_f', 'height_over_f', 'area_over_f2', 'depth_median_m'):
            self.assertEqual(error[name]['relative'], 0)

    def test_bad_metric_prevents_recommendation(self):
        row = study.measure(self.rectangle(), Camera(60, 40))
        row['errors'] = study.errors(row, row)
        self.assertTrue(study.qualified([row]))
        row['errors']['area_over_f2']['relative'] = .02001
        self.assertFalse(study.qualified([row]))
        self.assertFalse(study.qualified([]))

    def test_refused_comparison_has_no_zero_error(self):
        self.assertIsNone(study.errors({'status': 'REFUSED'}, {'status': 'MEASURED'}))

    def test_camera_reference_within_frozen_contract(self):
        self.assertEqual(study.RESOLUTIONS[-1], (2048, 1536))
        Camera(*study.RESOLUTIONS[-1])
        with self.assertRaises(ValueError):
            Camera(2560, 1920)

    def test_disjoint_fit_validation_and_old_primary(self):
        self.assertFalse(set(study.FIT_VIEWS) & set(study.VALIDATION_VIEWS))
        self.assertFalse(set(study.primary_grid().views) & set(study.VALIDATION_VIEWS))
        self.assertEqual(len(study.FIT_VIEWS), 6)
        self.assertEqual(len(study.VALIDATION_VIEWS), 24)

    def test_projected_axis_box_has_closed_form_area(self):
        actual = study.projected_box_area([1, 1, 1], [(0, 0, 2)])[0]
        expected = .28*.28/(2-.06)**2
        self.assertAlmostEqual(actual, expected, places=12)

    def test_fit_recovers_synthetic_anisotropic_control_without_validation(self):
        scales = [.8, 1.1, .7]
        areas = study.projected_box_area(scales, study.FIT_VIEWS)
        result = study.fit_scales(areas)
        self.assertTrue(result['success'])
        np.testing.assert_allclose(result['scales'], scales, atol=1e-6, rtol=0)

    def test_fit_requires_exact_calibration_shape(self):
        with self.assertRaises(ValueError):
            study.fit_scales([1, 2])

    def test_area_acceptance_boundaries(self):
        stats = {'median_absolute_relative_error': .05, 'p90_absolute_relative_error': .1,
                 'worst_absolute_relative_error': .2}
        self.assertTrue(study.area_pass(stats))
        for key in stats:
            changed = dict(stats)
            changed[key] += .00001
            self.assertFalse(study.area_pass(changed))

    def test_area_statistics_are_per_view_not_ratio_of_means(self):
        stats = study.area_statistics([2, 6], [2, 3])
        self.assertEqual(stats['median_absolute_relative_error'], .5)
        self.assertEqual(stats['worst_absolute_relative_error'], 1.)

    def test_nonfinite_area_is_not_silently_a_failed_numeric_gate(self):
        for observed, reference in (([np.nan], [1]), ([1], [np.inf]), ([-1], [1]), ([[1]], [[1]])):
            with self.assertRaises(ValueError):
                study.area_statistics(observed, reference)

    def test_boundary_contrast_excludes_constant_background(self):
        rgb = np.zeros((6, 6, 3))
        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, 2:4] = True
        rgb[mask] = .5
        self.assertAlmostEqual(study.boundary_contrast(rgb, mask), .5)

    def test_constant_scatter_has_no_fabricated_correlation(self):
        self.assertIsNone(study.correlation([1, 1, 1], [2, 3, 4])['pearson_r'])

    def test_mesh_contract_does_not_invent_normal_sources(self):
        names = {f.name for f in study.fields(study.MeshScene)}
        self.assertEqual(names, {'vertices', 'triangles', 'face_material', 'face_instance'})


if __name__ == '__main__':
    unittest.main()
