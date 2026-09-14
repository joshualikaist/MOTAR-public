"""CPU-only checks for the supplemental renderer measurement code."""
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_renderer_measurements as audit


class MeasurementTest(unittest.TestCase):
    def test_timing_summary_retains_count_mean_and_linear_percentile(self):
        s = audit.timing_summary([1, 2, 3, 4, 5])
        self.assertEqual((s['n'], s['mean_ms'], s['p50_ms']), (5, 3, 3))
        self.assertAlmostEqual(s['p95_ms'], 4.8)

    def test_timing_rejects_empty_nonpositive_or_nonfinite(self):
        for value in ([], [0], [-1], [float('nan')], [float('inf')], [[1]]):
            with self.assertRaises(ValueError):
                audit.timing_summary(value)

    def fixture(self):
        x = np.arange(100, dtype=np.float64).reshape(10, 10)
        rgb = np.repeat((x / 100)[..., None], 3, axis=-1)
        return rgb, x, np.ones((10, 10), dtype=bool)

    def test_linear_luminance_has_unit_r_squared(self):
        self.assertAlmostEqual(audit.numpy_statistics(*self.fixture())['range_luminance_r_squared'], 1)

    def test_constant_luminance_has_zero_r_squared_and_spread(self):
        rgb, x, mask = self.fixture()
        rgb.fill(0.5)
        stats = audit.numpy_statistics(rgb, x, mask)
        self.assertEqual(stats['range_luminance_r_squared'], 0)
        self.assertEqual(stats['within_range_bin_luminance_std_median'], 0)

    def test_population_std_and_twenty_equal_count_bins(self):
        stats = audit.numpy_statistics(*self.fixture())
        self.assertAlmostEqual(stats['within_range_bin_luminance_std_median'], np.std(np.arange(5) / 100))

    def test_debug_background_is_excluded(self):
        rgb, x, mask = self.fixture()
        mask[0] = False
        reference = audit.numpy_statistics(rgb, x, mask)
        rgb[0] = 900
        x[0] = -999
        self.assertEqual(reference, audit.numpy_statistics(rgb, x, mask))

    def test_insufficient_valid_pixels_refused(self):
        rgb, x, mask = self.fixture()
        mask[3:] = False
        with self.assertRaises(ValueError):
            audit.numpy_statistics(rgb, x, mask)

    def test_repeats_are_fixed(self):
        self.assertEqual((audit.REPEATS, audit.WARMUP, audit.SAMPLES), (3, 3, 10))

    def test_protocol_is_in_source_manifest(self):
        p = audit.provenance()
        key = str(audit.PROTOCOL.relative_to(ROOT))
        self.assertEqual(p['source_files'][key]['sha256'], hashlib.sha256(audit.PROTOCOL.read_bytes()).hexdigest())
        self.assertIn('tracked_at_head', p['source_files'][key])

    def test_simulator_import_is_rejected_without_importing_it(self):
        with patch.dict(sys.modules, {'aerial_gym.task.example': object()}):
            with self.assertRaises(RuntimeError):
                audit.isolation_guard()


if __name__ == '__main__':
    unittest.main()
