"""RC-R5 benchmark mechanics: the staged path, the stop rule, and how memory is reported.

The timings themselves are not asserted - a wall clock is not a unit test. What is asserted is that
the staged path reproduces the production render exactly, that the resource stop rule fires before
any work is done, and that an unavailable memory field says so instead of reading zero.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_renderer_characterization as bench
from renderer_validation.characterization_pipeline import CharacterizationCell
from renderer_validation.scene import Camera


class GridTest(unittest.TestCase):
    def test_grid_has_the_requested_number_of_distinct_views(self):
        for count in (1, 8, 32):
            grid = bench.benchmark_grid(count)
            self.assertEqual(len(grid), count)
            self.assertEqual(len(set(grid.views)), count)
            for _, elevation, distance in grid.views:
                self.assertEqual((elevation, distance),
                                 (bench.BENCHMARK_ELEVATION_DEG, bench.BENCHMARK_DISTANCE_M))

    def test_grid_refuses_counts_outside_the_supported_range(self):
        for count in (0, -1, 129, 2.5):
            with self.assertRaises(ValueError):
                bench.benchmark_grid(count)


class DistributionTest(unittest.TestCase):
    def test_distribution_reports_count_median_mean_and_p95(self):
        record = bench.distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(record["count"], 4)
        self.assertEqual(record["median_ms"], 2.5)
        self.assertEqual(record["mean_ms"], 2.5)
        self.assertEqual((record["min_ms"], record["max_ms"]), (1.0, 4.0))
        self.assertAlmostEqual(record["p95_ms"], 3.85)

    def test_distribution_refuses_empty_or_non_finite_samples(self):
        for samples in ([], [float("nan")], [1.0, float("inf")]):
            with self.assertRaises(ValueError):
                bench.distribution(samples)

    def test_timed_returns_one_sample_per_repeat(self):
        calls = []
        samples = bench.timed(lambda: calls.append(1), 5)
        self.assertEqual((len(samples), len(calls)), (5, 5))
        self.assertTrue(all(value >= 0.0 for value in samples))

    def test_projected_bytes_counts_every_exported_array(self):
        self.assertEqual(bench.projected_bytes(10, 10, 1), 4100)
        self.assertEqual(bench.projected_bytes(640, 480, 32), 640 * 480 * 32 * 41)


class MemoryReportTest(unittest.TestCase):
    def test_unavailable_fields_say_so_and_are_never_zero(self):
        record = bench.memory_record("cpu", None)
        self.assertEqual(record["torch_allocated_mib"], "UNAVAILABLE")
        self.assertEqual(record["torch_reserved_mib"], "UNAVAILABLE")
        self.assertEqual(record["warp_device_used_mib"], "UNAVAILABLE")
        self.assertIsInstance(record["rss_mib"], float)
        self.assertGreater(record["rss_mib"], 0.0)
        for key, value in record.items():
            self.assertNotEqual(value, 0, key)
        self.assertIn(record["nvml_used_mib"], ("UNAVAILABLE",)) if not self._nvml() else None

    def _nvml(self):
        try:
            import pynvml                                      # noqa: F401
        except ImportError:
            return False
        return True

    def test_nvml_absence_carries_its_reason(self):
        record = bench.memory_record("cpu", None)
        if record["nvml_used_mib"] == "UNAVAILABLE":
            self.assertIn("pynvml", record["nvml_reason"])
        else:
            self.assertNotIn("nvml_reason", record)


class StopRuleTest(unittest.TestCase):
    def test_an_oversized_cell_is_skipped_before_any_render(self):
        record = bench.run_cell("box_flat", "box_proxy", "flat", (2048, 2048), 128, "cpu", None)
        self.assertEqual(record["status"], "SKIPPED_RESOURCE_STOP")
        self.assertIn("projected output", record["reason"])
        self.assertNotIn("headline_total", record)

    def test_a_measured_cell_reports_both_totals_and_every_stage(self):
        original = (bench.WARMUP_RENDERS, bench.MEASURED_RENDERS)
        bench.WARMUP_RENDERS, bench.MEASURED_RENDERS = 1, 2
        try:
            record = bench.run_cell("box_flat", "box_proxy", "flat", (48, 36), 2, "cpu", None)
        finally:
            bench.WARMUP_RENDERS, bench.MEASURED_RENDERS = original
        self.assertEqual(record["status"], "MEASURED")
        self.assertIn("normal_face_pass_ms", record["stages"])
        self.assertIn("range_pass_ms", record["stages"])
        self.assertIn("finalize_ms", record["stages"])
        self.assertIn("clear_ms", record["stages"])
        self.assertGreater(record["headline_total"]["median_ms"], 0.0)
        self.assertGreater(record["staged_total"]["median_ms"], 0.0)
        self.assertIsInstance(record["instrumentation_overhead_ms"], float)
        self.assertIsNotNone(record["shading"])
        self.assertIsNotNone(record["host_copy"])
        self.assertGreater(record["images_per_second"], 0.0)
        self.assertEqual(record["memory"]["torch_allocated_mib"], "UNAVAILABLE")

    def test_a_geometry_only_configuration_reports_no_shading(self):
        original = (bench.WARMUP_RENDERS, bench.MEASURED_RENDERS)
        bench.WARMUP_RENDERS, bench.MEASURED_RENDERS = 1, 2
        try:
            record = bench.run_cell("analytic_sphere_geometry", "analytic_sphere", None,
                                    (48, 36), 1, "cpu", None)
        finally:
            bench.WARMUP_RENDERS, bench.MEASURED_RENDERS = original
        self.assertEqual(record["status"], "MEASURED")
        self.assertIsNone(record["shading"])
        self.assertIn("analytic_intersection_ms", record["stages"])


class StagedPathTest(unittest.TestCase):
    def cell(self, arm="box_proxy", count=2, width=64, height=48):
        return CharacterizationCell(arm, Camera(width=width, height=height, far_range_m=20.0),
                                    bench.benchmark_grid(count), device="cpu")

    def test_staged_mesh_render_reproduces_the_production_buffers_exactly(self):
        cell = self.cell()
        reference = cell.gbuffer()
        staged, stages = bench.staged_mesh_render(cell)
        self.assertTrue(bench.buffers_equal(reference, staged))
        self.assertEqual(sorted(stages), ["clear_ms", "finalize_ms", "normal_face_pass_ms",
                                          "range_pass_ms"])

    def test_buffers_equal_notices_a_single_changed_value(self):
        import torch
        from dataclasses import replace
        cell = self.cell()
        reference = cell.gbuffer()
        changed = replace(reference, range_m=reference.range_m.clone())
        flat = changed.range_m.view(-1)
        flat[0] = flat[0] + 1.0
        self.assertFalse(bench.buffers_equal(reference, changed))

    def test_staged_analytic_render_matches_the_analytic_arm(self):
        cell = self.cell(arm="analytic_sphere")
        reference = cell.gbuffer()
        staged, stages = bench.staged_analytic_render(cell)
        self.assertTrue(bench.buffers_equal(reference, staged))
        self.assertEqual(list(stages), ["analytic_intersection_ms"])


class ConfigurationTest(unittest.TestCase):
    def test_the_six_preregistered_configurations_are_present_and_distinct(self):
        self.assertEqual(len(bench.CONFIGURATIONS), 6)
        names = [name for name, _, _ in bench.CONFIGURATIONS]
        self.assertEqual(len(set(names)), 6)
        arms = {arm for _, arm, _ in bench.CONFIGURATIONS}
        self.assertEqual(arms, {"analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh"})
        self.assertIn(None, [shading for _, _, shading in bench.CONFIGURATIONS])

    def test_the_matrix_and_repeat_counts_match_the_preregistration(self):
        self.assertEqual(bench.RESOLUTIONS, ((160, 90), (320, 240), (640, 480)))
        self.assertEqual(bench.SCENE_COUNTS, (1, 8, 32))
        self.assertGreaterEqual(bench.WARMUP_RENDERS, 20)
        self.assertGreaterEqual(bench.MEASURED_RENDERS, 100)
        self.assertEqual(bench.OUTPUT_BYTES_MAX, int(1.5 * 1024 ** 3))
        self.assertEqual(bench.WARMUP_SECONDS_MAX, 2.0)
        self.assertEqual(bench.TORCH_RESERVED_MAX_BYTES, int(5.0 * 1024 ** 3))


if __name__ == "__main__":
    unittest.main()
