"""Recalculate archived descriptive statistics only; no simulator, control or GPU imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

RAW = (Path(__file__).resolve().parents[1] / "results" /
       "target_appearance_in_sim_2026-09-12" / "raw.json")


class AppearanceReportEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = RAW.read_bytes()
        cls.data = json.loads(cls.payload)

    def test_archived_bytes_are_unchanged(self):
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(),
                         "0589df29f51cace0ac7c756c29531f6f53021aaf739bf6793eac4d95046700a6")

    def test_sample_lengths_types_and_finiteness(self):
        for name in ("v2", "v3"):
            arm = self.data[name]
            self.assertEqual(arm["appearance"], name)
            for field in ("ranges", "pixels"):
                self.assertEqual(len(arm[field]), 3840)
                for value in arm[field]:
                    self.assertIn(type(value), (int, float))
                    self.assertTrue(math.isfinite(value))
                    self.assertGreater(value, 0)
            self.assertTrue(all(value == int(value) for value in arm["pixels"]))

    def test_stored_scalar_ranges_agree_not_full_trajectories(self):
        self.assertEqual(self.data["v2"]["ranges"], self.data["v3"]["ranges"])

    def test_pixel_counts_agree_not_masks_or_images(self):
        self.assertEqual(self.data["v2"]["pixels"], self.data["v3"]["pixels"])

    def test_all_defined_count_ratios_are_one(self):
        ratios = [right / left for left, right in
                  zip(self.data["v2"]["pixels"], self.data["v3"]["pixels"])]
        self.assertEqual(len(ratios), 3840)
        self.assertEqual(set(ratios), {1.0})

    def test_reported_bins_cover_all_samples_and_reproduce_medians(self):
        for name in ("v2", "v3"):
            arm = self.data[name]
            covered = 0
            for low, high, count, median in ((4, 5, 1196, 18.0),
                                             (5, 6, 1804, 6.0),
                                             (6, 7, 840, 6.0)):
                values = [px for distance, px in zip(arm["ranges"], arm["pixels"])
                          if low <= distance < high]
                self.assertEqual(len(values), count)
                self.assertEqual(statistics.median(values), median)
                covered += len(values)
            self.assertEqual(covered, len(arm["ranges"]))


if __name__ == "__main__":
    unittest.main()
