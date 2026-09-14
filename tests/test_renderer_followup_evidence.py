"""Independent arithmetic checks: no measurement estimators imported."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import unittest

from history_helpers import require_research_history

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def folder(stage):
    return ROOT / ("results/renderer_characterization_%s_2026-09-14" % stage)


def load(stage):
    return json.loads((folder(stage) / "summary.json").read_text())


class FollowupEvidence(unittest.TestCase):
    def test_all_receipts_match_files_committed_sources_and_preregistration_ancestor(self):
        require_research_history(ROOT)
        for stage in ("r5b", "r1b", "r2b", "r3b"):
            result = folder(stage)
            receipt = json.loads((result / "receipt.json").read_text())
            for name, sha in receipt["files_sha256"].items():
                self.assertEqual(hashlib.sha256((result/name).read_bytes()).hexdigest(), sha)
            source = receipt["source_manifest"]
            subprocess.run(["git", "merge-base", "--is-ancestor", source["preregistration_commit"],
                            source["commit"]], cwd=ROOT, check=True, capture_output=True)
            self.assertNotEqual(source["preregistration_commit"], source["commit"])
            for name, sha in source["source_sha256"].items():
                content = subprocess.check_output(["git", "show", source["commit"]+":"+name], cwd=ROOT)
                self.assertEqual(hashlib.sha256(content).hexdigest(), sha)
            self.assertEqual(load(stage)["causality_vs_d8b"], "NOT_TESTED")

    def test_r1b_all_288_rows_and_errors_recomputed_from_pixel_quantities(self):
        s = load("r1b")
        self.assertEqual(len(s["rows"]), 288)
        lookup = {(r["specimen"], tuple(r["resolution"]), r["view_index"]): r for r in s["rows"]}
        self.assertEqual(len(lookup), 288)
        for row in s["rows"]:
            self.assertEqual(row["status"], "MEASURED")
            ref = lookup[(row["specimen"], (2048, 1536), row["view_index"])]
            f, rf = row["focal_px"], ref["focal_px"]
            for raw, key, power in (("width_px", "width_over_f", 1), ("height_px", "height_over_f", 1),
                                    ("area_px", "area_over_f2", 2), ("depth_median_m", "depth_median_m", 0)):
                current, reference = row[raw]/f**power, ref[raw]/rf**power
                error = abs(current-reference)/reference
                self.assertAlmostEqual(error, row["errors"][key]["relative"], places=12)
            centre = (np.array(row["centroid_px"])-np.array(row["resolution"])/2)/f
            rc = (np.array(ref["centroid_px"])-np.array(ref["resolution"])/2)/rf
            error = np.linalg.norm(centre-rc)/(2*math.sqrt(ref["area_px"]/math.pi)/rf)
            self.assertAlmostEqual(error, row["errors"]["centroid_over_f"]["relative"], places=12)
        self.assertEqual(s["recommended_resolution"], [1280, 960])
        for level in s["levels"]:
            rows = [r for r in s["rows"] if r["resolution"] == level["resolution"]]
            self.assertEqual(len(rows), 72)
            worst = {k: max(r["errors"][k]["relative"] for r in rows) for k in level["worst_relative"]}
            self.assertEqual(worst, level["worst_relative"])
            passes = all(worst[k] <= (.02 if k in ("width_over_f", "height_over_f", "area_over_f2") else .01)
                         for k in worst)
            self.assertEqual(passes, level["qualified"])

    def test_r2b_separate_frozen_fit_and_validation_failure(self):
        s = load("r2b")
        fit = folder("r2b")/"fit.json"
        self.assertEqual(hashlib.sha256(fit.read_bytes()).hexdigest(), s["frozen_fit_sha256"])
        self.assertEqual(json.loads(fit.read_text())["fit"], s["fit"])
        fit_views = set(tuple(r["view"]) for r in s["phases"]["fit"]["rows"]["mesh"])
        val_views = set(tuple(r["view"]) for r in s["phases"]["validation"]["rows"]["mesh"])
        self.assertEqual((len(fit_views), len(val_views)), (6, 24))
        self.assertFalse(fit_views & val_views)
        for phase in s["phases"].values():
            gt = np.array([r["area_px"] for r in phase["rows"]["mesh"]])
            for arm in ("C0", "C1"):
                errors = np.abs(np.array([r["area_px"] for r in phase["rows"][arm]])/gt-1)
                stats = phase["statistics"][arm]
                np.testing.assert_array_equal(errors, stats["per_view_absolute_relative_error"])
                self.assertEqual(float(np.median(errors)), stats["median_absolute_relative_error"])
                self.assertEqual(float(np.quantile(errors, .9)), stats["p90_absolute_relative_error"])
                self.assertEqual(float(errors.max()), stats["worst_absolute_relative_error"])
        v = s["phases"]["validation"]["statistics"]["C1"]
        self.assertFalse(v["median_absolute_relative_error"] <= .05 and
                         v["p90_absolute_relative_error"] <= .10 and v["worst_absolute_relative_error"] <= .20)
        self.assertEqual(s["verdict"], "AREA_CONTROL_FAILED")

    def test_r3b_entropy_and_correlations_recomputed_no_synthetic_smoothing(self):
        s = load("r3b")
        self.assertEqual(len(s["rows"]), 72)
        for row in s["rows"]:
            h = row["normal_histogram"]
            counts = np.array(h["counts"])
            self.assertEqual(counts.sum(), row["area_px"])
            p = counts[counts > 0]/counts.sum()
            self.assertAlmostEqual(float(-(p*np.log2(p)).sum()), h["entropy_bits"], places=12)
            self.assertEqual(np.count_nonzero(counts), h["occupied_bins"])
            self.assertTrue(row["geometry_unchanged_by_shading"])
        for specimen, record in s["descriptive_correlations"].items():
            rows = [r for r in s["rows"] if r["specimen"] == specimen]
            x = np.array([r["normal_histogram"]["entropy_bits"] for r in rows])
            y = np.array([r["luminance_variance"] for r in rows])
            x, y = x-x.mean(), y-y.mean()
            self.assertAlmostEqual(float(x@y/np.sqrt((x@x)*(y@y))), record["pearson_r"], places=10)
        self.assertEqual(s["N1"]["status"], "UNSUPPORTED")
        self.assertEqual(s["N2"]["status"], "UNSUPPORTED")

    def test_r5b_24_cells_hashes_and_all_12000_raw_timing_samples(self):
        s = load("r5b")
        cells = s["cells"]
        self.assertEqual(len(cells), 24)
        self.assertEqual(len({(tuple(c["fixture"]), c["arm"], c["repeat"]) for c in cells}), 24)
        for c in cells:
            self.assertEqual(c["status"], "MEASURED")
            self.assertEqual(c["reference_hashes"], c["first_hashes"])
            self.assertEqual(c["reference_hashes"], c["output_hashes"])
            baseline = next(b for b in cells if b["fixture"] == c["fixture"] and b["arm"] == "A0")
            self.assertEqual(baseline["output_hashes"], c["output_hashes"])
            for key, values in c["raw_ms"].items():
                self.assertEqual(len(values), 100)
                for metric, value in (("mean_ms", np.mean(values)), ("median_ms", np.median(values)),
                                       ("p95_ms", np.quantile(values, .95))):
                    self.assertAlmostEqual(float(value), c["timing"][key][metric], places=10)
            self.assertAlmostEqual(c["scenes_per_second"], c["fixture"][2]*1000/np.mean(c["raw_ms"]["headline_total"]))
            if c["arm"] == "A1":
                self.assertIn("not full export", c["throughput_scope"])
        self.assertTrue(s["host_strategy_dominant"])
        self.assertTrue(all(f > .5 for f in s["a0_large_host_fractions"]))

    def test_first_rc_lineage_bytes_are_untouched(self):
        old = "32ee105309bd96b3b88f411da7b1f29d83341496"
        require_research_history(ROOT, old)
        paths = ["results/renderer_characterization_2026-09-13",
                 "docs/assets/paper/renderer-characterization-2026-09-13"]
        listed = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", old, "--", *paths],
                                         cwd=ROOT, text=True).splitlines()
        self.assertGreater(len(listed), 40)
        for path in listed:
            original = subprocess.check_output(["git", "show", old+":"+path], cwd=ROOT)
            self.assertEqual((ROOT/path).read_bytes(), original, path)


if __name__ == "__main__":
    unittest.main()
