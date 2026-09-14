"""Receipt and raw-evidence regression tests; no Warp/GPU execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from history_helpers import require_research_history
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_renderer_background_validation as compare

RUN = ROOT / "results/renderer_background_v1_2026-09-12"


class BackgroundComparisonTest(unittest.TestCase):
    def setUp(self):
        self.paths = [RUN / name / "run.json" for name in ("run1", "run2")]
        self.a, self.b = [compare.load_receipt(path) for path in self.paths]

    def test_real_receipts_pass(self):
        self.assertEqual(compare.compare_receipts(self.a, self.b)["verdict"], "PASS")

    def test_all_required_top_level_fields_are_mandatory(self):
        for key in compare.EXACT:
            with self.subTest(key=key):
                a = copy.deepcopy(self.a)
                del a[key]
                with self.assertRaises(ValueError):
                    compare.compare_receipts(a, a)

    def test_status_only_cannot_pass(self):
        with self.assertRaises(ValueError):
            compare.compare_receipts({"status": "TECHNICAL_PASS"}, {"status": "TECHNICAL_PASS"})

    def test_missing_nested_field_rejected_even_on_both_sides(self):
        for group, key in (("camera", "width"), ("runtime", "torch"), ("source", "files"),
                           ("checks", "0"), ("array_sha256", "depth_m")):
            with self.subTest(group=group):
                a = copy.deepcopy(self.a)
                del a[group][key]
                with self.assertRaises(ValueError):
                    compare.compare_receipts(a, a)

    def test_missing_per_view_metric_rejected(self):
        del self.a["per_view"][0]["light_rgb_mae"]
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_missing_nested_statistics_rejected(self):
        del self.a["per_view"][0]["appearance_statistics"]["flat"]["valid_pixels"]
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_per_view_mismatch_fails(self):
        self.b["per_view"][0]["light_rgb_mae"] += 0.1
        result = compare.compare_receipts(self.a, self.b)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["exact_comparisons"]["per_view"])

    def test_null_per_view_metric_rejected(self):
        self.a["per_view"][0]["light_rgb_mae"] = None
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_malformed_nested_statistics_rejected(self):
        self.a["per_view"][0]["appearance_statistics"]["flat"]["valid_pixels"] = "many"
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_empty_geometry_rejected(self):
        self.a["fixture"]["mesh"]["vertices"] = []
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_all_runtime_fields_compared(self):
        self.b["runtime"]["matmul_allow_tf32"] = not self.a["runtime"]["matmul_allow_tf32"]
        self.assertEqual(compare.compare_receipts(self.a, self.b)["verdict"], "FAIL")

    def test_failed_checks_cannot_be_hidden_by_pass_label(self):
        self.a["checks"]["0"]["lighting_sensitive"] = False
        self.assertEqual(compare.compare_receipts(self.a, self.a)["verdict"], "FAIL")

    def test_non_boolean_check_rejected(self):
        self.a["checks"]["0"]["lighting_sensitive"] = "true"
        with self.assertRaises(ValueError):
            compare.compare_receipts(self.a, self.a)

    def test_bad_schema_count_nan_and_hash_rejected(self):
        cases = (("schema", "unknown"), ("views", True), ("per_view", []),
                 ("seed", float("nan")), ("arrays_npz_sha256", "invalid"))
        for key, value in cases:
            with self.subTest(key=key):
                a = copy.deepcopy(self.a)
                a[key] = value
                with self.assertRaises(ValueError):
                    compare.compare_receipts(a, a)

    def test_dirty_source_cannot_pass(self):
        self.a["source"]["tracked_tree_clean"] = False
        self.assertEqual(compare.compare_receipts(self.a, self.a)["verdict"], "FAIL")

    def test_raw_arrays_and_historical_git_sources_verify(self):
        require_research_history(ROOT)
        for path, record in zip(self.paths, (self.a, self.b)):
            self.assertTrue(compare.verify_evidence(path, record))

    def test_tampered_raw_hash_rejected(self):
        self.a["arrays_npz_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Raw archive SHA mismatch"):
            compare.verify_evidence(self.paths[0], self.a)

    def test_tampered_decoded_array_hash_rejected(self):
        self.a["array_sha256"]["depth_m"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Decoded array SHA mismatch"):
            compare.verify_evidence(self.paths[0], self.a)

    def test_tampered_source_hash_rejected(self):

        require_research_history(ROOT)
        self.a["source"]["files"]["tools/run_renderer_background_validation.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source SHA mismatch"):
            compare.verify_evidence(self.paths[0], self.a)

    def test_missing_raw_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing raw arrays"):
                compare.verify_evidence(Path(directory) / "run.json", self.a)

    def test_malformed_duplicate_and_null_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            for content in ("{", "null", "[]", '{"status":1,"status":2}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    compare.load_receipt(path)

    def test_cli_read_only_pass(self):
        require_research_history(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(["--run", str(self.paths[0]), "--run", str(self.paths[1])]), 0)

    def test_cli_same_path_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            compare.main(["--run", str(self.paths[0]), "--run", str(self.paths[0])])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_valid_mismatch_exit_one(self):
        self.b["per_view"][0]["light_rgb_mae"] += 0.1
        with patch.object(compare, "load_receipt", side_effect=[self.a, self.b]), \
                patch.object(compare, "verify_evidence", return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(["--run", str(self.paths[0]), "--run", str(self.paths[1])]), 1)


if __name__ == "__main__":
    unittest.main()
