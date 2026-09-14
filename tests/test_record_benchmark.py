"""Only tiny synthetic fixtures; timing is not asserted as a performance gate."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import benchmark_record_integrity as benchmark
import run_record_validation_checks as regression


class RecordBenchmarkTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_valid(self):
        a, contract = benchmark.fixture(3)
        b, _ = benchmark.fixture(3)
        self.assertEqual(benchmark.encode(a), benchmark.encode(b))
        self.assertEqual(benchmark.audit_bytes(benchmark.encode(a), contract)["status"], "VALID_FOR_DECLARED_CONTRACT")

    def test_percentile_uses_linear_interpolation(self):
        self.assertAlmostEqual(benchmark.percentile([0, 10, 20], .95), 19.)

    def test_small_run_records_all_boundaries_and_samples(self):
        with patch.object(benchmark, "COUNTS", (3,)), patch.object(benchmark, "REPEATS", 1), patch.object(benchmark, "SAMPLES", 2):
            result = benchmark.benchmark()
        self.assertEqual(len(result["cells"]), 1)
        self.assertEqual(result["cells"][0]["timing"]["strict_parse"]["n"], 2)
        self.assertIn("live instrumentation overhead", result["not_estimated"])
        self.assertEqual(set(result["cells"][0]["raw_ms"]), {
            "serialize", "sha256", "strict_parse", "hash_parse_validate", "serialize_hash_parse_validate"})

    def test_test_allowlist_excludes_task_policy_control_suites(self):
        selected = [p for pattern in regression.PATTERNS for p in (ROOT/"tests").glob(pattern)]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(selected)
        self.assertFalse(any(p.name.startswith("test_navrl") for p in selected))


if __name__ == "__main__":
    unittest.main()
