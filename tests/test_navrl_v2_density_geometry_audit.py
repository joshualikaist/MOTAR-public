"""Receipt contract for the 2026-08-27 corrected-v2 density geometry audit.

Does not re-run the 23-minute CPU audit.  The JSON is the frozen measurement;
this test only checks that the gate arithmetic and SHA still match the documents.
"""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "results"
    / "navrl_v2_density_geometry_audit_2026-08-27"
    / "density_geometry_canonical_6_28.json"
)
EXPECTED_SHA = "6b1f1b36cf73409d0c09483c3e1767b7ff196aecf0b95769f0e07d8dffa268d5"


class TestNavrlV2DensityGeometryAudit(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_sha_and_schema(self):
        digest = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_SHA)
        self.assertEqual(self.data["schema_version"], "navrl.v2-density-geometry-audit.v1")
        self.assertEqual(self.data["goal_band"]["name"], "canonical_6_28")
        thresholds = self.data["thresholds"]
        self.assertEqual(thresholds["connectivity_min"], 0.95)
        self.assertEqual(thresholds["no_route_max"], 0.05)
        self.assertEqual(thresholds["generation_failures_max"], 0)

    def test_binding_verdict_keeps_205_and_rejects_300(self):
        binding = self.data["verdict"]["body_plus_tracking"]
        self.assertTrue(binding["205_passes"])
        self.assertEqual(binding["highest_passing_density"], 250)
        self.assertNotIn(300, binding["passing_densities"])
        row_205 = self.data["per_density"]["205"]
        row_300 = self.data["per_density"]["300"]
        self.assertEqual(row_205["generation_failures"], 0)
        self.assertGreaterEqual(row_205["body_plus_tracking"]["connectivity"], 0.95)
        self.assertLessEqual(row_205["body_plus_tracking"]["no_route_fraction"], 0.05)
        self.assertTrue(
            row_300["body_plus_tracking"]["connectivity"] < 0.95
            or row_300["body_plus_tracking"]["no_route_fraction"] > 0.05
        )


if __name__ == "__main__":
    unittest.main()
