"""Public landing/status metadata contracts. No simulator, network or exact test-count pin."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_public_docs as docs


class PublicDocsTest(unittest.TestCase):
    def test_declared_public_surfaces(self):
        self.assertEqual(docs.check(), [])

    def test_volatile_results_not_landing_content(self):
        landing = (ROOT / "README.md").read_text()
        self.assertEqual(docs.landing_errors(landing), [])
        for suffix in ("\n1535 tests passed", "\ncapture 82.1%", "\nseed-541"):
            self.assertTrue(docs.landing_errors(landing + suffix))

    def test_negative_outcomes_cannot_be_promoted(self):
        manifest = json.loads((ROOT / "docs/status_manifest.json").read_text())
        for key in ("D4", "D5", "D6", "D7", "D8", "D9"):
            candidate = copy.deepcopy(manifest)
            candidate["track_d"][key]["status"] = "COMPLETE"
            self.assertTrue(docs.manifest_errors(candidate), key)
        manifest["real_flight_validated"] = True
        self.assertTrue(docs.manifest_errors(manifest))

    def test_missing_evidence_and_links_fail(self):
        manifest = json.loads((ROOT / "docs/status_manifest.json").read_text())
        manifest["track_d"]["D1"]["evidence"] = "does-not-exist.md"
        self.assertTrue(docs.manifest_errors(manifest))
        self.assertTrue(docs.local_link_errors("[bad](missing.md)", ROOT / "README.md"))
        self.assertTrue(docs.local_link_errors("[bad](../../outside)", ROOT / "README.md"))

    def test_all_nine_figures_preserved_off_landing_page(self):
        import re
        landing = (ROOT / "README.md").read_text()
        overview = (ROOT / "docs/results_overview_2026-09-12.md").read_text()
        self.assertEqual(len(re.findall(r"!\[", landing)), 1)
        self.assertEqual(len(re.findall(r"!\[", overview)), 9)
        self.assertTrue((ROOT / "docs/archive/readme_9732d12_2026-09-12.md").is_file())

    def test_ci_no_simulator_install_and_no_unearned_badge(self):
        ci = (ROOT / ".github/workflows/cpu-validation.yml").read_text()
        for command in ("fetch-depth: 0", "motar_doctor.py", "validate_renderer_export.py",
                        "cffconvert", "check_public_docs.py --schema"):
            self.assertIn(command, ci)
        self.assertNotIn("pip install -e .", ci)
        self.assertNotIn("bootstrap_second_machine", ci)
        self.assertNotIn("badge.svg", (ROOT / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
