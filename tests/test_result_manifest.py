"""The result manifest must index results without interpreting them, and must not go stale.

The builder's job is identity, provenance, status strings and links. The risk it guards against is
a second, unreviewed place where research findings are stated, so these tests check that verdicts
are copied verbatim, that no research quantity is extracted, and that every path and hash the
committed manifest asserts is still true.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_result_manifest as builder

# A release snapshot ships RELEASE_MANIFEST.json describing itself, and may also carry the research
# index for provenance. Validate the one that describes the tree these tests are running in.
RELEASE_MANIFEST = ROOT / "results/RELEASE_MANIFEST.json"
MANIFEST = RELEASE_MANIFEST if RELEASE_MANIFEST.is_file() else ROOT / "results/MANIFEST.json"
FORBIDDEN_QUANTITIES = ("capture_rate", "crash_rate", "reach_rate", "mean_nc", "improvement",
                        "failure_cause", "pp", "p_value", "confidence_interval")


class InterpretationBoundaryTest(unittest.TestCase):
    def test_status_strings_are_copied_not_normalised(self):
        for value in ("GEOMETRY_DEFECT", "  AREA_MATCH_FAILED  ", "material_loss"):
            self.assertEqual(builder.verbatim_status({"verdict": value}), value.strip())
        self.assertIsNone(builder.verbatim_status({}))
        self.assertIsNone(builder.verbatim_status({"verdict": ""}))
        self.assertIsNone(builder.verbatim_status("not a document"))

    def test_summary_wins_over_receipt_and_the_source_is_recorded(self):
        self.assertEqual(builder.verbatim_status({"status": "X", "verdict": "Y"}), "Y")

    def test_lifecycle_uses_literal_tokens_only(self):
        self.assertEqual(builder.lifecycle("GEOMETRY_DEFECT"), "negative_result")
        self.assertEqual(builder.lifecycle("AREA_CONTROL_FAILED"), "negative_result")
        self.assertEqual(builder.lifecycle("SUPERSEDED_BY_X"), "superseded")
        self.assertEqual(builder.lifecycle("TRANSFER_CHARACTERIZED"), "recorded")
        self.assertEqual(builder.lifecycle(None), "unlabelled")
        # A verdict that merely sounds bad is not reclassified by meaning.
        self.assertEqual(builder.lifecycle("SLOW_BUT_COMPLETE"), "recorded")

    def test_the_builder_extracts_no_research_quantity(self):
        source = (ROOT / "tools/build_result_manifest.py").read_text()
        tree = ast.parse(source)
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        for name in FORBIDDEN_QUANTITIES:
            self.assertNotIn(name, literals, name)

    def test_scope_statement_is_present_in_the_manifest(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertIn("No research quantity", manifest["scope"])
        self.assertEqual(manifest["schema"], "result_manifest_v1")


class ValidationTest(unittest.TestCase):
    def entry(self, **overrides):
        row = {"result_id": "a", "parent": None, "container": False, "legacy_exception": False,
               "summary_path": "results/a/summary.json", "receipt_path": "results/a/receipt.json",
               "readme_path": "results/a/README.md", "dangling_preregistrations": [],
               "source_manifest_state": None}
        row.update(overrides)
        return row

    def test_duplicate_result_id_is_reported(self):
        issues = builder.validate([self.entry(), self.entry()], [])
        self.assertEqual(issues["duplicate_result_id"], ["a"])

    def test_modern_gaps_are_reported_but_legacy_ones_are_not(self):
        modern = self.entry(result_id="modern", summary_path=None)
        legacy = self.entry(result_id="legacy", summary_path=None, legacy_exception=True)
        issues = builder.validate([modern, legacy], [])
        self.assertEqual(issues["modern_missing_summary"], ["modern"])

    def test_a_container_is_not_reported_as_missing_its_own_summary(self):
        container = self.entry(result_id="track", container=True, summary_path=None,
                               receipt_path=None, readme_path="results/track/README.md")
        issues = builder.validate([container], [])
        self.assertEqual(issues["modern_missing_summary"], [])
        self.assertEqual(issues["orphan_candidates"], [])

    def test_orphan_and_dangling_preregistration_are_reported(self):
        orphan = self.entry(result_id="orphan", summary_path=None, receipt_path=None,
                            readme_path=None)
        dangling = self.entry(result_id="dangling",
                              dangling_preregistrations=["docs/preregistration_missing.md"])
        issues = builder.validate([orphan, dangling], ["loose.csv"])
        self.assertEqual(issues["orphan_candidates"], ["orphan"])
        self.assertEqual(issues["dangling_preregistration"],
                         {"dangling": ["docs/preregistration_missing.md"]})
        self.assertEqual(issues["loose_files_outside_any_result"], ["loose.csv"])

    def test_source_drift_is_reported_separately_from_missing_files(self):
        drifted = self.entry(result_id="drift",
                             source_manifest_state={"drifted": ["tools/x.py"], "missing": []})
        gone = self.entry(result_id="gone",
                          source_manifest_state={"drifted": [], "missing": ["tools/y.py"]})
        issues = builder.validate([drifted, gone], [])
        self.assertEqual(issues["source_manifest_drift"], {"drift": ["tools/x.py"]})
        self.assertEqual(issues["source_manifest_missing_files"], {"gone": ["tools/y.py"]})


class CommittedManifestTest(unittest.TestCase):
    """Every path and hash the committed manifest asserts must still be true."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_every_recorded_path_exists_and_stays_inside_the_repository(self):
        for entry in self.manifest["results"]:
            for key in ("result_path", "summary_path", "receipt_path", "readme_path",
                        "preregistration_path", "source_manifest_path", "config_path"):
                value = entry.get(key)
                if value is None:
                    continue
                path = (ROOT / value).resolve()
                self.assertIn(ROOT.resolve(), path.parents, value)
                self.assertTrue(path.exists(), value)

    def test_recorded_hashes_match_the_files(self):
        checked = 0
        for entry in self.manifest["results"]:
            for key in ("summary", "receipt", "readme", "preregistration", "source_manifest",
                        "config"):
                path, digest = entry.get(key + "_path"), entry.get(key + "_sha256")
                if not path or not digest:
                    continue
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest,
                                 path)
                checked += 1
        self.assertGreater(checked, 100)

    def test_counts_agree_with_the_indexed_rows(self):
        rows = self.manifest["results"]
        counts = self.manifest["counts"]
        self.assertEqual(counts["results"], len(rows))
        self.assertEqual(counts["legacy_exception"], sum(r["legacy_exception"] for r in rows))
        self.assertEqual(counts["containers"], sum(r["container"] for r in rows))
        self.assertEqual(counts["child_results"], sum(r["parent"] is not None for r in rows))
        self.assertEqual(sum(counts["lifecycle"].values()), len(rows))
        self.assertEqual(len({r["result_id"] for r in rows}), len(rows))

    def test_untracked_result_directories_are_surfaced(self):
        """A result that exists on one machine and in no commit is a provenance gap, not a result.

        A snapshot export drops it silently, so the manifest has to name it; otherwise the research
        tree and any release built from it disagree about how many results exist, with no record of
        why.
        """
        manifest = self.manifest
        if not manifest.get("git_available", True):
            self.skipTest("a snapshot without Git cannot determine tracking")
        untracked = manifest["issues"]["untracked_results"]
        self.assertEqual(manifest["counts"]["untracked_in_git"], len(untracked))
        by_id = {row["result_id"]: row for row in manifest["results"]}
        for result_id in untracked:
            self.assertIs(by_id[result_id]["tracked_in_git"], False)

    def test_a_snapshot_without_git_does_not_reclassify_legacy_results(self):
        """Without a .git directory the builder must say "unknown", not "modern"."""
        entry = builder.describe.__doc__ or ""
        self.assertIn("tracked", entry.lower() + builder.tracked_prefixes.__doc__.lower())
        rows = [row for row in self.manifest["results"]
                if row["legacy_exception"] is None]
        self.assertEqual(rows, [], "a git-backed build must classify every result")

    def test_no_duplicate_and_no_dangling_preregistration_is_recorded(self):
        self.assertEqual(self.manifest["issues"]["duplicate_result_id"], [])
        self.assertEqual(self.manifest["issues"]["dangling_preregistration"], {})

    def test_preserved_negative_results_are_still_listed_as_such(self):
        by_id = {row["result_id"]: row for row in self.manifest["results"]}
        expected = {
            "renderer_characterization_2026-09-13/R1_geometry": "GEOMETRY_DEFECT",
            "renderer_characterization_2026-09-13/R2_apparent_area": "AREA_MATCH_FAILED",
            "renderer_characterization_2026-09-13/R3_shading": "SHADING_GATE_FAILED",
            "renderer_characterization_r2b_2026-09-14": "AREA_CONTROL_FAILED",
        }
        for result_id, status in expected.items():
            self.assertIn(result_id, by_id, result_id)
            self.assertEqual(by_id[result_id]["status"], status)
            self.assertEqual(by_id[result_id]["lifecycle"], "negative_result")


if __name__ == "__main__":
    unittest.main()
