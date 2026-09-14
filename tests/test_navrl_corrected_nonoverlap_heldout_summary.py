from __future__ import annotations

import importlib.util
import subprocess
import unittest

from history_helpers import require_research_history
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heldout_summary",
    ROOT / "tools/summarize_navrl_corrected_nonoverlap_heldout.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RESULT = ROOT / "results/navrl_corrected_nonoverlap_physical_off_heldout_seed313"
MISSING_LOGS = sorted(
    f"{bars}bars.log" for bars in MODULE.EXPECTED_BARS if not (RESULT / f"{bars}bars.log").is_file()
)


class HeldoutSummaryTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_fraction(self):
        lo, hi = MODULE.wilson95(1715, 2049)
        self.assertLess(lo, 1715 / 2049)
        self.assertGreater(hi, 1715 / 2049)

    def test_runtime_sources_are_still_attested(self):

        require_research_history(ROOT)
        """Source attestation must survive disk reclamation, because later work cites this sweep.

        The snapshot directory and the cell logs were reclaimed, and eleven runtime files have
        since been edited in the worktree. Every one of the 330 files must still resolve to the
        SHA-256 the manifest recorded -- from the snapshot, the current file, the recorded commit,
        or the archived evaluator.
        """
        import json

        manifest = json.loads((RESULT / "source_manifest.json").read_text(encoding="utf-8"))
        _, verification = MODULE.verify_runtime_sources(RESULT.resolve(), manifest)
        self.assertEqual(sum(verification.values()), len(manifest["runtime_files"]))
        self.assertEqual(sum(verification.values()), 330)
        self.assertGreater(verification["git_history"], 0, "commit fallback never exercised")

    @unittest.skipIf(MISSING_LOGS, f"cell logs reclaimed for disk space: {MISSING_LOGS}")
    def test_frozen_results_validate_with_explicit_metadata_erratum(self):
        result = MODULE.summarize(RESULT)
        self.assertEqual(result["status"], "COMPLETE_VALID_WITH_METADATA_ERRATUM")
        self.assertEqual([cell["bars"] for cell in result["cells"]], [70, 85, 100, 115, 130, 145])
        self.assertFalse(result["metadata_erratum"]["outcome_affecting"])
        self.assertFalse(result["metadata_erratum"]["raw_evidence_modified"])
        self.assertFalse(result["interpretation"]["resume_authorized"])
        self.assertFalse(result["interpretation"]["routed_ppo_authorized"])
        self.assertLess(result["density_trend"]["capture_delta_70_to_145_pp"], -18.0)


class GitBlobRecoveryTest(unittest.TestCase):
    """The fallback that lets a pruned snapshot still be verified."""

    def test_matches_git_for_a_tracked_file_at_head(self):
        path = "tools/summarize_navrl_corrected_nonoverlap_heldout.py"
        blob = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"HEAD:{path}"])
        import hashlib

        self.assertEqual(
            MODULE.git_blob_sha256(ROOT, "HEAD", path), hashlib.sha256(blob).hexdigest()
        )

    def test_returns_none_without_a_commit_or_path(self):
        self.assertIsNone(MODULE.git_blob_sha256(ROOT, "", "tools/anything.py"))
        self.assertIsNone(MODULE.git_blob_sha256(ROOT, "HEAD", "no/such/file/anywhere.py"))
        self.assertIsNone(MODULE.git_blob_sha256(ROOT, "0" * 40, "tools/navrl_stats.py"))


if __name__ == "__main__":
    unittest.main()
