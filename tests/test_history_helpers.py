"""Release-snapshot detection remains valid after public follow-up commits."""
from pathlib import Path
import tempfile
import unittest

import history_helpers


ROOT = Path(__file__).resolve().parents[1]


class HistoryHelpersTest(unittest.TestCase):
    def test_current_checkout_matches_its_explicit_marker(self):
        self.assertEqual(
            history_helpers.is_public_snapshot(ROOT),
            (ROOT / "RELEASE_PROVENANCE.md").is_file(),
        )

    def test_explicit_release_marker_is_durable_without_commit_count_assumption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "RELEASE_PROVENANCE.md").write_text(
                "# Release provenance\n\n"
                "This directory is a **snapshot**, not a clone: content and none of its history.\n\n"
                "## History rewrite\n\n"
                "**None.** The research repository's history was not rewritten.\n"
            )
            self.assertTrue(history_helpers.is_public_snapshot(root))

    def test_file_name_alone_cannot_disable_history_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "RELEASE_PROVENANCE.md").write_text("not an attested release")
            self.assertFalse(history_helpers.is_public_snapshot(root))


if __name__ == "__main__":
    unittest.main()
