"""Every external number must carry a source that a reader can actually check.

The risk this guards against is a number entering the record with a plausible
label and no retrievable origin. Blogs, summary sites and arbitrary forks are not
acceptable origins for a primary value; a third-party measurement is acceptable
only when it is labelled as one.
"""
import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/literature_quantitative_ledger_2026-09-18.json"

# Primary-source hosts, in the preregistered priority order:
# primary paper, official project page, official repository.
ALLOWED_HOSTS = {"arxiv.org", "doi.org", "dx.doi.org", "ieeexplore.ieee.org",
                 "github.com", "proceedings.mlr.press", "openreview.net"}


class ExternalSources(unittest.TestCase):
    def setUp(self):
        self.entries = json.loads(LEDGER.read_text(encoding="utf-8"))["entries"]

    def test_ledger_is_not_empty(self):
        self.assertGreater(len(self.entries), 0)

    def test_every_source_url_is_a_checkable_primary_host(self):
        bad = []
        for e in self.entries:
            host = (urlparse(e["source_url"]).netloc or "").lower()
            host = host[4:] if host.startswith("www.") else host
            if host not in ALLOWED_HOSTS:
                bad.append(f"{e['work']} / {e['metric']}: host {host!r}")
        self.assertEqual([], bad, "\n".join(bad))

    def test_repository_values_are_labelled_official_code(self):
        for e in self.entries:
            host = (urlparse(e["source_url"]).netloc or "").lower()
            if host == "github.com":
                self.assertIn(e["source_type"], ("official_code", "official_project_page"),
                              f"{e['work']} cites a repository but claims {e['source_type']}")

    def test_third_party_measurements_are_named_as_such(self):
        for e in self.entries:
            if e["source_type"] == "third_party_benchmark":
                self.assertIn("third-party", e["allowed_interpretation"].lower(),
                              f"{e['work']}: third-party measurement not called out to the reader")

    def test_every_entry_records_the_version_actually_read(self):
        for e in self.entries:
            self.assertTrue(str(e.get("paper_version_date", "")).strip(),
                            f"{e['work']} / {e['metric']}: no paper_version_date")

    def test_unextracted_values_explain_where_they_were_looked_for(self):
        """A NOT_EXTRACTED must point at where the number should have been."""
        for e in self.entries:
            if e["value"] != "NOT_EXTRACTED":
                continue
            self.assertTrue(e["source_section"].strip(),
                            f"{e['work']}: NOT_EXTRACTED with no source_section")
            self.assertTrue(e["allowed_interpretation"].strip(),
                            f"{e['work']}: NOT_EXTRACTED with no explanation")

    def test_no_entry_asserts_a_value_it_also_calls_unextracted(self):
        for e in self.entries:
            text = json.dumps(e)
            if e["value"] == "NOT_EXTRACTED":
                self.assertNotRegex(
                    e.get("allowed_interpretation", ""), r"\b\d+(\.\d+)?\s*(pp|%)\b",
                    f"{e['work']}: NOT_EXTRACTED yet quotes a percentage")
            self.assertNotIn("TODO", text, f"{e['work']}: unfinished entry")


if __name__ == "__main__":
    unittest.main()
