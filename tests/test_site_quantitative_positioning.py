"""The public page's quantitative positioning section must stay bound to the ledger.

A number typed onto the site and nowhere else is unverifiable by a reader. Every
numeric value shown in the external-context table must therefore also exist in the
ledger, where it carries its metric wording, conditions and source section.

This also guards the running-evaluation boundary: a partially executed GPU matrix
must not reach the public page, whatever else changes.
"""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs/status/index.html"
LEDGER = ROOT / "docs/literature_quantitative_ledger_2026-09-18.json"
REGISTRY = ROOT / "docs/quantitative_positioning_registry.json"
FIGURE = ROOT / "docs/assets/paper/quantitative-positioning-2026-09-18.svg"

NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    return {float(m) for m in NUM.findall(text)}


class SiteSection(unittest.TestCase):
    def setUp(self):
        self.site = INDEX.read_text(encoding="utf-8")
        self.assertIn("<h3>6.2 Quantitative positioning</h3>", self.site)
        body = self.site.split("<h3>6.2 Quantitative positioning</h3>", 1)[1]
        self.section = body.split("<h3>6.3 Relation to published systems</h3>", 1)[0]

    def test_boundary_statement_is_present_and_unambiguous(self):
        for phrase in ("not head-to-head",
                       "under each publication",
                       "Class A matched external comparisons remain"):
            self.assertIn(phrase, self.section, f"missing boundary phrase: {phrase!r}")

    def test_section_states_class_a_is_zero(self):
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(0, registry["matched_external"]["class_a_count"])
        plain = re.sub(r"<[^>]+>", "", self.section)
        self.assertRegex(plain, r"(?i)class a matched external comparisons remain\s+zero")

    def test_every_work_in_the_table_exists_in_the_ledger(self):
        ledger_works = {e["work"] for e in json.loads(LEDGER.read_text(encoding="utf-8"))["entries"]}
        cells = re.findall(r"<tr><td><a href=\"[^\"]+\">([^<]+)</a></td>", self.section)
        self.assertGreaterEqual(len(cells), 10, "external table lost rows")
        for work in cells:
            self.assertIn(work, ledger_works, f"site shows {work!r}, absent from the ledger")

    def test_every_number_shown_on_the_site_exists_in_the_ledger(self):
        ledger_numbers = numbers(LEDGER.read_text(encoding="utf-8"))
        # Only the reported-number and benchmark-context cells carry external values.
        rows = re.findall(r"<tr>(.*?)</tr>", self.section, re.S)
        unbound = []
        for row in rows:
            cells = re.findall(r"<td>(.*?)</td>", row, re.S)
            for cell in cells[1:3]:
                plain = re.sub(r"<[^>]+>", " ", cell).replace("&nbsp;", " ")
                for value in numbers(plain):
                    if not any(abs(value - known) < 1e-9 for known in ledger_numbers):
                        unbound.append(f"{value} in {plain.strip()[:80]!r}")
        self.assertEqual([], unbound,
                         "site numbers with no ledger entry:\n  " + "\n  ".join(unbound))

    def test_the_positioning_figure_is_present_and_referenced(self):
        self.assertTrue(FIGURE.exists(), "positioning figure missing")
        self.assertIn("quantitative-positioning-2026-09-18.svg", self.section)

    def test_figure_marks_are_declared_not_to_be_scores(self):
        self.assertRegex(self.section, r"(?i)not a performance score")


class TableNumberingStaysConsistent(unittest.TestCase):
    def test_captions_are_unique_and_sequential(self):
        site = INDEX.read_text(encoding="utf-8")
        found = [int(n) for n in re.findall(r"<strong>Table (\d+)\.</strong>", site)]
        self.assertEqual(sorted(found), found, f"table captions out of order: {found}")
        self.assertEqual(len(found), len(set(found)), f"duplicate table numbers: {found}")

    def test_no_dangling_reference_to_a_missing_table(self):
        site = INDEX.read_text(encoding="utf-8")
        defined = {int(n) for n in re.findall(r"<strong>Table (\d+)\.</strong>", site)}
        referenced = {int(n) for n in re.findall(r"\bTable (\d+)\b", site)}
        self.assertTrue(referenced <= defined,
                        f"referenced but undefined: {sorted(referenced - defined)}")


class RunningEvaluationBoundary(unittest.TestCase):
    def test_no_partial_gpu_result_reaches_the_site(self):
        site = INDEX.read_text(encoding="utf-8")
        self.assertNotIn("target_motion_e0_e2_evaluation", site)
        for arm in ("H_historical", "E0_static", "E1_cv", "E2_obstacle_aware"):
            self.assertNotIn(arm, site, f"site references evaluation arm {arm}")

    def test_target_motion_row_is_either_pending_or_fully_sourced(self):
        """The evaluation may be pending or finished, but never half-reported.

        While it runs, the row carries no value. Once finished, it must carry a
        verdict AND a source. The state this forbids is a value on the site with
        no result document behind it.
        """
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        row = next(e for e in registry["internal_motar"]
                   if e["id"] == "target_motion_generalization")
        if row["verdict"] == "RESULT_PENDING":
            self.assertIsNone(row.get("value_pp"))
            self.assertIsNone(row.get("source_path"))
            self.assertIn("RESULT PENDING", INDEX.read_text(encoding="utf-8"))
            return
        self.assertIsNotNone(row.get("value_pp"), "finalized row carries no value")
        self.assertTrue(row.get("source_path"), "finalized row cites no source")
        self.assertTrue(row.get("ci95_pp"), "finalized row carries no interval")

    def test_a_finalized_row_never_claims_significance_at_three_seeds(self):
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        row = next(e for e in registry["internal_motar"]
                   if e["id"] == "target_motion_generalization")
        if row["verdict"] == "RESULT_PENDING":
            return
        blob = json.dumps(row).lower()
        self.assertNotIn("statistically significant", blob)
        self.assertIn("permutation", blob,
                      "the n=3 inference limitation must travel with the row")


if __name__ == "__main__":
    unittest.main()
