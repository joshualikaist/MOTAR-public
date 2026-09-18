"""The quantitative positioning registry must stay source-bound and non-comparative.

Three separate obligations are enforced here.

1. Every MOTAR number on the site must exist in the result file it cites. Copying a
   number into a document is how numbers drift; binding it to its source is how they
   stop drifting.
2. External reported numbers are context, never a contrast. No document may subtract
   an external value from a MOTAR value or phrase one as an advantage over the other,
   because no external work has been run under MOTAR's contract.
3. Class A (matched external comparison) must remain empty until such a run exists.
   An entry cannot be promoted into it by editing prose.
"""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/quantitative_positioning_registry.json"
LEDGER_JSON = ROOT / "docs/literature_quantitative_ledger_2026-09-18.json"
LEDGER_MD = ROOT / "docs/literature_quantitative_positioning_2026-09-18.md"
MATRIX_MD = ROOT / "docs/paper_claim_evidence_matrix_2026-09-18.md"
INDEX = ROOT / "docs/status/index.html"

# Wording that would assert a cross-benchmark performance ranking.
FORBIDDEN_PATTERNS = [
    r"\boutperform",
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bsuperior to\b",
    r"\bbeats\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bfirst ever\b",
    r"\bSOTA\b",
]
# A number immediately followed by a comparative preposition naming another system is
# the specific construction the preregistration bans (e.g. "12 pp better than OPEN").
SUBTRACTION_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:pp|%|percentage points)\s+(?:better|worse|higher|lower|above|below)\s+than",
    re.IGNORECASE,
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RegistryIsSourceBound(unittest.TestCase):
    def setUp(self):
        self.assertTrue(REGISTRY.exists(), f"missing {REGISTRY}")
        self.reg = load(REGISTRY)

    def test_internal_numbers_exist_in_their_cited_source(self):
        missing, pending = [], []
        for entry in self.reg["internal_motar"]:
            if entry.get("verdict") == "RESULT_PENDING":
                continue
            src, token = entry.get("source_path"), entry.get("source_value_token")
            self.assertTrue(src, f"{entry['id']}: no source_path")
            self.assertTrue(token, f"{entry['id']}: no source_value_token")
            path = ROOT / src
            if not path.exists():
                # A row may cite a result that lands on this branch only after the
                # research branch is merged. That is legal ONLY while the row says so
                # out loud; it is reported as a skip, never as a silent pass.
                if str(entry.get("integration_status", "")).startswith("PENDING_MERGE"):
                    pending.append(f"{entry['id']} -> {src}")
                    continue
                missing.append(f"{entry['id']}: source file absent: {src}")
                continue
            if token not in path.read_text(encoding="utf-8", errors="replace"):
                missing.append(f"{entry['id']}: value {token!r} not found in {src}")
        self.assertEqual([], missing, "\n".join(missing))
        if pending:
            self.skipTest("awaiting merge of the research branch: " + "; ".join(pending))

    def test_pending_entries_carry_no_value(self):
        for entry in self.reg["internal_motar"]:
            if entry.get("verdict") != "RESULT_PENDING":
                continue
            for key in ("value_pp", "value_percent", "value_utility"):
                self.assertIn(entry.get(key, None), (None,),
                              f"{entry['id']} is RESULT_PENDING but carries {key}")
            self.assertIsNone(entry.get("source_path"),
                              f"{entry['id']} is RESULT_PENDING but cites a source")

    def test_class_a_remains_zero(self):
        matched = self.reg["matched_external"]
        self.assertEqual(0, matched["class_a_count"])
        self.assertEqual([], matched["results"],
                         "a matched external comparison appeared without a preregistered run")


class ExternalLedgerDiscipline(unittest.TestCase):
    REQUIRED = (
        "work", "metric", "value", "unit", "environment", "task", "target_type",
        "sensor", "speed_condition", "simulation_or_real", "source_type",
        "source_url", "source_section", "motar_benchmark_match",
        "comparison_class", "allowed_interpretation", "forbidden_interpretation",
    )

    def entries(self):
        if not LEDGER_JSON.exists():
            return []
        return load(LEDGER_JSON).get("entries", [])

    def test_every_external_number_has_a_full_provenance_record(self):
        problems = []
        for i, e in enumerate(self.entries()):
            for field in self.REQUIRED:
                if field not in e:
                    problems.append(f"entry {i} ({e.get('work','?')}): missing {field}")
            if e.get("source_type") not in (
                    "primary_paper", "official_code", "official_project_page",
                    "third_party_benchmark"):
                problems.append(f"entry {i}: bad source_type {e.get('source_type')!r}")
            if not str(e.get("source_url", "")).startswith("http"):
                problems.append(f"entry {i}: source_url is not a URL")
        self.assertEqual([], problems, "\n".join(problems))

    def test_no_external_entry_claims_benchmark_parity(self):
        for e in self.entries():
            self.assertFalse(
                e.get("motar_benchmark_match"),
                f"{e.get('work')} claims motar_benchmark_match=true; "
                "that requires a Class A matched run, which does not exist")
            self.assertIn(e.get("comparison_class"), ("B", "C"))

    def test_unextracted_values_are_marked_not_numbers(self):
        for e in self.entries():
            if e.get("value") == "NOT_EXTRACTED":
                self.assertIsNone(e.get("value_numeric"),
                                  f"{e.get('work')}: NOT_EXTRACTED but carries a number")


class NoCrossBenchmarkClaims(unittest.TestCase):
    def documents(self):
        for path in (LEDGER_MD, MATRIX_MD, INDEX,
                     ROOT / "docs/relation_to_published_systems_2026-09-16.md"):
            if path.exists():
                yield path

    def test_no_comparative_superiority_wording(self):
        problems = []
        for path in self.documents():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                for pattern in FORBIDDEN_PATTERNS:
                    for match in re.finditer(pattern, line, re.IGNORECASE):
                        # Stating the prohibition is allowed, but only when the
                        # negation sits immediately before the term. A blanket
                        # "line contains 'not'" exemption would excuse
                        # "outperforms X, but not in real flight". The no-period rule keeps
                        # the negation inside the same clause.
                        window = line[max(0, match.start() - 120):match.start()].lower()
                        if re.search(r"\b(?:not|never|no|neither|without|cannot|"
                                     r"does not|is not|are not)\b[^.]*$", window):
                            continue
                        problems.append(
                            f"{path.name}:{line_no}: {pattern} :: {line.strip()[:110]}")
        self.assertEqual([], problems, "\n".join(problems))

    def test_no_external_minus_motar_subtraction(self):
        problems = []
        for path in self.documents():
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if SUBTRACTION_PATTERN.search(line):
                    problems.append(f"{path.name}:{line_no}: {line.strip()[:110]}")
        self.assertEqual([], problems, "\n".join(problems))


class RunningEvaluationStaysOutOfThePublicRecord(unittest.TestCase):
    """A partially executed GPU matrix must not reach the site or the registry."""

    def test_site_shows_no_target_motion_numbers_while_pending(self):
        reg = load(REGISTRY)
        pending = [e for e in reg["internal_motar"]
                   if e.get("verdict") == "RESULT_PENDING"]
        if not pending or not INDEX.exists():
            return
        text = INDEX.read_text(encoding="utf-8", errors="replace")
        # The per-cell result directory name must never appear on the site.
        self.assertNotIn("target_motion_e0_e2_evaluation", text,
                         "the site references a running evaluation's result directory")


if __name__ == "__main__":
    unittest.main()
