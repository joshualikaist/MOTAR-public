from __future__ import annotations

import importlib.util
import json
import unittest

from history_helpers import require_research_history
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stopcap_summary", ROOT / "tools/summarize_navrl_v2_stopcap_screen.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RESULT = ROOT / "results/navrl_v2_ep25000_stopcap_seed49_screen"


class StopcapScreenContractTest(unittest.TestCase):
    def test_canonical_schema2_receipts_validate(self):
        require_research_history(ROOT)
        rows = [MODULE.load_cell(RESULT, tag) for tag in MODULE.TAGS]
        self.assertEqual([row["tag"] for row in rows], list(MODULE.TAGS))
        self.assertEqual([row["mode"] for row in rows], [
            "off", "fixed", "riskcap", "stopcap", "ttc",
        ])

    def test_frozen_summary_keeps_four_separate_verdicts(self):
        summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
        self.assertNotIn("verdict", summary)
        self.assertEqual(summary["verdict_m1"], "IMPLEMENTATION_VOID")
        self.assertEqual(summary["verdict_q1"], "MECHANISM_UNSUPPORTED")
        self.assertEqual(summary["verdict_q2"], "NOT_JUDGED_M1_VOID")
        self.assertEqual(summary["verdict_q3"], "FILTER_DEPENDENT")
        self.assertFalse(any(summary["q2_gates"].values()))


if __name__ == "__main__":
    unittest.main()
