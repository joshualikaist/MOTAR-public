"""Archival corruption checks only; no simulator, GPU, or evaluation is launched."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from history_helpers import require_research_history
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13"


def load_audit():
    spec = importlib.util.spec_from_file_location("d8b_archive", CAMPAIGN / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveContracts(unittest.TestCase):
    def setUp(self):
        self.audit = load_audit()

    def test_complete_grid(self):
        self.audit.check_grid(list(self.audit.CELLS))

    def test_missing_cell(self):
        with self.assertRaisesRegex(ValueError, "missing/extra/duplicate"):
            self.audit.check_grid(list(self.audit.CELLS[:-1]))

    def test_duplicate_cell(self):
        with self.assertRaisesRegex(ValueError, "missing/extra/duplicate"):
            self.audit.check_grid(list(self.audit.CELLS[:-1]) + [self.audit.CELLS[0]])

    def test_extra_cell(self):
        with self.assertRaisesRegex(ValueError, "missing/extra/duplicate"):
            self.audit.check_grid(list(self.audit.CELLS) + ["unregistered"])

    def raw(self):
        return self.audit.read(CAMPAIGN / "cells/s593_analytic_flat/70bars.json")

    def test_counts_and_rates(self):
        self.audit.check_counts(self.raw())

    def test_inconsistent_rate_rejected(self):
        row = self.raw()
        row["outcome"]["capture_rate"] += 0.001
        with self.assertRaisesRegex(ValueError, "rate not derived"):
            self.audit.check_counts(row)

    def test_nan_rate_rejected(self):
        row = self.raw()
        row["outcome"]["capture_rate"] = float("nan")
        with self.assertRaises(ValueError):
            self.audit.check_counts(row)

    def test_noninteger_outcome_rejected(self):
        row = self.raw()
        row["outcome"]["captured"] = float(row["outcome"]["captured"])
        with self.assertRaisesRegex(ValueError, "invalid outcome count"):
            self.audit.check_counts(row)

    def test_accounting_mismatch_rejected(self):
        row = self.raw()
        row["outcome"]["timeout"] += 1
        with self.assertRaisesRegex(ValueError, "accounting"):
            self.audit.check_counts(row)

    def test_condition_change_rejected(self):
        rows = [{"seed": 1, "fixed": 2}, {"seed": 2, "fixed": 3}]
        with self.assertRaisesRegex(ValueError, "changed outside"):
            self.audit.same_except(rows, {"seed"}, "test")

    def test_duplicate_json_key_rejected(self):
        path = mock.Mock()
        path.read_text.return_value = '{"seed": 1, "seed": 2}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.audit.read(path)

    def test_portable_archive_without_local_snapshots_or_absolute_aliases(self):

        require_research_history(ROOT)
        with tempfile.TemporaryDirectory(prefix="d8b-archive-test-") as tmp:
            package = Path(tmp)
            # Only the retained evidence: no checkpoint, source snapshot or absolute symlink.
            names = ["summary.json", "source_bundle/source_manifest.json",
                     "source_bundle/python_environment.txt"]
            names += ["cells/%s/70bars.%s" % (cell, suffix)
                      for cell in self.audit.CELLS for suffix in ("json", "receipt.json", "log")]
            for name in names:
                destination = package / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CAMPAIGN / name, destination)
            with mock.patch.object(self.audit, "HERE", package):
                report = self.audit.audit()
                self.assertEqual(report["status"], "ARCHIVE_INTEGRITY_PASS")
                self.assertEqual(report["independent_primary"]["verdict"], "MATERIAL_LOSS")
                self.assertFalse(report["local_snapshots_checked"])

    def test_summary_primary_shape_tampering_rejected(self):

        require_research_history(ROOT)
        original = self.audit.read
        def changed(path):
            result = original(path)
            if path.name == "summary.json":
                result = copy.deepcopy(result)
                result["primary"]["values"] = []
            return result
        with mock.patch.object(self.audit, "read", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "primary shape mismatch"):
                self.audit.audit()


if __name__ == "__main__":
    unittest.main()
