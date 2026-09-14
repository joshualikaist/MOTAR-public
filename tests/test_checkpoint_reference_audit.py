"""The checkpoint keep-set must be derived, never remembered (OPERATIONS.md rule 8-A).

Prose already said "do not delete the terminal checkpoint" and four cited checkpoints were deleted
anyway. These tests pin the three keep reasons and the fail-safe, on synthetic run trees, so the
rule cannot quietly weaken.

Run: PYTHONNOUSERSITE=1 python tests/test_checkpoint_reference_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ckpt_audit", ROOT / "tools/audit_checkpoint_references.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PatternTest(unittest.TestCase):
    def test_only_periodic_saves_are_ever_deletable(self):
        self.assertTrue(AUDIT.INTERMEDIATE.match("last_gen_ppo_ep_1200_rew_31.4.pth"))
        self.assertTrue(AUDIT.INTERMEDIATE.match("last_gen_ppo_ep_500_rew__12.5_.pth"))
        # Anything the tool does not model must fall outside the deletable pattern.
        for name in ("gen_ppo.pth", "gen_ppo_rlnorm.pth", "hand_renamed_export.pth",
                     "last_gen_ppo_ep_XXXX.pth", "checkpoint_snapshot.pth"):
            self.assertIsNone(AUDIT.INTERMEDIATE.match(name), name)

    def test_run_checkpoint_pattern_excludes_fixtures_and_placeholders(self):
        self.assertTrue(AUDIT.RUN_CKPT.match("gen_ppo.pth"))
        self.assertTrue(AUDIT.RUN_CKPT.match("last_gen_ppo_ep_25000_rew_39.742134.pth"))
        for name in ("hostile.pth", "x.pth", "last_gen_ppo_ep_XXXX_rew_YY.pth", "model.pth"):
            self.assertIsNone(AUDIT.RUN_CKPT.match(name), name)


class TerminalTest(unittest.TestCase):
    """A run's final weights are kept even when nothing cites them yet."""

    def _runs(self, tree):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for run, names in tree.items():
            for name in names:
                path = root / run / "nn" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
        return root

    def test_highest_epoch_wins(self):
        root = self._runs({"runA": ["last_gen_ppo_ep_100_rew_1.0.pth",
                                    "last_gen_ppo_ep_900_rew_2.0.pth",
                                    "last_gen_ppo_ep_450_rew_1.5.pth"]})
        AUDIT.RUNS = root
        self.assertEqual(AUDIT.terminal_checkpoints()["runA"].name,
                         "last_gen_ppo_ep_900_rew_2.0.pth")

    def test_gen_ppo_is_the_terminal_when_a_run_saved_nothing_else(self):
        root = self._runs({"runB": ["gen_ppo.pth"]})
        AUDIT.RUNS = root
        self.assertEqual(AUDIT.terminal_checkpoints()["runB"].name, "gen_ppo.pth")

    def test_a_run_with_last_gen_does_not_fall_back_to_gen_ppo(self):
        root = self._runs({"runC": ["gen_ppo.pth", "last_gen_ppo_ep_10_rew_1.0.pth"]})
        AUDIT.RUNS = root
        self.assertEqual(AUDIT.terminal_checkpoints()["runC"].name,
                         "last_gen_ppo_ep_10_rew_1.0.pth")


class LiveRepositoryTest(unittest.TestCase):
    """The real tree: whatever a result cites must still be there, or be reported as lost."""

    def test_cited_checkpoints_are_reported_when_absent(self):
        cited = AUDIT.cited_by_results()
        self.assertTrue(cited, "no result JSON cites a run checkpoint; the scan is broken")
        for name in cited:
            self.assertTrue(AUDIT.RUN_CKPT.match(name), f"non-checkpoint leaked into the scan: {name}")

    def test_the_operations_rule_names_this_tool(self):
        text = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")
        self.assertIn("audit_checkpoint_references.py", text)
        self.assertIn("--verify", text)


if __name__ == "__main__":
    unittest.main()
