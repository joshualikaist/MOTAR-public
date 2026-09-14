"""CLAUDE.md and OPERATIONS.md must name the same working branch (OPERATIONS.md rule 0).

The 2026-09-07 deploy failure was not a git problem: the two branches never diverged. It was a
documentation problem. CLAUDE.md said the working branch was `research/navrl-env` while every
commit went to `main`, so pushes landed where the site was not served from. A stale sentence is
enough to lose a day, and nothing was checking it.

These tests read only the tracked documents. They do not touch git remotes or the network.

Run: PYTHONNOUSERSITE=1 python tests/test_branch_policy.py
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
OPERATIONS = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")

# The one branch every session works on. Changing it means changing both documents and this line.
WORKING_BRANCH = "main"
RETIRED = "research/navrl-env"


class BranchPolicyTest(unittest.TestCase):
    def test_operations_states_the_rule(self):
        self.assertIn("## 0. 브랜치 규칙", OPERATIONS)
        self.assertIn(f"`{WORKING_BRANCH}`", OPERATIONS)

    def test_operations_binds_the_pages_source_to_the_same_branch(self):
        """The failure mode was default-branch and Pages-source disagreeing; the rule must say both."""
        section = OPERATIONS.split("## 1. 처음 설치할 때")[0]
        self.assertIn("Pages", section)
        self.assertIn("/docs", section)
        self.assertIn(f"`{WORKING_BRANCH}`", section)

    def test_claude_md_names_the_same_branch(self):
        commit_rule = [line for line in CLAUDE.splitlines() if "커밋/푸시 전" in line]
        self.assertTrue(commit_rule, "CLAUDE.md lost its commit rule line")
        self.assertIn(WORKING_BRANCH, commit_rule[0])

    def test_the_retired_branch_appears_only_where_history_is_explained(self):
        """Structural, not word-matching: the retired name is confined to the "why" block.

        Anywhere else it would read as an instruction, which is exactly the sentence that cost a
        day. A future editor who wants to mention it elsewhere has to move it into that block or
        change this test deliberately.
        """
        marker = "**이 규칙이 생긴 이유**"
        self.assertIn(marker, OPERATIONS, "the rule lost the block that explains it")
        before, after = OPERATIONS.split(marker, 1)
        self.assertNotIn(RETIRED, before.split("## 1. 처음 설치할 때")[0],
                         f"{RETIRED} named in the rule itself, not in its history block")
        self.assertNotIn(RETIRED, CLAUDE, f"CLAUDE.md still names {RETIRED}")


if __name__ == "__main__":
    unittest.main()
