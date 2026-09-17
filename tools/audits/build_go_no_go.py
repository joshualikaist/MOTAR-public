#!/usr/bin/env python3
"""Decide GPU_EVALUATION GO / NO_GO from the preflight artifacts.

Reads only artifacts other tools produced, so the verdict cannot be asserted by
hand. Every criterion maps to a file that must exist and say the right thing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "docs/audits"
MATRIX = AUDITS / "target_motion_defect_impact_matrix_2026-09-17.json"


def git(*a):
    return subprocess.check_output(["git", *a], cwd=str(ROOT), text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", required=True)
    args = parser.parse_args()
    out = Path(args.preflight_dir)

    manifest = json.loads((out / "cell_manifest.json").read_text())
    comparison = json.loads((out / "contract_comparison.json").read_text())
    validity_path = out / "e2_target_validity.json"
    validity = json.loads(validity_path.read_text()) if validity_path.is_file() else None
    matrix = json.loads(MATRIX.read_text())

    criteria = {
        "all_48_cell_contracts_match_prereg": (
            comparison["cells_checked"] == 48
            and comparison["cells_matched"] == 48
            and comparison["cells_confounded"] == 0),
        "checkpoint_present_and_sha_verified": (
            manifest["checkpoint_present"] and manifest["checkpoint_sha_verified"]),
        "e2_target_mechanism_valid": bool(validity and validity["verdict"] == "PASS"),
        "no_type_b_defect_affects_any_arm": (
            matrix["summary"]["type_b_evaluation_invalidating"] == []),
        "e2_unaffected_by_d1": matrix["summary"]["e2_unaffected_by_d1"],
        "bounded_branch_returns_before_legacy_block": (
            matrix["structural_facts"]["bounded_branch"]["unconditional_return"]
            and matrix["structural_facts"]["legacy_block_after_branch_return"]),
        "bounded_executor_consumes_no_rng": (
            matrix["structural_facts"]["bounded_executor_consumes_no_rng"]),
        "git_tree_clean": git("status", "--porcelain") == "",
    }
    failed = sorted(k for k, v in criteria.items() if not v)
    verdict = "GPU_EVALUATION_GO" if not failed else "GPU_EVALUATION_NO_GO"

    report = {
        "kind": "e0_e2_gpu_evaluation_go_no_go",
        "git_commit": git("rev-parse", "HEAD"),
        "verdict": verdict,
        # A GO here means the preflight is satisfied, NOT that the run may start.
        "execution_status": "AUTHORIZATION_REQUIRED",
        "execution_note": (
            "GO is a statement about the preflight, not an authorisation. No GPU work "
            "runs until the operator approves it; the research-authority gate is "
            "separate and also applies."),
        "criteria": criteria,
        "failed_criteria": failed,
        "cells": manifest["total_cells"],
        "episodes_total": manifest["total_episodes"],
        "e2_validity_verdict": validity["verdict"] if validity else "NOT_RUN",
        "defect_arm_impact": matrix["summary"]["arm_impact"],
        "interpretation_contract": {
            "H": "in-distribution historical reproduction / reference",
            "E0_E1_E2": "frozen-policy generalization arms, all OUTSIDE training distribution",
            "H_vs_E2": ("target-distribution shift PLUS target-dynamics shift; "
                        "must NOT be read as 'E2 is harder'"),
            "E0_E1_E2_mutual": ("matched bounded-dynamics contrast, interpretable among "
                                "themselves, but all outside the historical training "
                                "distribution"),
            "matching_kind": "distributional, not paired-layout (see the RNG audit)",
        },
        "analysis_order_when_authorised": [
            "record integrity", "target validity gates", "contract equivalence",
            "target mechanism statistics", "observation distribution shift",
            "command-vs-actual control diagnostics", "close-approach outcomes",
            "retraining verdict",
        ],
    }
    (out / "go_no_go.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    width = max(len(k) for k in criteria)
    for name, ok in sorted(criteria.items()):
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
    print(f"\n{verdict}")
    print(f"execution_status = AUTHORIZATION_REQUIRED")
    if failed:
        print("failed: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
