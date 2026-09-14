#!/usr/bin/env python3
"""Apply the frozen decision rule to the seed-431 detector-coupling rerun."""

import json
import math
import sys
from pathlib import Path


ARMS = ["analytic_clean", "analytic_noise_1p0_binbias", "learned_v7"]


def delta_ci(base, arm):
    p0, p1 = base["captured"] / base["n"], arm["captured"] / arm["n"]
    delta = (p1 - p0) * 100
    se = math.sqrt(p0 * (1 - p0) / base["n"] + p1 * (1 - p1) / arm["n"])
    half = 1.959963985 * se * 100
    return delta, [delta - half, delta + half]


def main(root_raw):
    root = Path(root_raw)
    gate = json.loads((root / "quality_gate.json").read_text())
    cells = {}
    for arm in ARMS:
        p = root / arm / "205bars.json"
        j = json.loads(p.read_text())
        o = j["outcome"]
        cells[arm] = {
            "n": j["actual_episodes"], "captured": o["captured"],
            "capture": o["capture_rate"], "crash": o["crash_rate"],
            "timeout": o["timeout_rate"], "seed": j["condition"]["seed"],
        }
    base = cells[ARMS[0]]
    deltas = {}
    for arm in ARMS[1:]:
        d, ci = delta_ci(base, cells[arm])
        deltas[arm] = {"delta_pp": d, "ci95_pp": ci}

    dn, dv = deltas[ARMS[1]], deltas[ARMS[2]]
    if not gate["pass"]:
        verdict = "INCONCLUSIVE_NOISE_GATE_FAIL"
    elif dn["ci95_pp"][0] <= dv["delta_pp"] <= dn["ci95_pp"][1]:
        verdict = "SUPPORTED_OUTPUT_COUPLING_CONSISTENT_MATCHED_RETRAINING_REQUIRED"
    elif abs(dn["delta_pp"]) < 0.5 * abs(dv["delta_pp"]):
        verdict = "COUPLING_ALONE_INSUFFICIENT"
    else:
        verdict = "INCONCLUSIVE_EFFECT_MISMATCH"

    out = {"quality_gate": gate, "cells": cells, "deltas": deltas, "verdict": verdict}
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    rows = []
    for arm in ARMS:
        c = cells[arm]
        if arm == ARMS[0]:
            delta = "baseline"
        else:
            x = deltas[arm]
            delta = f"{x['delta_pp']:+.2f} pp [{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}]"
        rows.append(f"| `{arm}` | {c['n']:,} | {c['capture']*100:.2f}% | "
                    f"{c['crash']*100:.2f}% | {c['timeout']*100:.2f}% | {delta} |")
    md = "# Detector coupling bin-wise-bias rerun (seed 431)\n\n"
    md += f"**Frozen verdict: `{verdict}`.**\n\n"
    md += (f"Quality gate: injected std {gate['observed_std_m']:.4f} m vs profiled "
           f"{gate['target_std_m']:.4f} m ({gate['relative_error']*100:+.2f}%), "
           f"±10% gate = **{'PASS' if gate['pass'] else 'FAIL'}**.\n\n")
    md += "| arm | episodes | capture | crash | timeout | Δ capture vs clean |\n"
    md += "|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
    md += ("Interpretation ceiling: an eval-only reproduction supports that v7-shaped output "
           "errors hurt this frozen analytic-trained policy. Causal confirmation that the policy "
           "is specifically coupled requires matched retraining; this experiment cannot supply it.\n")
    (root / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
