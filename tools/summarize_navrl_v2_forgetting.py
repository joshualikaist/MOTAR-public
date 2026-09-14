#!/usr/bin/env python3
"""Validate the ep19100 versus ep24000 205-bar forgetting evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CHECKPOINTS = {
    19100: "82d1eeac1798b4b465274551ec2363fb377be781c0a58e930577ddb822f55044",
    24000: "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f",
}
CONDITIONS = {
    "uniform": ("uniform", None),
    "fast1p5": ("fixed", 1.5),
}


def difference_ci(new_success: int, new_total: int, old_success: int, old_total: int) -> tuple[float, float, float]:
    new = new_success / new_total
    old = old_success / old_total
    delta = new - old
    se = math.sqrt(new * (1.0 - new) / new_total + old * (1.0 - old) / old_total)
    radius = 1.959963984540054 * se
    return delta, delta - radius, delta + radius


def load_cell(root: Path, epoch: int, label: str) -> dict:
    path = root / f"ep{epoch}_{label}" / "205bars.json"
    if not path.is_file():
        raise SystemExit(f"missing forgetting result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    condition = payload.get("condition") or {}
    contract = payload.get("v2_evaluation_contract") or {}
    expected_mode, expected_speed = CONDITIONS[label]
    checks = {
        "checkpoint_sha256": payload.get("checkpoint_sha256") == CHECKPOINTS[epoch],
        "bars": int(condition.get("bars", -1)) == 205,
        "seed": int(condition.get("seed", -1)) == 42,
        "action": condition.get("action_selection") == "deterministic",
        "reflection": condition.get("reflection_mode") == "original",
        "speed_mode": condition.get("target_speed_mode") == expected_mode,
        "contract_speed_mode": contract.get("target_speed_distribution") == expected_mode,
        "pattern": condition.get("target_pattern") == "mixed",
        "full_goal_distribution": condition.get("full_goal_distribution") is True,
        "requested_episodes": int(payload.get("requested_episodes", -1)) == 2049,
        "actual_episodes": int(payload.get("actual_episodes", -1)) >= 2049,
    }
    if expected_speed is not None:
        for field in ("target_speed_mps", "target_speed_min_mps", "target_speed_max_mps"):
            try:
                checks[field] = abs(float(condition.get(field)) - expected_speed) <= 1e-6
            except (TypeError, ValueError):
                checks[field] = False
    else:
        checks["speed_min"] = abs(float(condition.get("target_speed_min_mps", -1)) - 0.3) <= 1e-6
        checks["speed_max"] = abs(float(condition.get("target_speed_max_mps", -1)) - 1.5) <= 1e-6
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"invalid {path}: {', '.join(failed)}")
    actual = int(payload["actual_episodes"])
    outcome = payload.get("outcome") or {}
    counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
    if any(value < 0 for value in counts) or sum(counts) != actual:
        raise SystemExit(f"invalid outcome accounting in {path}")
    return payload


def row(epoch: int, label: str, payload: dict) -> dict:
    actual = int(payload["actual_episodes"])
    outcome = payload["outcome"]
    causes = payload.get("crash_causes") or {}
    action = payload.get("action") or {}
    return {
        "epoch": epoch,
        "condition": label,
        "episodes": actual,
        "captured": int(outcome["captured"]),
        "capture_rate": int(outcome["captured"]) / actual,
        "crash_rate": int(outcome["crash"]) / actual,
        "timeout_rate": int(outcome["timeout"]) / actual,
        "bar_contact_rate": int(causes.get("bar_contact", 0)) / actual,
        "lateral_executed_edge98_rate": float(action["executed_edge98_rate"][1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    rows = {
        (epoch, label): row(epoch, label, load_cell(root, epoch, label))
        for epoch in CHECKPOINTS
        for label in CONDITIONS
    }

    comparisons = {}
    material_forgetting = False
    for label in CONDITIONS:
        old = rows[(19100, label)]
        new = rows[(24000, label)]
        delta, low, high = difference_ci(
            new["captured"], new["episodes"], old["captured"], old["episodes"]
        )
        verdict = "forgetting" if delta <= -0.03 and high < 0.0 else (
            "improvement" if delta >= 0.03 and low > 0.0 else "no-material-change"
        )
        material_forgetting |= verdict == "forgetting"
        comparisons[label] = {
            "ep24000_minus_ep19100_capture": delta,
            "capture_difference_ci95": [low, high],
            "crash_difference": new["crash_rate"] - old["crash_rate"],
            "bar_contact_difference": new["bar_contact_rate"] - old["bar_contact_rate"],
            "verdict": verdict,
            "rule": "forgetting if delta <= -0.03 and the independent 95% CI upper bound < 0",
        }

    payload = {
        "schema_version": 1,
        "experiment": "navrl_v2_ep19100_vs_ep24000_forgetting",
        "condition": {
            "bars": 205,
            "seed": 42,
            "requested_episodes_per_cell": 2049,
            "action_selection": "deterministic",
            "reflection_mode": "original",
        },
        "checkpoints": CHECKPOINTS,
        "cells": [rows[key] for key in sorted(rows)],
        "comparisons": comparisons,
        "material_forgetting_detected": material_forgetting,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# NavRL v2 ep19100 versus ep24000 forgetting evaluation",
        "",
        "Frozen policies, 205 bars, seed 42, deterministic/original inference, 2,049 requested "
        "episodes per independent cell.",
        "",
        "| condition | epoch | capture | crash | timeout | bar contact | lateral edge98 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in CONDITIONS:
        for epoch in CHECKPOINTS:
            item = rows[(epoch, label)]
            lines.append(
                f"| {label} | {epoch} | {item['capture_rate']:.2%} | {item['crash_rate']:.2%} | "
                f"{item['timeout_rate']:.2%} | {item['bar_contact_rate']:.2%} | "
                f"{item['lateral_executed_edge98_rate']:.2%} |"
            )
    lines.append("")
    for label, comparison in comparisons.items():
        low, high = comparison["capture_difference_ci95"]
        lines.append(
            f"- {label}: ep24000−ep19100 capture "
            f"**{comparison['ep24000_minus_ep19100_capture']:+.2%}** "
            f"(95% CI {low:+.2%}..{high:+.2%}); **{comparison['verdict']}**."
        )
    lines.extend(
        [
            "",
            f"Material forgetting detected: **{'YES' if material_forgetting else 'NO'}**.",
            "",
            "This is inference-only. Async resets make cells independent rather than episode-paired.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[forgetting] PASS | summary={root / 'summary.md'}")


if __name__ == "__main__":
    main()
