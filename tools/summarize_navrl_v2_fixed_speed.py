#!/usr/bin/env python3
"""Validate and summarize the frozen ep24000 fixed-target-speed evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_SHA256 = "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
SPEEDS = (0.3, 0.9, 1.5)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def difference_ci(a_success: int, a_total: int, b_success: int, b_total: int) -> tuple[float, float, float]:
    """Unpaired Wald interval for rate(a)-rate(b); cells use independent rollouts."""
    a = a_success / a_total
    b = b_success / b_total
    delta = a - b
    standard_error = math.sqrt(a * (1.0 - a) / a_total + b * (1.0 - b) / b_total)
    radius = 1.959963984540054 * standard_error
    return delta, delta - radius, delta + radius


def speed_dir(speed: float) -> str:
    return f"speed_{speed:g}".replace(".", "p")


def load_cell(root: Path, speed: float) -> dict:
    path = root / speed_dir(speed) / "205bars.json"
    if not path.is_file():
        raise SystemExit(f"missing fixed-speed result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    condition = payload.get("condition") or {}
    contract = payload.get("v2_evaluation_contract") or {}
    outcome = payload.get("outcome") or {}

    checks = {
        "checkpoint_sha256": payload.get("checkpoint_sha256") == EXPECTED_SHA256,
        "bars": int(condition.get("bars", -1)) == 205,
        "seed": int(condition.get("seed", -1)) == 42,
        "action": condition.get("action_selection") == "deterministic",
        "reflection": condition.get("reflection_mode") == "original",
        "speed_mode": condition.get("target_speed_mode") == "fixed",
        "contract_speed_mode": contract.get("target_speed_distribution") == "fixed",
        "pattern": condition.get("target_pattern") == "mixed",
        "full_goal_distribution": condition.get("full_goal_distribution") is True,
        "requested_episodes": int(payload.get("requested_episodes", -1)) == 2049,
        "actual_episodes": int(payload.get("actual_episodes", -1)) >= 2049,
    }
    for field in ("target_speed_mps", "target_speed_min_mps", "target_speed_max_mps"):
        try:
            checks[field] = abs(float(condition.get(field)) - speed) <= 1e-6
        except (TypeError, ValueError):
            checks[field] = False
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"invalid {path}: {', '.join(failed)}")

    actual = int(payload["actual_episodes"])
    counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
    if any(value < 0 for value in counts) or sum(counts) != actual:
        raise SystemExit(f"invalid outcome accounting in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()

    cells = {speed: load_cell(root, speed) for speed in SPEEDS}
    rows = []
    for speed, payload in cells.items():
        actual = int(payload["actual_episodes"])
        outcome = payload["outcome"]
        causes = payload.get("crash_causes") or {}
        captured = int(outcome["captured"])
        low, high = wilson(captured, actual)
        rows.append(
            {
                "target_speed_mps": speed,
                "episodes": actual,
                "captured": captured,
                "capture_rate": captured / actual,
                "capture_ci95": [low, high],
                "crash_rate": int(outcome["crash"]) / actual,
                "timeout_rate": int(outcome["timeout"]) / actual,
                "bar_contact_rate": int(causes.get("bar_contact", 0)) / actual,
                "lateral_executed_edge98_rate": float(payload["action"]["executed_edge98_rate"][1]),
                "mean_flight_speed_mps": float(payload["action"]["motion"]["mean_speed_mps"]),
                "mean_command_speed_mps": float(payload["action"]["motion"]["mean_command_speed_mps"]),
            }
        )

    low_speed = rows[0]
    high_speed = rows[-1]
    delta, ci_low, ci_high = difference_ci(
        high_speed["captured"],
        high_speed["episodes"],
        low_speed["captured"],
        low_speed["episodes"],
    )
    monotonic_nonincreasing = all(
        rows[index + 1]["capture_rate"] <= rows[index]["capture_rate"]
        for index in range(len(rows) - 1)
    )
    material = delta <= -0.03 and ci_high < 0.0

    summary = {
        "schema_version": 1,
        "experiment": "navrl_v2_ep24000_fixed_target_speed",
        "checkpoint_sha256": EXPECTED_SHA256,
        "condition": {
            "bars": 205,
            "seed": 42,
            "requested_episodes_per_cell": 2049,
            "action_selection": "deterministic",
            "reflection_mode": "original",
            "target_pattern": "mixed",
        },
        "cells": rows,
        "high_minus_low_capture": {
            "delta": delta,
            "ci95": [ci_low, ci_high],
            "material_speed_sensitivity": material,
            "rule": "delta <= -0.03 and the independent 95% CI upper bound < 0",
        },
        "high_minus_low_rates": {
            "crash": high_speed["crash_rate"] - low_speed["crash_rate"],
            "timeout": high_speed["timeout_rate"] - low_speed["timeout_rate"],
            "bar_contact": high_speed["bar_contact_rate"] - low_speed["bar_contact_rate"],
            "lateral_executed_edge98": (
                high_speed["lateral_executed_edge98_rate"]
                - low_speed["lateral_executed_edge98_rate"]
            ),
        },
        "capture_monotonic_nonincreasing": monotonic_nonincreasing,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# NavRL v2 ep24000 fixed-target-speed evaluation",
        "",
        "Frozen policy, 205 bars, seed 42, deterministic/original inference, mixed motion, "
        "2,049 requested episodes per independent cell.",
        "",
        "| speed (m/s) | episodes | capture (95% CI) | crash | timeout | bar contact |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target_speed_mps:.1f} | {episodes:,} | {capture_rate:.2%} "
            "({capture_ci95[0]:.2%}..{capture_ci95[1]:.2%}) | {crash_rate:.2%} | "
            "{timeout_rate:.2%} | {bar_contact_rate:.2%} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"High-minus-low capture: **{delta:+.2%}** (95% CI {ci_low:+.2%}..{ci_high:+.2%}).",
            "High-minus-low crash/bar-contact/timeout: "
            f"**{high_speed['crash_rate'] - low_speed['crash_rate']:+.2%} / "
            f"{high_speed['bar_contact_rate'] - low_speed['bar_contact_rate']:+.2%} / "
            f"{high_speed['timeout_rate'] - low_speed['timeout_rate']:+.2%}**.",
            "Lateral executed-edge98 changes from "
            f"**{low_speed['lateral_executed_edge98_rate']:.2%}** to "
            f"**{high_speed['lateral_executed_edge98_rate']:.2%}**; mean flight speed changes only "
            f"{low_speed['mean_flight_speed_mps']:.3f}→{high_speed['mean_flight_speed_mps']:.3f} m/s "
            f"while mean command speed is already {low_speed['mean_command_speed_mps']:.3f}→"
            f"{high_speed['mean_command_speed_mps']:.3f} m/s.",
            f"Pre-registered material speed sensitivity: **{'YES' if material else 'NO'}**.",
            f"Capture monotonically non-increasing: **{'YES' if monotonic_nonincreasing else 'NO'}**.",
            "",
            "This is an inference-only causal slice; it does not update the optimizer, running "
            "statistics, or checkpoint.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fixed-speed] PASS | summary={root / 'summary.md'}")


if __name__ == "__main__":
    main()
