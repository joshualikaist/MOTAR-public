#!/usr/bin/env python3
"""Aggregate a fixed-100 corridor evaluation and adjudicate the pre-registered pilot gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


BASELINE_CAPTURE = 0.6453
BASELINE_BAR_CONTACT = 0.3318
MIN_CAPTURE = 0.68
MIN_DELTA = 0.03


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"evaluation CSV is empty: {path}")
    required = {
        "bars",
        "target_speed",
        "pursuer_limit",
        "episodes",
        "capture_rate",
        "bar_contact_rate",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("evaluation CSV lacks columns: " + ", ".join(sorted(missing)))
    return rows


def evaluate(path: Path) -> Dict[str, Any]:
    rows = _load_rows(path)
    speeds = sorted(float(row["target_speed"]) for row in rows)
    if len(rows) != 4 or speeds != [0.0, 0.5, 1.0, 1.5]:
        raise ValueError(
            "corridor gate requires exactly the fixed-100 four-speed grid "
            "(target 0.0, 0.5, 1.0, 1.5)"
        )
    if any(int(float(row["bars"])) != 100 for row in rows):
        raise ValueError("corridor gate requires bars=100 for every cell")
    if any(abs(float(row["pursuer_limit"]) - 2.5) > 1e-6 for row in rows):
        raise ValueError("corridor gate requires pursuer_limit=2.5 for every cell")

    episodes = sum(int(row["episodes"]) for row in rows)
    captures = sum(
        int(row["episodes"]) * float(row["capture_rate"]) for row in rows
    )
    bar_contacts = sum(
        int(row["episodes"]) * float(row["bar_contact_rate"]) for row in rows
    )
    capture = captures / episodes
    bar_contact = bar_contacts / episodes
    delta = capture - BASELINE_CAPTURE

    # Independent-proportion normal interval versus the immutable ep12500 baseline. The baseline
    # evaluation had 4003 episodes; this interval is diagnostic, while the three preregistered
    # scalar gates below determine PASS/FAIL.
    baseline_n = 4003
    se = math.sqrt(
        capture * (1.0 - capture) / episodes
        + BASELINE_CAPTURE * (1.0 - BASELINE_CAPTURE) / baseline_n
    )
    ci = [delta - 1.96 * se, delta + 1.96 * se]
    checks = {
        "capture_at_least_68pct": capture >= MIN_CAPTURE,
        "gain_at_least_3pp": delta >= MIN_DELTA,
        "bar_contact_below_baseline": bar_contact < BASELINE_BAR_CONTACT,
    }
    return {
        "schema_version": 1,
        "source": str(path),
        "episodes": episodes,
        "target_speeds": speeds,
        "capture_rate": capture,
        "bar_contact_rate": bar_contact,
        "baseline": {
            "checkpoint": "ep12500",
            "capture_rate": BASELINE_CAPTURE,
            "bar_contact_rate": BASELINE_BAR_CONTACT,
            "episodes": baseline_n,
        },
        "capture_delta": delta,
        "capture_delta_95ci": ci,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit 3 when the pilot gate fails (default: report FAIL but exit successfully)",
    )
    args = parser.parse_args()
    try:
        result = evaluate(args.results_csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    lo, hi = result["capture_delta_95ci"]
    print(
        "[corridor-gate] "
        f"capture={100 * result['capture_rate']:.2f}% "
        f"delta={100 * result['capture_delta']:+.2f}pp "
        f"95%CI=[{100 * lo:+.2f},{100 * hi:+.2f}]pp "
        f"bar_contact={100 * result['bar_contact_rate']:.2f}% "
        f"verdict={result['verdict']}"
    )
    for name, passed in result["checks"].items():
        print(f"[corridor-gate] {'PASS' if passed else 'FAIL'} | {name}")
    return 3 if args.strict and result["verdict"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
