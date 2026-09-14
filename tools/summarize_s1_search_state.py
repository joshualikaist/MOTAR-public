#!/usr/bin/env python3
"""Validate and summarize the preregistered S1 search-state held-out cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("off", "geofence", "coverage", "belief")
DENSITIES = (70, 145)
SEED = 331
EPISODES = 2049
PRIMARY_MIN = 0.02
CRASH_GUARD = 0.02
TIMEOUT_GUARD = 0.01
MASKED_LOSS_FRACTION = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_ci(a_success, a_total, b_success, b_total):
    """Unpooled two-proportion normal CI for rate(a)-rate(b)."""
    pa, pb = a_success / a_total, b_success / b_total
    delta = pa - pb
    se = math.sqrt(pa * (1.0 - pa) / a_total + pb * (1.0 - pb) / b_total)
    radius = 1.959963984540054 * se
    return {"difference": delta, "ci95": [delta - radius, delta + radius]}


def wilson(successes, total):
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return [centre - radius, centre + radius]


def load_cell(root: Path, arm: str, bars: int, *, masked=False):
    directory = root / (arm + ("_masked" if masked else ""))
    result_path = directory / f"{bars}bars.json"
    receipt_path = directory / f"{bars}bars.receipt.json"
    if not result_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"missing result/receipt for {arm} masked={masked} bars={bars}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    condition = result.get("condition") or {}
    expected = {
        "seed": SEED,
        "bars": bars,
        "action_selection": "deterministic",
        "search_state": arm,
        "search_state_masked": masked,
        "search_state_telemetry": True,
    }
    mismatch = {key: (condition.get(key), value) for key, value in expected.items() if condition.get(key) != value}
    if mismatch:
        raise ValueError(f"condition mismatch in {result_path}: {mismatch}")
    if int(result.get("requested_episodes", -1)) != EPISODES:
        raise ValueError(f"requested episode mismatch in {result_path}")
    n = int(result.get("actual_episodes", -1))
    outcome = result.get("outcome") or {}
    counts = {key: int(outcome.get(key, -1)) for key in ("captured", "crash", "timeout")}
    if n < EPISODES or min(counts.values()) < 0 or sum(counts.values()) != n:
        raise ValueError(f"outcome accounting failure in {result_path}")
    if receipt.get("result_sha256") != sha256(result_path):
        raise ValueError(f"receipt digest mismatch for {result_path}")
    if receipt.get("runtime_git_dirty") is not False:
        raise ValueError(f"dirty runtime source in {receipt_path}")
    receipt_expected = {
        "seed": SEED,
        "bars": bars,
        "search_state": arm,
        "search_state_force_invalid": masked,
        "search_state_telemetry": True,
    }
    receipt_mismatch = {
        key: (receipt.get(key), value)
        for key, value in receipt_expected.items()
        if receipt.get(key) != value
    }
    if receipt_mismatch:
        raise ValueError(f"receipt condition mismatch in {receipt_path}: {receipt_mismatch}")

    acquisition = ((result.get("target_motion") or {}).get("first_acquisition") or {})
    if set(acquisition) < {"capture", "crash", "timeout"}:
        raise ValueError(f"first-acquisition telemetry missing in {result_path}")
    never = sum(int(acquisition[label]["never_acquired"]) for label in ("capture", "crash", "timeout"))
    crash_never = int(acquisition["crash"]["never_acquired"])
    acquired = n - never
    crash_acquired = counts["crash"] - crash_never
    if acquired < 0 or crash_acquired < 0:
        raise ValueError(f"invalid acquired-episode accounting in {result_path}")
    first_visible = acquisition["capture"].get("first_visible_step_median")
    search = result.get("search_state") or {}
    return {
        "arm": arm,
        "masked": masked,
        "bars": bars,
        "episodes": n,
        **counts,
        "capture_rate": counts["captured"] / n,
        "capture_ci95": wilson(counts["captured"], n),
        "crash_rate": counts["crash"] / n,
        "crash_ci95": wilson(counts["crash"], n),
        "timeout_rate": counts["timeout"] / n,
        "timeout_ci95": wilson(counts["timeout"], n),
        "never_acquired": never,
        "never_acquired_rate": never / n,
        "acquired": acquired,
        "crash_given_acquired": crash_acquired / acquired if acquired else None,
        "captured_first_visible_step_median": first_visible,
        "visible_fraction": {
            label: ((result["target_motion"]["outcome_telemetry"][label]).get("visible_fraction_step_weighted"))
            for label in ("capture", "crash", "timeout")
        },
        "blind_phase_mean_speed_mps": search.get("blind_phase_mean_speed_mps"),
        "blind_phase_bar_clearance_mean_m": search.get("blind_phase_bar_clearance_mean_m"),
        "first_visible_search_state": search.get("first_visible"),
        "result": str(result_path.resolve()),
        "result_sha256": sha256(result_path),
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256(receipt_path),
    }


def build_summary(root: Path):
    cells = {
        arm: {str(bars): load_cell(root, arm, bars) for bars in DENSITIES}
        for arm in ARMS
    }
    baseline = cells["off"]["70"]
    decisions = {}
    for arm in ARMS[1:]:
        cell = cells[arm]["70"]
        primary = difference_ci(
            baseline["never_acquired"], baseline["episodes"],
            cell["never_acquired"], cell["episodes"],
        )
        primary_pass = primary["difference"] >= PRIMARY_MIN and primary["ci95"][0] > 0.0
        crash_rise = (
            cell["crash_given_acquired"] - baseline["crash_given_acquired"]
            if cell["crash_given_acquired"] is not None and baseline["crash_given_acquired"] is not None
            else math.inf
        )
        timeout_rise = cell["timeout_rate"] - baseline["timeout_rate"]
        median_guard = (
            cell["captured_first_visible_step_median"] is not None
            and baseline["captured_first_visible_step_median"] is not None
            and cell["captured_first_visible_step_median"]
            <= baseline["captured_first_visible_step_median"]
        )
        guard_pass = crash_rise <= CRASH_GUARD and timeout_rise <= TIMEOUT_GUARD and median_guard
        decisions[arm] = {
            "never_acquired_control_minus_arm": primary,
            "primary_pass": primary_pass,
            "crash_given_acquired_rise": crash_rise,
            "timeout_rate_rise": timeout_rise,
            "captured_first_visible_median_nonincrease": median_guard,
            "guard_pass": guard_pass,
            "outcome": "FAIL",
        }

    masked = {}
    for arm in ARMS[1:]:
        directory = root / f"{arm}_masked"
        if not directory.exists():
            continue
        masked[arm] = {str(bars): load_cell(root, arm, bars, masked=True) for bars in DENSITIES}
        decision = decisions[arm]
        active = cells[arm]["70"]
        masked_cell = masked[arm]["70"]
        gain = decision["never_acquired_control_minus_arm"]["difference"]
        masked_loss = masked_cell["never_acquired_rate"] - active["never_acquired_rate"]
        mechanism_pass = gain > 0.0 and masked_loss >= MASKED_LOSS_FRACTION * gain
        decision["masked_loss"] = masked_loss
        decision["masked_loss_fraction_of_gain"] = masked_loss / gain if gain > 0.0 else None
        decision["mechanism_pass"] = mechanism_pass
        if decision["primary_pass"] and decision["guard_pass"]:
            decision["outcome"] = "PASS_MECHANISM" if mechanism_pass else "PASS_MECHANISM_UNRESOLVED"

    pending_mask = [
        arm for arm, decision in decisions.items()
        if decision["primary_pass"] and decision["guard_pass"] and arm not in masked
    ]
    if pending_mask:
        raise ValueError(
            "primary-passing arm(s) require preregistered masked cells: "
            + ", ".join(pending_mask)
        )

    a3_increment = cells["coverage"]["70"]["never_acquired_rate"] - cells["belief"]["70"]["never_acquired_rate"]
    return {
        "schema_version": 1,
        "experiment": "s1_explicit_blind_search_state",
        "condition": {
            "training_seed": 919,
            "heldout_seed": SEED,
            "training_bars": 70,
            "densities": list(DENSITIES),
            "requested_episodes_per_cell": EPISODES,
            "primary_density": 70,
            "secondary_density": 145,
        },
        "gates": {
            "primary_min_absolute_reduction": PRIMARY_MIN,
            "primary_ci_must_exclude_zero": True,
            "crash_given_acquired_rise_max": CRASH_GUARD,
            "timeout_rise_max": TIMEOUT_GUARD,
            "captured_first_visible_median_must_not_increase": True,
            "masked_loss_fraction_of_gain_min": MASKED_LOSS_FRACTION,
        },
        "cells": cells,
        "masked_cells": masked,
        "decisions": decisions,
        "belief_increment_over_coverage": {
            "never_acquired_reduction": a3_increment,
            "contributes_at_least_1pp": a3_increment >= 0.01,
        },
        "claim_boundary": [
            "70-bar same-budget fresh-arm comparison is primary",
            "145 bars is secondary OOD evidence",
            "no 205-bar, routed, hardware, sim-to-real, or adversarial-evader claim",
        ],
    }


def write_outputs(root: Path, payload):
    json_path = root / "summary.json"
    md_path = root / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# S1 explicit blind-search state",
        "",
        "| arm | bars | n | capture | crash | timeout | never acquired | crash / acquired | captured first-visible p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for bars in DENSITIES:
            cell = payload["cells"][arm][str(bars)]
            lines.append(
                f"| {arm} | {bars} | {cell['episodes']} | {cell['capture_rate']:.2%} | "
                f"{cell['crash_rate']:.2%} | {cell['timeout_rate']:.2%} | "
                f"{cell['never_acquired_rate']:.2%} | {cell['crash_given_acquired']:.2%} | "
                f"{cell['captured_first_visible_step_median']} |"
            )
    lines += ["", "## Preregistered decisions", ""]
    for arm, decision in payload["decisions"].items():
        primary = decision["never_acquired_control_minus_arm"]
        lines.append(
            f"- {arm}: **{decision['outcome']}**; never-acquired reduction "
            f"{primary['difference']:.2%}, 95% CI [{primary['ci95'][0]:.2%}, {primary['ci95'][1]:.2%}]."
        )
    lines += ["", "145 bars is secondary OOD evidence; it does not decide the gate."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest_path = root / "summary.sha256"
    digest_path.write_text(
        f"{sha256(json_path)}  {json_path.name}\n{sha256(md_path)}  {md_path.name}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_root",
        nargs="?",
        type=Path,
        default=Path("results/navrl_s1_search_state_seed331"),
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    payload = build_summary(root)
    write_outputs(root, payload)
    print(root / "summary.json")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[s1-summary] FAIL: {exc}")
