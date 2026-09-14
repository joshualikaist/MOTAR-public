#!/usr/bin/env python3
"""Summarize the frozen ep24000 reflection and second-seed evaluations."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - half, center + half]


def diff_ci(a_successes, a_total, b_successes, b_total, z=1.959963984540054):
    pa = a_successes / a_total
    pb = b_successes / b_total
    se = math.sqrt(pa * (1.0 - pa) / a_total + pb * (1.0 - pb) / b_total)
    diff = pb - pa
    return diff, [diff - z * se, diff + z * se]


def outcome_record(payload):
    outcome = payload["outcome"]
    n = int(payload["actual_episodes"])
    record = {"episodes": n}
    for count_name, rate_name in (
        ("captured", "capture_rate"),
        ("crash", "crash_rate"),
        ("timeout", "timeout_rate"),
    ):
        count = int(outcome[count_name])
        record[count_name] = count
        record[rate_name] = count / n
        record[rate_name + "_wilson95"] = wilson(count, n)
    record["bar_contact_rate"] = int(payload["crash_causes"]["bar_contact"]) / n
    return record


def pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed42", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    original_raw = load(root / "mirror_original" / "205bars.json")
    conjugate_raw = load(root / "mirror_conjugate" / "205bars.json")
    seed42_raw = load(args.seed42)
    seed43_raw = load(root / "seed43" / "205bars.json")
    bearing_raw = load(root / "bearing_original" / "205bars.json")
    action_pair = load(root / "mirror_original" / "reflection_actions.json")

    hashes = {
        item.get("checkpoint_sha256")
        for item in (original_raw, conjugate_raw, seed42_raw, seed43_raw, bearing_raw)
    }
    if len(hashes) != 1:
        raise SystemExit("checkpoint SHA differs across causal evaluation inputs")
    if original_raw["condition"].get("reflection_mode") != "original":
        raise SystemExit("original arm is mislabeled")
    if conjugate_raw["condition"].get("reflection_mode") != "conjugate":
        raise SystemExit("conjugate arm is mislabeled")
    if seed43_raw["condition"].get("seed") != 43:
        raise SystemExit("second-seed arm did not use seed 43")

    original = outcome_record(original_raw)
    conjugate = outcome_record(conjugate_raw)
    seed42 = outcome_record(seed42_raw)
    seed43 = outcome_record(seed43_raw)
    bearing_replay = outcome_record(bearing_raw)
    for name in ("episodes", "captured", "crash", "timeout"):
        if bearing_replay[name] != original[name]:
            raise SystemExit(
                "bearing instrumentation changed deterministic outcome %s: %s != %s"
                % (name, bearing_replay[name], original[name])
            )

    mirror_diffs = {}
    material = False
    for count_name in ("captured", "crash"):
        diff, ci = diff_ci(
            original[count_name], original["episodes"],
            conjugate[count_name], conjugate["episodes"],
        )
        name = "capture" if count_name == "captured" else "crash"
        mirror_diffs[name + "_conjugate_minus_original"] = diff
        mirror_diffs[name + "_diff95"] = ci
        if abs(diff) >= 0.02 and (ci[0] > 0.0 or ci[1] < 0.0):
            material = True

    seed_diff, seed_diff_ci = diff_ci(
        seed42["captured"], seed42["episodes"], seed43["captured"], seed43["episodes"]
    )
    ci_overlap = not (
        seed42["capture_rate_wilson95"][1] < seed43["capture_rate_wilson95"][0]
        or seed43["capture_rate_wilson95"][1] < seed42["capture_rate_wilson95"][0]
    )
    practical_replication = abs(seed_diff) <= 0.03 and ci_overlap

    bearing = bearing_raw.get("strata", {}).get("initial_target_bearing", {})
    negative = bearing.get("negative_y") or {}
    positive = bearing.get("positive_y") or {}
    if int(negative.get("episodes", 0)) <= 0 or int(positive.get("episodes", 0)) <= 0:
        raise SystemExit("initial target-bearing strata are missing")
    bearing_diff, bearing_diff_ci = diff_ci(
        int(negative["captured"]), int(negative["episodes"]),
        int(positive["captured"]), int(positive["episodes"]),
    )

    summary = {
        "schema_version": 1,
        "checkpoint_sha256": hashes.pop(),
        "preregistered_plan": "RESEARCH_PLAN.md §8.8",
        "training_performed": False,
        "mirror": {
            "original": original,
            "conjugate": conjugate,
            "differences": mirror_diffs,
            "material_chirality": material,
            "action_pair": action_pair,
            "initial_target_bearing": {
                "negative_y": negative,
                "positive_y": positive,
                "positive_minus_negative_capture": bearing_diff,
                "capture_diff95": bearing_diff_ci,
            },
            "interpretation": (
                "material chirality" if material else "no material aggregate chirality by preregistered rule"
            ),
            "pairing_note": (
                "actions are exact observation pairs; rollout outcomes are common-seed independent aggregates "
                "because asynchronous resets break episode pairing"
            ),
        },
        "second_seed": {
            "seed42": seed42,
            "seed43": seed43,
            "capture_seed43_minus_seed42": seed_diff,
            "capture_diff95": seed_diff_ci,
            "wilson95_overlap": ci_overlap,
            "practical_replication": practical_replication,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    action_mae = action_pair["mean_abs_error"]
    lines = [
        "# ep24000 causal checks 1--3",
        "",
        f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
        "- Training performed: **no** (frozen-weight inference only)",
        "- Preregistered plan: `RESEARCH_PLAN.md §8.8`",
        "",
        "## Mirror audit (seed 42, 205 bars)",
        "",
        "| arm | episodes | capture | crash | timeout | bar contact / episodes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| original pi | {original['episodes']} | {pct(original['capture_rate'])} | {pct(original['crash_rate'])} | {pct(original['timeout_rate'])} | {pct(original['bar_contact_rate'])} |",
        f"| conjugate M pi M | {conjugate['episodes']} | {pct(conjugate['capture_rate'])} | {pct(conjugate['crash_rate'])} | {pct(conjugate['timeout_rate'])} | {pct(conjugate['bar_contact_rate'])} |",
        "",
        f"Capture difference (conjugate-original): **{100*mirror_diffs['capture_conjugate_minus_original']:+.2f} pp** "
        f"(95% CI {100*mirror_diffs['capture_diff95'][0]:+.2f} to {100*mirror_diffs['capture_diff95'][1]:+.2f} pp).",
        f"Crash difference: **{100*mirror_diffs['crash_conjugate_minus_original']:+.2f} pp** "
        f"(95% CI {100*mirror_diffs['crash_diff95'][0]:+.2f} to {100*mirror_diffs['crash_diff95'][1]:+.2f} pp).",
        f"Preregistered aggregate outcome-chirality verdict: **{'YES' if material else 'NO'}**.",
        "",
        f"Exact action-pair MAE [x,y,z,yaw]: `{[round(x, 6) for x in action_mae]}`; "
        f"lateral sign mismatch: **{pct(action_pair['lateral_sign_mismatch_rate'])}** "
        f"over {action_pair['lateral_sign_comparable']} comparable samples.",
        "",
        "Outcome arms share a seed but are not episode-paired after asynchronous resets; only the action comparison is exactly paired.",
        "",
        f"Initial target-bearing outcome: negative-y {negative['captured']}/{negative['episodes']} "
        f"({pct(negative['capture_rate'])}) versus positive-y {positive['captured']}/{positive['episodes']} "
        f"({pct(positive['capture_rate'])}); positive-negative **{100*bearing_diff:+.2f} pp** "
        f"(95% CI {100*bearing_diff_ci[0]:+.2f} to {100*bearing_diff_ci[1]:+.2f} pp).",
        "",
        "## Independent seed replication (205 bars, original pi)",
        "",
        "| seed | episodes | capture | crash | timeout | bar contact / episodes |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 42 | {seed42['episodes']} | {pct(seed42['capture_rate'])} | {pct(seed42['crash_rate'])} | {pct(seed42['timeout_rate'])} | {pct(seed42['bar_contact_rate'])} |",
        f"| 43 | {seed43['episodes']} | {pct(seed43['capture_rate'])} | {pct(seed43['crash_rate'])} | {pct(seed43['timeout_rate'])} | {pct(seed43['bar_contact_rate'])} |",
        "",
        f"Capture difference (seed43-seed42): **{100*seed_diff:+.2f} pp** "
        f"(95% CI {100*seed_diff_ci[0]:+.2f} to {100*seed_diff_ci[1]:+.2f} pp).",
        f"Practical replication (<=3 pp and Wilson intervals overlap): **{'PASS' if practical_replication else 'FAIL'}**.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
