"""Validate and summarize the R3 detector/perception robustness screen."""

import argparse
import hashlib
import json
import math
from pathlib import Path


CELLS = (
    ("analytic_clean", {"detector": "bootstrap", "perturb": False}),
    ("dropout_0p3", {"detector": "bootstrap", "axis": "dropout", "value": 0.3}),
    ("latency_0p1s", {"detector": "bootstrap", "axis": "latency_s", "value": 0.1}),
    ("latency_0p2s", {"detector": "bootstrap", "axis": "latency_s", "value": 0.2}),
    ("range_error_0p15m", {"detector": "bootstrap", "axis": "range_error_m", "value": 0.15}),
    ("range_error_0p30m", {"detector": "bootstrap", "axis": "range_error_m", "value": 0.30}),
    ("learned_clean", {"detector": "learned", "perturb": False}),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diff_ci(a, b, field):
    count = {"capture_rate": "captured", "crash_rate": "crashes", "timeout_rate": "timeouts"}[
        field
    ]
    pa, pb = a[field], b[field]
    se = math.sqrt(pa * (1 - pa) / a["episodes"] + pb * (1 - pb) / b["episodes"])
    return pa - pb, [pa - pb - 1.96 * se, pa - pb + 1.96 * se]


def delta_with_ci(result):
    delta, interval = result
    return f"{delta * 100:+.2f} pp (95% CI {interval[0] * 100:+.2f}..{interval[1] * 100:+.2f})"


def load(root, tag, *, policy_sha, learned_sha=None):
    path = root / tag / "205bars.json"
    receipt_path = root / tag / "205bars.receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    condition = payload.get("condition") or {}
    contract = payload.get("v2_evaluation_contract") or {}
    outcome = payload.get("outcome") or {}
    governor = payload.get("speed_governor") or {}
    n = int(payload.get("actual_episodes", -1))
    detector_sha = contract.get("detector_checkpoint_sha256") or ""
    checks = {
        "schema": payload.get("schema_version") == 1,
        "episodes": int(payload.get("requested_episodes", -1)) == 2049 and n >= 2049,
        "policy_sha": payload.get("checkpoint_sha256") == policy_sha
        and receipt.get("source_checkpoint_sha256") == policy_sha,
        "condition": int(condition.get("bars", -1)) == 205
        and int(condition.get("seed", -1)) == 47
        and condition.get("action_selection") == "deterministic"
        and condition.get("reflection_mode") == "original"
        and condition.get("target_speed_mode") == "uniform",
        "runtime": contract.get("runtime_profile") == "main"
        and contract.get("speed_governor_mode") == "riskcap",
        "sensor_only": governor.get("sensor_only") is True
        and governor.get("direction_preserved") is True,
    }
    if tag == "learned_clean":
        checks["learned_detector"] = detector_sha == learned_sha
    else:
        checks["bootstrap_detector"] = detector_sha in ("", None)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid cell {tag}: {', '.join(failed)}")
    return {
        "tag": tag,
        "episodes": n,
        "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]),
        "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": float(
            (payload.get("crash_causes") or {}).get("bar_contact_share", 0.0)
        ),
        "mean_speed_mps": float(
            ((payload.get("action") or {}).get("motion") or {}).get("mean_speed_mps", 0.0)
        ),
        "perception_perturb": bool(contract.get("perception_perturb")),
        "detection_dropout": float(contract.get("detection_dropout", 0.0)),
        "detection_latency_s": float(contract.get("detection_latency_s", 0.0)),
        "range_error_m": float(contract.get("range_error_m", 0.0)),
        "detector_checkpoint_sha256": detector_sha or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--learned-detector-sha", required=True)
    args = parser.parse_args()

    rows = {}
    for tag, _meta in CELLS:
        rows[tag] = load(
            args.root,
            tag,
            policy_sha=args.policy_sha,
            learned_sha=args.learned_detector_sha if tag == "learned_clean" else None,
        )

    baseline = rows["analytic_clean"]
    learned = rows["learned_clean"]
    comparisons = {}
    for tag, row in rows.items():
        if tag == "analytic_clean":
            continue
        comparisons[tag] = {
            "capture_delta_pp": delta_with_ci(diff_ci(row, baseline, "capture_rate")),
            "crash_delta_pp": delta_with_ci(diff_ci(row, baseline, "crash_rate")),
            "timeout_delta_pp": delta_with_ci(diff_ci(row, baseline, "timeout_rate")),
        }
    comparisons["learned_vs_analytic"] = {
        "capture_delta_pp": delta_with_ci(diff_ci(learned, baseline, "capture_rate")),
        "crash_delta_pp": delta_with_ci(diff_ci(learned, baseline, "crash_rate")),
        "timeout_delta_pp": delta_with_ci(diff_ci(learned, baseline, "timeout_rate")),
    }

    summary = {
        "experiment": "navrl_v2_detector_robustness",
        "policy_sha256": args.policy_sha,
        "learned_detector_sha256": args.learned_detector_sha,
        "seed": 47,
        "bars": 205,
        "speed_governor": "riskcap",
        "baseline": baseline,
        "cells": rows,
        "comparisons_vs_analytic_clean": comparisons,
    }
    json_path = args.root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# NavRL v2 detector/perception robustness (R3 screen)",
        "",
        f"Policy SHA-256: `{args.policy_sha}`",
        f"Learned detector SHA-256: `{args.learned_detector_sha}`",
        "",
        "Held-out seed47 · 205 bars · U[0.3,1.5] m/s · deterministic · riskcap.",
        "",
        "| cell | episodes | capture | crash | timeout | bar contact | mean speed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, _meta in CELLS:
        row = rows[tag]
        lines.append(
            f"| {tag} | {row['episodes']} | {row['capture_rate'] * 100:.2f}% | "
            f"{row['crash_rate'] * 100:.2f}% | {row['timeout_rate'] * 100:.2f}% | "
            f"{row['bar_contact_rate'] * 100:.2f}% | {row['mean_speed_mps']:.3f} m/s |"
        )
    lines.extend(
        [
            "",
            "## Delta vs analytic_clean",
            "",
        ]
    )
    for tag, comp in comparisons.items():
        lines.append(
            f"- **{tag}**: capture {comp['capture_delta_pp']}, crash {comp['crash_delta_pp']}, "
            f"timeout {comp['timeout_delta_pp']}"
        )
    md_path = args.root / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
