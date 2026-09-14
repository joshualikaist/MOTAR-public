#!/usr/bin/env python3
"""Fail-closed summary for the frozen seed-313 corrected non-overlap held-out sweep.

Raw cell JSON/receipts are immutable.  The first completed sweep contains one known auxiliary
metadata error: ``v2_evaluation_contract.target_speed_max_mps`` was hard-coded to 1.5 even though
the measured task condition, launch log, validator and speed strata all attest U[0.3, 1.25].  This
tool records that erratum; it never rewrites the raw evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


EXPECTED_BARS = [70, 85, 100, 115, 130, 145]
EXPECTED_SEED = 313
EXPECTED_CHECKPOINT_SHA256 = (
    "541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha256(repository: Path, commit: str, path: str):
    """SHA-256 of a file's bytes as of `commit`, or None when git cannot produce them.

    Result roots carry an immutable copy of every runtime file, but a snapshot deleted to reclaim
    disk leaves the result unverifiable against a worktree that has since moved on. The recorded
    commit still holds those exact bytes: for the 09-02 held-out sweep this recovers 10 of the 11
    files that had drifted, and the eleventh is the archived evaluator. Verification stays
    cryptographic -- the recovered bytes must hash to the value the manifest recorded.
    """
    if not commit:
        return None
    try:
        blob = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "blob", f"{commit}:{path}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return hashlib.sha256(blob).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def wilson95(successes: int, total: int) -> list[float]:
    require(total > 0 and 0 <= successes <= total, "invalid binomial count")
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - half, center + half]


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isfinite(float(left)) and abs(float(left) - float(right)) <= tolerance


def verify_runtime_sources(root: Path, manifest: dict) -> tuple:
    """(source_hashes, verification counts) for every runtime file the manifest recorded.

    Evaluation was launched from a dirty worktree. Prefer the immutable local snapshot, then the
    byte-identical tracked file, then the recorded commit, then the explicitly archived evaluator
    (the only runtime file corrected after this sweep). Every branch is a SHA-256 equality: the
    question is only where the attested bytes are found, never whether they are checked.

    Separate from summarize() on purpose. Cell logs can be reclaimed for disk space while the
    source attestation stays intact, and that attestation is the part later work depends on.
    """
    source_hashes = {}
    verification = {"snapshot": 0, "current": 0, "git_history": 0, "archived_evaluator": 0}
    repository = Path(__file__).resolve().parents[1]
    commit = manifest.get("git_commit", "")
    # Files the worktree had already modified when the sweep launched: those alone may legitimately
    # need a source other than the recorded commit.
    dirty_at_launch = {line[3:] for line in manifest.get("git_status", []) if len(line) > 3}
    for entry in manifest["runtime_files"]:
        snapshot = root / entry["snapshot"]
        current = repository / entry["path"]
        archived = root / "evaluator_executed.sh"
        if snapshot.is_file() and sha256(snapshot) == entry["sha256"]:
            verification["snapshot"] += 1
        elif current.is_file() and sha256(current) == entry["sha256"]:
            verification["current"] += 1
        elif git_blob_sha256(repository, commit, entry["path"]) == entry["sha256"]:
            verification["git_history"] += 1
        elif (
            entry["path"] == "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
            and archived.is_file()
            and sha256(archived) == entry["sha256"]
        ):
            require(entry["path"] in dirty_at_launch,
                    f"archived evaluator used for a file that was clean at launch: {entry['path']}")
            verification["archived_evaluator"] += 1
        else:
            raise RuntimeError(
                f"runtime source unavailable or drifted: {entry['path']} "
                f"(no snapshot, current worktree differs, and commit {commit[:12]} does not hold it)")
        source_hashes[entry["path"]] = entry["sha256"]
    require(sum(verification.values()) == len(manifest["runtime_files"]),
            "runtime source accounting drift")
    return source_hashes, verification


def summarize(root: Path) -> dict:
    root = root.resolve()
    require(root.is_dir(), f"result root missing: {root}")
    manifest_path = root / "source_manifest.json"
    require(manifest_path.is_file(), "source manifest missing")
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(int(manifest.get("runtime_file_count", -1)) == len(manifest.get("runtime_files", [])),
            "source manifest file accounting drift")

    source_hashes, source_verification = verify_runtime_sources(root, manifest)

    csv_path = root / "results.csv"
    require(csv_path.is_file(), "results.csv missing")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = {int(row["bars"]): row for row in csv.DictReader(stream)}
    require(sorted(csv_rows) == EXPECTED_BARS, "CSV density set drift")

    cells = []
    nonces = set()
    evaluator_hashes = set()
    for bars in EXPECTED_BARS:
        result_path = root / f"{bars}bars.json"
        receipt_path = root / f"{bars}bars.receipt.json"
        log_path = root / f"{bars}bars.log"
        require(result_path.is_file() and receipt_path.is_file() and log_path.is_file(),
                f"cell artifacts missing at {bars} bars")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        condition = result.get("condition") or {}
        outcome = result.get("outcome") or {}
        causes = result.get("crash_causes") or {}

        require(receipt.get("bars") == bars and condition.get("bars") == bars,
                f"bar identity drift at {bars}")
        require(receipt.get("seed") == EXPECTED_SEED and condition.get("seed") == EXPECTED_SEED,
                f"seed drift at {bars}")
        require(receipt.get("source_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
                f"checkpoint receipt drift at {bars}")
        require(result.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
                f"checkpoint result drift at {bars}")
        require(receipt.get("evaluated_checkpoint_snapshot_sha256") == EXPECTED_CHECKPOINT_SHA256,
                f"checkpoint snapshot drift at {bars}")
        require(sha256(result_path) == receipt.get("result_sha256"),
                f"result receipt hash drift at {bars}")
        require(sha256(log_path) == receipt.get("log_sha256"), f"log hash drift at {bars}")
        require(receipt.get("runtime_source_manifest_sha256") == manifest_sha,
                f"source manifest receipt drift at {bars}")
        require(result.get("runtime_source_manifest_sha256") == manifest_sha,
                f"source manifest result drift at {bars}")
        evaluator_hash = receipt.get("evaluator_script_sha256")
        require(source_hashes.get("aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh")
                == evaluator_hash, f"evaluator is not the snapshotted runtime source at {bars}")
        evaluator_hashes.add(evaluator_hash)
        nonce = condition.get("evaluation_nonce")
        require(nonce and nonce == receipt.get("evaluation_nonce"), f"nonce drift at {bars}")
        require(nonce not in nonces, f"reused evaluation nonce at {bars}")
        nonces.add(nonce)

        expected_condition = {
            "action_selection": "deterministic",
            "target_speed_mode": "uniform",
            "target_speed_min_mps": 0.3,
            "target_speed_max_mps": 1.25,
            "target_pattern": "mixed",
            "target_route_mode": "off",
            "target_motion_model": "physx_ref5in_6dof_motor_wrench_v2_same_substep",
            "robot_name": "navrl_ref5in_v2_quad",
            "speed_governor_mode": "off",
            "episode_len_steps": 600,
            "goal_dist_min_m": 6.0,
            "goal_dist_max_m": 28.0,
        }
        for name, expected in expected_condition.items():
            got = condition.get(name)
            matches = _close(got, expected) if isinstance(expected, float) else got == expected
            require(matches, f"measured condition drift at {bars}: {name}={got!r}")
        require(result.get("speed_governor", {}).get("intervention_rate") == 0.0,
                f"governor intervened at {bars}")

        actual = int(result["actual_episodes"])
        counts = [int(outcome[name]) for name in ("captured", "crash", "timeout")]
        require(sum(counts) == actual and actual >= 2049, f"outcome accounting drift at {bars}")
        for rate_name, count in zip(("capture_rate", "crash_rate", "timeout_rate"), counts):
            require(_close(outcome[rate_name], count / actual), f"{rate_name} drift at {bars}")
            require(_close(float(csv_rows[bars][rate_name]), count / actual),
                    f"CSV {rate_name} drift at {bars}")
        require(int(causes.get("count", -1)) == counts[1], f"crash-cause count drift at {bars}")

        auxiliary_speed = (result.get("v2_evaluation_contract") or {}).get(
            "target_speed_max_mps"
        )
        require(_close(auxiliary_speed, 1.5),
                "raw metadata no longer matches the registered 1.5-m/s erratum")
        action = result.get("action") or {}
        cells.append({
            "bars": bars,
            "actual_episodes": actual,
            "captured": counts[0],
            "crash": counts[1],
            "timeout": counts[2],
            "capture_rate": counts[0] / actual,
            "capture_wilson95": wilson95(counts[0], actual),
            "crash_rate": counts[1] / actual,
            "crash_wilson95": wilson95(counts[1], actual),
            "timeout_rate": counts[2] / actual,
            "timeout_wilson95": wilson95(counts[2], actual),
            "bar_contact": int(causes.get("bar_contact", 0)),
            "bar_contact_rate": int(causes.get("bar_contact", 0)) / actual,
            "out_of_bounds": int(causes.get("out_of_bounds", 0)),
            "lateral_mean_abs": float((action.get("mean_abs") or [0, 0])[1]),
            "lateral_positive_rate": float(action.get("positive_y_rate", 0.0)),
            "lateral_edge98_rate": float((action.get("executed_edge98_rate") or [0, 0])[1]),
            "metadata_erratum_observed_speed_max_mps": auxiliary_speed,
            "measured_speed_max_mps": condition["target_speed_max_mps"],
            "result_sha256": sha256(result_path),
            "receipt_sha256": sha256(receipt_path),
        })

    require(len(evaluator_hashes) == 1, "evaluator changed between held-out cells")
    first, last = cells[0], cells[-1]
    density_steps = (last["bars"] - first["bars"]) / 15.0
    return {
        "schema": "motar_corrected_nonoverlap_physical_off_heldout_v1",
        "status": "COMPLETE_VALID_WITH_METADATA_ERRATUM",
        "claim_boundary": (
            "one incomplete route-off seed-911 policy at trained densities 70-145; "
            "no 205/routed/hardware/sim-to-real claim"
        ),
        "seed": EXPECTED_SEED,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_manifest_sha256": manifest_sha,
        "runtime_git_commit": manifest.get("git_commit"),
        "runtime_git_dirty": bool(manifest.get("git_dirty")),
        "runtime_source_snapshot_verified": True,
        "runtime_source_verification": source_verification,
        "runtime_source_file_count": len(manifest["runtime_files"]),
        "evaluator_script_sha256": next(iter(evaluator_hashes)),
        "metadata_erratum": {
            "field": "v2_evaluation_contract.target_speed_max_mps",
            "recorded": 1.5,
            "correct": 1.25,
            "outcome_affecting": False,
            "evidence": (
                "condition.target_speed_max_mps, launch log, pre-write validator and speed strata "
                "all attest 1.25; only the redundant auxiliary serializer was hard-coded"
            ),
            "raw_evidence_modified": False,
        },
        "cells": cells,
        "density_trend": {
            "capture_delta_70_to_145_pp": 100.0 * (last["capture_rate"] - first["capture_rate"]),
            "crash_delta_70_to_145_pp": 100.0 * (last["crash_rate"] - first["crash_rate"]),
            "mean_capture_delta_per_15_bars_pp": (
                100.0 * (last["capture_rate"] - first["capture_rate"]) / density_steps
            ),
            "mean_crash_delta_per_15_bars_pp": (
                100.0 * (last["crash_rate"] - first["crash_rate"]) / density_steps
            ),
        },
        "interpretation": {
            "timeout_is_primary_bottleneck": False,
            "bar_contact_is_primary_failure": True,
            "persistent_lateral_action_chirality": True,
            "resume_authorized": False,
            "second_curriculum_authorized": False,
            "routed_ppo_authorized": False,
        },
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Corrected non-overlap physical route-off held-out result",
        "",
        f"Status: **{summary['status']}**",
        "",
        "| bars | n | capture (Wilson 95%) | crash | timeout | bar contact |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lo, hi = cell["capture_wilson95"]
        lines.append(
            f"| {cell['bars']} | {cell['actual_episodes']} | {100*cell['capture_rate']:.2f}% "
            f"[{100*lo:.2f}, {100*hi:.2f}] | {100*cell['crash_rate']:.2f}% | "
            f"{100*cell['timeout_rate']:.2f}% | {100*cell['bar_contact_rate']:.2f}% |"
        )
    trend = summary["density_trend"]
    lines += [
        "",
        f"70→145 capture change: **{trend['capture_delta_70_to_145_pp']:+.2f} pp**; "
        f"mean **{trend['mean_capture_delta_per_15_bars_pp']:+.2f} pp / 15 bars**.",
        "Timeout stays below 0.4%; the density loss is almost entirely bar contact, not timeout.",
        "",
        "## Metadata erratum",
        "",
        "Raw `v2_evaluation_contract.target_speed_max_mps` says 1.5 m/s because of a redundant "
        "serializer constant. The measured condition, log, runtime validator and speed strata all "
        "prove `U[0.3,1.25] m/s`. Raw artifacts were not edited; the serializer is fixed for future runs.",
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"] + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--sources-only", action="store_true",
                        help="verify the runtime source attestation and stop; works on a root whose "
                             "cell logs were reclaimed for disk space")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.sources_only:
        root = args.result_root.resolve()
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
        _, verification = verify_runtime_sources(root, manifest)
        print(json.dumps({"root": str(root), "runtime_files": len(manifest["runtime_files"]),
                          "verified_by": verification}, indent=2, sort_keys=True))
        return
    summary = summarize(args.result_root)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(summary["status"])
    print(json.dumps(summary["density_trend"], indent=2))


if __name__ == "__main__":
    main()
