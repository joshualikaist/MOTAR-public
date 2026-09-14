#!/usr/bin/env python3
"""Read-only archival checks; stdlib only, no simulator/runner import or execution.

Before finalization: python -B <this file> --integrity-only --local
After finalization:  python -B <this file> [--local]
The default validates Git blobs instead of requiring redundant source snapshots.
--local additionally hashes the original checkpoint/source copies. Nothing is written.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (593, 599, 601)
ARMS = ("analytic_flat", "mesh_flat", "mesh_shaded")
CELLS = tuple("s%d_%s" % (seed, arm) for seed in SEEDS for arm in ARMS)
RUNTIME = "1e0eed8b963ed3ccdae93cc2e9ea67d9f9ddbaab"
PREREG_COMMIT = "0aba9816d789c37e7d0ef8874e0a66d3d501299b"
PREREG = "docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md"
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
D8A_SHA = "92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee"
V3_SHA = "c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0"


def check(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def blob(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=str(ROOT))


def check_grid(names):
    check(len(names) == len(CELLS) and set(names) == set(CELLS), "missing/extra/duplicate cell")


def same_except(rows, excluded, label):
    values = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
    check(all(value == values[0] for value in values), label + " changed outside registered factors")


def check_counts(result):
    check(type(result["actual_episodes"]) is int and result["actual_episodes"] >= 2049,
          "actual episode count invalid")
    check(result["requested_episodes"] == 2049, "requested episodes changed")
    outcome = result["outcome"]
    counts = [outcome[key] for key in ("captured", "crash", "timeout")]
    check(all(type(n) is int and n >= 0 for n in counts), "invalid outcome count")
    check(sum(counts) == result["actual_episodes"], "outcome accounting mismatch")
    for count, field in zip(counts, ("capture_rate", "crash_rate", "timeout_rate")):
        check(math.isclose(outcome[field], count / result["actual_episodes"], abs_tol=1e-15,
                           rel_tol=0), "rate not derived from counts: " + field)


def audit(local=False, integrity_only=False):
    d8a = ROOT / "results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/receipt.json"
    check(digest(d8a) == D8A_SHA, "D8-A receipt drift")
    check(read(d8a)["decision"]["verdict"] == "TECHNICAL_GO", "D8-A gate not satisfied")
    check(blob(PREREG_COMMIT, PREREG) == (ROOT / PREREG).read_bytes(), "preregistration drift")
    subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, RUNTIME],
                   cwd=str(ROOT), check=True)
    bundle = HERE / "source_bundle"
    manifest_path = bundle / "source_manifest.json"
    manifest = read(manifest_path)
    check(manifest["git_commit"] == RUNTIME and manifest["git_dirty"] is False,
          "runtime manifest identity/cleanliness mismatch")
    check(manifest["repository_git_dirty"] is False, "repository was dirty")
    check(manifest["runtime_roots"] == ["aerial_gym", "resources/robots",
          "resources/models/environment_assets/objects", "tools/renderer_validation"], "roots drift")
    entries = manifest["runtime_files"]
    check(len(entries) == manifest["runtime_file_count"] == 370, "source file count mismatch")
    check(len({e["path"] for e in entries}) == len(entries), "duplicate source path")
    for entry in entries:
        raw = blob(RUNTIME, entry["path"])
        check(len(raw) == entry["size_bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
              "Git source mismatch: " + entry["path"])
        if local:
            check((bundle / entry["snapshot"]).read_bytes() == raw, "local source snapshot drift")
    environment = bundle / "python_environment.txt"
    check(digest(environment) == manifest["python_environment_sha256"], "environment hash drift")
    env_text = environment.read_text()
    check("python_executable=/home/fair/miniconda3/envs/aerialgym/bin/python3.8\n" in env_text,
          "Python environment mismatch")
    check("\nninja==1.13.0\n" in env_text, "recorded ninja package mismatch")
    directories = sorted(path.name for path in (HERE / "cells").iterdir() if path.is_dir())
    check_grid(directories)
    results, receipts = [], []
    for name in CELLS:
        seed, arm = name.split("_", 1)
        seed = int(seed[1:])
        cell = HERE / "cells" / name
        result_path = cell / "70bars.json"
        result, receipt = read(result_path), read(cell / "70bars.receipt.json")
        check(digest(result_path) == receipt["result_sha256"], name + ": result hash mismatch")
        check(digest(cell / "70bars.log") == receipt["log_sha256"], name + ": log hash mismatch")
        for record in (result, receipt):
            check(record["runtime_git_commit"] == RUNTIME and record["runtime_git_dirty"] is False,
                  name + ": runtime mismatch")
            check(record["runtime_source_manifest_sha256"] == digest(manifest_path), "manifest binding")
            check(record["python_environment_manifest_sha256"] == digest(environment), "env binding")
            check(record["evaluated_checkpoint_snapshot_sha256"] == CHECKPOINT_SHA, "checkpoint binding")
        check(result["checkpoint_sha256"] == receipt["source_checkpoint_sha256"] == CHECKPOINT_SHA,
              "source checkpoint identity")
        check(receipt["evaluated_detector_snapshot_sha256"] == "", "unexpected detector artifact")
        check(hashlib.sha256(blob(RUNTIME, "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"))
              .hexdigest() == result["evaluator_script_sha256"] == receipt["evaluator_script_sha256"],
              "evaluator source binding")
        if local:
            check(digest(cell / "checkpoint_snapshot.pth") == CHECKPOINT_SHA, "checkpoint snapshot drift")
        check_counts(result)
        check(result["actual_episodes"] == receipt["actual_episodes"] and
              result["requested_episodes"] == receipt["requested_episodes"], "receipt episode mismatch")
        c, v = result["condition"], result["v2_evaluation_contract"]
        check(c["seed"] == v["seed"] == receipt["seed"] == seed, "seed binding")
        check(c["evaluation_nonce"] == receipt["evaluation_nonce"], "nonce binding")
        for key, expected in {"bars": 70, "distractor_count": 0, "num_envs": 128,
                              "p9_empirical_error_enabled": False, "target_render_mode": arm}.items():
            check(c[key] == expected, "condition mismatch: " + key)
        for key, expected in {"arena_xy_m": 40.0, "goal_dist_min_m": 22.5, "goal_dist_max_m": 28.0,
                              "target_speed_min_mps": 0.3, "target_speed_max_mps": 1.5,
                              "target_pattern": "mixed", "action_selection": "deterministic",
                              "reflection_mode": "original", "speed_governor_mode": "off",
                              "detector_min_pixels": 2, "target_camera_max_range_m": 20.0,
                              "detector_threshold": 0.55, "detector_checkpoint_sha256": "",
                              "perception_perturb": False, "detection_dropout_active": 0.0,
                              "detection_latency_s": 0.0, "range_error_m": 0.0}.items():
            check(v[key] == expected, "fixed contract mismatch: " + key)
        for key in v:
            if key.startswith("appearance_appearance_") or key in (
                    "camera_mount_rot_deg", "camera_mount_trans_m", "camera_fov_scale_err"):
                check(v[key] == 0.0, "active appearance perturbation: " + key)
        if arm == "analytic_flat":
            check(c["d8_dynamic_mesh_treatment"] is False and c["d8_dynamic_mesh_contract"] is None
                  and c["d8_dynamic_mesh_asset_sha256"] == "", "analytic treatment mismatch")
        else:
            check(c["d8_dynamic_mesh_treatment"] is True and c["d8_dynamic_mesh_asset_sha256"] == V3_SHA,
                  "mesh treatment mismatch")
            contract = c["d8_dynamic_mesh_contract"]
            check(contract["mode"] == arm and contract["triangles"] == 1548 and contract["materials"] == 4
                  and contract["ambient"] == 0.45 and contract["directional"] == 0.55,
                  "mesh constants mismatch")
            check(contract["light_direction"] == [-0.4504281282424927, -0.3002854287624359,
                                                   0.8407991528511047], "light drift")
            check(contract["material_gains"] == [0.5741929411888123, 0.550000011920929,
                                                  0.7868121266365051, 1.0], "material drift")
        log = (cell / "70bars.log").read_text()
        check("actor obs=898, critic states=906" in log, "runtime observation dimensions absent")
        check("py=/home/fair/miniconda3/envs/aerialgym/bin/python " in log, "runtime Python absent")
        results.append(result)
        receipts.append(receipt)
    check(len({r["evaluation_nonce"] for r in receipts}) == 9, "duplicate nonce")
    same_except([r["v2_evaluation_contract"] for r in results], {"seed"}, "evaluation contract")
    same_except([r["condition"] for r in results], {"seed", "evaluation_nonce", "target_render_mode",
                "d8_dynamic_mesh_treatment", "d8_dynamic_mesh_asset_sha256", "d8_dynamic_mesh_contract"},
                "runtime condition")
    same_except(receipts, {"seed", "evaluation_nonce", "started_at_utc", "completed_at_utc",
                "evaluated_checkpoint_snapshot", "result_json", "result_sha256", "log_file", "log_sha256"},
                "receipt")
    report = {"status": "ARCHIVE_INTEGRITY_PASS", "cells": 9, "seeds": list(SEEDS),
              "arms": list(ARMS), "actual_episodes": [r["actual_episodes"] for r in results],
              "source_files": len(entries), "local_snapshots_checked": local,
              "scope": "Artifact and contract checks; manual gate review is recorded in AUDIT.md. "
                       "Does not attest historical ninja binary SHA or dynamic actor noninterference."}
    if not integrity_only:
        summary = read(HERE / "summary.json")
        check_grid([row["cell"] for row in summary["cells"]])
        by_name = dict(zip(CELLS, results))
        def count_rate(name):
            row = by_name[name]
            return row["outcome"]["captured"] / row["actual_episodes"]
        differences = [100 * (count_rate("s%d_mesh_shaded" % seed) -
                              count_rate("s%d_analytic_flat" % seed)) for seed in SEEDS]
        mean = statistics.mean(differences)
        # For df=2, t_.975 has this closed form (independent of runner's hard-coded critical value).
        tcrit = math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2))
        half = tcrit * statistics.stdev(differences) / math.sqrt(3)
        ci = [mean - half, mean + half]
        primary = summary["primary"]
        check(len(primary["values"]) == 3 and len(primary["ci95"]) == 2, "primary shape mismatch")
        check(primary["margin_pp"] == -3.0 and primary["df"] == 2, "analysis rule drift")
        for observed, expected in zip(primary["values"] + [primary["mean"]] + primary["ci95"],
                                      differences + [mean] + ci):
            check(math.isclose(observed, expected, abs_tol=1e-10, rel_tol=0), "primary recompute mismatch")
        verdict = ("MATERIAL_LOSS" if ci[1] < -3.0 else "NO_MATERIAL_LOSS_WITHIN_MARGIN"
                   if ci[0] > -3.0 else "INCONCLUSIVE_POLICY_SENSITIVITY")
        check(summary["verdict"] == verdict and all(summary["integrity"].values()), "verdict mismatch")
        for row in summary["cells"]:
            name = row["cell"]
            raw = by_name[name]
            check(row["actual_episodes"] == raw["actual_episodes"] and
                  row["requested_episodes"] == raw["requested_episodes"], "summary counts mismatch")
            for key in ("captured", "crash", "timeout", "capture_rate", "crash_rate", "timeout_rate"):
                check(row[key] == raw["outcome"][key], "summary cell mismatch: " + key)
            check(row["result_sha256"] == digest(HERE / "cells" / name / "70bars.json"), "summary hash")
        report["independent_primary"] = {"values": differences, "mean": mean, "ci95": ci,
                                          "verdict": verdict, "df": 2}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.local, args.integrity_only), indent=2))
