#!/usr/bin/env python3
"""Run the preregistered D8-B frozen-policy observation sensitivity campaign.

This launcher never trains. It reuses the canonical D9 default-detector/N=0 evaluation
environment, changes only the preregistered seed and D8 treatment factors, and delegates each
2,049-episode cell to the generic evaluator. Run from a committed clean tree:

  python -B tools/run_dynamic_mesh_policy_sensitivity_d8b.py preflight
  python -B tools/run_dynamic_mesh_policy_sensitivity_d8b.py evaluate [cell]
  python -B tools/run_dynamic_mesh_policy_sensitivity_d8b.py finalize
  python -B tools/run_dynamic_mesh_policy_sensitivity_d8b.py verify
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
CHECKPOINT = ROOT / (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
PREREG = ROOT / "docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md"
D8A = ROOT / "results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/receipt.json"
D8A_SHA = "92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee"
V3 = ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf"
V3_SHA = "c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0"
OUTPUT = ROOT / "results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
SUMMARY = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "README.md"
PYTHON = Path("/home/fair/miniconda3/envs/aerialgym/bin/python")
NINJA = PYTHON.with_name("ninja")

SEEDS = (593, 599, 601)
ARMS = ("analytic_flat", "mesh_flat", "mesh_shaded")
ENV_MODES = {"analytic_flat": "off", "mesh_flat": "mesh_flat", "mesh_shaded": "mesh_shaded"}
CELLS = tuple((f"s{seed}_{arm}", seed, arm) for seed in SEEDS for arm in ARMS)
EPISODES = 2049
BARS = 70
MARGIN_PP = -3.0
T_CRITICAL_DF2_95 = 4.302652729911275
EXTRA_RUNTIME_ROOTS = (
    "resources/models/environment_assets/objects",
    "tools/renderer_validation",
)
SOURCE_FILES = (
    "aerial_gym/task/navrl_task/navrl_task.py",
    "aerial_gym/task/navrl_task/navrl_detector.py",
    "aerial_gym/task/navrl_task/navrl_dynamic_mesh_treatment.py",
    "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh",
    "tools/run_dynamic_mesh_policy_sensitivity_d8b.py",
    "tools/run_navrl_distractor_envelope.py",
    "docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md",
    "results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/receipt.json",
    "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf",
)


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D9 = load_module("d8b_d9_environment", ROOT / "tools/run_navrl_distractor_envelope.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def cell_factors(name):
    for cell, seed, arm in CELLS:
        if cell == name:
            return seed, arm
    raise ContractError(f"unknown D8-B cell {name!r}; expected {[row[0] for row in CELLS]}")


def cell_dir(name):
    return OUTPUT / "cells" / name


def cell_paths(name):
    directory = cell_dir(name)
    return {
        "result": directory / f"{BARS}bars.json",
        "receipt": directory / f"{BARS}bars.receipt.json",
        "log": directory / f"{BARS}bars.log",
        "snapshot": directory / "checkpoint_snapshot.pth",
        "manifest": directory / "source_manifest.json",
    }


def launcher_contract():
    require(PYTHON.is_file(), f"canonical Python missing: {PYTHON}")
    require(NINJA.is_file() and os.access(str(NINJA), os.X_OK), f"matching ninja missing: {NINJA}")
    return {
        "python": str(PYTHON.resolve()),
        "ninja": str(NINJA.resolve()),
        "ninja_version": subprocess.check_output([str(NINJA), "--version"], text=True).strip(),
        "ninja_sha256": sha256_file(NINJA),
    }


def evaluation_env(name, preflight):
    seed, arm = cell_factors(name)
    # Reuse the already audited default-detector, zero-distractor environment. D8-B adds exactly
    # its output paths, fresh seed, extra source attestation roots, and one treatment factor.
    env = D9.evaluation_env("default_n0", preflight=preflight, force=False)
    launcher = launcher_contract()
    env.update({
        "PATH": str(PYTHON.parent) + os.pathsep + env.get("PATH", ""),
        "NAVRL_NINJA": launcher["ninja"],
        "PYTHONPATH": str(ROOT),
        "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
        "NAVRL_SEED": str(seed),
        "NAVRL_V2_DENSITIES": str(BARS),
        "NAVRL_V2_RESULT_DIR": str(cell_dir(name)),
        "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
        "NAVRL_DYNAMIC_MESH_TREATMENT": ENV_MODES[arm],
        "NAVRL_EVAL_EXTRA_RUNTIME_ROOTS": ",".join(EXTRA_RUNTIME_ROOTS),
        "NAVRL_DISTRACTOR_COUNT": "0",
        "NAVRL_DETECTOR_CHECKPOINT": "",
        "NAVRL_DETECTOR_THRESHOLD": "0.55",
        "NAVRL_P9_ERROR_MODEL": "",
        "NAVRL_P9_EXPECTED_SHA256": "",
    })
    require("NAVRL_V2_FORCE" not in env, f"{name}: blanket evaluator force is forbidden")
    require(env["NAVRL_DYNAMIC_MESH_TREATMENT"] == ENV_MODES[arm], f"{name}: treatment drift")
    require(env["NAVRL_DISTRACTOR_COUNT"] == "0", f"{name}: distractor drift")
    require(env["NAVRL_DETECTOR_CHECKPOINT"] == "", f"{name}: detector artifact drift")
    require(env["NAVRL_DETECT_WIDTH"] == env["NAVRL_CAMERA_WIDTH"] == "160", f"{name}: width drift")
    require(env["NAVRL_DETECT_HEIGHT"] == env["NAVRL_CAMERA_HEIGHT"] == "90", f"{name}: height drift")
    return env


def prereg_commit():
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(PREREG.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def preflight(require_output_absent=True):
    require(CHECKPOINT.is_file() and sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "checkpoint drift")
    require(V3.is_file() and sha256_file(V3) == V3_SHA, "v3 asset drift")
    require(D8A.is_file() and sha256_file(D8A) == D8A_SHA, "D8-A receipt drift")
    d8a = load_json(D8A)
    require(d8a.get("decision", {}).get("verdict") == "TECHNICAL_GO", "D8-A was not GO")
    if require_output_absent:
        require(not OUTPUT.exists(), f"D8-B output already exists: {OUTPUT}")
    runtime_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         "aerial_gym", "resources/models/environment_assets/objects", "tools/renderer_validation",
         "tools/run_dynamic_mesh_policy_sensitivity_d8b.py"], cwd=ROOT, text=True,
    ).strip()
    require(not runtime_status, f"D8-B runtime source is dirty: {runtime_status}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen = prereg_commit()
    require(frozen and frozen != head, "D8-B preregistration must precede implementation")
    require(subprocess.run(["git", "merge-base", "--is-ancestor", frozen, head], cwd=ROOT).returncode == 0,
            "D8-B preregistration is not an ancestor of runtime HEAD")
    for name, _, _ in CELLS:
        env = evaluation_env(name, preflight=True)
        require(env.get("NAVRL_PREFLIGHT_ONLY") == "1", f"{name}: preflight flag missing")
    return {
        "runtime_git_commit": head,
        "runtime_source_clean": True,
        "preregistration_commit": frozen,
        "d8a_receipt_sha256": D8A_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "v3_asset_sha256": V3_SHA,
        "launcher": launcher_contract(),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
    }


def run_evaluator(name, preflight_only=False):
    command = ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)]
    return subprocess.run(command, cwd=ROOT, env=evaluation_env(name, preflight_only), check=False)


def manifest_map(path):
    manifest = load_json(path.resolve())
    entries = manifest.get("runtime_files") or []
    require(manifest.get("schema_version") == 2 and entries, "source manifest malformed")
    require(manifest.get("git_dirty") is False, "runtime source manifest is dirty")
    roots = tuple(manifest.get("runtime_roots") or ())
    require(roots == ("aerial_gym", "resources/robots", *EXTRA_RUNTIME_ROOTS),
            f"runtime roots drift: {roots}")
    mapping = {}
    for entry in entries:
        relative = str(entry.get("path", ""))
        snapshot = (path.resolve().parent / str(entry.get("snapshot", ""))).resolve()
        require(relative and relative not in mapping and snapshot.is_file(), f"bad manifest entry: {relative}")
        require(sha256_file(snapshot) == entry.get("sha256"), f"snapshot hash drift: {relative}")
        mapping[relative] = (entry.get("sha256"), int(entry.get("size_bytes", -1)))
    for required in (
        "aerial_gym/task/navrl_task/navrl_dynamic_mesh_treatment.py",
        "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf",
        "tools/renderer_validation/urdf_asset.py",
    ):
        require(required in mapping, f"runtime manifest omitted {required}")
    return manifest, mapping


def pooled_visibility(result):
    rows = result["target_motion"]["outcome_telemetry"].values()
    visible = sum(int(row["visible_steps"]) for row in rows)
    observed = sum(int(row["observation_steps"]) for row in rows)
    return visible / observed if observed else None


def pooled_never_acquired(result):
    rows = result["target_motion"]["first_acquisition"].values()
    never = sum(int(row["never_acquired"]) for row in rows)
    episodes = sum(int(row["episodes"]) for row in rows)
    return never / episodes if episodes else None


def verify_cell(name):
    seed, arm = cell_factors(name)
    paths = cell_paths(name)
    for path in paths.values():
        require(path.exists(), f"{name}: missing {path.name}")
    result, receipt = load_json(paths["result"]), load_json(paths["receipt"])
    require(sha256_file(paths["result"]) == receipt.get("result_sha256"), f"{name}: result hash drift")
    require(sha256_file(paths["snapshot"]) == CHECKPOINT_SHA, f"{name}: checkpoint snapshot drift")
    require(receipt.get("source_checkpoint_sha256") == CHECKPOINT_SHA, f"{name}: source checkpoint drift")
    require(receipt.get("evaluated_checkpoint_snapshot_sha256") == CHECKPOINT_SHA,
            f"{name}: evaluated checkpoint drift")
    require(receipt.get("runtime_git_dirty") is False, f"{name}: runtime was dirty")
    require(int(receipt.get("seed", -1)) == seed and int(receipt.get("bars", -1)) == BARS,
            f"{name}: receipt condition drift")
    actual = int(result.get("actual_episodes", -1))
    requested = int(result.get("requested_episodes", -1))
    outcome = result.get("outcome") or {}
    require(requested == EPISODES and actual >= EPISODES, f"{name}: episode contract failed")
    require(sum(int(outcome.get(key, -1)) for key in ("captured", "crash", "timeout")) == actual,
            f"{name}: outcome accounting failed")
    condition = result.get("condition") or {}
    expected_mode = ENV_MODES[arm]
    require(int(condition.get("seed", -1)) == seed and int(condition.get("bars", -1)) == BARS,
            f"{name}: result seed/bars drift")
    require(int(condition.get("distractor_count", -1)) == 0, f"{name}: distractors not zero")
    require(condition.get("target_render_mode") == ("analytic_flat" if expected_mode == "off" else expected_mode),
            f"{name}: effective renderer mode drift")
    attached = bool(condition.get("d8_dynamic_mesh_treatment"))
    require(attached == (expected_mode != "off"), f"{name}: treatment attachment drift")
    if attached:
        require(condition.get("d8_dynamic_mesh_asset_sha256") == V3_SHA, f"{name}: asset attestation drift")
        contract = condition.get("d8_dynamic_mesh_contract") or {}
        require(contract.get("mode") == expected_mode and int(contract.get("triangles", 0)) > 0,
                f"{name}: treatment contract missing")
    else:
        require(condition.get("d8_dynamic_mesh_asset_sha256") == "", f"{name}: analytic asset leakage")
        require(condition.get("d8_dynamic_mesh_contract") is None, f"{name}: analytic contract leakage")
    manifest, mapping = manifest_map(paths["manifest"])
    return {
        "cell": name,
        "seed": seed,
        "arm": arm,
        "requested_episodes": requested,
        "actual_episodes": actual,
        "captured": int(outcome["captured"]),
        "crash": int(outcome["crash"]),
        "timeout": int(outcome["timeout"]),
        "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]),
        "timeout_rate": float(outcome["timeout_rate"]),
        "never_acquired_rate": pooled_never_acquired(result),
        "target_visible_fraction": pooled_visibility(result),
        "action_mean_abs": result["action"]["mean_abs"],
        "motion_mean_speed_mps": result["action"]["motion"]["mean_speed_mps"],
        "outcome_steps": result["speed_governor"]["outcome_steps"],
        "result_sha256": sha256_file(paths["result"]),
        "receipt_sha256": sha256_file(paths["receipt"]),
        "log_sha256": sha256_file(paths["log"]),
        "source_manifest_sha256": sha256_file(paths["manifest"].resolve()),
        "source_manifest_git_commit": manifest["git_commit"],
        "runtime_file_count": len(mapping),
        "effective_condition": {
            "target_render_mode": condition["target_render_mode"],
            "d8_dynamic_mesh_treatment": attached,
            "d8_dynamic_mesh_asset_sha256": condition["d8_dynamic_mesh_asset_sha256"],
            "d8_dynamic_mesh_contract": condition["d8_dynamic_mesh_contract"],
        },
    }


def paired_t(values):
    require(len(values) == len(SEEDS), "paired interval requires exactly three seed differences")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_DF2_95 * sd / math.sqrt(len(values))
    return {"values": values, "mean": mean, "ci95": [mean - half, mean + half],
            "unit": "paired evaluation seed", "df": 2, "t_critical": T_CRITICAL_DF2_95}


def contrast(rows, arm, field):
    by_key = {(row["seed"], row["arm"]): row for row in rows}
    values = [100.0 * (by_key[(seed, arm)][field] - by_key[(seed, "analytic_flat")][field])
              for seed in SEEDS]
    return paired_t(values)


def classify_primary(primary, integrity):
    if not all(integrity.values()):
        return "INVALID_D8B_EXECUTION"
    if primary["ci95"][1] < MARGIN_PP:
        return "MATERIAL_LOSS"
    if primary["ci95"][0] > MARGIN_PP:
        return "NO_MATERIAL_LOSS_WITHIN_MARGIN"
    return "INCONCLUSIVE_POLICY_SENSITIVITY"


def build_summary(provenance):
    rows = [verify_cell(name) for name, _, _ in CELLS]
    manifest_shas = {row["source_manifest_sha256"] for row in rows}
    manifest_commits = {row["source_manifest_git_commit"] for row in rows}
    integrity = {
        "all_nine_cells": len(rows) == 9,
        "single_source_manifest": len(manifest_shas) == 1,
        "single_runtime_commit": manifest_commits == {provenance["runtime_git_commit"]},
        "checkpoint_pinned": True,
        "episode_accounting": True,
        "effective_treatments_attested": True,
        "no_training": True,
    }
    primary = contrast(rows, "mesh_shaded", "capture_rate")
    verdict = classify_primary(primary, integrity)
    secondary = {
        f"{arm}_minus_analytic_{field}": contrast(rows, arm, field)
        for arm in ("mesh_flat", "mesh_shaded")
        for field in ("capture_rate", "crash_rate", "timeout_rate", "never_acquired_rate",
                      "target_visible_fraction")
    }
    return {
        "schema": "dynamic_mesh_policy_sensitivity_d8b_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "D8-B",
        "verdict": verdict,
        "interpretation": (
            "Frozen-policy sensitivity in one checkpoint lineage only; no detector improvement, "
            "shortcut reduction, adaptation, training, identity, real-flight, or deployment claim."
        ),
        "preregistration": str(PREREG.relative_to(ROOT)),
        "provenance": provenance,
        "design": {"arms": list(ARMS), "seeds": list(SEEDS), "episodes_per_cell": EPISODES,
                   "cells": [row[0] for row in CELLS], "policy_frozen": True,
                   "detector": "built-in AppearanceTargetSegmenter", "detector_threshold": 0.55,
                   "distractors": 0, "training_started": False},
        "integrity": integrity,
        "primary": {"name": "mesh_shaded_minus_analytic_capture_rate_pp",
                    "margin_pp": MARGIN_PP, **primary},
        "secondary": secondary,
        "cells": rows,
    }


def summary_markdown(payload):
    primary = payload["primary"]
    lines = [
        "# D8-B — frozen-policy sensitivity to mesh-derived observation",
        "",
        f"## Verdict: **`{payload['verdict']}`**",
        "",
        "The verdict uses only the preregistered mesh-shaded minus analytic capture contrast.",
        f"Mean: **{primary['mean']:+.3f} pp**, seed-level 95% t CI "
        f"[{primary['ci95'][0]:+.3f}, {primary['ci95'][1]:+.3f}] pp; margin {MARGIN_PP:+.1f} pp.",
        "",
        "| seed | analytic capture | mesh-flat capture | mesh-shaded capture |",
        "|---:|---:|---:|---:|",
    ]
    by_key = {(row["seed"], row["arm"]): row for row in payload["cells"]}
    for seed in SEEDS:
        lines.append(
            f"| {seed} | {100*by_key[(seed,'analytic_flat')]['capture_rate']:.2f}% | "
            f"{100*by_key[(seed,'mesh_flat')]['capture_rate']:.2f}% | "
            f"{100*by_key[(seed,'mesh_shaded')]['capture_rate']:.2f}% |"
        )
    lines += [
        "", "All nine cells use the same frozen policy/checkpoint and built-in detector. No PPO or "
        "adaptation ran. The seed is the uncertainty unit; episodes/frames are not independent "
        "replicates for the reported CI.",
        "", "This result does not measure color-shortcut reduction because D8-B contains no "
        "distractors. D9 remains a separate preregistered audit. See `summary.json` and cell "
        "receipts for full secondary metrics and provenance.", "",
    ]
    return "\n".join(lines)


def run_preflight():
    provenance = preflight(require_output_absent=True)
    # The generic evaluator provenance check should accept both the baseline and an intervention;
    # it does not instantiate the simulator, so this consumes no result cell.
    for name in ("s593_analytic_flat", "s593_mesh_shaded"):
        completed = run_evaluator(name, preflight_only=True)
        require(completed.returncode == 0, f"generic evaluator preflight failed for {name}")
    print(json.dumps({"status": "PREFLIGHT_PASS", "provenance": provenance}, indent=2))


def run_evaluate(target=None):
    require(OUTPUT.exists() or target is None, "single-cell evaluation requires initialized output")
    names = [target] if target else [row[0] for row in CELLS]
    if not OUTPUT.exists():
        preflight(require_output_absent=True)
    else:
        preflight(require_output_absent=False)
    for index, name in enumerate(names, 1):
        cell_factors(name)
        if cell_dir(name).exists():
            print(f"[D8-B] {name}: exists, verifying and skipping", flush=True)
            verify_cell(name)
            continue
        print(f"[D8-B] cell {index}/{len(names)} {name}: {EPISODES} episodes", flush=True)
        completed = run_evaluator(name, preflight_only=False)
        require(completed.returncode == 0, f"{name}: evaluator failed with {completed.returncode}")
        verify_cell(name)
        print(f"[D8-B] {name}: VERIFIED", flush=True)
    pending = [name for name, _, _ in CELLS if not cell_dir(name).exists()]
    print(f"[D8-B] evaluation pass complete; pending={pending}")


def run_finalize():
    require(OUTPUT.is_dir() and not SUMMARY.exists() and not SUMMARY_MD.exists(),
            "output missing or summary already exists")
    provenance = preflight(require_output_absent=False)
    payload = build_summary(provenance)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(summary_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "primary": payload["primary"]}, indent=2))


def run_verify():
    require(SUMMARY.is_file() and SUMMARY_MD.is_file(), "D8-B summary is incomplete")
    recorded = load_json(SUMMARY)
    expected = build_summary(recorded["provenance"])
    recorded_without_time = {k: v for k, v in recorded.items() if k != "created_at_utc"}
    expected_without_time = {k: v for k, v in expected.items() if k != "created_at_utc"}
    require(recorded_without_time == expected_without_time, "D8-B summary does not recompute")
    require(SUMMARY_MD.read_text(encoding="utf-8") == summary_markdown(recorded),
            "D8-B README does not recompute")
    print(f"[D8-B] VERIFY PASS: {recorded['verdict']}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    require(command in ("preflight", "evaluate", "finalize", "verify"),
            "usage: run_dynamic_mesh_policy_sensitivity_d8b.py preflight|evaluate [cell]|finalize|verify")
    if command == "preflight":
        require(len(sys.argv) == 2, "preflight takes no cell")
        return run_preflight()
    if command == "evaluate":
        require(len(sys.argv) <= 3, "evaluate accepts at most one cell")
        return run_evaluate(sys.argv[2] if len(sys.argv) == 3 else None)
    require(len(sys.argv) == 2, f"{command} takes no cell")
    return run_finalize() if command == "finalize" else run_verify()


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"[D8-B] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
