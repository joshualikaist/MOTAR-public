#!/usr/bin/env python3
"""Run and attest the post-P2 ref5in outcome-strata diagnostic.

This is descriptive follow-up, not a retry of the failed P2 decision cell.  It uses a new seed and
more episodes to separate distance, target-speed and motion-pattern failure composition.  Nothing
in this script can unlock P3 training.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
PYTHON = Path("/home/fair/miniconda3/envs/aerialgym/bin/python")
CHECKPOINT = RL_ROOT / (
    "runs/ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197/nn/"
    "last_gen_ppo_ep_900_rew_137.08087.pth"
)
CHECKPOINT_SHA = "f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf"
P2_ATTESTATION = ROOT / "results/navrl_ref5in_p2_seed313/attestation.json"
OUTPUT = ROOT / "results/navrl_ref5in_outcome_diagnostic_seed317"
CELL = OUTPUT / "ref5in"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
EPISODES = 8193
SEED = 317


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_p2_module():
    spec = importlib.util.spec_from_file_location(
        "ref5in_p2_contract", ROOT / "tools/attest_navrl_ref5in_p2.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def canonical_env(preflight: bool = False) -> dict[str, str]:
    p2 = load_p2_module()
    env = p2.canonical_env(CELL, preflight=preflight)
    env.update(
        {
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_RESULT_DIR": str(CELL),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
        }
    )
    return env


def verify_prerequisites(*, require_clean: bool) -> None:
    require(PYTHON.is_file(), f"canonical Python missing: {PYTHON}")
    require(CHECKPOINT.is_file(), f"P1c checkpoint missing: {CHECKPOINT}")
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "P1c checkpoint changed")
    require(P2_ATTESTATION.is_file(), "P2 attestation missing")
    p2 = load_json(P2_ATTESTATION)
    require(p2.get("verdict") == "FAIL" and p2.get("unlocks") == "none", "P2 is not fail-closed")
    require((p2.get("p1c") or {}).get("checkpoint_sha256") == CHECKPOINT_SHA, "P2/P1c lineage mismatch")
    if require_clean:
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        require(not status.strip(), "tracked source is dirty; commit diagnostic code first")


def run_evaluator(*, preflight: bool = False) -> None:
    command = ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)]
    subprocess.run(command, cwd=ROOT, env=canonical_env(preflight), check=True)


def wilson(
    count: int, total: int, z: float = 1.959963984540054
) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = count / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [center - half, center + half]


def verify_axis(name: str, cells: dict[str, Any], totals: tuple[int, int, int, int]) -> None:
    require(isinstance(cells, dict) and cells, f"missing {name} strata")
    observed = [0, 0, 0, 0]
    cause_totals = {key: 0 for key in ("bar_contact", "below", "above", "out_of_bounds")}
    for label, cell in cells.items():
        episodes = int(cell.get("episodes", -1))
        captured = int(cell.get("successes", -1))
        crash = int(cell.get("crash", -1))
        timeout = int(cell.get("timeout", -1))
        require(min(episodes, captured, crash, timeout) >= 0, f"negative {name}/{label} count")
        require(captured + crash + timeout == episodes, f"incomplete {name}/{label} outcome accounting")
        causes = cell.get("crash_causes") or {}
        require(sum(int(causes.get(key, -1)) for key in cause_totals) == crash,
                f"incomplete {name}/{label} crash-cause accounting")
        for key in cause_totals:
            cause_totals[key] += int(causes[key])
        for index, value in enumerate((episodes, captured, crash, timeout)):
            observed[index] += value
    require(tuple(observed) == totals, f"{name} totals {tuple(observed)} != {totals}")


def verify_result() -> dict[str, Any]:
    result_path = CELL / "70bars.json"
    receipt_path = CELL / "70bars.receipt.json"
    snapshot = CELL / "checkpoint_snapshot.pth"
    for path in (result_path, receipt_path, snapshot, CELL / "70bars.log"):
        require(path.is_file(), f"missing diagnostic artifact: {path}")
    result = load_json(result_path)
    receipt = load_json(receipt_path)
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "source checkpoint changed")
    require(sha256_file(snapshot) == CHECKPOINT_SHA, "evaluated checkpoint snapshot mismatch")
    require(receipt.get("source_checkpoint_sha256") == CHECKPOINT_SHA, "receipt checkpoint mismatch")
    require(sha256_file(result_path) == receipt.get("result_sha256"), "result hash mismatch")
    condition = result.get("condition") or {}
    expected = {
        "seed": SEED,
        "bars": 70,
        "robot_name": "navrl_ref5in_quad",
        "action_selection": "deterministic",
        "reflection_mode": "original",
        "speed_governor_mode": "off",
        "target_speed_mode": "uniform",
        "target_speed_min_mps": 0.3,
        "target_speed_max_mps": 1.5,
        "goal_dist_min_m": 6.0,
        "goal_dist_max_m": 28.0,
    }
    mismatches = {key: (condition.get(key), value) for key, value in expected.items()
                  if condition.get(key) != value}
    require(not mismatches, f"diagnostic condition mismatch: {mismatches}")
    outcome = result.get("outcome") or {}
    counts = tuple(int(outcome.get(key, -1)) for key in ("captured", "crash", "timeout"))
    actual = int(result.get("actual_episodes", -1))
    require(int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
            "episode contract mismatch")
    require(sum(counts) == actual, "global outcome accounting mismatch")
    totals = (actual, *counts)
    strata = result.get("strata") or {}
    for axis in ("distance", "speed", "pattern"):
        verify_axis(axis, strata.get(axis), totals)
    global_causes = result.get("crash_causes") or {}
    require(sum(int(global_causes.get(key, 0)) for key in
                ("bar_contact", "below", "above", "out_of_bounds")) == counts[1],
            "global crash-cause accounting mismatch")
    return result


def enrich_axis(cells: dict[str, Any]) -> dict[str, Any]:
    enriched = {}
    for label, cell in cells.items():
        n = int(cell["episodes"])
        item = dict(cell)
        item["capture_wilson95"] = wilson(int(cell["successes"]), n)
        item["crash_wilson95"] = wilson(int(cell["crash"]), n)
        item["timeout_wilson95"] = wilson(int(cell["timeout"]), n)
        enriched[label] = item
    return enriched


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    strata = result["strata"]
    axes = {name: enrich_axis(strata[name]) for name in ("distance", "speed", "pattern")}
    distance = axes["distance"]
    speed = axes["speed"]
    q0, q3 = distance["q0"], distance["q3"]
    timeout_rates = [float(cell["timeout_rate"]) for cell in speed.values()]
    mechanisms = {
        "distance_q3_minus_q0_capture_pp": 100.0 * (q3["capture_rate"] - q0["capture_rate"]),
        "distance_q3_minus_q0_crash_pp": 100.0 * (q3["crash_rate"] - q0["crash_rate"]),
        "distance_q3_minus_q0_timeout_pp": 100.0 * (q3["timeout_rate"] - q0["timeout_rate"]),
        "speed_timeout_range_pp": 100.0 * (max(timeout_rates) - min(timeout_rates)),
        "long_range_timeout_channel_ge_3pp": q3["timeout_rate"] - q0["timeout_rate"] >= 0.03,
        "long_range_crash_channel_ge_5pp": q3["crash_rate"] - q0["crash_rate"] >= 0.05,
        "speed_timeout_channel_ge_3pp": max(timeout_rates) - min(timeout_rates) >= 0.03,
    }
    payload = {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_outcome_diagnostic.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_p2_descriptive_outcome_strata_seed317",
        "decision_authority": "none",
        "p3_unlocked": False,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "condition": result["condition"],
        "outcome": result["outcome"],
        "strata": axes,
        "mechanism_screen": mechanisms,
        "limitations": [
            "post-P2 diagnostic; cannot revise the seed313 decision",
            "single evaluation seed and one 70-bar density",
            "bins are equal-width input ranges, not randomized treatment arms",
            "associations are descriptive and do not establish causality",
        ],
    }
    return payload


def write_summary(summary: dict[str, Any]) -> None:
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in post-P2 outcome diagnostic",
        "",
        "이 평가는 P2 재시험이 아니라 seed 317의 기술적 진단이다. P3를 해제하지 않는다.",
        "",
        "| distance bin | episodes | capture | crash | timeout | contact | OOB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, cell in summary["strata"]["distance"].items():
        causes = cell["crash_causes"]
        lines.append(
            f"| {label} | {cell['episodes']:,} | {cell['capture_rate']*100:.2f}% | "
            f"{cell['crash_rate']*100:.2f}% | {cell['timeout_rate']*100:.2f}% | "
            f"{causes['bar_contact']} | {causes['out_of_bounds']} |"
        )
    screen = summary["mechanism_screen"]
    lines.extend(
        [
            "",
            "## 사전 고정한 기술적 판독",
            "",
            f"- distance q3−q0 capture: {screen['distance_q3_minus_q0_capture_pp']:+.2f} pp",
            f"- distance q3−q0 crash: {screen['distance_q3_minus_q0_crash_pp']:+.2f} pp",
            f"- distance q3−q0 timeout: {screen['distance_q3_minus_q0_timeout_pp']:+.2f} pp",
            f"- speed-bin timeout range: {screen['speed_timeout_range_pp']:.2f} pp",
            "",
            "이 수치는 연관 진단이며 인과효과나 P2 PASS로 해석하지 않는다.",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: ... {preflight|run|finalize|verify}",
    )
    if mode == "preflight":
        verify_prerequisites(require_clean=False)
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        run_evaluator(preflight=True)
        require(not OUTPUT.exists(), "preflight created output")
        print("[ref5in-outcome-diagnostic] PREFLIGHT PASS")
        return 0
    if mode == "run":
        verify_prerequisites(require_clean=True)
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        run_evaluator()
        result = verify_result()
        summary = summarize(result)
        write_summary(summary)
        print(json.dumps(summary["mechanism_screen"], indent=2))
        return 0
    if mode == "finalize":
        verify_prerequisites(require_clean=False)
        result = verify_result()
        write_summary(summarize(result))
        print("[ref5in-outcome-diagnostic] FINALIZE PASS")
        return 0
    verify_prerequisites(require_clean=False)
    result = verify_result()
    expected = summarize(result)
    recorded = load_json(OUTPUT / "summary.json")
    for key in ("scope", "decision_authority", "p3_unlocked", "checkpoint_sha256",
                "condition", "outcome", "strata", "mechanism_screen", "limitations"):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-outcome-diagnostic] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[ref5in-outcome-diagnostic] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
