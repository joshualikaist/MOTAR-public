#!/usr/bin/env python3
"""Correct the speed-bin contract and add distance×pattern cells to the seed-317 diagnostic."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/navrl_ref5in_outcome_diagnostic_seed317"
OUTPUT = ROOT / "results/navrl_ref5in_outcome_diagnostic_v2_seed317"


def load_base():
    spec = importlib.util.spec_from_file_location(
        "ref5in_outcome_base", ROOT / "tools/run_navrl_ref5in_outcome_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUTPUT = OUTPUT
    module.CELL = OUTPUT / "ref5in"
    module.SOURCE_BUNDLE = OUTPUT / "source_bundle"
    return module


BASE = load_base()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BASE.ContractError(message)


def verify_joint(result: dict) -> None:
    joint = (result.get("strata") or {}).get("distance_by_pattern") or {}
    require(set(joint) == {"q0", "q1", "q2", "q3"}, "distance×pattern rows missing")
    flattened = {}
    for distance, cells in joint.items():
        for pattern, cell in cells.items():
            flattened[f"{distance}/{pattern}"] = cell
    outcome = result["outcome"]
    totals = (
        int(result["actual_episodes"]),
        int(outcome["captured"]),
        int(outcome["crash"]),
        int(outcome["timeout"]),
    )
    BASE.verify_axis("distance_by_pattern", flattened, totals)


def verify_parity(result: dict) -> None:
    original = BASE.load_json(ORIGINAL / "ref5in/70bars.json")
    for key in ("outcome", "crash_causes"):
        require(result.get(key) == original.get(key), f"behavioral parity failed: {key}")
    for key in ("distance", "pattern", "initial_target_bearing"):
        require(result["strata"].get(key) == original["strata"].get(key),
                f"behavioral parity failed: strata/{key}")
    edges = result["strata"].get("speed_bin_edges_mps") or []
    expected = [0.3, 0.6, 0.9, 1.2, 1.5]
    require(
        len(edges) == len(expected)
        and all(math.isclose(float(a), b, abs_tol=1e-12) for a, b in zip(edges, expected)),
        "corrected speed support is not [0.3,1.5]",
    )


def enriched_joint(result: dict) -> dict:
    return {
        distance: BASE.enrich_axis(cells)
        for distance, cells in result["strata"]["distance_by_pattern"].items()
    }


def summary(result: dict) -> dict:
    payload = BASE.summarize(result)
    payload["schema_version"] = 2
    payload["producer"] = "tools/run_navrl_ref5in_outcome_diagnostic_v2.py"
    payload["scope"] = "post_p2_descriptive_outcome_strata_v2_seed317"
    payload["supersedes"] = {
        "artifact": str(ORIGINAL),
        "scope": "speed strata only",
        "reason": "v1 speed bins covered [0,1.5] instead of applied [0.3,1.5] support",
        "distance_pattern_global_outcomes_remain_valid": True,
    }
    payload["strata"]["distance_by_pattern"] = enriched_joint(result)
    payload["behavioral_parity_with_v1"] = True
    payload["limitations"].append(
        "same-seed deterministic replay corrects telemetry; it is not independent replication"
    )
    return payload


def write(payload: dict) -> None:
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in post-P2 outcome diagnostic v2",
        "",
        "동일 seed·checkpoint의 deterministic replay로 speed bin을 실제 [0.3,1.5] 지원범위에 맞췄다.",
        "전역 outcome과 distance/pattern/bearing count가 v1과 byte-level JSON 값으로 동일해야 유효하다.",
        "",
        "| speed bin | episodes | capture | crash | timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, cell in payload["strata"]["speed"].items():
        lines.append(
            f"| {label} | {cell['episodes']:,} | {cell['capture_rate']*100:.2f}% | "
            f"{cell['crash_rate']*100:.2f}% | {cell['timeout_rate']*100:.2f}% |"
        )
    lines.extend(["", "## Distance × pattern", ""])
    for distance, cells in payload["strata"]["distance_by_pattern"].items():
        for pattern in ("cv", "waypoint"):
            cell = cells[pattern]
            lines.append(
                f"- {distance}/{pattern}: n={cell['episodes']:,}, capture "
                f"{cell['capture_rate']*100:.2f}%, crash {cell['crash_rate']*100:.2f}%, "
                f"timeout {cell['timeout_rate']*100:.2f}%"
            )
    lines.extend(
        ["", "이 평가는 계측 정정이며 P2 판정이나 P3 차단 상태를 바꾸지 않는다.", ""]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def verify_result() -> dict:
    result = BASE.verify_result()
    verify_joint(result)
    verify_parity(result)
    return result


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: ... {preflight|run|finalize|verify}",
    )
    if mode == "preflight":
        BASE.verify_prerequisites(require_clean=False)
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        BASE.run_evaluator(preflight=True)
        require(not OUTPUT.exists(), "preflight created output")
        print("[ref5in-outcome-diagnostic-v2] PREFLIGHT PASS")
        return 0
    if mode == "run":
        BASE.verify_prerequisites(require_clean=True)
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        BASE.run_evaluator()
        payload = summary(verify_result())
        write(payload)
        print(json.dumps(payload["mechanism_screen"], indent=2))
        return 0
    if mode == "finalize":
        BASE.verify_prerequisites(require_clean=False)
        payload = summary(verify_result())
        write(payload)
        print("[ref5in-outcome-diagnostic-v2] FINALIZE PASS")
        return 0
    BASE.verify_prerequisites(require_clean=False)
    expected = summary(verify_result())
    recorded = BASE.load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p3_unlocked", "checkpoint_sha256", "condition",
        "outcome", "strata", "mechanism_screen", "supersedes", "behavioral_parity_with_v1",
        "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-outcome-diagnostic-v2] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BASE.ContractError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"[ref5in-outcome-diagnostic-v2] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
