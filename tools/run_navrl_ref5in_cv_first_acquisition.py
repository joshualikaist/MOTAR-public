#!/usr/bin/env python3
"""Preregistered first-acquisition / never-acquired diagnostic for ref5in (RESEARCH_PLAN 8.28).

Why this run exists. The seed-353 telemetry showed away-heading timeouts sit at a 0.59%
step-weighted visible fraction against 15.02% for captures -- a +14.43 pp gap that missed its
preregistered 20 pp screen. A visible fraction cannot separate "acquired the target once and lost
it" from "never acquired it at all", and those imply opposite interventions (tracker memory versus
the range contract). The code audit then found the sharper fact: the hard-distance cell samples
goals in [22.5, 28] m while the target camera reaches 20 m and the LiDAR 12 m, so every episode
begins with the target token zeroed and the actor holding only a position prior.

So this measures, per outcome, how many episodes ever acquire the target at all and how late the
ones that do acquire it. Policy, reward, sensor ranges, arena and horizon are unchanged; this is a
frozen replay with one extra counter.

Contract is inherited wholesale from the audited near-open runner (immutable shared source bundle,
per-cell receipts, checkpoint SHA pinning, runtime-clean source map, and the narrow single-line
provenance override), with only the output root and eval seed rebound.

Usage: tools/run_navrl_ref5in_cv_first_acquisition.py {preflight|run|finalize|verify}
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/navrl_ref5in_cv_first_acquisition_seed359"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
TELEMETRY_SUMMARY = ROOT / "results/navrl_ref5in_cv_heading_telemetry_seed353/summary.json"
SEED = 359
MODES = ("toward", "away")
EXPECTED_MISMATCH = "cfg_target_pattern: checkpoint=mixed expected=cv"

# Preregistered screens (RESEARCH_PLAN 8.28). Fixed here, before any measurement, and never
# recomputed from the results.
NEVER_ACQUIRED_GAP_THRESHOLD = 0.30      # away timeout minus away capture, primary
DELAYED_ACQUISITION_STEP_THRESHOLD = 100  # away timeout minus away capture, secondary


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN = load_module(
    "ref5in_cv_near_open_base", ROOT / "tools/run_navrl_ref5in_cv_heading_near_open.py"
)
require = RUN.require
ContractError = RUN.ContractError
load_json = RUN.load_json
P2 = RUN.P2

# Assert the inherited knobs BEFORE rebinding, so a change to the base contract fails closed
# instead of silently redefining this experiment.
require(RUN.BARS == 1 and RUN.EPISODES == 2049, "near-open base contract changed")
require(RUN.MODES == MODES, "near-open base modes changed")
RUN.OUTPUT = OUTPUT
RUN.SOURCE_BUNDLE = SOURCE_BUNDLE
RUN.SEED = SEED


def verify_prerequisites(*, require_clean: bool) -> tuple[dict, dict]:
    dense = RUN.verify_prerequisites(require_clean=require_clean)
    require(TELEMETRY_SUMMARY.is_file(), "seed353 telemetry summary missing")
    prior = load_json(TELEMETRY_SUMMARY)
    require(
        prior.get("checkpoint_sha256") == RUN.CHECKPOINT_SHA,
        "telemetry checkpoint lineage mismatch",
    )
    # This follow-up is authorised precisely BECAUSE the FOV screen was not met; if a future edit
    # flips that screen to supported, the motivating question has changed and this run should be
    # re-preregistered rather than reused.
    require(
        prior.get("screen", {}).get("interpretation")
        == "fov_screen_not_met_inspect_wall_association",
        "telemetry screen does not authorize the first-acquisition follow-up",
    )
    require(prior.get("p3_unlocked") is False, "telemetry diagnostic unlocked P3")
    return dense, prior


def validate_first_acquisition(mode: str, result: dict) -> dict:
    """Re-derive every exported count from the outcome block; abort on any disagreement."""
    target_motion = result.get("target_motion") or {}
    payload = target_motion.get("first_acquisition") or {}
    outcome = result["outcome"]
    expected_counts = {
        "capture": int(outcome["captured"]),
        "crash": int(outcome["crash"]),
        "timeout": int(outcome["timeout"]),
    }
    require(
        set(payload) == set(expected_counts),
        f"{mode}: first_acquisition outcome labels mismatch",
    )
    total = 0
    for label, row in payload.items():
        episodes = int(row["episodes"])
        require(
            episodes == expected_counts[label],
            f"{mode}/{label}: first_acquisition episodes {episodes} != {expected_counts[label]}",
        )
        never = int(row["never_acquired"])
        acquired = int(row["acquired"])
        require(never >= 0 and acquired >= 0, f"{mode}/{label}: negative acquisition counts")
        require(
            never + acquired == episodes,
            f"{mode}/{label}: never+acquired != episodes",
        )
        # A cohort with no acquisitions must report null, never 0 -- a 0 would read as the fastest
        # possible acquisition and invert the finding this experiment exists to make.
        if acquired == 0:
            require(
                row["first_visible_step_mean"] is None
                and row["first_visible_step_median"] is None,
                f"{mode}/{label}: acquired==0 must report null first-visible statistics",
            )
        else:
            require(
                row["first_visible_step_mean"] is not None
                and row["first_visible_step_median"] is not None,
                f"{mode}/{label}: acquired>0 must report first-visible statistics",
            )
            require(
                row["first_visible_step_mean"] > 0
                and row["first_visible_step_median"] > 0,
                f"{mode}/{label}: first-visible step must be a 1-based observation index",
            )
        require(
            int(row["camera_never_acquired"]) >= never,
            f"{mode}/{label}: camera cannot acquire more often than the fused signal",
        )
        if episodes:
            require(
                abs(row["never_acquired_rate"] - never / episodes) < 1e-9,
                f"{mode}/{label}: never_acquired_rate inconsistent",
            )
        total += episodes
    require(
        total == int(result["actual_episodes"]),
        f"{mode}: first_acquisition episodes do not sum to the cell total",
    )
    return payload


def verify_all() -> dict[str, dict]:
    results: dict[str, dict] = {}
    runtime_maps = []
    for mode in MODES:
        # verify_cell returns (result, RECEIPT). The receipt necessarily differs per cell -- it
        # carries the completion timestamp, the evaluation nonce and the per-cell log/result
        # hashes -- so comparing receipts across cells would always fail. The cross-cell invariant
        # is over the runtime BYTE MAP, which has to be read out of the manifest the receipt
        # points at.
        result, receipt = RUN.verify_cell(mode)
        validate_first_acquisition(mode, result)
        results[mode] = result
        manifest = Path(receipt["runtime_source_manifest"]).resolve()
        mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
        RUN.BASE.verify_runtime_clean_manifest(metadata, mode)
        runtime_maps.append(mapping)
    require(
        runtime_maps[0] == runtime_maps[1],
        "first-acquisition cells used different runtime byte maps",
    )
    return results


def acquisition_payload(row: dict) -> dict:
    keys = (
        "episodes",
        "never_acquired",
        "never_acquired_rate",
        "acquired",
        "first_visible_step_mean",
        "first_visible_step_median",
        "visible_hidden_transitions",
        "visible_hidden_transitions_mean_per_episode",
        "camera_never_acquired",
        "camera_never_acquired_rate",
        "camera_first_visible_step_mean",
    )
    return {key: row[key] for key in keys}


def build_summary(results: dict[str, dict], prior: dict) -> dict:
    cells = {}
    for mode, result in results.items():
        standard = RUN.cell_summary(result)
        standard["first_acquisition"] = {
            label: acquisition_payload(row)
            for label, row in result["target_motion"]["first_acquisition"].items()
        }
        cells[mode] = standard

    away = cells["away"]["first_acquisition"]
    capture_never = away["capture"]["never_acquired_rate"]
    timeout_never = away["timeout"]["never_acquired_rate"]
    never_gap = timeout_never - capture_never
    primary_support = never_gap >= NEVER_ACQUIRED_GAP_THRESHOLD

    capture_first = away["capture"]["first_visible_step_mean"]
    timeout_first = away["timeout"]["first_visible_step_mean"]
    delay_gap = (
        timeout_first - capture_first
        if capture_first is not None and timeout_first is not None
        else None
    )
    secondary_support = (
        (not primary_support)
        and delay_gap is not None
        and delay_gap >= DELAYED_ACQUISITION_STEP_THRESHOLD
    )

    if primary_support:
        interpretation = "initial_acquisition_range_contract_channel_supported"
    elif secondary_support:
        interpretation = "delayed_acquisition_channel_supported"
    else:
        interpretation = "initial_observability_sole_explanation_withheld"

    return {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_cv_first_acquisition.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_d1_frozen_cv_first_acquisition_seed359",
        "decision_authority": "none",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        "checkpoint": str(RUN.CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": RUN.CHECKPOINT_SHA,
        "generic_provenance_override": {
            "used": True,
            "sole_verified_mismatch": EXPECTED_MISMATCH,
            "reason": "controlled CV-only evaluation of a mixed-pattern training checkpoint",
        },
        "condition": {
            "seed": SEED,
            "bars": RUN.BARS,
            "requested_episodes_per_cell": RUN.EPISODES,
            "goal_dist_m": [22.5, 28.0],
            "target_pattern": "cv",
            "target_speed_mps": [0.3, 1.5],
            "action_selection": "deterministic",
            "episode_len_steps": 600,
            "modes": list(MODES),
            "sensor_ranges_m": {"target_camera": 20.0, "lidar": 12.0},
        },
        "telemetry_seed353_visibility_gap": prior["screen"][
            "away_capture_minus_timeout_visibility"
        ],
        "cells": cells,
        "screen": {
            "primary_metric": "fused_never_acquired_rate",
            "away_capture_never_acquired_rate": capture_never,
            "away_timeout_never_acquired_rate": timeout_never,
            "away_timeout_minus_capture_never_acquired": never_gap,
            "never_acquired_gap_threshold": NEVER_ACQUIRED_GAP_THRESHOLD,
            "initial_acquisition_channel_support": primary_support,
            "away_capture_first_visible_step_mean": capture_first,
            "away_timeout_first_visible_step_mean": timeout_first,
            "away_timeout_minus_capture_first_visible_step": delay_gap,
            "delayed_acquisition_step_threshold": DELAYED_ACQUISITION_STEP_THRESHOLD,
            "delayed_acquisition_channel_support": secondary_support,
            "interpretation": interpretation,
        },
        "limitations": [
            "telemetry-only mechanism diagnostic; cannot revise P2/D1 or unlock P3",
            "visibility is an associational measurement, not a causal effect or a remedy",
            "never-acquired is conditioned on the outcome, which is itself downstream of the "
            "trajectory; episodes that crash early have fewer chances to acquire",
            "one frozen training seed and one held-out evaluation seed",
            "camera-only acquisition is a secondary channel; the preregistered primary is fused",
        ],
    }


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in first-acquisition / never-acquired telemetry (seed 359)",
        "",
        "정책·보상·센서 range·아레나·horizon을 바꾸지 않은 동결 replay 진단이다. P3를 해제하지 않고",
        "P2/D1 판정도 바꾸지 않는다.",
        "",
        "| heading | outcome | episodes | never acq. | first-visible mean | median | vis→hid /ep |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for outcome in ("capture", "crash", "timeout"):
            row = payload["cells"][mode]["first_acquisition"][outcome]
            mean = row["first_visible_step_mean"]
            median = row["first_visible_step_median"]
            lines.append(
                f"| {mode} | {outcome} | {row['episodes']:,} | "
                f"{row['never_acquired_rate']*100:.2f}% | "
                f"{'—' if mean is None else f'{mean:.1f}'} | "
                f"{'—' if median is None else median} | "
                f"{row['visible_hidden_transitions_mean_per_episode']:.3f} |"
            )
    screen = payload["screen"]
    gap = screen["away_timeout_minus_capture_never_acquired"]
    delay = screen["away_timeout_minus_capture_first_visible_step"]
    lines.extend(
        [
            "",
            f"사전 판정: `{screen['interpretation']}`",
            "",
            f"- primary: away timeout−capture never-acquired **{gap*100:+.2f} pp** "
            f"(임계 {screen['never_acquired_gap_threshold']*100:.0f} pp) → "
            f"{'통과' if screen['initial_acquisition_channel_support'] else '미달'}",
            # "미달"은 secondary가 실제로 평가되어 탈락했을 때만 쓴다. primary가 통과하면 이
            # screen은 사전등록상 적용 대상이 아니므로, 값이 임계를 넘더라도 "해당 없음"이다.
            f"- secondary: away timeout−capture first-visible "
            f"**{'—' if delay is None else f'{delay:+.1f} step'}** "
            f"(임계 {screen['delayed_acquisition_step_threshold']} step) → "
            + (
                "해당 없음 (primary 통과 시 미적용)"
                if screen["initial_acquisition_channel_support"]
                else ("통과" if screen["delayed_acquisition_channel_support"] else "미달")
            ),
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: ... {preflight|run|finalize|verify}",
    )
    _, prior = verify_prerequisites(require_clean=mode == "run")
    if mode == "preflight":
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        mismatch = RUN.verify_narrow_override()
        require(mismatch == EXPECTED_MISMATCH, "unexpected preflight mismatch")
        print("[ref5in-cv-first-acq] PREFLIGHT PASS")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        mismatch = RUN.verify_narrow_override()
        require(mismatch == EXPECTED_MISMATCH, "unexpected preflight mismatch")
        for heading in MODES:
            print(f"[ref5in-cv-first-acq] RUN {heading}", flush=True)
            RUN.run_command(heading, preflight=False, force=True)
        payload = build_summary(verify_all(), prior)
        write_summary(payload)
        print(f"[ref5in-cv-first-acq] COMPLETE | {payload['screen']['interpretation']}")
        return 0

    expected = build_summary(verify_all(), prior)
    if mode == "finalize":
        write_summary(expected)
        print("[ref5in-cv-first-acq] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed",
        "p3_unlocked", "checkpoint_sha256", "generic_provenance_override", "condition",
        "telemetry_seed353_visibility_gap", "cells", "screen", "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-cv-first-acq] VERIFY PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContractError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[ref5in-cv-first-acq] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
