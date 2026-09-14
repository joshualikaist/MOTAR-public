#!/usr/bin/env python3
"""Run the preregistered outcome-level visibility/reflection diagnostic for ref5in."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/navrl_ref5in_cv_heading_telemetry_seed353"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
NEAR_OPEN_SUMMARY = ROOT / "results/navrl_ref5in_cv_heading_near_open_seed347/summary.json"
SEED = 353
MODES = ("toward", "away")
EXPECTED_MISMATCH = "cfg_target_pattern: checkpoint=mixed expected=cv"


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

# Reuse the already audited launch and cell-contract implementation with a new immutable output
# root and held-out seed. Assert its fixed knobs before rebinding so source drift fails closed.
require(RUN.BARS == 1 and RUN.EPISODES == 2049, "near-open base contract changed")
require(RUN.MODES == MODES, "near-open base modes changed")
RUN.OUTPUT = OUTPUT
RUN.SOURCE_BUNDLE = SOURCE_BUNDLE
RUN.SEED = SEED


def verify_prerequisites(*, require_clean: bool) -> tuple[dict, dict]:
    dense = RUN.verify_prerequisites(require_clean=require_clean)
    require(NEAR_OPEN_SUMMARY.is_file(), "seed347 near-open summary missing")
    near = load_json(NEAR_OPEN_SUMMARY)
    require(
        near.get("checkpoint_sha256") == RUN.CHECKPOINT_SHA,
        "near-open checkpoint lineage mismatch",
    )
    require(
        near.get("screen", {}).get("interpretation")
        == "obstacle_occlusion_not_necessary_kinematic_fov_wall_channel_sufficient",
        "near-open screen does not authorize telemetry follow-up",
    )
    require(near.get("p3_unlocked") is False, "near-open diagnostic unlocked P3")
    return dense, near


def validate_telemetry(mode: str, result: dict) -> dict:
    target_motion = result.get("target_motion") or {}
    outcome_telemetry = target_motion.get("outcome_telemetry") or {}
    speed_wall = target_motion.get("speed_by_wall_reflection") or {}
    outcome = result["outcome"]
    expected_counts = {
        "capture": int(outcome["captured"]),
        "crash": int(outcome["crash"]),
        "timeout": int(outcome["timeout"]),
    }
    for label, expected in expected_counts.items():
        row = outcome_telemetry.get(label) or {}
        require(int(row.get("episodes", -1)) == expected, f"{mode}/{label} count mismatch")
        observations = int(row.get("observation_steps", -1))
        visible = int(row.get("visible_steps", -1))
        require(observations >= expected and 0 <= visible <= observations,
                f"{mode}/{label} visibility accounting failed")
        for prefix in ("wall", "bar"):
            total = int(row.get(prefix + "_reflections", -1))
            any_count = int(row.get(prefix + "_reflection_any", -1))
            any_rate = row.get(prefix + "_reflection_any_rate")
            require(total >= any_count >= 0 and any_count <= expected,
                    f"{mode}/{label} {prefix} accounting failed")
            require(
                (any_rate is None and expected == 0)
                or abs(float(any_rate) - any_count / max(1, expected)) <= 1e-12,
                f"{mode}/{label} {prefix} rate mismatch",
            )

    aggregate = {label: 0 for label in expected_counts}
    for speed_label in ("q0", "q1", "q2", "q3"):
        require(speed_label in speed_wall, f"{mode} missing speed row {speed_label}")
        for reflection_label in ("zero", "one_or_more"):
            row = speed_wall[speed_label].get(reflection_label) or {}
            counts = {
                "capture": int(row.get("captured", -1)),
                "crash": int(row.get("crash", -1)),
                "timeout": int(row.get("timeout", -1)),
            }
            require(min(counts.values()) >= 0, f"{mode} negative speed/reflection count")
            require(sum(counts.values()) == int(row.get("episodes", -1)),
                    f"{mode} speed/reflection row mismatch")
            for label, count in counts.items():
                aggregate[label] += count
    require(aggregate == expected_counts, f"{mode} speed/reflection total mismatch: {aggregate}")
    return outcome_telemetry


def verify_all() -> dict[str, dict]:
    results, runtime_maps = {}, []
    for mode in MODES:
        result, receipt = RUN.verify_cell(mode)
        validate_telemetry(mode, result)
        results[mode] = result
        manifest = Path(receipt["runtime_source_manifest"]).resolve()
        mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
        RUN.BASE.verify_runtime_clean_manifest(metadata, mode)
        runtime_maps.append(mapping)
    require(runtime_maps[0] == runtime_maps[1], "telemetry cells used different runtime byte maps")
    return results


def outcome_payload(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "episodes",
            "observation_steps",
            "visible_steps",
            "visible_fraction_step_weighted",
            "wall_reflections",
            "wall_reflections_mean_per_episode",
            "wall_reflection_any",
            "wall_reflection_any_rate",
            "bar_reflections",
            "bar_reflections_mean_per_episode",
            "bar_reflection_any",
            "bar_reflection_any_rate",
        )
    }


def wall_association_by_speed(result: dict) -> dict:
    table = result["target_motion"]["speed_by_wall_reflection"]
    payload = {}
    for speed_label in ("q0", "q1", "q2", "q3"):
        zero = table[speed_label]["zero"]
        reflected = table[speed_label]["one_or_more"]
        zero_rate = zero["capture_rate"]
        reflected_rate = reflected["capture_rate"]
        payload[speed_label] = {
            "zero_reflection_episodes": zero["episodes"],
            "reflected_episodes": reflected["episodes"],
            "zero_reflection_capture_rate": zero_rate,
            "reflected_capture_rate": reflected_rate,
            "capture_difference_reflected_minus_zero": (
                float(reflected_rate - zero_rate)
                if reflected_rate is not None and zero_rate is not None
                else None
            ),
        }
    return payload


def build_summary(results: dict[str, dict], near: dict) -> dict:
    cells = {}
    for mode, result in results.items():
        standard = RUN.cell_summary(result)
        standard["outcome_telemetry"] = {
            label: outcome_payload(row)
            for label, row in result["target_motion"]["outcome_telemetry"].items()
        }
        standard["wall_capture_association_by_speed"] = wall_association_by_speed(result)
        cells[mode] = standard

    away = cells["away"]["outcome_telemetry"]
    capture_visible = away["capture"]["visible_fraction_step_weighted"]
    timeout_visible = away["timeout"]["visible_fraction_step_weighted"]
    visibility_gap = capture_visible - timeout_visible
    fov_support = visibility_gap >= 0.20
    return {
        "schema_version": 1,
        "producer": "tools/run_navrl_ref5in_cv_heading_telemetry.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_d1_frozen_cv_heading_outcome_telemetry_seed353",
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
        },
        "near_open_seed347_timeout_difference": near["screen"]["near_open_timeout_difference"],
        "cells": cells,
        "screen": {
            "away_capture_visible_fraction_step_weighted": capture_visible,
            "away_timeout_visible_fraction_step_weighted": timeout_visible,
            "away_capture_minus_timeout_visibility": visibility_gap,
            "fov_track_loss_support_threshold": 0.20,
            "fov_track_loss_channel_support": fov_support,
            "interpretation": (
                "prioritize_fov_memory_or_target_tracker_intervention"
                if fov_support
                else "fov_screen_not_met_inspect_wall_association"
            ),
        },
        "limitations": [
            "telemetry-only mechanism diagnostic; cannot revise P2/D1 or unlock P3",
            "step-weighted visibility is descriptive and is not an episode-paired causal effect",
            "wall-reflection association is selected by episode duration and target speed",
            "one frozen training seed and one held-out evaluation seed",
        ],
    }


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ref5in outcome-level visibility/reflection telemetry",
        "",
        "이 결과는 정책을 바꾸지 않은 원인 진단이며 P3를 해제하지 않는다.",
        "",
        "| heading | outcome | episodes | visible | wall any | wall mean | bar any |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for outcome in ("capture", "crash", "timeout"):
            row = payload["cells"][mode]["outcome_telemetry"][outcome]
            lines.append(
                f"| {mode} | {outcome} | {row['episodes']:,} | "
                f"{row['visible_fraction_step_weighted']*100:.2f}% | "
                f"{row['wall_reflection_any_rate']*100:.2f}% | "
                f"{row['wall_reflections_mean_per_episode']:.3f} | "
                f"{row['bar_reflection_any_rate']*100:.2f}% |"
            )
    screen = payload["screen"]
    lines.extend(
        [
            "",
            f"사전 screen: `{screen['interpretation']}`",
            f"away capture−timeout visibility: {screen['away_capture_minus_timeout_visibility']*100:+.2f} pp",
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(mode in {"preflight", "run", "finalize", "verify"},
            "usage: ... {preflight|run|finalize|verify}")
    _, near = verify_prerequisites(require_clean=mode == "run")
    if mode == "preflight":
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        mismatch = RUN.verify_narrow_override()
        require(mismatch == EXPECTED_MISMATCH, "unexpected preflight mismatch")
        print("[ref5in-cv-telemetry] PREFLIGHT PASS")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        mismatch = RUN.verify_narrow_override()
        require(mismatch == EXPECTED_MISMATCH, "unexpected preflight mismatch")
        for heading in MODES:
            print(f"[ref5in-cv-telemetry] RUN {heading}", flush=True)
            RUN.run_command(heading, preflight=False, force=True)
        payload = build_summary(verify_all(), near)
        write_summary(payload)
        print(f"[ref5in-cv-telemetry] COMPLETE | {payload['screen']['interpretation']}")
        return 0

    expected = build_summary(verify_all(), near)
    if mode == "finalize":
        write_summary(expected)
        print("[ref5in-cv-telemetry] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed",
        "p3_unlocked", "checkpoint_sha256", "generic_provenance_override", "condition",
        "near_open_seed347_timeout_difference", "cells", "screen", "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-cv-telemetry] VERIFY PASS")
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
        print(f"[ref5in-cv-telemetry] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
