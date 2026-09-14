#!/usr/bin/env python3
"""Preregistered initial-observability causal control for ref5in (RESEARCH_PLAN 8.29).

Seed 359 showed 87.52% of away-heading timeouts never acquire the target at all, against 0.00% for
captures. That is an association: outcome is downstream of the trajectory, so an episode that ends
early has fewer chances to acquire anything. This control holds the task fixed and changes ONLY the
observability, which is the one manipulation that can separate the two.

Two arms, away heading only, differing in exactly one value: the target camera's detection range,
20 m against 28 m. The hard-distance contract samples goals in [22.5, 28] m, so arm A cannot see
the goal region at episode start and arm B can. Goal distances, arena, bar count, target motion,
policy, reward, controller, governor and horizon are identical between arms.

Why not the distance-restriction alternative. Shrinking the goal range into the existing camera
range would also change path length, time budget and bar exposure together, and D0 already reports
6-11.5 m timeouts at 0.06% -- it would largely re-derive a known number through a confounded
manipulation.

Stated before running, because it decides how a null reads: the frozen policy was trained against a
20 m camera. Arm B pushes the target-token distribution outside that training support, so if
timeouts do NOT fall, that is NOT evidence against initial unobservability -- it is indistinguishable
from "this policy cannot exploit long-range detections". Only a fall is strong evidence, and only in
one direction.

Usage: tools/run_navrl_ref5in_camera_range_control.py {preflight|run|finalize|verify}
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/navrl_ref5in_camera_range_control_seed367"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
FIRST_ACQ_SUMMARY = ROOT / "results/navrl_ref5in_cv_first_acquisition_seed359/summary.json"
SEED = 367
HEADING = "away"
# (arm directory name, target camera detection range in metres)
ARMS = (("camera_20m", 20.0), ("camera_28m", 28.0))
EXPECTED_MISMATCH = "cfg_target_pattern: checkpoint=mixed expected=cv"
PRODUCER = "tools/run_navrl_ref5in_camera_range_control.py"
SUMMARY_SCOPE = "post_d1_frozen_camera_range_causal_control_seed367"
INCLUDE_OOB_FORENSICS = False

# Preregistered gates. Fixed here before any measurement; never recomputed from the cells.
TIMEOUT_DROP_THRESHOLD = 0.20   # arm A minus arm B timeout rate, primary
CRASH_RISE_LIMIT = 0.10         # arm B minus arm A crash rate; above this the drop is not a win


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

require(RUN.BARS == 1 and RUN.EPISODES == 2049, "near-open base contract changed")
RUN.OUTPUT = OUTPUT
RUN.SOURCE_BUNDLE = SOURCE_BUNDLE
RUN.SEED = SEED


def cell_dir(arm: str) -> Path:
    return OUTPUT / "cells" / arm


def canonical_env(arm: str, camera_range: float, *, preflight: bool, force: bool) -> dict[str, str]:
    """Inherit the audited cell environment and change exactly one value.

    The heading is pinned to `away` for BOTH arms: the arm axis here is observability, not heading,
    and letting the heading vary would reintroduce the confound seed 359 was run to isolate.
    """
    env = P2.canonical_env(cell_dir(arm), preflight=preflight)
    env.update(
        {
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(RUN.BARS),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(arm)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            "NAVRL_V2_TARGET_PATTERN": "cv",
            "NAVRL_EVAL_CV_INITIAL_HEADING": HEADING,
            "NAVRL_DETECTOR_MAX_RANGE": str(camera_range),
        }
    )
    if force:
        env["NAVRL_V2_FORCE"] = "1"
    return env


def run_command(arm: str, camera_range: float, *, preflight: bool, force: bool, check: bool = True):
    return subprocess.run(
        ["bash", str(RUN.EVALUATOR), str(RUN.CHECKPOINT), str(RUN.EPISODES)],
        cwd=ROOT,
        env=canonical_env(arm, camera_range, preflight=preflight, force=force),
        text=True,
        stdout=subprocess.PIPE if not check else None,
        stderr=subprocess.STDOUT if not check else None,
        check=check,
    )


def verify_cell(arm: str, camera_range: float) -> tuple[dict, dict]:
    """Arm-aware copy of the audited cell verifier.

    The base verifier asserts `receipt["cv_initial_heading"] == mode`, which cannot hold here
    because both arms share one heading and differ by camera range instead. Every other guard is
    reproduced: artifact existence, result-hash agreement with the receipt, and the pinned contract
    fields -- plus the arm's own camera range, without which the two cells' provenance would be
    indistinguishable.
    """
    directory = cell_dir(arm)
    result_path = directory / f"{RUN.BARS}bars.json"
    receipt_path = directory / f"{RUN.BARS}bars.receipt.json"
    require(result_path.is_file(), f"{arm}: result json missing")
    require(receipt_path.is_file(), f"{arm}: receipt missing")
    result = load_json(result_path)
    receipt = load_json(receipt_path)
    require(
        P2.sha256_file(result_path) == receipt["result_sha256"],
        f"{arm}: result json does not match its receipt hash",
    )
    snapshot = directory / "checkpoint_snapshot.pth"
    require(snapshot.is_file(), f"{arm}: checkpoint snapshot missing")
    require(
        P2.sha256_file(snapshot) == RUN.CHECKPOINT_SHA,
        f"{arm}: evaluated checkpoint is not the pinned D1 terminal checkpoint",
    )
    for key, value in {
        "seed": SEED,
        "bars": RUN.BARS,
        "requested_episodes": RUN.EPISODES,
        "source_checkpoint_sha256": RUN.CHECKPOINT_SHA,
        "goal_dist_min_m": 22.5,
        "goal_dist_max_m": 28.0,
        "target_pattern": "cv",
        "cv_initial_heading": HEADING,
        "target_camera_max_range_m": camera_range,
    }.items():
        require(receipt.get(key) == value, f"{arm} receipt mismatch: {key}")
    # The evaluator records the manipulated value in the RECEIPT, not in the result condition
    # block. The receipt is hash-bound to the result above, so this is equally strong provenance --
    # but it only proves what was REQUESTED of the process. The behavioural confirmation that the
    # perception module actually used the longer range lives in verify_all().
    require(
        float(receipt["target_camera_max_range_m"]) == camera_range,
        f"{arm}: receipt camera range is not the arm's value",
    )
    # The evaluator drains whole 128-env batches, so a cell finishes at or just past the request.
    # The audited base verifier uses the same `requested == N and actual >= N` pair; asserting
    # exact equality here was my error, not a contract violation -- one arm landed on 2,050.
    require(
        int(result.get("requested_episodes", -1)) == RUN.EPISODES
        and int(result["actual_episodes"]) >= RUN.EPISODES,
        f"{arm}: episode contract mismatch",
    )
    return result, receipt


def verify_all() -> dict[str, dict]:
    results: dict[str, dict] = {}
    runtime_maps = []
    for arm, camera_range in ARMS:
        result, receipt = verify_cell(arm, camera_range)
        results[arm] = result
        manifest = Path(receipt["runtime_source_manifest"]).resolve()
        mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
        RUN.BASE.verify_runtime_clean_manifest(metadata, arm)
        runtime_maps.append(mapping)
    require(
        runtime_maps[0] == runtime_maps[1],
        "camera-range arms used different runtime byte maps",
    )
    # The manipulation must be the ONLY difference: identical bytes, identical seed, identical
    # heading. Everything else that differs would be a confound.
    a, b = (results[arm] for arm, _ in ARMS)
    for key in ("seed", "bars", "goal_dist_min_m", "goal_dist_max_m", "cv_initial_heading",
                "target_pattern", "speed_governor_mode", "action_selection", "episode_len_steps",
                "robot_name", "robot_asset_sha256"):
        require(
            a["condition"].get(key) == b["condition"].get(key),
            f"arms differ in {key}, which must be held fixed",
        )

    # Manipulation check, and the only one that is behavioural rather than bookkeeping. A receipt
    # proves the env var was requested; it cannot prove the perception module honoured it. If the
    # camera really reaches further, strictly fewer episodes should end without ever acquiring the
    # target. Deliberately a DIRECTION test with no magnitude threshold: a tuned number here could
    # be accused of being chosen to fit, whereas "strictly fewer" cannot.
    def pooled_never(result):
        rows = result["target_motion"]["first_acquisition"].values()
        never = sum(int(r["never_acquired"]) for r in rows)
        episodes = sum(int(r["episodes"]) for r in rows)
        return never / episodes if episodes else None

    control_never, treated_never = pooled_never(a), pooled_never(b)
    require(
        control_never is not None and treated_never is not None,
        "first-acquisition telemetry missing; cannot confirm the manipulation took effect",
    )
    require(
        treated_never < control_never,
        "the 28 m arm did not acquire the target more often than the 20 m arm -- the camera "
        f"range change did not take effect ({treated_never:.4f} !< {control_never:.4f})",
    )
    return results


def _pooled_never(result: dict):
    rows = result["target_motion"]["first_acquisition"].values()
    never = sum(int(r["never_acquired"]) for r in rows)
    episodes = sum(int(r["episodes"]) for r in rows)
    return never / episodes if episodes else None


def first_acquisition_payload(row: dict) -> dict:
    keys = (
        "episodes", "never_acquired", "never_acquired_rate", "acquired",
        "first_visible_step_mean", "first_visible_step_median",
        "visible_hidden_transitions_mean_per_episode",
    )
    return {key: row[key] for key in keys}


def verify_prerequisites(*, require_clean: bool) -> dict:
    RUN.verify_prerequisites(require_clean=require_clean)
    require(FIRST_ACQ_SUMMARY.is_file(), "seed359 first-acquisition summary missing")
    prior = load_json(FIRST_ACQ_SUMMARY)
    require(
        prior.get("checkpoint_sha256") == RUN.CHECKPOINT_SHA,
        "first-acquisition checkpoint lineage mismatch",
    )
    # This control is authorised precisely because the acquisition channel screen passed. If that
    # ever changes, the motivating question is different and this must be re-preregistered.
    require(
        prior.get("screen", {}).get("interpretation")
        == "initial_acquisition_range_contract_channel_supported",
        "seed359 screen does not authorize the camera-range control",
    )
    require(prior.get("p3_unlocked") is False, "first-acquisition diagnostic unlocked P3")
    return prior


def build_summary(results: dict[str, dict], prior: dict) -> dict:
    try:
        checkpoint_display = str(RUN.CHECKPOINT.relative_to(ROOT))
    except ValueError:
        # A clean git worktree may deliberately read the immutable checkpoint from the primary
        # workspace so ignored multi-GB run artifacts do not have to be copied into the worktree.
        checkpoint_display = str(RUN.CHECKPOINT)
    cells = {}
    for arm, camera_range in ARMS:
        result = results[arm]
        standard = RUN.cell_summary(result)
        standard["target_camera_max_range_m"] = camera_range
        standard["first_acquisition"] = {
            label: first_acquisition_payload(row)
            for label, row in result["target_motion"]["first_acquisition"].items()
        }
        if INCLUDE_OOB_FORENSICS:
            oob = result.get("target_motion", {}).get("oob_exit_forensics")
            require(
                isinstance(oob, dict) and isinstance(oob.get("by_acquisition"), dict),
                f"{arm}: acquisition-stratified OOB forensics missing",
            )
            standard["oob_exit_forensics"] = oob
        cells[arm] = standard

    control, treated = cells["camera_20m"], cells["camera_28m"]
    timeout_drop = control["timeout_rate"] - treated["timeout_rate"]
    crash_rise = treated["crash_rate"] - control["crash_rate"]
    primary_support = timeout_drop >= TIMEOUT_DROP_THRESHOLD
    failure_mode_shifted = crash_rise >= CRASH_RISE_LIMIT

    if primary_support and not failure_mode_shifted:
        interpretation = "initial_unobservability_dominant_cause_supported"
    elif primary_support and failure_mode_shifted:
        interpretation = "timeout_reduced_but_failure_mode_shifted_to_crash"
    else:
        interpretation = "no_supported_reduction_null_is_uninterpretable_see_limitation"

    return {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SUMMARY_SCOPE,
        "decision_authority": "none",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        "checkpoint": checkpoint_display,
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
            "cv_initial_heading": HEADING,
            "target_speed_mps": [0.3, 1.5],
            "action_selection": "deterministic",
            "episode_len_steps": 600,
            "manipulated_variable": "vision.detector_max_range",
            "arms_m": {arm: rng for arm, rng in ARMS},
        },
        "manipulation_check": {
            "metric": "pooled never-acquired rate (all outcomes)",
            "control": _pooled_never(results["camera_20m"]),
            "treated": _pooled_never(results["camera_28m"]),
            "note": "direction test only; confirms the longer camera range actually changed what "
                    "the perception module could see, which a receipt alone cannot show",
        },
        "seed359_away_timeout_never_acquired_rate": prior["screen"][
            "away_timeout_never_acquired_rate"
        ],
        "cells": cells,
        "screen": {
            "primary_metric": "timeout_rate_reduction",
            "control_timeout_rate": control["timeout_rate"],
            "treated_timeout_rate": treated["timeout_rate"],
            "timeout_rate_drop": timeout_drop,
            "timeout_drop_threshold": TIMEOUT_DROP_THRESHOLD,
            "initial_unobservability_causal_support": primary_support,
            "control_crash_rate": control["crash_rate"],
            "treated_crash_rate": treated["crash_rate"],
            "crash_rate_rise": crash_rise,
            "crash_rise_limit": CRASH_RISE_LIMIT,
            "failure_mode_shifted": failure_mode_shifted,
            "interpretation": interpretation,
        },
        "limitations": [
            "the frozen policy was trained with a 20 m camera; the 28 m arm is outside its "
            "training support, so a NULL result cannot be read as 'initial unobservability is not "
            "causal' -- it is indistinguishable from 'this policy cannot exploit long-range "
            "detections'. Only a reduction is strong evidence, and only in one direction",
            "one frozen training seed and one held-out evaluation seed",
            "away heading only; toward was already near-ceiling and is not re-measured",
            "changing the camera range changes the observation distribution, not just whether the "
            "target is observable at spawn",
            "diagnostic only; cannot revise P2/D1 or unlock P3",
        ],
    }


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    s = payload["screen"]
    lines = [
        "# ref5in 초기 관측가능성 인과 대조 (seed 367)",
        "",
        "target camera range **한 값만** 바꾼 2-arm 대조다. 정책·보상·아레나·horizon·heading 불변.",
        "P2/D1 재판정 없음, P3 해제 없음.",
        "",
        "| arm | camera range | capture | crash | timeout | never-acq (timeout) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, rng in ARMS:
        c = payload["cells"][arm]
        na = c["first_acquisition"]["timeout"]["never_acquired_rate"]
        lines.append(
            f"| {arm} | {rng:.0f} m | {c['capture_rate']*100:.2f}% | "
            f"{c['crash_rate']*100:.2f}% | {c['timeout_rate']*100:.2f}% | "
            f"{'—' if na is None else f'{na*100:.2f}%'} |"
        )
    lines.extend([
        "",
        f"사전 판정: `{s['interpretation']}`",
        "",
        f"- primary: timeout 감소 **{s['timeout_rate_drop']*100:+.2f} pp** "
        f"(임계 {s['timeout_drop_threshold']*100:.0f} pp) → "
        f"{'통과' if s['initial_unobservability_causal_support'] else '미달'}",
        f"- guard: crash 증가 **{s['crash_rate_rise']*100:+.2f} pp** "
        f"(한계 {s['crash_rise_limit']*100:.0f} pp) → "
        f"{'실패모드 이동' if s['failure_mode_shifted'] else '이동 없음'}",
        "",
        "**한계(사전 명시):** 정책은 20 m로 학습됐다. 28 m arm은 학습 범위 밖이므로 timeout이",
        "줄지 않아도 \"비관측이 원인이 아니다\"로 읽을 수 없다 — \"이 정책이 장거리 검출을",
        "활용하지 못한다\"와 구분되지 않는다. 감소가 관측될 때만 단방향으로 강한 증거다.",
        "",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: ... {preflight|run|finalize|verify}",
    )
    prior = verify_prerequisites(require_clean=mode == "run")
    if mode == "preflight":
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        arm, rng = ARMS[0]
        completed = run_command(arm, rng, preflight=True, force=False, check=False)
        require(completed.returncode == 2, "preflight did not fail closed on the known mismatch")
        lines = [
            ln.strip() for ln in (completed.stdout or "").splitlines()
            if ln.strip().startswith("cfg_")
        ]
        require(lines == [EXPECTED_MISMATCH], f"unexpected preflight mismatch: {lines}")
        print("[ref5in-camera-range] PREFLIGHT PASS")
        return 0
    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        for arm, rng in ARMS:
            print(f"[ref5in-camera-range] RUN {arm} ({rng:.0f} m)", flush=True)
            run_command(arm, rng, preflight=False, force=True)
        payload = build_summary(verify_all(), prior)
        write_summary(payload)
        print(f"[ref5in-camera-range] COMPLETE | {payload['screen']['interpretation']}")
        return 0

    expected = build_summary(verify_all(), prior)
    if mode == "finalize":
        write_summary(expected)
        print("[ref5in-camera-range] FINALIZE PASS")
        return 0
    recorded = load_json(OUTPUT / "summary.json")
    for key in (
        "scope", "decision_authority", "p2_verdict_changed", "d1_verdict_changed",
        "p3_unlocked", "checkpoint_sha256", "generic_provenance_override", "condition",
        "seed359_away_timeout_never_acquired_rate", "cells", "screen", "limitations",
    ):
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print("[ref5in-camera-range] VERIFY PASS")
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
        print(f"[ref5in-camera-range] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
