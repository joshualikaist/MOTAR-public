#!/usr/bin/env python3
"""CPU-only, fail-closed verifier for the braking-aware route-v3 gates.

The simulator launcher writes one immutable JSON summary.  This module deliberately imports no
Isaac Gym or torch code, so receipts can be checked on a CPU host and unit tests can exercise every
scientific verdict without starting the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
# The lower contract is an explicit environment declaration, exactly like the braking probe's
# variant contract.  The canonical grid stays the byte-frozen default; the lower grid replaces
# only the top registered speed per its own preregistration.
CONTRACT_VARIANT = os.environ.get(
    "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", "canonical_1p5"
).strip().lower()
if CONTRACT_VARIANT == "canonical_1p5":
    PREREG = ROOT / "docs/preregistration_braking_aware_route_v3_2026-09-01.md"
    PREREG_SHA256 = "cceecb9ad4a538e7bc2bc9171436e823ef18652e9c971e0d6fa8174279df6056"
    EXECUTION_AUTHORITY = None
    EXECUTION_AUTHORITY_SHA256 = None
    SPEEDS = (0.6, 0.9, 1.2, 1.5)
elif CONTRACT_VARIANT == "baseline_1p25":
    PREREG = ROOT / (
        "docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md"
    )
    PREREG_SHA256 = "17e7f35e350087bf2733aca70dfca7210efe667ad1845dd053f7334ddf1645b4"
    EXECUTION_AUTHORITY = ROOT / (
        "docs/preregistration_braking_aware_route_v3_lower1p25_"
        "matched_spawn_gpu_authority_2026-09-01.md"
    )
    EXECUTION_AUTHORITY_SHA256 = (
        "70b8a08f0c95040a86c43e1be5ac11d0b688b9a6874f3cfc4548f914882f085f"
    )
    SPEEDS = (0.6, 0.9, 1.2, 1.25)
else:
    raise RuntimeError(
        "unknown NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=%r" % CONTRACT_VARIANT
    )
SCHEMA = "navrl_braking_route_v3_gate_v1"
ROUTE_MODE = "global_astar_braking_v3"
ROUTE_ARMS = ("off", ROUTE_MODE)
STAGES = {
    "pilot": {"seed": 829, "densities": (70,), "cells": 8},
    "confirmatory": {"seed": 839, "densities": (70, 115, 160, 205), "cells": 32},
}
ENVS = 32
STEPS = 300
WARMUP_STEPS = 20

# Byte-for-byte corrected-r2 per-cell thresholds.  V3 adds mechanism gates; it does not relax these.
PHYSICAL_GATES = {
    "tracking_rmse_mps_max": 0.35,
    "mean_speed_ratio_min": 0.80,
    "contact_step_fraction_max": 0.01,
    "off_bounded_local_step_infeasible_fraction_max": 0.01,
    "routed_local_step_invalidation_fraction_max": 0.01,
    "motor_saturation_fraction_max": 0.15,
    "max_tilt_deg_max": 60.0,
    "invalid_state_fraction_max": 0.0,
}

# Contract expected from the core v3 telemetry.  Keep these names centralized for reconciliation.
V3_DIAGNOSTIC_FIELDS = (
    "runtime_replan_unsafe_start_count",
    "soft_envelope_exit_count",
    "accepted_terminal_stop_certificate_count",
    "accepted_command_count",
    "plan_attempt_count",
    "plan_success_count",
    "fallback_interval_count",
    "goal_completion_count",
)

IDENTITY_HASH_FIELDS = (
    "source_manifest_sha256",
    "import_origin_sha256",
    "robot_config_sha256",
    "robot_urdf_sha256",
    "sim_config_sha256",
    "task_config_sha256",
    "cell_runner_sha256",
    "braking_receipt_sha256",
)


class IntegrityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def record_id(route_mode: str, speed: float, bars: int) -> str:
    # Lossless decimal key: identical bytes to the historical one-decimal ids for the canonical
    # grid, while 1.25 in the lower contract can never alias 1.2.
    return f"route_{route_mode}__speed_{format(float(speed), '.12g')}__bars_{bars}"


def physical_gate_metrics(row: Mapping) -> dict[str, bool]:
    local = (
        row["off_bounded_local_step_infeasible_fraction"]
        <= PHYSICAL_GATES["off_bounded_local_step_infeasible_fraction_max"]
        if row["route_mode"] == "off"
        else row["routed_local_step_invalidation_fraction"]
        <= PHYSICAL_GATES["routed_local_step_invalidation_fraction_max"]
    )
    return {
        "tracking": row["tracking_rmse_mps"] <= PHYSICAL_GATES["tracking_rmse_mps_max"],
        "speed": row["mean_speed_ratio"] >= PHYSICAL_GATES["mean_speed_ratio_min"],
        "contact": row["contact_step_fraction"] <= PHYSICAL_GATES["contact_step_fraction_max"],
        "arm_specific_local_feasibility": local,
        "motors": row["motor_saturation_fraction"] <= PHYSICAL_GATES["motor_saturation_fraction_max"],
        "tilt": row["max_tilt_deg"] <= PHYSICAL_GATES["max_tilt_deg_max"],
        "state_displacement": row["invalid_state_fraction"] <= PHYSICAL_GATES["invalid_state_fraction_max"],
    }


def validate_identity(identity: Mapping) -> None:
    require(identity.get("runtime_clean") is True, "runtime-clean guard is not true")
    require(identity.get("physical_geometry_version") == "v2", "physical geometry is not v2")
    require(identity.get("placement_mode") == "footprint_clearance", "placement is not footprint_clearance")
    require(identity.get("preregistration_sha256") == PREREG_SHA256, "preregistration SHA drift")
    require(identity.get("robot_name") == "navrl_ref5in_v2_quad", "robot identity drift")
    require(identity.get("target_dynamics") == "physical", "target is not physical")
    require(identity.get("target_pattern") == "waypoint", "target pattern is not waypoint")
    if EXECUTION_AUTHORITY is not None:
        require(
            identity.get("execution_authority_sha256") == EXECUTION_AUTHORITY_SHA256,
            "GPU execution-authority SHA drift",
        )
    for name in IDENTITY_HASH_FIELDS:
        require(_hash(identity.get(name)), f"missing/invalid identity hash: {name}")


def validate_cell(row: Mapping, stage: str, identity: Mapping) -> None:
    config = STAGES[stage]
    route = row.get("route_mode")
    speed = row.get("speed_mps")
    bars = row.get("bars")
    require(route in ROUTE_ARMS and speed in SPEEDS and bars in config["densities"], "cell outside frozen grid")
    require(row.get("record_id") == record_id(route, speed, bars), "record id/header mismatch")
    require(row.get("seed") == config["seed"], "cell seed drift")
    require(row.get("envs") == ENVS and row.get("steps") == STEPS, "env/step contract drift")
    require(row.get("warmup_steps") == WARMUP_STEPS, "warmup contract drift")
    require(row.get("identity") == identity, "cell identity differs from summary identity")
    for name in ("initial_layout_sha256", "initial_robot_pose_sha256", "initial_target_pose_sha256"):
        require(_hash(row.get(name)), f"missing matched-arm digest: {name}")

    numeric = (
        "tracking_rmse_mps", "mean_speed_ratio", "contact_step_fraction",
        "motor_saturation_fraction", "max_tilt_deg", "invalid_state_fraction",
    )
    for name in numeric:
        require(_finite(row.get(name)) and row[name] >= 0.0, f"invalid physical metric: {name}")
    local = "off_bounded_local_step_infeasible_fraction" if route == "off" else "routed_local_step_invalidation_fraction"
    absent = "routed_local_step_invalidation_fraction" if route == "off" else "off_bounded_local_step_infeasible_fraction"
    require(_finite(row.get(local)) and 0.0 <= row[local] <= 1.0, f"invalid local metric: {local}")
    require(row.get(absent) is None, f"inapplicable local metric must be null: {absent}")
    for name in ("contact_step_fraction", "motor_saturation_fraction", "invalid_state_fraction"):
        require(row[name] <= 1.0, f"fraction outside [0,1]: {name}")
    gates = physical_gate_metrics(row)
    require(row.get("physical_gates") == gates, "physical gate recomputation mismatch")
    require(row.get("physical_pass") == all(gates.values()), "physical pass is not conjunctive")

    diagnostics = row.get("v3_diagnostics")
    if route == "off":
        require(diagnostics is None, "route-off cell must not carry v3 diagnostics")
    else:
        require(isinstance(diagnostics, Mapping), "routed cell lacks v3 diagnostics")
        for name in V3_DIAGNOSTIC_FIELDS:
            require(_finite(diagnostics.get(name)) and diagnostics[name] >= 0, f"missing v3 diagnostic: {name}")
        require(diagnostics["plan_success_count"] <= diagnostics["plan_attempt_count"], "plan successes exceed attempts")
        require(diagnostics["accepted_terminal_stop_certificate_count"] <= diagnostics["accepted_command_count"], "certificates exceed accepted commands")


def validate_payload(payload: Mapping, pilot_summary_sha256: str | None = None) -> None:
    require(sha256_file(PREREG) == PREREG_SHA256, "working-tree preregistration bytes drift")
    if EXECUTION_AUTHORITY is not None:
        require(
            EXECUTION_AUTHORITY.is_file()
            and sha256_file(EXECUTION_AUTHORITY) == EXECUTION_AUTHORITY_SHA256,
            "working-tree GPU execution-authority bytes drift",
        )
    require(payload.get("schema") == SCHEMA, "wrong v3 gate schema")
    stage = payload.get("stage")
    require(stage in STAGES, "unknown stage")
    config = STAGES[stage]
    require(payload.get("seed") == config["seed"], "summary seed drift")
    require(payload.get("route_arms") == list(ROUTE_ARMS), "route arms drift")
    require(payload.get("speeds_mps") == list(SPEEDS), "speed grid drift")
    require(payload.get("densities") == list(config["densities"]), "density grid drift")
    require(payload.get("envs") == ENVS and payload.get("steps") == STEPS, "summary env/step drift")
    require(payload.get("warmup_steps") == WARMUP_STEPS, "summary warmup drift")
    require(payload.get("physical_gates_preregistered") == PHYSICAL_GATES, "threshold override/drift")
    identity = payload.get("identity")
    require(isinstance(identity, Mapping), "summary identity missing")
    validate_identity(identity)

    cells = payload.get("cells")
    require(isinstance(cells, list) and len(cells) == config["cells"], "incomplete cell grid")
    expected = {record_id(route, speed, bars) for route in ROUTE_ARMS for speed in SPEEDS for bars in config["densities"]}
    require({row.get("record_id") for row in cells} == expected, "cell grid mismatch/duplicate")
    for row in cells:
        validate_cell(row, stage, identity)
    for speed in SPEEDS:
        for bars in config["densities"]:
            pair = [r for r in cells if r["speed_mps"] == speed and r["bars"] == bars]
            require(len(pair) == 2, "matched arm pair missing")
            for name in ("initial_layout_sha256", "initial_robot_pose_sha256", "initial_target_pose_sha256"):
                require(len({r[name] for r in pair}) == 1, f"matched-arm identity drift: {name}")

    if stage == "pilot":
        require(payload.get("pilot_authorization") is None, "pilot cannot consume a pilot authorization")
    else:
        authorization = payload.get("pilot_authorization")
        require(isinstance(authorization, Mapping), "confirmatory run lacks pilot authorization")
        require(_hash(authorization.get("summary_sha256")), "pilot authorization hash missing")
        require(authorization.get("verdict") == "PASS_PILOT_AUTHORIZES_CONFIRMATORY", "pilot did not authorize confirmatory")
        if pilot_summary_sha256 is not None:
            require(authorization["summary_sha256"] == pilot_summary_sha256, "confirmatory pilot binding drift")


def derive_verdict(payload: Mapping) -> dict[str, object]:
    try:
        validate_payload(payload)
    except IntegrityError as exc:
        return {"execution_integrity": "VOID_EXECUTION", "gate": "NOT_INTERPRETED", "reason": str(exc)}
    cells = payload["cells"]
    routed70 = [row for row in cells if row["route_mode"] == ROUTE_MODE and row["bars"] == 70]
    diagnostics = [row["v3_diagnostics"] for row in routed70]
    attempts = sum(row["plan_attempt_count"] for row in diagnostics)
    successes = sum(row["plan_success_count"] for row in diagnostics)
    accepted = sum(row["accepted_command_count"] for row in diagnostics)
    certificates = sum(row["accepted_terminal_stop_certificate_count"] for row in diagnostics)
    fallback = sum(row["fallback_interval_count"] for row in diagnostics)
    commanded = ENVS * STEPS * len(routed70)
    low = next(row for row in routed70 if row["speed_mps"] == 0.6)
    mechanism = (
        sum(row["runtime_replan_unsafe_start_count"] for row in diagnostics) == 0
        and sum(row["soft_envelope_exit_count"] for row in diagnostics) == 0
        and accepted > 0 and certificates == accepted
        and attempts > 0 and successes / attempts >= 0.99
        and fallback / commanded <= 0.01
        and low["v3_diagnostics"]["goal_completion_count"] / ENVS >= 0.50
        and all(row["mean_speed_ratio"] >= 0.80 for row in routed70)
    )
    physical = all(row["physical_pass"] for row in cells)
    passed = mechanism and physical
    stage = payload["stage"]
    return {
        "execution_integrity": f"PASS_{STAGES[stage]['cells']}_CELL_INTEGRITY",
        "gate": (
            "PASS_PILOT_AUTHORIZES_CONFIRMATORY" if stage == "pilot" and passed
            else "PASS_CONFIRMATORY_AUTHORIZES_SEPARATE_PPO_SMOKE" if passed
            else "FAIL_BLOCKS_CONFIRMATORY" if stage == "pilot"
            else "FAIL_BLOCKS_PPO"
        ),
        "mechanism_pass": mechanism,
        "all_corrected_r2_per_cell_gates_pass": physical,
        "inputs": {
            "runtime_replan_unsafe_start_count_70": sum(row["runtime_replan_unsafe_start_count"] for row in diagnostics),
            "soft_envelope_exit_count_70": sum(row["soft_envelope_exit_count"] for row in diagnostics),
            "terminal_certificate_fraction_70": certificates / accepted if accepted else 0.0,
            "plan_success_fraction_70": successes / attempts if attempts else 0.0,
            "fallback_interval_fraction_70": fallback / commanded,
            "goal_completions_per_env_70_speed_0p6": low["v3_diagnostics"]["goal_completion_count"] / ENVS,
        },
        "long_training_authorized": False,
    }


def verify_summary(path: Path, pilot_path: Path | None = None) -> int:
    require(path.is_file(), f"summary missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    pilot_sha = None
    if pilot_path is not None:
        require(pilot_path.is_file(), "pilot summary missing")
        pilot_payload = json.loads(pilot_path.read_text(encoding="utf-8"))
        validate_payload(pilot_payload)
        pilot_verdict = derive_verdict(pilot_payload)
        require(pilot_verdict["gate"] == "PASS_PILOT_AUTHORIZES_CONFIRMATORY", "pilot FAIL blocks confirmatory")
        pilot_sha = sha256_file(pilot_path)
    validate_payload(payload, pilot_sha)
    recomputed = derive_verdict(payload)
    require(payload.get("verdict") == recomputed, "stored verdict differs from recomputation")
    print(f"VERIFIED {path} {recomputed['gate']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--pilot-summary", type=Path)
    args = parser.parse_args(argv)
    try:
        return verify_summary(args.summary.resolve(), args.pilot_summary.resolve() if args.pilot_summary else None)
    except (IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"VOID_EXECUTION: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
