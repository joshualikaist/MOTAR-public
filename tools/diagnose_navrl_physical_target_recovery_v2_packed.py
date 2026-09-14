#!/usr/bin/env python3
"""Evaluation-only packed-telemetry diagnosis for a finalized recovery-v2 gate.

Reads an already-verified gate directory. It does not launch GPU, alter the receipt,
change gates, retune the controller, or authorize training.

Packed status 19 is generic ``recovery_no_connector``. The labels below are descriptive
classes from state/margin at interval start, not a new reason code or a loosened gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STATE_NORMAL = 0
STATE_BRAKE = 1
STATE_CONNECT = 2
STATE_ROUTE = 3
STATE_NO_CONNECTOR = 4

STATUS_NO_CONNECTOR = 19
STATUS_HARD_BREACH = 20
STATUS_LOCAL_INFEASIBLE_SOFT_FREE = 21
STATUS_BRAKE_TIMEOUT = 22
STATUS_CONNECT_TIMEOUT = 23

HEADING_VALID_SPEED_MPS = 0.10
ACCEL_ENVELOPE_MPS2 = 4.0
INTERVAL_S = 0.10
ENVELOPE_DV_MPS = ACCEL_ENVELOPE_MPS2 * INTERVAL_S

STATE_NAME = {
    STATE_NORMAL: "normal",
    STATE_BRAKE: "brake",
    STATE_CONNECT: "connect",
    STATE_ROUTE: "route",
    STATE_NO_CONNECTOR: "no_connector",
}


def _unit(xy: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(xy, axis=-1, keepdims=True)
    return xy / np.clip(norm, 1e-9, None)


def classify_no_connector_entry(
    state_before: int,
    status_after: int,
    soft_margin_before_m: float,
) -> str:
    """Map one NO_CONNECTOR transition onto a packed-telemetry class."""
    if int(status_after) == STATUS_HARD_BREACH:
        return "hard_breach"
    if int(status_after) == STATUS_LOCAL_INFEASIBLE_SOFT_FREE:
        return "local_infeasible_soft_free"
    if int(status_after) == STATUS_BRAKE_TIMEOUT:
        return "brake_timeout"
    if int(status_after) == STATUS_CONNECT_TIMEOUT:
        return "connect_timeout"
    if int(status_after) != STATUS_NO_CONNECTOR:
        return "unexpected_status_%s" % int(status_after)
    before = int(state_before)
    if before == STATE_BRAKE:
        return "brake_no_anchor_likely"
    if before == STATE_CONNECT:
        if float(soft_margin_before_m) > 0.0:
            return "connect_failed_resume_likely"
        return "connect_failed_certificate_likely"
    if before == STATE_NORMAL:
        return "same_interval_brake_no_anchor_likely"
    if before == STATE_ROUTE:
        return "route_to_no_connector"
    return "other"


def connect_tracking(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    active = payload["state_after"] == STATE_CONNECT
    count = int(active.sum())
    empty = {
        "connect_intervals": count,
        "command_toward_anchor_fraction": None,
        "realized_toward_anchor_fraction": None,
        "mean_command_mps": None,
        "mean_realized_mps": None,
        "mean_tracking_error_mps": None,
        "rest_fraction": None,
        "rest_command_toward_anchor_fraction": None,
        "rest_realized_toward_anchor_fraction": None,
        "rest_mean_command_mps": None,
        "rest_mean_realized_mps": None,
        "rest_realized_over_envelope_dv": None,
        "selected_candidate_index_counts": {},
        "horizon_min_m": None,
        "actual_regression_count": 0,
        "actual_max_increase_m": 0.0,
    }
    if count == 0:
        return empty
    command = np.asarray(payload["command_xy"])[active]
    realized = np.asarray(payload["velocity_after_xy"])[active]
    before_vel = np.asarray(payload["velocity_before_xy"])[active]
    toward = np.asarray(payload["anchor_xy"])[active] - np.asarray(
        payload["position_before_xy"]
    )[active]
    command_dot = (_unit(command) * _unit(toward)).sum(-1)
    realized_dot = (_unit(realized) * _unit(toward)).sum(-1)
    command_speed = np.linalg.norm(command, axis=-1)
    realized_speed = np.linalg.norm(realized, axis=-1)
    before_speed = np.linalg.norm(before_vel, axis=-1)
    rest = before_speed < HEADING_VALID_SPEED_MPS
    actual = (
        np.asarray(payload["anchor_distance_after_m"])[active]
        - np.asarray(payload["anchor_distance_m"])[active]
    )
    selected = np.asarray(payload["candidate_selected_index"])[active]
    selected_counts = {int(k): int(v) for k, v in zip(*np.unique(selected, return_counts=True))}
    horizon = np.asarray(payload["planned_horizon_progress_m"])[active]
    result = dict(empty)
    result.update({
        "connect_intervals": count,
        "command_toward_anchor_fraction": float((command_dot > 0.0).mean()),
        "realized_toward_anchor_fraction": float((realized_dot > 0.0).mean()),
        "mean_command_mps": float(command_speed.mean()),
        "mean_realized_mps": float(realized_speed.mean()),
        "mean_tracking_error_mps": float(np.linalg.norm(realized - command, axis=-1).mean()),
        "rest_fraction": float(rest.mean()),
        "selected_candidate_index_counts": selected_counts,
        "horizon_min_m": float(horizon.min()) if horizon.size else None,
        "actual_regression_count": int((actual > 1e-5).sum()),
        "actual_max_increase_m": float(actual.max()) if actual.size else 0.0,
    })
    if bool(rest.any()):
        result["rest_command_toward_anchor_fraction"] = float((command_dot[rest] > 0.0).mean())
        result["rest_realized_toward_anchor_fraction"] = float((realized_dot[rest] > 0.0).mean())
        result["rest_mean_command_mps"] = float(command_speed[rest].mean())
        result["rest_mean_realized_mps"] = float(realized_speed[rest].mean())
        result["rest_realized_over_envelope_dv"] = float(
            realized_speed[rest].mean() / ENVELOPE_DV_MPS
        )
    return result


def occupancy(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    after = np.asarray(payload["state_after"])
    counts = {name: 0 for name in STATE_NAME.values()}
    for code, name in STATE_NAME.items():
        counts[name] = int((after == code).sum())
    counts["intervals"] = int(after.size)
    return counts


def no_connector_entries(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    before = np.asarray(payload["state_before"])
    after = np.asarray(payload["state_after"])
    status = np.asarray(payload["status_after"])
    soft = np.asarray(payload["soft_margin_before_m"])
    enter = (after == STATE_NO_CONNECTOR) & (before != STATE_NO_CONNECTOR)
    tallies: Counter[str] = Counter()
    for state, code, margin in zip(before[enter], status[enter], soft[enter]):
        tallies[classify_no_connector_entry(int(state), int(code), float(margin))] += 1
    tallies["entries"] = int(enter.sum())
    return dict(tallies)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as packed:
        return {name: packed[name] for name in packed.files}


def diagnose_gate(gate_dir: Path) -> dict[str, Any]:
    gate_dir = gate_dir.resolve()
    receipt = json.loads((gate_dir / "receipt.json").read_text(encoding="utf-8"))
    summary = json.loads((gate_dir / "summary.json").read_text(encoding="utf-8"))
    verdict = receipt.get("verdict") or {}
    cells = []
    pooled_entries: Counter[str] = Counter()
    pooled_occ: Counter[str] = Counter()
    pool70_attempts = 0
    pool70_successes = 0
    pool70_fallback = 0
    pool70_goals = 0
    for cell in summary["cells"]:
        record = {
            "record_id": cell["record_id"],
            "route_mode": cell["route_mode"],
            "speed_mps": cell["speed_mps"],
            "bars": cell["bars"],
            "pass": cell["pass"],
            "failed_gates": sorted(name for name, ok in cell["gates"].items() if not ok),
            "mean_speed_ratio": cell.get("mean_speed_ratio"),
            "plan_success_fraction": cell["route"].get("plan_success_fraction"),
            "fallback_interval_fraction": cell["route"].get("fallback_interval_fraction"),
            "goal_completions_per_env": cell["route"].get("goal_completions_per_env"),
        }
        if cell["route_mode"] != "off":
            payload = _load_npz(gate_dir / "raw" / cell["telemetry"]["path"])
            record["occupancy"] = occupancy(payload)
            record["no_connector_classes"] = no_connector_entries(payload)
            record["connect_tracking"] = connect_tracking(payload)
            pooled_entries.update(
                {k: v for k, v in record["no_connector_classes"].items() if k != "entries"}
            )
            pooled_entries["entries"] += record["no_connector_classes"]["entries"]
            pooled_occ.update(record["occupancy"])
            delta = cell["route"].get("counter_delta") or {}
            if int(cell["bars"]) == 70:
                pool70_attempts += int(delta.get("plan_attempts") or 0)
                pool70_successes += int(delta.get("plan_successes") or 0)
                pool70_fallback += int(delta.get("fallback_intervals") or 0)
                pool70_goals += int(delta.get("goal_completions") or 0)
        cells.append(record)
    recovery_cells = [row for row in cells if row["route_mode"] != "off"]
    off_pass = sum(1 for row in cells if row["route_mode"] == "off" and row["pass"])
    return {
        "schema": "navrl_physical_target_recovery_v2_packed_forensics_v1",
        "gate_dir": str(gate_dir),
        "execution_integrity": verdict.get("execution_integrity"),
        "route_mechanism": verdict.get("route_mechanism"),
        "long_training_authorized": verdict.get("long_training_authorized"),
        "cells_pass": sum(1 for row in cells if row["pass"]),
        "cells_total": len(cells),
        "off_cells_pass": off_pass,
        "recovery_cells_pass": sum(1 for row in recovery_cells if row["pass"]),
        "recovery_cells_total": len(recovery_cells),
        "mechanism_pool_70bar": {
            "plan_attempts": pool70_attempts,
            "plan_successes": pool70_successes,
            "plan_success_fraction": (
                pool70_successes / pool70_attempts if pool70_attempts else None
            ),
            "fallback_intervals": pool70_fallback,
            "fallback_interval_fraction": pool70_fallback / 38400.0 if pool70_fallback else None,
            "goal_completions": pool70_goals,
        },
        "pooled_no_connector_classes": dict(pooled_entries),
        "pooled_occupancy": dict(pooled_occ),
        "claim_boundaries": [
            "Does not repair or supersede the canonical 1.5 m/s contract.",
            "Does not authorize PPO, gain/0.45 retuning, env-count change, or a 32-cell rerun.",
            "Packed class labels are descriptive; they are not new status codes.",
        ],
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = diagnose_gate(args.gate_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
