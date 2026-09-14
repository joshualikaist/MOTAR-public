#!/usr/bin/env python3
"""Fail-closed validator for the two-zone target-token replay contract.

The contract is JSON (not YAML) so this tool stays dependency-free on the flight computer.  A
real profile must choose ``near_boundary_m`` before a replay is accepted; this validator never
chooses that boundary.  In the far zone, range is explicitly invalid rather than silently
encoded as zero.  In the near zone, a dropout is allowed but must still carry ``range_valid=0``
and no numeric range value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "trial_id",
    "timestamp_ns",
    "zone",
    "azimuth_deg",
    "elevation_deg",
    "bearing_rate_dps",
    "confidence",
    "range_valid",
    "range_m",
    "range_sigma_m",
    "measurement_age_ms",
    "track_covariance",
)


class ReplayError(ValueError):
    pass


def _finite(value: Any, *, name: str, row: int, optional: bool = False) -> Optional[float]:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ReplayError("row %d: %s must be numeric" % (row, name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError("row %d: %s must be numeric" % (row, name)) from exc
    if not math.isfinite(number):
        raise ReplayError("row %d: %s must be finite" % (row, name))
    return number


def _bool(value: Any, *, name: str, row: int) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"0", "1", "true", "false"}:
        return value.strip().lower() in {"1", "true"}
    raise ReplayError("row %d: %s must be boolean" % (row, name))


def _timestamp(value: Any, *, row: int) -> int:
    if isinstance(value, bool):
        raise ReplayError("row %d: timestamp_ns must be an integer" % row)
    try:
        stamp = int(str(value), 10)
    except (TypeError, ValueError) as exc:
        raise ReplayError("row %d: timestamp_ns must be an integer" % row) from exc
    return stamp


def load_contract(path: Path) -> Dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError("cannot read contract: %s" % exc) from exc
    if not isinstance(contract, dict):
        raise ReplayError("contract must be a JSON object")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ReplayError("unsupported contract schema_version")
    if contract.get("source_kind") != "real_log":
        raise ReplayError("two-zone contract requires source_kind=real_log")
    boundary = _finite(contract.get("near_boundary_m"), name="near_boundary_m", row=0)
    if boundary is None or boundary <= 0.0:
        raise ReplayError("near_boundary_m must be positive")
    if contract.get("far_range_policy") != "invalid":
        raise ReplayError("far_range_policy must be invalid")
    return contract


def validate_replay(input_path: Path, contract_path: Path) -> Dict[str, Any]:
    contract = load_contract(contract_path)
    if not input_path.is_file():
        raise ReplayError("replay input does not exist: %s" % input_path)
    events: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    last_stamp: Dict[str, int] = {}
    with input_path.open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError("row %d: invalid JSON: %s" % (row_number, exc.msg)) from exc
            if not isinstance(raw, dict):
                raise ReplayError("row %d: event must be an object" % row_number)
            if raw.get("record_type") == "manifest":
                continue
            missing = [field for field in REQUIRED_FIELDS if field not in raw]
            if missing:
                raise ReplayError("row %d: missing fields: %s" % (row_number, ", ".join(missing)))
            trial_id = str(raw["trial_id"]).strip()
            zone = str(raw["zone"]).strip().lower()
            if not trial_id or zone not in {"near", "far"}:
                raise ReplayError("row %d: trial_id and zone=near|far are required" % row_number)
            # Do not parse nanosecond clocks through float: a float loses integer precision above
            # 2^53, which is already below ordinary Unix-nanosecond timestamps.
            stamp = _timestamp(raw["timestamp_ns"], row=row_number)
            if trial_id in last_stamp and stamp < last_stamp[trial_id]:
                issues.append({"code": "TIMESTAMP_REGRESSION", "row": row_number, "trial_id": trial_id})
            last_stamp[trial_id] = stamp
            confidence = _finite(raw["confidence"], name="confidence", row=row_number)
            if confidence < 0.0 or confidence > 1.0:
                issues.append({"code": "CONFIDENCE_OUT_OF_RANGE", "row": row_number})
            age = _finite(raw["measurement_age_ms"], name="measurement_age_ms", row=row_number)
            if age < 0.0:
                issues.append({"code": "NEGATIVE_MEASUREMENT_AGE", "row": row_number})
            covariance = _finite(raw["track_covariance"], name="track_covariance", row=row_number)
            if covariance < 0.0:
                issues.append({"code": "NEGATIVE_TRACK_COVARIANCE", "row": row_number})
            valid = _bool(raw["range_valid"], name="range_valid", row=row_number)
            range_value = _finite(raw["range_m"], name="range_m", row=row_number, optional=True)
            sigma = _finite(raw["range_sigma_m"], name="range_sigma_m", row=row_number, optional=True)
            if valid and (range_value is None or sigma is None or range_value <= 0.0 or sigma < 0.0):
                issues.append({"code": "INVALID_RANGE_PAYLOAD", "row": row_number})
            if not valid and (range_value is not None or sigma is not None):
                issues.append({"code": "INVALID_RANGE_NOT_NULL", "row": row_number})
            if zone == "far" and valid:
                issues.append({"code": "FAR_RANGE_MUST_BE_INVALID", "row": row_number})
            events.append(raw)
    if not events:
        issues.append({"code": "EMPTY_REPLAY"})
    verdict = "PASS" if events and not issues else ("FAIL" if issues else "EMPTY")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract_path.resolve()),
        "input": str(input_path.resolve()),
        "event_count": len(events),
        "trial_count": len(last_stamp),
        "near_boundary_m": contract["near_boundary_m"],
        "issues": issues,
        "verdict": verdict,
        "claim_status": "MEASURED_CANDIDATE" if verdict == "PASS" else "NOT_VALIDATED",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_jsonl", type=Path)
    parser.add_argument("contract_json", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_replay(args.replay_jsonl, args.contract_json)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ReplayError, OSError, ValueError) as exc:
        print("two-zone replay error: %s" % exc, file=sys.stderr)
        return 2
    print("two-zone replay verdict=%s events=%d" % (report["verdict"], report["event_count"]))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
