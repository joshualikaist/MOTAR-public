#!/usr/bin/env python3
"""Build a trial-level sensor profile from held-out measurement CSV.

The CSV is deliberately separate from the transport telemetry JSONL: it contains paired sensor
and ground-truth measurements after the recorder has joined them by timestamp.  This tool reports
diagnostic distributions only; it never turns frame counts into independent trials and never
chooses a PPO noise parameter.  A real-log profile is a ``MEASURED_CANDIDATE`` until its trial
manifest, calibration hashes, split, and quality gates are reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REQUIRED_COLUMNS = (
    "trial_id",
    "distance_m",
    "lighting",
    "motion",
    "target_present",
    "detected",
    "range_valid",
    "ground_truth_azimuth_deg",
    "estimated_azimuth_deg",
    "ground_truth_range_m",
    "estimated_range_m",
    "confidence",
    "source_stamp_ns",
    "host_receive_stamp_ns",
)


class ProfileError(ValueError):
    pass


def _number(value: str, *, name: str, row: int, allow_empty: bool = False) -> Optional[float]:
    value = value.strip()
    if not value and allow_empty:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ProfileError("row %d: %s must be numeric" % (row, name)) from exc
    if not math.isfinite(result):
        raise ProfileError("row %d: %s must be finite" % (row, name))
    return result


def _integer(value: str, *, name: str, row: int) -> int:
    try:
        result = int(value.strip(), 10)
    except ValueError as exc:
        raise ProfileError("row %d: %s must be an integer" % (row, name)) from exc
    return result


def _boolean(value: str, *, name: str, row: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ProfileError("row %d: %s must be 0/1 or true/false" % (row, name))


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values) if values else None,
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _cell_key(row: Mapping[str, Any]) -> Tuple[float, str, str]:
    return (float(row["distance_m"]), str(row["lighting"]), str(row["motion"]))


def read_measurements(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise ProfileError("measurement CSV does not exist: %s" % path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ProfileError("measurement CSV has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ProfileError("measurement CSV missing columns: %s" % ", ".join(missing))
        last_stamp: Dict[str, int] = {}
        last_host: Dict[str, int] = {}
        for row_number, raw in enumerate(reader, start=2):
            if not raw or all(not str(value or "").strip() for value in raw.values()):
                continue
            trial_id = str(raw.get("trial_id") or "").strip()
            lighting = str(raw.get("lighting") or "").strip()
            motion = str(raw.get("motion") or "").strip()
            if not trial_id or not lighting or not motion:
                raise ProfileError("row %d: trial_id, lighting and motion are required" % row_number)
            target_present = _boolean(str(raw.get("target_present") or ""), name="target_present", row=row_number)
            detected = _boolean(str(raw.get("detected") or ""), name="detected", row=row_number)
            range_valid = _boolean(str(raw.get("range_valid") or ""), name="range_valid", row=row_number)
            gt_az = _number(str(raw.get("ground_truth_azimuth_deg") or ""), name="ground_truth_azimuth_deg", row=row_number, allow_empty=True)
            est_az = _number(str(raw.get("estimated_azimuth_deg") or ""), name="estimated_azimuth_deg", row=row_number, allow_empty=True)
            gt_range = _number(str(raw.get("ground_truth_range_m") or ""), name="ground_truth_range_m", row=row_number, allow_empty=True)
            est_range = _number(str(raw.get("estimated_range_m") or ""), name="estimated_range_m", row=row_number, allow_empty=True)
            if target_present and (gt_az is None or gt_range is None):
                raise ProfileError("row %d: target-present rows need ground truth azimuth and range" % row_number)
            if detected and est_az is None:
                raise ProfileError("row %d: detected rows need estimated azimuth" % row_number)
            if range_valid and est_range is None:
                raise ProfileError("row %d: range-valid rows need estimated range" % row_number)
            source_stamp = _integer(str(raw.get("source_stamp_ns") or ""), name="source_stamp_ns", row=row_number)
            host_stamp = _integer(str(raw.get("host_receive_stamp_ns") or ""), name="host_receive_stamp_ns", row=row_number)
            if trial_id in last_stamp and source_stamp < last_stamp[trial_id]:
                raise ProfileError("row %d: source timestamp regressed within trial %s" % (row_number, trial_id))
            if trial_id in last_host and host_stamp < last_host[trial_id]:
                raise ProfileError("row %d: host timestamp regressed within trial %s" % (row_number, trial_id))
            last_stamp[trial_id] = source_stamp
            last_host[trial_id] = host_stamp
            rows.append(
                {
                    "trial_id": trial_id,
                    "distance_m": _number(str(raw.get("distance_m") or ""), name="distance_m", row=row_number),
                    "lighting": lighting,
                    "motion": motion,
                    "target_present": target_present,
                    "detected": detected,
                    "range_valid": range_valid,
                    "gt_az": gt_az,
                    "est_az": est_az,
                    "gt_range": gt_range,
                    "est_range": est_range,
                    "confidence": _number(str(raw.get("confidence") or ""), name="confidence", row=row_number),
                    "source_stamp_ns": source_stamp,
                    "host_receive_stamp_ns": host_stamp,
                }
            )
    if not rows:
        raise ProfileError("measurement CSV contains no rows")
    return rows


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    trials: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        trials.setdefault(str(row["trial_id"]), []).append(row)
    trial_summaries: List[Dict[str, Any]] = []
    for trial_id, trial_rows in sorted(trials.items()):
        tp = sum(1 for row in trial_rows if row["target_present"] and row["detected"])
        fn = sum(1 for row in trial_rows if row["target_present"] and not row["detected"])
        fp = sum(1 for row in trial_rows if not row["target_present"] and row["detected"])
        bearing = [abs(float(row["est_az"]) - float(row["gt_az"])) for row in trial_rows if row["detected"] and row["gt_az"] is not None and row["est_az"] is not None]
        ranges = [abs(float(row["est_range"]) - float(row["gt_range"])) for row in trial_rows if row["range_valid"] and row["gt_range"] is not None and row["est_range"] is not None]
        delays = [(int(row["host_receive_stamp_ns"]) - int(row["source_stamp_ns"])) / 1_000_000.0 for row in trial_rows]
        trial_summaries.append(
            {
                "trial_id": trial_id,
                "distance_m": float(trial_rows[0]["distance_m"]),
                "lighting": trial_rows[0]["lighting"],
                "motion": trial_rows[0]["motion"],
                "frames": len(trial_rows),
                "detection_recall": tp / float(tp + fn) if tp + fn else None,
                "false_positive_rate": fp / float(fp + sum(1 for row in trial_rows if not row["target_present"])) if any(not row["target_present"] for row in trial_rows) else None,
                "range_valid_fraction": sum(1 for row in trial_rows if row["range_valid"]) / float(sum(1 for row in trial_rows if row["target_present"])) if any(row["target_present"] for row in trial_rows) else None,
                "bearing_abs_error_deg": _distribution(bearing),
                "range_abs_error_m": _distribution(ranges),
                "source_to_host_latency_ms": _distribution(delays),
            }
        )
    cells: Dict[Tuple[float, str, str], List[Dict[str, Any]]] = {}
    for trial in trial_summaries:
        cells.setdefault((trial["distance_m"], trial["lighting"], trial["motion"]), []).append(trial)
    cell_summaries: List[Dict[str, Any]] = []
    for key, cell_trials in sorted(cells.items()):
        recalls = [trial["detection_recall"] for trial in cell_trials if trial["detection_recall"] is not None]
        valid = [trial["range_valid_fraction"] for trial in cell_trials if trial["range_valid_fraction"] is not None]
        latency = [trial["source_to_host_latency_ms"]["p95"] for trial in cell_trials if trial["source_to_host_latency_ms"]["p95"] is not None]
        cell_summaries.append(
            {
                "distance_m": key[0],
                "lighting": key[1],
                "motion": key[2],
                "trial_count": len(cell_trials),
                "trial_macro_detection_recall": _mean(recalls),
                "trial_macro_range_valid_fraction": _mean(valid),
                "trial_p95_latency_ms_mean": _mean(latency),
            }
        )
    return {"trial_count": len(trial_summaries), "trials": trial_summaries, "cells": cell_summaries}


def build_profile(input_path: Path, output_path: Path, *, source_kind: str, run_id: str) -> Dict[str, Any]:
    if source_kind not in {"real_log", "synthetic_fixture"}:
        raise ProfileError("source_kind must be real_log or synthetic_fixture")
    rows = read_measurements(input_path)
    profile = {
        "schema_version": 1,
        "source_kind": source_kind,
        "claim_status": "SYNTHETIC_ONLY" if source_kind == "synthetic_fixture" else "MEASURED_CANDIDATE",
        "run_id": run_id,
        "input": str(input_path.resolve()),
        "frame_count": len(rows),
        "profile": _summarize(rows),
        "quality": {
            "trial_unit_reporting": True,
            "ground_truth_required": True,
            "threshold_decision": "NOT_APPLIED",
            "note": "This profile does not select simulator noise or a two-zone boundary.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return profile


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-kind", required=True, choices=("real_log", "synthetic_fixture"))
    args = parser.parse_args(argv)
    try:
        profile = build_profile(args.input_csv, args.output_json, source_kind=args.source_kind, run_id=args.run_id)
    except (ProfileError, OSError) as exc:
        print("sensor profile error: %s" % exc, file=sys.stderr)
        return 2
    print("profiled %d frames / %d trials claim_status=%s output=%s" % (profile["frame_count"], profile["profile"]["trial_count"], profile["claim_status"], args.output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
