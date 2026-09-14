#!/usr/bin/env python3
"""Fail-closed timestamp/frame contract and offline telemetry summary.

The input is JSON Lines.  A run may start with one manifest record::

    {"record_type": "manifest", "schema_version": 1,
     "source_kind": "real_log", "run_id": "..."}

Every event then contains ``topic``, ``seq``, ``source_stamp_ns``,
``host_receive_stamp_ns``, ``frame_id`` and ``parent_frame_id``.  Sensor events may
also carry ``sync_group``.  Policy/command events may carry
``policy_input_stamp_ns`` and ``command_publish_stamp_ns``.

This tool deliberately does not invent a sensor noise model, modify a policy, or
claim sim-to-real validity.  It only checks source integrity and reports measured
latency/skew statistics.  ``fixture`` output is explicitly ``SYNTHETIC_ONLY``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
DEFAULT_TOPIC_FRAMES: Dict[str, Tuple[str, str]] = {
    "camera": ("camera_optical_frame", "base_link"),
    "lidar": ("lidar_frame", "base_link"),
    "ego_state": ("base_link", "odom"),
    "policy_input": ("base_link", "odom"),
    "command": ("base_link", "odom"),
}
DEFAULT_REQUIRED_TOPICS = tuple(DEFAULT_TOPIC_FRAMES)


class ContractError(ValueError):
    """Raised for malformed input or an invalid contract."""


@dataclass(frozen=True)
class TelemetryEvent:
    topic: str
    seq: int
    source_stamp_ns: int
    host_receive_stamp_ns: int
    frame_id: str
    parent_frame_id: str
    sync_group: Optional[int] = None
    policy_input_stamp_ns: Optional[int] = None
    command_publish_stamp_ns: Optional[int] = None
    row_number: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], row_number: int) -> "TelemetryEvent":
        required = (
            "topic",
            "seq",
            "source_stamp_ns",
            "host_receive_stamp_ns",
            "frame_id",
            "parent_frame_id",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ContractError(f"row {row_number}: missing event fields: {', '.join(missing)}")

        def integer(name: str, *, optional: bool = False) -> Optional[int]:
            value = raw.get(name)
            if value is None and optional:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"row {row_number}: {name} must be an integer")
            return int(value)

        def text(name: str) -> str:
            value = raw.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"row {row_number}: {name} must be a non-empty string")
            return value

        topic = text("topic")
        return cls(
            topic=topic,
            seq=int(integer("seq")),
            source_stamp_ns=int(integer("source_stamp_ns")),
            host_receive_stamp_ns=int(integer("host_receive_stamp_ns")),
            frame_id=text("frame_id"),
            parent_frame_id=text("parent_frame_id"),
            sync_group=integer("sync_group", optional=True),
            policy_input_stamp_ns=integer("policy_input_stamp_ns", optional=True),
            command_publish_stamp_ns=integer("command_publish_stamp_ns", optional=True),
            row_number=row_number,
        )


@dataclass(frozen=True)
class TelemetryContract:
    required_topics: Tuple[str, ...] = DEFAULT_REQUIRED_TOPICS
    topic_frames: Mapping[str, Tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_TOPIC_FRAMES)
    )
    max_sync_skew_ns: Optional[int] = None
    max_sensor_to_host_latency_ns: Optional[int] = None


@dataclass
class TelemetryReport:
    source_kind: str
    run_id: Optional[str]
    event_count: int
    topic_counts: Dict[str, int]
    metrics: Dict[str, Any]
    issues: List[Dict[str, Any]]
    verdict: str
    claim_status: str
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "claim_status": self.claim_status,
            "run_id": self.run_id,
            "event_count": self.event_count,
            "topic_counts": self.topic_counts,
            "metrics": self.metrics,
            "issues": self.issues,
            "verdict": self.verdict,
        }


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values_ns: Sequence[int]) -> Dict[str, Any]:
    values_ms = [value / 1_000_000.0 for value in values_ns]
    if not values_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "count": len(values_ms),
        "p50_ms": _percentile(values_ms, 50),
        "p95_ms": _percentile(values_ms, 95),
        "p99_ms": _percentile(values_ms, 99),
        "max_ms": max(values_ms),
    }


def _issue(code: str, message: str, **details: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def read_jsonl(path: Path) -> Tuple[Dict[str, Any], List[TelemetryEvent]]:
    if not path.is_file():
        raise ContractError(f"telemetry input does not exist: {path}")
    manifest: Dict[str, Any] = {}
    events: List[TelemetryEvent] = []
    with path.open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"row {row_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ContractError(f"row {row_number}: JSON value must be an object")
            if raw.get("record_type") == "manifest":
                if manifest:
                    raise ContractError(f"row {row_number}: duplicate manifest record")
                manifest = dict(raw)
                continue
            events.append(TelemetryEvent.from_mapping(raw, row_number))
    return manifest, events


def validate_events(
    events: Sequence[TelemetryEvent],
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    contract: TelemetryContract = TelemetryContract(),
    source_kind: Optional[str] = None,
) -> TelemetryReport:
    manifest = manifest or {}
    kind = str(source_kind or manifest.get("source_kind") or "real_log")
    run_id = manifest.get("run_id")
    issues: List[Dict[str, Any]] = []
    if not manifest:
        issues.append(_issue("MISSING_MANIFEST", "JSONL input has no manifest record"))
    elif manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue("MANIFEST_SCHEMA_MISMATCH", "manifest schema version differs from contract", observed=manifest.get("schema_version"), expected=SCHEMA_VERSION))
    topic_counts: Dict[str, int] = {}
    by_topic: Dict[str, List[TelemetryEvent]] = {}
    for event in events:
        topic_counts[event.topic] = topic_counts.get(event.topic, 0) + 1
        by_topic.setdefault(event.topic, []).append(event)
        if event.topic not in contract.topic_frames:
            issues.append(_issue("UNKNOWN_TOPIC", f"event topic is not in the contract: {event.topic}", row=event.row_number))

    for topic in contract.required_topics:
        if not by_topic.get(topic):
            issues.append(_issue("MISSING_TOPIC", f"required topic has no events: {topic}", topic=topic))

    source_to_host: Dict[str, List[int]] = {}
    policy_to_command: List[int] = []
    sync_groups: Dict[int, Dict[str, int]] = {}
    sequence_gaps: Dict[str, int] = {}
    for topic, topic_events in by_topic.items():
        last_source: Optional[int] = None
        last_host: Optional[int] = None
        last_seq: Optional[int] = None
        for event in sorted(topic_events, key=lambda item: item.row_number):
            expected_frame = contract.topic_frames.get(topic)
            if expected_frame is not None and (event.frame_id, event.parent_frame_id) != expected_frame:
                issues.append(
                    _issue(
                        "FRAME_CONTRACT_MISMATCH",
                        f"{topic} frame edge differs from contract",
                        row=event.row_number,
                        observed=[event.frame_id, event.parent_frame_id],
                        expected=list(expected_frame),
                    )
                )
            if last_source is not None and event.source_stamp_ns < last_source:
                issues.append(_issue("SOURCE_TIMESTAMP_REGRESSION", f"{topic} source timestamp regressed", row=event.row_number))
            if last_host is not None and event.host_receive_stamp_ns < last_host:
                issues.append(_issue("HOST_TIMESTAMP_REGRESSION", f"{topic} host timestamp regressed", row=event.row_number))
            if last_seq is not None:
                if event.seq <= last_seq:
                    issues.append(_issue("SEQUENCE_REGRESSION", f"{topic} sequence is not strictly increasing", row=event.row_number))
                elif event.seq > last_seq + 1:
                    sequence_gaps[topic] = sequence_gaps.get(topic, 0) + event.seq - last_seq - 1
            delay = event.host_receive_stamp_ns - event.source_stamp_ns
            if delay < 0:
                issues.append(_issue("NEGATIVE_SOURCE_TO_HOST_LATENCY", f"{topic} host received before source stamp", row=event.row_number))
            source_to_host.setdefault(topic, []).append(delay)
            if event.sync_group is not None:
                sync_groups.setdefault(event.sync_group, {})[topic] = event.source_stamp_ns
            if event.policy_input_stamp_ns is not None and event.command_publish_stamp_ns is not None:
                delta = event.command_publish_stamp_ns - event.policy_input_stamp_ns
                if delta < 0:
                    issues.append(_issue("NEGATIVE_POLICY_TO_COMMAND_LATENCY", "command precedes policy input", row=event.row_number))
                policy_to_command.append(delta)
            last_source = event.source_stamp_ns
            last_host = event.host_receive_stamp_ns
            last_seq = event.seq

    sync_skew: List[int] = []
    for sync_group, stamps in sorted(sync_groups.items()):
        if len(stamps) < 2:
            continue
        skew = max(stamps.values()) - min(stamps.values())
        sync_skew.append(skew)
        if contract.max_sync_skew_ns is not None and skew > contract.max_sync_skew_ns:
            issues.append(_issue("SYNC_SKEW_EXCEEDED", "synchronized sensor group exceeds skew gate", sync_group=sync_group, skew_ms=skew / 1_000_000.0))

    for topic, values in source_to_host.items():
        if contract.max_sensor_to_host_latency_ns is not None and topic in {"camera", "lidar", "ego_state"}:
            for value in values:
                if value > contract.max_sensor_to_host_latency_ns:
                    issues.append(_issue("SOURCE_TO_HOST_LATENCY_EXCEEDED", f"{topic} exceeds latency gate", topic=topic, latency_ms=value / 1_000_000.0))

    metrics: Dict[str, Any] = {
        "source_to_host_latency": {topic: _distribution(values) for topic, values in sorted(source_to_host.items())},
        "policy_to_command_latency": _distribution(policy_to_command),
        "sync_skew": _distribution(sync_skew),
        "sequence_gaps": sequence_gaps,
        "sync_groups": len(sync_groups),
    }
    verdict = "PASS" if events and not issues else ("FAIL" if issues else "EMPTY")
    claim_status = "SYNTHETIC_ONLY" if kind.startswith("synthetic") else "MEASURED_CANDIDATE"
    return TelemetryReport(kind, str(run_id) if run_id is not None else None, len(events), topic_counts, metrics, issues, verdict, claim_status)


def write_jsonl_fixture(path: Path, *, groups: int = 5) -> None:
    if groups < 1:
        raise ValueError("groups must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"record_type": "manifest", "schema_version": SCHEMA_VERSION, "source_kind": "synthetic_fixture", "run_id": "fixture"}) + "\n")
        topic_frames = DEFAULT_TOPIC_FRAMES
        for index in range(groups):
            sync_group = index
            base = index * 10_000_000
            rows = [
                ("camera", index, base, 1_000_000, topic_frames["camera"]),
                ("lidar", index, base + 2_000_000, 2_000_000, topic_frames["lidar"]),
                ("ego_state", index, base + 1_000_000, 1_500_000, topic_frames["ego_state"]),
                ("policy_input", index, base + 2_000_000, 4_000_000, topic_frames["policy_input"]),
            ]
            for topic, seq, source, receive_delay, (frame, parent) in rows:
                stream.write(json.dumps({"topic": topic, "seq": seq, "source_stamp_ns": source, "host_receive_stamp_ns": source + receive_delay, "frame_id": frame, "parent_frame_id": parent, "sync_group": sync_group if topic in {"camera", "lidar", "ego_state"} else None}) + "\n")
            stream.write(json.dumps({"topic": "command", "seq": index, "source_stamp_ns": base + 6_000_000, "host_receive_stamp_ns": base + 7_000_000, "frame_id": "base_link", "parent_frame_id": "odom", "policy_input_stamp_ns": base + 2_000_000, "command_publish_stamp_ns": base + 6_000_000}) + "\n")


def _load_contract(args: argparse.Namespace) -> TelemetryContract:
    max_skew = None if args.max_sync_skew_ms is None else int(args.max_sync_skew_ms * 1_000_000)
    max_latency = None if args.max_sensor_to_host_latency_ms is None else int(args.max_sensor_to_host_latency_ms * 1_000_000)
    return TelemetryContract(max_sync_skew_ns=max_skew, max_sensor_to_host_latency_ns=max_latency)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="write an explicitly synthetic JSONL fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--groups", type=int, default=5)
    validate = sub.add_parser("validate", help="validate JSONL and write a JSON report")
    validate.add_argument("input", type=Path)
    validate.add_argument("--report", type=Path)
    validate.add_argument("--source-kind", default=None)
    validate.add_argument("--max-sync-skew-ms", type=float, default=None)
    validate.add_argument("--max-sensor-to-host-latency-ms", type=float, default=None)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        write_jsonl_fixture(args.output, groups=args.groups)
        print(f"SYNTHETIC_ONLY fixture written: {args.output}")
        return 0
    try:
        manifest, events = read_jsonl(args.input)
        report = validate_events(events, manifest=manifest, contract=_load_contract(args), source_kind=args.source_kind)
    except (ContractError, OSError, ValueError) as exc:
        print(f"telemetry validation error: {exc}", file=sys.stderr)
        return 2
    output = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    print(f"telemetry verdict={report.verdict} claim_status={report.claim_status} events={report.event_count}", file=sys.stderr)
    return 0 if report.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
