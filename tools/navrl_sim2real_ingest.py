#!/usr/bin/env python3
"""Convert a real-log CSV into the canonical sim-to-real telemetry JSONL contract.

This is intentionally a narrow, lossless adapter.  It does not infer frames, timestamps,
sequence numbers, or ``source_kind``.  A caller must provide ``--run-id`` and explicitly choose
``--source-kind real_log`` (or ``synthetic_fixture`` for a test fixture).  Unknown columns are
preserved as ``extra`` fields so a later profile/replay tool can use them without changing the
telemetry contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = (
    "topic",
    "seq",
    "source_stamp_ns",
    "host_receive_stamp_ns",
    "frame_id",
    "parent_frame_id",
)
OPTIONAL_COLUMNS = (
    "sync_group",
    "policy_input_stamp_ns",
    "command_publish_stamp_ns",
)
INTEGER_COLUMNS = set(REQUIRED_COLUMNS[1:4] + OPTIONAL_COLUMNS)


class IngestError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: str, *, column: str, row: int, allow_empty: bool = False) -> Optional[int]:
    value = value.strip()
    if not value and allow_empty:
        return None
    if not value:
        raise IngestError("row %d: %s is empty" % (row, column))
    try:
        return int(value, 10)
    except ValueError as exc:
        raise IngestError("row %d: %s must be an integer" % (row, column)) from exc


def _text(value: str, *, column: str, row: int) -> str:
    value = value.strip()
    if not value:
        raise IngestError("row %d: %s is empty" % (row, column))
    return value


def convert_csv(
    input_path: Path,
    output_path: Path,
    *,
    run_id: str,
    source_kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    if source_kind not in {"real_log", "synthetic_fixture"}:
        raise IngestError("source_kind must be real_log or synthetic_fixture")
    if not run_id.strip():
        raise IngestError("run_id must be non-empty")
    if not input_path.is_file():
        raise IngestError("input CSV does not exist: %s" % input_path)

    events: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise IngestError("input CSV has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise IngestError("CSV missing required columns: %s" % ", ".join(missing))
        known = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not str(value or "").strip() for value in row.values()):
                continue
            event: Dict[str, Any] = {}
            for column in REQUIRED_COLUMNS:
                if column in INTEGER_COLUMNS:
                    event[column] = _integer(str(row.get(column) or ""), column=column, row=row_number)
                else:
                    event[column] = _text(str(row.get(column) or ""), column=column, row=row_number)
            for column in OPTIONAL_COLUMNS:
                value = row.get(column)
                if value is not None and str(value).strip():
                    event[column] = _integer(str(value), column=column, row=row_number)
            extra = {
                key: value
                for key, value in row.items()
                if key and key not in known and value is not None and str(value).strip()
            }
            if extra:
                event["extra"] = extra
            events.append(event)
    if not events:
        raise IngestError("input CSV contains no event rows")

    manifest: Dict[str, Any] = {
        "record_type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "source_kind": source_kind,
        "run_id": run_id,
        "input_format": "csv",
        "input_sha256": _sha256(input_path),
        "event_count": len(events),
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return len(events)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-kind", required=True, choices=("real_log", "synthetic_fixture"))
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        metadata = None
        if args.metadata_json is not None:
            metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise IngestError("metadata JSON must contain an object")
        count = convert_csv(
            args.input_csv,
            args.output_jsonl,
            run_id=args.run_id,
            source_kind=args.source_kind,
            metadata=metadata,
        )
    except (IngestError, OSError, json.JSONDecodeError) as exc:
        print("ingest error: %s" % exc, file=sys.stderr)
        return 2
    claim = "SYNTHETIC_ONLY" if args.source_kind == "synthetic_fixture" else "MEASURED_CANDIDATE"
    print("converted %d events claim_status=%s output=%s" % (count, claim, args.output_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
