"""Validate P4 candidate JSONL against its receipt, JSON Schema, and semantic constraints."""

import argparse
import json
from pathlib import Path

from perception_candidates import read_jsonl, sha256_file, validate_candidate_record


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA = REPOSITORY / "docs" / "specs" / "motar_perception_candidates_v1.schema.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    receipt = json.loads(args.receipt.read_text())
    if sha256_file(args.candidates) != receipt["output_sha256"]:
        raise SystemExit("[candidate-verify] output hash mismatch")
    if sha256_file(SCHEMA) != receipt["candidate_schema_sha256"]:
        raise SystemExit("[candidate-verify] schema hash mismatch")
    if args.manifest and sha256_file(args.manifest) != receipt["manifest_sha256"]:
        raise SystemExit("[candidate-verify] manifest hash mismatch")
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))
    except ImportError:
        validator = None
    previous = {}
    rows = 0
    candidate_rows = read_jsonl(args.candidates)
    manifest_rows = read_jsonl(args.manifest) if args.manifest else None
    for rows, candidate in enumerate(candidate_rows, 1):
        if validator is not None:
            validator.validate(candidate)
        validate_candidate_record(candidate, previous)
        if manifest_rows is not None:
            try:
                source = next(manifest_rows)
            except StopIteration:
                raise SystemExit("[candidate-verify] candidates outnumber manifest")
            expected = (source["frame_id"], source["source_sequence_id"], source["frame_index"],
                        source["capture_timestamp_ns"])
            observed = (candidate["frame_id"], candidate["source_sequence_id"],
                        candidate["frame_index"], candidate["capture_timestamp_ns"])
            if observed != expected:
                raise SystemExit("[candidate-verify] manifest order/identity mismatch")
    if rows != receipt["records"]:
        raise SystemExit("[candidate-verify] record count mismatch")
    if manifest_rows is not None:
        try:
            next(manifest_rows)
            raise SystemExit("[candidate-verify] manifest outnumbers candidates")
        except StopIteration:
            pass
    print("[candidate-verify] PASS: %d records" % rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
