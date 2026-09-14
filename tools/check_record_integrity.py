#!/usr/bin/env python3
"""Read-only, stdlib JSON record-envelope checks. No model/task imports or metric analysis."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

MAX_BYTES = 64 * 1024**2
TYPES = {"integer", "number", "string", "boolean"}
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class StrictJSONError(ValueError):
    pass


def strict_loads(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StrictJSONError("DUPLICATE_OBJECT_KEY")
            result[key] = value
        return result

    def constant(_):
        raise StrictJSONError("NONSTANDARD_NUMERIC_TOKEN")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def validate_contract(contract):
    allowed = {"schema_version", "records_key", "expected_count", "fields", "unique_fields", "context_key", "context_fields"}
    if (not isinstance(contract, dict) or set(contract)-allowed
            or type(contract.get("schema_version")) is not int or contract["schema_version"] != 1):
        raise ValueError("Unsupported contract")
    if not isinstance(contract.get("records_key"), str) or not contract["records_key"]:
        raise ValueError("records_key is required")
    count = contract.get("expected_count")
    if type(count) is not int or count < 0:
        raise ValueError("Explicit nonnegative expected_count required")
    fields = contract.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Explicit envelope fields required")
    for name, spec in fields.items():
        if not isinstance(name, str) or not isinstance(spec, dict) or set(spec)-{"type", "nullable", "minimum", "format"}:
            raise ValueError("Invalid field contract")
        if spec.get("type") not in TYPES or type(spec.get("nullable", False)) is not bool:
            raise ValueError("Invalid field type/nullability")
        if "format" in spec and (spec["format"] != "sha256" or spec["type"] != "string"):
            raise ValueError("Only sha256 string format supported")
        if "minimum" in spec and (spec["type"] not in {"integer", "number"}
                or type(spec["minimum"]) not in {int, float} or not math.isfinite(spec["minimum"])):
            raise ValueError("Invalid numeric minimum")
    unique = contract.get("unique_fields")
    if not isinstance(unique, list) or not unique or len(unique) != len(set(unique)) or not set(unique) <= set(fields):
        raise ValueError("Unique keys must name contracted fields")
    if any(fields[key].get("nullable", False) for key in unique):
        raise ValueError("Identity cannot be nullable")
    context = contract.get("context_fields", [])
    if not isinstance(context, list) or not set(context) <= set(fields):
        raise ValueError("Invalid context fields")
    if context and (not isinstance(contract.get("context_key"), str) or not contract["context_key"]):
        raise ValueError("context_key required")


def issue_for(value, spec):
    if value is None:
        return None if spec.get("nullable", False) else "NULL_REQUIRED_VALUE"
    kind = spec["type"]
    valid = ((kind == "integer" and type(value) is int)
             or (kind == "number" and type(value) in (int, float))
             or (kind == "string" and type(value) is str)
             or (kind == "boolean" and type(value) is bool))
    if not valid:
        return "WRONG_TYPE"
    if type(value) is float and not math.isfinite(value):
        return "NONFINITE_VALUE"
    if "minimum" in spec and value < spec["minimum"]:
        return "BELOW_MINIMUM"
    if spec.get("format") == "sha256" and not SHA256.fullmatch(value):
        return "INVALID_SHA256_FORMAT"
    return None


def audit_bytes(data, contract):
    """Values never enter the report; only field names, row positions and issue counts do."""
    validate_contract(contract)
    if len(data) > MAX_BYTES:
        raise ValueError("Input exceeds 64 MiB safety bound")
    report = {"scope": "Record envelope only; no domain metrics, labels or outcomes evaluated",
              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "record_count": None,
              "expected_count": contract["expected_count"], "issues": [], "issue_counts": {},
              "coverage": {}, "duplicate_records": 0, "nonfinite_values": 0}
    counts = Counter()

    def issue(code, location):
        counts[code] += 1
        if len(report["issues"]) < 40:
            report["issues"].append({"code": code, "location": location})

    def done():
        report["issue_counts"] = dict(sorted(counts.items()))
        report["issue_count"] = sum(counts.values())
        report["issues_truncated"] = report["issue_count"] > len(report["issues"])
        report["status"] = "INVALID_FOR_DECLARED_CONTRACT" if counts else "VALID_FOR_DECLARED_CONTRACT"
        return report

    try:
        document = strict_loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        code = str(error) if isinstance(error, StrictJSONError) else "MALFORMED_JSON"
        issue(code, "document")
        return done()
    # Walk iteratively so all numerical leaves, including uncontracted fields, are checked.
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif type(value) is float and not math.isfinite(value):
            report["nonfinite_values"] += 1
    if report["nonfinite_values"]:
        issue("NONFINITE_DOCUMENT", "document")
    if not isinstance(document, dict) or not isinstance(document.get(contract["records_key"]), list):
        issue("MISSING_RECORD_ARRAY", "document")
        return done()
    rows = document[contract["records_key"]]
    report["record_count"] = len(rows)
    if len(rows) != contract["expected_count"]:
        issue("RECORD_COUNT_MISMATCH", contract["records_key"])
    context = document.get(contract.get("context_key", ""), {})
    if not isinstance(context, dict):
        issue("INVALID_CONTEXT", "context")
        context = {}
    for name in contract.get("context_fields", []):
        error = "MISSING_FIELD" if name not in context else issue_for(context[name], contract["fields"][name])
        if error:
            issue(error, "context."+name)
    seen = set()
    coverage = {k: {"missing": 0, "null": 0, "invalid": 0} for k in contract["fields"]}
    for index, row in enumerate(rows):
        prefix = "records[%d]" % index
        if not isinstance(row, dict):
            issue("RECORD_NOT_OBJECT", prefix)
            for stat in coverage.values():
                stat["missing"] += 1
            continue
        for name, spec in contract["fields"].items():
            stat = coverage[name]
            if name not in row:
                stat["missing"] += 1
                issue("MISSING_FIELD", prefix+"."+name)
                continue
            stat["null"] += row[name] is None
            error = issue_for(row[name], spec)
            if error:
                stat["invalid"] += 1
                issue(error, prefix+"."+name)
        keys = contract["unique_fields"]
        if all(key in row and issue_for(row[key], contract["fields"][key]) is None for key in keys):
            identity = tuple(row[key] for key in keys)
            if identity in seen:
                report["duplicate_records"] += 1
                issue("DUPLICATE_IDENTITY", prefix)
            seen.add(identity)
        for name in contract.get("context_fields", []):
            if name in row and name in context and (type(row[name]) is not type(context[name]) or row[name] != context[name]):
                issue("CONTEXT_MISMATCH", prefix+"."+name)
    report["coverage"] = coverage
    return done()


def git_pin(path, repository, revision, observed_sha):
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    commit = subprocess.check_output(["git", "rev-parse", "--verify", revision+"^{commit}"],
                                     cwd=repository, text=True).strip()
    original = subprocess.check_output(["git", "show", commit+":"+relative], cwd=repository)
    expected = hashlib.sha256(original).hexdigest()
    return {"commit": commit, "path": relative, "expected_sha256": expected,
            "status": "MATCH" if expected == observed_sha else "MISMATCH"}


def write_exclusive(path, report):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, help="Required for the unchanged v1 historical audit")
    parser.add_argument("--envelope-v2", action="store_true", help="Use the distinct generic v2 schema; never upgrades v1 input")
    parser.add_argument("--expected-records", type=int, help="External expected count for v2, not inferred from rows")
    parser.add_argument("--verify-files", action="store_true", help="V2 only: explicitly rehash referenced files; default is structural validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--git-ref", help="Compare input bytes with an existing committed Git blob; no fetch")
    args = parser.parse_args(argv)
    if args.envelope_v2:
        if args.contract is not None:
            parser.error("--contract and --envelope-v2 are mutually exclusive")
        from record_envelope_v2 import SCHEMA, audit_v2_bytes
        args.contract = SCHEMA
    elif args.contract is None or args.verify_files or args.expected_records is not None:
        parser.error("v1 requires --contract; --verify-files/--expected-records require --envelope-v2")
    if args.output.exists() or args.output.resolve() in (args.input.resolve(), args.contract.resolve()):
        raise FileExistsError("Output must be a new file distinct from inputs")
    start = time.perf_counter()
    contract_bytes = args.contract.read_bytes()
    with args.input.open("rb") as stream:
        data = stream.read(MAX_BYTES+1)
    if args.envelope_v2:
        report = audit_v2_bytes(data, args.expected_records, args.repository, args.verify_files)
    else:
        report = audit_bytes(data, strict_loads(contract_bytes))
    report["contract_sha256"] = hashlib.sha256(contract_bytes).hexdigest()
    report["input"] = str(args.input)
    if args.git_ref:
        report["git_pin"] = git_pin(args.input, args.repository, args.git_ref, report["sha256"])
        if report["git_pin"]["status"] != "MATCH":
            report["status"] = "INVALID_FOR_DECLARED_CONTRACT"
    report["input_unchanged"] = hashlib.sha256(args.input.read_bytes()).hexdigest() == report["sha256"]
    report["contract_unchanged"] = args.contract.read_bytes() == contract_bytes
    if not report["input_unchanged"] or not report["contract_unchanged"]:
        report["status"] = "INVALID_FOR_DECLARED_CONTRACT"
    report["elapsed_seconds"] = time.perf_counter()-start
    report["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                         "platform": platform.platform()}
    write_exclusive(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "record_count", "issue_counts", "elapsed_seconds")}))
    return 0 if report["status"] == "VALID_FOR_DECLARED_CONTRACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
