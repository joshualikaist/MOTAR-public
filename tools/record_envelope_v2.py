"""Standalone generic provenance producer. No integration with a task, model or runtime logger."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs/record_envelope_v2.json"
DEFAULT_SOURCES = ("tools/record_envelope_v2.py", "tools/check_record_integrity.py", "docs/record_envelope_v2.json")
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
REQUIRED = {"schema_version", "schema_sha256", "run_id", "source_git_commit", "source_dirty",
            "source_files_sha256", "seed", "density_bars", "checkpoint_path", "checkpoint_sha256",
            "config_path", "config_sha256", "created_at_utc", "record_count", "records"}


class EnvelopeError(ValueError):
    """Stable machine code, without serializing a supplied value or reading a model."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise EnvelopeError(code)


def strict(data):
    # Lazy import keeps the historical checker API available without a module cycle.
    from check_record_integrity import strict_loads
    try:
        return strict_loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise EnvelopeError("STRICT_JSON_INVALID") from error


def encoded(value):
    try:
        return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n").encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise EnvelopeError("NON_JSON_OR_NONFINITE_VALUE") from error


def content_sha256(path):
    """Hash actual regular-file bytes, detecting concurrent inode/content-metadata changes."""
    try:
        require(stat.S_ISREG(Path(path).stat().st_mode), "INPUT_NOT_REGULAR_FILE")
        with Path(path).open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "INPUT_NOT_REGULAR_FILE")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = Path(path).stat()
    except OSError as error:
        raise EnvelopeError("INPUT_FILE_UNAVAILABLE") from error
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(signature(before) == signature(after) == signature(current), "INPUT_CHANGED_DURING_HASH")
    return digest.hexdigest()


def schema_digest():
    """Refuse missing/malformed schema files, not just hash arbitrary bytes under that name."""
    try:
        data = SCHEMA.read_bytes()
    except OSError as error:
        raise EnvelopeError("SCHEMA_UNAVAILABLE") from error
    definition = strict(data)
    require(isinstance(definition, dict) and definition.get("type") == "object"
            and definition.get("additionalProperties") is False
            and set(definition.get("required", [])) == REQUIRED
            and set(definition.get("properties", {})) == REQUIRED
            and definition["properties"]["schema_version"].get("const") == 2,
            "SCHEMA_DEFINITION_INVALID")
    return hashlib.sha256(data).hexdigest()


def source_path(repository, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name, "INVALID_SOURCE_PATH")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
            "INVALID_SOURCE_PATH")
    root = Path(repository).resolve()
    path = (root/name).resolve()
    require(root in path.parents, "SOURCE_PATH_ESCAPES_REPOSITORY")
    return path


def git_state(repository):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repository, text=True, stderr=subprocess.PIPE).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise EnvelopeError("GIT_PROVENANCE_UNAVAILABLE") from error
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == Path(repository).resolve(), "REPOSITORY_ROOT_REQUIRED")
    commit = git("rev-parse", "--verify", "HEAD")
    require(bool(COMMIT.fullmatch(commit)), "INVALID_GIT_COMMIT")
    require(git("cat-file", "-t", commit) == "commit", "INVALID_GIT_COMMIT")
    return commit, bool(git("status", "--porcelain=v1", "--untracked-files=all"))


def validate_document(document, expected_records=None):
    encoded(document)  # Reject cycles, unsupported objects and NaN/Inf before a tree walk.
    require(isinstance(document, dict) and set(document) == REQUIRED, "SCHEMA_FIELDS_MISMATCH")
    require(type(document["schema_version"]) is int and document["schema_version"] == 2, "SCHEMA_VERSION_MISMATCH")
    for key in ("schema_sha256", "checkpoint_sha256", "config_sha256"):
        require(type(document[key]) is str and bool(HEX.fullmatch(document[key])), "INVALID_"+key.upper())
    require(document["schema_sha256"] == schema_digest(), "SCHEMA_HASH_MISMATCH")
    require(type(document["source_git_commit"]) is str and bool(COMMIT.fullmatch(document["source_git_commit"])),
            "INVALID_SOURCE_GIT_COMMIT")
    require(type(document["source_dirty"]) is bool, "INVALID_SOURCE_DIRTY")
    for key in ("seed", "density_bars", "record_count"):
        require(type(document[key]) is int and document[key] >= 0, "INVALID_"+key.upper())
    for key in ("checkpoint_path", "config_path"):
        require(type(document[key]) is str and bool(document[key]) and "\x00" not in document[key], "INVALID_"+key.upper())
    try:
        identifier = uuid.UUID(document["run_id"])
        require(identifier.version == 4 and str(identifier) == document["run_id"], "INVALID_RUN_ID")
        datetime.strptime(document["created_at_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
        require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", document["created_at_utc"])), "INVALID_UTC_TIMESTAMP")
    except (ValueError, AttributeError, TypeError) as error:
        if isinstance(error, EnvelopeError):
            raise
        raise EnvelopeError("INVALID_RUN_ID_OR_UTC_TIMESTAMP") from error
    sources = document["source_files_sha256"]
    require(isinstance(sources, dict) and bool(sources), "SOURCE_MANIFEST_REQUIRED")
    for name, digest in sources.items():
        # Syntax only here: validation does not follow paths embedded in arbitrary JSON.
        require(isinstance(name, str) and bool(name) and "\\" not in name and not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts and str(PurePosixPath(name)) == name, "INVALID_SOURCE_PATH")
        require(type(digest) is str and bool(HEX.fullmatch(digest)), "INVALID_SOURCE_SHA256")
    require(type(document["records"]) is list, "RECORD_ARRAY_REQUIRED")
    require(len(document["records"]) == document["record_count"], "RECORD_COUNT_MISMATCH")
    if expected_records is not None:
        require(type(expected_records) is int and expected_records >= 0, "INVALID_EXPECTED_RECORD_COUNT")
        require(document["record_count"] == expected_records, "EXPECTED_RECORD_COUNT_MISMATCH")
    seen = set()
    for row in document["records"]:
        require(type(row) is dict, "RECORD_NOT_OBJECT")
        require(type(row.get("record_id")) is str and bool(row["record_id"]), "RECORD_ID_REQUIRED")
        require(row["record_id"] not in seen, "DUPLICATE_RECORD_ID")
        seen.add(row["record_id"])
        for key in (REQUIRED-{"records", "record_count"}) & row.keys():
            require(type(row[key]) is type(document[key]) and row[key] == document[key], "CONTEXT_RECORD_MISMATCH")
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            require(all(type(k) is str for k in value), "NON_STRING_JSON_KEY")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif type(value) is float:
            require(math.isfinite(value), "NONFINITE_VALUE")
        else:
            require(value is None or type(value) in (str, int, bool), "NON_JSON_VALUE")
    return document


def verify_inputs(document, repository):
    """Explicit file revalidation; locations are read only on this opt-in path."""
    commit, _ = git_state(repository)
    require(commit == document["source_git_commit"], "SOURCE_GIT_COMMIT_CHANGED")
    for kind in ("checkpoint", "config"):
        require(content_sha256(document[kind+"_path"]) == document[kind+"_sha256"], kind.upper()+"_HASH_MISMATCH")
    for name, expected in document["source_files_sha256"].items():
        require(content_sha256(source_path(repository, name)) == expected, "SOURCE_FILE_HASH_MISMATCH")


@dataclass(frozen=True)
class Provenance:
    metadata_bytes: bytes
    repository: Path

    @property
    def metadata(self):
        return strict(self.metadata_bytes)  # An owned copy: caller mutations cannot alter capture.


def preflight(*, repository, seed, density_bars, checkpoint_path, config_path, expected_records,
              source_files=DEFAULT_SOURCES):
    for name, value in (("seed", seed), ("density_bars", density_bars), ("expected_records", expected_records)):
        require(type(value) is int and value >= 0, "INVALID_"+name.upper())
    require(bool(source_files) and len(source_files) == len(set(source_files)), "SOURCE_MANIFEST_REQUIRED")
    repository = Path(repository).resolve()
    before = git_state(repository)
    sources = {name: content_sha256(source_path(repository, name)) for name in source_files}
    document = {"schema_version": 2, "schema_sha256": schema_digest(), "run_id": str(uuid.uuid4()),
                "source_git_commit": before[0], "source_dirty": before[1], "source_files_sha256": sources,
                "seed": seed, "density_bars": density_bars,
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "record_count": 0, "records": []}
    for kind, supplied in (("checkpoint", checkpoint_path), ("config", config_path)):
        require(isinstance(supplied, (str, Path)) and bool(str(supplied)), "MISSING_"+kind.upper()+"_PATH")
        path = Path(supplied).resolve()
        document[kind+"_path"] = str(path)
        document[kind+"_sha256"] = content_sha256(path)
    require(git_state(repository) == before, "GIT_CHANGED_DURING_PREFLIGHT")
    validate_document(document, 0)
    verify_inputs(document, repository)
    document.pop("records")
    document["record_count"] = expected_records
    return Provenance(encoded(document), repository)


def audit_v2_bytes(data, expected_records=None, repository=None, verify_files=False):
    from check_record_integrity import MAX_BYTES
    report = {"schema_version": 2, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
              "status": "INVALID_FOR_DECLARED_CONTRACT", "record_count": None, "issue_counts": {},
              "scope": "Generic envelope only; no domain outcomes assessed", "file_identity_validation": "NOT_RUN",
              "expected_count_source": "caller" if expected_records is not None else "declared record_count"}
    try:
        require(len(data) <= MAX_BYTES, "INPUT_SIZE_LIMIT")
        document = strict(data)
        validate_document(document, expected_records)
        report["record_count"] = document["record_count"]
        if verify_files:
            require(repository is not None, "REPOSITORY_REQUIRED")
            verify_inputs(document, repository)
            report["file_identity_validation"] = "PASS"
        report["status"] = "VALID_FOR_DECLARED_CONTRACT"
    except EnvelopeError as error:
        report["issue_counts"] = {error.code: 1}
    return report


def postflight(path, capture, expected_sha256):
    metadata = capture.metadata
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "ARTIFACT_HASH_MISMATCH")
    document = validate_document(strict(data), metadata["record_count"])
    require({k: v for k, v in document.items() if k != "records"} == metadata, "CAPTURED_METADATA_MISMATCH")
    verify_inputs(document, capture.repository)
    require(content_sha256(path) == expected_sha256, "ARTIFACT_CHANGED_DURING_POSTFLIGHT")
    return {"status": "POSTFLIGHT_PASS", "artifact_sha256": expected_sha256,
            "schema_sha256": document["schema_sha256"], "run_id": document["run_id"],
            "checks": {k: "PASS" for k in ("strict_json", "schema", "expected_records", "unique_record_ids",
                     "finite_values", "context_consistency", "checkpoint_hash", "config_hash", "source_hashes", "artifact_hash")},
            "scope": "Generic artifact provenance only; no operational validation"}


def publish(capture, records, output):
    """Publish a NEW artifact plus receipt; failed postflight retains raw bytes but no PASS receipt."""
    from check_record_integrity import MAX_BYTES, write_exclusive
    output = Path(output)
    receipt = output.with_name(output.name+".receipt.json")
    failure = output.with_name(output.name+".failure.json")
    require(not any(p.exists() for p in (output, receipt, failure)), "OUTPUT_ALREADY_EXISTS")
    require(type(records) is list, "RECORD_ARRAY_REQUIRED")
    document = {**capture.metadata, "records": records}
    validate_document(document, capture.metadata["record_count"])
    payload = encoded(document)
    require(len(payload) <= MAX_BYTES, "INPUT_SIZE_LIMIT")
    verify_inputs(document, capture.repository)
    digest = hashlib.sha256(payload).hexdigest()
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        checked = postflight(output, capture, digest)
        write_exclusive(receipt, checked)
    except (EnvelopeError, OSError) as error:
        write_exclusive(failure, {"status": "INVALID_ARTIFACT", "error": getattr(error, "code", type(error).__name__),
                                  "raw_artifact_preserved": True, "success_receipt_written": False})
        raise
    return checked
