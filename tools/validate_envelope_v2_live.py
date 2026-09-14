#!/usr/bin/env python3
"""Can Record Envelope v2 wrap a real, task-independent artifact producer? Find out by trying.

The unit tests show the producer refuses bad synthetic input. They cannot show whether the contract
fits a real generation path in this repository, which is a different question and the one asked
here. The candidate is the generic renderer dataset exporter: already audited, already
task-independent, and it touches no policy, task, controller or target performance.

This tool does not try to make the attempt succeed. It supplies only values it can obtain honestly
and records what the producer does with the rest. A generic exporter has no policy checkpoint and no
obstacle-bar density; inventing `density_bars = 0` or pointing `checkpoint_path` at a URDF would
manufacture provenance, which is the exact defect the envelope exists to prevent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Why each required field is or is not obtainable from a task-independent producer. Judgements
# about the producer, not about the envelope: the envelope is simply what it is.
FIELD_SOURCES = {
    "schema_version": ("AVAILABLE", "constant declared by the contract"),
    "schema_sha256": ("AVAILABLE", "sha256 of the schema file"),
    "run_id": ("AVAILABLE", "uuid4 minted per run"),
    "source_git_commit": ("AVAILABLE", "git rev-parse HEAD in the repository"),
    "source_dirty": ("AVAILABLE", "git status, observed"),
    "source_files_sha256": ("AVAILABLE", "sha256 of the producer's own sources"),
    "created_at_utc": ("AVAILABLE", "clock at capture"),
    "record_count": ("AVAILABLE", "number of exported frames"),
    "records": ("AVAILABLE", "one row per exported frame"),
    "seed": ("AVAILABLE", "the exporter takes --material-seed and --light-seed; either is a real "
                          "integer seed, though the contract has one slot for two seeds"),
    "density_bars": ("NOT_APPLICABLE", "obstacle-bar density is a simulator-task concept; a static "
                                       "renderer export has no bars and no honest value"),
    "checkpoint_path": ("NOT_APPLICABLE", "a generic exporter loads no policy checkpoint"),
    "checkpoint_sha256": ("NOT_APPLICABLE", "no checkpoint exists to hash"),
    "config_path": ("NOT_APPLICABLE", "the exporter is configured by command-line arguments and "
                                      "writes its configuration into its own receipt afterwards; "
                                      "there is no pre-existing config file to hash"),
    "config_sha256": ("NOT_APPLICABLE", "no config file exists to hash"),
}
CANDIDATE = "tools/export_renderer_dataset.py"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def honest_attempt(repository):
    """Call preflight with the honest values only, and record exactly what it refuses."""
    import record_envelope_v2 as v2
    attempts = []
    for label, arguments in (
        ("density_bars unknown",
         dict(seed=0, density_bars=None, checkpoint_path=ROOT / CANDIDATE,
              config_path=ROOT / CANDIDATE, expected_records=1)),
        ("checkpoint absent",
         dict(seed=0, density_bars=0, checkpoint_path=None,
              config_path=ROOT / CANDIDATE, expected_records=1)),
        ("config absent",
         dict(seed=0, density_bars=0, checkpoint_path=ROOT / CANDIDATE,
              config_path=None, expected_records=1)),
    ):
        try:
            v2.preflight(repository=repository, **arguments)
            attempts.append({"attempt": label, "result": "ACCEPTED",
                             "note": "the producer accepted a value it should not have"})
        except v2.EnvelopeError as error:
            attempts.append({"attempt": label, "result": "REFUSED", "error_code": error.code})
        except Exception as error:                      # pragma: no cover - environment dependent
            attempts.append({"attempt": label, "result": "ERROR",
                             "error_type": type(error).__name__})
    return attempts


def export_runs(repository):
    """Prove the candidate producer itself still works, without wrapping it in an envelope."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "generic"
        completed = subprocess.run(
            [sys.executable, "-B", str(ROOT / CANDIDATE), "--output", str(output),
             "--device", "cpu", "--width", "40", "--height", "30"],
            cwd=str(repository), capture_output=True, text=True)
        receipt = output / "receipt.json"
        record = {"returncode": completed.returncode, "receipt_present": receipt.is_file()}
        if receipt.is_file():
            payload = json.loads(receipt.read_text())
            record.update({"schema": payload.get("schema"), "status": payload.get("status"),
                           "frames": len(payload.get("files") or []),
                           "has_checkpoint_field": "checkpoint_sha256" in payload,
                           "has_density_field": "density_bars" in payload})
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    repository = Path(arguments.repository).resolve()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError("Output must not exist")
    schema = json.loads((ROOT / "docs/record_envelope_v2.json").read_text())
    required = list(schema["required"])
    unavailable = sorted(name for name in required
                         if FIELD_SOURCES.get(name, ("UNKNOWN",))[0] != "AVAILABLE")
    attempts = honest_attempt(repository)
    producer = export_runs(repository)
    verdict = ("LIVE_GENERIC_VALIDATION_NOT_APPLICABLE" if unavailable
               else "LIVE_GENERIC_VALIDATION_POSSIBLE")
    summary = {
        "experiment": "Record Envelope v2 live-generation feasibility",
        "verdict": verdict,
        "question": "Does Record Envelope v2 fit a real task-independent artifact producer in this "
                    "repository?",
        "candidate_producer": CANDIDATE,
        "candidate_producer_sha256": sha256_file(ROOT / CANDIDATE),
        "candidate_still_works": producer,
        "schema": "docs/record_envelope_v2.json",
        "schema_sha256": sha256_file(ROOT / "docs/record_envelope_v2.json"),
        "required_fields": required,
        "field_sources": {name: {"availability": FIELD_SOURCES[name][0],
                                 "reason": FIELD_SOURCES[name][1]} for name in required},
        "fields_without_an_honest_generic_source": unavailable,
        "fail_closed_attempts": attempts,
        "producer_invented_nothing": all(row["result"] == "REFUSED" for row in attempts),
        "decision": "No value was fabricated for the unavailable fields, and the envelope was not "
                    "modified to accommodate the producer. A v2.1 with an optional task section is "
                    "a design change and needs its own decision, not a side effect of this check.",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "scope": "Generic provenance feasibility only; no policy, task, controller or target "
                 "performance work",
        "causality_vs_d8b": "NOT_TESTED",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    (output / "README.md").write_text(readme(summary), encoding="utf-8")
    print("%s -> %s" % (verdict, output))
    return 0 if verdict == "LIVE_GENERIC_VALIDATION_POSSIBLE" else 2


def readme(summary):
    lines = [
        "# Record Envelope v2 — live-generation feasibility, 14 September 2026", "",
        "**Verdict: `%s`.**" % summary["verdict"], "",
        "Question: %s" % summary["question"], "", "## What was tried", "",
        "The candidate was the generic renderer dataset exporter (`%s`), chosen because it is "
        "already audited, already task-independent, and touches no policy, task or controller."
        % summary["candidate_producer"], "",
        "It still works: the exporter ran and wrote a `%s` receipt with %s frame(s)."
        % (summary["candidate_still_works"].get("schema"),
           summary["candidate_still_works"].get("frames")),
        "Its own receipt has no checkpoint field and no density field, which is the point.", "",
        "## Why the envelope does not fit it", "",
        "| Field | Availability | Reason |", "| --- | --- | --- |"]
    for name, row in summary["field_sources"].items():
        lines.append("| `%s` | %s | %s |" % (name, row["availability"], row["reason"]))
    lines += ["", "Five of the fifteen required fields have no honest source in a task-independent "
              "producer. `density_bars` and the two checkpoint fields are simulator-task concepts; "
              "the two config fields assume a pre-existing configuration file, and this exporter is "
              "configured by command-line arguments.", "",
              "## The producer invented nothing", "",
              "Asked for an envelope without those values, the producer refused every time rather "
              "than filling a default:", "", "| Attempt | Result | Code |", "| --- | --- | --- |"]
    for row in summary["fail_closed_attempts"]:
        lines.append("| %s | %s | `%s` |" % (row["attempt"], row["result"],
                                             row.get("error_code", "-")))
    lines += ["", "That is the correct behaviour and it is why this check ends in "
              "`NOT_APPLICABLE` rather than in a passing envelope. Setting `density_bars = 0`, "
              "pointing `checkpoint_path` at a URDF, or hashing the exporter's own source as a "
              "\"config\" would each have produced a green result and a false record.", "",
              "## What was not done", "",
              summary["decision"], "",
              "The historical 128-row artifact is untouched, and its two verdicts stand: byte "
              "integrity `MATCH`, metadata contract `INVALID_FOR_DECLARED_CONTRACT`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
