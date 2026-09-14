#!/usr/bin/env python3
"""Index every result directory by identity, provenance and links. It never reads a research claim.

What this tool does: names each result, records where its receipt, summary, preregistration and
source manifest live, hashes those files, copies the verdict string verbatim, and reports what is
missing or inconsistent.

What it deliberately does not do: extract a capture rate, a performance delta, a failure cause or
any other number that means something about the research. A manifest that interprets results
becomes a second, unreviewed place where findings are stated, and the two drift. Verdict strings are
copied exactly as written, never normalised, summarised or re-graded.

Old directories are not forced into the current shape. A result created before the convention
existed is recorded with `legacy_exception` and its missing files are listed as facts, not defects.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# The modern result shape (preregistration, README, config, summary, receipt, source manifest) was
# introduced with the renderer characterization track. Directories first committed before this are
# recorded as legacy: their absent files are history, not defects to repair now.
CONVENTION_SINCE = "2026-09-13"
STATUS_KEYS = ("verdict", "status", "run_verdict", "overall")
NEGATIVE_TOKENS = ("FAIL", "VOID", "INCONCLUSIVE", "PARTIAL_EVIDENCE", "BLOCKED", "NOT_RUN",
                   "DEFECT", "LOSS", "NOT_APPLICABLE", "UNSUPPORTED", "WITHDRAWN")
SUPERSEDED_TOKENS = ("SUPERSEDED", "RETRACTED")
PREREGISTRATION_PATTERN = re.compile(r"(docs/preregistration[\w./-]*\.md|PREREGISTRATION\.md)")
MAX_TEXT_BYTES = 4 * 1024 * 1024


def external_contract():
    path = ROOT / "docs/external_data_manifest.json"
    return json.loads(path.read_text())


def reference_availability(relative, expected):
    """An explicit removed path AND its historical hash are required for the exception."""
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        return "MISSING_REQUIRED_REPOSITORY_ARTIFACT"
    if path.is_file():
        return "PRESENT_HASH_MATCH" if sha256_file(path) == expected else "HASH_MISMATCH"
    for item in external_contract()["excluded_assets"]:
        if item["path"] == relative and item["sha256"] == expected:
            return "EXTERNAL_DATA_NOT_REDISTRIBUTED"
    return "MISSING_REQUIRED_REPOSITORY_ARTIFACT"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args, cwd=ROOT):
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def small_json(path):
    """Read a JSON file only if it is small enough to be metadata rather than data."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def first_heading(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def verbatim_status(document):
    """The verdict exactly as the result wrote it. No mapping, no grading, no normalisation."""
    if not isinstance(document, dict):
        return None
    for key in STATUS_KEYS:
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def lifecycle(status):
    """A coarse label derived only from literal tokens already present in the verdict string."""
    if not status:
        return "unlabelled"
    upper = status.upper()
    if any(token in upper for token in SUPERSEDED_TOKENS):
        return "superseded"
    if any(token in upper for token in NEGATIVE_TOKENS):
        return "negative_result"
    return "recorded"


def referenced_preregistrations(directory):
    """Preregistration paths the result itself names, so a dangling reference can be seen."""
    found = set()
    for name in ("README.md", "summary.json", "receipt.json", "PREREGISTRATION.md"):
        path = directory / name
        if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found.update(PREREGISTRATION_PATTERN.findall(text))
    return sorted(found)


def source_manifest_state(directory):
    """Re-hash what a result's own source manifest pinned. Drift is reported, not judged.

    A result records the source as it stood when it ran. Later commits move the tree, so a
    mismatch here usually means the repository progressed, not that the result is wrong. Calling it
    a defect would be a false alarm; not reporting it at all would hide a real reproduction risk.
    """
    document = small_json(directory / "source_manifest.json")
    if not isinstance(document, dict):
        return None
    hashes = document.get("source_sha256")
    if not isinstance(hashes, dict):
        return {"commit": document.get("commit"), "files_checked": 0, "matching": 0,
                "drifted": [], "missing": []}
    matching, drifted, missing, external = 0, [], [], []
    for relative, expected in sorted(hashes.items()):
        state = reference_availability(relative, expected)
        if state == "EXTERNAL_DATA_NOT_REDISTRIBUTED":
            external.append({"path": relative, "sha256": expected,
                             "reference_kind": "HISTORICAL_INPUT_REFERENCE"})
        elif state == "MISSING_REQUIRED_REPOSITORY_ARTIFACT":
            missing.append(relative)
        elif state == "PRESENT_HASH_MATCH":
            matching += 1
        else:
            drifted.append(relative)
    return {"commit": document.get("commit"), "files_checked": len(hashes), "matching": matching,
            "drifted": drifted, "missing": missing, "external_data_not_redistributed": external}


def first_commit_date(relative):
    date = git("log", "--diff-filter=A", "--format=%aI", "--", relative)
    return (date.splitlines() or [""])[-1] or None


def git_available():
    return bool(git("rev-parse", "--git-dir"))


def tracked_prefixes():
    """Every tracked path under results/, so an untracked result directory is visible as one.

    An untracked result directory is a provenance gap: it exists on this machine and in no commit.
    A snapshot export drops it silently, so the manifest should say which entries are committed.
    """
    listing = git("ls-files", "results")
    return {line.split("/")[1] for line in listing.splitlines() if "/" in line}


def inherited_classification(path):
    """Reuse created_at/legacy/tracked from a manifest built where Git was available.

    A release snapshot has no Git directory, so it cannot date its own results. Importing the dates
    from the research manifest keeps the classification truthful and records where it came from,
    which is better than silently calling every legacy directory modern.
    """
    document = json.loads(Path(path).read_text())
    return {entry["result_id"]: entry for entry in document.get("results", [])}, {
        "source": str(path), "sha256": sha256_file(path),
        "built_at_utc": document.get("built_at_utc"),
        "repository_commit": document.get("repository_commit")}


def child_results(directory):
    """Immediate subdirectories that are themselves results, for track directories.

    A track directory holds several experiments and has no summary of its own. Reporting it as
    "missing a summary" would be wrong twice over: it invents a defect, and it hides the real
    results one level down.
    """
    return sorted(child for child in directory.iterdir()
                  if child.is_dir() and (child / "summary.json").is_file())


def describe(directory, parent=None, tracked=None, inherited=None):
    relative = directory.relative_to(ROOT).as_posix()
    summary = small_json(directory / "summary.json")
    receipt = small_json(directory / "receipt.json")
    status = verbatim_status(summary) or verbatim_status(receipt)
    created = first_commit_date(relative)
    legacy = bool(created) and created[:10] < CONVENTION_SINCE
    children = child_results(directory) if parent is None else []
    entry = {
        "result_id": directory.name if parent is None else "%s/%s" % (parent, directory.name),
        "parent": parent,
        "container": bool(children) and not (directory / "summary.json").is_file(),
        "title": first_heading(directory / "README.md"),
        "status": status,
        "status_source": ("summary.json" if verbatim_status(summary) else
                          ("receipt.json" if verbatim_status(receipt) else None)),
        "lifecycle": lifecycle(status),
        "result_path": relative,
        "created_at": created,
        "legacy_exception": legacy,
        "claim_scope": (summary or {}).get("scope") if isinstance(summary, dict) else None,
        "causality_vs_d8b": (summary or {}).get("causality_vs_d8b") if isinstance(summary, dict) else None,
    }
    for key, name in (("summary", "summary.json"), ("receipt", "receipt.json"),
                      ("readme", "README.md"), ("preregistration", "PREREGISTRATION.md"),
                      ("source_manifest", "source_manifest.json"), ("config", "config.json")):
        path = directory / name
        entry[key + "_path"] = (path.relative_to(ROOT).as_posix() if path.is_file() else None)
        entry[key + "_sha256"] = sha256_file(path) if path.is_file() else None
    entry["source_git_commit"] = None
    for document in (small_json(directory / "source_manifest.json"), receipt, summary):
        if isinstance(document, dict):
            commit = document.get("commit") or (document.get("source_manifest") or {}).get("commit") \
                if isinstance(document.get("source_manifest"), dict) else document.get("commit")
            if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
                entry["source_git_commit"] = commit
                break
    entry["referenced_preregistrations"] = referenced_preregistrations(directory)
    entry["dangling_preregistrations"] = [name for name in entry["referenced_preregistrations"]
                                          if not (ROOT / name).is_file()
                                          and not (directory / name).is_file()]
    entry["source_manifest_state"] = source_manifest_state(directory)
    top_level = entry["result_id"].split("/")[0]
    entry["tracked_in_git"] = (None if tracked is None else top_level in tracked)
    if inherited is not None:
        previous = inherited.get(entry["result_id"])
        if previous is not None:
            entry["created_at"] = previous.get("created_at")
            entry["legacy_exception"] = previous.get("legacy_exception")
            if entry["tracked_in_git"] is None:
                entry["tracked_in_git"] = previous.get("tracked_in_git")
            entry["classification_inherited"] = True
        else:
            entry["classification_inherited"] = False
            entry["legacy_exception"] = None
    contract = external_contract()
    entry["external_data"] = ({"dataset": contract["dataset"],
                               "current_release_availability": contract["current_release_availability"],
                               "reference_kind": "HISTORICAL_INPUT_REFERENCE",
                               "contract_path": "docs/external_data_manifest.json",
                               "contract_sha256": sha256_file(ROOT / "docs/external_data_manifest.json")}
                              if directory.name in contract["required_for"] else None)
    entry["files"] = sum(1 for _ in directory.rglob("*") if _.is_file())
    entry["child_result_ids"] = [child.name for child in children]
    return entry


def validate(entries, loose_files):
    seen, duplicates = set(), []
    for entry in entries:
        if entry["result_id"] in seen:
            duplicates.append(entry["result_id"])
        seen.add(entry["result_id"])
    # An entry whose legacy status is unknown cannot be held to the modern convention.
    modern = [e for e in entries
              if e["legacy_exception"] is False and not e["container"]]
    issues = {
        "duplicate_result_id": duplicates,
        "modern_missing_summary": [e["result_id"] for e in modern if not e["summary_path"]],
        "modern_missing_receipt": [e["result_id"] for e in modern if not e["receipt_path"]],
        "modern_missing_readme": [e["result_id"] for e in modern if not e["readme_path"]],
        "dangling_preregistration": {e["result_id"]: e["dangling_preregistrations"]
                                     for e in entries if e["dangling_preregistrations"]},
        "source_manifest_drift": {e["result_id"]: e["source_manifest_state"]["drifted"]
                                  for e in entries
                                  if e["source_manifest_state"]
                                  and e["source_manifest_state"]["drifted"]},
        "source_manifest_missing_files": {e["result_id"]: e["source_manifest_state"]["missing"]
                                          for e in entries
                                          if e["source_manifest_state"]
                                          and e["source_manifest_state"]["missing"]},
        "untracked_results": [e["result_id"] for e in entries if e.get("tracked_in_git") is False],
        "unclassified_results": [e["result_id"] for e in entries if e["legacy_exception"] is None],
        "orphan_candidates": [e["result_id"] for e in entries
                              if not e["container"]
                              and not any((e["summary_path"], e["receipt_path"], e["readme_path"]))],
        "loose_files_outside_any_result": loose_files,
    }
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "MANIFEST.json")
    parser.add_argument("--print-issues", action="store_true")
    parser.add_argument("--inherit-classification", type=Path,
                        help="Manifest built where Git was available, for a snapshot with no .git")
    arguments = parser.parse_args(argv)
    have_git = git_available()
    tracked = tracked_prefixes() if have_git else None
    inherited, inheritance = (None, None)
    if arguments.inherit_classification:
        inherited, inheritance = inherited_classification(arguments.inherit_classification)
    directories = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    loose = sorted(p.name for p in RESULTS.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    entries = []
    for directory in directories:
        entry = describe(directory, tracked=tracked, inherited=inherited)
        entries.append(entry)
        if entry["container"]:
            entries.extend(describe(child, parent=directory.name, tracked=tracked,
                                    inherited=inherited)
                           for child in child_results(directory))
    issues = validate(entries, loose)
    lifecycles = {}
    for entry in entries:
        lifecycles[entry["lifecycle"]] = lifecycles.get(entry["lifecycle"], 0) + 1
    manifest = {
        "schema": "result_manifest_v1",
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain")),
        "convention_since": CONVENTION_SINCE,
        "git_available": have_git,
        "classification_inherited_from": inheritance,
        "counts": {"results": len(entries),
                   "legacy_exception": sum(e["legacy_exception"] is True for e in entries),
                   "untracked_in_git": sum(e.get("tracked_in_git") is False for e in entries),
                   "containers": sum(e["container"] for e in entries),
                   "child_results": sum(e["parent"] is not None for e in entries),
                   "modern": sum(e["legacy_exception"] is False and not e["container"]
                                 for e in entries),
                   "with_summary": sum(bool(e["summary_path"]) for e in entries),
                   "with_receipt": sum(bool(e["receipt_path"]) for e in entries),
                   "with_preregistration": sum(bool(e["preregistration_path"]) for e in entries),
                   "lifecycle": lifecycles},
        "scope": "Identity, provenance, status strings and links only. No research quantity is "
                 "extracted or interpreted, and no verdict is normalised or re-graded.",
        "issues": issues,
        "results": entries,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("indexed %d results (%d legacy) -> %s"
          % (len(entries), manifest["counts"]["legacy_exception"], output))
    if arguments.print_issues:
        print(json.dumps({k: v for k, v in issues.items() if v}, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
