#!/usr/bin/env python3
"""Read-only, redacted audit of blobs reachable from all local Git refs (not reflogs).

Finite pattern scan, NOT a guarantee of absence of credentials. Never prints matched values.
Large/binary blobs are inventoried rather than silently considered secret-free.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

PATTERNS = {
    "private_key_header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "aws_access_key_id": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential_assignment_candidate": re.compile(
        rb"(?im)^\s*(?:password|api_key|access_token|secret_key)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
    "personal_absolute_path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}


def findings(payload):
    return {name: len(pattern.findall(payload)) for name, pattern in PATTERNS.items()
            if pattern.search(payload)}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit(root, limit=2 * 1024 ** 2):
    objects = git(root, "rev-list", "--objects", "--all").decode().splitlines()
    paths = dict(line.split(" ", 1) if " " in line else (line, "") for line in objects)
    current = set(line.split()[1] for line in git(root, "ls-files", "-s").decode().splitlines())
    process = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    counts = Counter()
    matches, inventory = [], []
    try:
        for oid, path in paths.items():
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if len(header) != 3:
                raise RuntimeError("Unexpected cat-file response")
            _, kind, size = header
            size = int(size)
            payload = bytearray()
            remaining = size
            while remaining:
                block = process.stdout.read(min(1024 ** 2, remaining))
                if not block:
                    raise RuntimeError("Truncated Git object")
                if size <= limit and kind == "blob":
                    payload.extend(block)
                remaining -= len(block)
            if process.stdout.read(1) != b"\n":
                raise RuntimeError("Missing Git object delimiter")
            if kind != "blob":
                continue
            counts["blobs"] += 1
            counts["blob_bytes"] += size
            suffix = Path(path).suffix.lower()
            row = {"blob": oid, "example_path": path, "bytes": size, "in_current_index": oid in current}
            if size > limit:
                counts["oversized_not_content_scanned"] += 1
            elif b"\x00" in payload:
                counts["binary_not_content_scanned"] += 1
            else:
                counts["text_scanned"] += 1
                found = findings(bytes(payload))
                if found:
                    matches.append(dict(row, patterns=found))
                    counts.update({"matched_" + key: value for key, value in found.items()})
            if size >= 5 * 1024 ** 2 or suffix in (".zip", ".7z", ".tar", ".gz", ".jpg", ".jpeg", ".png", ".pth", ".pt"):
                inventory.append(row)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if process.returncode:
        raise RuntimeError("Git object reader failed")
    return {"schema": "public_history_audit_v1", "source_commit": git(root, "rev-parse", "HEAD").decode().strip(),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_note": "source_commit identifies the audited repository; tool content separately hashed",
            "refs_sha256": hashlib.sha256(git(root, "show-ref")).hexdigest(),
            "scope": "All local refs after fetch; reachable unique blobs; excludes reflogs, unreferenced objects and remote-only hidden refs",
            "status": "REVIEW_REQUIRED", "text_size_limit_bytes": limit, "counts": dict(counts),
            "matches_redacted": matches, "binary_archive_image_weight_inventory": inventory,
            "limits": ["Patterns are not exhaustive and can match documentation/test placeholders",
                       "Binary and oversized contents not scanned for secrets",
                       "example_path is one rev-list path per blob, not every historical alias",
                       "No history rewrite or data removal performed"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite an audit")
    result = audit(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
