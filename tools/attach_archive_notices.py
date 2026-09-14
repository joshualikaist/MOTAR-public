#!/usr/bin/env python3
"""Add licence notices to an explicitly hash-identified ZIP, preserving all existing member bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def attach(archive, expected_sha, notices, receipt):
    archive, receipt = Path(archive), Path(receipt)
    if receipt.exists():
        raise FileExistsError("Receipt already exists")
    original = archive.read_bytes()
    if sha(original) != expected_sha:
        raise ValueError("Archive changed; inspect the exact target before editing")
    additions = {Path(path).name: Path(path).read_bytes() for path in notices}
    if len(additions) != len(notices):
        raise ValueError("Duplicate notice basename")
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(set(source.namelist())) or source.testzip():
            raise ValueError("Duplicate or corrupt original members")
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    names = {info.filename for info, _ in entries}
    if names.intersection(additions):
        raise ValueError("Notice name already exists; never overwrite a member")
    handle, temporary = tempfile.mkstemp(prefix=".notice-pack-", suffix=".zip", dir=str(archive.parent))
    os.close(handle)
    temp = Path(temporary)
    try:
        with zipfile.ZipFile(temp, "w") as target:
            for info, data in entries:
                target.writestr(info, data)
            for name, data in sorted(additions.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                target.writestr(info, data)
        with zipfile.ZipFile(temp) as target:
            if target.testzip():
                raise ValueError("Generated ZIP failed CRC")
            for info, data in entries:
                if target.read(info.filename) != data:
                    raise ValueError("Original member changed")
        result = {"schema": "archive_notice_attachment_v1", "archive": archive.name,
                  "before_sha256": expected_sha, "after_sha256": sha(temp.read_bytes()),
                  "original_members_sha256": {info.filename: sha(data) for info, data in entries},
                  "added_members_sha256": {name: sha(data) for name, data in additions.items()},
                  "status": "NOTICES_ATTACHED_ORIGINAL_MEMBERS_PRESERVED",
                  "history_limit": "Earlier Git revisions still contain the old archive"}
        if sha(archive.read_bytes()) != expected_sha:
            raise ValueError("Archive changed concurrently")
        os.replace(temp, archive)
        with receipt.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return result
    finally:
        if temp.exists():
            temp.unlink()  # Only this tool's own temporary file, never an input/archive.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--notice", type=Path, nargs="+", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(attach(args.archive, args.expected_sha256, args.notice, args.receipt)))


if __name__ == "__main__":
    main()
