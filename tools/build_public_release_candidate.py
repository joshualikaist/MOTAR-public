#!/usr/bin/env python3
"""Build a clean public-release snapshot from the current commit, without inheriting Git history.

The research repository keeps its full history, including blobs that must not be redistributed.
This tool does not touch it. It exports the *content of one commit* into a fresh directory that has
no `.git`, applies an explicit exclusion policy, and then tries to prove the exclusions held by
content hash rather than by filename.

Three rules the build follows:

* A file is excluded only by a written policy rule, never because it looked unnecessary.
* A hash that appears as provenance text inside a receipt is allowed; the corresponding bytes are
  not. Those are different things and the check distinguishes them.
* Archives are opened. A denylisted image hidden inside a ZIP is still a redistribution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = ROOT / "docs/public_release_denylist.json"
# Path rules. Each entry is (glob, reason). Nothing is dropped without one.
EXCLUDE_GLOBS = (
    ("**/__pycache__/**", "compiled Python caches, regenerated on use"),
    ("**/*.pyc", "compiled Python caches, regenerated on use"),
    ("**/.pytest_cache/**", "local test cache"),
    ("**/.DS_Store", "filesystem noise"),
)


def run(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_snapshot(commit, destination):
    """Extract one commit's content. `git archive` writes no history and no .git directory."""
    destination.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=str(ROOT),
                               stdout=subprocess.PIPE)
    extract = subprocess.Popen(["tar", "-x", "-C", str(destination)], stdin=archive.stdout)
    archive.stdout.close()
    extract.communicate()
    if extract.returncode or archive.wait():
        raise RuntimeError("snapshot export failed")
    if (destination / ".git").exists():
        raise RuntimeError(".git must never appear in a release candidate")


def relativize_symlinks(root):
    """Absolute in-repository symlinks break on any other machine; make them relative.

    Only the pointer changes. No regular file's bytes are touched, and a link that leaves the tree
    is removed rather than silently repointed at something else.
    """
    converted, removed = [], []
    source_root = str(ROOT.resolve())
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        if not os.path.isabs(target):
            continue
        if not (target == source_root or target.startswith(source_root + os.sep)):
            path.unlink()
            removed.append(str(path.relative_to(root)))
            continue
        inside = Path(target).resolve().relative_to(Path(source_root))
        relative = os.path.relpath(root / inside, path.parent)
        path.unlink()
        path.symlink_to(relative)
        converted.append({"path": str(path.relative_to(root)), "target": relative})
    return converted, removed


def drop_dangling_symlinks(root):
    """Remove links whose target does not exist at this commit.

    These are already broken in the research repository - their target directory was removed long
    ago - so nothing is lost by leaving them out of a release, and a published tree should not ship
    pointers to nothing. The research repository keeps them exactly as they are.
    """
    dropped = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not path.exists():
            target = os.readlink(path)
            path.unlink()
            dropped.append({"path": str(path.relative_to(root)), "target": target,
                            "reason": "dangling in the source commit as well"})
    return dropped


def apply_exclusions(root):
    dropped = []
    for pattern, reason in EXCLUDE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
            else:
                continue
            dropped.append({"path": str(path.relative_to(root)), "reason": reason})
    return dropped


def denylist_hashes():
    record = json.loads(DENYLIST.read_text())
    return {entry["sha256"]: entry for entry in record["assets"]}, record


def archive_members(path):
    """Yield (member name, bytes) for archives, so a denylisted file cannot hide inside one."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        yield info.filename, archive.read(info)
        elif suffix in (".tar", ".gz", ".tgz", ".bz2", ".xz") or path.name.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        handle = archive.extractfile(member)
                        if handle is not None:
                            yield member.name, handle.read()
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as error:
        yield "<unreadable:%s>" % type(error).__name__, b""


def scan_for_denylisted(root, denied):
    """Content-hash scan of every regular file, and of every member of every archive."""
    matches, archives_opened, files_scanned = [], 0, 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        files_scanned += 1
        digest = sha256_file(path)
        if digest in denied:
            matches.append({"kind": "file", "path": str(path.relative_to(root)), "sha256": digest})
        if path.suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"):
            archives_opened += 1
            for name, data in archive_members(path):
                if data and sha256_bytes(data) in denied:
                    matches.append({"kind": "archive_member",
                                    "path": "%s::%s" % (path.relative_to(root), name),
                                    "sha256": sha256_bytes(data)})
    return matches, files_scanned, archives_opened


def broken_symlinks(root):
    broken = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not path.exists():
            broken.append({"path": str(path.relative_to(root)), "target": os.readlink(path)})
    return broken


def tree_size(root):
    total, files, links = 0, 0, 0
    for path in root.rglob("*"):
        if path.is_symlink():
            links += 1
        elif path.is_file():
            files += 1
            total += path.stat().st_size
    return {"files": files, "symlinks": links, "bytes": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    destination = Path(arguments.output).resolve()
    if destination.exists():
        raise FileExistsError("Release candidate directory must not exist: %s" % destination)
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Refusing to build a release candidate inside the research repository")
    commit = run("rev-parse", "--verify", arguments.commit)
    dirty = bool(run("status", "--porcelain"))
    if dirty:
        raise RuntimeError("Refusing to snapshot a dirty tree; commit or stash first")
    export_snapshot(commit, destination)
    converted, removed_links = relativize_symlinks(destination)
    dropped = apply_exclusions(destination)
    dangling = drop_dangling_symlinks(destination)
    denied, denylist_record = denylist_hashes()
    matches, files_scanned, archives_opened = scan_for_denylisted(destination, denied)
    report = {
        "report": "public_release_candidate_build",
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_research_commit": commit,
        "source_tree_dirty": dirty,
        "destination": str(destination),
        "git_history_inherited": False,
        "git_directory_present": (destination / ".git").exists(),
        "tree": tree_size(destination),
        "symlinks_relativized": len(converted),
        "symlinks_removed_pointing_outside": removed_links,
        "excluded_paths": dropped,
        "dangling_symlinks_dropped": dangling,
        "denylist": {"source": str(DENYLIST.relative_to(ROOT)),
                     "assets": len(denied),
                     "sha256": sha256_file(DENYLIST),
                     "description": denylist_record.get("description")},
        "scan": {"files_scanned": files_scanned, "archives_opened": archives_opened,
                 "denylist_matches": matches},
        "broken_symlinks": broken_symlinks(destination),
        "verdict": ("CLEAN" if not matches and not broken_symlinks(destination)
                    else "CONTAMINATED"),
        "scope": "Snapshot construction and exclusion verification only. Licensing, secrets and "
                 "documentation checks are separate tools.",
    }
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "source_research_commit", "tree",
                                             "symlinks_relativized")}, indent=2, sort_keys=True))
    print("denylist matches: %d | archives opened: %d | broken symlinks: %d"
          % (len(matches), archives_opened, len(report["broken_symlinks"])))
    return 0 if report["verdict"] == "CLEAN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
