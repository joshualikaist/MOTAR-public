#!/usr/bin/env python3
"""Create or verify an immutable-by-hash NavRL training source receipt.

The policy checkpoint records the manifest path and digest.  The bundle contains the exact Python,
launcher/config and robot-URDF bytes used to start a run, plus the Python package environment.
Ignored run outputs are intentionally excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


RUNTIME_ROOTS = ("aerial_gym", "resources/robots")
RUNTIME_EXTRA_PATHS = ("tools/create_navrl_source_bundle.py",)
RUNTIME_EXTENSIONS = {
    ".csv",
    ".json",
    ".py",
    ".pyx",
    ".sh",
    ".toml",
    ".urdf",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=not binary
    )


def git_paths(repo: Path, *args: str) -> list[Path]:
    raw = git(repo, *args, binary=True)
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def repository_root() -> Path:
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip()).resolve()


def runtime_paths(repo: Path) -> list[Path]:
    paths = set(git_paths(repo, "ls-files", "-z", "--", *RUNTIME_ROOTS))
    paths.update(
        git_paths(
            repo,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *RUNTIME_ROOTS,
        )
    )
    # The receipt creator is part of the executable provenance even though it is outside the two
    # simulator roots.  Pin it explicitly rather than snapshotting every offline analysis tool.
    for name in RUNTIME_EXTRA_PATHS:
        path = Path(name)
        if (repo / path).is_file():
            paths.add(path)
    return sorted(
        path
        for path in paths
        if path.suffix.lower() in RUNTIME_EXTENSIONS
        and "__pycache__" not in path.parts
    )


def create(output: Path, require_clean: bool) -> dict:
    repo = repository_root()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"source bundle output is not empty: {output}")

    # A training receipt promises clean *executable input*, not a cosmetically clean repository.
    # Large result tables and draft Markdown frequently coexist with a run and cannot change the
    # simulator.  Blocking on those files encouraged people to bypass the receipt altogether.
    # Restrict the hard gate to the exact roots snapshotted below, while retaining the whole-tree
    # state as transparent metadata.
    runtime_status = git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *RUNTIME_ROOTS,
        *RUNTIME_EXTRA_PATHS,
    ).splitlines()
    repository_status = git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    if require_clean and runtime_status:
        preview = "\n".join(runtime_status[:20])
        extra = max(0, len(runtime_status) - 20)
        if extra:
            preview += f"\n... and {extra} more paths"
        raise SystemExit(
            "refusing a clean-contract training receipt from dirty runtime sources:\n"
            + preview
        )

    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in runtime_paths(repo):
        source = (repo / relative).resolve()
        if not source.is_file() or repo not in source.parents:
            raise SystemExit(f"invalid runtime source path: {relative}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "snapshot": destination.relative_to(output).as_posix(),
            }
        )
    if not entries:
        raise SystemExit("no NavRL runtime sources found")

    environment = output / "python_environment.txt"
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        freeze = f"pip freeze failed ({exc.returncode})\n{exc.output}"
    environment.write_text(
        f"python_executable={Path(sys.executable).resolve()}\n"
        f"python_version={sys.version.replace(chr(10), ' ')}\n"
        f"platform={platform.platform()}\n\n[pip-freeze]\n{freeze}",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "purpose": "navrl_training_source_receipt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo),
        "git_commit": git(repo, "rev-parse", "HEAD").strip(),
        # Backward-compatible names: these now deliberately describe the snapshotted runtime
        # surface.  Repository-wide dirtiness is separately visible and is not a training blocker.
        "git_dirty": bool(runtime_status),
        "git_status": runtime_status,
        "repository_git_dirty": bool(repository_status),
        "repository_git_status": repository_status,
        "runtime_roots": list(RUNTIME_ROOTS),
        "runtime_file_count": len(entries),
        "runtime_files": entries,
        "python_environment": environment.relative_to(output).as_posix(),
        "python_environment_sha256": sha256_file(environment),
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "runtime_file_count": len(entries),
    }


def verify(manifest_path: Path, expected_sha256: str = "") -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"source manifest is missing: {manifest_path}")
    actual_manifest_sha = sha256_file(manifest_path)
    if expected_sha256 and actual_manifest_sha != expected_sha256:
        raise SystemExit("source manifest SHA-256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported training source manifest schema")
    repo = Path(manifest["repository_root"]).resolve()
    entries = manifest.get("runtime_files") or []
    if not entries or len(entries) != int(manifest.get("runtime_file_count", -1)):
        raise SystemExit("invalid runtime source accounting")
    for entry in entries:
        original = (repo / entry["path"]).resolve()
        snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
        expected = entry["sha256"]
        if not original.is_file() or sha256_file(original) != expected:
            raise SystemExit(f"runtime source changed: {entry['path']}")
        if not snapshot.is_file() or sha256_file(snapshot) != expected:
            raise SystemExit(f"runtime snapshot changed: {entry['snapshot']}")
    environment = (manifest_path.parent / manifest["python_environment"]).resolve()
    if (
        not environment.is_file()
        or sha256_file(environment) != manifest["python_environment_sha256"]
    ):
        raise SystemExit("Python environment receipt changed")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "git_commit": manifest["git_commit"],
        "git_dirty": bool(manifest["git_dirty"]),
        "runtime_file_count": len(entries),
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--output", required=True, type=Path)
    create_parser.add_argument("--require-clean", action="store_true")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True, type=Path)
    verify_parser.add_argument("--expected-sha256", default="")
    args = parser.parse_args()
    result = (
        create(args.output, args.require_clean)
        if args.command == "create"
        else verify(args.manifest, args.expected_sha256)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
