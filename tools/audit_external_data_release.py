"""Read-only release inventory. No dataset decoding, downloading or research computation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "f7496004a1027ecf21b666d6da2d040adb16dec3"
INVENTORY = ROOT / "docs/eth_ds5_public_release_inventory_2026-09-14.json"
EDITABLE_NOTES = {"results/eth_ds5_intake_2026-09-10/HUMAN_REVIEW_HANDOFF.md"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(revision):
    rows = {}
    for row in git("ls-tree", "-rlz", revision).split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, kind, oid, size = header.split()
        if kind == b"blob":
            rows[name.decode()] = {"git_blob": oid.decode(), "size": int(size)}
    return rows


def counts(rows):
    result = {"files": len(rows), "tracked_bytes": sum(r["size"] for r in rows.values())}
    for name, extensions in {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"},
                             "archive": {".zip", ".tar", ".gz", ".7z"},
                             "checkpoint": {".pth", ".pt"}, "npz": {".npz"}}.items():
        selected = [r for p, r in rows.items() if Path(p).suffix.lower() in extensions]
        result[name] = {"files": len(selected), "bytes": sum(r["size"] for r in selected)}
    return result


def snapshot():
    rows = tree(BASELINE)
    assets, evidence = [], {}
    for path, row in rows.items():
        if not path.startswith("results/eth_ds5_"):
            continue
        data = git("cat-file", "blob", row["git_blob"])
        if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".zip"}:
            evidence[path] = {"sha256": sha(data), "size": len(data),
                              "editorial_change_allowed": path in EDITABLE_NOTES}
            continue
        asset = dict(row, path=path, sha256=sha(data), classification="DERIVED_DATASET_ASSET",
                     type=Path(path).suffix[1:].upper(),
                     last_change_commit=git("log", "-1", "--format=%H", BASELINE, "--", path).decode().strip())
        # Lexical dependency candidates; no interpretation of research values.
        found = subprocess.run(["git", "grep", "-n", "-I", "-F", "-e", path,
                                "-e", Path(path).name, "-e", asset["sha256"], BASELINE, "--"],
                               cwd=str(ROOT), stdout=subprocess.PIPE)
        if found.returncode not in (0, 1):
            raise RuntimeError("dependency search failed")
        references = set()
        for line in found.stdout.decode().splitlines():
            _, source, number, _ = line.split(":", 3)
            references.add((source, int(number)))
        asset["reference_candidates"] = [{"path": p, "line": n} for p, n in sorted(references)]
        if asset["type"] == "ZIP":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                asset["members"] = [{"path": i.filename, "size": i.file_size,
                                     "sha256": sha(archive.read(i))} for i in archive.infolist()]
        assets.append(asset)
    if len(assets) != 14 or sum(a["type"] == "JPG" for a in assets) != 13:
        raise RuntimeError("baseline asset set differs; investigate before removal")
    return {"schema": "eth_ds5_release_inventory_v1", "baseline_commit": BASELINE,
            "scope": "Paths, hashes and lexical dependencies only; no scientific recomputation.",
            "before": counts(rows), "assets": assets, "preserved_evidence": evidence}


def audit(inventory):
    rows = tree("HEAD")
    indexed = set(git("ls-files", "-z").decode().split("\0"))
    reachable = {line.split()[0] for line in git("rev-list", "--objects", "--all").decode().splitlines()}
    assets = [{"path": a["path"], "tracked_in_index": a["path"] in indexed,
               "present_on_disk": (ROOT / a["path"]).exists(),
               "in_HEAD": a["path"] in rows,
               "historical_blob_reachable": a["git_blob"] in reachable} for a in inventory["assets"]]
    changed, notes = [], []
    for path, pin in inventory["preserved_evidence"].items():
        current = ROOT / path
        if not current.is_file() or sha(current.read_bytes()) != pin["sha256"]:
            (notes if pin["editorial_change_allowed"] else changed).append(path)
    # Renames cannot hide exact image bytes. Inspect all current blobs and ZIP members.
    banned = {a["sha256"] for a in inventory["assets"]}
    copies, archives = [], []
    for path, row in rows.items():
        data = git("cat-file", "blob", row["git_blob"])
        if sha(data) in banned:
            copies.append(path)
        if Path(path).suffix.lower() == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [{"path": i.filename, "size": i.file_size,
                            "sha256": sha(archive.read(i))} for i in archive.infolist()]
            archives.append({"path": path, "members": members})
            copies.extend(path + "!" + m["path"] for m in members if m["sha256"] in banned)
    unresolved = any(a["tracked_in_index"] or a["in_HEAD"] for a in assets) or bool(copies)
    return {"schema": "external_data_release_audit_v1", "commit": git("rev-parse", "HEAD").decode().strip(),
            "working_tree_dirty": bool(git("status", "--porcelain").strip()),
            "tool_sha256": sha(Path(__file__).read_bytes()),
            "CURRENT_TREE_BLOCKER": "OPEN" if unresolved else "RESOLVED",
            "HISTORICAL_REDISTRIBUTION_RISK": "OPEN" if any(a["historical_blob_reachable"] for a in assets) else "NOT_FOUND",
            "assets": assets, "unexpected_evidence_changes": changed, "editorial_notes_changed": notes,
            "before": inventory["before"], "after_HEAD": counts(rows),
            "exact_asset_copies": copies, "remaining_archive_inventory": archives,
            "limits": "Exact-content and ZIP-member scan, not visual similarity or full legal clearance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot() if args.mode == "snapshot" else audit(json.loads(INVENTORY.read_text()))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output)
    return int(bool(result.get("unexpected_evidence_changes")) or result.get("CURRENT_TREE_BLOCKER") == "OPEN")


if __name__ == "__main__":
    raise SystemExit(main())
