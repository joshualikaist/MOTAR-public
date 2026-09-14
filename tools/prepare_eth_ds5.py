"""Pinned ETH ds5 intake; CPU-only, no detector fitting or range-error claims.

Metadata is fetched by default. --video explicitly fetches cam0 split archives.
Every file is checked against its upstream Git blob id and a local SHA-256.
Incomplete downloads resume only after a matching HTTP Content-Range response.
"""
import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import time
import urllib.request

COMMIT = "2c857c97be71834d0791ae8ee4984ffb62b7680a"
REPO = "CenekAlbl/drone-tracking-datasets"
RAW = "https://raw.githubusercontent.com/" + REPO + "/" + COMMIT + "/"
META = (
    "LICENSE", "dataset5/LICENSE", "dataset5/README.md", "dataset5/cameras.txt",
    "dataset5/drones.txt", "dataset5/camera-locations/campos.txt",
    "dataset5/pose/fused_pose.txt", "dataset5/videos/cam0/cam0_frame_ts.txt",
    "dataset5/raw-data/sync_coefficients_cam2pc.txt",
    "dataset5/raw-data/leverarm_prism_in_body.txt",
    "dataset5/raw-data/tran_mat_tps2local.txt",
    "dataset5/raw-data/project_time_origin_in_utc.txt",
    "calibration/sony5100/sony5100.json",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError("unsafe upstream path")
    target = root / relative
    if root.resolve() not in target.resolve().parents:
        raise ValueError("path escapes output root")
    return target


def digests(path):
    blob = hashlib.sha1(("blob %d\0" % path.stat().st_size).encode())
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            blob.update(chunk)
            sha.update(chunk)
    return blob.hexdigest(), sha.hexdigest()


def download(root, entry, opener=urllib.request.urlopen):
    path = safe_path(root, entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        blob, sha = digests(path)
        if path.stat().st_size != entry["size"] or blob != entry["sha"]:
            raise ValueError("existing file is corrupt; preserved: " + str(path))
        return sha
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink():
        raise ValueError("refusing symlink partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size"]:
        raise ValueError("oversize partial; preserved")
    if offset < entry["size"]:
        request = urllib.request.Request(RAW + entry["path"], headers={
            "Range": "bytes=%d-" % offset, "Accept-Encoding": "identity",
            "User-Agent": "MOTAR-ETH-intake"})
        with opener(request, timeout=30) as response:
            status = response.status
            if status == 206:
                expected = "bytes %d-%d/%d" % (offset, entry["size"] - 1, entry["size"])
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("Content-Range mismatch; partial preserved")
            elif status != 200 or offset:
                raise ValueError("server did not honor resume; partial preserved")
            with partial.open("ab" if offset else "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    if stream.tell() + len(chunk) > entry["size"]:
                        raise ValueError("response exceeds upstream size")
                    stream.write(chunk)
    if partial.stat().st_size != entry["size"]:
        raise ValueError("incomplete download; rerun to resume")
    blob, sha = digests(partial)
    if blob != entry["sha"]:
        raise ValueError("Git blob mismatch; partial preserved")
    partial.replace(path)
    return sha


def numeric_table(path, columns):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        row = list(map(float, line.split()))
        if len(row) != columns or not all(map(math.isfinite, row)):
            raise ValueError("malformed numeric row: " + str(path))
        rows.append(row)
    if not rows:
        raise ValueError("empty table")
    return rows


def fetch_verified(root, entry, attempts=8):
    """Retry interrupted transport, never retry semantic/hash failures silently."""
    for attempt in range(attempts):
        try:
            return download(root, entry)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and not str(exc).startswith("incomplete download"):
                raise
            print("RETRY", entry["path"], attempt + 1, type(exc).__name__, str(exc), flush=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(attempt + 1, 5))


def increasing(values):
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("timestamps/ids must be strictly increasing")


def audit(pose, frames, camera):
    increasing([r[0] for r in pose])
    increasing([r[0] for r in frames])
    increasing([r[1] for r in frames])
    if any(r[0] < 1 or not r[0].is_integer() for r in frames):
        raise ValueError("frame ids must be positive integers")
    if any(not r[-1].is_integer() or any(s < 0 for s in r[7:10]) for r in pose):
        raise ValueError("invalid status or position uncertainty")
    overlap = [r for r in frames if pose[0][0] <= r[1] <= pose[-1][0]]
    ranges = [math.sqrt(sum((r[i+1]-camera[i])**2 for i in range(3))) for r in pose]
    return {
        "status": "METADATA_VERIFIED_MEASUREMENT_BLOCKED",
        "target": "drone0 / Pixhawk only", "pose_rows": len(pose),
        "pose_time_s": [pose[0][0], pose[-1][0]],
        "video_timestamp_rows": len(frames), "video_time_s": [frames[0][1], frames[-1][1]],
        "temporal_overlap_frames": len(overlap),
        "overlap_time_s": [overlap[0][1], overlap[-1][1]] if overlap else None,
        "tracking_status_counts_uninterpreted": dict(Counter(str(int(r[-1])) for r in pose)),
        "raw_slant_range_m_not_quality_filtered": [min(ranges), max(ranges)],
        "slant_range_is_not_optical_depth": True,
        "frame_timestamps_already_project_time": True,
        "blockers": ["verify calibration against ds5 lens/crop settings",
                     "verify TrackingStatus semantics and attitude reference conventions",
                     "identify drone0 and obtain independently reviewed bounding boxes",
                     "validate camera orientation/reprojection before optical-depth claims"],
        "range_model_fit": False, "ppo_started": False,
    }


def annotation_queue(pose, frames, stride=150):
    if stride < 1:
        raise ValueError("stride must be positive")
    times = [r[0] for r in pose]
    overlap = [r for r in frames if times[0] <= r[1] <= times[-1]]
    queue = []
    for frame, timestamp in overlap[::stride]:
        right = min(bisect.bisect_left(times, timestamp), len(times) - 1)
        left = max(0, right - 1)
        queue.append({"frame_id": int(frame), "opencv_index": int(frame)-1,
                      "project_timestamp_s": timestamp, "target_id": "drone0",
                      "pose_bracket_rows_zero_based": [left, right],
                      "pose_bracket_gap_s": times[right] - times[left],
                      "tracking_status_uninterpreted": [pose[left][-1], pose[right][-1]],
                      "box_xyxy": None, "identity_verified": False,
                      "annotation_status": "PENDING_MANUAL_REVIEW",
                      "measurement_eligible": False})
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    args.output.mkdir(parents=True, exist_ok=True)
    tree_path = args.output / "upstream_tree.json"
    if not tree_path.exists():
        url = "https://api.github.com/repos/%s/git/trees/%s?recursive=1" % (REPO, COMMIT)
        with urllib.request.urlopen(url, timeout=30) as response:
            tree = json.load(response)
        if tree.get("truncated") or tree.get("sha") != COMMIT:
            raise ValueError("incomplete or wrong upstream tree")
        write_json(tree_path, tree)
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated") or tree.get("sha") != COMMIT:
        raise ValueError("invalid cached tree")
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    archives = sorted(p for p in entries if re.fullmatch(r"dataset5/videos/cam0/cam0\.(z\d+|zip)", p))
    paths = list(META) + (archives if args.video else [])
    missing = sum(entries[p]["size"] for p in paths if not safe_path(args.output, p).exists())
    # Reserve another archive-sized footprint for eventual extraction, plus 2 GiB headroom.
    reserve = sum(entries[p]["size"] for p in archives) if args.video else 0
    if shutil.disk_usage(args.output).free < missing + reserve + 2 * 1024**3:
        raise ValueError("insufficient disk headroom; no files deleted")
    def fetch(p):
        sha = fetch_verified(args.output, entries[p])
        print("VERIFIED", p, flush=True)
        return {"path": p, "bytes": entries[p]["size"],
                "git_blob_sha1": entries[p]["sha"], "sha256": sha}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(fetch, paths))
    pose = numeric_table(args.output / "dataset5/pose/fused_pose.txt", 11)
    frames = numeric_table(args.output / "dataset5/videos/cam0/cam0_frame_ts.txt", 2)
    camera_lines = (args.output / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t"))
    report = audit(pose, frames, camera)
    report.update({"source_commit": COMMIT, "source_repository": REPO,
                   "license": "CC-BY-NC-SA-4.0", "cam0_archive_bytes": sum(entries[p]["size"] for p in archives),
                   "video_archives_downloaded": args.video, "video_extracted": False,
                   "files": files, "runtime": {"python": platform.python_version(), "executable": sys.executable},
                   "tool_sha256": digests(Path(__file__))[1]})
    write_json(args.report, report)
    write_json(args.report.with_name("annotation_queue.json"), {
        "source_commit": COMMIT, "purpose": "pilot annotation feasibility, not a measurement split",
        "stride_frames": 150, "frames": annotation_queue(pose, frames)})
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
