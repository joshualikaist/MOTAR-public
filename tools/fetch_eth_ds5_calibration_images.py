"""Fetch a pinned camera's published chessboard images, so the calibration can be recomputed.

The dataset publishes a calibration JSON per camera and, separately, the images it was made from. When a
published JSON does not describe the recording it is supposed to, the images are the only way to tell a
stale or mis-assigned file apart from a genuine lens difference. Every file is verified against its
upstream Git blob id, exactly as the metadata intake does; nothing is fetched that is not in the pinned
tree.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import sys

from prepare_eth_ds5 import COMMIT, REPO, digests, fetch_verified, safe_path, write_json


def image_paths(tree, camera):
    pattern = re.compile(r"calibration/%s/calibration_images/\d+\.jpg" % re.escape(camera))
    return sorted(e["path"] for e in tree["tree"] if e["type"] == "blob" and pattern.fullmatch(e["path"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="the existing pinned dataset root")
    parser.add_argument("--camera", default="sony5100")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    tree_path = args.output / "upstream_tree.json"
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated") or tree.get("sha") != COMMIT:
        raise ValueError("invalid cached tree")
    entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    paths = image_paths(tree, args.camera)
    if not paths:
        raise ValueError("no calibration images for " + args.camera)
    missing = sum(entries[p]["size"] for p in paths if not safe_path(args.output, p).exists())
    if shutil.disk_usage(args.output).free < missing + 512 * 1024 ** 2:
        raise ValueError("insufficient disk headroom; no files deleted")
    from concurrent.futures import ThreadPoolExecutor

    def fetch(path):
        sha = fetch_verified(args.output, entries[path])
        return {"path": path, "bytes": entries[path]["size"], "git_blob_sha1": entries[path]["sha"], "sha256": sha}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(fetch, paths))
    write_json(args.report, {"status": "CALIBRATION_IMAGES_VERIFIED", "source_commit": COMMIT,
                             "source_repository": REPO, "camera": args.camera, "count": len(files),
                             "bytes": sum(f["bytes"] for f in files), "files": files,
                             "tool_sha256": digests(Path(__file__))[1],
                             "runtime": {"python": platform.python_version(), "executable": sys.executable}})
    print(json.dumps({"camera": args.camera, "count": len(files),
                      "bytes": sum(f["bytes"] for f in files)}, indent=2))


if __name__ == "__main__":
    main()
