"""Extract only the verified cam0 MP4 from the pinned split ZIP into a new directory."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

from prepare_eth_ds5 import COMMIT, digests, safe_path, write_json


def video_member(listing):
    if "----------\n" not in listing:
        raise ValueError("missing archive member listing")
    blocks = listing.split("----------\n", 1)[1].strip().split("\n\n")
    entries = [dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line) for block in blocks]
    candidates = []
    for entry in entries:
        p = PurePosixPath(entry.get("Path", ""))
        if p.is_absolute() or ".." in p.parts or "\\" in str(p):
            raise ValueError("unsafe archive member")
        if p.name.lower() == "cam0.mp4":
            if any("link" in key.lower() for key in entry) or entry.get("Folder") == "+":
                raise ValueError("video must be a regular file")
            if int(entry["Size"]) <= 0:
                raise ValueError("empty video")
            candidates.append(entry)
    if len(candidates) != 1:
        raise ValueError("expected exactly one cam0.mp4")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sevenzip", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing extraction output")
    receipt = json.loads(args.receipt.read_text())
    if receipt["source_commit"] != COMMIT or not receipt["video_archives_downloaded"]:
        raise ValueError("intake is not a complete pinned video acquisition")
    prefix = "dataset5/videos/cam0/"
    entries = {f["path"]: f for f in receipt["files"] if f["path"].startswith(prefix) and f["path"].split(".")[-1] != "txt"}
    expected = {prefix + ("cam0.z%02d" % i) for i in range(1, 43)} | {prefix + "cam0.zip"}
    if set(entries) != expected:
        raise ValueError("missing/unexpected split archive parts")
    for name, entry in entries.items():
        path = safe_path(args.dataset, name)
        if path.stat().st_size != entry["bytes"] or digests(path) != (entry["git_blob_sha1"], entry["sha256"]):
            raise ValueError("archive hash mismatch: " + name)
    archive = args.dataset / prefix / "cam0.zip"
    listing = subprocess.check_output([args.sevenzip, "l", "-slt", str(archive)], text=True)
    member = video_member(listing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < int(member["Size"]) + 2 * 1024**3:
        raise ValueError("insufficient extraction headroom")
    args.output.mkdir()
    subprocess.run([args.sevenzip, "x", "-y", "-o" + str(args.output), str(archive), member["Path"]], check=True)
    video = safe_path(args.output, member["Path"])
    if not video.is_file() or video.is_symlink() or video.stat().st_size != int(member["Size"]):
        raise ValueError("extracted video size/type mismatch")
    result = {"status": "EXTRACTED_NOT_CALIBRATION_VALIDATED", "source_commit": COMMIT,
              "video": member["Path"], "video_bytes": video.stat().st_size,
              "video_sha256": digests(video)[1], "intake_receipt_sha256": digests(args.receipt)[1],
              "tool_sha256": digests(Path(__file__))[1],
              "sevenzip_binary_sha256": digests(Path(args.sevenzip))[1]}
    write_json(args.output / "extraction_receipt.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
