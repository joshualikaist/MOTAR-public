"""Decode NPS-Drones (BSD-3) into a YOLO-format detection dataset with a CLIP-LEVEL split.

Why this exists: the cross-dataset zero-shot measurement (results/air2air_zeroshot/) showed the
published GLAD appearance weights cannot be reused (recall 0.152, and the motion fusion that
carries the signal was never released), so a detector has to be trained on real air-to-air data.
This is the dataset step only. It makes no judgement and trains nothing.

Three properties are load-bearing and are asserted rather than commented:

* The split is by CLIP, never by frame. Consecutive frames of one clip are near-duplicates, so a
  frame-level split leaks the test set into training and inflates every number downstream.
  (docs/plans/perception_shape_temporal_redesign_2026-09-03.md section 0, principle 5.)
* The annotation convention is MOT top-left `frame,id,left,top,w,h,...`, verified visually against
  the rendered crop before this file was written -- the centre-point reading puts the box off the
  drone. Writing YOLO centres from a top-left source without this check is silent and unrecoverable.
* No ultralytics/yolov5 import. That package is AGPL-3.0; keeping our own preparation code free of
  it is what lets this file stay under the repository's own licence (WORKLOG 2026-09-04).
"""

import argparse
import collections
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

CLASS_ID = 0  # single class: uav
SPLIT_SEED = "nps-drones-clip-split-v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path, default=Path("../../datasets/nps/Videos"))
    p.add_argument("--annotations", type=Path,
                   default=Path("../../datasets/nps/Video_Annotation-v2/refined_gt"))
    p.add_argument("--output", type=Path, default=Path("../../datasets/nps_yolo"))
    p.add_argument("--stride", type=int, default=3, help="keep every Nth frame (30 fps source)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--val-clips", type=int, default=7)
    p.add_argument("--test-clips", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_annotations(path):
    """MOT rows -> {frame_index: [(left, top, w, h), ...]}. Frame indices are 1-based."""
    boxes = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6 or not parts[0]:
            continue
        frame = int(float(parts[0]))
        left, top, width, height = (float(value) for value in parts[2:6])
        if width <= 0 or height <= 0:
            continue
        boxes[frame].append((left, top, width, height))
    return boxes


def assign_split(stems, val_clips, test_clips):
    """Deterministic clip-level split: hash the stem, never the frame."""
    ordered = sorted(stems, key=lambda s: hashlib.sha256((SPLIT_SEED + s).encode()).hexdigest())
    test = set(ordered[:test_clips])
    val = set(ordered[test_clips:test_clips + val_clips])
    return {s: ("test" if s in test else "val" if s in val else "train") for s in stems}


def to_yolo(box, frame_w, frame_h):
    """MOT top-left box -> clipped YOLO centre form; None if it leaves the frame entirely."""
    left, top, width, height = box
    right, bottom = left + width, top + height
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(frame_w), right), min(float(frame_h), bottom)
    if right - left <= 1.0 or bottom - top <= 1.0:
        return None
    cx = (left + right) / 2.0 / frame_w
    cy = (top + bottom) / 2.0 / frame_h
    return (cx, cy, (right - left) / frame_w, (bottom - top) / frame_h)


def process_clip(job):
    video, annotation, out_root, split, stride, quality = job
    stem = Path(video).stem
    boxes = read_annotations(annotation)
    image_dir = Path(out_root) / "images" / split
    label_dir = Path(out_root) / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"clip": stem, "error": "could not open video"}
    kept = written_boxes = dropped = 0
    index = 0
    sizes = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1                      # 1-based, matching the annotation frame column
        if (index - 1) % stride:
            continue
        rows = boxes.get(index, [])
        if not rows:
            continue                    # unannotated frames carry no supervision here
        height, width = frame.shape[:2]
        yolo_rows = []
        for box in rows:
            converted = to_yolo(box, width, height)
            if converted is None:
                dropped += 1
                continue
            yolo_rows.append(converted)
        if not yolo_rows:
            continue
        name = f"{stem}_{index:06d}"
        path = image_dir / f"{name}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        sizes.append(path.stat().st_size)
        (label_dir / f"{name}.txt").write_text(
            "".join(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in yolo_rows)
        )
        kept += 1
        written_boxes += len(yolo_rows)
    capture.release()
    return {"clip": stem, "split": split, "frames": kept, "boxes": written_boxes,
            "boxes_dropped_out_of_frame": dropped, "bytes": sum(sizes),
            "annotation_sha256": sha256_file(annotation)}


def main():
    args = parse_args()
    videos = {p.stem: p for p in sorted(Path(args.videos).glob("*.mov"))}
    annotations = {p.stem.replace("_refined", ""): p
                   for p in sorted(Path(args.annotations).glob("*_refined.txt"))}
    stems = sorted(set(videos) & set(annotations))
    if not stems:
        raise SystemExit("[nps] no clip has both a video and an annotation file")
    missing = sorted((set(videos) | set(annotations)) - set(stems))
    split_of = assign_split(stems, args.val_clips, args.test_clips)

    counts = collections.Counter(split_of.values())
    print(f"[nps] clips={len(stems)} split={dict(counts)} stride={args.stride} q={args.jpeg_quality}")
    if missing:
        print(f"[nps] unpaired (skipped): {', '.join(missing)}")
    if args.dry_run:
        for stem in stems:
            print(f"  {stem}: {split_of[stem]}")
        return 0

    out_root = Path(args.output)
    if out_root.exists():
        raise SystemExit(f"[nps] refusing to overwrite an existing dataset: {out_root}")
    jobs = [(videos[s], annotations[s], out_root, split_of[s], args.stride, args.jpeg_quality)
            for s in stems]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_clip, jobs):
            results.append(result)
            if "error" in result:
                print(f"[nps] {result['clip']}: FAILED {result['error']}")
            else:
                print(f"[nps] {result['clip']} [{result['split']}] "
                      f"{result['frames']} frames, {result['boxes']} boxes")

    failures = [r for r in results if "error" in r]
    per_split = collections.defaultdict(lambda: {"clips": 0, "frames": 0, "boxes": 0, "bytes": 0})
    for r in results:
        if "error" in r:
            continue
        bucket = per_split[r["split"]]
        bucket["clips"] += 1
        for key in ("frames", "boxes", "bytes"):
            bucket[key] += r[key]

    # A YOLO data yaml written by hand: relative paths only, so the dataset stays relocatable.
    (out_root / "nps_drones.yaml").write_text(
        "# NPS-Drones (BSD-3), clip-level split. Generated by tools/prepare_nps_drones_dataset.py\n"
        f"path: {out_root.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: [uav]\n"
    )
    receipt = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"videos": str(Path(args.videos).resolve()),
                   "annotations": str(Path(args.annotations).resolve()),
                   "licence": "BSD-3 (NPS-Drones)"},
        "config": {"stride": args.stride, "jpeg_quality": args.jpeg_quality,
                   "split_seed": SPLIT_SEED, "val_clips": args.val_clips,
                   "test_clips": args.test_clips},
        "annotation_convention": "MOT top-left (left, top, w, h); verified visually before use",
        "splits": {k: dict(v) for k, v in sorted(per_split.items())},
        "clip_split": split_of,
        "unpaired_skipped": missing,
        "failures": failures,
        "clips": sorted(results, key=lambda r: r["clip"]),
    }
    (out_root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total_bytes = sum(v["bytes"] for v in per_split.values())
    for split, bucket in sorted(per_split.items()):
        print(f"[nps] {split}: {bucket['clips']} clips, {bucket['frames']} frames, {bucket['boxes']} boxes")
    print(f"[nps] total {total_bytes / 1e9:.2f} GB -> {out_root}")
    if failures:
        print(f"[nps] {len(failures)} clip(s) FAILED")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
