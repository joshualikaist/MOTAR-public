"""Build an auditable NPS clip manifest with source-video-derived timestamps and GT boxes."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image

from perception_candidates import canonical_line, sha256_file


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = WORKSPACE / "datasets" / "nps_yolo"
FRAME_RE = re.compile(r"^(Clip_\d+)_(\d{6})$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_yolo_boxes(path, width, height):
    boxes = []
    for line in Path(path).read_text().splitlines():
        values = line.split()
        if len(values) != 5 or int(values[0]) != 0:
            raise ValueError("unexpected NPS label row in %s" % path)
        center_x, center_y, box_w, box_h = (float(value) for value in values[1:])
        x1 = max(0.0, (center_x - box_w / 2.0) * width)
        y1 = max(0.0, (center_y - box_h / 2.0) * height)
        x2 = min(float(width), (center_x + box_w / 2.0) * width)
        y2 = min(float(height), (center_y + box_h / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid NPS box in %s" % path)
        boxes.append([x1, y1, x2, y2])
    return boxes


def video_metadata(video):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("could not open source video %s" % video)
    result = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "source_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width_px": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height_px": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if result["fps"] <= 0.0:
        raise ValueError("invalid source FPS in %s" % video)
    return result


def build_rows(dataset, split):
    receipt = json.loads((dataset / "receipt.json").read_text())
    video_root = Path(receipt["source"]["videos"])
    allowed_clips = {clip for clip, value in receipt["clip_split"].items() if value == split}
    metadata = {clip: video_metadata(video_root / (clip + ".mov")) for clip in allowed_clips}
    rows = []
    for image_path in sorted((dataset / "images" / split).glob("*.jpg")):
        match = FRAME_RE.match(image_path.stem)
        if not match:
            raise ValueError("unexpected NPS frame name: %s" % image_path.name)
        clip, source_frame_number_text = match.groups()
        if clip not in allowed_clips:
            raise ValueError("frame clip is outside receipt split: %s" % clip)
        source_frame_number = int(source_frame_number_text)
        with Image.open(image_path) as image:
            width, height = image.size
        source = metadata[clip]
        if (width, height) != (source["width_px"], source["height_px"]):
            raise ValueError("decoded/source geometry mismatch for %s" % image_path)
        label_path = dataset / "labels" / split / (image_path.stem + ".txt")
        if not label_path.is_file():
            raise ValueError("missing label for %s" % image_path)
        frame_index = source_frame_number - 1
        rows.append({
            "schema_version": "motar.nps-sequence-manifest.v1",
            "frame_id": image_path.stem,
            "source_sequence_id": clip,
            "frame_index": frame_index,
            "source_frame_number": source_frame_number,
            "capture_timestamp_ns": int(round(frame_index * 1e9 / source["fps"])),
            "image": str(image_path.relative_to(dataset)),
            "width_px": width,
            "height_px": height,
            "ground_truth_xyxy": read_yolo_boxes(label_path, width, height),
        })
    rows.sort(key=lambda row: (int(row["source_sequence_id"].split("_")[1]), row["frame_index"]))
    return rows, metadata, receipt


def write_manifest(output, rows):
    digest = hashlib.sha256()
    with output.open("w") as stream:
        for row in rows:
            line = canonical_line(row)
            stream.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise SystemExit("[nps-manifest] refusing existing output/receipt")
    rows, videos, source_receipt = build_rows(dataset, args.split)
    if not rows:
        raise SystemExit("[nps-manifest] no frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_manifest(output, rows)
    sequences = Counter(row["source_sequence_id"] for row in rows)
    receipt = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "dataset": str(dataset),
        "source_receipt_sha256": sha256_file(dataset / "receipt.json"),
        "manifest": str(output),
        "manifest_sha256": digest,
        "frames": len(rows),
        "ground_truth_boxes": sum(len(row["ground_truth_xyxy"]) for row in rows),
        "sequences": dict(sorted(sequences.items())),
        "video_metadata": dict(sorted(videos.items())),
        "timestamp_method": "(one_based_source_frame_number - 1) / OpenCV container FPS",
        "source_split_seed": source_receipt["config"]["split_seed"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[nps-manifest] %s frames, %s sequences -> %s" % (
        len(rows), len(sequences), output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
