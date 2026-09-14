"""Validate Det-Fly and build a lightweight, immutable evaluation index.

The official download contains Pascal VOC XML files and JPEGs in two numbered groups (010 and
020).  Its README says the imagery spans sky, urban, field and mountain backgrounds, but neither
the XML nor the directory names provide that four-way label.  This tool therefore preserves the
official numbered group and records ``background=None`` instead of inventing a mapping.

No image is copied or recompressed.  The output is an index plus a receipt that binds it to the
download manifest.  The command fails closed on missing pairs, malformed XML/JPEG dimensions,
unexpected classes, or invalid boxes.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tool modules
from runtime_fingerprint import runtime_fingerprint  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = WORKSPACE / "datasets" / "detfly"
DEFAULT_OUTPUT = WORKSPACE / "datasets" / "detfly_index"
SIZE_EDGES_PX = (0.0, 8.0, 12.0, 20.0, 32.0, 64.0, 128.0, math.inf)
JPEG_SOF_MARKERS = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpeg_size(path):
    """Read JPEG width and height from a SOF marker without decoding the 4K image."""
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("missing JPEG SOI marker")
        while True:
            byte = stream.read(1)
            while byte and byte != b"\xff":
                byte = stream.read(1)
            if not byte:
                break
            marker_byte = stream.read(1)
            while marker_byte == b"\xff":
                marker_byte = stream.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in (0x01, 0xD8, 0xD9):
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError("invalid JPEG segment length")
            if marker in JPEG_SOF_MARKERS:
                header = stream.read(5)
                if len(header) != 5:
                    break
                height = int.from_bytes(header[1:3], "big")
                width = int.from_bytes(header[3:5], "big")
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive JPEG dimensions")
                return width, height
            stream.seek(length - 2, os.SEEK_CUR)
    raise ValueError("JPEG SOF marker not found")


def required_text(root, path):
    value = root.findtext(path)
    if value is None or not value.strip():
        raise ValueError("missing XML field " + path)
    return value.strip()


def size_bin(width, height):
    side = math.sqrt(max(width * height, 0.0))
    for index, (lower, upper) in enumerate(zip(SIZE_EDGES_PX, SIZE_EDGES_PX[1:])):
        if lower <= side < upper:
            return index
    raise AssertionError("unreachable")


def size_label(index):
    lower, upper = SIZE_EDGES_PX[index], SIZE_EDGES_PX[index + 1]
    return "%d-%spx" % (lower, "inf" if math.isinf(upper) else "%d" % upper)


def parse_annotation(xml_path, source):
    root = ET.parse(str(xml_path)).getroot()
    width = int(required_text(root, "size/width"))
    height = int(required_text(root, "size/height"))
    depth = int(required_text(root, "size/depth"))
    if width <= 0 or height <= 0 or depth not in (1, 3, 4):
        raise ValueError("invalid XML image dimensions")

    relative_xml = xml_path.relative_to(source)
    if len(relative_xml.parts) != 3:
        raise ValueError("expected Annotations/<group>/<stem>.xml")
    group = relative_xml.parts[1]
    image_path = source / "JPEGImages" / group / (xml_path.stem + ".jpg")
    if not image_path.is_file():
        raise FileNotFoundError(str(image_path))
    actual_width, actual_height = jpeg_size(image_path)
    if (actual_width, actual_height) != (width, height):
        raise ValueError("XML dimensions %sx%s != JPEG %sx%s" % (
            width, height, actual_width, actual_height))

    objects = []
    for obj in root.findall("object"):
        name = required_text(obj, "name")
        if name.casefold() != "uav":
            raise ValueError("unexpected class %r" % name)
        xmin = float(required_text(obj, "bndbox/xmin"))
        ymin = float(required_text(obj, "bndbox/ymin"))
        xmax = float(required_text(obj, "bndbox/xmax"))
        ymax = float(required_text(obj, "bndbox/ymax"))
        if not (0.0 <= xmin < xmax <= width and 0.0 <= ymin < ymax <= height):
            raise ValueError("invalid bounding box %r" % ([xmin, ymin, xmax, ymax],))
        difficult = int(obj.findtext("difficult", default="0"))
        truncated = int(obj.findtext("truncated", default="0"))
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise ValueError("difficult/truncated must be binary")
        # A box spanning half the frame cannot be a UAV at these ranges: the largest plausible
        # annotation in this distribution is 333 px tall and the 95th percentile diagonal is
        # 298 px. Exactly one box in 13,270 trips this (010/0106343: 3840x17, the full frame
        # width). It is FLAGGED, not dropped -- removing inconvenient ground truth before seeing
        # results is how a miss rate gets quietly improved. One box can move recall by 0.008 pp.
        implausible = (xmax - xmin) >= width * 0.5 or (ymax - ymin) >= height * 0.5
        objects.append({
            "class_id": 0,
            "class_name": "uav",
            "xyxy": [xmin, ymin, xmax, ymax],
            "difficult": bool(difficult),
            "truncated": bool(truncated),
            "implausible_size": implausible,
            "equivalent_side_px": math.sqrt((xmax - xmin) * (ymax - ymin)),
        })
    # Frames with no UAV are legitimate evidence for a detector: they measure false positives.
    # Exactly one frame in 13,271 has none (020/0201689). Refusing it would discard the only
    # negative frame the distribution offers.

    return {
        "image": str(image_path.relative_to(source)),
        "image_sha256": sha256_file(image_path),
        "annotation": str(relative_xml),
        "annotation_sha256": sha256_file(xml_path),
        "source_group": group,
        "background": None,
        "width": width,
        "height": height,
        "objects": objects,
    }


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("[detfly] refusing to overwrite existing index: %s" % output)
    download_receipt = source / "download_receipt.json"
    if not download_receipt.is_file():
        raise SystemExit("[detfly] missing download_receipt.json")
    upstream = json.loads(download_receipt.read_text())
    if upstream.get("verification_failures"):
        raise SystemExit("[detfly] download receipt contains verification failures")
    if upstream.get("verified_by_size") != upstream.get("files"):
        raise SystemExit("[detfly] download receipt is not complete")

    annotations = sorted((source / "Annotations").glob("*/*.xml"))
    images = sorted((source / "JPEGImages").glob("*/*.jpg"))
    if not annotations:
        raise SystemExit("[detfly] no annotation XML files found")

    rows = []
    failures = []
    groups = collections.Counter()
    size_bins = collections.Counter()
    difficult = truncated = boxes = 0
    for index, xml_path in enumerate(annotations, 1):
        try:
            row = parse_annotation(xml_path, source)
            rows.append(row)
            groups[row["source_group"]] += 1
            for obj in row["objects"]:
                boxes += 1
                difficult += int(obj["difficult"])
                truncated += int(obj["truncated"])
                size_bins[size_label(size_bin(
                    obj["xyxy"][2] - obj["xyxy"][0],
                    obj["xyxy"][3] - obj["xyxy"][1]))] += 1
        except Exception as exc:  # retain every bad path in the receipt before failing
            failures.append({"annotation": str(xml_path.relative_to(source)),
                             "error": "%s: %s" % (type(exc).__name__, exc)})
        if index % 1000 == 0:
            print("[detfly] validated %s/%s XML/JPEG pairs" % (index, len(annotations)), flush=True)

    indexed_images = {row["image"] for row in rows}
    extra_images = sorted(str(path.relative_to(source)) for path in images
                          if str(path.relative_to(source)) not in indexed_images)
    expected_count = int(upstream.get("files", 0)) // 2
    if len(annotations) != expected_count or len(images) != expected_count:
        failures.append({"error": "pair count does not match upstream manifest",
                         "annotations": len(annotations), "images": len(images),
                         "expected_each": expected_count})
    if extra_images:
        failures.append({"error": "images without matching annotation", "count": len(extra_images),
                         "examples": extra_images[:20]})

    if failures:
        print("[detfly] FAIL: %s validation failure(s); first: %s" % (len(failures), failures[0]))
        return 2

    output.mkdir(parents=True)
    index_path = output / "index.jsonl"
    with index_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    content_manifest = [
        {"image": row["image"], "image_sha256": row["image_sha256"],
         "annotation": row["annotation"], "annotation_sha256": row["annotation_sha256"]}
        for row in rows
    ]
    content_manifest_sha256 = hashlib.sha256(
        json.dumps(content_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "schema_version": 1,
        # What produced these numbers. Omitting it in 2026-09 made a cache that a
        # different torch/cuDNN could not reproduce look like non-determinism.
        "runtime": runtime_fingerprint(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_repository": "https://github.com/Jake-WU/Det-Fly",
        "download_receipt_sha256": sha256_file(download_receipt),
        "upstream_manifest_sha256": upstream.get("stable_manifest_sha256"),
        "index_sha256": sha256_file(index_path),
        "content_manifest_sha256": content_manifest_sha256,
        "content_hash_algorithm": "SHA-256 for every JPEG and XML; aggregate is SHA-256 of the stable JSON manifest",
        "images": len(rows),
        "boxes": boxes,
        "difficult_boxes": difficult,
        "truncated_boxes": truncated,
        "source_groups": dict(sorted(groups.items())),
        "size_bins_equivalent_side_px": dict(sorted(size_bins.items())),
        "background_labels": {
            "status": "UNAVAILABLE_IN_DISTRIBUTED_METADATA",
            "official_categories": ["sky", "urban", "field", "mountain"],
            "index_value": None,
            "substitute_slice": "source_group (010/020)",
        },
        "failures": [],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[detfly] PASS: %s images, %s boxes -> %s" % (len(rows), boxes, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
