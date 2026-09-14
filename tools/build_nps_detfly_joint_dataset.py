"""Audit Det-Fly splits and build the NPS+Det-Fly detector dataset.

The distributed Det-Fly metadata has no video id.  Random frame splitting would therefore leak
adjacent frames.  This builder uses only source structure -- never detector predictions:

* the larger official source group is held out whole as test;
* the other group is divided at its largest numeric frame-id gap;
* the larger side of that gap is train and the smaller side is validation.

The sealed test group is absent from the YOLO training YAML.  NPS train/validation tiles are
referenced in text manifests without copying them.  Det-Fly train/validation frames are cropped at
native scale with the same 640/128 sliding-window geometry used by P3.  All positive tiles and one
deterministic, target-free negative tile per usable frame are materialized because a symlink cannot
represent a crop.  The original 4K JPEGs are never copied or modified.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DETFLY = WORKSPACE / "datasets" / "detfly"
DEFAULT_DETFLY_INDEX = WORKSPACE / "datasets" / "detfly_index"
DEFAULT_NPS_TILES = WORKSPACE / "datasets" / "nps_yolo_tiles640"
DEFAULT_OUTPUT = WORKSPACE / "datasets" / "nps_detfly_joint_v1"
TILE = 640
OVERLAP = 128
VISIBLE_FRACTION = 0.60
MIN_VISIBLE_SIDE_PX = 4.0
NEGATIVES_PER_FRAME = 1
JPEG_QUALITY = 90
NEGATIVE_SEED = "motar-nps-detfly-joint-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detfly", type=Path, default=DEFAULT_DETFLY)
    parser.add_argument("--detfly-index", type=Path, default=DEFAULT_DETFLY_INDEX)
    parser.add_argument("--nps-tiles", type=Path, default=DEFAULT_NPS_TILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-perceptual-audit", action="store_true",
                        help="Skip the diagnostic dHash audit; structural and SHA audits still run.")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("w") as stream:
        for row in rows:
            line = canonical_bytes(row) + b"\n"
            stream.write(line.decode())
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_index(path):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except Exception as error:
                raise ValueError("invalid index JSON on line %d: %s" % (line_number, error))
            rows.append(row)
    return rows


def numeric_frame_id(row):
    group = str(row["source_group"])
    stem = Path(row["image"]).stem
    if not stem.startswith(group):
        raise ValueError("image stem %s does not start with source group %s" % (stem, group))
    suffix = stem[len(group):]
    if not suffix.isdigit():
        raise ValueError("non-numeric Det-Fly frame suffix: %s" % stem)
    return int(suffix)


def choose_structural_split(rows):
    """Return split assignments using source groups and the largest id gap only."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row["source_group"])].append(row)
    if len(groups) != 2:
        raise ValueError("expected exactly two Det-Fly source groups, found %r" % sorted(groups))
    ordered_groups = sorted(groups, key=lambda group: (-len(groups[group]), group))
    test_group, development_group = ordered_groups
    development = sorted(groups[development_group], key=numeric_frame_id)
    gaps = []
    for index, (left, right) in enumerate(zip(development, development[1:]), 1):
        left_id = numeric_frame_id(left)
        right_id = numeric_frame_id(right)
        gaps.append((right_id - left_id, -left_id, index, left_id, right_id))
    if not gaps:
        raise ValueError("development source group has fewer than two frames")
    gap_width, _negative_left, cut, left_id, right_id = max(gaps)
    if gap_width <= 1:
        raise ValueError("no natural numeric gap exists for train/validation separation")
    sides = [development[:cut], development[cut:]]
    train_side = 0 if len(sides[0]) >= len(sides[1]) else 1
    assignments = {}
    for row in groups[test_group]:
        assignments[row["image"]] = "test"
    for side_index, side in enumerate(sides):
        split = "train" if side_index == train_side else "val"
        for row in side:
            assignments[row["image"]] = split
    if len(assignments) != len(rows):
        raise AssertionError("split assignment count mismatch")
    policy = {
        "algorithm": "largest_source_group_is_test; other_group_largest_numeric_gap; larger_side_is_train",
        "selection_inputs": ["source_group", "numeric frame id", "group/frame counts"],
        "prediction_metrics_read_by_builder": False,
        "post_p3_split_warning": (
            "The split was created after the zero-shot P3 report existed. The executable rule does "
            "not read predictions, confidence, boxes, background appearance, or P3 metrics."
        ),
        "test_group": test_group,
        "development_group": development_group,
        "development_gap": {
            "left_id": left_id,
            "right_id": right_id,
            "id_difference": gap_width,
            "missing_ids_between": gap_width - 1,
        },
        "train_side": "lower_ids" if train_side == 0 else "higher_ids",
    }
    return assignments, policy


def split_audit(rows, assignments, policy):
    split_rows = collections.defaultdict(list)
    for row in rows:
        split_rows[assignments[row["image"]]].append(row)
    digest_splits = collections.defaultdict(set)
    for row in rows:
        digest_splits[row["image_sha256"]].add(assignments[row["image"]])
    cross_split_duplicates = sorted(
        digest for digest, splits in digest_splits.items() if len(splits) > 1
    )
    if cross_split_duplicates:
        raise ValueError("exact JPEG duplicates cross splits: %d" % len(cross_split_duplicates))
    test_groups = sorted(set(str(row["source_group"]) for row in split_rows["test"]))
    development_groups = sorted(set(
        str(row["source_group"]) for split in ("train", "val") for row in split_rows[split]
    ))
    if set(test_groups) & set(development_groups):
        raise ValueError("source group crosses the test/development boundary")
    summary = {}
    for split in ("train", "val", "test"):
        selected = split_rows[split]
        summary[split] = {
            "images": len(selected),
            "boxes": sum(len(row["objects"]) for row in selected),
            "source_groups": dict(sorted(collections.Counter(
                str(row["source_group"]) for row in selected).items()
            )),
            "negative_frames": sum(not row["objects"] for row in selected),
            "implausible_boxes_flagged": sum(
                bool(obj.get("implausible_size")) for row in selected for obj in row["objects"]
            ),
        }
    return {
        "splits": summary,
        "test_source_group_disjoint": True,
        "exact_cross_split_jpeg_duplicates": 0,
        "exact_unique_jpeg_sha256": len(digest_splits),
        "development_boundary": policy["development_gap"],
    }


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def perceptual_audit(rows, detfly, assignments):
    """Diagnose low-frequency similarity; never use it to choose the split."""
    import cv2
    import numpy as np
    from PIL import Image

    descriptors = collections.defaultdict(list)
    thumbnail_hash_splits = collections.defaultdict(set)
    group_previous = {}
    adjacent_mae = collections.defaultdict(list)
    across_gap_mae = collections.defaultdict(list)
    ordered = sorted(rows, key=lambda row: (str(row["source_group"]), numeric_frame_id(row)))
    started = time.monotonic()
    for index, row in enumerate(ordered, 1):
        image_path = detfly / row["image"]
        with Image.open(str(image_path)) as image:
            image.draft("RGB", (160, 90))
            rgb = np.asarray(image.convert("RGB").resize(
                (64, 36), Image.Resampling.BILINEAR), dtype=np.uint8)
        gray = ((rgb[:, :, 0].astype(np.uint16) * 77
                 + rgb[:, :, 1].astype(np.uint16) * 150
                 + rgb[:, :, 2].astype(np.uint16) * 29) >> 8).astype(np.uint8)
        small = np.asarray(Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR))
        descriptor = np.packbits((small[:, 1:] > small[:, :-1]).reshape(-1))
        split = assignments[row["image"]]
        descriptors[split].append(descriptor)
        thumbnail_hash_splits[sha256_bytes(rgb.tobytes())].add(split)
        group = str(row["source_group"])
        previous = group_previous.get(group)
        if previous is not None:
            previous_id, previous_rgb = previous
            difference = float(np.abs(rgb.astype(np.int16) - previous_rgb.astype(np.int16)).mean())
            target = adjacent_mae if numeric_frame_id(row) == previous_id + 1 else across_gap_mae
            target[group].append(difference)
        group_previous[group] = (numeric_frame_id(row), rgb)
        if index % 500 == 0:
            print("[joint] perceptual audit %d/%d images (%.1fs)" % (
                index, len(ordered), time.monotonic() - started), flush=True)

    development = np.asarray(descriptors["train"] + descriptors["val"], dtype=np.uint8)
    test = np.asarray(descriptors["test"], dtype=np.uint8)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).match(development, test)
    distances = [float(match.distance) for match in matches]
    thumbnail_cross_split = sum(len(splits) > 1 for splits in thumbnail_hash_splits.values())
    return {
        "method": "64x36 RGB thumbnail + 64-bit difference hash",
        "role": "diagnostic_only_not_a_split_input",
        "exact_64x36_thumbnail_hashes_crossing_splits": thumbnail_cross_split,
        "development_to_test_nearest_dhash_hamming": {
            "min": percentile(distances, 0),
            "p01": percentile(distances, 1),
            "p05": percentile(distances, 5),
            "p50": percentile(distances, 50),
            "max": percentile(distances, 100),
            "at_most_2": sum(distance <= 2 for distance in distances),
            "at_most_4": sum(distance <= 4 for distance in distances),
        },
        "caveat": (
            "Low-entropy sky frames collide under dHash, so low Hamming distance is not treated as "
            "duplicate evidence. Exact JPEG SHA and official source-group isolation are the gates."
        ),
        "consecutive_rgb_mae": {
            group: {
                "adjacent_p50": percentile(adjacent_mae[group], 50),
                "adjacent_p95": percentile(adjacent_mae[group], 95),
                "gap_p50": percentile(across_gap_mae[group], 50),
                "gap_p95": percentile(across_gap_mae[group], 95),
            } for group in sorted(adjacent_mae)
        },
    }


def tile_origins(length, tile=TILE, overlap=OVERLAP):
    if length <= tile:
        return [0]
    step = tile - overlap
    origins = list(range(0, length - tile + 1, step))
    final = length - tile
    if origins[-1] != final:
        origins.append(final)
    return origins


def intersection(box, origin_x, origin_y, tile=TILE):
    xmin, ymin, xmax, ymax = [float(value) for value in box]
    left = max(xmin, float(origin_x))
    top = max(ymin, float(origin_y))
    right = min(xmax, float(origin_x + tile))
    bottom = min(ymax, float(origin_y + tile))
    if right <= left or bottom <= top:
        return None
    return [left - origin_x, top - origin_y, right - origin_x, bottom - origin_y]


def visible_labels(objects, origin_x, origin_y):
    labels = []
    for obj in objects:
        box = [float(value) for value in obj["xyxy"]]
        clipped = intersection(box, origin_x, origin_y)
        if clipped is None:
            continue
        original_area = (box[2] - box[0]) * (box[3] - box[1])
        visible_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if visible_area / original_area < VISIBLE_FRACTION:
            continue
        if clipped[2] - clipped[0] < MIN_VISIBLE_SIDE_PX:
            continue
        if clipped[3] - clipped[1] < MIN_VISIBLE_SIDE_PX:
            continue
        labels.append(clipped)
    return labels


def has_any_intersection(objects, origin_x, origin_y):
    return any(intersection(obj["xyxy"], origin_x, origin_y) is not None for obj in objects)


def choose_tiles(row):
    if any(bool(obj.get("implausible_size")) for obj in row["objects"]):
        return [], "implausible_annotation"
    objects = list(row["objects"])
    candidates = [(x, y) for y in tile_origins(int(row["height"]))
                  for x in tile_origins(int(row["width"]))]
    positives = []
    negative_candidates = []
    for x, y in candidates:
        labels = visible_labels(objects, x, y)
        if labels:
            positives.append((x, y, labels, "positive"))
        elif not has_any_intersection(objects, x, y):
            negative_candidates.append((x, y))
    ranked_negatives = sorted(
        negative_candidates,
        key=lambda origin: sha256_bytes((
            NEGATIVE_SEED + "|" + row["image_sha256"] + "|%d|%d" % origin
        ).encode()),
    )[:NEGATIVES_PER_FRAME]
    selected = positives + [(x, y, [], "negative") for x, y in ranked_negatives]
    return sorted(selected, key=lambda item: (item[1], item[0], item[3])), None


def yolo_line(box):
    xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2.0 / TILE
    center_y = (ymin + ymax) / 2.0 / TILE
    width = (xmax - xmin) / TILE
    height = (ymax - ymin) / TILE
    values = (center_x, center_y, width, height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized YOLO box outside [0,1]: %r" % (values,))
    return "0 %.8f %.8f %.8f %.8f" % values


def detfly_split_rows(rows, assignments):
    for row in sorted(rows, key=lambda item: (str(item["source_group"]), numeric_frame_id(item))):
        copy = dict(row)
        copy["frame_id"] = numeric_frame_id(row)
        copy["split"] = assignments[row["image"]]
        yield copy


def materialize_detfly(rows, assignments, detfly, temporary, final_output):
    from PIL import Image, __version__ as pillow_version

    manifest = []
    counters = collections.Counter()
    selected = [row for row in rows if assignments[row["image"]] in ("train", "val")]
    started = time.monotonic()
    for frame_index, row in enumerate(selected, 1):
        split = assignments[row["image"]]
        chosen, exclusion = choose_tiles(row)
        if exclusion is not None:
            counters["excluded_frames_" + exclusion] += 1
            continue
        source_image = detfly / row["image"]
        with Image.open(str(source_image)) as opened:
            image = opened.convert("RGB")
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError("image size drift for %s" % source_image)
            for x, y, boxes, kind in chosen:
                stem = "detfly_%s_x%d_y%d" % (Path(row["image"]).stem, x, y)
                relative_image = Path("images") / split / (stem + ".jpg")
                relative_label = Path("labels") / split / (stem + ".txt")
                image_path = temporary / relative_image
                label_path = temporary / relative_label
                crop = image.crop((x, y, x + TILE, y + TILE))
                if crop.size != (TILE, TILE):
                    raise ValueError("non-640 crop for %s" % source_image)
                crop.save(str(image_path), format="JPEG", quality=JPEG_QUALITY, subsampling=2)
                label_text = "\n".join(yolo_line(box) for box in boxes)
                if label_text:
                    label_text += "\n"
                label_path.write_text(label_text)
                record = {
                    "dataset": "detfly",
                    "split": split,
                    "kind": kind,
                    "source_image": row["image"],
                    "source_image_sha256": row["image_sha256"],
                    "source_group": row["source_group"],
                    "origin_xy": [x, y],
                    "image": str(final_output / relative_image),
                    "image_sha256": sha256_file(image_path),
                    "label": str(final_output / relative_label),
                    "label_sha256": sha256_file(label_path),
                    "boxes": len(boxes),
                }
                manifest.append(record)
                counters[split + "_tiles"] += 1
                counters[split + "_" + kind + "_tiles"] += 1
                counters[split + "_boxes"] += len(boxes)
        if frame_index % 250 == 0:
            print("[joint] Det-Fly tiles %d/%d frames, %d tiles (%.1fs)" % (
                frame_index, len(selected), len(manifest), time.monotonic() - started), flush=True)
    return manifest, dict(sorted(counters.items())), pillow_version


def nps_manifest(nps_tiles):
    records = []
    for split in ("train", "val"):
        image_dir = nps_tiles / "images" / split
        label_dir = nps_tiles / "labels" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.is_file():
                raise FileNotFoundError(str(label_path))
            label_lines = [line for line in label_path.read_text().splitlines() if line.strip()]
            records.append({
                "dataset": "nps",
                "split": split,
                "kind": "positive" if label_lines else "negative",
                "source_unit": image_path.stem.split("_", 2)[0] + "_" + image_path.stem.split("_", 2)[1],
                "image": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "label": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "boxes": len(label_lines),
            })
    return records


def validate_inputs(detfly, detfly_index, nps_tiles):
    index_path = detfly_index / "index.jsonl"
    index_receipt_path = detfly_index / "receipt.json"
    download_receipt_path = detfly / "download_receipt.json"
    nps_receipt_path = nps_tiles / "receipt.json"
    for path in (index_path, index_receipt_path, download_receipt_path, nps_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    index_receipt = load_json(index_receipt_path)
    if sha256_file(index_path) != index_receipt.get("index_sha256"):
        raise ValueError("Det-Fly index SHA-256 does not match its receipt")
    if index_receipt.get("images") != 13271 or index_receipt.get("failures"):
        raise ValueError("Det-Fly index is incomplete")
    nps_receipt = load_json(nps_receipt_path)
    if nps_receipt.get("test_split") != "not tiled on purpose; judge whole frames with a sliding window":
        raise ValueError("NPS test split contract drift")
    return {
        "detfly_index_sha256": sha256_file(index_path),
        "detfly_index_receipt_sha256": sha256_file(index_receipt_path),
        "detfly_download_receipt_sha256": sha256_file(download_receipt_path),
        "nps_tile_receipt_sha256": sha256_file(nps_receipt_path),
    }


def build(args):
    detfly = args.detfly.resolve()
    detfly_index = args.detfly_index.resolve()
    nps_tiles = args.nps_tiles.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("[joint] refusing to overwrite existing output: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s.building." % output.name, dir=str(output.parent)))
    try:
        for split in ("train", "val"):
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
        (temporary / "manifests").mkdir()
        (temporary / "receipts").mkdir()

        inputs = validate_inputs(detfly, detfly_index, nps_tiles)
        rows = load_index(detfly_index / "index.jsonl")
        assignments, policy = choose_structural_split(rows)
        audit = split_audit(rows, assignments, policy)
        split_count, split_manifest_sha = write_jsonl(
            temporary / "manifests" / "detfly_split.jsonl",
            detfly_split_rows(rows, assignments),
        )
        test_rows = [row for row in detfly_split_rows(rows, assignments) if row["split"] == "test"]
        test_count, test_manifest_sha = write_jsonl(
            temporary / "manifests" / "detfly_test_sealed.jsonl", test_rows
        )
        perceptual = None
        if not args.skip_perceptual_audit:
            perceptual = perceptual_audit(rows, detfly, assignments)
        split_receipt = {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "policy": policy,
            "audit": audit,
            "perceptual_audit": perceptual,
            "detfly_index_sha256": inputs["detfly_index_sha256"],
            "split_manifest_rows": split_count,
            "split_manifest_sha256": split_manifest_sha,
            "sealed_test_manifest_rows": test_count,
            "sealed_test_manifest_sha256": test_manifest_sha,
            "training_yaml_contains_test": False,
        }
        write_json(temporary / "receipts" / "detfly_split_receipt.json", split_receipt)

        detfly_records, detfly_counts, pillow_version = materialize_detfly(
            rows, assignments, detfly, temporary, output
        )
        print("[joint] hashing existing NPS tile set", flush=True)
        nps_records = nps_manifest(nps_tiles)
        joint_records = sorted(nps_records + detfly_records,
                               key=lambda row: (row["split"], row["dataset"], row["image"]))
        joint_count, joint_manifest_sha = write_jsonl(
            temporary / "manifests" / "joint_samples.jsonl", joint_records
        )
        split_paths = collections.defaultdict(list)
        for record in joint_records:
            split_paths[record["split"]].append(record["image"])
        for split in ("train", "val"):
            (temporary / (split + ".txt")).write_text("\n".join(split_paths[split]) + "\n")
        yaml_text = (
            "# NPS + Det-Fly v1. Test is deliberately absent; see sealed test receipt.\n"
            "path: %s\n"
            "train: train.txt\n"
            "val: val.txt\n"
            "nc: 1\n"
            "names: [uav]\n" % output
        )
        if "test:" in yaml_text:
            raise AssertionError("training YAML unexpectedly exposes test")
        (temporary / "joint_detector.yaml").write_text(yaml_text)

        dataset_counts = {}
        for split in ("train", "val"):
            records = [row for row in joint_records if row["split"] == split]
            dataset_counts[split] = {
                "samples": len(records),
                "boxes": sum(row["boxes"] for row in records),
                "by_dataset": dict(sorted(collections.Counter(row["dataset"] for row in records).items())),
                "positive_samples": sum(row["kind"] == "positive" for row in records),
                "negative_samples": sum(row["kind"] == "negative" for row in records),
            }
        receipt = {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "output": str(output),
            "inputs": inputs,
            "split_receipt_sha256": sha256_file(temporary / "receipts" / "detfly_split_receipt.json"),
            "sealed_test_manifest_sha256": test_manifest_sha,
            "joint_manifest_rows": joint_count,
            "joint_manifest_sha256": joint_manifest_sha,
            "train_manifest_sha256": sha256_file(temporary / "train.txt"),
            "val_manifest_sha256": sha256_file(temporary / "val.txt"),
            "training_yaml_sha256": sha256_file(temporary / "joint_detector.yaml"),
            "training_yaml_contains_test": False,
            "detfly_tiling": {
                "tile": TILE,
                "overlap": OVERLAP,
                "visible_fraction": VISIBLE_FRACTION,
                "min_visible_side_px": MIN_VISIBLE_SIDE_PX,
                "negatives_per_frame": NEGATIVES_PER_FRAME,
                "negative_seed": NEGATIVE_SEED,
                "jpeg_quality": JPEG_QUALITY,
                "counts": detfly_counts,
            },
            "counts": dataset_counts,
            "software": {
                "python": platform.python_version(),
                "pillow": pillow_version,
            },
        }
        write_json(temporary / "receipt.json", receipt)
        os.rename(str(temporary), str(output))
        print("[joint] PASS: %d indexed train/val samples -> %s" % (joint_count, output), flush=True)
        return receipt
    except BaseException:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
