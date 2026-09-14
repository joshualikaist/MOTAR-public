"""Fail-closed verifier for the generated NPS+Det-Fly v1 detector dataset."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = WORKSPACE / "datasets" / "nps_detfly_joint_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    digest = hashlib.sha256()
    rows = []
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                rows.append(json.loads(line))
            except Exception as error:
                raise RuntimeError("invalid JSONL line %d in %s: %s" % (
                    line_number, path, error))
    return rows, digest.hexdigest()


def validate_label(path):
    boxes = 0
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 5, "label must have five columns: %s:%d" % (path, line_number))
        require(fields[0] == "0", "only class 0 is allowed: %s:%d" % (path, line_number))
        values = [float(value) for value in fields[1:]]
        require(all(0.0 <= value <= 1.0 for value in values),
                "normalized label outside [0,1]: %s:%d" % (path, line_number))
        require(values[2] > 0.0 and values[3] > 0.0,
                "non-positive normalized box: %s:%d" % (path, line_number))
        boxes += 1
    return boxes


def verify(dataset):
    from PIL import Image

    dataset = Path(dataset).resolve()
    receipt = load_json(dataset / "receipt.json")
    split_receipt_path = dataset / "receipts" / "detfly_split_receipt.json"
    split_receipt = load_json(split_receipt_path)
    require(receipt["training_yaml_contains_test"] is False, "receipt exposes test to training")
    require(split_receipt["training_yaml_contains_test"] is False, "split receipt exposes test")
    require(sha256_file(split_receipt_path) == receipt["split_receipt_sha256"],
            "split receipt SHA-256 mismatch")

    artifacts = {
        "joint_detector.yaml": "training_yaml_sha256",
        "train.txt": "train_manifest_sha256",
        "val.txt": "val_manifest_sha256",
    }
    for relative, receipt_key in artifacts.items():
        require(sha256_file(dataset / relative) == receipt[receipt_key],
                "%s SHA-256 mismatch" % relative)
    yaml_text = (dataset / "joint_detector.yaml").read_text()
    require("test:" not in yaml_text, "training YAML contains a test key")

    joint_path = dataset / "manifests" / "joint_samples.jsonl"
    records, joint_sha = load_jsonl(joint_path)
    require(joint_sha == receipt["joint_manifest_sha256"], "joint manifest SHA-256 mismatch")
    require(len(records) == receipt["joint_manifest_rows"], "joint manifest row-count mismatch")

    test_path = dataset / "manifests" / "detfly_test_sealed.jsonl"
    test_rows, test_sha = load_jsonl(test_path)
    require(test_sha == receipt["sealed_test_manifest_sha256"], "sealed test SHA-256 mismatch")
    require(test_sha == split_receipt["sealed_test_manifest_sha256"],
            "sealed test SHA-256 disagrees between receipts")
    require(len(test_rows) == split_receipt["sealed_test_manifest_rows"],
            "sealed test row-count mismatch")
    require(all(row["split"] == "test" and row["source_group"] == "020" for row in test_rows),
            "sealed test contains a non-test or non-020 row")

    expected_paths = {
        split: (dataset / (split + ".txt")).read_text().splitlines()
        for split in ("train", "val")
    }
    actual_paths = collections.defaultdict(list)
    unit_splits = collections.defaultdict(set)
    digest_splits = collections.defaultdict(set)
    source_images = set()
    counts = collections.defaultdict(lambda: collections.Counter())
    for index, record in enumerate(records, 1):
        split = record["split"]
        require(split in ("train", "val"), "joint manifest contains forbidden split %r" % split)
        require(record["dataset"] in ("nps", "detfly"), "unknown dataset in joint manifest")
        image_path = Path(record["image"])
        label_path = Path(record["label"])
        require(image_path.is_file(), "missing image %s" % image_path)
        require(label_path.is_file(), "missing label %s" % label_path)
        require(sha256_file(image_path) == record["image_sha256"],
                "image SHA-256 mismatch: %s" % image_path)
        require(sha256_file(label_path) == record["label_sha256"],
                "label SHA-256 mismatch: %s" % label_path)
        with Image.open(str(image_path)) as image:
            require(image.size == (640, 640), "non-640 image: %s -> %r" % (image_path, image.size))
        label_boxes = validate_label(label_path)
        require(label_boxes == record["boxes"], "box count mismatch: %s" % label_path)
        require((record["kind"] == "positive") == (label_boxes > 0),
                "positive/negative kind disagrees with labels: %s" % label_path)
        actual_paths[split].append(str(image_path))
        unit = record.get("source_unit") if record["dataset"] == "nps" else record.get("source_image")
        require(unit, "missing source unit for %s" % image_path)
        unit_splits[(record["dataset"], unit)].add(split)
        digest_splits[record["image_sha256"]].add(split)
        if record["dataset"] == "detfly":
            require(str(record["source_group"]) == "010", "Det-Fly 020 leaked into train/val")
            source_images.add(record["source_image"])
        counts[split]["samples"] += 1
        counts[split]["boxes"] += label_boxes
        counts[split][record["dataset"]] += 1
        counts[split][record["kind"]] += 1
        if index % 5000 == 0:
            print("[joint-verify] %d/%d samples" % (index, len(records)), flush=True)

    require(all(len(splits) == 1 for splits in unit_splits.values()),
            "a source frame/clip crosses train and validation")
    require(all(len(splits) == 1 for splits in digest_splits.values()),
            "an exact image duplicate crosses train and validation")
    test_source_images = set(row["image"] for row in test_rows)
    require(not (source_images & test_source_images), "sealed test source image leaked into joint data")
    for split in ("train", "val"):
        require(actual_paths[split] == expected_paths[split], "%s.txt order/content mismatch" % split)
        expected = receipt["counts"][split]
        require(counts[split]["samples"] == expected["samples"], "%s sample count drift" % split)
        require(counts[split]["boxes"] == expected["boxes"], "%s box count drift" % split)
        require(counts[split]["nps"] == expected["by_dataset"]["nps"], "%s NPS count drift" % split)
        require(counts[split]["detfly"] == expected["by_dataset"]["detfly"],
                "%s Det-Fly count drift" % split)

    report = {
        "status": "PASS",
        "samples": len(records),
        "sealed_test_images": len(test_rows),
        "source_units_crossing_train_val": 0,
        "exact_images_crossing_train_val": 0,
        "sealed_test_sources_in_training": 0,
        "training_yaml_contains_test": False,
    }
    print("[joint-verify] PASS " + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    verify(parse_args().dataset)


if __name__ == "__main__":
    main()
