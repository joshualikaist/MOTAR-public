"""Evaluate the frozen NPS-Drones YOLOv5 detector on Det-Fly without retuning.

The NPS model was trained on native-scale 640 px tiles.  Det-Fly frames are therefore evaluated
with the same 640 px native-scale sliding window (128 px overlap), not by shrinking 3840x2160 to
640.  Per-tile detections are merged with image-level NMS before scoring.  The command writes raw
per-image predictions as compressed JSONL and reports whole-dataset, official source-group and
ground-truth pixel-size slices.

Four semantic background names are not present in the distributed Det-Fly metadata.  This tool
does not infer them from pixels; it reports the official numbered source groups 010/020 instead.
"""

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tool modules
from runtime_fingerprint import runtime_fingerprint  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = WORKSPACE / "datasets" / "detfly_index"
DEFAULT_WEIGHTS = (WORKSPACE / "detector_runs" / "runs" / "nps_det" /
                   "s_tiles640_b8_e40" / "weights" / "best.pt")
DEFAULT_YOLOV5 = WORKSPACE / "datasets" / "yolov5"
DEFAULT_OUTPUT = WORKSPACE / "detector_runs" / "results" / "nps_to_detfly_zeroshot"
EXPECTED_WEIGHTS_SHA256 = "ccd65dc37ec2fce765e0860232922287323c49179a4ea0e12cb6ad1b4f28f580"
SIZE_EDGES_PX = (0.0, 8.0, 12.0, 20.0, 32.0, 64.0, 128.0, math.inf)
IOU_THRESHOLDS = (0.3, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolov5", type=Path, default=DEFAULT_YOLOV5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-manifest", type=Path, default=None,
                        help="Optional audited subset JSONL, e.g. the sealed joint-detector test.")
    parser.add_argument("--expected-selection-manifest-sha256", default=None,
                        help="Required with --selection-manifest; prevents selection substitution.")
    parser.add_argument("--expected-weights-sha256", default=EXPECTED_WEIGHTS_SHA256)
    parser.add_argument("--evaluation-kind", choices=("zero_shot", "joint_heldout"),
                        default="zero_shot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence-floor", type=float, default=0.001)
    parser.add_argument("--operating-confidence", type=float, default=0.25)
    parser.add_argument("--tile-nms-iou", type=float, default=0.45)
    parser.add_argument("--image-nms-iou", type=float, default=0.45)
    parser.add_argument("--max-tile-detections", type=int, default=100)
    parser.add_argument("--max-image-detections", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=None,
                        help="smoke-test prefix only; omitted for the preregistered full run")
    parser.add_argument("--half", action="store_true", help="FP16 CUDA inference")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="divide the frame by this factor before tiling. 1.0 is the native-scale "
                             "arm. 4.0 is the scale-matched arm: Det-Fly targets have a median "
                             "diagonal of 109 px against 25 px in the NPS training tiles, so "
                             "native-scale zero-shot confounds domain shift with a 4.3x scale "
                             "shift. Boxes are reported in ORIGINAL pixel coordinates either way.")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tile_origins(extent, tile, overlap):
    if extent <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile")
    origins = list(range(0, extent - tile + 1, step))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def greedy_matches(predictions, ground_truth, iou_threshold):
    """Return confidence-ordered (score, matched_gt_or_None) pairs for one image."""
    if not predictions:
        return []
    gt = np.asarray(ground_truth, dtype=np.float32).reshape((-1, 4))
    used = set()
    matches = []
    for pred in sorted(predictions, key=lambda row: -row[4]):
        matched = None
        if len(gt):
            overlaps = box_iou_one_to_many(np.asarray(pred[:4], dtype=np.float32), gt)
            for candidate in np.argsort(-overlaps):
                candidate = int(candidate)
                if float(overlaps[candidate]) < iou_threshold:
                    break
                if candidate not in used:
                    matched = candidate
                    used.add(candidate)
                    break
        matches.append((float(pred[4]), matched))
    return matches


def size_bin(box):
    side = math.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 0.0))
    for index, (lower, upper) in enumerate(zip(SIZE_EDGES_PX, SIZE_EDGES_PX[1:])):
        if lower <= side < upper:
            return index
    raise AssertionError("unreachable")


def size_label(index):
    lower, upper = SIZE_EDGES_PX[index], SIZE_EDGES_PX[index + 1]
    return "%d-%spx" % (lower, "inf" if math.isinf(upper) else "%d" % upper)


class MetricSlice:
    """Confidence-ranked detections for one image/group/GT-conditioned slice."""

    def __init__(self):
        self.gt = 0
        self.scores = []
        self.labels = []

    def add(self, matches, selected_gt):
        selected_gt = set(selected_gt)
        self.gt += len(selected_gt)
        for score, matched in matches:
            if matched is None:
                self.scores.append(score)
                self.labels.append(0)
            elif matched in selected_gt:
                self.scores.append(score)
                self.labels.append(1)
            # A detection assigned to an out-of-slice GT is ignored for this slice.

    def report(self, operating_confidence):
        return ranked_report(np.asarray(self.scores), np.asarray(self.labels), self.gt,
                             operating_confidence)


def ranked_report(scores, labels, gt_count, operating_confidence, already_sorted=False):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    if not already_sorted:
        order = np.argsort(-scores)
        labels = labels[order]
        scores = scores[order]
    true_positive = np.cumsum(labels)
    false_positive = np.cumsum(1 - labels)
    recall = true_positive / gt_count if gt_count else np.zeros_like(true_positive, dtype=float)
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    ap = 0.0
    if gt_count and len(labels):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        changed = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    selected = scores >= operating_confidence
    tp = int(labels[selected].sum()) if len(labels) else 0
    fp = int(selected.sum() - tp)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt_count if gt_count else None
    return {
        "gt_boxes": gt_count,
        "detections_at_operating_confidence": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": gt_count - tp,
        "precision": p,
        "recall": r,
        "f1": (2.0 * p * r / (p + r)) if r is not None and p + r else 0.0,
        "average_precision": ap,
    }


def load_jsonl(path, max_images=None):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if max_images is not None and len(rows) >= max_images:
                    break
    return rows


def load_selection(index_dir, selection_manifest, expected_sha256, max_images):
    index_path = index_dir / "index.jsonl"
    if selection_manifest is None:
        return load_jsonl(index_path, max_images), None
    selection_manifest = selection_manifest.resolve()
    if not expected_sha256:
        raise ValueError("--expected-selection-manifest-sha256 is required with a selection")
    observed_sha = sha256_file(selection_manifest)
    if observed_sha != expected_sha256:
        raise ValueError("selection manifest hash mismatch: %s" % observed_sha)
    full_index = {row["image"]: row for row in load_jsonl(index_path)}
    selected = load_jsonl(selection_manifest, max_images)
    for row in selected:
        indexed = full_index.get(row.get("image"))
        if indexed is None:
            raise ValueError("selection image is absent from the prepared index")
        for key, value in indexed.items():
            if row.get(key) != value:
                raise ValueError("selection/index mismatch for %s field %s" % (row["image"], key))
    if len({row["image"] for row in selected}) != len(selected):
        raise ValueError("selection manifest contains duplicate images")
    return selected, observed_sha


def main():
    args = parse_args()
    index_dir = args.index.resolve()
    weights = args.weights.resolve()
    yolov5 = args.yolov5.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("[zeroshot] refusing to overwrite existing output: %s" % output)
    if not 0.0 <= args.confidence_floor <= args.operating_confidence <= 1.0:
        raise SystemExit("[zeroshot] require 0 <= confidence-floor <= operating-confidence <= 1")
    if not (index_dir / "index.jsonl").is_file() or not (index_dir / "receipt.json").is_file():
        raise SystemExit("[zeroshot] Det-Fly index is incomplete")
    weight_sha = sha256_file(weights)
    if weight_sha != args.expected_weights_sha256:
        raise SystemExit("[zeroshot] frozen checkpoint hash mismatch: %s" % weight_sha)

    index_receipt = json.loads((index_dir / "receipt.json").read_text())
    source = Path(index_receipt["source_dataset"])
    try:
        rows, selection_sha = load_selection(
            index_dir, args.selection_manifest, args.expected_selection_manifest_sha256,
            args.max_images)
    except ValueError as error:
        raise SystemExit("[zeroshot] %s" % error)
    if not rows:
        raise SystemExit("[zeroshot] index contains no rows")

    sys.path.insert(0, str(yolov5))
    import cv2
    import torch
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    from torchvision.ops import nms

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[zeroshot] CUDA requested but unavailable")
    device = torch.device(args.device)
    use_half = bool(args.half and device.type == "cuda")
    model = DetectMultiBackend(str(weights), device=device, fp16=use_half)
    stride = int(model.stride)
    if args.tile % stride:
        raise SystemExit("[zeroshot] tile size must be divisible by model stride %s" % stride)
    model.warmup(imgsz=(args.batch_size, 3, args.tile, args.tile))

    output.mkdir(parents=True)
    raw_path = output / "predictions.jsonl.gz"
    groups = sorted({row["source_group"] for row in rows})
    group_to_id = {group: index for index, group in enumerate(groups)}
    capacity = len(rows) * args.max_image_detections
    score_store = np.empty(capacity, dtype=np.float32)
    group_store = np.empty(capacity, dtype=np.int8)
    match_size_store = {threshold: np.full(capacity, -1, dtype=np.int8)
                        for threshold in IOU_THRESHOLDS}
    match_difficulty_store = {threshold: np.full(capacity, -1, dtype=np.int8)
                              for threshold in IOU_THRESHOLDS}
    detection_count = 0
    gt_counts = {
        "overall": 0,
        "groups": collections.Counter(),
        "sizes": collections.Counter(),
        "difficulty": collections.Counter(),
    }

    started = time.monotonic()
    inference_seconds = 0.0
    tile_count = 0
    raw_prediction_count = 0
    with gzip.open(str(raw_path), "wt", encoding="utf-8", compresslevel=6) as raw_stream:
        for image_index, row in enumerate(rows, 1):
            image_path = source / row["image"]
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("could not decode %s" % image_path)
            height, width = frame.shape[:2]
            if (width, height) != (row["width"], row["height"]):
                raise RuntimeError("index/image size mismatch for %s" % row["image"])
            if args.downscale != 1.0:
                # Scale-matched arm. INTER_AREA is the correct filter for shrinking; every box the
                # model emits is multiplied back by the same factor below, so predictions, ground
                # truth and the size bins all stay in original 4K pixels.
                #
                # The factor must be IDENTICAL on both axes. Clamping each axis to the tile size
                # independently silently distorted the aspect ratio (2160/4 = 540 -> clamped to
                # 640, i.e. y scaled by 3.375 while x scaled by 4.0) and put every prediction in
                # the wrong place. Short axes are PADDED after the uniform resize instead; the pad
                # goes bottom-right so tile origins and the inverse mapping are unaffected.
                scaled_w = max(1, int(round(width / args.downscale)))
                scaled_h = max(1, int(round(height / args.downscale)))
                frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                pad_x, pad_y = max(0, args.tile - scaled_w), max(0, args.tile - scaled_h)
                if pad_x or pad_y:
                    frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x,
                                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
                height, width = frame.shape[:2]
            origins = [(x, y) for y in tile_origins(height, args.tile, args.overlap)
                       for x in tile_origins(width, args.tile, args.overlap)]
            translated = []
            for batch_start in range(0, len(origins), args.batch_size):
                batch_origins = origins[batch_start:batch_start + args.batch_size]
                tiles = [frame[y:y + args.tile, x:x + args.tile] for x, y in batch_origins]
                array = np.ascontiguousarray(np.stack(tiles)[:, :, :, ::-1].transpose(0, 3, 1, 2))
                tensor = torch.from_numpy(array).to(device)
                tensor = tensor.half() if use_half else tensor.float()
                tensor /= 255.0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.monotonic()
                with torch.no_grad():
                    batch_prediction = model(tensor)
                detections = non_max_suppression(
                    batch_prediction, args.confidence_floor, args.tile_nms_iou,
                    max_det=args.max_tile_detections)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.monotonic() - before
                tile_count += len(batch_origins)
                for detection, (origin_x, origin_y) in zip(detections, batch_origins):
                    if len(detection):
                        detection = detection[:, :5].detach().float().cpu()
                        detection[:, [0, 2]] += origin_x
                        detection[:, [1, 3]] += origin_y
                        translated.append(detection)

            predictions = []
            if translated:
                merged = torch.cat(translated, dim=0)
                if args.downscale != 1.0:
                    # Back to original 4K pixels BEFORE image-level NMS, so the IoU threshold and
                    # the size bins mean the same thing in both arms.
                    merged[:, :4] *= float(args.downscale)
                kept = nms(merged[:, :4], merged[:, 4], args.image_nms_iou)
                kept = kept[:args.max_image_detections]
                predictions = merged[kept].numpy().tolist()
                predictions.sort(key=lambda item: -item[4])
            raw_prediction_count += len(predictions)
            gt_boxes = [obj["xyxy"] for obj in row["objects"]]
            gt_size_labels = [size_label(size_bin(box)) for box in gt_boxes]
            gt_counts["overall"] += len(gt_boxes)
            gt_counts["groups"][row["source_group"]] += len(gt_boxes)
            gt_counts["sizes"].update(gt_size_labels)
            gt_counts["difficulty"].update(
                "difficult" if obj["difficult"] else "non_difficult" for obj in row["objects"])
            store_start = detection_count
            store_stop = store_start + len(predictions)
            if store_stop > capacity:
                raise AssertionError("prediction store capacity exceeded")
            if predictions:
                score_store[store_start:store_stop] = [pred[4] for pred in predictions]
                group_store[store_start:store_stop] = group_to_id[row["source_group"]]
            for threshold in IOU_THRESHOLDS:
                matches = greedy_matches(predictions, gt_boxes, threshold)
                size_values = match_size_store[threshold][store_start:store_stop]
                difficulty_values = match_difficulty_store[threshold][store_start:store_stop]
                for prediction_index, (_score, matched) in enumerate(matches):
                    if matched is not None:
                        size_values[prediction_index] = size_bin(gt_boxes[matched])
                        difficulty_values[prediction_index] = int(row["objects"][matched]["difficult"])
            detection_count = store_stop

            raw_stream.write(json.dumps({
                "image": row["image"],
                "source_group": row["source_group"],
                "background": row["background"],
                # Original frame geometry: ground truth, predictions and size bins are all in
                # original pixels, so reporting the working size here would be misleading.
                "width": row["width"],
                "height": row["height"],
                "ground_truth": row["objects"],
                "predictions_xyxy_confidence": predictions,
            }, separators=(",", ":")) + "\n")
            if image_index % 25 == 0 or image_index == len(rows):
                elapsed = time.monotonic() - started
                rate = image_index / max(elapsed, 1e-9)
                eta = (len(rows) - image_index) / max(rate, 1e-9)
                print("[zeroshot] %s/%s images · %.2f image/s · ETA %.1f min" % (
                    image_index, len(rows), rate, eta / 60.0), flush=True)

    elapsed_seconds = time.monotonic() - started
    score_store = score_store[:detection_count]
    group_store = group_store[:detection_count]
    order = np.argsort(-score_store)
    ranked_scores = score_store[order]
    ranked_groups = group_store[order]
    metrics = {}
    for threshold in IOU_THRESHOLDS:
        ranked_sizes = match_size_store[threshold][:detection_count][order]
        ranked_difficulty = match_difficulty_store[threshold][:detection_count][order]
        overall_labels = (ranked_sizes >= 0).astype(np.int8)
        by_group = {}
        for group in groups:
            selected = ranked_groups == group_to_id[group]
            by_group[group] = ranked_report(
                ranked_scores[selected], overall_labels[selected], gt_counts["groups"][group],
                args.operating_confidence, already_sorted=True)
        by_size = {}
        for bin_index in range(len(SIZE_EDGES_PX) - 1):
            label = size_label(bin_index)
            selected = (ranked_sizes < 0) | (ranked_sizes == bin_index)
            by_size[label] = ranked_report(
                ranked_scores[selected], (ranked_sizes[selected] == bin_index).astype(np.int8),
                gt_counts["sizes"][label], args.operating_confidence, already_sorted=True)
        by_difficulty = {}
        for value, label in ((0, "non_difficult"), (1, "difficult")):
            selected = (ranked_difficulty < 0) | (ranked_difficulty == value)
            by_difficulty[label] = ranked_report(
                ranked_scores[selected], (ranked_difficulty[selected] == value).astype(np.int8),
                gt_counts["difficulty"][label], args.operating_confidence, already_sorted=True)
        metrics[str(threshold)] = {
            "overall": ranked_report(ranked_scores, overall_labels, gt_counts["overall"],
                                     args.operating_confidence, already_sorted=True),
            "by_source_group": by_group,
            "by_gt_equivalent_side_px": by_size,
            "by_difficulty": by_difficulty,
        }

    report = {
        "schema_version": 1,
        "experiment": ("nps_to_detfly_zero_shot" if args.evaluation_kind == "zero_shot"
                       else "nps_detfly_joint_to_detfly_heldout"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cross_dataset": args.evaluation_kind == "zero_shot",
        "complete_dataset": args.max_images is None and args.selection_manifest is None,
        "complete_selection": args.max_images is None,
        "images": len(rows),
        "tiles": tile_count,
        "raw_predictions": raw_prediction_count,
        "weights": str(weights),
        "weights_sha256": weight_sha,
        "trained_on": ("NPS-Drones 640px native-scale tiles"
                       if args.evaluation_kind == "zero_shot"
                       else "NPS-Drones + Det-Fly 010 train/validation tiles"),
        "evaluated_on": "Det-Fly",
        "arm": "native_scale" if args.downscale == 1.0 else f"scale_matched_{args.downscale:g}x",
        "downscale": args.downscale,
        "index_sha256": sha256_file(index_dir / "index.jsonl"),
        "index_receipt_sha256": sha256_file(index_dir / "receipt.json"),
        "selection_manifest": str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        "selection_manifest_sha256": selection_sha,
        "yolov5_revision": git_revision(yolov5),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fp16": use_half,
        "config": {
            "tile": args.tile,
            "overlap": args.overlap,
            "batch_size": args.batch_size,
            "confidence_floor": args.confidence_floor,
            "operating_confidence": args.operating_confidence,
            "tile_nms_iou": args.tile_nms_iou,
            "image_nms_iou": args.image_nms_iou,
            "max_tile_detections": args.max_tile_detections,
            "max_image_detections": args.max_image_detections,
            "iou_thresholds": list(IOU_THRESHOLDS),
            "size_edges_equivalent_side_px": [None if math.isinf(x) else x for x in SIZE_EDGES_PX],
            "max_images": args.max_images,
        },
        "background_slice": {
            "status": "UNAVAILABLE_IN_DISTRIBUTED_METADATA",
            "official_categories": ["sky", "urban", "field", "mountain"],
            "reported_substitute": "source_group (010/020)",
        },
        "size_slice_method": (
            "GT-conditioned: detections matched to out-of-bin GT are ignored; unmatched "
            "detections are false positives in every bin"
        ),
        "ap_scope": (
            "confidence >= confidence_floor after per-tile NMS and image-level top-K cap; "
            "AP is a lower bound if lower-score detections exist"
        ),
        "elapsed_seconds": elapsed_seconds,
        "model_inference_seconds": inference_seconds,
        "metrics": metrics,
        "raw_predictions_file": raw_path.name,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": 1,
        # What produced these numbers. Omitting it in 2026-09 made a cache that a
        # different torch/cuDNN could not reproduce look like non-determinism.
        "runtime": runtime_fingerprint(),
        "report_sha256": sha256_file(report_path),
        "raw_predictions_sha256": sha256_file(raw_path),
        "weights_sha256": weight_sha,
        "index_sha256": report["index_sha256"],
        "complete_dataset": report["complete_dataset"],
        "complete_selection": report["complete_selection"],
        "selection_manifest_sha256": selection_sha,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    primary = metrics["0.3"]["overall"]
    print("[zeroshot] IoU 0.3: P %.4f R %.4f AP %.4f" % (
        primary["precision"], primary["recall"] or 0.0, primary["average_precision"]))
    print("[zeroshot] PASS: report and raw predictions -> %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
