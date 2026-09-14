"""Run the frozen detector over an NPS sequence manifest and emit P4 Top-K candidate records."""

import argparse
import gzip
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from perception_candidates import (
    SCHEMA_VERSION, appearance_descriptor, canonical_line, read_jsonl, sha256_file,
    validate_candidate_record,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tool modules
from runtime_fingerprint import runtime_fingerprint  # noqa: E402


WORKSPACE = Path(__file__).resolve().parents[3]
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_YOLOV5 = WORKSPACE / "datasets" / "yolov5"
ENCODER_CONFIG = REPOSITORY / "configs" / "perception_appearance_encoder_v1.json"
SCHEMA = REPOSITORY / "docs" / "specs" / "motar_perception_candidates_v1.schema.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolov5", type=Path, default=DEFAULT_YOLOV5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=5, choices=(5,))
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test prefix only.")
    return parser.parse_args()


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


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def main():
    args = parse_args()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise SystemExit("[candidates] refusing existing output/receipt")
    if not 0.0 <= args.confidence_floor <= 1.0:
        raise SystemExit("[candidates] invalid confidence floor")
    manifest = args.manifest.resolve()
    manifest_receipt_path = args.manifest_receipt.resolve()
    manifest_receipt = json.loads(manifest_receipt_path.read_text())
    if sha256_file(manifest) != manifest_receipt["manifest_sha256"]:
        raise SystemExit("[candidates] manifest receipt hash mismatch")
    weights = args.weights.resolve()
    weight_sha = sha256_file(weights)
    if weight_sha != args.expected_weights_sha256:
        raise SystemExit("[candidates] detector weights hash mismatch: %s" % weight_sha)
    rows = list(read_jsonl(manifest))
    if args.max_frames is not None:
        rows = rows[:args.max_frames]
    if not rows:
        raise SystemExit("[candidates] empty manifest selection")
    dataset = Path(manifest_receipt["dataset"])
    encoder_config = json.loads(ENCODER_CONFIG.read_text())
    encoder_sha = sha256_file(ENCODER_CONFIG)

    yolov5 = args.yolov5.resolve()
    sys.path.insert(0, str(yolov5))
    import cv2
    import torch
    from models.common import DetectMultiBackend
    from torchvision.ops import nms
    from utils.general import non_max_suppression

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[candidates] CUDA requested but unavailable")
    use_half = bool(args.half and device.type == "cuda")
    model = DetectMultiBackend(str(weights), device=device, fp16=use_half)
    stride = int(model.stride)
    if args.tile % stride:
        raise SystemExit("[candidates] tile must be divisible by detector stride")
    model.warmup(imgsz=(args.batch_size, 3, args.tile, args.tile))

    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    latencies_ms = []
    candidate_count = 0
    tile_count = 0
    started = time.monotonic()
    with gzip.open(str(output), "wt", encoding="utf-8", compresslevel=6) as stream:
        for position, row in enumerate(rows, 1):
            frame = cv2.imread(str(dataset / row["image"]), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("could not decode %s" % row["image"])
            height, width = frame.shape[:2]
            if (width, height) != (row["width_px"], row["height_px"]):
                raise RuntimeError("manifest/image geometry mismatch")
            began = time.monotonic_ns()
            origins = [(x, y) for y in tile_origins(height, args.tile, args.overlap)
                       for x in tile_origins(width, args.tile, args.overlap)]
            translated = []
            for start in range(0, len(origins), args.batch_size):
                batch_origins = origins[start:start + args.batch_size]
                tiles = [frame[y:y + args.tile, x:x + args.tile] for x, y in batch_origins]
                array = np.ascontiguousarray(np.stack(tiles)[:, :, :, ::-1].transpose(0, 3, 1, 2))
                tensor = torch.from_numpy(array).to(device)
                tensor = tensor.half() if use_half else tensor.float()
                tensor /= 255.0
                with torch.no_grad():
                    predictions = model(tensor)
                detections = non_max_suppression(
                    predictions, args.confidence_floor, args.nms_iou, max_det=100)
                tile_count += len(batch_origins)
                for detection, (origin_x, origin_y) in zip(detections, batch_origins):
                    if len(detection):
                        detection = detection[:, :5].detach().float().cpu()
                        detection[:, [0, 2]] += origin_x
                        detection[:, [1, 3]] += origin_y
                        translated.append(detection)
            selected = []
            if translated:
                merged = torch.cat(translated, dim=0)
                merged[:, [0, 2]] = merged[:, [0, 2]].clamp(0.0, float(width))
                merged[:, [1, 3]] = merged[:, [1, 3]].clamp(0.0, float(height))
                kept = nms(merged[:, :4], merged[:, 4], args.nms_iou)[:args.top_k]
                selected = merged[kept].numpy().tolist()
                # Python's sort is stable, so exact confidence ties retain torchvision NMS order.
                selected.sort(key=lambda item: -item[4])
            candidates = []
            for rank, detection in enumerate(selected):
                x1, y1, x2, y2, confidence = (float(value) for value in detection)
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append({
                    "rank": len(candidates),
                    "u_px": (x1 + x2) / 2.0,
                    "v_px": (y1 + y2) / 2.0,
                    "width_px": x2 - x1,
                    "height_px": y2 - y1,
                    "confidence": confidence,
                    "appearance_64d": appearance_descriptor(
                        frame, (x1, y1, x2, y2), encoder_config["context_scale"]),
                })
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ns = time.monotonic_ns() - began
            latencies_ms.append(latency_ns / 1e6)
            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_id": row["frame_id"],
                "source_sequence_id": row["source_sequence_id"],
                "frame_index": row["frame_index"],
                "capture_timestamp_ns": row["capture_timestamp_ns"],
                "inference_completed_timestamp_ns": row["capture_timestamp_ns"] + latency_ns,
                "image": {
                    "width_px": width,
                    "height_px": height,
                    "coordinate_frame": "pixel_top_left_u_right_v_down",
                },
                "detector": {
                    "name": "motar.nps-detfly-yolov5s.v1",
                    "weights_sha256": weight_sha,
                    "confidence_floor": args.confidence_floor,
                    "nms_iou": args.nms_iou,
                },
                "appearance_encoder": {
                    "name": encoder_config["name"],
                    "weights_sha256": encoder_sha,
                    "dimension": 64,
                    "normalization": "l2_unit",
                },
                "candidates": candidates,
            }
            validate_candidate_record(record, previous)
            stream.write(canonical_line(record))
            candidate_count += len(candidates)
            if position % 100 == 0 or position == len(rows):
                rate = position / max(time.monotonic() - started, 1e-9)
                print("[candidates] %d/%d frames · %.2f frame/s" % (
                    position, len(rows), rate), flush=True)

    receipt = {
        "schema_version": 1,
        # What produced these numbers. Omitting it in 2026-09 made a cache that a
        # different torch/cuDNN could not reproduce look like non-determinism.
        "runtime": runtime_fingerprint(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_schema": SCHEMA_VERSION,
        "candidate_schema_sha256": sha256_file(SCHEMA),
        "records": len(rows),
        "candidates": candidate_count,
        "tiles": tile_count,
        "complete_manifest": args.max_frames is None,
        "manifest_sha256": sha256_file(manifest),
        "manifest_receipt_sha256": sha256_file(manifest_receipt_path),
        "weights_sha256": weight_sha,
        "appearance_encoder_config_sha256": encoder_sha,
        "appearance_encoder_note": "parameter-free; schema weights_sha256 carries config identity",
        "output_sha256": sha256_file(output),
        "yolov5_revision": git_revision(yolov5),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fp16": use_half,
        "config": {
            "tile": args.tile, "overlap": args.overlap, "batch_size": args.batch_size,
            "confidence_floor": args.confidence_floor, "nms_iou": args.nms_iou,
            "top_k": args.top_k, "max_frames": args.max_frames,
        },
        "latency_ms": {
            "scope": "frame-array available through detector, NMS and appearance encoding",
            "mean": statistics.fmean(latencies_ms),
            "p50": quantile(latencies_ms, 0.50),
            "p95": quantile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[candidates] PASS -> %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
