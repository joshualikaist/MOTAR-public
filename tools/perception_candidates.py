"""Shared, dependency-light helpers for MOTAR perception candidate records."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "motar.perception-candidates.v1"
TOP_K = 5
APPEARANCE_DIMENSION = 64


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_line(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def open_jsonl(path, mode="rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(str(path), mode, encoding=None if "b" in mode else "utf-8")
    return path.open(mode, encoding=None if "b" in mode else "utf-8")


def read_jsonl(path):
    with open_jsonl(path, "rt") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as error:
                    raise ValueError("invalid JSONL line %d: %s" % (line_number, error))


def appearance_descriptor(frame_bgr, xyxy, context_scale=1.25):
    """Return the parameter-free v1 64-D descriptor for one clipped detector box."""
    import cv2

    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max((x2 - x1) * context_scale / 2.0, 0.5)
    half_h = max((y2 - y1) * context_scale / 2.0, 0.5)
    left = max(0, int(math.floor(center_x - half_w)))
    top = max(0, int(math.floor(center_y - half_h)))
    right = min(width, int(math.ceil(center_x + half_w)))
    bottom = min(height, int(math.ceil(center_y + half_h)))
    if right <= left or bottom <= top:
        raise ValueError("candidate crop is empty after clipping")
    crop = frame_bgr[top:bottom, left:right]
    rgb = cv2.cvtColor(cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float64) / 255.0
    grid = rgb.reshape(4, 2, 4, 2, 3).mean(axis=(1, 3)).reshape(-1)
    gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114)
    histogram, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    vector = np.concatenate([grid, histogram.astype(np.float64) / gray.size])
    if vector.shape != (APPEARANCE_DIMENSION,):
        raise AssertionError("appearance descriptor dimension changed")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("appearance descriptor has invalid norm")
    return (vector / norm).tolist()


def validate_candidate_record(record, previous_by_sequence=None, norm_tolerance=1e-5):
    """Validate semantic constraints not expressible in the JSON Schema."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate schema version mismatch")
    required = {
        "frame_id", "source_sequence_id", "frame_index", "capture_timestamp_ns",
        "inference_completed_timestamp_ns", "image", "detector", "appearance_encoder",
        "candidates",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError("missing candidate fields: %s" % missing)
    if record["capture_timestamp_ns"] > record["inference_completed_timestamp_ns"]:
        raise ValueError("inference completes before capture")
    candidates = record["candidates"]
    if len(candidates) > TOP_K:
        raise ValueError("more than Top-K candidates")
    width = int(record["image"]["width_px"])
    height = int(record["image"]["height_px"])
    previous_confidence = math.inf
    for expected_rank, candidate in enumerate(candidates):
        if candidate["rank"] != expected_rank:
            raise ValueError("candidate ranks are not contiguous")
        confidence = float(candidate["confidence"])
        if not 0.0 <= confidence <= previous_confidence:
            raise ValueError("candidate confidence is invalid or not non-increasing")
        previous_confidence = confidence
        u, v = float(candidate["u_px"]), float(candidate["v_px"])
        box_w, box_h = float(candidate["width_px"]), float(candidate["height_px"])
        if box_w <= 0.0 or box_h <= 0.0:
            raise ValueError("candidate has non-positive extent")
        if u - box_w / 2.0 < -1e-5 or u + box_w / 2.0 > width + 1e-5:
            raise ValueError("candidate exceeds horizontal image bounds")
        if v - box_h / 2.0 < -1e-5 or v + box_h / 2.0 > height + 1e-5:
            raise ValueError("candidate exceeds vertical image bounds")
        appearance = np.asarray(candidate["appearance_64d"], dtype=np.float64)
        if appearance.shape != (APPEARANCE_DIMENSION,) or not np.isfinite(appearance).all():
            raise ValueError("appearance must be 64 finite values")
        if abs(float(np.linalg.norm(appearance)) - 1.0) > norm_tolerance:
            raise ValueError("appearance is not L2 unit normalized")
    if previous_by_sequence is not None:
        sequence = record["source_sequence_id"]
        previous = previous_by_sequence.get(sequence)
        current = (int(record["frame_index"]), int(record["capture_timestamp_ns"]))
        if previous is not None and not (current[0] > previous[0] and current[1] > previous[1]):
            raise ValueError("frame index/timestamp is not strictly increasing within sequence")
        previous_by_sequence[sequence] = current
    return True
