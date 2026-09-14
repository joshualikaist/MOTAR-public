"""Build audited P7c optical-flow/GMC residual features for train or validation only."""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

import cv2
import numpy as np

from perception_candidates import canonical_line, open_jsonl, sha256_file
from perception_temporal import (
    MOTION_FEATURE_DIMENSION, TOP_K, load_aligned_records,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tool modules
from runtime_fingerprint import runtime_fingerprint  # noqa: E402


SCHEMA = "motar.perception-motion.v1"
REPOSITORY = Path(__file__).resolve().parents[1]
FEATURE_FIELDS = (
    "box_gmc_residual_du_norm",
    "box_gmc_residual_dv_norm",
    "box_dlog_width",
    "box_dlog_height",
    "local_backward_flow_u_norm",
    "local_backward_flow_v_norm",
    "gmc_backward_flow_u_norm",
    "gmc_backward_flow_v_norm",
    "residual_backward_flow_u_norm",
    "residual_backward_flow_v_norm",
    "gmc_inlier_ratio",
    "gmc_valid",
)
DEFAULT_CONFIG = {
    "flow_max_width_px": 640,
    "farneback": {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 21,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    },
    "gmc_grid_step_px_at_flow_scale": 16,
    "gmc_candidate_mask_scale": 1.5,
    "gmc_ransac_threshold_px_at_flow_scale": 1.5,
    "gmc_min_correspondences": 12,
    "candidate_match_center_weight": 1.0,
    "candidate_match_log_size_weight": 0.05,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-test-contract", type=Path,
                        help="Explicit final-test execution contract after architecture freeze")
    return parser.parse_args()


def git_value(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY), *arguments], text=True).strip()


def gray_at_flow_scale(image_path, max_width):
    frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError("could not decode motion frame %s" % image_path)
    height, width = frame.shape
    scale = min(1.0, float(max_width) / width)
    if scale < 1.0:
        frame = cv2.resize(
            frame, (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA)
    return frame, scale


def candidate_mask(shape, candidates, scale, expansion):
    mask = np.zeros(shape, dtype=np.bool_)
    height, width = shape
    for candidate in candidates:
        half_w = float(candidate["width_px"]) * scale * expansion / 2.0
        half_h = float(candidate["height_px"]) * scale * expansion / 2.0
        center_x = float(candidate["u_px"]) * scale
        center_y = float(candidate["v_px"]) * scale
        left = max(0, int(math.floor(center_x - half_w)))
        top = max(0, int(math.floor(center_y - half_h)))
        right = min(width, int(math.ceil(center_x + half_w)))
        bottom = min(height, int(math.ceil(center_y + half_h)))
        mask[top:bottom, left:right] = True
    return mask


def estimate_backward_flow_and_gmc(previous_gray, current_gray, current_candidates,
                                   scale, config):
    flow = compute_flow(previous_gray, current_gray, config)
    return estimate_gmc(flow, current_candidates, scale, config)


def compute_flow(previous_gray, current_gray, config=DEFAULT_CONFIG):
    """Candidate-independent backward flow; safe to run alongside detection."""
    if previous_gray.shape != current_gray.shape:
        raise ValueError("motion frame geometry changed within sequence")
    flow_cfg = config["farneback"]
    flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None,
        flow_cfg["pyr_scale"], flow_cfg["levels"], flow_cfg["winsize"],
        flow_cfg["iterations"], flow_cfg["poly_n"], flow_cfg["poly_sigma"],
        flow_cfg["flags"])
    return flow


def estimate_gmc(flow, current_candidates, scale, config=DEFAULT_CONFIG):
    """Candidate-dependent GMC, run on the caller thread after detection."""
    height, width = flow.shape[:2]
    step = int(config["gmc_grid_step_px_at_flow_scale"])
    ys = np.arange(step // 2, height, step, dtype=np.int32)
    xs = np.arange(step // 2, width, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    source = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    excluded = candidate_mask(
        (height, width), current_candidates, scale,
        float(config["gmc_candidate_mask_scale"]))
    keep = ~excluded[source[:, 1].astype(int), source[:, 0].astype(int)]
    source = source[keep]
    sampled_flow = flow[source[:, 1].astype(int), source[:, 0].astype(int)]
    destination = source + sampled_flow
    finite = np.isfinite(destination).all(axis=1)
    source, destination = source[finite], destination[finite]
    affine, inliers = None, None
    if len(source) >= int(config["gmc_min_correspondences"]):
        cv2.setRNGSeed(17)
        affine, inliers = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC,
            ransacReprojThreshold=float(config["gmc_ransac_threshold_px_at_flow_scale"]),
            maxIters=2000, confidence=0.99, refineIters=10)
    valid = affine is not None and np.isfinite(affine).all()
    if not valid:
        affine = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_ratio = 0.0
    else:
        inlier_ratio = float(np.asarray(inliers).mean()) if inliers is not None else 0.0
    return flow, np.asarray(affine, dtype=np.float64), inlier_ratio, bool(valid)


def affine_point(affine, x, y):
    return np.asarray([
        affine[0, 0] * x + affine[0, 1] * y + affine[0, 2],
        affine[1, 0] * x + affine[1, 1] * y + affine[1, 2],
    ], dtype=np.float64)


def local_flow(flow, candidate, scale):
    height, width = flow.shape[:2]
    center_x = float(candidate["u_px"]) * scale
    center_y = float(candidate["v_px"]) * scale
    half_w = max(float(candidate["width_px"]) * scale / 2.0, 1.0)
    half_h = max(float(candidate["height_px"]) * scale / 2.0, 1.0)
    left = max(0, int(math.floor(center_x - half_w)))
    top = max(0, int(math.floor(center_y - half_h)))
    right = min(width, int(math.ceil(center_x + half_w)))
    bottom = min(height, int(math.ceil(center_y + half_h)))
    crop = flow[top:bottom, left:right].reshape(-1, 2)
    finite = crop[np.isfinite(crop).all(axis=1)]
    if not len(finite):
        return np.zeros(2, dtype=np.float64)
    return np.median(finite, axis=0).astype(np.float64)


def closest_previous_candidate(mapped_current, current, previous_candidates, width, height,
                               config):
    if not previous_candidates:
        return None
    best, best_cost = None, math.inf
    for previous in previous_candidates:
        center_cost = (
            ((mapped_current[0] / width) - (float(previous["u_px"]) / width)) ** 2
            + ((mapped_current[1] / height) - (float(previous["v_px"]) / height)) ** 2)
        size_cost = (
            math.log(max(float(current["width_px"]), 1e-6)
                     / max(float(previous["width_px"]), 1e-6)) ** 2
            + math.log(max(float(current["height_px"]), 1e-6)
                       / max(float(previous["height_px"]), 1e-6)) ** 2)
        cost = (float(config["candidate_match_center_weight"]) * center_cost
                + float(config["candidate_match_log_size_weight"]) * size_cost)
        if cost < best_cost:
            best, best_cost = previous, cost
    return best


def candidate_motion_features(flow, affine, inlier_ratio, gmc_valid, current_candidates,
                              previous_candidates, width, height, scale, config):
    result = np.zeros((TOP_K, MOTION_FEATURE_DIMENSION), dtype=np.float32)
    scaled_width, scaled_height = width * scale, height * scale
    for candidate in current_candidates:
        rank = int(candidate["rank"])
        center = np.asarray([
            float(candidate["u_px"]) * scale,
            float(candidate["v_px"]) * scale,
        ], dtype=np.float64)
        mapped = affine_point(affine, center[0], center[1])
        gmc_flow = mapped - center
        observed_flow = local_flow(flow, candidate, scale)
        residual_flow = observed_flow - gmc_flow
        previous = closest_previous_candidate(
            mapped / scale, candidate, previous_candidates, width, height, config)
        box_motion = np.zeros(4, dtype=np.float64)
        if previous is not None:
            box_motion = np.asarray([
                (mapped[0] - float(previous["u_px"]) * scale) / scaled_width,
                (mapped[1] - float(previous["v_px"]) * scale) / scaled_height,
                np.clip(math.log(float(candidate["width_px"])
                                 / max(float(previous["width_px"]), 1e-6)), -2.0, 2.0),
                np.clip(math.log(float(candidate["height_px"])
                                 / max(float(previous["height_px"]), 1e-6)), -2.0, 2.0),
            ])
        result[rank] = np.asarray([
            *box_motion,
            observed_flow[0] / scaled_width,
            observed_flow[1] / scaled_height,
            gmc_flow[0] / scaled_width,
            gmc_flow[1] / scaled_height,
            residual_flow[0] / scaled_width,
            residual_flow[1] / scaled_height,
            inlier_ratio,
            float(gmc_valid),
        ], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite motion feature")
    return result


def main():
    args = parse_args()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise SystemExit("[motion] refusing existing output/receipt")
    aligned, manifest_meta, candidate_meta = load_aligned_records(
        args.manifest, args.manifest_receipt, args.candidates, args.candidate_receipt)
    test_open = manifest_meta["split"] == "test" and args.frozen_test_contract is not None
    if args.frozen_test_contract is not None and not args.frozen_test_contract.is_file():
        raise SystemExit("[motion] final-test contract missing")
    if manifest_meta["split"] not in ("train", "val") and not test_open:
        raise SystemExit("[motion] only train/validation are allowed before architecture freeze")
    dataset_root = Path(manifest_meta["dataset"])
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_by_sequence = {}
    began = time.monotonic()
    gmc_valid_frames = 0
    with open_jsonl(output, "wt") as stream:
        for index, aligned_row in enumerate(aligned):
            source = aligned_row["source"]
            candidates = aligned_row["candidate_record"]["candidates"]
            image_path = dataset_root / source["image"]
            current_gray, scale = gray_at_flow_scale(
                image_path, int(config["flow_max_width_px"]))
            previous = previous_by_sequence.get(source["source_sequence_id"])
            if previous is None:
                values = np.zeros((TOP_K, MOTION_FEATURE_DIMENSION), dtype=np.float32)
                gmc_valid, inlier_ratio = False, 0.0
            else:
                flow, affine, inlier_ratio, gmc_valid = estimate_backward_flow_and_gmc(
                    previous["gray"], current_gray, candidates, scale, config)
                values = candidate_motion_features(
                    flow, affine, inlier_ratio, gmc_valid, candidates,
                    previous["candidates"], int(source["width_px"]),
                    int(source["height_px"]), scale, config)
                gmc_valid_frames += int(gmc_valid)
            stream.write(canonical_line({
                "schema_version": SCHEMA,
                "frame_id": source["frame_id"],
                "source_sequence_id": source["source_sequence_id"],
                "frame_index": source["frame_index"],
                "capture_timestamp_ns": source["capture_timestamp_ns"],
                "candidate_motion": values.tolist(),
                "gmc_valid": gmc_valid,
                "gmc_inlier_ratio": inlier_ratio,
            }))
            previous_by_sequence[source["source_sequence_id"]] = {
                "gray": current_gray,
                "candidates": candidates,
            }
            if (index + 1) % 500 == 0:
                print("[motion] %d/%d frames" % (index + 1, len(aligned)), flush=True)
    receipt = {
        "schema_version": 1,
        # What produced these numbers. Omitting it in 2026-09 made a cache that a
        # different torch/cuDNN could not reproduce look like non-determinism.
        "runtime": runtime_fingerprint(),
        "feature_schema": SCHEMA,
        "feature_dimension": MOTION_FEATURE_DIMENSION,
        "feature_fields": list(FEATURE_FIELDS),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": git_value("rev-parse", "HEAD"),
        "source_git_dirty": bool(git_value(
            "status", "--porcelain", "--untracked-files=no")),
        "split": manifest_meta["split"],
        "records": len(aligned),
        "sequences": len(manifest_meta["sequences"]),
        "gmc_valid_frames": gmc_valid_frames,
        "gmc_valid_rate_excluding_first_frames": (
            gmc_valid_frames / max(len(aligned) - len(manifest_meta["sequences"]), 1)),
        "manifest_sha256": manifest_meta["manifest_sha256"],
        "dataset_receipt_sha256": manifest_meta["source_receipt_sha256"],
        "candidates_sha256": candidate_meta["output_sha256"],
        "config": config,
        "output_sha256": sha256_file(output),
        "elapsed_seconds": time.monotonic() - began,
        "test_status": "FINAL_FROZEN_EVALUATION" if test_open else "NOT_READ_OR_PROCESSED",
        "test_used": test_open,
        "frozen_test_contract_sha256": (
            sha256_file(args.frozen_test_contract) if test_open else None),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[motion] PASS: %d frames, GMC %.4f -> %s" % (
        len(aligned), receipt["gmc_valid_rate_excluding_first_frames"], output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
