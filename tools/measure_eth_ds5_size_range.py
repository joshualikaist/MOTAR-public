"""E3-S: how accurately does apparent size in cam0 recover the target's metric slant range?

Runs exactly the analysis fixed in results/eth_ds5_e3s_2026-09-10/PREREGISTRATION.md. The estimator uses
the target's position ground truth and a focal length, and nothing else: no camera orientation, no
principal point, no extrinsic. That is what lets E3-S proceed while the orientation and the timing offset
are still unresolved.

The unit of analysis is a block of project time, never a frame, because neighbouring frames of a tracked
target are not independent. Every evaluated frame is out of sample under leave-one-block-out.

The output is size-based range estimation performance. It is not a range error attributable to attitude,
and it must not be read as one; that separation is E3-P, which is gated.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json
from prepare_eth_ds5_review import interpolate_pose

BLOCK_SECONDS = 15.0
RANGE_BINS = ((30.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 110.0))
BIN_MIN_BLOCKS = 3
BIN_MIN_FRAMES_PER_BLOCK = 20
MIN_EVAL_BLOCKS = 6
MIN_EVAL_FRAMES = 500
G2_MEDIAN_ABS_REL = 0.25
G3_BLOCK_SPREAD = 0.20
BORDER_PX = 40


def radial_magnification(pixels, K, dist):
    """Area magnification of the radial model at each pixel, as sqrt(radial x tangential).

    Used only for the pre-registered distortion sensitivity: a measured size is divided by this to ask
    what it would have been without the published distortion.
    """
    import cv2
    import numpy as np
    K = np.asarray(K, np.float64)
    points = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    normalised = cv2.undistortPoints(points, K, np.asarray(dist, np.float64)).reshape(-1, 2)
    r = np.linalg.norm(normalised, axis=1)
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    radial = 1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6
    derivative = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
    return np.sqrt(np.abs(derivative * radial))


def load_points(track, pose, stamps, camera, shift_frames, period, K=None, dist=None, correct_distortion=False):
    """Included measurements with their ground-truth range, following the pre-registered exclusions."""
    import numpy as np
    rows = []
    for key, value in sorted(track["track"].items(), key=lambda kv: int(kv[0])):
        frame_id = int(key)
        t = stamps.get(frame_id)
        if t is None:
            continue
        x, y = value["x"], value["y"]
        if not (BORDER_PX <= x <= 1920 - BORDER_PX and BORDER_PX <= y <= 1080 - BORDER_PX):
            continue
        state = interpolate_pose(pose, t + shift_frames * period)
        if state is None:
            continue
        distance = float(np.linalg.norm(np.asarray(state["xyz_m"]) - np.asarray(camera)))
        rows.append({"frame_id": frame_id, "t": t, "range_m": distance, "x": x, "y": y,
                     "sqrt_pixels": float(value["pixels"]) ** 0.5,
                     "extent_y": float(value["extent_y"]), "extent_x": float(value["extent_x"]),
                     "tracking_status": state["tracking_status_pair_uninterpreted"],
                     "bracket_gap_s": state["bracket_gap_s"]})
    if correct_distortion and rows:
        magnification = radial_magnification([[r["x"], r["y"]] for r in rows], K, dist)
        for row, m in zip(rows, magnification):
            for key in ("sqrt_pixels", "extent_y", "extent_x"):
                row[key] = row[key] / float(m)
    return rows


def block_index(t):
    return int(t // BLOCK_SECONDS)


def group_blocks(points):
    grouped = {}
    for point in points:
        grouped.setdefault(block_index(point["t"]), []).append(point)
    return dict(sorted(grouped.items()))


def fit_size(points, focal, measure):
    """Effective target size: the median of (apparent size x range / focal) over the fitting frames."""
    values = [p[measure] * p["range_m"] / focal for p in points if p[measure] > 0]
    if not values:
        raise ValueError("no usable size measurements to fit")
    return statistics.median(values)


def relative_errors(points, size, focal, measure):
    """Signed relative range error of Z_hat = focal * size / apparent size."""
    errors = []
    for p in points:
        if p[measure] <= 0:
            continue
        estimate = focal * size / p[measure]
        errors.append((estimate - p["range_m"]) / p["range_m"])
    return errors


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def leave_one_block_out(blocks, focal, measure):
    """Fit on every block but one, evaluate on the one held out. Every frame is out of sample."""
    results = {}
    for held in blocks:
        others = [p for index, points in blocks.items() if index != held for p in points]
        if not others:
            continue
        size = fit_size(others, focal, measure)
        errors = relative_errors(blocks[held], size, focal, measure)
        if not errors:
            continue
        absolute = [abs(e) for e in errors]
        ranges = [p["range_m"] for p in blocks[held]]
        results[held] = {"frames": len(errors), "fitted_size_m": size,
                         "range_span_m": [min(ranges), max(ranges)],
                         "median_signed_rel": statistics.median(errors),
                         "median_abs_rel": statistics.median(absolute),
                         "p90_abs_rel": percentile(absolute, 0.9),
                         "errors": errors, "points": blocks[held]}
    return results


def bin_summary(per_block):
    """Per range bin, aggregated over blocks, honouring the pre-registered minimum sample."""
    summary = {}
    for low, high in RANGE_BINS:
        contributions = {}
        for index, entry in per_block.items():
            inside = [(p, e) for p, e in zip(entry["points"], entry["errors"]) if low <= p["range_m"] < high]
            if len(inside) >= BIN_MIN_FRAMES_PER_BLOCK:
                contributions[index] = inside
        label = "%d-%d m" % (low, high)
        if len(contributions) < BIN_MIN_BLOCKS:
            summary[label] = {"status": "INSUFFICIENT", "blocks": len(contributions),
                              "frames": sum(len(v) for v in contributions.values())}
            continue
        block_signed = [statistics.median([e for _, e in v]) for v in contributions.values()]
        block_absolute = [statistics.median([abs(e) for _, e in v]) for v in contributions.values()]
        block_metres = [statistics.median([abs(e) * p["range_m"] for p, e in v]) for v in contributions.values()]
        summary[label] = {"status": "REPORTED", "blocks": len(contributions),
                          "frames": sum(len(v) for v in contributions.values()),
                          "bias_median_signed_rel": statistics.median(block_signed),
                          "median_abs_rel": statistics.median(block_absolute),
                          "p90_abs_rel_across_blocks": percentile(block_absolute, 0.9),
                          "median_abs_error_m": statistics.median(block_metres),
                          "per_block_median_abs_rel": dict(zip(map(str, contributions), block_absolute))}
    return summary


def verdict(per_block, bins):
    """The four gates, exactly as preregistered."""
    medians = [entry["median_abs_rel"] for entry in per_block.values()]
    frames = sum(entry["frames"] for entry in per_block.values())
    spread = (percentile(medians, 0.9) - percentile(medians, 0.1)) if medians else None
    reported = [name for name, value in bins.items() if value["status"] == "REPORTED"]
    gates = {
        "G1_coverage": {"pass": len(per_block) >= MIN_EVAL_BLOCKS and frames >= MIN_EVAL_FRAMES,
                        "blocks": len(per_block), "frames": frames,
                        "required": [MIN_EVAL_BLOCKS, MIN_EVAL_FRAMES]},
        "G2_central": {"pass": bool(medians) and statistics.median(medians) < G2_MEDIAN_ABS_REL,
                       "median_of_block_medians": statistics.median(medians) if medians else None,
                       "threshold": G2_MEDIAN_ABS_REL},
        "G3_stability": {"pass": spread is not None and spread < G3_BLOCK_SPREAD,
                         "p90_minus_p10": spread, "threshold": G3_BLOCK_SPREAD},
        "G4_range_coverage": {"pass": len(reported) >= 3, "bins_reported": reported, "threshold": 3}}
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    return {"status": "SIZE_RANGE_USABLE" if not failed else "SIZE_RANGE_NOT_USABLE",
            "failed_gates": failed, "gates": gates}


def analyse(track, pose, stamps, camera, focal, period, measure, shift, K=None, dist=None, correct=False):
    points = load_points(track, pose, stamps, camera, shift, period, K, dist, correct)
    blocks = {i: v for i, v in group_blocks(points).items() if len(v) >= BIN_MIN_FRAMES_PER_BLOCK}
    per_block = leave_one_block_out(blocks, focal, measure)
    bins = bin_summary(per_block)
    outcome = verdict(per_block, bins)
    lean = {str(i): {k: v for k, v in entry.items() if k not in ("errors", "points")}
            for i, entry in per_block.items()}
    for name, value in bins.items():
        value.pop("points", None)
    return {"points": len(points), "blocks": len(blocks), "per_block": lean, "bins": bins, **outcome}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-frames", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    track = json.loads(args.track.read_text())
    if track["source_commit"] != COMMIT:
        raise ValueError("source commit mismatch")
    calibration = json.loads(args.calibration.read_text())
    focal = (calibration["K-matrix"][0][0] + calibration["K-matrix"][1][1]) / 2
    period = 1.0 / calibration["fps"]
    pose = numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11)
    frames = numeric_table(args.dataset / "dataset5/videos/cam0/cam0_frame_ts.txt", 2)
    stamps = {int(f): t for f, t in frames}
    camera_lines = (args.dataset / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t"))
    primary = analyse(track, pose, stamps, camera, focal, period, "sqrt_pixels", args.shift_frames)
    quality = load_points(track, pose, stamps, camera, args.shift_frames, period)
    statuses = {}
    for point in quality:
        statuses[str(sorted(set(point["tracking_status"])))] = statuses.get(
            str(sorted(set(point["tracking_status"]))), 0) + 1
    sensitivities = {
        "time_shift": {str(s): analyse(track, pose, stamps, camera, focal, period, "sqrt_pixels", s)
                       for s in (0.0, -3.6)},
        "distortion_corrected": analyse(track, pose, stamps, camera, focal, period, "sqrt_pixels",
                                        args.shift_frames, calibration["K-matrix"], calibration["distCoeff"], True),
        "size_measure": {m: analyse(track, pose, stamps, camera, focal, period, m, args.shift_frames)
                         for m in ("extent_y", "extent_x")}}
    receipt = {"status": primary["status"], "stage": "E3-S", "source_commit": COMMIT,
               "preregistration": "results/eth_ds5_e3s_2026-09-10/PREREGISTRATION.md",
               "focal_px": focal, "block_seconds": BLOCK_SECONDS, "shift_frames": args.shift_frames,
               "measurement_noise_px": track["measurement_noise"]["rms_px"],
               "data_quality": {"tracked_frames": track["frames_tracked"], "included_frames": len(quality),
                                "tracking_status_counts_uninterpreted": statuses,
                                "max_pose_bracket_gap_s": max(p["bracket_gap_s"] for p in quality),
                                "range_span_m": [min(p["range_m"] for p in quality),
                                                 max(p["range_m"] for p in quality)]},
               "primary": primary, "sensitivities": sensitivities,
               "interpretation": "size-based range estimation performance only; it bundles viewing-aspect"
                                 " size change, segmentation noise, motion blur, unmodelled distortion and"
                                 " the unresolved timing offset, and separates none of them. E3-P is blocked.",
               "track_sha256": digests(args.track)[1], "calibration_sha256": digests(args.calibration)[1],
               "tool_sha256": digests(Path(__file__))[1],
               "runtime": {"python": platform.python_version(), "executable": sys.executable}}
    write_json(args.output / "e3s_result.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("primary", "sensitivities")}, indent=2))
    print("PRIMARY:", json.dumps({k: v for k, v in primary.items() if k != "per_block"}, indent=2))


if __name__ == "__main__":
    main()
