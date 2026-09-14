"""Select the ETH ds5 cam0 frames that can actually resolve the frame_id/container-index alignment.

The stride-150 pilot samples the flight uniformly, so the target moves a median of about 1.5 px between
frames there. Neighbouring alignment shifts then fit almost equally well and the reprojection check
returns several consistent alignments instead of one. This tool picks frames where the ground truth
moves enough per frame for the shifts to separate, using only pose geometry and the calibration's focal
length: no image is read, no box is invented, and the criteria are fixed before any annotation exists.

Ground-truth motion alone is not enough: the drone leaves cam0's field of view during the high passes,
and a frame where the target is not in the picture cannot be annotated. The field of view therefore has
to be predicted, which needs an approximate camera pointing. That pointing is an explicit input, never a
silent guess, and it is recorded with its provenance. It only decides which frames a human is shown; the
reprojection check still re-fits the orientation from reviewed boxes alone, so a wrong pointing wastes
review effort but cannot make a wrong alignment pass.

Selection is preregistration material. Record the thresholds with the queue and do not retune them after
seeing reviewed boxes.
"""
import argparse
import bisect
import json
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json
from prepare_eth_ds5_review import interpolate_pose
import math

from check_eth_ds5_reprojection import alignment_discrimination, reproject, valid_distortion_radius


def discrimination_per_frame(pose, camera, frames, period, focal):
    """Pixels of image motion per frame of timing error, for every frame inside pose support."""
    values = {}
    for frame_id, timestamp in frames:
        measured = alignment_discrimination(pose, camera, [timestamp], period, focal)
        if measured["frames"]:
            values[int(frame_id)] = measured["median_px"]
    return values


def level_camera(azimuth_deg, elevation_deg):
    """Rotation of a roll-free camera pointing at the given ENU azimuth (from East) and elevation."""
    import numpy as np
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    z = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    x = np.cross([0.0, 0.0, 1.0], z)
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        raise ValueError("pointing straight up or down leaves the roll-free frame undefined")
    x = x / norm
    return np.vstack([x, np.cross(z, x), z])


def predicted_pixel(rotation, camera, point, K, dist, resolution, margin_px):
    """Pixel of a ground-truth point under an assumed pointing, or None when it is not in the picture."""
    import numpy as np
    direction = np.asarray(rotation) @ (np.asarray(point) - np.asarray(camera))
    if direction[2] <= 0:
        return None
    if float(np.linalg.norm(direction[:2] / direction[2])) >= valid_distortion_radius(dist):
        return None
    pixel = reproject(np.eye(3), np.zeros(3), [direction], K, dist)[0]
    width, height = resolution
    if margin_px <= pixel[0] <= width - margin_px and margin_px <= pixel[1] <= height - margin_px:
        return [float(pixel[0]), float(pixel[1])]
    return None


def select(values, ranges, min_discrimination, min_range_m, min_separation, limit, in_frame=None):
    """Highest-motion frames first, keeping them apart in time so one manoeuvre cannot dominate."""
    if min_separation < 1 or limit < 1:
        raise ValueError("separation and limit must be positive")
    eligible = [f for f, v in values.items() if v >= min_discrimination and ranges[f] >= min_range_m
                and (in_frame is None or in_frame.get(f) is not None)]
    chosen = []
    for frame in sorted(eligible, key=lambda f: -values[f]):
        if all(abs(frame - other) >= min_separation for other in chosen):
            chosen.append(frame)
        if len(chosen) >= limit:
            break
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-discrimination-px", type=float, default=4.0,
                        help="pixels of image motion per frame of timing error; below about 4 px the"
                             " alignment shifts cannot be separated once annotation noise is included")
    parser.add_argument("--min-range-m", type=float, default=25.0,
                        help="skip the close-range take-off phase, where a small centring error is a"
                             " large angular error")
    parser.add_argument("--min-separation-frames", type=int, default=30)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pointing-az-deg", type=float, required=True,
                        help="approximate optical-axis azimuth, degrees from East counter-clockwise (ENU)")
    parser.add_argument("--pointing-el-deg", type=float, required=True,
                        help="approximate optical-axis elevation in degrees")
    parser.add_argument("--pointing-source", required=True,
                        help="where the pointing came from; recorded verbatim in the queue receipt")
    parser.add_argument("--fov-margin-px", type=float, default=120.0,
                        help="border kept clear, absorbing camera roll that the roll-free model ignores")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output; preserve previous selections")
    calibration = json.loads(args.calibration.read_text())
    focal = (calibration["K-matrix"][0][0] + calibration["K-matrix"][1][1]) / 2
    period = 1.0 / calibration["fps"]
    pose = numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11)
    frames = numeric_table(args.dataset / "dataset5/videos/cam0/cam0_frame_ts.txt", 2)
    camera_lines = (args.dataset / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t"))
    overlap = [(f, t) for f, t in frames if pose[0][0] <= t <= pose[-1][0]]
    values = discrimination_per_frame(pose, camera, overlap, period, focal)
    times = {int(f): t for f, t in overlap}
    ranges = {}
    for frame_id in values:
        sample = interpolate_pose(pose, times[frame_id])
        ranges[frame_id] = sum((sample["xyz_m"][i] - camera[i]) ** 2 for i in range(3)) ** 0.5
    rotation = level_camera(args.pointing_az_deg, args.pointing_el_deg)
    predicted = {}
    for frame_id in values:
        sample = interpolate_pose(pose, times[frame_id])
        predicted[frame_id] = predicted_pixel(rotation, camera, sample["xyz_m"], calibration["K-matrix"],
                                              calibration["distCoeff"], calibration["resolution"],
                                              args.fov_margin_px)
    chosen = select(values, ranges, args.min_discrimination_px, args.min_range_m,
                    args.min_separation_frames, args.limit, predicted)
    pose_times = [r[0] for r in pose]
    queue = []
    for frame_id in chosen:
        timestamp = times[frame_id]
        right = min(bisect.bisect_left(pose_times, timestamp), len(pose_times) - 1)
        left = max(0, right - 1)
        queue.append({"frame_id": frame_id, "opencv_index": frame_id - 1,
                      "project_timestamp_s": timestamp, "target_id": "drone0",
                      "pose_bracket_rows_zero_based": [left, right],
                      "pose_bracket_gap_s": pose_times[right] - pose_times[left],
                      "tracking_status_uninterpreted": [pose[left][-1], pose[right][-1]],
                      "alignment_discrimination_px": values[frame_id],
                      "gt_slant_range_m": ranges[frame_id],
                      "predicted_pixel_under_assumed_pointing": predicted[frame_id],
                      "box_xyxy": None, "identity_verified": False,
                      "annotation_status": "PENDING_MANUAL_REVIEW",
                      "measurement_eligible": False})
    if not queue:
        raise ValueError("no frame met the selection criteria; loosen them explicitly, never silently")
    selected = [values[f] for f in chosen]
    write_json(args.output, {
        "source_commit": COMMIT,
        "purpose": "resolve the frame_id/container-index alignment; not a measurement split",
        "criteria": {"min_discrimination_px": args.min_discrimination_px, "min_range_m": args.min_range_m,
                     "min_separation_frames": args.min_separation_frames, "limit": args.limit,
                     "fov_margin_px": args.fov_margin_px, "fixed_before_any_box": True},
        "assumed_pointing": {"azimuth_deg_from_east_ccw": args.pointing_az_deg,
                             "elevation_deg": args.pointing_el_deg, "roll_deg_assumed": 0.0,
                             "source": args.pointing_source,
                             "used_for": "choosing which frames a human is shown",
                             "not_used_for": "any measurement; the reprojection check re-fits the"
                                             " orientation from reviewed boxes alone",
                             "verified": False},
        "overlap_frames_considered": len(values),
        "overlap_frames_predicted_in_view": sum(1 for v in predicted.values() if v is not None),
        "discrimination_px": {"selected_min": min(selected), "selected_median": sorted(selected)[len(selected) // 2],
                              "selected_max": max(selected)},
        "opencv_index_is_a_hypothesis": "container index is chosen at render time by --index-offset",
        "calibration_sha256": digests(args.calibration)[1],
        "tool_sha256": digests(Path(__file__))[1],
        "runtime": {"python": platform.python_version(), "executable": sys.executable},
        "stride_frames": None, "frames": queue})
    print(json.dumps({"selected": chosen, "discrimination_px_median": sorted(selected)[len(selected) // 2],
                      "range_m": [min(ranges[f] for f in chosen), max(ranges[f] for f in chosen)]}, indent=2))


if __name__ == "__main__":
    main()
