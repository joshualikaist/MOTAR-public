"""Build human-review material for the ETH ds5 cam0 pilot frames: is drone0 visible, and where?

Nothing here identifies the target. The tool only adds (a) the ground-truth geometry of drone0 relative
to cam0 at each frame time (range, bearing, elevation, speed; camera orientation is unknown, so no
pixel prediction), (b) temporal-difference motion cues with candidate blobs that a reviewer must judge,
and (c) an empty review template. Motion candidates are hints, never boxes.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json

TEMPLATE_COLUMNS = ["frame_id", "opencv_index", "project_timestamp_s", "gt_slant_range_m", "gt_azimuth_deg_from_east_ccw",
                    "gt_elevation_deg", "gt_speed_mps", "drone0_visible", "box_x1", "box_y1", "box_x2", "box_y2",
                    "other_drones_visible_count", "confidence", "reviewer", "notes"]


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def interpolate_pose(pose, t, max_gap_s=0.5):
    """Linear interpolation of drone0 pose at project time t; None outside support or across long gaps."""
    times = [r[0] for r in pose]
    if t < times[0] or t > times[-1]:
        return None
    import bisect
    right = bisect.bisect_left(times, t)
    if right == 0:
        right = 1
    left = right - 1
    gap = times[right] - times[left]
    if gap > max_gap_s or gap <= 0:
        return None
    w = (t - times[left]) / gap
    a, b = pose[left], pose[right]
    xyz = [a[i] + w * (b[i] - a[i]) for i in (1, 2, 3)]
    rpy = [wrap_deg(a[i] + w * wrap_deg(b[i] - a[i])) for i in (4, 5, 6)]
    velocity = [(b[i] - a[i]) / gap for i in (1, 2, 3)]
    return {"xyz_m": xyz, "rpy_deg_uninterpreted": rpy, "std_xyz_m": [max(a[i], b[i]) for i in (7, 8, 9)],
            "speed_mps": math.sqrt(sum(v * v for v in velocity)), "bracket_gap_s": gap,
            "tracking_status_pair_uninterpreted": [a[10], b[10]]}


def bearing_from_camera(camera_xyz, target_xyz):
    d = [t - c for t, c in zip(target_xyz, camera_xyz)]
    horizontal = math.hypot(d[0], d[1])
    return {"slant_range_m": math.sqrt(sum(v * v for v in d)), "horizontal_distance_m": horizontal,
            "azimuth_deg_from_east_ccw": math.degrees(math.atan2(d[1], d[0])),
            "elevation_deg": math.degrees(math.atan2(d[2], horizontal))}


def motion_candidates(prev, cur, nxt, threshold=24, min_area=4, max_candidates=8):
    """Blobs that differ from BOTH temporal neighbours; returns boxes sorted by area (hints only)."""
    import cv2
    import numpy as np
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in (prev, cur, nxt)]
    diff = np.minimum(np.abs(g[1] - g[0]), np.abs(g[1] - g[2]))
    mask = (diff > threshold).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area >= min_area:
            boxes.append({"xyxy": [x, y, x + w, y + h], "area_px": area})
    boxes.sort(key=lambda b: -b["area_px"])
    return boxes[:max_candidates], int(mask.sum())


def render_panel(frame, candidates, lines, crops=3, crop_half=48, zoom=3):
    import cv2
    import numpy as np
    canvas = frame.copy()
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = c["xyxy"]
        cv2.rectangle(canvas, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 255), 1)
        cv2.putText(canvas, "m%d" % i, (x1 - 6, max(12, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    strip = np.zeros((22 * (len(lines) + 1), canvas.shape[1], 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(strip, text, (8, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles = []
    for c in candidates[:crops]:
        cx, cy = [(c["xyxy"][k] + c["xyxy"][k + 2]) // 2 for k in (0, 1)]
        x1, y1 = max(0, cx - crop_half), max(0, cy - crop_half)
        crop = frame[y1:y1 + 2 * crop_half, x1:x1 + 2 * crop_half]
        crop = cv2.resize(crop, (2 * crop_half * zoom, 2 * crop_half * zoom), interpolation=cv2.INTER_NEAREST)
        tiles.append(cv2.copyMakeBorder(crop, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    if tiles:
        row = np.concatenate(tiles, axis=1)
        pad = np.zeros((row.shape[0], max(0, canvas.shape[1] - row.shape[1]), 3), np.uint8)
        row = np.concatenate([row, pad], axis=1)[:, :canvas.shape[1]]
        canvas = np.concatenate([canvas, row], axis=0)
    return np.concatenate([strip, canvas], axis=0)


def read_frame(video, index):
    import cv2
    if not video.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError("seek failed")
    ok, frame = video.read()
    if not ok:
        raise ValueError("decode failed at %d" % index)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor-frames", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output; preserve previous review material")
    pilot = json.loads((args.pilot_dir / "receipt.json").read_text())
    if pilot["source_commit"] != COMMIT:
        raise ValueError("source commit mismatch")
    if pilot["video_sha256"] != digests(args.video)[1]:
        raise ValueError("video differs from the pilot receipt")
    pose = numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11)
    camera_lines = (args.dataset / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t"))
    import cv2
    cv2.setNumThreads(1)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError("cannot open video")
    args.output.mkdir(parents=True)
    rows, thumbs = [], []
    try:
        for row in pilot["frames"]:
            if row["box_xyxy"] is not None or row["identity_verified"] or row["measurement_eligible"]:
                raise ValueError("pilot receipt must be unannotated")
            image = args.pilot_dir / row["image"]
            if digests(image)[1] != row["image_sha256"]:
                raise ValueError("pilot image hash mismatch: " + row["image"])
            frame = cv2.imread(str(image))
            index = row.get("container_index", row["opencv_index"])
            prev = read_frame(video, max(0, index - args.neighbor_frames))
            nxt = read_frame(video, min(pilot["frame_count"] - 1, index + args.neighbor_frames))
            candidates, moving_px = motion_candidates(prev, frame, nxt)
            gt = interpolate_pose(pose, row["project_timestamp_s"])
            geometry = bearing_from_camera(camera, gt["xyz_m"]) if gt else None
            lines = ["frame %d  container_index %d (= frame_id %+d)  t=%.3f s  (alignment %s)" % (
                row["frame_id"], index, index - row["frame_id"], row["project_timestamp_s"],
                pilot["frame_index_alignment"])]
            if gt:
                lines.append("drone0 GT: range %.1f m  az %.1f deg(E ccw)  elev %.1f deg  speed %.2f m/s  status %s" % (
                    geometry["slant_range_m"], geometry["azimuth_deg_from_east_ccw"], geometry["elevation_deg"],
                    gt["speed_mps"], gt["tracking_status_pair_uninterpreted"]))
            else:
                lines.append("drone0 GT: NOT INTERPOLABLE at this time")
            lines.append("yellow m# = motion cue vs frames %+d/%+d; NOT identity; %d candidates" % (
                -args.neighbor_frames, args.neighbor_frames, len(candidates)))
            panel = render_panel(frame, candidates, lines)
            panel_path = args.output / ("review_%06d.jpg" % row["frame_id"])
            if not cv2.imwrite(str(panel_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError("panel write failed")
            thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, str(row["frame_id"]), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            thumbs.append(thumb)
            rows.append({"frame_id": row["frame_id"], "container_index": index, "opencv_index": index,
                         "project_timestamp_s": row["project_timestamp_s"],
                         "index_alignment_unresolved": row.get("index_alignment_unresolved", True),
                         "drone0_gt": gt, "drone0_geometry_from_cam0": geometry,
                         "motion_candidates_hint_only": candidates, "moving_pixels": moving_px,
                         "panel": panel_path.name, "panel_sha256": digests(panel_path)[1],
                         "source_image": row["image"], "source_image_sha256": row["image_sha256"]})
    finally:
        video.release()
    import numpy as np
    cols = 6
    while len(thumbs) % cols:
        thumbs.append(np.zeros((180, 320, 3), np.uint8))
    sheet = np.concatenate([np.concatenate(thumbs[i:i + cols], axis=1) for i in range(0, len(thumbs), cols)], axis=0)
    sheet_path = args.output / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    template = args.output / "review_template.csv"
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for r in rows:
            g = r["drone0_geometry_from_cam0"] or {}
            writer.writerow({"frame_id": r["frame_id"], "opencv_index": r["opencv_index"],
                             "project_timestamp_s": r["project_timestamp_s"],
                             "gt_slant_range_m": g.get("slant_range_m", ""), "gt_azimuth_deg_from_east_ccw": g.get("azimuth_deg_from_east_ccw", ""),
                             "gt_elevation_deg": g.get("elevation_deg", ""),
                             "gt_speed_mps": (r["drone0_gt"] or {}).get("speed_mps", "")})
    receipt = {"status": "REVIEW_MATERIAL_READY_IDENTITY_UNVERIFIED", "source_commit": COMMIT,
               "camera_xyz_m": camera, "pilot_receipt_sha256": digests(args.pilot_dir / "receipt.json")[1],
               "video_sha256": pilot["video_sha256"], "frame_index_alignment": pilot["frame_index_alignment"],
               "container_index_offset": pilot.get("container_index_offset"),
               "container_index_rule": pilot.get("container_index_rule"),
               "neighbor_frames": args.neighbor_frames, "frames": rows,
               "contact_sheet_sha256": digests(sheet_path)[1], "review_template_sha256": digests(template)[1],
               "tool_sha256": digests(Path(__file__))[1],
               "runtime": {"python": platform.python_version(), "executable": sys.executable, "opencv": cv2.__version__}}
    write_json(args.output / "receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
