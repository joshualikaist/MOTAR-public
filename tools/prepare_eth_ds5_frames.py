"""Decode a pending ETH annotation pilot, without inventing boxes or target identity."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from fractions import Fraction

from prepare_eth_ds5 import COMMIT, digests, write_json


def validate_video(stream, expected_count, calibration, allow_count_mismatch=False):
    if stream.get("codec_type") != "video":
        raise ValueError("not a video stream")
    count = int(stream.get("nb_frames", -1))
    if count < 1:
        raise ValueError("unknown frame count")
    if count != expected_count and not allow_count_mismatch:
        raise ValueError("video/timestamp frame count mismatch")
    fps = float(Fraction(stream["avg_frame_rate"]))
    if abs(fps - calibration["fps"]) > 0.01:
        raise ValueError("video FPS differs from calibration candidate")
    dimensions_match = [int(stream["width"]), int(stream["height"])] == calibration["resolution"]
    return {"frame_count": count, "published_timestamp_rows": expected_count,
            "frame_index_alignment": "ASSUMED_frame_id_minus_1" if count == expected_count else "UNRESOLVED_count_mismatch",
            "fps": fps, "dimensions_match_calibration_candidate": dimensions_match,
            "calibration_validated": False,
            "note": "Dimensions/FPS are necessary diagnostics, not lens/crop/reprojection validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--intake-receipt", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--allow-count-mismatch", action="store_true",
                        help="render review images even if container frames != timestamp rows; the receipt then"
                             " marks every frame's index alignment unresolved (review only, never measurement)")
    parser.add_argument("--index-offset", type=int, default=-1,
                        help="container_index = frame_id + OFFSET. The published stamps do not fix this integer;"
                             " 0 is the value supported by the video track's edit list, -1 reproduces the first"
                             " pilot. Whichever is used is recorded, and the reprojection check searches shifts.")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output; preserve previous pilot")
    queue = json.loads(args.queue.read_text())
    receipt = json.loads(args.intake_receipt.read_text())
    if queue["source_commit"] != COMMIT or receipt["source_commit"] != COMMIT:
        raise ValueError("source commit mismatch")
    calibration = json.loads(args.calibration.read_text())
    expected = next(f["sha256"] for f in receipt["files"] if f["path"] == "calibration/sony5100/sony5100.json")
    if digests(args.calibration)[1] != expected:
        raise ValueError("calibration hash mismatch")
    info = json.loads(subprocess.check_output([
        args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(args.video)], text=True))
    result = validate_video(info["streams"][0], receipt["video_timestamp_rows"], calibration, args.allow_count_mismatch)
    unresolved = result["frame_index_alignment"] != "ASSUMED_frame_id_minus_1"
    import cv2
    cv2.setNumThreads(1)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise ValueError("video decoder could not open file")
    args.output.mkdir(parents=True)
    rendered = []
    try:
        for row in queue["frames"]:
            if row["box_xyxy"] is not None or row["measurement_eligible"] or row["identity_verified"]:
                raise ValueError("expected an unannotated pilot queue")
            index = row["frame_id"] + args.index_offset
            if not 0 <= index < result["frame_count"]:
                raise ValueError("container index %d out of range for frame_id %d" % (index, row["frame_id"]))
            if not video.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError("frame seek failed")
            ok, frame = video.read()
            if not ok or round(video.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise ValueError("frame decode/position mismatch")
            path = args.output / ("frame_%06d.png" % row["frame_id"])
            if not cv2.imwrite(str(path), frame):
                raise ValueError("image write failed")
            rendered.append(dict(row, image=path.name, image_sha256=digests(path)[1],
                                 container_index=index, opencv_index=index,
                                 queue_declared_opencv_index=row["opencv_index"],
                                 index_alignment_unresolved=unresolved))
    finally:
        video.release()
    status = "PILOT_IMAGES_READY_ANNOTATION_PENDING" + ("_ALIGNMENT_UNRESOLVED" if unresolved else "")
    result.update({"status": status, "source_commit": COMMIT,
                   "container_index_offset": args.index_offset,
                   "container_index_rule": "container_index = frame_id + %d" % args.index_offset,
                   "reprojection_shift_of_true_offset_d": "shift = %d - d" % args.index_offset,
                   "video_sha256": digests(args.video)[1], "video_stream": info["streams"][0],
                   "queue_sha256": digests(args.queue)[1], "intake_receipt_sha256": digests(args.intake_receipt)[1],
                   "tool_sha256": digests(Path(__file__))[1], "frames": rendered,
                   "runtime": {"python": platform.python_version(), "executable": sys.executable,
                               "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads()}})
    write_json(args.output / "receipt.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "video_stream")}, indent=2))


if __name__ == "__main__":
    main()
