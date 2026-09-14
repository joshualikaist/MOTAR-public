"""Track the reviewed target through consecutive cam0 frames, to turn a handful of boxes into many.

A sparse review answers whether a frame shows the target. It cannot say whether the camera model is
right, because a handful of points lets any extra parameter absorb the error. This tool propagates
reviewed centres through neighbouring frames by local intensity centroiding, which gives hundreds of
measurements whose noise can be measured directly from trajectory smoothness.

The search window follows the target's apparent size: a fixed window silently clips a close target's
blob, which would bias every size measurement derived from it. The window each frame is set from the
previous accepted extent, and a measurement whose blob reaches the window edge is rejected rather than
recorded truncated.

It measures positions; it decides nothing. Every centre is an image measurement seeded by a human- or
review-supplied centre, and ground truth is never consulted.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import COMMIT, digests, write_json


def centroid(gray, seed, half, min_contrast, max_pixels):
    """Darkness-weighted centre of the dark blob near `seed`, or None when it is not resolved."""
    import numpy as np
    x0, y0 = max(0, int(seed[0]) - half), max(0, int(seed[1]) - half)
    window = gray[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
    if window.size == 0:
        return None
    background = float(np.percentile(window, 85))
    contrast = background - float(window.min())
    if contrast < min_contrast:
        return None
    dark = np.clip(background - window, 0, None)
    dark[dark < 0.4 * contrast] = 0
    ys, xs = np.nonzero(dark)
    if len(xs) < 4 or len(xs) > max_pixels:
        return None
    weights = dark[ys, xs]
    return {"x": float((xs * weights).sum() / weights.sum()) + x0,
            "y": float((ys * weights).sum() / weights.sum()) + y0,
            "contrast": contrast, "pixels": int(len(xs)),
            "extent_x": int(xs.max() - xs.min() + 1), "extent_y": int(ys.max() - ys.min() + 1)}


def smoothness_noise(track, window=15, degree=2):
    """Measurement noise, from the deviation of consecutive centres about a local polynomial.

    The target's true image path is smooth over a fraction of a second, so whatever does not fit a low
    order polynomial there is measurement noise. This needs no ground truth and no camera model.
    """
    import numpy as np
    frames = sorted(track)
    residuals = []
    for start in range(0, len(frames) - window, window):
        block = frames[start:start + window]
        if block[-1] - block[0] != window - 1:
            continue
        f = np.asarray(block, float)
        x = np.asarray([track[i]["x"] for i in block])
        y = np.asarray([track[i]["y"] for i in block])
        residuals.append(np.hypot(x - np.polyval(np.polyfit(f, x, degree), f),
                                  y - np.polyval(np.polyfit(f, y, degree), f)))
    if not residuals:
        return None
    stacked = np.concatenate(residuals)
    return {"rms_px": float(np.sqrt((stacked ** 2).mean())), "samples": int(len(stacked)),
            "window_frames": window, "polynomial_degree": degree}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True,
                        help="JSON mapping container frame index to [x, y] reviewed centres")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--half-window", type=int, default=14, help="minimum, and the window used at a seed")
    parser.add_argument("--max-half-window", type=int, default=40)
    parser.add_argument("--min-contrast", type=float, default=25.0)
    parser.add_argument("--max-pixels", type=int, default=1500)
    parser.add_argument("--max-step-px", type=float, default=14.0)
    parser.add_argument("--skip-below-row", type=float, default=740.0,
                        help="rows below this are ground clutter for cam0; tracking stops rather than"
                             " latching onto people or fence posts")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    seeds = {int(k): v for k, v in json.loads(args.seeds.read_text()).items()}
    if not seeds:
        raise ValueError("no seeds")
    import cv2
    import numpy as np
    cv2.setNumThreads(4)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("cannot open video")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.first)
    track, previous, before = {}, None, None
    half = args.half_window
    try:
        for frame_id in range(args.first, args.last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id in seeds:
                predicted = seeds[frame_id]
            elif previous is not None and before is not None:
                predicted = (2 * previous[1] - before[1], 2 * previous[2] - before[2])
            elif previous is not None:
                predicted = (previous[1], previous[2])
            else:
                continue
            if predicted[1] > args.skip_below_row:
                before, previous = previous, None
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            found = centroid(gray, predicted, half, args.min_contrast, args.max_pixels)
            if found is None:
                before, previous, half = previous, None, args.half_window
                continue
            if max(found["extent_x"], found["extent_y"]) >= 2 * half - 1:
                # the blob reaches the window edge, so its extent is truncated: widen and drop this frame
                half = min(args.max_half_window, max(half + 8, half * 2))
                before, previous = previous, None
                continue
            found["half_window"] = half
            if previous is not None and np.hypot(found["x"] - previous[1], found["y"] - previous[2]) > args.max_step_px:
                before, previous = previous, None
                continue
            track[frame_id] = found
            half = int(min(args.max_half_window,
                           max(args.half_window, round(0.9 * max(found["extent_x"], found["extent_y"])) + 10)))
            before, previous = previous, (frame_id, found["x"], found["y"])
    finally:
        capture.release()
    if not track:
        raise ValueError("tracking produced no measurements")
    noise = smoothness_noise(track)
    frames = sorted(track)
    write_json(args.output, {
        "status": "TRACK_MEASURED_NO_GROUND_TRUTH_USED", "source_commit": COMMIT,
        "video_sha256": digests(args.video)[1], "seeds_sha256": digests(args.seeds)[1],
        "frames_tracked": len(track), "frame_range": [frames[0], frames[-1]],
        "seed_frames": sorted(seeds), "measurement_noise": noise,
        "parameters": {"half_window": args.half_window, "max_half_window": args.max_half_window,
                       "adaptive_window": True, "min_contrast": args.min_contrast,
                       "max_pixels": args.max_pixels, "max_step_px": args.max_step_px,
                       "skip_below_row": args.skip_below_row},
        "tool_sha256": digests(Path(__file__))[1],
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "opencv": cv2.__version__},
        "track": {str(f): {k: (round(v, 3) if isinstance(v, float) else v) for k, v in track[f].items()}
                  for f in frames}})
    print(json.dumps({"frames_tracked": len(track), "frame_range": [frames[0], frames[-1]],
                      "measurement_noise": noise}, indent=2))


if __name__ == "__main__":
    main()
