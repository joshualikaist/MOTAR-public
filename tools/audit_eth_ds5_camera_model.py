"""Ask which camera-model hypothesis, if any, explains cam0's reprojection residual, with held-out tests.

The published calibration leaves a residual far above the tracking noise. Any extra free parameter will
shrink a residual on the data it was fitted to, so each hypothesis is scored on measurements it never
saw, and the camera rotation is refitted on the held-out split so that extrinsic drift is not mistaken
for an intrinsic error. A hypothesis that only wins on its training split is reported as not generalising.

The tool never rewrites the calibration. Its output is evidence about where the inconsistency lives.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json
from prepare_eth_ds5_review import interpolate_pose
from check_eth_ds5_reprojection import camera_rays, fit_rotation, reproject

HYPOTHESES = (("published", ()), ("radial", ("k",)), ("radial_and_focal", ("k", "f")),
              ("focal", ("f",)), ("principal_point", ("pp",)))


def apply_parameters(K, dist, values, free):
    """Camera matrix and coefficients under the free parameters, in a fixed order."""
    import numpy as np
    K = np.array(K, float).copy()
    dist = np.array(dist, float).copy()
    index = 1
    if "k" in free:
        dist[0], dist[1], dist[4] = values[index:index + 3]
        index += 3
    if "pp" in free:
        K[0, 2] += values[index]
        K[1, 2] += values[index + 1]
        index += 2
    if "f" in free:
        K[0, 0] *= values[index]
        K[1, 1] *= values[index]
        index += 1
    return K, dist


def residual(points, pose, camera, K0, dist0, values, free, period, indices=None, rotation=None):
    import numpy as np
    K, dist = apply_parameters(K0, dist0, values, free)
    shift = values[0]
    world, pixels = [], []
    chosen = points if indices is None else [points[i] for i in indices]
    for point in chosen:
        state = interpolate_pose(pose, point["t"] + shift * period)
        if state is None:
            continue
        world.append(state["xyz_m"])
        pixels.append(point["pixel"])
    world, pixels = np.asarray(world), np.asarray(pixels)
    if len(world) < 8:
        return None, None
    if rotation is None:
        rays = camera_rays(pixels, K, dist)
        bearings = world - np.asarray(camera)
        bearings = bearings / np.linalg.norm(bearings, axis=1, keepdims=True)
        rotation = fit_rotation(rays, bearings)[0]
    error = reproject(rotation, camera, world, K, dist) - pixels
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))), rotation


def initial(free, dist0, shift):
    values = [shift]
    if "k" in free:
        values += [dist0[0], dist0[1], dist0[4]]
    if "pp" in free:
        values += [0.0, 0.0]
    if "f" in free:
        values += [1.0]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--initial-shift", type=float, default=-3.6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    import numpy as np
    from scipy.optimize import minimize
    track = json.loads(args.track.read_text())
    if track["source_commit"] != COMMIT:
        raise ValueError("source commit mismatch")
    calibration = json.loads(args.calibration.read_text())
    K0 = calibration["K-matrix"]
    dist0 = calibration["distCoeff"]
    period = 1.0 / calibration["fps"]
    pose = numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11)
    frames = numeric_table(args.dataset / "dataset5/videos/cam0/cam0_frame_ts.txt", 2)
    stamp = {int(f): t for f, t in frames}
    camera_lines = (args.dataset / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t"))
    points = []
    for key, value in sorted(track["track"].items(), key=lambda kv: int(kv[0])):
        frame_id = int(key)
        if frame_id % args.stride or frame_id not in stamp:
            continue
        t = stamp[frame_id]
        if any(interpolate_pose(pose, t + s * period) is None for s in (-6, 0, 2)):
            continue
        points.append({"frame_id": frame_id, "t": t, "pixel": [value["x"], value["y"]]})
    if len(points) < 40:
        raise ValueError("too few usable tracked points")
    times = np.array([p["t"] for p in points])
    radius = np.linalg.norm(np.array([p["pixel"] for p in points]) - np.array([K0[0][2], K0[1][2]]), axis=1)
    splits = {"alternating": (np.arange(len(points)) % 2 == 0, np.arange(len(points)) % 2 == 1),
              "early_to_late": (times < np.median(times), times >= np.median(times)),
              "late_to_early": (times >= np.median(times), times < np.median(times)),
              "centre_to_edge": (radius < np.median(radius), radius >= np.median(radius))}
    results = {}
    for split, (train_mask, test_mask) in splits.items():
        train = np.nonzero(train_mask)[0]
        test = np.nonzero(test_mask)[0]
        entry = {}
        for name, free in HYPOTHESES:
            start = initial(free, dist0, args.initial_shift)
            if free:
                solution = minimize(
                    lambda v: (residual(points, pose, camera, K0, dist0, v, free, period, train)[0] or 1e6),
                    start, method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 20000, "maxfev": 20000}).x
            else:
                solution = minimize(
                    lambda v: (residual(points, pose, camera, K0, dist0, [v[0]], (), period, train)[0] or 1e6),
                    [start[0]], method="Nelder-Mead", options={"xatol": 1e-6}).x
            train_rms, _ = residual(points, pose, camera, K0, dist0, solution, free, period, train)
            test_rms, _ = residual(points, pose, camera, K0, dist0, solution, free, period, test)
            K, dist = apply_parameters(K0, dist0, solution, free)
            entry[name] = {"train_rms_px": train_rms, "held_out_rms_px": test_rms,
                           "shift_frames": float(solution[0]),
                           "K-matrix": K.tolist(), "distCoeff": dist.tolist(),
                           "generalises": bool(test_rms is not None and train_rms is not None
                                               and test_rms <= 1.5 * train_rms)}
        results[split] = entry
    published_held_out = [results[s]["published"]["held_out_rms_px"] for s in results]
    winners = {s: min(results[s], key=lambda n: results[s][n]["held_out_rms_px"]) for s in results}
    consistent = len(set(winners.values())) == 1 and set(winners.values()) != {"published"}
    noise = track["measurement_noise"]["rms_px"]
    best_held_out = min(results[s][winners[s]]["held_out_rms_px"] for s in results)
    status = ("CAMERA_MODEL_HYPOTHESIS_CONFIRMED" if consistent and best_held_out < 3 * noise
              else "NO_CAMERA_MODEL_HYPOTHESIS_EXPLAINS_THE_RESIDUAL")
    receipt = {"status": status, "source_commit": COMMIT, "points": len(points),
               "measurement_noise_px": noise,
               "published_held_out_rms_px": published_held_out,
               "winner_per_split": winners, "same_winner_everywhere": consistent,
               "best_held_out_rms_px": best_held_out,
               "held_out_over_noise": best_held_out / noise if noise else None,
               "results": results,
               "track_sha256": digests(args.track)[1], "calibration_sha256": digests(args.calibration)[1],
               "tool_sha256": digests(Path(__file__))[1],
               "note": "A hypothesis counts only if it wins on every split and reaches the measurement noise."
                       " Shrinking a training residual proves nothing: extra parameters always can.",
               "runtime": {"python": platform.python_version(), "executable": sys.executable}}
    write_json(args.output / "camera_model_audit.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
