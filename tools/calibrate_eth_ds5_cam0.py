"""Recompute the Sony a5100 intrinsics from the dataset's own chessboard images, and test the published file.

The published `calibration/sony5100/sony5100.json` did not reproject drone0 consistently in ds5 cam0. Two
explanations survive that observation: the file is stale or mis-assigned, or the ds5 recording used a
different lens or zoom than the calibration session. This tool separates them by asking whether the
published coefficients describe the very images they claim to come from.

Both models are scored the same way, on images held out of the fit: intrinsics are fixed, each held-out
board's pose is solved by PnP, and the corner reprojection error is measured. A published file that is
correct for these images will match the recomputed one there. Nothing here is applied to ds5 cam0; that
remains a separate question, because these images only constrain the configuration they were shot in.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import digests, write_json

PATTERNS = ((9, 6), (8, 6), (7, 6), (9, 7), (8, 5), (7, 5), (6, 5), (6, 4), (5, 4))


def detect(path, patterns=PATTERNS):
    """Corners of the largest chessboard the image resolves, refined to sub-pixel accuracy.

    A smaller grid inside a larger board is still a rigid planar target, so images that only resolve a
    sub-grid stay usable; each image carries its own object points.
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    for pattern in patterns:
        ok, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (7, 7), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4))
            grid = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            grid[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
            return pattern, grid, corners.reshape(-1, 2), gray.shape[::-1]
    return None


def held_out_error(K, dist, samples):
    """Reprojection error of fixed intrinsics on boards whose pose is solved independently."""
    import cv2
    import numpy as np
    errors, corners_used = [], 0
    for grid, corners in samples:
        ok, rvec, tvec = cv2.solvePnP(grid, corners.reshape(-1, 1, 2), np.asarray(K, np.float64),
                                      np.asarray(dist, np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        projected, _ = cv2.projectPoints(grid, rvec, tvec, np.asarray(K, np.float64), np.asarray(dist, np.float64))
        errors.append(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1))
        corners_used += len(corners)
    if not errors:
        return None
    stacked = np.concatenate(errors)
    return {"rms_px": float(np.sqrt(stacked.mean())), "max_px": float(np.sqrt(stacked.max())),
            "boards": len(errors), "corners": corners_used}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--published-calibration", type=Path, required=True)
    parser.add_argument("--images-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-images", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    receipt = json.loads(args.images_receipt.read_text())
    published = json.loads(args.published_calibration.read_text())
    paths = sorted(args.images_dir.glob("*.jpg"))
    by_name = {Path(f["path"]).name: f for f in receipt["files"]}
    for path in paths:
        entry = by_name.get(path.name)
        if entry is None or digests(path)[1] != entry["sha256"]:
            raise ValueError("calibration image not covered by the verified receipt: " + path.name)
    import cv2
    import numpy as np
    cv2.setNumThreads(4)
    detections, size = [], None
    for path in paths:
        found = detect(path)
        if found is None:
            continue
        pattern, grid, corners, image_size = found
        if size is None:
            size = image_size
        elif size != image_size:
            raise ValueError("calibration images differ in resolution")
        detections.append({"name": path.name, "pattern": list(pattern), "grid": grid, "corners": corners})
    if len(detections) < args.min_images:
        raise ValueError("too few usable boards: %d" % len(detections))
    if list(size) != list(published["resolution"]):
        raise ValueError("calibration images do not match the published resolution")
    # Deterministic split: alternate images, so both halves span the whole session.
    train = [d for i, d in enumerate(detections) if i % 2 == 0]
    test = [d for i, d in enumerate(detections) if i % 2 == 1]

    def fit(subset):
        rms, K, dist, _, _ = cv2.calibrateCamera([d["grid"] for d in subset], [d["corners"] for d in subset],
                                                 tuple(size), None, None)
        return float(rms), K, dist.ravel()

    train_rms, K_train, dist_train = fit(train)
    all_rms, K_all, dist_all = fit(detections)
    samples = [(d["grid"], d["corners"]) for d in test]
    recomputed = held_out_error(K_train, dist_train, samples)
    published_error = held_out_error(published["K-matrix"], published["distCoeff"], samples)
    ratio = published_error["rms_px"] / recomputed["rms_px"] if recomputed["rms_px"] else float("inf")
    verdict = ("PUBLISHED_FILE_MATCHES_ITS_OWN_IMAGES" if ratio < 1.5
               else "PUBLISHED_FILE_DOES_NOT_DESCRIBE_ITS_OWN_IMAGES")
    result = {"status": verdict, "source_commit": receipt["source_commit"], "camera": receipt["camera"],
              "images_detected": len(detections), "images_total": len(paths),
              "patterns_used": {str(tuple(d["pattern"])): 1 for d in detections} and
                               {str(k): sum(1 for d in detections if tuple(d["pattern"]) == k)
                                for k in {tuple(d["pattern"]) for d in detections}},
              "resolution": list(size),
              "held_out_rms_px": {"recomputed": recomputed, "published": published_error,
                                  "published_over_recomputed": ratio},
              "recomputed_from_half": {"fit_rms_px": train_rms, "K-matrix": K_train.tolist(),
                                       "distCoeff": dist_train.tolist(), "boards": len(train)},
              "recomputed_from_all": {"fit_rms_px": all_rms, "K-matrix": K_all.tolist(),
                                      "distCoeff": dist_all.tolist(), "boards": len(detections)},
              "published": {"K-matrix": published["K-matrix"], "distCoeff": published["distCoeff"]},
              "note": "These images constrain the configuration they were shot in. Agreement here does not"
                      " transfer the calibration to ds5 cam0 by itself.",
              "published_calibration_sha256": digests(args.published_calibration)[1],
              "images_receipt_sha256": digests(args.images_receipt)[1],
              "tool_sha256": digests(Path(__file__))[1],
              "runtime": {"python": platform.python_version(), "executable": sys.executable, "opencv": cv2.__version__}}
    write_json(args.output / "calibration_check.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("recomputed_from_half", "recomputed_from_all")}, indent=2))
    print("recomputed_from_all:", json.dumps(result["recomputed_from_all"], indent=2))


if __name__ == "__main__":
    main()
