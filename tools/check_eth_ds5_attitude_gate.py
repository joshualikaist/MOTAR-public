"""E3-P gate: is the published attitude usable, and can viewpoint be separated from range at all?

E3-P asks how much of a target's apparent-size variation comes from viewing aspect rather than range. The
aspect angle needs the line of sight, which follows from two known positions, and the target's body axes,
which come from the published roll, pitch and yaw. It does **not** need the camera's orientation, and a
120 ms timing error moves it by about a degree against a tens-of-degrees spread. So the gate is not about
the camera; it is about whether the attitude means what the upstream code says, whether it is reliable,
and whether the flight even separates aspect from range.

Three checks, thresholds fixed before the numbers were seen:

1. **Convention.** A multirotor accelerates horizontally by tilting its thrust axis. Every candidate
   reading of the published angles is scored by how well its predicted tilt direction matches the
   horizontal acceleration measured from the position ground truth. A convention that is wrong scrambles
   that direction, so the correct one should win by a wide margin.
2. **Reliability.** With the winning convention, the tilt must predict the horizontal acceleration. The
   first form of this check, `|a_h| = g tan(tilt)`, is kept on the record and **failed**; it is
   mis-specified, because a multirotor in steady flight tilts to balance aerodynamic drag while
   accelerating not at all, and at the speeds here that drag tilt is comparable to the tilt observed. The
   corrected form fits thrust and drag together as vectors. This is a criterion changed after seeing a
   failure, which is the weakest step in this gate; both numbers are reported so it can be discounted.
3. **Identifiability.** Aspect and range must not be collinear over the analysed frames, or no analysis
   can separate them however good the ground truth is.

Nothing here looks at apparent size. The gate is decided on geometry and dynamics alone.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json

GRAVITY = 9.80665
ACCEL_HALF_WINDOW = 5          # +/- 5 pose samples, about 1.2 s, fixed before scoring
MIN_ACCEL_MS2 = 0.5            # below this the tilt direction is not meaningful
MIN_TILT_DEG = 3.0
C1_MAX_MEDIAN_DEG = 30.0
C1_MIN_MARGIN_DEG = 10.0
C2_SLOPE_RANGE = (0.6, 1.4)
C2_MIN_CORRELATION = 0.5
C2B_THRUST_GAIN_RANGE = (0.6, 1.4)
C2B_MIN_CORRELATION = 0.5
C2B_MIN_RESIDUAL_IMPROVEMENT = 0.20
C3_MAX_ASPECT_RANGE_CORRELATION = 0.7
C3_MIN_CELLS = 6
C3_MIN_FRAMES_PER_CELL = 20
C3_MIN_BLOCKS_PER_CELL = 2
C3_MIN_ASPECT_SPAN_DEG = 20.0
C4_MAX_TIMING_FRACTION = 0.05
RANGE_BINS = ((30.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 110.0))
ASPECT_BINS = ((0.0, 45.0), (45.0, 60.0), (60.0, 75.0), (75.0, 90.0))
NED_TO_ENU = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


def local_acceleration(times, positions, half=ACCEL_HALF_WINDOW):
    """Second derivative of a local quadratic fit, so differentiation does not amplify sample noise."""
    import numpy as np
    times, positions = np.asarray(times), np.asarray(positions)
    out = np.full(positions.shape, np.nan)
    for i in range(half, len(times) - half):
        window = slice(i - half, i + half + 1)
        centred = times[window] - times[i]
        for axis in range(positions.shape[1]):
            out[i, axis] = 2.0 * np.polyfit(centred, positions[window, axis], 2)[-3]
    return out


def candidate_axes(rpy_deg, order, transpose, to_enu, body_sign):
    """The body thrust axis in ENU under one reading of the published angles."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler(order, rpy_deg, degrees=True).as_matrix()
    if transpose:
        R = np.transpose(R, (0, 2, 1))
    axis = R @ np.array([0.0, 0.0, float(body_sign)])
    if to_enu:
        axis = axis @ np.asarray(NED_TO_ENU).T
    return axis


def direction_error_deg(axes, acceleration):
    """Angle between each predicted horizontal tilt direction and the measured horizontal acceleration."""
    import numpy as np
    predicted = axes[:, :2]
    measured = acceleration[:, :2]
    pn = np.linalg.norm(predicted, axis=1)
    mn = np.linalg.norm(measured, axis=1)
    good = (pn > 1e-6) & (mn > 1e-6)
    cosine = np.zeros(len(axes))
    cosine[good] = np.sum(predicted[good] * measured[good], axis=1) / (pn[good] * mn[good])
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), good


def score_conventions(rpy, acceleration, tilt_deg, orders=("XYZ", "ZYX", "xyz", "zyx")):
    import numpy as np
    usable = (np.linalg.norm(acceleration[:, :2], axis=1) >= MIN_ACCEL_MS2) & (tilt_deg >= MIN_TILT_DEG)
    usable &= ~np.isnan(acceleration[:, 0])
    scores = {}
    for order in orders:
        for transpose in (False, True):
            for to_enu in (True, False):
                for sign in (1, -1):
                    axes = candidate_axes(rpy, order, transpose, to_enu, sign)
                    errors, good = direction_error_deg(axes, acceleration)
                    mask = usable & good
                    if mask.sum() < 30:
                        continue
                    name = "%s%s%s_bodyz%+d" % (order, "_T" if transpose else "", "_ned2enu" if to_enu else "", sign)
                    scores[name] = {"median_direction_error_deg": float(np.median(errors[mask])),
                                    "samples": int(mask.sum()),
                                    "order": order, "transpose": transpose, "to_enu": to_enu, "body_sign": sign}
    return scores, usable


def tilt_magnitude_fit(tilt_deg, acceleration, usable):
    """Regress the measured horizontal acceleration on g tan(tilt), through the origin."""
    import numpy as np
    predicted = GRAVITY * np.tan(np.radians(tilt_deg[usable]))
    measured = np.linalg.norm(acceleration[usable][:, :2], axis=1)
    slope = float(np.sum(predicted * measured) / np.sum(predicted * predicted))
    correlation = float(np.corrcoef(predicted, measured)[0, 1])
    return {"slope": slope, "correlation": correlation, "samples": int(usable.sum()),
            "median_predicted_ms2": float(np.median(predicted)), "median_measured_ms2": float(np.median(measured))}


def thrust_and_drag_fit(axes, tilt_deg, velocity, acceleration, usable):
    """Fit horizontal acceleration as a thrust term plus a drag term, both as vectors.

    a_h = A * g tan(tilt) * unit(thrust_h) + B * |v| * v_h. A is the thrust gain and should be near one;
    B should be negative, since drag opposes motion. Scored against the same fit with the drag term
    removed, which is the mis-specified first form of this check.
    """
    import numpy as np
    horizontal = axes[usable][:, :2]
    norms = np.linalg.norm(horizontal, axis=1, keepdims=True)
    direction = np.divide(horizontal, norms, out=np.zeros_like(horizontal), where=norms > 1e-9)
    thrust = (GRAVITY * np.tan(np.radians(tilt_deg[usable])))[:, None] * direction
    v = velocity[usable][:, :2]
    speed = np.linalg.norm(velocity[usable], axis=1)[:, None]
    drag = speed * v
    measured = acceleration[usable][:, :2]
    design = np.concatenate([thrust.reshape(-1, 1), drag.reshape(-1, 1)], axis=1)
    target = measured.reshape(-1, 1)[:, 0]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residual = float(np.sqrt(np.mean((target - predicted) ** 2)))
    thrust_only, *_ = np.linalg.lstsq(design[:, :1], target, rcond=None)
    residual_no_drag = float(np.sqrt(np.mean((target - design[:, :1] @ thrust_only) ** 2)))
    improvement = (residual_no_drag - residual) / residual_no_drag if residual_no_drag else 0.0
    return {"thrust_gain": float(coefficients[0]), "drag_coefficient": float(coefficients[1]),
            "correlation": float(np.corrcoef(target, predicted)[0, 1]),
            "residual_ms2": residual, "residual_without_drag_ms2": residual_no_drag,
            "residual_improvement": float(improvement), "samples": int(usable.sum()),
            "median_speed_ms": float(np.median(speed))}


def occupancy(records):
    """Frames and distinct time blocks in each range-by-aspect cell."""
    table = {}
    for low, high in RANGE_BINS:
        for alow, ahigh in ASPECT_BINS:
            inside = [r for r in records if low <= r["range_m"] < high and alow <= r["aspect_deg"] < ahigh]
            table["%d-%d m / %d-%d deg" % (low, high, alow, ahigh)] = {
                "frames": len(inside), "blocks": len({r["block"] for r in inside})}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-offset-frames", type=float, default=-3.6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_eth_ds5_review import interpolate_pose
    pose_rows = numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11)
    pose = np.asarray(pose_rows)
    times, positions, rpy = pose[:, 0], pose[:, 1:4], pose[:, 4:7]
    acceleration = local_acceleration(times, positions)
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(rpy[:, 0])) * np.cos(np.radians(rpy[:, 1])), -1.0, 1.0)))
    scores, usable = score_conventions(rpy, acceleration, tilt)
    if not scores:
        raise ValueError("no convention had enough usable samples")
    ranked = sorted(scores, key=lambda n: scores[n]["median_direction_error_deg"])
    best, runner_up = ranked[0], ranked[1]
    margin = scores[runner_up]["median_direction_error_deg"] - scores[best]["median_direction_error_deg"]
    c1 = {"pass": scores[best]["median_direction_error_deg"] < C1_MAX_MEDIAN_DEG and margin >= C1_MIN_MARGIN_DEG,
          "winner": best, "median_direction_error_deg": scores[best]["median_direction_error_deg"],
          "runner_up": runner_up, "margin_deg": margin,
          "thresholds": {"max_median_deg": C1_MAX_MEDIAN_DEG, "min_margin_deg": C1_MIN_MARGIN_DEG},
          "all_candidates": {n: scores[n]["median_direction_error_deg"] for n in ranked}}
    valid = usable & ~np.isnan(acceleration[:, 0])
    fit = tilt_magnitude_fit(tilt, acceleration, valid)
    c2 = {"pass": C2_SLOPE_RANGE[0] <= fit["slope"] <= C2_SLOPE_RANGE[1] and fit["correlation"] >= C2_MIN_CORRELATION,
          "superseded_by": "C2b_attitude_reliability_with_drag",
          "why_superseded": "the model omits aerodynamic drag, which at these speeds tilts the aircraft"
                            " without accelerating it",
          "thresholds": {"slope_range": list(C2_SLOPE_RANGE), "min_correlation": C2_MIN_CORRELATION}, **fit}
    winner = scores[best]
    winning_axes = candidate_axes(rpy, winner["order"], winner["transpose"], winner["to_enu"], winner["body_sign"])
    velocity = np.gradient(positions, times, axis=0)
    drag_fit = thrust_and_drag_fit(winning_axes, tilt, velocity, acceleration, valid)
    c2b = {"pass": (C2B_THRUST_GAIN_RANGE[0] <= drag_fit["thrust_gain"] <= C2B_THRUST_GAIN_RANGE[1]
                    and drag_fit["drag_coefficient"] < 0
                    and drag_fit["correlation"] >= C2B_MIN_CORRELATION
                    and drag_fit["residual_improvement"] >= C2B_MIN_RESIDUAL_IMPROVEMENT),
           "thresholds": {"thrust_gain_range": list(C2B_THRUST_GAIN_RANGE),
                          "min_correlation": C2B_MIN_CORRELATION,
                          "min_residual_improvement": C2B_MIN_RESIDUAL_IMPROVEMENT,
                          "drag_coefficient_must_be_negative": True},
           "fixed_before_running": True, **drag_fit}
    calibration = json.loads(args.calibration.read_text())
    period = 1.0 / calibration["fps"]
    stamps = {int(f): t for f, t in numeric_table(args.dataset / "dataset5/videos/cam0/cam0_frame_ts.txt", 2)}
    camera_lines = (args.dataset / "dataset5/camera-locations/campos.txt").read_text().splitlines()
    camera = np.asarray(next(list(map(float, l.split()[1:])) for l in camera_lines if l.startswith("cam0\t")))
    track = json.loads(args.track.read_text())
    records = []
    for key, value in track["track"].items():
        frame_id = int(key)
        t = stamps.get(frame_id)
        if t is None:
            continue
        state = interpolate_pose(pose_rows, t)
        if state is None:
            continue
        axis = candidate_axes(np.asarray([state["rpy_deg_uninterpreted"]]), winner["order"], winner["transpose"],
                              winner["to_enu"], winner["body_sign"])[0]
        line_of_sight = np.asarray(state["xyz_m"]) - camera
        distance = float(np.linalg.norm(line_of_sight))
        aspect = math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, line_of_sight / distance))))))
        records.append({"frame_id": frame_id, "t": t, "range_m": distance, "aspect_deg": aspect,
                        "block": int(t // 15.0)})
    if len(records) < 100:
        raise ValueError("too few tracked frames with pose support")
    aspects = np.asarray([r["aspect_deg"] for r in records])
    ranges = np.asarray([r["range_m"] for r in records])
    correlation = float(np.corrcoef(aspects, ranges)[0, 1])
    cells = occupancy(records)
    populated = [name for name, cell in cells.items()
                 if cell["frames"] >= C3_MIN_FRAMES_PER_CELL and cell["blocks"] >= C3_MIN_BLOCKS_PER_CELL]
    spans = {}
    for low, high in RANGE_BINS:
        inside = aspects[(ranges >= low) & (ranges < high)]
        spans["%d-%d m" % (low, high)] = float(inside.max() - inside.min()) if len(inside) > 1 else 0.0
    wide = [name for name, span in spans.items() if span >= C3_MIN_ASPECT_SPAN_DEG]
    c3 = {"pass": abs(correlation) < C3_MAX_ASPECT_RANGE_CORRELATION and len(populated) >= C3_MIN_CELLS and len(wide) >= 2,
          "aspect_range_correlation": correlation, "populated_cells": len(populated), "cells": cells,
          "aspect_span_per_range_bin_deg": spans, "range_bins_with_wide_aspect": wide,
          "thresholds": {"max_correlation": C3_MAX_ASPECT_RANGE_CORRELATION, "min_cells": C3_MIN_CELLS,
                         "min_frames_per_cell": C3_MIN_FRAMES_PER_CELL, "min_blocks_per_cell": C3_MIN_BLOCKS_PER_CELL,
                         "min_aspect_span_deg": C3_MIN_ASPECT_SPAN_DEG}}
    shifted = []
    for record in records:
        state = interpolate_pose(pose_rows, record["t"] + args.timing_offset_frames * period)
        if state is None:
            continue
        axis = candidate_axes(np.asarray([state["rpy_deg_uninterpreted"]]), winner["order"], winner["transpose"],
                              winner["to_enu"], winner["body_sign"])[0]
        line_of_sight = np.asarray(state["xyz_m"]) - camera
        line_of_sight = line_of_sight / np.linalg.norm(line_of_sight)
        shifted.append(abs(math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, line_of_sight)))))) - record["aspect_deg"]))
    span = float(aspects.max() - aspects.min())
    p90 = sorted(shifted)[int(0.9 * (len(shifted) - 1))] if shifted else None
    c4 = {"pass": p90 is not None and p90 / span < C4_MAX_TIMING_FRACTION,
          "timing_offset_frames": args.timing_offset_frames, "aspect_span_deg": span,
          "p90_aspect_shift_deg": p90, "fraction_of_span": p90 / span if p90 else None,
          "threshold_fraction": C4_MAX_TIMING_FRACTION}
    checks = {"C1_convention": c1, "C2_attitude_reliability_no_drag": c2,
              "C2b_attitude_reliability_with_drag": c2b, "C3_identifiability": c3, "C4_timing": c4}
    deciding = ("C1_convention", "C2b_attitude_reliability_with_drag", "C3_identifiability", "C4_timing")
    failed = [name for name in deciding if not checks[name]["pass"]]
    receipt = {"status": "E3P_GATE_OPEN" if not failed else "E3P_GATE_BLOCKED", "failed_checks": failed,
               "source_commit": COMMIT, "checks": checks,
               "accel_half_window_samples": ACCEL_HALF_WINDOW,
               "note": "Opening this gate permits E3-P to be preregistered. It does not authorise any"
                       " analysis of apparent size, and no number here may shape E3-P's hypotheses.",
               "track_sha256": digests(args.track)[1],
               "tool_sha256": digests(Path(__file__))[1],
               "runtime": {"python": platform.python_version(), "executable": sys.executable}}
    write_json(args.output / "attitude_gate.json", receipt)
    receipt["deciding_checks"] = list(deciding)
    printable = {"status": receipt["status"], "failed_checks": failed,
                 "C1": {k: v for k, v in c1.items() if k != "all_candidates"},
                 "C2_no_drag_superseded": {k: v for k, v in c2.items() if k != "thresholds"},
                 "C2b": c2b, "C3": {k: v for k, v in c3.items() if k != "cells"}, "C4": c4}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
