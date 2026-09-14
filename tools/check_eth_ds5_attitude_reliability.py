"""Is ds5's published attitude authentic, that is, does it carry real frame-specific motion information?

Runs the criterion fixed in results/eth_ds5_e3p_reliability_2026-09-10/PREREGISTRATION.md. It replaces two
earlier forms that failed and that both asked the wrong question: one discarded direction and assumed drag
away, the other required drag to be large.

A multirotor accelerates horizontally by tilting its thrust axis, so a genuine attitude stream predicts a
horizontal acceleration measured independently from position ground truth. The test is whether it beats
200 permutations of the attitude-to-frame assignment, out of sample, at every differentiation window in a
fixed sweep. Sweeping rather than choosing the window is what stops the window becoming a free parameter.

A pass certifies authenticity, not accuracy in degrees.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

from prepare_eth_ds5 import COMMIT, digests, numeric_table, write_json
from check_eth_ds5_attitude_gate import GRAVITY, candidate_axes, local_acceleration

HALF_WINDOWS = (2, 3, 4, 5, 7, 9)
PERMUTATIONS = 200
PERMUTATION_PERCENTILE = 99.0
THRUST_GAIN_RANGE = (0.5, 1.5)
MIN_HELD_OUT_CORRELATION = 0.35
MIN_ACCEL_MS2 = 0.5
MIN_TILT_DEG = 3.0
CONVENTION = {"order": "xyz", "transpose": False, "to_enu": True, "body_sign": -1}


def design(axes, tilt_deg, velocity):
    """Thrust and drag regressors, stacked over both horizontal components."""
    import numpy as np
    horizontal = axes[:, :2]
    norms = np.linalg.norm(horizontal, axis=1, keepdims=True)
    direction = np.divide(horizontal, norms, out=np.zeros_like(horizontal), where=norms > 1e-9)
    thrust = (GRAVITY * np.tan(np.radians(tilt_deg)))[:, None] * direction
    speed = np.linalg.norm(velocity, axis=1)[:, None]
    drag = speed * velocity[:, :2]
    return np.stack([thrust.reshape(-1), drag.reshape(-1)], axis=1)


def fit_and_score(regressors_fit, target_fit, regressors_score, target_score):
    """Least squares on one split, correlation and gains scored on the other."""
    import numpy as np
    coefficients, *_ = np.linalg.lstsq(regressors_fit, target_fit, rcond=None)
    predicted = regressors_score @ coefficients
    if np.std(predicted) < 1e-12 or np.std(target_score) < 1e-12:
        return {"thrust_gain": float(coefficients[0]), "drag_coefficient": float(coefficients[1]),
                "held_out_correlation": 0.0}
    return {"thrust_gain": float(coefficients[0]), "drag_coefficient": float(coefficients[1]),
            "held_out_correlation": float(np.corrcoef(target_score, predicted)[0, 1])}


def evaluate_window(times, positions, rpy, half, permutations=PERMUTATIONS, seed=0):
    """One differentiation window: both split directions, plus the permutation null."""
    import numpy as np
    acceleration = local_acceleration(times, positions, half=half)
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(rpy[:, 0])) * np.cos(np.radians(rpy[:, 1])), -1.0, 1.0)))
    velocity = np.gradient(positions, times, axis=0)
    usable = (~np.isnan(acceleration[:, 0]) & (np.linalg.norm(acceleration[:, :2], axis=1) >= MIN_ACCEL_MS2)
              & (tilt >= MIN_TILT_DEG))
    if usable.sum() < 60:
        return {"status": "INSUFFICIENT_SAMPLES", "samples": int(usable.sum())}
    index = np.nonzero(usable)[0]
    axes = candidate_axes(rpy, CONVENTION["order"], CONVENTION["transpose"], CONVENTION["to_enu"],
                          CONVENTION["body_sign"])
    target = acceleration[index][:, :2].reshape(-1)
    regressors = design(axes[index], tilt[index], velocity[index])
    midpoint = np.median(times[index])
    early = times[index] < midpoint
    masks = {"early_to_late": (early, ~early), "late_to_early": (~early, early)}
    rng = np.random.default_rng(seed)
    out = {"samples": int(usable.sum()), "directions": {}}
    for name, (fit_mask, score_mask) in masks.items():
        rows_fit = np.repeat(fit_mask, 2)
        rows_score = np.repeat(score_mask, 2)
        real = fit_and_score(regressors[rows_fit], target[rows_fit], regressors[rows_score], target[rows_score])
        null = []
        for _ in range(permutations):
            order = rng.permutation(len(index))
            shuffled = design(axes[index][order], tilt[index][order], velocity[index])
            null.append(fit_and_score(shuffled[rows_fit], target[rows_fit],
                                      shuffled[rows_score], target[rows_score])["held_out_correlation"])
            
        threshold = float(np.percentile(null, PERMUTATION_PERCENTILE))
        real.update({"null_percentile_%g" % PERMUTATION_PERCENTILE: threshold,
                     "null_max": float(np.max(null)), "null_median": float(np.median(null)),
                     "beats_null": real["held_out_correlation"] > threshold,
                     "gain_in_range": THRUST_GAIN_RANGE[0] <= real["thrust_gain"] <= THRUST_GAIN_RANGE[1],
                     "drag_negative": real["drag_coefficient"] < 0,
                     "correlation_sufficient": real["held_out_correlation"] >= MIN_HELD_OUT_CORRELATION})
        real["pass"] = bool(real["beats_null"] and real["gain_in_range"] and real["drag_negative"]
                            and real["correlation_sufficient"])
        out["directions"][name] = real
    out["pass"] = all(d["pass"] for d in out["directions"].values())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output")
    import numpy as np
    pose = np.asarray(numeric_table(args.dataset / "dataset5/pose/fused_pose.txt", 11))
    times, positions, rpy = pose[:, 0], pose[:, 1:4], pose[:, 4:7]
    windows = {str(h): evaluate_window(times, positions, rpy, h, args.permutations, args.seed)
               for h in HALF_WINDOWS}
    failed = [h for h, w in windows.items() if not w.get("pass")]
    receipt = {"status": "ATTITUDE_RELIABLE" if not failed else "ATTITUDE_NOT_RELIABLE",
               "failed_windows": failed, "source_commit": COMMIT,
               "preregistration": "results/eth_ds5_e3p_reliability_2026-09-10/PREREGISTRATION.md",
               "convention": CONVENTION, "half_windows": list(HALF_WINDOWS),
               "permutations": args.permutations, "seed": args.seed,
               "thresholds": {"thrust_gain_range": list(THRUST_GAIN_RANGE),
                              "min_held_out_correlation": MIN_HELD_OUT_CORRELATION,
                              "null_percentile": PERMUTATION_PERCENTILE},
               "certifies": "authenticity of the attitude stream, not its accuracy in degrees",
               "windows": windows, "tool_sha256": digests(Path(__file__))[1],
               "runtime": {"python": platform.python_version(), "executable": sys.executable}}
    write_json(args.output / "attitude_reliability.json", receipt)
    print(json.dumps({"status": receipt["status"], "failed_windows": failed,
                      "summary": {h: {d: {k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in val.items()
                                          if k in ("held_out_correlation", "thrust_gain", "drag_coefficient",
                                                   "null_percentile_99", "pass")}
                                      for d, val in w.get("directions", {}).items()}
                                  for h, w in windows.items()}}, indent=2))


if __name__ == "__main__":
    main()
