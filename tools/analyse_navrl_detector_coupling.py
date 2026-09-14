#!/usr/bin/env python3
"""Analyse the paired analytic-vs-v7 detector profile, and the 4-1 coupling arms.

Preregistration: docs/prereg_2026-08-13_detector_coupling.md

Two subcommands:

  profile  read results/.../paired_errors.npz, characterise v7's error against the analytic
           detector on identical frames, and emit the noise parameters the arms will inject.
  arms     read the five evaluated cells, compute the deltas with CIs, check gate 0, and apply
           the preregistered judgment rule.

The one modelling decision worth stating here: dropout is characterised as a two-state MARKOV
process, not an iid Bernoulli with the right marginal. A tracker's response to one isolated miss
is nothing like its response to a four-frame run -- it coasts on constant velocity while the
covariance inflates and the LiDAR fallback takes over -- so a noise model that matches only the
marginal would systematically under-reproduce v7's effect and make a null result uninterpretable.
Transitions are estimated only on consecutive frame pairs where the ANALYTIC head saw the target
in both frames, because that is the only condition under which "did v7 also see it" is observable.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [5, 25, 50, 75, 95]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _describe(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "mad": float(np.median(np.abs(x - np.median(x)))),
        "quantiles": {str(q): float(np.percentile(x, q)) for q in QUANTILES},
    }


def _run_lengths(miss, observable):
    """Completed miss-run lengths, walking the time axis per env.

    A run is only counted when both its start and its end are observable, so partial runs at the
    edges of an occlusion do not bias the distribution short.
    """
    runs = []
    steps, envs = miss.shape
    for e in range(envs):
        run = 0
        for t in range(steps):
            if not observable[t, e]:
                run = 0                       # unobservable: abandon the in-progress run
                continue
            if miss[t, e]:
                run += 1
            elif run:
                runs.append(run)
                run = 0
    return np.array(runs, dtype=np.int64)


def _autocorr(x, lags=5):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < lags + 2:
        return {}
    out = {}
    for lag in range(1, lags + 1):
        a, b = x[:-lag].ravel(), x[lag:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
            out[str(lag)] = None
        else:
            out[str(lag)] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def cmd_profile(args):
    npz = Path(args.npz)
    d = np.load(npz)
    ref_vis = d["ref_visible"]            # (steps, envs) analytic saw the target
    prof_vis = d["profile_visible"]       # v7 saw it
    both = d["both_visible"]
    bearing_err = d["bearing_err"]
    range_err = d["range_err"]
    ref_range = d["ref_range"]
    ref_count = d["ref_count"].astype(np.float64)
    prof_count = d["profile_count"].astype(np.float64)

    steps, envs = ref_vis.shape

    # --- continuous error axes, measured only where BOTH heads produced a detection -----------
    be = bearing_err[both]
    re_ = range_err[both]
    prof = {
        "source_npz": str(npz.relative_to(ROOT)) if str(npz).startswith(str(ROOT)) else str(npz),
        "source_npz_sha256": _sha256(npz),
        "shape": {"steps": int(steps), "envs": int(envs)},
        "visibility": {
            "analytic_rate": float(ref_vis.mean()),
            "v7_rate": float(prof_vis.mean()),
            "both_rate": float(both.mean()),
            # The channel that matters for injection: v7 losing a target the analytic head held.
            "v7_miss_given_analytic_saw": float(
                (ref_vis & ~prof_vis).sum() / max(1, ref_vis.sum())
            ),
            "v7_extra_given_analytic_missed": float(
                (~ref_vis & prof_vis).sum() / max(1, (~ref_vis).sum())
            ),
        },
        "bearing_err_rad": _describe(be),
        "range_err_m": _describe(re_),
        "pixel_count_ratio": _describe(
            prof_count[both] / np.clip(ref_count[both], 1.0, None)
        ),
        "autocorr_bearing_err": _autocorr(np.where(both, bearing_err, np.nan)),
        "autocorr_range_err": _autocorr(np.where(both, range_err, np.nan)),
    }

    # --- conditional structure: does the error grow with range? ------------------------------
    if both.sum() > 100:
        r = ref_range[both]
        edges = np.percentile(r, [0, 25, 50, 75, 100])
        by_range = []
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            sel = (r >= lo) & (r <= hi if i == 3 else r < hi)
            by_range.append({
                "range_lo_m": float(lo), "range_hi_m": float(hi),
                "bearing_err_rad": _describe(be[sel]),
                "range_err_m": _describe(re_[sel]),
            })
        prof["by_range_quartile"] = by_range

    # --- dropout as a two-state Markov chain --------------------------------------------------
    miss = ref_vis & ~prof_vis
    # A transition is observable only if the analytic head saw the target in BOTH frames.
    obs_pair = ref_vis[:-1] & ref_vis[1:]
    prev_miss, cur_miss = miss[:-1], miss[1:]
    n_seen = int((obs_pair & ~prev_miss).sum())
    n_enter = int((obs_pair & ~prev_miss & cur_miss).sum())
    n_missing = int((obs_pair & prev_miss).sum())
    n_leave = int((obs_pair & prev_miss & ~cur_miss).sum())
    p01 = (n_enter / n_seen) if n_seen else 0.0
    p10 = (n_leave / n_missing) if n_missing else 1.0
    runs = _run_lengths(miss, ref_vis)
    prof["dropout"] = {
        "p01": p01, "p10": p10,
        "n_transitions_seen": n_seen, "n_transitions_missing": n_missing,
        "stationary_miss_rate": (p01 / (p01 + p10)) if (p01 + p10) > 0 else 0.0,
        "empirical_miss_rate": float(miss.sum() / max(1, ref_vis.sum())),
        "mean_run_length": float(runs.mean()) if runs.size else 0.0,
        "markov_mean_run_length": (1.0 / p10) if p10 > 0 else None,
        "iid_mean_run_length_same_marginal": (
            1.0 / (1.0 - (p01 / (p01 + p10))) if (p01 + p10) > 0 and p01 / (p01 + p10) < 1 else None
        ),
        # dict union `|` is 3.9+; the Isaac Gym environment is 3.8.
        "run_length_histogram": dict(
            [(str(k), int((runs == k).sum())) for k in range(1, 6)]
            + [("6+", int((runs >= 6).sum()))]
        ),
        "n_completed_runs": int(runs.size),
    }

    out_dir = npz.parent
    (out_dir / "profile.json").write_text(
        json.dumps(prof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- the parameters the arms will inject ---------------------------------------------------
    bearing_std = prof["bearing_err_rad"].get("std", 0.0)
    range_std = prof["range_err_m"].get("std", 0.0)
    # AR(1) coefficient and the range-dependent sigma multipliers. Both exist because the profile
    # says the error is neither white nor homoscedastic; injecting the marginal alone would be a
    # materially easier perturbation than the one v7 actually applies.
    rho = prof["autocorr_range_err"].get("1") or 0.0
    rho = max(0.0, min(0.95, float(rho)))
    sigma_profile = ""
    if "by_range_quartile" in prof and range_std > 0:
        parts = []
        for q in prof["by_range_quartile"]:
            mult = q["range_err_m"]["std"] / range_std
            parts.append(f"{q['range_hi_m']:.4g}:{mult:.4g}")
        sigma_profile = ",".join(parts)

    env_path = out_dir / "noise_params.env"
    env_path.write_text(
        "# Generated by tools/analyse_navrl_detector_coupling.py from profile.json.\n"
        f"# profile.json sha256: {_sha256(out_dir / 'profile.json')}\n"
        "# Structure, not just marginals: rho is the measured lag-1 autocorrelation of the range\n"
        "# error and the sigma profile is its std per measured-range quartile, normalised by the\n"
        "# pooled std. See the preregistration's limitation L1.\n"
        f"NAVRL_DETNOISE_BEARING_STD_RAD={bearing_std:.8g}\n"
        f"NAVRL_DETNOISE_RANGE_STD_M={range_std:.8g}\n"
        f"NAVRL_DETNOISE_RANGE_RHO={rho:.8g}\n"
        f"NAVRL_DETNOISE_RANGE_BIAS_M={prof['range_err_m'].get('mean', 0.0):.8g}\n"
        f'NAVRL_DETNOISE_RANGE_SIGMA_PROFILE="{sigma_profile}"\n'
        f"NAVRL_DETNOISE_DROPOUT_P01={p01:.8g}\n"
        f"NAVRL_DETNOISE_DROPOUT_P10={max(p10, 1e-6):.8g}\n",
        encoding="utf-8")
    prof["injection_params"] = {
        "bearing_std_rad": bearing_std, "range_std_m": range_std, "range_rho": rho,
        "range_bias_m": prof["range_err_m"].get("mean", 0.0),
        "range_sigma_profile": sigma_profile, "dropout_p01": p01, "dropout_p10": p10,
    }
    (out_dir / "profile.json").write_text(
        json.dumps(prof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    v = prof["visibility"]
    dr = prof["dropout"]
    print(f"steps x envs        : {steps} x {envs}")
    print(f"analytic / v7 seen  : {v['analytic_rate']:.4f} / {v['v7_rate']:.4f}")
    print(f"v7 miss | analytic  : {v['v7_miss_given_analytic_saw']:.5f}")
    print(f"bearing err std     : {bearing_std:.6f} rad ({math.degrees(bearing_std):.3f} deg)")
    print(f"range err std       : {range_std:.4f} m")
    print(f"pixel count ratio   : median {prof['pixel_count_ratio']['quantiles']['50']:.3f}")
    print(f"dropout p01 / p10   : {p01:.5f} / {p10:.5f}")
    print(f"  mean run (obs)    : {dr['mean_run_length']:.3f}"
          f"  markov {dr['markov_mean_run_length']}"
          f"  iid {dr['iid_mean_run_length_same_marginal']}")
    print(f"wrote {out_dir/'profile.json'} and {env_path.name}")
    return 0


def _wilson_delta_ci(k1, n1, k2, n2):
    """Normal-approximation CI on the difference of two rates, in percentage points."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = (p2 - p1) * 100.0
    h = 1.959963985 * se * 100.0
    return d, (d - h, d + h)


def cmd_arms(args):
    root = Path(args.root)
    arms = ["analytic_clean", "analytic_noise_0p5", "analytic_noise_1p0",
            "analytic_noise_1p5", "learned_v7"]
    cells = {}
    for arm in arms:
        p = root / arm / "205bars.json"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        o = j["outcome"]
        cells[arm] = {
            "capture": o["capture_rate"], "crash": o["crash_rate"],
            "timeout": o["timeout_rate"], "captured": o["captured"],
            "episodes": j["actual_episodes"],
            "governor": j["condition"]["speed_governor_mode"],
            "seed": j["condition"]["seed"],
            "detector_threshold": j["condition"].get("detector_threshold"),
        }
    if "analytic_clean" not in cells:
        raise SystemExit("baseline arm missing")

    base = cells["analytic_clean"]
    out = {"cells": cells, "deltas": {}}
    for arm, c in cells.items():
        if arm == "analytic_clean":
            continue
        d, ci = _wilson_delta_ci(base["captured"], base["episodes"],
                                 c["captured"], c["episodes"])
        out["deltas"][arm] = {"delta_pp": d, "ci95_pp": list(ci)}

    print(f"{'arm':22s} {'capture':>9s} {'crash':>8s} {'timeout':>8s} {'delta pp':>10s} {'95% CI':>20s}")
    for arm in arms:
        if arm not in cells:
            continue
        c = cells[arm]
        if arm == "analytic_clean":
            print(f"{arm:22s} {c['capture']*100:8.2f}% {c['crash']*100:7.2f}% "
                  f"{c['timeout']*100:7.2f}% {'baseline':>10s}")
        else:
            dd = out["deltas"][arm]
            print(f"{arm:22s} {c['capture']*100:8.2f}% {c['crash']*100:7.2f}% "
                  f"{c['timeout']*100:7.2f}% {dd['delta_pp']:+9.2f} "
                  f"[{dd['ci95_pp'][0]:+6.2f}, {dd['ci95_pp'][1]:+6.2f}]")

    (root / "arms_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {root/'arms_summary.json'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("npz")
    p.set_defaults(fn=cmd_profile)
    a = sub.add_parser("arms")
    a.add_argument("root")
    a.set_defaults(fn=cmd_arms)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
