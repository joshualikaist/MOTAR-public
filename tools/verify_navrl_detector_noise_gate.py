#!/usr/bin/env python3
"""Pre-arm quality gate for the preregistered detector-coupling bin-bias rerun.

Replays the shipped injection model on the exact analytic-driven range trace used to profile v7.
The gate is unchanged from the first probe: injected pooled range-error std must be within ±10%
of v7's measured 0.705294 m. Bin means are reported as an implementation audit, but they do not
move or add to the preregistered decision boundary.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_profile(raw):
    edges, values = [], []
    for item in raw.split(","):
        hi, value = item.split(":")
        edges.append(float(hi))
        values.append(float(value))
    return torch.tensor(edges), torch.tensor(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--profile-json", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()

    profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    data = np.load(args.npz)
    ranges = torch.from_numpy(data["ref_range"]).float()
    valid = torch.from_numpy(data["both_visible"])
    params = profile["injection_params"]
    sigma_edges, sigma_mults = parse_profile(params["range_sigma_profile"])
    bins = profile["by_range_quartile"]
    bias_raw = ",".join(
        f"{b['range_hi_m']:.9g}:{b['range_err_m']['mean']:.9g}" for b in bins
    )
    bias_edges, bias_values = parse_profile(bias_raw)

    generator = torch.Generator().manual_seed(args.seed)
    ar = torch.zeros(ranges.shape[1])
    rho = float(params["range_rho"])
    pooled_sigma = float(params["range_std_m"])
    injected = []
    for t in range(ranges.shape[0]):
        sigma_idx = torch.bucketize(ranges[t], sigma_edges).clamp(max=len(sigma_mults) - 1)
        bias_idx = torch.bucketize(ranges[t], bias_edges).clamp(max=len(bias_values) - 1)
        white = torch.randn(ranges.shape[1], generator=generator)
        ar = rho * ar + math.sqrt(max(0.0, 1.0 - rho * rho)) * white
        injected.append(ar * pooled_sigma * sigma_mults[sigma_idx] + bias_values[bias_idx])
    injected = torch.stack(injected)
    sample = injected[valid]

    target_std = float(profile["range_err_m"]["std"])
    observed_std = float(sample.std(unbiased=False))
    rel_error = observed_std / target_std - 1.0
    passed = abs(rel_error) <= args.tolerance
    bin_rows = []
    all_idx = torch.bucketize(ranges, bias_edges).clamp(max=len(bias_values) - 1)
    for i, b in enumerate(bins):
        x = injected[valid & (all_idx == i)]
        bin_rows.append({
            "range_lo_m": b["range_lo_m"],
            "range_hi_m": b["range_hi_m"],
            "target_bias_m": b["range_err_m"]["mean"],
            "observed_bias_m": float(x.mean()),
            "target_within_bin_std_m": b["range_err_m"]["std"],
            "observed_within_bin_std_m": float(x.std(unbiased=False)),
            "samples": int(x.numel()),
        })

    out = {
        "gate": "injected pooled range-error std within profiled v7 std ±10%",
        "profile_npz": args.npz,
        "profile_json": args.profile_json,
        "noise_seed": args.seed,
        "target_std_m": target_std,
        "observed_std_m": observed_std,
        "relative_error": rel_error,
        "tolerance": args.tolerance,
        "pass": passed,
        "bias_profile": bias_raw,
        "bins": bin_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
