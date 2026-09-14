#!/usr/bin/env python3
"""Reset latency of the real non-overlap bar placer (CPU, no Isaac Gym).

Measures AssetManager.reset_idx in footprint_clearance mode for a single env and for the
128-env training batch, at each audited density.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

import audit_navrl_v2_density_geometry as audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=int, nargs="+",
                    default=[64, 70, 100, 130, 160, 190, 205, 220, 250, 300])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--output", type=Path,
                    default=audit.ROOT / "results/navrl_v2_density_geometry_audit_2026-08-27/placer_latency.json")
    args = ap.parse_args()

    pool_sizes, _ = audit.bar_pool_sizes()
    out = {}
    for envs in (1, 128):
        gen = audit.LayoutGenerator(num_envs=envs, pool_sizes=pool_sizes, seed=7)
        for density in args.densities:
            gen.generate(density)  # warm up
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                gen.generate(density)
                samples.append(time.perf_counter() - t0)
            s = np.asarray(samples)
            out.setdefault(str(density), {})["envs_%d" % envs] = {
                "median_s": float(np.median(s)),
                "p95_s": float(np.percentile(s, 95)),
                "max_s": float(s.max()),
            }
            print(f"[latency] envs={envs:3d} D={density:3d} median={np.median(s)*1000:8.1f} ms", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"schema_version": "navrl.v2-placer-latency.v1", "repeats": args.repeats,
         "note": "CPU torch, placement_mode=footprint_clearance, surface_clearance=0.45 m; "
                 "this is the AssetManager placement cost only, not the full task reset",
         "per_density": out}, indent=2, sort_keys=True) + "\n")
    print("[latency] wrote", args.output)


if __name__ == "__main__":
    main()
