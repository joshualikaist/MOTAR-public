#!/usr/bin/env python3
"""CPU reset/replan latency benchmark for the opt-in physical-target route planner."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUTE = load_module(
    "navrl_target_route_benchmark_standalone",
    ROOT / "aerial_gym/task/navrl_task/target_route_planner.py",
)
GEOMETRY = load_module(
    "navrl_density_geometry_benchmark_standalone",
    ROOT / "tools/analyze_navrl_density_feasibility.py",
)


def safe_start(rng, bars, support, tracking, boundary):
    lo, hi = boundary + support, 40.0 - boundary - support
    for _ in range(10000):
        point = np.array(
            [rng.uniform(float(lo[0]), float(hi[0])), rng.uniform(float(lo[1]), float(hi[1]))]
        )
        valid = True
        for x, y, width, depth in bars:
            delta = np.maximum(
                np.abs(point - np.array([x, y]))
                - np.array([0.5 * width, 0.5 * depth])
                - support
                - tracking,
                0.0,
            )
            if np.linalg.norm(delta) <= 1e-9:
                valid = False
                break
        if valid:
            return point
    raise RuntimeError("could not sample a route-safe target start")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--densities", nargs="+", type=int, default=[70, 150, 205, 300])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=825)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    sizes = GEOMETRY.load_bar_sizes(ROOT, "bars_h3")
    config = ROUTE.RoutePlannerConfig(
        resolution_m=0.25,
        tracking_margin_m=0.45,
        boundary_margin_m=1.25,
        max_expansions=50000,
        max_waypoints=128,
        replan_cooldown_steps=10,
        goal_tolerance_m=0.05,
    )
    planner = ROUTE.DeterministicAStarRoutePlanner(config)
    support = ROUTE.conservative_xy_support_from_box((0.28, 0.28, 0.12))
    rows = []
    for density in args.densities:
        elapsed_ms, expanded, statuses = [], [], {}
        for trial in range(args.trials):
            rng = random.Random(args.seed + density * 100003 + trial)
            bars, _, _, _ = GEOMETRY.place_bars_navrl_band(
                density, rng, sizes, (0.0, 40.0, 0.0, 40.0), touch=0.4, gap=1.6
            )
            start = safe_start(rng, bars, support, config.tracking_margin_m, config.boundary_margin_m)
            centers = np.array([[bar[0], bar[1]] for bar in bars], dtype=np.float64)
            half = 0.5 * np.array([[bar[2], bar[3]] for bar in bars], dtype=np.float64)
            begin = time.perf_counter()
            result = planner.plan_to_connected_goal(
                start, centers, half, np.array([0.0, 0.0]), np.array([40.0, 40.0]),
                support, min_goal_distance_m=6.0, selector=rng.random(),
            )
            elapsed_ms.append(1000.0 * (time.perf_counter() - begin))
            expanded.append(result.expanded_nodes)
            statuses[result.status] = statuses.get(result.status, 0) + 1
        values = np.asarray(elapsed_ms)
        row = {
            "bars": density,
            "trials": args.trials,
            "mean_ms_per_env": float(values.mean()),
            "p50_ms_per_env": float(np.quantile(values, 0.50)),
            "p95_ms_per_env": float(np.quantile(values, 0.95)),
            "max_ms_per_env": float(values.max()),
            "mean_expanded_nodes": float(np.mean(expanded)),
            "status_counts": statuses,
        }
        row["projected_serial_128_env_s"] = 128.0 * row["mean_ms_per_env"] / 1000.0
        row["latency_gate_pass"] = bool(
            row["mean_ms_per_env"] <= 100.0
            and row["p95_ms_per_env"] <= 150.0
            and row["projected_serial_128_env_s"] <= 10.0
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    payload = {
        "schema": "navrl_target_route_cpu_benchmark_v1",
        "seed": args.seed,
        "densities": args.densities,
        "contract": {
            "placement": "navrl_band touch=0.4 gap=1.6",
            "bar_pool": "bars_h3 actual per-asset AABB",
            "route_resolution_m": config.resolution_m,
            "target_support_xy_m": support.tolist(),
            "tracking_margin_m": config.tracking_margin_m,
            "boundary_margin_m": config.boundary_margin_m,
            "connected_goal_min_distance_m": 6.0,
            "device": "CPU",
            "scope": "sequential per-env reset/replan cost; no PhysX/GPU/policy",
        },
        "rows": rows,
        "gate": {
            "mean_ms_per_env_max": 100.0,
            "p95_ms_per_env_max": 150.0,
            "projected_serial_128_env_s_max": 10.0,
            "availability_at_70_min": 0.90,
            "availability_at_70": (
                rows[0]["status_counts"].get("ok", 0) / rows[0]["trials"]
                if rows and rows[0]["bars"] == 70 else None
            ),
            "verdict": (
                "PASS_CPU_ENGINEERING_GATE"
                if all(row["latency_gate_pass"] for row in rows)
                and rows and rows[0]["bars"] == 70
                and rows[0]["status_counts"].get("ok", 0) / rows[0]["trials"] >= 0.90
                else "BLOCKED_PERFORMANCE"
            ),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
