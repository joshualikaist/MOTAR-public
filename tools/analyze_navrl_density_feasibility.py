#!/usr/bin/env python3
"""Monte-Carlo free-space audit for legacy and v2 NavRL bar fields.

This deliberately ignores the learned policy.  It reproduces the environment's XY placement
rule, inflates every bar by the drone footprint plus a requested clearance, and asks whether the
resulting free-space topology is connected.  It therefore separates a physical/layout ceiling
from a perception, control, curriculum, or PPO ceiling.

Defaults match the v2 recovery contract: 40 x 40 m, full-width ``navrl_band`` placement,
touch/gap=0.4/1.6 m, full-height ``bars_h3`` assets, and densities through 300 bars.  Pass
``--preset legacy-v1`` only when reproducing the old 24 m random+relax results.
"""

import argparse
from collections import deque
import csv
from pathlib import Path
import random
import re

import numpy as np


DRONE_HALF = 0.28 * 0.5


def load_bar_sizes(root, pool):
    sizes = []
    pattern = re.compile(r'<box size="([0-9.]+) ([0-9.]+) ([0-9.]+)"')
    for path in sorted(
        (root / "resources/models/environment_assets" / pool).glob("*.urdf")
    ):
        match = pattern.search(path.read_text(encoding="utf-8"))
        if match:
            sizes.append((float(match.group(1)), float(match.group(2))))
    if not sizes:
        raise RuntimeError(f"bar URDF sizes not found for pool {pool!r}")
    return sizes


def _uniform_xy(rng, band):
    x0, x1, y0, y1 = band
    return rng.uniform(x0, x1), rng.uniform(y0, y1)


def place_bars_legacy(count, rng, sizes, band, *, batch=32, attempts_before_relax=128):
    """Mirror AssetManager's legacy random placement with spacing relaxation."""
    placed = []
    min_spacing = 1.5
    attempts = 0
    relaxations = 0
    for _ in range(count):
        while True:
            candidates = [_uniform_xy(rng, band) for _ in range(batch)]
            valid = [
                all(
                    (x - px) ** 2 + (y - py) ** 2 >= min_spacing**2
                    for px, py, _, _ in placed
                )
                for x, y in candidates
            ]
            if any(valid):
                idx = valid.index(True)
                w, d = rng.choice(sizes)
                placed.append((candidates[idx][0], candidates[idx][1], w, d))
                break
            attempts += batch
            if attempts >= attempts_before_relax:
                min_spacing *= 0.8
                attempts = 0
                relaxations += 1
    return placed, min_spacing, relaxations, 0


def place_bars_navrl_band(
    count,
    rng,
    sizes,
    band,
    *,
    touch=0.4,
    gap=1.6,
    batch=32,
    attempts_before_merge=1280,
):
    """Mirror ``AssetManager._navrl_band_xy_spacing`` including merge fallback."""
    if count <= 0:
        return [], gap, 0, 0
    x, y = _uniform_xy(rng, band)
    w, d = rng.choice(sizes)
    placed = [(x, y, w, d)]
    merges = 0
    for _ in range(1, count):
        attempts = 0
        while True:
            candidates = [_uniform_xy(rng, band) for _ in range(batch)]
            valid = []
            for cx, cy in candidates:
                in_forbidden_band = any(
                    touch < ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < gap
                    for px, py, _, _ in placed
                )
                valid.append(not in_forbidden_band)
            if any(valid):
                cx, cy = candidates[valid.index(True)]
                w, d = rng.choice(sizes)
                placed.append((cx, cy, w, d))
                break
            attempts += batch
            if attempts >= attempts_before_merge:
                px, py, _, _ = rng.choice(placed)
                angle = 2.0 * np.pi * rng.random()
                radius = 0.5 * touch * rng.random()
                x0, x1, y0, y1 = band
                cx = min(x1, max(x0, px + radius * np.cos(angle)))
                cy = min(y1, max(y0, py + radius * np.sin(angle)))
                w, d = rng.choice(sizes)
                placed.append((cx, cy, w, d))
                merges += 1
                break
    return placed, gap, 0, merges


def occupancy_grid(bars, arena, resolution, side_clearance):
    n = int(round(arena / resolution)) + 1
    occupied = np.zeros((n, n), dtype=bool)
    inflation = DRONE_HALF + side_clearance
    for x, y, w, d in bars:
        xa = max(0, int(np.floor((x - w * 0.5 - inflation) / resolution)))
        xb = min(n - 1, int(np.ceil((x + w * 0.5 + inflation) / resolution)))
        ya = max(0, int(np.floor((y - d * 0.5 - inflation) / resolution)))
        yb = min(n - 1, int(np.ceil((y + d * 0.5 + inflation) / resolution)))
        occupied[xa : xb + 1, ya : yb + 1] = True
    return occupied


def _components(free):
    labels = np.full(free.shape, -1, dtype=np.int32)
    sizes = []
    neighbors = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    for sx, sy in zip(*np.nonzero(free & (labels < 0))):
        # np.nonzero is materialized before the flood fills below, so cells reached by an earlier
        # component still appear in this iterator.  Skip them instead of relabelling each cell as
        # a new singleton component.
        if labels[sx, sy] >= 0:
            continue
        label = len(sizes)
        queue = deque([(int(sx), int(sy))])
        labels[sx, sy] = label
        size = 0
        while queue:
            x, y = queue.popleft()
            size += 1
            for dx, dy in neighbors:
                xx, yy = x + dx, y + dy
                if (
                    0 <= xx < free.shape[0]
                    and 0 <= yy < free.shape[1]
                    and free[xx, yy]
                    and labels[xx, yy] < 0
                ):
                    labels[xx, yy] = label
                    queue.append((xx, yy))
        sizes.append(size)
    return labels, np.asarray(sizes, dtype=np.int64)


def topology_metrics(bars, arena, resolution, side_clearance):
    occupied = occupancy_grid(bars, arena, resolution, side_clearance)
    free = ~occupied
    labels, sizes = _components(free)
    total = int(sizes.sum())
    if total == 0:
        return False, 0.0, 0.0, 0.0

    # Boundary crossing is a conservative legacy-compatible diagnostic.  General-spawn v2 is
    # better represented by largest-component coverage and random-pair connectivity below.
    start_x = min(free.shape[0] - 1, int(round(0.5 / resolution)))
    goal_x = max(0, int(round((arena - 0.5) / resolution)))
    start_labels = set(int(v) for v in labels[start_x, free[start_x]] if v >= 0)
    goal_labels = set(int(v) for v in labels[goal_x, free[goal_x]] if v >= 0)
    crossing = bool(start_labels & goal_labels)

    largest_fraction = float(sizes.max()) / total
    # Probability that two uniformly sampled free cells lie in the same component.
    pair_connectivity = float(np.sum(sizes.astype(np.float64) ** 2)) / float(total**2)
    free_fraction = float(free.mean())
    return crossing, largest_fraction, pair_connectivity, free_fraction


def _preset(args):
    if args.preset == "legacy-v1":
        args.arena = 24.0
        args.band_x = (0.13, 0.96)
        args.placement = "random"
        args.bar_pool = "bars"
        if args.densities is None:
            args.densities = [75, 110, 130, 150]
    else:
        if args.densities is None:
            args.densities = [130, 160, 190, 205, 220, 250, 300]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("v2", "legacy-v1"), default="v2")
    parser.add_argument("--densities", nargs="+", type=int)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arena", type=float, default=40.0)
    parser.add_argument("--band-x", nargs=2, type=float, default=(0.0, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--placement", choices=("navrl_band", "random"), default="navrl_band")
    parser.add_argument("--touch", type=float, default=0.4)
    parser.add_argument("--gap", type=float, default=1.6)
    parser.add_argument("--bar-pool", default="bars_h3")
    parser.add_argument("--csv-out", type=Path)
    args = _preset(parser.parse_args())
    if not 0.0 <= args.band_x[0] < args.band_x[1] <= 1.0:
        raise ValueError("--band-x ratios must satisfy 0 <= MIN < MAX <= 1")
    if args.trials <= 0 or args.resolution <= 0.0 or args.arena <= 0.0:
        raise ValueError("trials, resolution and arena must be positive")

    root = Path(__file__).resolve().parents[1]
    sizes = load_bar_sizes(root, args.bar_pool)
    band = (
        args.band_x[0] * args.arena,
        args.band_x[1] * args.arena,
        0.0,
        args.arena,
    )
    fields = (
        "density",
        "side_clearance_m",
        "crossing_rate",
        "largest_component_fraction",
        "random_pair_connectivity",
        "free_area_fraction",
        "placement_fallback_rate",
        "mean_fallback_count",
    )
    print(",".join(fields))
    output_rows = []
    for density in args.densities:
        samples = {margin: [] for margin in args.margins}
        fallback_counts = []
        for trial in range(args.trials):
            rng = random.Random(args.seed + density * 100003 + trial)
            if args.placement == "navrl_band":
                bars, _, relaxations, merges = place_bars_navrl_band(
                    density,
                    rng,
                    sizes,
                    band,
                    touch=args.touch,
                    gap=args.gap,
                )
                fallback = merges
            else:
                bars, _, relaxations, merges = place_bars_legacy(
                    density, rng, sizes, band
                )
                fallback = relaxations
            fallback_counts.append(fallback)
            for margin in args.margins:
                samples[margin].append(
                    topology_metrics(bars, args.arena, args.resolution, margin)
                )
        fallback_counts = np.asarray(fallback_counts, dtype=float)
        for margin in args.margins:
            values = np.asarray(samples[margin], dtype=float)
            row = {
                "density": density,
                "side_clearance_m": margin,
                "crossing_rate": values[:, 0].mean(),
                "largest_component_fraction": values[:, 1].mean(),
                "random_pair_connectivity": values[:, 2].mean(),
                "free_area_fraction": values[:, 3].mean(),
                "placement_fallback_rate": np.mean(fallback_counts > 0),
                "mean_fallback_count": fallback_counts.mean(),
            }
            output_rows.append(row)
            print(
                "%d,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f"
                % tuple(row[field] for field in fields)
            )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    print("\nStopping-distance reference (a=2 m/s^2, control+perception latency=0.1 s):")
    for speed in (0.5, 1.0, 1.5, 2.0, 2.5):
        stopping = speed * 0.1 + speed * speed / (2.0 * 2.0)
        print("  %.1f m/s -> %.3f m" % (speed, stopping))


if __name__ == "__main__":
    main()
