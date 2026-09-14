#!/usr/bin/env python3
"""검증 4 stage ①: static reachability oracle for bar-contact episodes.

Input: the per-episode dump written by the instrumented evaluation
(NAVRL_EPISODE_DUMP=<path>.npz): per terminated episode, the active bar centres, the spawn
position, the final drone/target positions and the outcome code.

Question: for the episodes that ENDED in bar contact, did a collision-free static path from the
spawn to the target's final position exist at all? Bars are boxes with footprints in
[0.4, 0.8] m; per-bar sizes are not dumped, so the oracle brackets the answer with disk radii:

  optimistic   r = 0.2 (min half-width) + 0.2 (drone radius)          = 0.40 m
  governor     r = 0.45 (riskcap path half-width) + 0.2               = 0.65 m
  pessimistic  r = 0.566 (max half-diagonal) + 0.2                    = 0.766 m

If contact episodes are connected even at the PESSIMISTIC radius, geometry did not force the
ceiling and the residual crash rate is a perception/representation/control property. If many are
disconnected at the OPTIMISTIC radius, the arena itself caps the achievable rate.

CPU-only: occupancy grid at 0.1 m + KD-tree distance stamp + connected-component labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ARENA_MIN_M = 0.0
ARENA_MAX_M = 40.0
GRID_RES_M = 0.10
RADII = {"optimistic_0p40": 0.40, "governor_0p65": 0.65, "pessimistic_0p766": 0.766}
OUTCOME_NAMES = {0: "capture", 1: "bar_contact", 2: "below", 3: "above",
                 4: "out_of_bounds", 5: "timeout"}


def episode_connected(bars_xy, spawn_xy, goal_xy, radius):
    """Is goal reachable from spawn on the free-space grid with bars inflated by `radius`?"""
    n = int(round((ARENA_MAX_M - ARENA_MIN_M) / GRID_RES_M))
    axis = (np.arange(n) + 0.5) * GRID_RES_M + ARENA_MIN_M
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dist, _ = cKDTree(bars_xy).query(points, k=1)
    free = (dist > radius).reshape(n, n)

    def cell(p):
        i = int(np.clip((p[0] - ARENA_MIN_M) / GRID_RES_M, 0, n - 1))
        j = int(np.clip((p[1] - ARENA_MIN_M) / GRID_RES_M, 0, n - 1))
        return i, j

    si, sj = cell(spawn_xy)
    gi, gj = cell(goal_xy)
    # A spawn/goal cell inside an inflated disk (spawn clearance, moving target) is snapped to
    # the nearest free cell within 0.6 m rather than declared unreachable by quantisation.
    labels, _ = ndimage.label(free)

    def label_near(i, j):
        if labels[i, j] > 0:
            return labels[i, j]
        r = int(0.6 / GRID_RES_M)
        i0, i1 = max(0, i - r), min(n, i + r + 1)
        j0, j1 = max(0, j - r), min(n, j + r + 1)
        window = labels[i0:i1, j0:j1]
        free_i, free_j = np.nonzero(window > 0)
        if free_i.size == 0:
            return 0
        distance2 = (free_i + i0 - i) ** 2 + (free_j + j0 - j) ** 2
        nearest = int(np.argmin(distance2))
        return int(window[free_i[nearest], free_j[nearest]])

    ls, lg = label_near(si, sj), label_near(gi, gj)
    return bool(ls > 0 and ls == lg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/navrl_v2_bar_ceiling/reachability.json"))
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="0 = all bar-contact episodes")
    args = parser.parse_args()

    data = np.load(args.dump)
    outcome = data["outcome"].astype(int)
    bars = data["bars_xy"].astype(np.float32)       # (E, B, 2)
    spawn = data["spawn"].astype(np.float32)        # (E, 3)
    target = data["target_end"].astype(np.float32)  # (E, 3)

    counts = {OUTCOME_NAMES.get(k, str(k)): int((outcome == k).sum())
              for k in sorted(set(outcome.tolist()))}
    contact_idx = np.nonzero(outcome == 1)[0]
    if args.max_episodes:
        contact_idx = contact_idx[: args.max_episodes]

    connected = {name: 0 for name in RADII}
    for count, episode in enumerate(contact_idx, 1):
        bxy = bars[episode]
        finite = np.isfinite(bxy).all(axis=1)
        in_arena = (
            (bxy[:, 0] >= ARENA_MIN_M) & (bxy[:, 0] <= ARENA_MAX_M)
            & (bxy[:, 1] >= ARENA_MIN_M) & (bxy[:, 1] <= ARENA_MAX_M)
        )
        bxy = bxy[finite & in_arena]
        for name, radius in RADII.items():
            if episode_connected(bxy, spawn[episode, :2], target[episode, :2], radius):
                connected[name] += 1
        if count % 100 == 0:
            print(f"[oracle] {count}/{len(contact_idx)} contact episodes")

    n_contact = max(1, len(contact_idx))
    report = {
        "dump": str(args.dump),
        "episodes_total": int(outcome.shape[0]),
        "outcome_counts": counts,
        "contact_episodes_checked": int(len(contact_idx)),
        "grid_res_m": GRID_RES_M,
        "connected_fraction": {name: connected[name] / n_contact for name in RADII},
        "radii_m": RADII,
        "reading": (
            "connected means a static 2-D path to the final target position exists at this "
            "radius; it does not test dynamics, target motion, or the episode horizon"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["connected_fraction"], indent=2))
    print(f"[oracle] wrote {args.output}")


if __name__ == "__main__":
    main()
