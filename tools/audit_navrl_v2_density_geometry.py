#!/usr/bin/env python3
"""CPU-only geometric audit of the corrected v2 arena: is free space still connected at density D?

Nothing here imports Isaac Gym or touches a GPU.  Layouts are produced by the REAL placer
(``AssetManager._footprint_clearance_xy_spacing``, the ``footprint_clearance`` non-overlap mode
used by ``train_navrl_physical_fresh.sh``), with the REAL sampled bar footprints read from the
``bars_h3`` URDF pool -- never an assumed 0.60 m square (the flaw recorded for the earlier
topology dump in docs/diagnostic_synthesis_2026-08-21.md).

Start/goal pairs mirror ``NavRLTask._randomize_general_drone_spawn`` and
``NavRLTask._sample_general_target`` rather than a uniform box.  Since 2026-08-27 the spawn
mirror is footprint aware (each bar inflated by its own circumradius, matching the task); pass
``--spawn-clearance center`` to reproduce the frozen receipt, which was measured with the older
flat 0.65 m bar-centre rule and recorded a 27.45% start-in-inflated-obstacle rate at 205 bars.

Every distance is exact rectangle geometry: bars are axis aligned (bar min/max_state_ratio fix
roll=pitch=yaw=0), so the clearance field is min over bars of ||max(|p-c| - h, 0)||.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------
# Import the real placer without executing aerial_gym/__init__.py (which pulls in Isaac Gym).
# --------------------------------------------------------------------------------------------
def load_asset_manager_class():
    for name, rel in (
        ("aerial_gym", "aerial_gym"),
        ("aerial_gym.utils", "aerial_gym/utils"),
        ("aerial_gym.env_manager", "aerial_gym/env_manager"),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(ROOT / rel)]
            sys.modules[name] = pkg
    module = importlib.import_module("aerial_gym.env_manager.asset_manager")
    return module.AssetManager


# --------------------------------------------------------------------------------------------
# Measured constants (every one sourced from the repo; see RESULTS["provenance"]).
# --------------------------------------------------------------------------------------------
ARENA_XY = 40.0                 # NAVRL_ARENA_XY, train_navrl_physical_fresh.sh:137
SURFACE_CLEARANCE = 0.45        # NAVRL_PLACEMENT_SURFACE_CLEARANCE_M, navrl_bars_env
                                # obstacle_surface_clearance. Cite the symbol, not a line:
                                # the line numbers here had already drifted by 5 to 700 lines.
PLACEMENT_MODE = "footprint_clearance"
MAX_BARS = 300                  # NAVRL_MAX_BARS, train_navrl_physical_fresh.sh:144
BAR_POOL = "bars_h3"
BAR_X_RATIO = (0.0, 1.0)        # NAVRL_BAR_X_MIN/MAX = 0/1
BAR_Z_RATIO = 0.5

ROBOT_BOX_XY = 0.28             # resources/robots/quad/quad_navrl_ref5in.urdf collision box
ROBOT_TIP_AABB = 0.2825634      # documented prop-tip span in the same URDF header
TRACKING_RESERVE = 0.45         # navrl_task_config physical_tracking_margin (symbol, not line)

# Legacy (pre-2026-08-27) spawn rule: flat distance to the nearest bar CENTRE, blind to that
# bar's footprint.  Kept only so `--spawn-clearance center` reproduces the frozen receipt
# results/navrl_v2_density_geometry_audit_2026-08-27/density_geometry_canonical_6_28.json.
SPAWN_BAR_CENTER_CLEARANCE = 0.65
# Current spawn rule, mirroring NavRLTask._spawn_required_surface_margin_m: every candidate must
# clear EACH bar's own surface (centre distance minus that bar's circumradius) by the same
# inflation the connectivity graph below uses.
SPAWN_SURFACE_MARGIN_M = 0.5 * ROBOT_TIP_AABB * math.sqrt(2.0) + TRACKING_RESERVE
SPAWN_WALL_MARGIN = 1.0             # navrl_task _randomize_general_drone_spawn,
                                    # local spawn_wall_margin (symbol, not line)
GOAL_WALL_MARGIN = 1.25             # max(1.0, wall_margin 0.5 + boundary_margin 0.75)
TARGET_BOX_XYZ = (0.28, 0.28, 0.12)
GOAL_DIST_BANDS = {"canonical_6_28": (6.0, 28.0), "hard_22p5_28": (22.5, 28.0)}

GRID_RES = 0.05
SNAP_RADIUS = 0.6               # mirrors tools/analyze_navrl_v2_reachability.py
WIDTH_LEVELS = 36
WIDTH_STEP = 0.05


def bar_pool_sizes():
    sizes = []
    names = []
    for path in sorted(glob.glob(str(ROOT / "resources/models/environment_assets" / BAR_POOL / "*.urdf"))):
        box = ET.parse(path).getroot().find(".//collision/geometry/box")
        sizes.append([float(v) for v in box.get("size").split()])
        names.append(os.path.basename(path))
    return np.asarray(sizes, dtype=np.float64), names


class LayoutGenerator:
    """Thin wrapper around the real AssetManager placer."""

    def __init__(self, num_envs, pool_sizes, seed):
        import torch

        self.torch = torch
        AssetManager = load_asset_manager_class()
        self.num_envs = int(num_envs)
        self.pool_sizes = pool_sizes
        self.rng = random.Random(seed)
        A = MAX_BARS
        E = self.num_envs
        half = np.zeros((E, A, 3), dtype=np.float64)
        for e in range(E):
            picks = self.rng.choices(range(len(pool_sizes)), k=A)   # asset_loader.randomly_pick_assets_from_folder
            half[e] = 0.5 * pool_sizes[picks]
        self.half_np = half
        min_ratio = np.zeros((E, A, 13), dtype=np.float64)
        max_ratio = np.zeros((E, A, 13), dtype=np.float64)
        min_ratio[..., 0] = BAR_X_RATIO[0]
        max_ratio[..., 0] = BAR_X_RATIO[1]
        min_ratio[..., 1] = 0.0
        max_ratio[..., 1] = 1.0
        min_ratio[..., 2] = BAR_Z_RATIO
        max_ratio[..., 2] = BAR_Z_RATIO
        min_ratio[..., 6] = 1.0
        max_ratio[..., 6] = 1.0
        gtd = {
            "env_asset_state_tensor": torch.zeros((E, A, 13), dtype=torch.float32),
            "asset_min_state_ratio": torch.tensor(min_ratio, dtype=torch.float32),
            "asset_max_state_ratio": torch.tensor(max_ratio, dtype=torch.float32),
            "asset_collision_half_extents": torch.tensor(half, dtype=torch.float32),
            "env_bounds_min": torch.zeros((E, 3), dtype=torch.float32),
            "env_bounds_max": torch.tensor(
                np.tile([ARENA_XY, ARENA_XY, 3.0], (E, 1)), dtype=torch.float32
            ),
        }
        gtd["env_bounds_max"][:, 2] = 3.0
        self.am = AssetManager(
            gtd,
            num_keep_in_env=0,
            min_xy_spacing=1.5,                 # NavRLBarsEnvCfg.env.min_obstacle_xy_spacing
            placement_mode=PLACEMENT_MODE,
            placement_attempts_before_relax=128,
            placement_relax_factor=0.8,
            placement_candidate_batch_size=32,
            placement_touch_dist=0.4,
            placement_gap_dist=1.6,
            placement_surface_clearance=SURFACE_CLEARANCE,
        )

    def generate(self, density, env_ids=None):
        """Return (centers[E, D, 2], halves[E, D, 2]) or raise RuntimeError (generation failure)."""
        torch = self.torch
        ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids)
        self.am.reset_idx(ids, num_obstacles_per_env=int(density))
        centers = self.am.env_asset_state_tensor[ids][:, : int(density), 0:2].numpy().astype(np.float64)
        halves = self.half_np[ids.numpy()][:, : int(density), 0:2]
        return centers, halves


# --------------------------------------------------------------------------------------------
# Exact clearance field.
# --------------------------------------------------------------------------------------------
def clearance_field(centers, halves, res=GRID_RES, k=24):
    n = int(round(ARENA_XY / res))
    axis = (np.arange(n) + 0.5) * res
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    tree = cKDTree(centers)
    kk = min(k, centers.shape[0])
    _, idx = tree.query(pts, k=kk)
    idx = idx.reshape(pts.shape[0], kk)
    delta = np.abs(pts[:, None, :] - centers[idx]) - halves[idx]
    np.maximum(delta, 0.0, out=delta)
    dist = np.sqrt((delta ** 2).sum(axis=2)).min(axis=1)
    return dist.reshape(n, n), n, res


def cell_of(p, n, res):
    i = int(np.clip(p[0] / res, 0, n - 1))
    j = int(np.clip(p[1] / res, 0, n - 1))
    return i, j


def label_near(labels, i, j, n, res, snap=SNAP_RADIUS):
    if labels[i, j] > 0:
        return int(labels[i, j])
    r = int(snap / res)
    i0, i1 = max(0, i - r), min(n, i + r + 1)
    j0, j1 = max(0, j - r), min(n, j + r + 1)
    window = labels[i0:i1, j0:j1]
    fi, fj = np.nonzero(window > 0)
    if fi.size == 0:
        return 0
    d2 = (fi + i0 - i) ** 2 + (fj + j0 - j) ** 2
    return int(window[fi[np.argmin(d2)], fj[np.argmin(d2)]])


# --------------------------------------------------------------------------------------------
# Start/goal sampling mirroring the task.
# --------------------------------------------------------------------------------------------
def sample_pairs(centers, halves, rng, n_pairs, goal_band, spawn_clearance="footprint"):
    """Mirror _randomize_general_drone_spawn + _sample_general_target (physical target).

    `spawn_clearance` selects the spawn acceptance predicate:
      footprint - current task code: min over bars of (centre distance - that bar's circumradius)
                  >= SPAWN_SURFACE_MARGIN_M.
      center    - the pre-fix flat rule, for reproducing the frozen 2026-08-27 receipt.
    """
    bar_tree = cKDTree(centers)
    bar_circumradius = np.linalg.norm(halves, axis=1)
    neighbours = min(24, centers.shape[0])
    # Only a bar whose centre is nearer than this can decide the footprint-aware test.
    violation_bound = SPAWN_SURFACE_MARGIN_M + float(bar_circumradius.max())
    lo_s, hi_s = SPAWN_WALL_MARGIN, ARENA_XY - SPAWN_WALL_MARGIN
    lo_g, hi_g = GOAL_WALL_MARGIN, ARENA_XY - GOAL_WALL_MARGIN
    bar_radius = float(np.linalg.norm(halves, axis=1).max())
    target_radius = math.sqrt(sum((0.5 * v) ** 2 for v in TARGET_BOX_XYZ))
    goal_clearance = bar_radius + target_radius + TRACKING_RESERVE
    min_d, max_d = goal_band

    starts = np.empty((n_pairs, 2))
    goals = np.empty((n_pairs, 2))
    start_rejected = 0
    goal_rejected = 0
    for p in range(n_pairs):
        s = np.array([rng.uniform(lo_s, hi_s), rng.uniform(lo_s, hi_s)])
        ok = False
        for _ in range(64):                       # navrl_task.py:1815
            cand = np.array([rng.uniform(lo_s, hi_s), rng.uniform(lo_s, hi_s)])
            if spawn_clearance == "center":
                accepted = bar_tree.query(cand)[0] >= SPAWN_BAR_CENTER_CLEARANCE
            else:
                dist, idx = bar_tree.query(cand, k=neighbours)
                dist = np.atleast_1d(dist)
                idx = np.atleast_1d(idx)
                if dist[-1] < violation_bound:
                    # A farther bar could still be the binding one; fall back to the full scan.
                    worst = float(
                        (np.linalg.norm(centers - cand, axis=1) - bar_circumradius).min()
                    )
                else:
                    worst = float((dist - bar_circumradius[idx]).min())
                accepted = worst >= SPAWN_SURFACE_MARGIN_M
            if accepted:
                s, ok = cand, True
                break
        start_rejected += 0 if ok else 1
        g = np.array([rng.uniform(lo_g, hi_g), rng.uniform(lo_g, hi_g)])
        ok = False
        for _ in range(1024):                     # attempts=1024 for physical dynamics
            cand = np.array([rng.uniform(lo_g, hi_g), rng.uniform(lo_g, hi_g)])
            d = np.linalg.norm(cand - s)
            if not (min_d <= d <= max_d):
                continue
            if bar_tree.query(cand)[0] >= goal_clearance:
                g, ok = cand, True
                break
        goal_rejected += 0 if ok else 1
        starts[p] = s
        goals[p] = g
    return starts, goals, start_rejected, goal_rejected, goal_clearance


# --------------------------------------------------------------------------------------------
def analyse_layout(centers, halves, rng, n_pairs, inflation, goal_band, spawn_clearance="footprint"):
    field, n, res = clearance_field(centers, halves)
    levels = [inflation + i * WIDTH_STEP for i in range(WIDTH_LEVELS)]
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)   # 4-connectivity
    label_stack = []
    for lv in levels:
        lab, _ = ndimage.label(field >= lv, structure=structure)
        label_stack.append(lab)

    free0 = field >= inflation
    free_cells = int(free0.sum())
    lab0 = label_stack[0]
    if free_cells:
        counts = np.bincount(lab0.ravel())
        counts[0] = 0
        lcc_frac = float(counts.max()) / free_cells
    else:
        lcc_frac = 0.0

    starts, goals, s_rej, g_rej, goal_clearance = sample_pairs(
        centers, halves, rng, n_pairs, goal_band, spawn_clearance=spawn_clearance
    )
    start_infeasible = int((~free0[tuple(np.array([cell_of(s, n, res) for s in starts]).T)]).sum())
    goal_infeasible = int((~free0[tuple(np.array([cell_of(g, n, res) for g in goals]).T)]).sum())

    connected = 0
    widths = []
    for s, g in zip(starts, goals):
        si, sj = cell_of(s, n, res)
        gi, gj = cell_of(g, n, res)
        best = None
        for li, lab in enumerate(label_stack):
            ls = label_near(lab, si, sj, n, res)
            lg = label_near(lab, gi, gj, n, res)
            if ls > 0 and ls == lg:
                best = li
            else:
                break
        if best is not None:
            connected += 1
            widths.append(2.0 * levels[best])
    return {
        "free_area_frac_of_arena": free_cells / float(n * n),
        "lcc_frac_of_free": lcc_frac,
        "pairs": n_pairs,
        "connected": connected,
        "start_rejection_exhausted": s_rej,
        "goal_rejection_exhausted": g_rej,
        "start_in_inflated_obstacle": start_infeasible,
        "goal_in_inflated_obstacle": goal_infeasible,
        "goal_bar_center_clearance_m": goal_clearance,
        "widths": widths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=int, nargs="+",
                    default=[64, 70, 100, 130, 160, 190, 205, 220, 250, 300])
    ap.add_argument("--layouts", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--band", default="canonical_6_28", choices=sorted(GOAL_DIST_BANDS))
    ap.add_argument("--spawn-clearance", default="footprint", choices=("footprint", "center"),
                    dest="spawn_clearance",
                    help="footprint (default): mirror the current per-bar surface test in "
                         "NavRLTask._randomize_general_drone_spawn. center: the pre-2026-08-27 "
                         "flat 0.65 m bar-centre rule, kept to reproduce the frozen receipt.")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/navrl_v2_density_geometry_audit_2026-08-27/density_geometry.json")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    pool_sizes, pool_names = bar_pool_sizes()
    robot_radius = 0.5 * ROBOT_TIP_AABB * math.sqrt(2.0)          # yaw-invariant circumradius
    inflations = {
        "body_only": robot_radius,
        "body_plus_tracking": robot_radius + TRACKING_RESERVE,
    }
    band = GOAL_DIST_BANDS[args.band]

    gen = LayoutGenerator(num_envs=args.layouts, pool_sizes=pool_sizes, seed=args.seed)
    results = {}
    t_all = time.time()
    for density in args.densities:
        gen_failures = 0
        t0 = time.time()
        try:
            centers_b, halves_b = gen.generate(density)
        except RuntimeError:
            # attribute failures per layout: retry env by env with the same placer
            centers_b, halves_b, gen_failures = [], [], 0
            for e in range(args.layouts):
                try:
                    c, h = gen.generate(density, env_ids=[e])
                    centers_b.append(c[0]); halves_b.append(h[0])
                except RuntimeError:
                    gen_failures += 1
            centers_b = np.asarray(centers_b); halves_b = np.asarray(halves_b)
        gen_time = time.time() - t0
        per_layout_time = gen_time / max(1, args.layouts)

        density_entry = {
            "generation_failures": gen_failures,
            "layouts_generated": int(len(centers_b)),
            "gen_time_batch_s": gen_time,
            "gen_time_per_layout_s": per_layout_time,
        }
        for infl_name, infl in inflations.items():
            rng = random.Random(args.seed * 1000 + density)
            agg = []
            for e in range(len(centers_b)):
                agg.append(
                    analyse_layout(
                        centers_b[e], halves_b[e], rng, args.pairs, infl, band,
                        spawn_clearance=args.spawn_clearance,
                    )
                )
            pairs_total = sum(a["pairs"] for a in agg)
            conn = sum(a["connected"] for a in agg)
            widths = np.concatenate([np.asarray(a["widths"]) for a in agg]) if conn else np.zeros(0)
            lcc = np.asarray([a["lcc_frac_of_free"] for a in agg])
            per_layout_conn = np.asarray([a["connected"] / a["pairs"] for a in agg])
            density_entry[infl_name] = {
                "inflation_radius_m": infl,
                "free_area_frac_of_arena_mean": float(np.mean([a["free_area_frac_of_arena"] for a in agg])),
                "lcc_frac_of_free_mean": float(lcc.mean()),
                "lcc_frac_of_free_min": float(lcc.min()),
                "pairs_total": int(pairs_total),
                "connectivity": conn / float(pairs_total),
                "connectivity_layout_sem": float(per_layout_conn.std(ddof=1) / math.sqrt(len(per_layout_conn))),
                "no_route_fraction": 1.0 - conn / float(pairs_total),
                "corridor_width_m_p05": float(np.percentile(widths, 5)) if conn else None,
                "corridor_width_m_median": float(np.median(widths)) if conn else None,
                "corridor_width_m_min": float(widths.min()) if conn else None,
                "start_in_inflated_obstacle_frac":
                    sum(a["start_in_inflated_obstacle"] for a in agg) / float(pairs_total),
                "goal_in_inflated_obstacle_frac":
                    sum(a["goal_in_inflated_obstacle"] for a in agg) / float(pairs_total),
                "start_rejection_exhausted_frac":
                    sum(a["start_rejection_exhausted"] for a in agg) / float(pairs_total),
                "goal_rejection_exhausted_frac":
                    sum(a["goal_rejection_exhausted"] for a in agg) / float(pairs_total),
                "goal_bar_center_clearance_m": agg[0]["goal_bar_center_clearance_m"],
            }
        results[str(density)] = density_entry
        print(f"[audit] D={density:3d} done in {time.time()-t0:.1f}s "
              f"body={density_entry['body_only']['connectivity']:.4f} "
              f"body+track={density_entry['body_plus_tracking']['connectivity']:.4f} "
              f"genfail={gen_failures}", flush=True)

    verdict = {}
    for infl_name in inflations:
        passing = [int(d) for d in args.densities
                   if results[str(d)][infl_name]["connectivity"] >= 0.95
                   and results[str(d)][infl_name]["no_route_fraction"] <= 0.05
                   and results[str(d)]["generation_failures"] == 0]
        verdict[infl_name] = {
            "highest_passing_density": max(passing) if passing else None,
            "passing_densities": passing,
            "205_passes": 205 in passing,
        }

    payload = {
        "schema_version": "navrl.v2-density-geometry-audit.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "goal_band": {"name": args.band, "min_m": band[0], "max_m": band[1]},
        "spawn_clearance_mode": args.spawn_clearance,
        "layouts_per_density": args.layouts,
        "pairs_per_layout": args.pairs,
        "grid_res_m": GRID_RES,
        "thresholds": {"connectivity_min": 0.95, "no_route_max": 0.05, "generation_failures_max": 0},
        "inflation_m": inflations,
        "provenance": {
            "arena_xy_m": [ARENA_XY, "train_navrl_physical_fresh.sh:137 NAVRL_ARENA_XY=40"],
            "placement_mode": [PLACEMENT_MODE, "train_navrl_physical_fresh.sh:142"],
            "surface_clearance_m": [SURFACE_CLEARANCE, "navrl_bars_env obstacle_surface_clearance"],
            "placement_touch_dist_m": [0.4, "asset_manager.py:21 (unused by footprint_clearance)"],
            "placement_gap_dist_m": [1.6, "asset_manager.py:22 (unused by footprint_clearance)"],
            "min_xy_spacing_m": [1.5, "navrl_bars_env.py min_obstacle_xy_spacing"],
            "bar_pool": [BAR_POOL, "%d URDFs, footprint xy in [%.4f, %.4f] m" %
                         (len(pool_names), pool_sizes[:, :2].min(), pool_sizes[:, :2].max())],
            "bar_circumradius_m": [float(np.linalg.norm(0.5 * pool_sizes[:, :2], axis=1).min()),
                                   float(np.linalg.norm(0.5 * pool_sizes[:, :2], axis=1).max())],
            "robot_collision_box_xy_m": [ROBOT_BOX_XY, "quad_navrl_ref5in.urdf base_link collision box"],
            "robot_prop_tip_aabb_m": [ROBOT_TIP_AABB, "quad_navrl_ref5in.urdf header / navrl_ref5in_quad_config.py:11"],
            "robot_inflation_radius_m": [robot_radius, "half of prop-tip AABB * sqrt(2) (yaw invariant)"],
            "tracking_reserve_m": [TRACKING_RESERVE, "navrl_task_config physical_tracking_margin"],
            "spawn_clearance_mode": [
                args.spawn_clearance,
                "footprint = navrl_task.py _randomize_general_drone_spawn per-bar surface test; "
                "center = pre-2026-08-27 flat rule kept only to reproduce the frozen receipt",
            ],
            "spawn_surface_margin_m": [
                SPAWN_SURFACE_MARGIN_M,
                "navrl_task.py _spawn_required_surface_margin_m "
                "(robot inflation radius + physical_tracking_margin)",
            ],
            "spawn_bar_center_clearance_m": [
                SPAWN_BAR_CENTER_CLEARANCE,
                "legacy flat centre rule, applied only with --spawn-clearance center",
            ],
            "spawn_wall_margin_m": [SPAWN_WALL_MARGIN, "navrl_task _randomize_general_drone_spawn spawn_wall_margin"],
            "goal_wall_margin_m": [GOAL_WALL_MARGIN, "navrl_task.py:2070-2086 max(1.0, 0.5+0.75)"],
        },
        "verdict": verdict,
        "per_density": results,
        "total_runtime_s": time.time() - t_all,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"[audit] wrote {args.output}")


if __name__ == "__main__":
    main()
