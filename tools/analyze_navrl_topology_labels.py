#!/usr/bin/env python3
"""Offline GT topology labels for NavRL/MOTAR evaluation layouts.

This tool never imports or calls the simulator.  It consumes layout snapshots, rasterises bars
with the same axis-aligned footprint inflation convention used by the existing density
feasibility audit, and emits labels that can later be joined to frozen-policy outcomes.

Accepted inputs
---------------
``.json`` snapshot contract (preferred; exact bar sizes)::

  {
    "schema_version": "motar.topology-layout.v1",
    "arena": {"min_xy_m": [0, 0], "max_xy_m": [40, 40]},
    "layouts": [{
      "layout_id": "seed42-episode000001",
      "start_xy_m": [4, 20], "goal_xy_m": [34, 20],
      "bars": [{"center_xy_m": [10, 20], "size_xy_m": [0.6, 0.6]}]
    }]
  }

``.npz`` legacy ``NAVRL_EPISODE_DUMP`` files are also accepted.  They contain ``bars_xy``,
``spawn`` and ``target_end`` but not per-bar footprints.  In that mode ``--default-bar-size-m``
is required and every result is marked ``bar_size_source=assumed_default``.  Such labels are
diagnostic only; do not use them as exact publication values.

The local cul-de-sac label is deliberately a proxy, not a planner claim.  It restricts free
space to a sensor-radius disc around the start and counts angular exit arcs.  Zero exits is
enclosed, one narrow exit is cul-de-sac-like, two opposed exits is corridor-like, and broad
angular coverage is open space.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


SCHEMA_VERSION = "motar.topology-layout.v1"
OUTPUT_SCHEMA_VERSION = "motar.topology-labels.v1"
NEIGHBORS_8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class Layout:
    layout_id: str
    arena_min_xy_m: np.ndarray
    arena_max_xy_m: np.ndarray
    start_xy_m: np.ndarray
    goal_xy_m: np.ndarray
    bars_xy_m: np.ndarray
    bars_size_xy_m: np.ndarray
    bar_size_source: str
    source_metadata: dict[str, Any]


def _xy(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [x, y] pair, got {value!r}")
    return array


def load_json_layouts(path: Path) -> list[Layout]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {payload.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    arena = payload.get("arena", {})
    arena_min = _xy(arena.get("min_xy_m"), "arena.min_xy_m")
    arena_max = _xy(arena.get("max_xy_m"), "arena.max_xy_m")
    if np.any(arena_max <= arena_min):
        raise ValueError("arena.max_xy_m must be greater than arena.min_xy_m on both axes")
    layouts = []
    for index, raw in enumerate(payload.get("layouts", [])):
        raw_bars = raw.get("bars", [])
        centers = np.asarray([bar["center_xy_m"] for bar in raw_bars], dtype=np.float64)
        sizes = np.asarray([bar["size_xy_m"] for bar in raw_bars], dtype=np.float64)
        if not raw_bars:
            centers = np.empty((0, 2), dtype=np.float64)
            sizes = np.empty((0, 2), dtype=np.float64)
        if centers.shape != sizes.shape or centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"layout {index}: bars must contain center_xy_m and size_xy_m pairs")
        if not np.isfinite(centers).all() or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError(f"layout {index}: bar centres/sizes must be finite and sizes positive")
        known = {"layout_id", "start_xy_m", "goal_xy_m", "bars"}
        layouts.append(
            Layout(
                layout_id=str(raw.get("layout_id", f"layout-{index:06d}")),
                arena_min_xy_m=arena_min,
                arena_max_xy_m=arena_max,
                start_xy_m=_xy(raw.get("start_xy_m"), f"layout {index}.start_xy_m"),
                goal_xy_m=_xy(raw.get("goal_xy_m"), f"layout {index}.goal_xy_m"),
                bars_xy_m=centers,
                bars_size_xy_m=sizes,
                bar_size_source="snapshot_exact",
                source_metadata={key: value for key, value in raw.items() if key not in known},
            )
        )
    if not layouts:
        raise ValueError("snapshot contains no layouts")
    return layouts


def load_legacy_npz_layouts(
    path: Path,
    *,
    default_bar_size_m: float | None,
    arena_min_xy_m: tuple[float, float],
    arena_max_xy_m: tuple[float, float],
) -> list[Layout]:
    if default_bar_size_m is None or default_bar_size_m <= 0:
        raise ValueError(
            "legacy episode dumps omit bar footprints; pass a positive --default-bar-size-m"
        )
    with np.load(path) as data:
        required = {"bars_xy", "spawn", "target_end"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"legacy dump missing arrays: {sorted(missing)}")
        bars_all = data["bars_xy"].astype(np.float64)
        spawn = data["spawn"].astype(np.float64)
        target = data["target_end"].astype(np.float64)
        outcomes = data["outcome"].astype(int) if "outcome" in data.files else None
    layouts = []
    for index in range(bars_all.shape[0]):
        bars = bars_all[index]
        finite = np.isfinite(bars).all(axis=1)
        bars = bars[finite]
        metadata: dict[str, Any] = {"legacy_episode_index": index}
        if outcomes is not None:
            metadata["outcome_code"] = int(outcomes[index])
        layouts.append(
            Layout(
                layout_id=f"{path.stem}-episode-{index:06d}",
                arena_min_xy_m=np.asarray(arena_min_xy_m, dtype=np.float64),
                arena_max_xy_m=np.asarray(arena_max_xy_m, dtype=np.float64),
                start_xy_m=spawn[index, :2],
                goal_xy_m=target[index, :2],
                bars_xy_m=bars,
                bars_size_xy_m=np.full((bars.shape[0], 2), default_bar_size_m),
                bar_size_source="assumed_default",
                source_metadata=metadata,
            )
        )
    return layouts


def _grid_axes(layout: Layout, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    extent = layout.arena_max_xy_m - layout.arena_min_xy_m
    shape = np.maximum(1, np.ceil(extent / resolution_m).astype(int))
    x = layout.arena_min_xy_m[0] + (np.arange(shape[0]) + 0.5) * resolution_m
    y = layout.arena_min_xy_m[1] + (np.arange(shape[1]) + 0.5) * resolution_m
    return x, y


def rasterize_free_space(
    layout: Layout, resolution_m: float, vehicle_half_width_m: float, side_clearance_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned footprint inflation matching ``occupancy_grid`` in the density audit."""
    x, y = _grid_axes(layout, resolution_m)
    occupied = np.zeros((x.size, y.size), dtype=bool)
    inflation = vehicle_half_width_m + side_clearance_m
    for center, size in zip(layout.bars_xy_m, layout.bars_size_xy_m):
        inside_x = np.abs(x - center[0]) <= size[0] * 0.5 + inflation
        inside_y = np.abs(y - center[1]) <= size[1] * 0.5 + inflation
        occupied |= inside_x[:, None] & inside_y[None, :]
    return ~occupied, x, y


def _nearest_free_cell(
    point_xy: np.ndarray, free: np.ndarray, x: np.ndarray, y: np.ndarray, snap_radius_m: float
) -> tuple[int, int] | None:
    i = int(np.clip(np.searchsorted(x, point_xy[0]), 0, x.size - 1))
    j = int(np.clip(np.searchsorted(y, point_xy[1]), 0, y.size - 1))
    candidates = {(i, j), (max(0, i - 1), j), (i, max(0, j - 1))}
    viable = [(ii, jj) for ii, jj in candidates if free[ii, jj]]
    if not viable:
        radius_cells = int(math.ceil(snap_radius_m / max(x[1] - x[0] if x.size > 1 else 1, y[1] - y[0] if y.size > 1 else 1)))
        i0, i1 = max(0, i - radius_cells), min(x.size, i + radius_cells + 1)
        j0, j1 = max(0, j - radius_cells), min(y.size, j + radius_cells + 1)
        ii, jj = np.nonzero(free[i0:i1, j0:j1])
        viable = [(int(a + i0), int(b + j0)) for a, b in zip(ii, jj)]
    if not viable:
        return None
    dist2 = [float(np.sum((np.array([x[ii], y[jj]]) - point_xy) ** 2)) for ii, jj in viable]
    winner = int(np.argmin(dist2))
    return viable[winner] if dist2[winner] <= snap_radius_m**2 else None


def shortest_path(
    free: np.ndarray, start: tuple[int, int], goal: tuple[int, int], resolution_m: float
) -> tuple[float, list[tuple[int, int]]]:
    """8-neighbour Dijkstra with diagonal corner-cut prevention."""
    distance = np.full(free.shape, np.inf, dtype=np.float64)
    parent_i = np.full(free.shape, -1, dtype=np.int32)
    parent_j = np.full(free.shape, -1, dtype=np.int32)
    distance[start] = 0.0
    queue = [(0.0, start[0], start[1])]
    while queue:
        current, i, j = heapq.heappop(queue)
        if current != distance[i, j]:
            continue
        if (i, j) == goal:
            break
        for di, dj, step in NEIGHBORS_8:
            ii, jj = i + di, j + dj
            if not (0 <= ii < free.shape[0] and 0 <= jj < free.shape[1] and free[ii, jj]):
                continue
            if di and dj and (not free[i + di, j] or not free[i, j + dj]):
                continue
            candidate = current + step * resolution_m
            if candidate < distance[ii, jj]:
                distance[ii, jj] = candidate
                parent_i[ii, jj], parent_j[ii, jj] = i, j
                heapq.heappush(queue, (candidate, ii, jj))
    if not np.isfinite(distance[goal]):
        return math.inf, []
    path = [goal]
    cursor = goal
    while cursor != start:
        cursor = (int(parent_i[cursor]), int(parent_j[cursor]))
        path.append(cursor)
    path.reverse()
    return float(distance[goal]), path


def _angular_exit_bins(angles: np.ndarray, angular_bin_deg: float = 5.0) -> np.ndarray:
    n_bins = int(round(360.0 / angular_bin_deg))
    occupied = np.zeros(n_bins, dtype=bool)
    if angles.size:
        occupied[np.floor((angles + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins] = True
    return occupied


def _angular_exit_arcs(occupied: np.ndarray) -> int:
    if occupied.all():
        return 1
    # Circular false->true transitions count connected angular arcs.
    return int(np.sum(occupied & ~np.roll(occupied, 1)))


def local_dead_end_proxy(
    free: np.ndarray,
    start: tuple[int, int],
    goal_distance_m: float,
    x: np.ndarray,
    y: np.ndarray,
    sensor_range_m: float,
    resolution_m: float,
) -> dict[str, Any]:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    sx, sy = x[start[0]], y[start[1]]
    radius = np.hypot(xx - sx, yy - sy)
    local_free = free & (radius <= sensor_range_m + 0.5 * resolution_m)
    labels, _ = ndimage.label(local_free, structure=np.ones((3, 3), dtype=np.int8))
    component = labels == labels[start]
    shell = np.abs(radius - sensor_range_m) <= 1.5 * resolution_m
    exit_i, exit_j = np.nonzero(component & shell)
    angles = np.arctan2(y[exit_j] - sy, x[exit_i] - sx)
    exit_bins = _angular_exit_bins(angles)
    arcs = _angular_exit_arcs(exit_bins)
    coverage = float(exit_bins.mean())
    # Arena boundaries can clip the sensor disc.  Compare against the angular shell that exists
    # inside the arena so an open spawn near a wall is not called a cul-de-sac.
    available_i, available_j = np.nonzero(shell)
    available_angles = np.arctan2(y[available_j] - sy, x[available_i] - sx)
    available_bins = _angular_exit_bins(available_angles)
    available_count = max(1, int(available_bins.sum()))
    available_normalized_coverage = float((exit_bins & available_bins).sum()) / available_count
    goal_outside = goal_distance_m > sensor_range_m
    broad_open = available_normalized_coverage >= 0.75
    culdesac = bool(goal_outside and not broad_open and arcs <= 1)
    if broad_open or not goal_outside:
        severity = 0.0
    elif arcs == 0:
        severity = 1.0
    elif arcs == 1:
        severity = 0.5 + 0.5 * (1.0 - coverage)
    else:
        severity = 0.0
    return {
        "local_exit_arc_count": arcs,
        "local_exit_angular_coverage": coverage,
        "local_exit_available_normalized_coverage": available_normalized_coverage,
        "local_culdesac_proxy": culdesac,
        "local_dead_end_severity_proxy": severity,
    }


def _surface_distance_to_rectangles(points: np.ndarray, layout: Layout) -> np.ndarray:
    if layout.bars_xy_m.size == 0:
        return np.full(points.shape[0], np.inf)
    delta = np.abs(points[:, None, :] - layout.bars_xy_m[None, :, :]) - layout.bars_size_xy_m[None, :, :] * 0.5
    outside = np.maximum(delta, 0.0)
    return np.linalg.norm(outside, axis=2).min(axis=1)


def obstacle_cluster_counts(
    layout: Layout, sensor_range_m: float, cluster_gap_m: float
) -> tuple[int, int]:
    if layout.bars_xy_m.size == 0:
        return 0, 0
    start = layout.start_xy_m[None, :]
    delta = np.abs(start - layout.bars_xy_m) - layout.bars_size_xy_m * 0.5
    distance = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    visible = np.nonzero(distance <= sensor_range_m)[0]
    if not visible.size:
        return 0, 0
    centers = layout.bars_xy_m[visible]
    sizes = layout.bars_size_xy_m[visible]
    parent = np.arange(visible.size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[bb] = aa

    for i in range(visible.size):
        for j in range(i):
            axis_gap = np.abs(centers[i] - centers[j]) - 0.5 * (sizes[i] + sizes[j])
            surface_gap = float(np.linalg.norm(np.maximum(axis_gap, 0.0)))
            if surface_gap <= cluster_gap_m:
                union(i, j)
    return int(visible.size), len({find(i) for i in range(visible.size)})


def label_layout(
    layout: Layout,
    *,
    resolution_m: float,
    vehicle_half_width_m: float,
    side_clearance_m: float,
    sensor_range_m: float,
    cluster_gap_m: float,
    snap_radius_m: float,
) -> dict[str, Any]:
    free, x, y = rasterize_free_space(
        layout, resolution_m, vehicle_half_width_m, side_clearance_m
    )
    start = _nearest_free_cell(layout.start_xy_m, free, x, y, snap_radius_m)
    goal = _nearest_free_cell(layout.goal_xy_m, free, x, y, snap_radius_m)
    path_length, path = (math.inf, []) if start is None or goal is None else shortest_path(
        free, start, goal, resolution_m
    )
    direct = float(np.linalg.norm(layout.goal_xy_m - layout.start_xy_m))
    detour = (
        max(1.0, path_length / direct)
        if path and direct > resolution_m
        else (1.0 if path else None)
    )
    if path:
        points = np.asarray([[x[i], y[j]] for i, j in path])
        obstacle_surface = _surface_distance_to_rectangles(points, layout)
        boundary = np.minimum.reduce(
            [
                points[:, 0] - layout.arena_min_xy_m[0],
                layout.arena_max_xy_m[0] - points[:, 0],
                points[:, 1] - layout.arena_min_xy_m[1],
                layout.arena_max_xy_m[1] - points[:, 1],
            ]
        )
        raw_clearance = float(np.minimum(obstacle_surface, boundary).min())
        usable_clearance = raw_clearance - vehicle_half_width_m
    else:
        raw_clearance = None
        usable_clearance = None
    visible_obstacles, visible_clusters = obstacle_cluster_counts(
        layout, sensor_range_m, cluster_gap_m
    )
    dead_end = (
        local_dead_end_proxy(
            free, start, direct, x, y, sensor_range_m, resolution_m
        )
        if start is not None
        else {
            "local_exit_arc_count": 0,
            "local_exit_angular_coverage": 0.0,
            "local_exit_available_normalized_coverage": 0.0,
            "local_culdesac_proxy": True,
            "local_dead_end_severity_proxy": 1.0,
        }
    )
    return {
        "layout_id": layout.layout_id,
        "path_exists": bool(path),
        "shortest_path_length_m": None if not path else path_length,
        "straight_line_distance_m": direct,
        "shortest_path_detour_ratio": detour,
        "minimum_raw_center_clearance_along_path_m": raw_clearance,
        "minimum_usable_side_clearance_along_path_m": usable_clearance,
        "obstacle_count_within_sensor_range": visible_obstacles,
        "cluster_count_within_sensor_range": visible_clusters,
        **dead_end,
        "bar_count": int(layout.bars_xy_m.shape[0]),
        "bar_size_source": layout.bar_size_source,
        "source_metadata": layout.source_metadata,
        "metadata": {
            "grid_resolution_m": resolution_m,
            "vehicle_half_width_m": vehicle_half_width_m,
            "side_clearance_m": side_clearance_m,
            "inflation_m": vehicle_half_width_m + side_clearance_m,
            "sensor_range_m": sensor_range_m,
            "cluster_surface_gap_m": cluster_gap_m,
            "endpoint_snap_radius_m": snap_radius_m,
            "arena_min_xy_m": layout.arena_min_xy_m.tolist(),
            "arena_max_xy_m": layout.arena_max_xy_m.tolist(),
            "raster_inflation_convention": "axis_aligned_bar_half_extent_plus_vehicle_half_width_plus_side_clearance",
            "culdesac_definition": "sensor-disc reachable exit arcs; diagnostic proxy, not dynamic reachability",
        },
    }


def contract_example() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arena": {"min_xy_m": [0.0, 0.0], "max_xy_m": [40.0, 40.0]},
        "layouts": [
            {
                "layout_id": "seed42-episode000001",
                "start_xy_m": [4.0, 20.0],
                "goal_xy_m": [34.0, 20.0],
                "bars": [
                    {"center_xy_m": [20.0, 12.0], "size_xy_m": [0.6, 0.6]}
                ],
                "outcome": "capture",
                "seed": 42,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-contract-example", type=Path)
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--vehicle-half-width-m", type=float, default=0.14)
    parser.add_argument("--side-clearance-m", type=float, default=0.20)
    parser.add_argument("--sensor-range-m", type=float, default=12.0)
    parser.add_argument("--cluster-gap-m", type=float, default=0.45)
    parser.add_argument("--endpoint-snap-radius-m", type=float, default=0.60)
    parser.add_argument("--default-bar-size-m", type=float)
    parser.add_argument("--arena-min-xy-m", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--arena-max-xy-m", type=float, nargs=2, default=(40.0, 40.0))
    parser.add_argument("--max-layouts", type=int, default=0)
    args = parser.parse_args()

    if args.write_contract_example:
        args.write_contract_example.parent.mkdir(parents=True, exist_ok=True)
        args.write_contract_example.write_text(
            json.dumps(contract_example(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote snapshot contract example: {args.write_contract_example}")
        if args.input is None:
            return
    if args.input is None or args.output is None:
        parser.error("input and --output are required unless only --write-contract-example is used")
    positive = (
        args.resolution_m,
        args.vehicle_half_width_m,
        args.sensor_range_m,
        args.endpoint_snap_radius_m,
    )
    if any(value <= 0 for value in positive) or args.side_clearance_m < 0 or args.cluster_gap_m < 0:
        parser.error("resolution, footprint, sensor range and snap radius must be positive; gaps non-negative")

    if args.input.suffix.lower() == ".json":
        layouts = load_json_layouts(args.input)
    elif args.input.suffix.lower() == ".npz":
        layouts = load_legacy_npz_layouts(
            args.input,
            default_bar_size_m=args.default_bar_size_m,
            arena_min_xy_m=tuple(args.arena_min_xy_m),
            arena_max_xy_m=tuple(args.arena_max_xy_m),
        )
    else:
        parser.error("input must be a .json topology snapshot or legacy .npz episode dump")
    if args.max_layouts > 0:
        layouts = layouts[: args.max_layouts]
    labels = [
        label_layout(
            layout,
            resolution_m=args.resolution_m,
            vehicle_half_width_m=args.vehicle_half_width_m,
            side_clearance_m=args.side_clearance_m,
            sensor_range_m=args.sensor_range_m,
            cluster_gap_m=args.cluster_gap_m,
            snap_radius_m=args.endpoint_snap_radius_m,
        )
        for layout in layouts
    ]
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source": str(args.input),
        "layout_count": len(labels),
        "exact_bar_size_layout_count": sum(
            label["bar_size_source"] == "snapshot_exact" for label in labels
        ),
        "limitations": [
            "labels are static 2-D geometry diagnostics; they do not model target motion, dynamics, or episode horizon",
            "local_culdesac_proxy is sensor-disc topology, not evidence that a policy planned or rejected a route",
            "legacy NPZ results use an assumed bar size because NAVRL_EPISODE_DUMP omitted bars_size_xy",
        ],
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(labels)} topology labels: {args.output}")


if __name__ == "__main__":
    main()
