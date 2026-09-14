# Offline topology-label snapshot contract

`tools/analyze_navrl_topology_labels.py` assigns static 2-D difficulty labels without importing
the simulator or changing actor observations, rewards, termination, placement, or policy code.

## Exact snapshot format

Use `motar.topology-layout.v1` JSON. Every layout needs the arena bounds, start and goal XY, and
the center **and actual XY footprint** of every active bar. Generate a valid example with:

```bash
python tools/analyze_navrl_topology_labels.py \
  --write-contract-example results/topology_layout_example.json
```

Evaluate it with explicit geometric assumptions:

```bash
python tools/analyze_navrl_topology_labels.py \
  results/topology_layout_example.json \
  --output results/topology_layout_example_labels.json \
  --resolution-m 0.10 \
  --vehicle-half-width-m 0.14 \
  --side-clearance-m 0.20 \
  --sensor-range-m 12.0 \
  --cluster-gap-m 0.45
```

The defaults match the current 0.28 m collision-box width, 0.20 m diagnostic side-clearance,
12 m LiDAR range, the deployed `cluster_sector` representation's 0.45 m cluster gap, and the
existing 0.10 m feasibility grid. They are recorded in every output row and must be changed
explicitly when analysing another platform or clearance convention.

## Labels and boundaries of interpretation

- `path_exists`: static grid path from start to final goal, with bars inflated by vehicle
  half-width plus requested side-clearance.
- `shortest_path_detour_ratio`: grid shortest-path length divided by Euclidean start-goal length.
- `minimum_usable_side_clearance_along_path_m`: minimum distance from the shortest path to a bar
  surface or arena boundary, minus vehicle half-width. This reports geometric room, not stopping
  distance or dynamic feasibility.
- `local_culdesac_proxy`: within a sensor-radius disc around the start, an enclosed component or
  a component with one narrow angular exit while the goal is outside sensor range. It is a local
  topology proxy, not evidence that the policy planned, rejected, or selected a route.
- `obstacle_count_within_sensor_range` and `cluster_count_within_sensor_range`: footprint-distance
  counts from the start. Bars whose surface gap is at most `cluster_surface_gap_m` share a cluster.

All labels are GT/offline diagnostics. They must never be concatenated into actor/critic inputs or
used to alter reward/termination in the same experiment lineage.

## Existing-result limitation

The historical `NAVRL_EPISODE_DUMP=<path>.npz` contract contains `bars_xy`, `spawn`, `target_end`
and `outcome`, so it can be processed directly, but it does **not** contain `bars_size_xy`.
Processing therefore requires an explicit assumed square footprint:

```bash
python tools/analyze_navrl_topology_labels.py \
  results/navrl_v2_bar_ceiling/episodes_seed167.npz \
  --default-bar-size-m 0.60 \
  --output results/navrl_v2_bar_ceiling/topology_labels_assumed_0p60.json
```

Those rows are marked `bar_size_source=assumed_default`. Since the actual bar pool ranges from
0.4 to 0.8 m, they are suitable for exploratory association only, not an exact publication
number. Existing summary JSON/CSV files that saved only aggregate outcome rates cannot be
retrospectively topology-labelled at all.

For future frozen-policy evaluations, export the JSON snapshot beside the evaluation receipt.
The export should be evaluation-only and contain the actual asset footprint selected for each
active bar. This repository intentionally does not wire the exporter into training semantics.
