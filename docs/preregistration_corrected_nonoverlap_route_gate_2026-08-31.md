# Preregistration — corrected non-overlap route/physical engineering gate

Date frozen: 2026-08-31 (Asia/Seoul), before any corrected-route GPU result was read.

## Question and scope

Does the physical waypoint target move safely and execute global routes in the corrected,
footprint-aware non-overlap environment strongly enough to justify one short **fresh** PPO smoke?
This is an engineering gate only. It loads no pursuer policy/checkpoint and cannot support a
capture, hardware, sim-to-real, or long-training claim.

The historical routed gate is not reused as evidence because it instantiated `navrl_band`,
`navrl_ref5in_quad`, and a 0.280 m target proxy. This gate binds the corrected lineage:

- arena `40×40×3 m`, `bars_h3`, full-width placement band;
- `NAVRL_PLACEMENT_MODE=footprint_clearance`;
- per-footprint surface clearance `0.45 m`, overlap fallback forbidden;
- pursuer `navrl_ref5in_v2_quad` and target proxy `0.283×0.283×0.12 m`;
- physical 6-DoF target, waypoint pattern, `global_astar_v1` vs matched route-off control;
- target speed `0.6/0.9/1.2/1.5 m/s`;
- training-support densities `70/115/160/205` bars;
- `NAVRL_MAX_BARS=300` only to match the later asset allocation. 300 bars is excluded from the
  training-authority gate because the frozen geometry audit already classifies it as disconnected.

## Execution contract

- seed 829, 32 environments, 300 policy intervals/cell, first 20 excluded from tracking error;
- 2 route arms × 4 speeds × 4 densities = exactly 32 cells;
- neutral pursuer action passes through the canonical action mapper; no PPO is loaded;
- matched arms must have identical initial bar, robot, and target pose digests per density;
- each child records imported source origin, runtime config/URDF hashes, CUDA/Python identity,
  initial-layout digest, route counters, reset timing, and an immutable result receipt;
- any missing/duplicate cell, source drift, non-v2 robot, 0.283 m proxy drift, placement drift,
  or receipt mismatch makes the entire execution VOID.

## Frozen physical-cell gates

Every routed 1.5 m/s training-density cell must pass all of:

- tracking RMSE ≤ 0.35 m/s;
- realized/commanded mean-speed ratio ≥ 0.80;
- contact-step fraction ≤ 0.01;
- routed local-step invalidation fraction ≤ 0.01;
- motor-saturation fraction ≤ 0.15;
- maximum tilt ≤ 60°;
- invalid-state fraction = 0.

The 70-bar route-mechanism pool, aggregated over all four speeds, must also satisfy:

- plan success ≥ 0.99;
- fallback interval fraction ≤ 0.01;
- same-goal reselection count = 0 at 70 bars and across all routed cells;
- at 70 bars and 0.6 m/s, goal completions per environment ≥ 0.5.

## Decision rule

Only `PASS_32_CELL_INTEGRITY + PASS_ROUTE_MECHANISM + PASS_FULL_1P5_CONTRACT` authorizes a separately
frozen 70-bar, 500-epoch, fresh PPO smoke. A density-conditioned pass alone does not authorize the
current `U[0.3,1.5] m/s` training launcher. Any failure stops before PPO; thresholds, speed support,
clearance, density cells, or seed are not changed after observing results.

Even a full pass does **not** authorize long training. The smoke must independently pass its
predeclared optimizer, action-support, outcome, environment-identity, and held-out checks before a
70→205 run may start. Warm-start and historical checkpoint reuse are forbidden.
