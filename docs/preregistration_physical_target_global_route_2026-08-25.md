# Preregistration — physical target global route (`global_astar_v1`)

Date frozen: 2026-08-25, before the final 16-layout latency benchmark.

## Scope and lineage boundary

This is an engineering gate for a **new, fresh-only** target-motion lineage. It does not
authorize PPO training and it does not change any legacy, bounded, or existing physical
transition. The model identifier is
`physx_ref5in_6dof_global_astar_aabb_v1`; the launcher must pin
`NAVRL_TARGET_DYNAMICS=physical`, `NAVRL_TARGET_PATTERN=waypoint`, and
`NAVRL_TARGET_ROUTE_MODE=global_astar_v1`. Existing checkpoints are inadmissible.

The planner may use ground-truth bar AABBs because it generates the simulated target's
environment dynamics; these values are not actor observations. The actor observation,
reward, and pursuer policy remain unchanged.

## Frozen geometry contract

- arena bounds are occupied outside their finite interior;
- every bar uses its asset-specific XY AABB, not a nominal radius;
- target support is the circumscribed radius of the declared 0.28 × 0.28 × 0.12 m box
  projected onto both XY axes (safe under future yaw/tilt), plus a 0.45 m tracking reserve;
- occupancy grid resolution is 0.25 m;
- A* uses deterministic eight-neighbour expansion without diagonal corner cutting;
- line-of-sight smoothing uses the same continuous inflated AABBs and arena bounds;
- a goal is selected at least 6 m away inside the start's connected component and A* runs
  once. Occupancy and connectivity are not recomputed for candidate goals;
- no path, invalid geometry, expansion exhaustion, local rollout failure, or stale support
  yields zero velocity and an explicit reason. There is no unchecked straight-line fallback.
  A local rollout failure remains stopped through cooldown, then resamples a destination in the
  current connected component instead of deterministically recreating the failed route;
- route cursor reach uses the task's 0.5 m waypoint reach and a passed-waypoint test.

Bulk-evaluation telemetry keeps cumulative goal completions separate from connected-goal
replans and invalidations by reason (`local_step_infeasible`, `support_contract_changed`,
`goal_changed`). It also measures planning batch count/size and wall time around the complete
device-to-host copy plus CPU plan. Timing is observational only and never changes control.

## Gates

1. Synthetic CPU tests must pass for blocked-straight detour, complete disconnection,
   narrow-corridor pass/fail, reflection and repeated-call determinism, AABB-corner LOS,
   invalid numeric input, conservative support, connected-goal selection, and waypoint
   overshoot, and local-failure route replacement.
2. Launcher preflights must show: canonical base = route off; canonical physical rejects a
   stale route request; only the routed fresh wrapper admits `global_astar_v1 + waypoint`.
3. At 70/150/205/300 bars, 16 deterministic `navrl_band` layouts use actual `bars_h3`
   AABBs. For each density, sequential mean planning latency must be at most 100 ms/env,
   p95 at most 150 ms/env, and `128 × mean` at most 10 s. Failure of any latency gate is
   `BLOCKED_PERFORMANCE`; no training launcher may be recommended.
4. Route availability at the starting curriculum density (70 bars) must be at least 90%.
   High-density no-route results are reported, not retuned away.

The 128-env figure is a conservative serial projection, not measured parallel throughput.
No GPU or PhysX behavior is tested here.

## Known geometric ceiling and claim boundary

An independent seed-825 raster topology probe used route-equivalent **obstacle inflation**
(20 layouts/density, 0.15 m analysis grid): its 0.14 m level-box half-width plus 0.517 m
side-clearance totals 0.6570 m, 0.118 mm above the route's 0.2068816 m support plus 0.45 m
tracking reserve. It found largest-component / sampled-pair
connectivity of 0.9999/0.9998 at 70 bars, 0.9966/0.9932 at 150, 0.9577/0.9211 at 205,
and 0.3998/0.2247 at 300; arena-crossing availability was 1/1/1/0.3. The 300-bar
rasterized free space is therefore strongly fragmented in this probe.
Same-component target roaming can prevent commanding an impossible crossing, but it cannot
support an arena-wide roaming claim at 300 bars. These external probe values are contextual
evidence and are not substituted for the benchmark gates above. The probe does not mirror the
route's 1.25 m + support boundary exclusion, 0.25 m grid, diagonal no-corner-cutting, or exact LOS,
so it is not an exact planner reachability certificate.

## After a pass

The only next authority is a short fresh physical simulator smoke checking route telemetry,
target/bar contacts, controller tracking, and actual reset stalls. Long training remains
blocked until that separate gate is preregistered and passed.
