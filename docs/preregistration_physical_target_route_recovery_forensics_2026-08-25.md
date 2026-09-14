# Preregistration — routed physical-target recovery forensics

Date frozen: 2026-08-25. This is an evaluation-only diagnostic for the existing
`global_astar_v1` physical-target transition. It does not authorize PPO training, a planner
change, a target-command change, a reward/observation/termination change, or a change to the
original attempt2 evaluator or its artifacts.

## Provenance and frozen cells

The diagnostic refuses to run if the preserved attempt2 `summary.json` does not have SHA-256
`e5e4560464dc3a2080d904c2f8d2247e0c65e671dd63ea08d1b507ec65fc7197`. It records the current
execution git commit and the SHA-256 of the complete runtime source manifest, imported task/planner,
target-motion/controller, diagnostic tool, and instantiated BaseSim/robot sources in a separate
receipt. Each child also records Python/torch/CUDA/Isaac Gym/GPU-driver/ninja identities and the
post-construction runtime contract (actual density/speed, route settings, physical parameters,
arena/bar placement, physics timing, and ref5in config/URDF). Attempt2 is read-only input and
remains the authority for the prior gate. Later `--verify` checks that the recorded commit object
still exists and that all bound bytes/manifests are unchanged; it deliberately permits a
descendant/cherry-pick result commit when those bytes match, while rejecting byte drift or a forged
receipt.

The instantiated contract is fail-closed against drift in the 300-bar capacity, physical box
`[0.28,0.28,0.12]`, obstacle clearance `0.77`, motor arm `0.0777817`, yaw ratio `.01`, velocity /
altitude gains `2.5/4.0`, attitude/rate gains `[.08,.08,.04]` / `[.04,.04,.03]`, exact speed
min/final, route tolerance `.05`, expansion/waypoint limits `50000/128`, support
`0.2068816086567407`, RL dt `.1`, and arena/bar placement values. Route segment clearance is
computed once when a plan is observed and reused by fallback telemetry; this is an observer-side
performance optimization and does not alter planner decisions or commands.

Frozen runtime contract: seed `827`, route mode `global_astar_v1`, physical dynamics, waypoint
pattern, 32 environments, 300 RL intervals, and a 20-step warmup marker. All forensic transition
events include all 300 intervals (the warmup marker is retained only for parity with attempt2;
it is not an event exclusion). The minimal diagnostic grid is
the route-on slice `{70, 150, 205, 300} bars × {0.6, 1.5} m/s` (8 cells). This retains all four
attempt2 density knots and both speed endpoints while omitting the already-populated route-off
control arm and redundant interior speeds. Attempt2 established the relevant range at every
density: route-on local invalidation was 0.125–0.635% and fallback 32.6–85.0%; the 8-cell slice
tests whether that mechanism persists at both ends of each axis. No cell may be added, removed,
or substituted after events are inspected.

Run command (GPU, not run as part of this CPU-only change):

```text
<aerialgym-python> tools/diagnose_navrl_physical_target_route_recovery.py --run
```

Outputs go only under
`results/navrl_physical_target_route_recovery_forensics_seed827/`; they must never be written to
`results/navrl_physical_target_routed_gate_seed827_attempt2/`.

## Observer and claim boundary

The tool attaches a process-local observer to the existing route-manager calls. It consumes the
same start, bar AABB, boundary, support, route, speed, and validity values that the transition
already uses. CPU conversion occurs only in the diagnostic child. The observer is not imported by
the task and is absent from the normal control path. A malformed/non-finite diagnostic value is
reported as null and cannot become a passing zero.

At each invalidation, replan (including the initial plan), and fallback interval the record stores:

- exact hard clearance to the bars and boundary with target support only;
- exact soft clearance to the bars and boundary with target support plus the frozen `0.45 m`;
- local rounded-corner soft-bar free (`Euclidean distance from the support-inflated AABB >=
  0.4501 m`), route square-AABB soft free, and their disagreement. The former matches the
  existing bounded rollout check; the latter matches the global route's axis inflation. This is
  diagnostic geometry only and does not alter either transition;
- `unsafe_start_reason` (`obstacle`, `boundary`, `both`, `none`, or null when geometry is invalid);
- active-route cross-track error and whole-route polyline error;
- requested target speed and realized planar target speed when available;
- originating invalidation id, fallback age in RL steps, and fallback interval count per origin.

“Hard boundary” includes the existing runtime wall reserve (`wall_margin`, currently 0.50 m) and
target support; “soft boundary” uses the route's existing boundary reserve (currently 1.25 m,
wall plus physical boundary reserve). Thus the boundary hard/soft difference is 0.75 m, while
the obstacle hard/soft difference is exactly the frozen `0.45 m`; the tracking reserve is never
added to a boundary a second time. Bar distances are exact Euclidean point-to-closed-AABB
distances after the corresponding support inflation, not center-distance proxies. Route segment
clearances are also recorded with the exact closed-AABB corner test, so a raster corner shortcut
cannot be mistaken for a safe connector.

For unsafe-start plan/replan events the observer searches the existing planner grid at `0.25 m`
resolution in the planner's deterministic `±3`-cell (7×7) anchor neighbourhood. It reports
whether a nearest soft-free anchor exists and its distance. An anchor is accepted only if the
continuous connector from the observed target point to that anchor is exactly hard-free for the
bars and closed arena boundary. Invalidation and fallback events record clearance, route error,
speed, age, and origin but omit the expensive anchor search; anchor existence is an unsafe-start
recovery question, not a fallback command. This is a diagnostic classification only: it never
supplies a command or changes a planner decision. The neighbourhood is intentionally minimal
because it is the same bounded anchor search already used by the route planner.

## Analysis rules fixed before data

1. Report event counts by cell, invalidation reason, status, and origin id. A fallback with no
   observed originating invalidation is reported as `unattributed`; it is not silently assigned.
2. Report the distribution (count, median, p90, maximum) of hard/soft clearances, cross-track
   and polyline error, target speed, anchor distance, and fallback age. Nulls remain nulls.
3. Compare obstacle-versus-boundary unsafe-start attribution and, separately, the existence of an
   exact-hard-safe connector to a soft-free anchor. These are descriptive decompositions, not
   gates and not evidence of global route absence.
4. Keep `initial_plan` and `replan` status counts separate. The prior 70-bar raw attempt2 counts
   (attempts/replans/successes) were 336/304/38 at 0.6, 384/351/47 at 0.9, 352/312/61 at 1.2,
   and 406/360/69 at 1.5 m/s; therefore the initial 32 plans must not be treated as the whole
   failure population.
5. Before inspecting this diagnostic, freeze the following descriptive decision rule (not a
   planner or training gate): support of a recovery-dominant engineering hypothesis requires
   `unsafe_start`/`unsafe_start_cell` replan count `n >= 64`, Wilson 95% lower bound above 0.5
   for both (a) hard-free but soft-unsafe starts and (b) existing soft-free anchors with an exact
   hard-free connector, and attributed fallback intervals per local invalidation greater than the
   cooldown of 10 steps. Only the first unsafe replan per unique originating
   `local_step_infeasible` invalidation votes; repeats, unattributed initial failures, and
   other-origin replans are reported separately. Otherwise report `HARD_UNSAFE` when the
   hard-free/soft-unsafe ratio is at
   most 0.5, `ANCHOR_INSUFFICIENT` when the connector ratio is at most 0.5, or `INCONCLUSIVE`.
   These labels are descriptive only and do not authorize a fix or retuning.
6. Do not lower `0.45 m`, alter wall/support margins, tune speed/acceleration/turn/lookahead,
   alter route resolution, select a different anchor radius, or rerun a cell after seeing data.
   No result authorizes PPO or changes the original attempt2 verdict.

## Frozen deliverables

Each cell JSON is hashed in `receipt.json`, and the receipt contains the frozen contract, tool
hash, current source hashes, attempt2 hash, and explicit `attempt2_artifacts_read_only` and
`original_evaluator_unchanged` markers. A receipt or cell with a contract mismatch is invalid and
must be discarded rather than repaired in place.
