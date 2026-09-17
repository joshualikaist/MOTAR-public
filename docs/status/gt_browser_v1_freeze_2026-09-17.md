# GT_BROWSER_V1 — browser tracking preview freeze record, 2026-09-17

```text
BROWSER GT TRACKING PREVIEW
SIMULATION ONLY
NOT PPO
NOT PHYSX
NOT RESEARCH PERFORMANCE EVIDENCE
```

This freezes the **browser visualization** of moving-object tracking and close
approach. It is an engineering record. It reports no policy performance, no
PhysX result, and it changes no historical verdict.

| Gate | State |
|---|---|
| `logic_validation` | **PASS** |
| `determinism` | **PASS** |
| `browser_validation` | **PASS** |
| `memory_stability` | **PASS** |
| `responsive_validation` | **PASS** |
| `evidence_boundary` | **PASS** |
| **`GT_BROWSER_V1`** | **FROZEN** |

## Tracking objective

The reference is a relative position error of zero:

```text
d(t) = ||p_tracker(t) - p_target(t)|| -> 0
```

`0` is a tracking objective in the browser coordinate frame. It is not impact,
collision, or contact dynamics. The preview holds a **display-only standoff**
of 1.55 m so the two meshes do
not overlap or jitter; that standoff is a rendering offset and is deliberately
**not** connected to the historical 0.5 m capture radius, which stays with the
historical lineage as provenance.

## 1. Logic validation — 120 runs

Densities [70, 115, 160, 205], target speeds
[0.3, 0.9, 1.5] m/s, 10 seeds per
cell, 60 simulated seconds each, fixed
10 Hz. Harness
`tools/validate_gt_browser_tracking.js`; raw report `logic_validation.json`.

### Hard invariants — every one must be zero

| Invariant | Count |
|---|---|
| `nan_or_inf` | **0** |
| `target_obstacle_penetration` | **0** |
| `tracker_obstacle_penetration` | **0** |
| `wall_violation` | **0** |
| `teleport` | **0** |
| `speed_limit_violation` | **0** |
| `acceleration_limit_violation` | **0** |
| `turn_rate_violation` | **0** |
| `unsafe_route_segment` | **0** |
| `command_without_safe_route` | **0** |
| `blind_motion_after_route_invalidation` | **0** |

**All zero across all 120 runs.**

### Tracking diagnostics (cell means)

| bars | v (m/s) | d₀ | d_final | d_median | d_min | Δd | decreasing | direct LOS | stall | no route | replans/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 70 | 0.3 | 16.84 | 1.41 | 1.53 | 0.23 | 15.43 | 52.7 % | 88.6 % | 0.20 % | 0.00 % | 0.16 |
| 70 | 0.9 | 21.35 | 1.57 | 1.52 | 0.27 | 19.78 | 58.7 % | 84.2 % | 0.30 % | 0.00 % | 0.30 |
| 70 | 1.5 | 15.55 | 1.61 | 1.53 | 0.29 | 13.94 | 63.6 % | 84.6 % | 0.28 % | 0.00 % | 0.44 |
| 115 | 0.3 | 15.18 | 1.43 | 1.52 | 0.31 | 13.75 | 51.4 % | 83.9 % | 0.43 % | 0.00 % | 0.30 |
| 115 | 0.9 | 17.71 | 1.52 | 1.53 | 0.33 | 16.19 | 58.7 % | 80.9 % | 0.55 % | 0.00 % | 0.45 |
| 115 | 1.5 | 18.90 | 1.65 | 1.63 | 0.18 | 17.25 | 64.4 % | 67.9 % | 0.58 % | 0.00 % | 0.75 |
| 160 | 0.3 | 19.29 | 1.45 | 1.53 | 0.11 | 17.84 | 52.8 % | 74.2 % | 0.58 % | 0.00 % | 0.49 |
| 160 | 0.9 | 17.50 | 1.52 | 1.55 | 0.18 | 15.98 | 58.1 % | 74.0 % | 0.68 % | 0.03 % | 0.61 |
| 160 | 1.5 | 19.50 | 2.11 | 1.85 | 0.14 | 17.39 | 62.6 % | 57.9 % | 0.95 % | 0.02 % | 1.06 |
| 205 | 0.3 | 14.81 | 1.35 | 1.53 | 0.05 | 13.46 | 51.6 % | 73.8 % | 0.60 % | 0.02 % | 0.48 |
| 205 | 0.9 | 16.08 | 2.17 | 1.74 | 0.07 | 13.91 | 57.3 % | 54.4 % | 1.17 % | 5.18 % | 1.36 |
| 205 | 1.5 | 20.74 | 1.95 | 2.13 | 0.11 | 18.79 | 61.0 % | 44.5 % | 1.28 % | 0.00 % | 1.32 |

Across all 120 runs: distance reduction mean **16.14 m**
(min 4.75, max 26.56);
minimum distance mean **0.190 m** (min 0.004);
terminal tracking error mean **1.602 m**
against the 1.55 m display standoff;
RMS tracking error mean 4.493 m;
route switches mean 36.0;
target goal completions total 135;
valid-route-but-stalled mean 0.38 s per 60 s;
heading-oscillation period-2 events (>25° reversals) total
**0**;
bounded emergency holds mean 7.7 ticks per 600
(1.28 %);
connected-spawn relocations: target 12,
tracker 16 of 120 spawns.

`decreasing` is measured over the whole run. Once the tracker has converged onto
the standoff band it oscillates about it, so the figure settles near 50-65 % by
construction; it is an approach-phase statistic, not a quality score. The
convergence claim rests on `d_final`, `d_median` and the distance reduction.

## 2. Determinism

`tests/test_status_gt_tracking_determinism.js`:

- identical seed replays byte-identical state digests over 300 steps;
- **render cadence independence**: 30 / 60 / 120 FPS commit the *same* trajectory
  over 30 s of wall clock, compared digest-by-digest. The fixed 10 Hz clock only
  decides *when* a step commits; the step always receives the fixed dt;
- a paused renderer commits zero steps at any cadence;
- obstacle-free convergence: from a positive initial distance the tracker reaches
  the standoff band and *stays* there (median, p90 and max bounds), at target
  speeds 0.3 / 0.9 / 1.5 m/s;
- the certified look-ahead never hands the follower a blocked chord, and
  line-of-sight tracking never publishes a stale route carrot.

Under obstacles, monotone distance decrease is **not** asserted. The engineering
diagnostic is only that long-term progress is positive where a safe route exists.

## 3. Browser validation — real Chrome

`/usr/bin/google-chrome`, 65 s per viewport, WebGL via
ANGLE/SwiftShader, driven by `tools/browser_validation/`. The probe is injected
only for `?probe=1` on the local validation server; the published page is not
modified.

| viewport | measured CSS px | s | fps | frame ms median | p95 | sim steps/s | planner/s | LiDAR/s | heap MB start/end | geom/tex | teleports | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| desktop | 1440x761 | 65.0 | 53.8 | 17.4 | 26.7 | 9.57 | 0.69 | 9.66 | 4.91 / 10.48 | 39 / 3 | 0 | PASS |
| tablet | 1024x629 | 65.0 | 57.7 | 16.7 | 22.0 | 9.57 | 0.69 | 9.66 | 4.91 / 11.08 | 39 / 3 | 0 | PASS |
| mobile | 390x844 (iframe host) | 65.0 | 59.9 | 16.7 | 17.9 | 9.57 | 0.69 | 9.66 | 7.49 / 7.27 | 39 / 2 | 0 | PASS |

Each run scripted: camera view cycling, pause/resume, click/tap ground goal,
auto roam, bar-count changes, target-speed changes, LiDAR toggle. All
42 scripted
interactions executed with **zero** errors and zero page exceptions. The four
bar-count/target-speed changes rebuild the scene on purpose; those rebuilds are
counted separately (`rebuilds`) and excluded from the teleport check, which is
bounded by *committed simulation steps* rather than wall clock.

Sensor HUD reached all three states across the runs, computed independently of
the GT tracking planner: DETECTED 1115, OCCLUDED 196, OUT OF FOV 639
(samples, all viewports).

The GT evidence badge was present in every run:

```text
BROWSER GT PREVIEW
NOT PPO · NOT PHYSX · NOT RESEARCH EVIDENCE
```

**Mobile note.** Headless Chrome will not open a window narrower than ~500 CSS
px, so `--window-size=390` silently yields a 500 px viewport. The narrow
breakpoint is therefore hosted in an iframe sized to exactly 390x844 — an iframe
is its own viewport, so the page's media queries evaluate at the true width —
and the verdict asserts that `window.innerWidth` matches the requested width.
Without that assertion the mobile row would report PASS at whatever width the
browser happened to grant.

## 4. Measured vs not measured

Reported above are measured values only. Not measured in this freeze:

```text
GPU-accelerated frame timing on real hardware   NOT_MEASURED
real mobile devices / touch hardware            NOT_MEASURED
Safari / Firefox engines                        NOT_MEASURED
long-session (> 65 s) memory behaviour          NOT_MEASURED
network-constrained first paint                 NOT_MEASURED
```

Frame times come from SwiftShader software rasterization in headless Chrome.
They bound the CPU-side cost of the scene; they are **not** a claim about
frame rate on a user's GPU.

## 5. Evidence boundary

Unchanged by this work, and verified:

- actor observation tensors and every `aerial_gym` task observation builder;
- reward, controller, safety filter, and all checkpoints;
- all recorded research verdicts and lifecycle states — P10 `INCONCLUSIVE`,
  D8b `MATERIAL_LOSS`, D8b causality `NOT_TESTED`, D8c `PLANNED`, D9 `NOT_RUN`,
  live RGB -> policy `NOT_TESTED`, bearing / true metric range / persistent ID
  `BLOCKED`, TM-E0/E1/E2 `IMPLEMENTED`, TM-E2 policy comparison `NOT_TESTED`,
  TM-E3/E4 `PLANNED`;
- `docs/assets/paper/overview-2026-09-13/` stays hash-pinned and was not
  regenerated.

`tests/test_browser_gt_state_not_in_policy_observation.py` enforces that browser
GT target state and the GT obstacle map never reach an actor observation or a
historical policy input.

## 6. Next research

`docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md` is preregistered.

```text
E0/E1/E2 preregistration   READY
experiment execution       NOT_RUN
new PPO training           NOT_RUN
reward modification        NONE
```
