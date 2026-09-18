# Browser arena motion audit — 2026-09-17

This is a code-level diagnosis of the public 3D arena. It is not a research
result, PPO measurement, PhysX validation, or a change to any historical
checkpoint.

Sources: `docs/status/arena.js`, `arena_motion.js`, `arena_route.js`.

## Evidence boundary

```text
browser visualization improvement != new research result
browser GT tracking != PPO performance
browser planner != simulator policy
browser motion != PhysX evidence
```

The actor observation tensor and `aerial_gym` task observation builders are
unchanged. Browser GT fields do not exist in the policy observation.

## Current target (historical display lineages)

| Mode | What it is | Obstacle handling | Notes |
|---|---|---|---|
| `legacy` | Checkpointed virtual point | Local heading candidates + circular push-out | Can clip/teleport-correct |
| `bounded` | TM-E2 local steering | 4.0 m/s², 150°/s, 1.0 s lookahead | Not policy-compared |
| `physical-style` | Bounded command + lag/attitude | Illustrative, not PhysX | Body-frame roll/pitch already used |
| `routed-preview` | Browser A* + bounded/lagged follow | Fail-closed zero command | Historical `global_astar_v1` geometry |

`routed-preview` already uses a global A* route (`arena_route.js`) with exact
inflated AABB safety, 0.25 m grid, 0.45 m tracking margin, and certified
corner handoff. Goal replacement exists; the 30 s episode watchdog in
`arena.js` still resets the scene.

## Current pursuer (historical browser)

`arena.js` calls:

```javascript
Motion.steerPursuerStep(pursuerX, pursuerY, targetX, targetY, ...)
```

| Capability | Historical browser pursuer |
|---|---|
| GT target position | YES — exact browser coordinates |
| GT target velocity used | NO |
| GT target heading used | NO |
| GT obstacle AABB | YES — for local swept clearance only |
| Arena bounds | YES |
| Global path planning | NO |
| Target prediction | NO |
| Persistent route memory | NO |
| Continuous follow | NO |
| Local lookahead | YES — 0.9 s along a straight heading |
| Bounded acceleration | NO — step is a heading pick, not `limitPlanarVelocity` |
| Bounded turn rate | Weak — heading continuity is a score term, not a hard rate |
| Capture reset | YES — swept 0.5 m |
| Time reset | YES — 30 s |
| Position push-out | YES — `pushPursuerOut` after the proposed step |

The local planner scores headings

```text
direct target heading + [0, ±15, ±30, ±45, ±60, ±90, ±120, ±150, 180]
```

over a short swept path. In dense bars this is a local-minimum heuristic: a
blocking cluster between pursuer and the current target point has no global
detour, so the agent weaves, stalls, or looks “stupid” even though it already
has exact target coordinates.

That is why the historical display is not “missing GT.” It has GT position and
still cannot route around clutter.

## Clock

Fixed 10 Hz simulation clock with render interpolation is already correct and
must be preserved. 30/60/144 FPS must not change the committed trajectory.

## Rendering notes (pre-change)

- Trails rebuilt `BufferGeometry` on every accepted point (`dispose` + `setFromPoints`).
- LiDAR ran every animation frame (`all beams × all bars`).
- HUD used `document.getElementById` inside `animate`.
- Pursuer visual bank used world-axis `vel.y`, not body-frame acceleration.
- Altitude used `sin(frame * …)` bobbing.
- Bars already use `THREE.InstancedMesh`.
- Camera FOV / occlusion HUD is independent of the pursuer command and must stay that way.

## Intended browser-only addition (not TM-E3)

A presentation pursuer `gt-route-track` and target `gt-free-roam` may use:

```text
exact target position
exact target planar velocity
exact target heading
exact obstacle AABB map
arena bounds
```

They remain labelled:

```text
BROWSER GT PREVIEW
NOT PPO · NOT PHYSX · NOT RESEARCH EVIDENCE
```

They are not a reactive-evader research implementation (TM-E3) and must not be
registered as one.

---

# Post-implementation validation — same day, after the preview shipped

The sections above diagnosed the HISTORICAL browser. This section records what the
`gt-route-track` / `gt-free-roam` preview actually did once it was measured over a
density x target-speed x seed matrix, and the four defects that measurement found.

Harness: `tools/validate_gt_browser_tracking.js` (120 runs: densities 70/115/160/205
x target speeds 0.3/0.9/1.5 m/s x 10 seeds x 60 simulated seconds, fixed 10 Hz).
Report: `results/browser_gt_tracking_validation_2026-09-17/logic_validation.json`.

This is browser engineering validation. It is not a PPO measurement, not PhysX, and
not research performance evidence.

## What the first measurement found

The preview passed its own unit tests and still tracked badly. Measured over the
matrix, the tracker was **stalled 47-87 % of every run**, and at several cells the
relative distance at 60 s was LARGER than at spawn (160 bars / 0.3 m/s: 18.13 m ->
18.49 m). One hard invariant was failing outright.

| Defect | Symptom | Root cause |
|---|---|---|
| **D1 unbounded fail-closed stop** | `acceleration_limit_violation` > 0 | `integrateBounded` zeroed the velocity the moment a swept step was unsafe. 2.5 m/s -> 0 in one 0.1 s tick is 25 m/s² against a 4.0 m/s² contract, and the rejected command was re-issued unchanged next tick, so the agent latched into a stop/stutter loop. |
| **D2 uncertified look-ahead chord** | tracker wedged against the safety envelope holding a valid route and a full-speed command | The carrot was a point 1.8 m ALONG the polyline, but the follower drives the STRAIGHT line to it. Where the route doubles back around a bar that chord leaves the certified corridor and points into the obstacle. Measured at 205 bars: command heading 202° against a next-waypoint heading of -69°. |
| **D3 unroutable follow point** | `no_safe_route_fraction` up to 74 % at 205 bars, 5-7.8 replans/s | The tracker's support disc plus tracking margin is larger than the target's, so a gap the target slips through is closed for the tracker. `nearestSafe` gave up instead of snapping the reference to the nearest certified-safe cell. |
| **D4 sealed-pocket spawn** | both aircraft frozen from step 0, `no_path` forever | At the upper densities `navrl_band` merges touching bars into compound walls, sealing pockets. The zero command was CORRECT there — the spawn was not. Connectivity is a spawn precondition, not a controller problem. |

D4 is worth stating plainly: the fail-closed behaviour was right and the scenario was
wrong. Fixing it in the controller would have produced a preview that drives through
walls to look busy.

## Fixes

- **D1** `integrateBounded` now selects among bounded candidates (scales 1, 0.7, 0.45,
  0.25, 0.1, 0) and takes the first whose swept segment is certified. Every candidate
  goes through the same `limitPlanarVelocity` bound, so acceleration and turn rate hold
  whichever is accepted; scale 0 is the maximum-rate bounded brake. An emergency hold
  remains for the case where even the brake would sweep the envelope, and it is counted.
- **D2** `carrotPoint` now returns the FARTHEST look-ahead candidate joined to the pose
  by a certified segment, and `null` when every candidate is blocked — which forces a
  replan instead of a blind chord. `advanceAlongRoute` also advances past waypoints the
  agent overshot, so the follower stops aiming backwards. A new `approachVelocity`
  shapes commanded speed by the turn still owed, so the agent slows into corners rather
  than barrelling through them 15° per tick.
- **D3** `nearestSafe` falls back to `snapToSafe`; the no-route retry is throttled to
  `noRouteRetryS` so an unroutable pose cannot cost one A* per tick.
- **D4** `createSession` resolves a connected spawn from the occupancy grid the A* cache
  already built (`freeComponents`), deterministically relocating an agent out of a
  sealed pocket. It reads cached occupancy and adds no obstacle geometry of its own.

## Result over the same matrix

| | before | after |
|---|---|---|
| hard invariant violations | `acceleration_limit_violation` non-zero | **all 11 categories 0 / 120 runs** |
| stall fraction | 47-87 % | 0.2-1.3 % |
| no-safe-route fraction | up to 74 % | 0.0 % (one cell 5.2 %) |
| median relative distance | 8.7-18.5 m | 1.52-2.13 m |
| final relative distance | 6.5-18.5 m | 1.35-2.17 m (from 14.8-21.4 m at spawn) |
| replans/s | up to 7.8 | 0.16-1.36 |

**Interception episode (2026-09-18 follow-up).** The measurements below were taken in
CONTINUOUS tracking mode, where the standoff is held permanently. The site now defaults
to an INTERCEPTION EPISODE instead: the follow standoff ramps from 1.55 m at the CLOSE
boundary to 0 at the INTERCEPT boundary, the episode ends on the task's swept 0.5 m
capture or its 60 s timeout, holds the outcome ~1.5 s, and resets. A permanent standoff
made capture structurally impossible; that is the conceptual bug this follow-up fixes.
Continuous mode is preserved for the tracking demonstration.

The 1.5 m floor is the **display-only standoff** (`CONTRACT.standoffM = 1.55 m`) that
keeps the two meshes from overlapping in the preview. The documented tracking reference
remains a relative position error of zero; the standoff is a rendering offset, not a
capture radius, and it is not connected to the historical 0.5 m capture semantics.

## What did not change

Actor observation tensors, `aerial_gym` task observation builders, reward, checkpoints
and every recorded research verdict are untouched. `docs/assets/paper/overview-2026-09-13/`
remains hash-pinned and was not regenerated.
