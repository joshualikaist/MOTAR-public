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
