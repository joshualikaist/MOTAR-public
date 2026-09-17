# How the target moves — target motion algorithms in MOTAR

Canonical answer to "what algorithm does the target use?". Every technical claim
below is bound to a source file and function named in §12. This document records
existing behaviour; it introduces no new result and changes no verdict.

---

## 1. Executive explanation

The target in MOTAR is not a point that ignores obstacles and flies in a straight
line. Depending on the research setting it can be static, move at constant
velocity, chase random waypoints, or steer around obstacles under acceleration
and turn-rate limits.

In the 3-D demo on the site, the target knows the exact positions of the bars.
It chooses a destination it can actually reach, plans a route that never passes
through an obstacle, and then follows that route with limits on speed,
acceleration and heading change. When it arrives it picks a new destination and
keeps going. If no safe route exists it stops and replans rather than clipping
through a bar.

That browser motion is an **explanatory GT preview**. It is not the PPO policy's
input, not a PhysX rollout, and not a performance result.

---

## 2. Which target algorithm are we talking about?

This is the distinction that matters most, and the one most easily got wrong.
**MOTAR does not have one target algorithm.** It has four lineages, and they do
not share a planner.

| Lineage | Global route | Local step | Obstacle knowledge | Pursuer-reactive |
|---|---|---|---|---|
| **A. Historical legacy** | none | full-speed heading candidates, 90° continuity preference | GT bar centres, centre-distance clearance | no |
| **B. TM-E2 obstacle-aware (bounded)** | **none** | receding-horizon candidate rollout, first step only | GT bar geometry | no |
| **C. Physical lineage** | **optional** A* (`NAVRL_TARGET_ROUTE_MODE`, default `off`) | braking-aware route step into a PhysX actor | GT bar AABBs | no |
| **D. Browser GT free-roam** | **yes** — reachable-set flood or A* | bounded candidate follower | exact browser AABBs | no |

```text
                          Target motion
                                |
            +-------------------+-------------------+
            |                                       |
    Research simulator                        Browser preview
            |                                       |
   TM-E0  static                          reachable goal selection
   TM-E1  constant velocity                          |
   TM-E2  local receding horizon           global route  (A*)
            |                                       |
   no global route                          bounded follower
   (A* only in the optional                          |
    physical route mode)                   explanatory visualization
```

Two consequences follow, and both are load-bearing:

> **TM-E2 does not use A\*.** It is a *local* receding-horizon planner. Saying
> "the MOTAR target uses A\*" is wrong for the research target under its default
> and its TM-E2 configuration.

> **A\* appears in the research simulator only in the physical lineage**, only
> when `NAVRL_TARGET_ROUTE_MODE` is explicitly turned on (it defaults to `off`),
> and that mode is refused for virtual/bounded targets. Its route mechanism gate
> is recorded as `FAIL_ROUTE_MECHANISM`.

The browser preview and TM-E2 are **different algorithms**. The browser mirrors
the *geometry contract* of the physical lineage's `global_astar_v1`, not TM-E2.

---

## 3. 30-second explanation

For the browser demo on the site:

1. The target picks a place to go.
2. It finds a route there that avoids the obstacles.
3. It follows the route without changing speed or direction abruptly.
4. When it arrives, it picks a new place.
5. If there is no safe route, it stops and replans instead of passing through a bar.

```text
goal selection  ->  route planning  ->  bounded following  ->  goal reached  ->  next goal
```

---

## 4. Two-minute explanation

The browser target is a **hierarchical motion generator**: a global layer decides
*where* to go, a local layer decides *how far it may actually move this tick*.

```text
        Reachable goal selection
                    |
        Obstacle inflation (configuration space)
                    |
        Global route  (reachable-set flood, or A* for a fixed goal)
                    |
        Route simplification (farthest-visible shortcut)
                    |
        Bounded local tracking (speed / accel / turn-rate)
                    |
        Position and velocity update
                    |
              Goal reached?
               |          |
              No         Yes
               |          |
            continue   next goal
```

**Reachable goal selection.** Candidate destinations are not arbitrary
coordinates. They must be inside the workspace, outside the inflated obstacles,
and connected to the target's current cell.

**Obstacle inflation.** The target is not treated as a point. Each bar is grown
by the target's own support radius plus a safety margin, so a route that keeps
the target's *centre* clear also keeps its *body* clear.

**Global route.** Over that inflated map the planner produces an ordered list of
waypoints that never crosses an obstacle.

**Route simplification.** The raw grid path is staircase-shaped. Redundant
intermediate points are dropped whenever a straight segment between two
non-adjacent points is certified safe.

**Bounded local tracking.** The follower converts the route into one bounded
velocity command per tick, under speed, acceleration and heading-rate limits.

**Update and repeat.** On arrival a new destination is selected and the loop
continues. There is no capture-and-reset in the GT preview; following is
continuous.

---

## 5. Browser GT free-roam, precisely

Source: `docs/status/arena_demo_planner.js`, `arena_route.js`, `arena_motion.js`.

### 5.1 Goal selection

The browser has **two** goal paths, and they use **different planners**:

**AUTO ROAM** (`Route.planToConnectedGoal`) does *not* sample a random point and
then plan to it. It runs a **uniform-cost (Dijkstra) flood** from the target's
current cell across the free grid, then selects a destination from the cells the
flood actually reached. A cell is a candidate when all of the following hold:

```text
reachable from the current cell   (finite flood distance)
at least  minGoalDistanceM        from the current position
farther than goalExclusionRadiusM from the previous goal
```

A seeded selector picks one candidate; the route comes back from the flood's
parent chain, so **reachability is guaranteed by construction rather than by
rejection sampling**. If the candidate set is empty the planner returns
`no_connected_goal` and the target holds position.

**CLICK / TAP GOAL** classifies the clicked point first
(`out_of_bounds` / `inside_bar` / `unsafe` / `ok`) and rejects it with HUD
feedback unless it is `ok`. Only then does it run **A\*** (`Route.plan`) to that
fixed destination.

The tracking UAV's own route to its follow point also uses **A\***.

### 5.2 Obstacle inflation

`inflatedGeometry` grows every bar by

```text
half_extent = 0.5 * bar_width + agent_support + trackingMarginM
```

and shrinks the admissible workspace by `boundaryMarginM + agent_support`. The
target's support radius is the conservative circumradius of its physical box
(`conservativeXYSupportFromBox`); the tracking UAV uses its own, larger radius —
which is why a gap the target can slip through may be closed for the tracker.
Exact values are in §9.

### 5.3 A\*

```text
state        grid cell at resolutionM
cost         travel distance, 1 per orthogonal step, sqrt(2) per diagonal
heuristic    Euclidean distance to the goal cell, in cell units
neighbours   8-connected
corner rule  a diagonal move is rejected unless BOTH orthogonal neighbours are free
collision    inflated obstacle geometry, exact closed-AABB slab test
output       ordered waypoints
limits       maxExpansions, maxWaypoints; exceeding either fails closed
```

The corner rule matters: without it an 8-connected path can slip diagonally
between two touching bars that a real body cannot pass.

### 5.4 Route simplification — shortcut, not spline

Greedy **farthest-visible shortcut** (`smoothRaw`). From each anchor the
algorithm tries the farthest remaining point first and walks backwards until a
segment passes the exact closed-AABB test, then makes that point the next
anchor. Every accepted edge is re-verified continuously.

This is **waypoint reduction, not curve fitting.** There is no spline, no
B-spline, no polynomial smoothing anywhere in the route. Waypoints are removed;
none is invented.

### 5.5 Bounded following

The planner says where; the follower says how far.

The follower picks a look-ahead point on the route (`carrotPoint`) and takes the
**farthest candidate reachable from the current pose by a certified-safe
segment**. A look-ahead point along the polyline is not enough: the follower
drives the straight chord to it, and on a route that doubles back around a bar
that chord can leave the corridor. If no candidate is reachable, the follower
returns nothing and forces a replan rather than driving a blocked chord.

The commanded speed is then shaped by the turn still owed (`approachVelocity`),
so the target slows into corners instead of carrying full speed through them,
and passed through the bounded step selector (`integrateBounded`), which tries
progressively slower versions of the request and takes the first whose swept
segment is certified. Every candidate passes the **same** velocity bound, so the
speed, acceleration and turn-rate contract holds whichever is accepted.

### 5.6 Fail-closed behaviour

```text
no certified candidate this tick  ->  bounded brake, hold the certified pose
no valid route                    ->  zero command, throttled replan
look-ahead blocked                ->  immediate replan, no blind chord
```

The target never passes through a bar, never teleports, and never takes an
unsafe shortcut to look busy. Across the 120-run validation matrix this was
measured, not assumed: obstacle penetration, wall violation, teleport and
"non-zero command when no safe route exists" were all **zero**
(`docs/status/gt_browser_v1_freeze_2026-09-17.md`).

### 5.7 Spawn connectivity

At high density the bar layout merges touching bars into compound walls, which
can seal off pockets. A spawn inside a sealed pocket is genuinely unroutable —
the zero command would be *correct* and the scenario *wrong*. Both agents are
therefore required at session creation to start in a free-space component large
enough to roam, and are deterministically relocated otherwise. The controller is
never coerced into passing through an obstacle to escape.

---

## 6. Research TM-E0 / TM-E1 / TM-E2

Source: `aerial_gym/task/navrl_task/target_motion.py`,
selected by `NAVRL_TARGET_BEHAVIOR_LEVEL` (default `historical`).

**TM-E0 `e0_static`** — speed zero. A baseline, not a motion algorithm.

**TM-E1 `e1_cv`** — constant-velocity nominal reference. Honest caveat: CV
targets **bounce off bars** (the held heading is reflected off the composite
push normal), and there is a declared **evaluation-only** intervention
(`NAVRL_EVAL_CV_INITIAL_HEADING`, default `random`) that can set the *initial*
heading relative to the pursuer (`away`, `toward`, `tangent_left`,
`tangent_right`). That is an initial condition, not reactive evasion — the target
never updates its heading in response to where the pursuer goes afterwards.

**TM-E2 `e2_obstacle_aware`** — the bounded waypoint executor
(`bounded_drone_target_step`). This is **local receding-horizon**, not global
routing:

```text
nominal waypoint direction
        |
candidate heading offsets  x  candidate cruise-speed scales  +  one stop command
        |
roll each candidate forward over the lookahead under the SAME accel / turn bounds
        |
feasibility per rollout sample: inside the workspace AND clear of the bars
        |
score, then execute the FIRST step only
        |
replan next RL step
```

Selection has two regimes. When a candidate survives the whole horizon it is
scored on cruise scale and turn cost, so the target prefers to keep speed and
turn as little as possible. When **no** candidate survives, the target does not
pretend it succeeded: it falls back to the candidate with the longest safe
prefix, breaking ties on clearance and boundary margin, and exposes an
infeasible flag to telemetry. A trapped rollout is reported as trapped.

TM-E2 carries **no persistent waypoint list, no grid, and no A\***. It cannot see
around a corner: its horizon is one lookahead long. That is a real difference in
capability from the browser preview, not a presentational one.

TM-E3 `e3_reactive` and TM-E4 `e4_learned` raise `NotImplementedError`. They fail
closed rather than silently substituting scripted motion.

---

## 7. Historical legacy and physical lineages

**Legacy** (`steer_target_step`) — full-speed heading candidates with a 90°
heading-continuity preference; if no candidate is clear the least-bad endpoint is
returned for the caller's projection fallback. Wall reflection and post-step bar
push-out exist in this lineage. There is **no acceleration or turn-rate bound**:
the episode speed is applied immediately. This is a historical kinematic
contract, not physical target evidence.

**Physical** — a bounded planner supplies a velocity reference to a PhysX target
actor with motor lag and tilt/thrust limits. This is the only research lineage
that may enable the global A* route
(`NAVRL_TARGET_ROUTE_MODE` ∈ `global_astar_v1`, `global_astar_recovery_v2`,
`global_astar_braking_v3`; default `off`), and the mode is **refused** for
virtual/bounded targets and for non-waypoint patterns. Implementation exists, but
the preserved route/recovery evaluations failed their preregistered mechanism
gates: the recorded verdict is `FAIL_ROUTE_MECHANISM` and stays that way.
Implementation is not a passing benchmark.

---

## 8. GT information boundary

This is the question behind most of the others, and the answer has two halves
that must not be merged.

```text
          TARGET SIDE                          PURSUER / POLICY SIDE
   ------------------------------       ---------------------------------
   obstacle geometry  (GT)              camera + LiDAR derived observation
   arena bounds       (GT)              declared ego / history inputs
            |                                        |
            v                                        v
   benchmark motion generator            PPO actor observation
            |                                        |
            +--------- NEVER CROSSES ----------------+
```

The target may use privileged simulator geometry **because it is a benchmark
motion generator, not a perception agent.** The research question is the
*pursuer's* perception and navigation behaviour; if the target walked into bars
or teleported, the benchmark itself would be corrupted.

`resolve_target_behavior_contract` records this explicitly per level:
`uses_privileged_obstacle_gt` is true only for TM-E2, and
`uses_privileged_pursuer_gt` is **false for every level**.

The browser GT tracking preview is a separate explanatory layer under the same
rule. `tests/test_browser_gt_state_not_in_policy_observation.py` enforces that
browser GT fields never appear in an actor observation or a historical policy
input.

> **GT target generator ≠ policy privilege.**

---

## 9. Constraints and values

Motion bounds enforced by the follower (browser) and the rollout (TM-E2):

```text
||v_t||                          <=  v_max
||v_t - v_{t-1}||                <=  a_max * dt
|wrap(psi_t - psi_{t-1})|        <=  omega_max * dt
```

Route safety condition:

```text
segment(w_i, w_{i+1})  intersect  inflated_obstacles  =  empty      for every i
```

Read from source at the time of writing:

| Quantity | Value | Source |
|---|---|---|
| browser simulation clock | 10 Hz fixed | `arena.js` |
| grid resolution | 0.25 m | `arena_route.js` `CONTRACT.resolutionM` |
| tracking margin | 0.45 m | `CONTRACT.trackingMarginM` |
| boundary margin | 1.25 m | `CONTRACT.boundaryMarginM` |
| max A* expansions | 50 000 | `CONTRACT.maxExpansions` |
| max waypoints | 128 | `CONTRACT.maxWaypoints` |
| min goal distance | 6.0 m | `CONTRACT.minGoalDistanceM` |
| goal exclusion radius | 1.0 m | `CONTRACT.goalExclusionRadiusM` |
| target physical box | 0.28 x 0.28 x 0.12 m | `CONTRACT.physicalBoxXYZ` |
| tracking UAV radius | 0.25 m | `arena_motion.js` `pursuerRadius` |
| max acceleration | 4.0 m/s² | `boundedMaxAccel` |
| max heading rate | 150 °/s | `boundedMaxTurnRate` |
| target speed range | 0.0 – 1.5 m/s | `targetSpeedMin` / `targetSpeedMax` |
| tracker speed cap | 2.5 m/s | `pursuerSpeedMax` |
| browser look-ahead | 1.8 m | `arena_demo_planner.js` `lookAheadM` |
| TM-E2 heading offsets | 24, symmetric 0°…180° | `BOUNDED_TURN_ANGLES_DEG` |
| TM-E2 cruise scales | 1.0, 0.5, 0.25, plus one stop | `bounded_drone_target_step` |
| legacy heading offsets | 10, symmetric 0°…180° | `TURN_ANGLES_DEG` |
| legacy continuity window | 90° | `HEADING_CONTINUITY_RAD` |

TM-E2's lookahead, accel and turn-rate limits are per-run configuration rather
than literals in the motion module; they are recorded in each run's receipt and
must be read from there, not assumed.

---

## 10. Pseudocode — browser GT free-roam

```text
while simulation_running:

    if goal is missing or reached:
        # AUTO ROAM: flood for reachable cells, choose among them, route comes
        # back with the goal. CLICK GOAL: classify the click, then A* to it.
        goal, route = select_reachable_goal_and_route(position)

    if route is missing or its next leg is no longer certified:
        route = replan(position, goal)

    if route does not exist:
        command = zero_velocity          # fail closed, do not clip through bars
        schedule_throttled_replan()
        continue

    lookahead = farthest_route_point_reachable_by_a_certified_segment(position, route)

    if lookahead is None:
        replan_now()                     # never drive a blocked chord
        continue

    desired = direction_to(lookahead) * speed_shaped_by_remaining_turn()

    command = first_candidate_whose_swept_segment_is_certified(
        current_velocity, desired,
        scales = [1.0, 0.7, 0.45, 0.25, 0.1, 0.0],   # 0.0 = bounded brake
        bounds = (speed_limit, max_accel, max_turn_rate)
    )

    move(command)
```

## 10b. Pseudocode — research TM-E2

Deliberately shown separately, because it is a different algorithm.

```text
every RL step:

    desired = direction_towards_current_waypoint * episode_speed

    candidates = [ rotate(desired, a) * s
                   for a in heading_offsets
                   for s in cruise_scales ] + [ stop ]

    for each candidate:
        roll forward over the lookahead under the SAME accel / turn bounds
        feasible = every rollout sample is inside the workspace and clear of bars
        record safe_prefix_length, min_clearance, boundary_margin

    if any candidate is feasible for the whole horizon:
        pick the one with the best cruise scale and least turn
    else:
        pick the longest safe prefix, tie-break on clearance
        report infeasible to telemetry

    execute the FIRST step of the chosen candidate only
    # no route is stored; the whole thing is recomputed next RL step
```

---

## 11. Known limitations

**Browser GT free-roam**

```text
uses GT obstacle geometry          2-D path planning only
presentation-grade planner         not adversarial
not learned                        not pursuer-reactive
not PhysX                          not PPO
not a performance measurement
```

**TM-E2**

```text
scripted, not learned              not pursuer-reactive
local horizon only, no global route
cannot plan around a corner beyond its lookahead
2-D kinematic: no rigid-body attitude, motors or contact
policy-performance comparison NOT TESTED
```

**Physical lineage**

```text
implementation exists
route/recovery mechanism gate FAILED  (FAIL_ROUTE_MECHANISM)
```

**All lineages**

```text
no target reacts to the pursuer's position after reset
TM-E3 (reactive) and TM-E4 (learned) are PLANNED and fail closed
```

---

## 12. Source files

Research simulator:

```text
aerial_gym/task/navrl_task/target_motion.py
    resolve_target_behavior_contract   behaviour-level contract, GT flags
    limit_planar_velocity              speed / accel / turn-rate bound
    bounded_drone_target_step          TM-E2 receding-horizon executor
    braking_aware_route_step           physical routed step
    steer_target_step                  historical legacy step
    initial_cv_velocity                evaluation-only CV initial heading

aerial_gym/task/navrl_task/target_route_planner.py
    _NEIGHBORS                         8-connected neighbourhood
    plan                               A* with Euclidean heuristic
    plan_to_connected_goal             reachable-goal flood
    (greedy farthest-visible smoothing inside plan)

aerial_gym/task/navrl_task/navrl_task.py
    target dynamics and route-mode gating, target advance
```

Browser preview:

```text
docs/status/arena_route.js             geometry contract, A*, flood, smoothing
docs/status/arena_motion.js            fixed clock, limitPlanarVelocity, limits
docs/status/arena_demo_planner.js      goal logic, look-ahead, bounded follower
docs/status/arena.js                   scene, HUD, controls
```

Related documents:

```text
docs/target_behavior_ladder_2026-09-16.md            TM-E0..E4 ladder and status
docs/status/gt_browser_v1_freeze_2026-09-17.md       measured browser validation
docs/status/arena_motion_audit_2026-09-17.md         browser motion audit
docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md
```

---

## 13. FAQ

**Q1. What algorithm does the target use?**
There is no single one — see §2. In the browser demo: pick a reachable
destination, plan an obstacle-free route, follow it under speed/acceleration/
turn limits, pick a new destination on arrival. In the research simulator the
default is the historical lineage; TM-E2 is a local receding-horizon obstacle-
aware executor.

**Q2. Does it use A\*?**
The browser preview does, for a fixed destination (click/tap goal, and the
tracking UAV's own route). Its auto-roam uses a reachable-set flood instead and
takes the route from that flood. In the research simulator, **TM-E2 does not use
A\***; A* exists only in the physical lineage's optional route mode, which
defaults to `off`.

**Q3. Does the target run away from the pursuer?**
No. No implemented target behaviour reacts to the pursuer's position. The
browser free-roam and research TM-E2 are both *obstacle-aware but
pursuer-independent*. **Obstacle-aware ≠ adversarial.** A declared
evaluation-only intervention can set a CV target's *initial* heading relative to
the pursuer, but nothing updates in response to the pursuer afterwards. Reactive
(TM-E3) and learned/self-play (TM-E4) evaders are PLANNED and fail closed when
selected.

**Q4. Does the target know where the obstacles are?**
Yes, in the browser preview and in TM-E2, by design — see §8.

**Q5. So does the pursuer get GT too?**
No. The target's privileged geometry is never passed to the PPO actor
observation. `uses_privileged_pursuer_gt` is false for every behaviour level, and
a test enforces that browser GT fields never reach a policy input.

**Q6. How is the destination chosen?**
A candidate must be inside the workspace, outside the inflated obstacles,
connected to the current cell, at least a minimum distance away, and outside an
exclusion radius around the previous goal. In auto-roam, reachability is
guaranteed by construction because candidates come from a flood that already
reached them.

**Q7. What if there is no route?**
Zero command and replan. The target holds position; it does not pass through a
bar to keep moving.

**Q8. Can the target pass through obstacles?**
No. Measured across 120 runs × 60 s: zero obstacle penetrations, zero wall
violations, zero teleports.

**Q9. Is the target's speed constant?**
Not in the browser preview — it has a desired cruise speed but real speed is
shaped by the bounded follower, and it slows into corners. Historical CV is a
constant-velocity nominal law with reflection behaviour at bars. TM-E2 selects
among cruise-speed scales including a stop.

**Q10. Is the site's target motion the same as the research target?**
No. See the table in §2 and the comparison in §14. They differ in the planner,
not only in presentation.

**Q11. What is TM-E2?**
The obstacle-aware scripted target level: a bounded local receding-horizon
executor that uses privileged obstacle geometry, is implemented, and has **not**
been policy-compared.

**Q12. Are TM-E3 / TM-E4 implemented?**
No. Both are PLANNED and raise an error if selected, rather than quietly
behaving like a scripted target.

**Q13. Why make target motion this complicated?**
A random walk is simpler but degrades as obstacle density rises: it gets stuck
against walls and needs unrealistic reflection or teleport corrections, which
corrupt the benchmark. Separating *where to go* from *how to move* keeps the
target moving continuously while remaining physically plausible and
path-feasible.

**Q14. Can the same motion be reproduced?**
Yes for the browser preview: the bar layout, goal selection and motion are
seeded, and identical seed and configuration replay identically. This was
verified to be independent of render frame rate — 30, 60 and 120 FPS commit the
same trajectory.

**Q15. Doesn't an A\* target make things easier for the PPO agent?**
The A\* is how the *target* generates its own trajectory; it does not hand the
pursuer a solution path to the target. The browser A\* is a visualization motion
generator and is not PPO performance evidence at all. And the research TM-E2
target does not use A\* in the first place.

---

## 14. Browser GT free-roam vs research TM-E2

|  | Browser GT free-roam | Research TM-E2 |
|---|---|---|
| Purpose | visualization / explanation | simulator target behaviour |
| Obstacle knowledge | exact browser AABBs | simulator GT bar geometry |
| Global route | **yes** — A* for a fixed goal, reachable-set flood for auto-roam | **no** — none |
| Planning horizon | whole route to the goal | one lookahead, recomputed each RL step |
| Route storage | persistent waypoint list | none |
| Local bounded motion | yes | yes |
| Pursuer-reactive | no | no |
| Changes PPO input | no | no |
| Scientific result | no | implementation only |
| Policy comparison | no | **NOT TESTED** |

The two share the *idea* of obstacle-aware bounded motion and the same velocity
bound. They do not share a planner, and they must not be described as one
algorithm.
