# Preregistration — physical target two-envelope recovery (`v2`)

Date frozen: 2026-08-25, after route-recovery forensics and before implementation validation.

## Scope and lineage

This is a fresh-only safety-lineage change for the explicit route mode
`global_astar_recovery_v2`, model `physx_ref5in_6dof_global_astar_aabb_v2_two_envelope_recovery`.
It does not alter legacy,
bounded, route-off physical, or the existing `global_astar_v1` transition. Existing checkpoints
are inadmissible; no PPO training is authorized by this document.

The recovery controller is target-side environment dynamics. It is not exposed to the pursuer
observation, reward, policy action, or critic. A missing or non-finite safety input fails closed.

## Frozen geometry

The declared physical box is `[0.28, 0.28, 0.12]` m. Its all-orientation circumscribed XY support
is `S = 0.2068816087` m. A bar's asset-specific XY AABB half-extents are `b_i`.

* Hard envelope: the closed AABBs `b_i + S` and center wall bounds
  `arena + runtime_wall_margin(0.50) + S`. A point or continuous segment touching this set is
  unsafe; the existing `1e-4` m strictness is retained.
* Soft/normal envelope: the closed AABBs `b_i + S + tracking_margin(0.45)` and center wall
  bounds `arena + route_boundary_margin(1.25) + S`. The `0.45` value is unchanged.
* Both envelopes use exact AABB slab/segment tests. The rounded Euclidean local check is not a
  recovery certificate; corner disagreement is recorded as a safety failure. Continuous
  PhysX-substep chord certificates inflate the hard AABB by the central `1e-4` strictness plus
  a derived `0.0123` m reachable-tube reserve. The latter is the upward ceiling of the
  prospective CONNECT RL-step chord, `g*tan(45deg)*0.1^2/8 = 0.0122625` m; it also dominates
  the ten 0.01 s PhysX-substep chord bound and is not a tuning knob.
* Recovery may leave the soft envelope only while the hard envelope remains certified. A
  7x7 nearest-cell search (radius 3 cells at the fixed 0.25 m grid) chooses the nearest anchor
  outside an additional fixed 0.25 m soft hysteresis envelope. The point-to-anchor connector is accepted only after an exact continuous
  hard-envelope segment certificate (including walls). No straight-line, random, or unchecked
  fallback is allowed.

## State machine

`NORMAL -> BRAKE -> CONNECT -> ROUTE` is the only successful transition. `NORMAL` enters
`BRAKE` when the actual target is hard-free but soft-unsafe. Contact, non-finite state, hard
breach, or an absent safety certificate enters a latched fail-closed state and never writes a
position or resets the actor.

* `BRAKE`: submit planar zero velocity. The zero-command rollout must have an exact hard-safe
  swept certificate. Continue until actual XY speed is at most the existing 0.10 m/s stop probe
  threshold and the measured lower-bound braking stop distance fits the hard certificate.
  If zero braking is not certified, test the same-step deterministic escape candidates under the
  fixed dynamics bounds; if none is certified, submit zero and latch `NO_CONNECTOR`.
* `CONNECT`: after braking, search the deterministic 7x7 anchor set. The fixed anchor may require
  multiple RL intervals because the maximum certified connector is 1.237 m while a 0.6 m/s target
  cannot traverse it in one second. Every submitted interval must independently carry a complete
  bounded rollout certificate (fixed `4 m/s²`, `150 deg/s`, `0.1 s` RL step, `1.0 s` lookahead),
  stay hard-safe at every sample, and make nonnegative progress toward the unchanged anchor
  (fixed tolerance `1e-6` m per interval). It is
  not required to reach the soft-free endpoint in one interval. The full anchor segment is exact
  hard-safe, and ROUTE handoff still requires the actual actor to be soft-free beyond hysteresis.
  PhysX position is never assigned by the recovery code.
* `ROUTE`: require soft clearance greater than the hysteresis and a newly planned route whose
  first handoff connector certificate covers the current position. Resume cached waypoint
  following only after that certificate; otherwise remain fail-closed.

The exit hysteresis is one fixed grid cell (`0.25` m), selected from the existing route
resolution; it does not tune `0.45`. Enter at soft clearance `<= 0`; leave CONNECT only when all
soft clearances exceed `0.25` m and the route handoff certificate is valid. The existing 10-step
(1.0 s) ordinary replan cooldown is retained for ordinary route invalidations; recovery has no
unchecked cooldown bypass.

BRAKE timeout is the measured stop-time p95 plus the existing 0.20 s reserve. CONNECT timeout is
derived per environment from the worst point-to-anchor 7x7 diagonal distance
`sqrt(2)*(3+0.5)*0.25` (the nearest cell center may be 0.5 cell away on each axis), the fixed
acceleration ramp `v/4`, worst half-turn `pi/(150 deg/s)`, and the same 0.20 s reserve, divided
by the fixed 0.1 s RL interval and rounded up. These are budgets, not tuning parameters.

## Dynamics and PhysX watchdog

Recovery uses the existing reference limits exactly: `a_ref=4 m/s²`, `omega_ref=150 deg/s`,
speed no greater than the episode target-speed cap, and `10` physics substeps per `0.1 s` command
interval. A candidate is checked as a continuous segment between samples, not only by endpoints.
The fixed 45-degree tilt bound implies the derived horizontal acceleration bound
`g*tan(45°)=9.81 m/s²` for the one-physics-substep reachable tube; motor thrust and mass remain
the declared `9.60 N`, `1.20 kg` values.

A dedicated zero-command PhysX probe must record stop-time and stop-distance quantiles for every
registered target speed. The p05 deceleration and p95 stop time are gate inputs; the stopping
certificate uses the validated monotone speed-indexed p95 stop-distance lookup with a ceiling
speed selection (never interpolation), plus the measured p95 lateral stopping tube. The brake
certificate inflates bars and shrinks walls by that lateral tube in addition to the central hard
reserve. These are measured inputs, not post-hoc tuning, and are
supplied only through a hashed receipt with `schema=navrl_target_recovery_braking_receipt_v1` and
`probe_schema=navrl_target_recovery_braking_probe_v1`. The probe must
show `a_brake_p05 > 0`, finite stop distance, no contact, no invalid OBB, motor saturation
`<=0.15`, and max tilt `<=60°`; otherwise this lineage is blocked. `T_brake_budget` is the
probe p95 stop time plus the existing 0.20 s diagnostic reserve, not a new learned parameter.

## Fail-closed rules and telemetry

NaN/Inf, empty/invalid geometry, hard-envelope breach, contact, watchdog failure, no anchor,
unsafe connector, route-plan failure, or recovery timeout emits zero planar command, increments a
reason-specific counter, and never clamps/teleports/resets the physical target. An unavoidable
subsequent PhysX contact is a normal terminal failure, not a recovery success.
BRAKE timeout and CONNECT timeout are distinct status/reason codes and counters.

Each interval records state and transition, recovery age and fixed timeout budget, signed hard/soft minimum clearances, violating bar or
wall, target position/velocity, command, stop distance/margin, anchor cell/distance, connector
clearance, candidate count/full-horizon/safe-prefix, route status, and no-connector reason.
Each physics substep records hard margin, support, contact force, OBB validity, velocity error,
motor saturation, and tilt. Recovery fields are evaluation telemetry only.

## Acceptance gates (unchanged 32-cell grid)

The existing 32 fresh-process cells remain fixed: seed 827; route arms off/on; bars
70/150/205/300; speeds 0.6/0.9/1.2/1.5 m/s; 32 envs; 300 RL intervals; 40x40x3 m;
`bars_h3`/`navrl_band`; neutral pursuer action; physics `0.01 s x 10`.

Existing controller gates are unchanged: tracking RMSE `<=0.35 m/s`, realized/requested ratio
`>=0.80`, contact `<=0.01`, arm-specific local invalidation `<=0.01`, motor saturation `<=0.15`,
max tilt `<=60°`, finite invalid OBB fraction `0`. Recovery adds safety gates: hard-envelope
breach/contact/invalid during a certified recovery `=0`; every no-connector case emits zero and
no position write/reset; and all state transitions/timeout reasons are present in telemetry.
The route mechanism gates remain unchanged (70-bar plan success `>=0.99`, fallback `<=0.01`,
0.6 m/s goal completions/env `>=0.5`). 300-bar topology remains limited/fragmented and is not
converted into an arena-wide connectivity claim.

## Provenance/checkpoint contract

The source manifest must hash `target_route_planner.py`, `target_motion.py`, `physical_target.py`,
`navrl_task.py`, `navrl_task_config.py`, the recovery launcher, and this preregistration. Checkpoints must record the
recovery schema/model id, both envelope margins, `S`, strict epsilon, hysteresis, fixed dynamics,
braking-probe digest and source digest. Missing or mismatched recovery fields are a hard refusal;
no shape-compatible v1 or older checkpoint may load. This document and implementation are separate
commits; CPU tests are required before any simulator smoke.
