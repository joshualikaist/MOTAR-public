# Browser interception episode — conceptual-correctness fix, 2026-09-18

```text
BROWSER INTERCEPTION PREVIEW
SIMULATION ONLY · NOT PPO · NOT PHYSX · NOT RESEARCH PERFORMANCE EVIDENCE
Mirrors the research task's termination semantics; not a performance measurement.
```

The GT browser preview showed *continuous following* as its only behaviour, which
misrepresents MOTAR: the research task is detection → tracking → **pursuit → close
approach → interception/capture**. This makes the default demo an interception
episode with real terminal semantics, and keeps continuous tracking as a separate
mode.

## A. Why interception was missing — exact cause

`arena.js:simulationStep` had the GT tracking branch end in an early `return`:

```javascript
if (pursuerDisplayMode === 'gt-route-track') {
    Planner.stepPursuerGt(gtSession, dt);
    ... return;            // <-- returned here
}
const captured = Motion.sweptCapture(..., 0.5);   // never reached in GT mode
if (episode.age >= 30 || captured) resetEpisode(true);
```

The capture / watchdog code lived only on the *historical* path below that
`return`, so the default GT preview never evaluated an outcome. The pursuer
followed forever.

## B. The `return` and the bypass

The bypass was `arena.js:902` (`return;` inside the `gt-route-track` block). It
skipped `sweptCapture` and the `episode.age >= 30` watchdog entirely. Fixed by
restructuring `simulationStep` into `stepTarget → stepPursuer → common outcome
check → terminal transition`: **both** the historical and GT paths now reach one
`evaluateEpisodeOutcome()`; no pursuit path returns before it. A test enforces
that no `return;` sits between the motion steps and the outcome check
(`tests/test_browser_capture_provenance.py`).

## C. How the 1.55 m standoff blocked capture

`predictedFollowPoint` aimed a point `standoffM = 1.55 m` *behind* the target
whenever the pursuer got close, so the aim point was never on the target and
capture at 0.5 m was structurally impossible. The docs already called 1.55 m a
"display-only standoff", not a capture radius — but nothing ever reduced it.

Now the standoff is a **continuous function of distance** in interception mode:
1.55 m at the CLOSE boundary (4.0 m), linearly to 0 at the INTERCEPT boundary
(1.2 m), 0 inside. The aim point reaches the target, so capture is reachable
without any sudden jump toward it (`Planner.desiredStandoff`, monotone; tested).
Continuous mode keeps the constant 1.55 m standoff.

## D. What the hesitation actually is — measured, four sources separated

Over the 120-run continuous matrix (same one used for GT_BROWSER_V1):

| candidate cause | measurement | verdict |
|---|---|---|
| 10 Hz decision clock | 30/60/120 FPS commit identical trajectories (determinism test) | **not a cause** — render rate is independent of motion |
| replanning | 0.64 replans/s mean (up to 5.75/s dense) | event-driven *while moving*; not a stop |
| goal handoff | **pause 0.035 s mean** (max ~1.0 s in a few cells); target dwell < 0.05 m/s = 0.11 s / 60 s | negligible |
| emergency hold | 7.7 ticks / 600 = **1.28%** of steps | dominant non-motion contributor; a safety device |

The visible "thinking", to the extent it exists, is the bounded emergency hold —
the fail-closed brake when no certified candidate exists. That is a safety
guarantee and is **kept**. Goal-handoff pause is already negligible (0.035 s), so
a speculative next-route prefetch would trade the fail-closed guarantee for no
measurable gain; it was **not** implemented, on the evidence. The 10 Hz clock is
not a hesitation source.

## E. State machine

```text
SPAWN → CHASE → CLOSE APPROACH → INTERCEPT → CAPTURED → terminal hold → RESET
                                           ↘ (elapsed ≥ timeout) → TIMEOUT → hold → RESET
historical local-heuristic bar contact       → ABORT → hold → RESET
```

Phases are distance-gated (`Planner.episodePhase`): CHASE ≥ 4.0 m, CLOSE
[1.2, 4.0) m, INTERCEPT < 1.2 m. In INTERCEPT the aim point is a short-horizon
prediction of the target itself, not a following point. Obstacle safety and
bounded dynamics hold in every phase.

## F. Capture / timeout termination

One shared function, `Motion.episodeOutcome`, is called with the same inputs by
the browser's `evaluateEpisodeOutcome` (arena.js) and the planner's
`applyEpisodeOutcome` (headless). Capture is the **swept** test
(`sweptMinDistance < radius`) so a fast closing pass cannot tunnel between two
10 Hz samples. Timeout is the task's episode budget, not a browser watchdog. On
a terminal state both agents freeze, the HUD shows the outcome for ~1.5 s, then
exactly one reset fires.

The capture radius (0.5 m) and timeout (600 steps × 0.1 s = 60 s) are the
**research task's own values**, carried into the page with provenance
(`viewer.js`, comments citing `navrl_task_config.success_radius`, the v2 launcher
budget, and `cfg_rl_step_dt_s`) and bound to the research source by
`tests/test_browser_capture_provenance.py`. They are not new browser constants.

## G. Before / after (120 runs each, continuous mode)

The refactor did not change continuous tracking: **every metric is identical
before and after** — goal transitions, handoff pause, dwell, stall, emergency
holds, replans, route switches — confirming the tracking demo is preserved
byte-for-byte. Both are ALL-ZERO on every hard invariant.

## H. Interception validation (120 runs, interception mode)

```text
episodes terminated   676        captures 676        timeouts 0
time to capture       8.8 s mean (cell average)
closest distance      0.39 m mean (< 0.5 m capture radius, as required)
hard invariants       ALL ZERO across all 120 runs
```

Terminal-correctness invariants, all zero: CAPTURED only when the swept criterion
held; no missed capture; no motion during a terminal state; exactly one reset per
terminal; no stale route/goal/tracker state after reset. Raw report:
`results/browser_interception_v1_2026-09-18/`.

## I. Invariant regression

All eleven GT_BROWSER_V1 hard invariants remain zero across all three matrices
(before-continuous, after-continuous, after-interception): NaN/Inf, target and
tracker obstacle penetration, wall violation, teleport, speed / acceleration /
turn-rate limit, unsafe route segment, command without safe route, blind motion
after route invalidation.

## J. Scope

Browser visualization only. **Zero changes** under `aerial_gym/`, `resources/`,
`configs/`. No PPO, PhysX, checkpoint, reward, controller or research result is
touched or reinterpreted; the 48-cell preregistered evaluation is unaffected.
