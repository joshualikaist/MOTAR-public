# GT_BROWSER_EPISODE_V1 — browser freeze record, 2026-09-18

```text
BROWSER GT PREVIEW
SIMULATION ONLY · NOT PPO · NOT PHYSX · NOT RESEARCH PERFORMANCE EVIDENCE
```

This freezes the browser preview's **episode semantics, termination provenance and
public terminology**. It extends
[`gt_browser_v1_freeze_2026-09-17.md`](gt_browser_v1_freeze_2026-09-17.md), which
froze the tracking contract; that record stands unchanged.

| Gate | State |
|---|---|
| `continuous_tracking_contract` | **FROZEN** |
| `close_approach_episode_contract` | **FROZEN** |
| `termination_provenance` | **FROZEN** |
| `public_terminology` | **FROZEN** |
| `browser_planner_tuning` | **CLOSED** |
| **`GT_BROWSER_EPISODE_V1`** | **FROZEN** |

## What is frozen

**Continuous tracking.** The GT_BROWSER_V1 behaviour: target roams, tracker holds
the 1.55 m display standoff, no terminal outcome, no timeout, no reset. Verified
byte-identical before and after the episode work across all 120-run metrics.

**Close-approach episode (site default).** `TRACKING → CLOSE APPROACH → FINAL
APPROACH → terminal`, standoff a continuous function of distance (1.55 m at the
CLOSE boundary → 0 at the FINAL APPROACH boundary), terminal freeze, ~1.5 s hold,
exactly one reset.

**Termination provenance.** Capture radius and episode budget come from
[`research_task_contract.json`](research_task_contract.json), generated from
research source by `tools/build_research_task_contract.py`. Every consumer — page,
arena, harness, tests — binds to that one file. Nothing re-declares 0.5 / 600 / 0.1.

**Public terminology.** The HUD mapping below is fixed.

## Frozen parameters

Changing any of these now requires a preregistration or a recorded amendment, not
an edit:

```text
standoffM                 1.55 m      display standoff (continuous; CLOSE boundary value)
closeEnterM               4.0 m       TRACKING -> CLOSE APPROACH
interceptEnterM           1.2 m       CLOSE APPROACH -> FINAL APPROACH
interceptHorizonS         0.3 s       final-approach aim prediction
terminalHoldS             1.5 s       terminal banner hold
predictionHorizonMin/Max  0.2 / 0.9 s follow-point prediction
lookAheadM                1.8 m       certified look-ahead
replanPeriodS             0.5 s       replan throttle
noRouteRetryS             0.2 s       no-route retry throttle
A* resolutionM            0.25 m      route grid
trackingMarginM           0.45 m      obstacle inflation
```

Termination values are **not** in this list: they are not browser parameters. They
are whatever the research contract says.

## Terminology mapping — UI only

| internal (unchanged everywhere) | user-facing HUD |
|---|---|
| `CHASE` | TRACKING |
| `CLOSE` | CLOSE APPROACH |
| `INTERCEPT` | FINAL APPROACH |
| `CAPTURED` | APPROACH COMPLETE |
| `TIMEOUT` | TIMEOUT |
| `ABORT` | RESET / ABORT |

Internal state names, `success_radius`, `capture`, and every historical research
record keep their original vocabulary. Only the HUD label changes.
`tests/test_browser_capture_provenance.py` enforces both halves: the labels map,
and the internals are not renamed.

## Static vs dynamic status

```text
docs/status/status.json                  DYNAMIC dashboard snapshot
                                         active run, latest run, historical summaries
docs/status/research_task_contract.json  STATIC browser/research task contract
                                         success_radius, episode budget, RL dt, arena geometry
```

They are deliberately separate. Regenerating the dynamic snapshot to carry three
static fields would churn unrelated live run data; the static contract is
generated from research source and does not move when a run finishes.

Fail behaviour: the page refuses to boot and the harness refuses to run if the
contract is missing or malformed. `arena.js` retains explicitly labelled
development fallbacks for a bare `Arena.init()` with no `configure()`; on the
production path `terminationFromContract` is true and fallbacks are used zero times.

## Engineering metrics are not research results

The browser validation numbers — 120 runs, episodes terminated, time to approach
completion, closest distance — are **browser engineering validation**. They are
recorded in [`interception_episode_2026-09-18.md`](interception_episode_2026-09-18.md)
and are deliberately **not** placed in the site's research result tables. The GT
browser planner uses exact target state and exact obstacle geometry; it is not the
PPO policy, not PhysX, and not research performance evidence.

## After this freeze

Browser planner tuning is closed. Work returns to the frozen-policy H/E0/E1/E2
comparison: `docs/audits/retraining_readiness_audit_2026-09-17.md` and the 48-cell
preflight in `results/target_motion_e0_e2_preflight_2026-09-17/`.
