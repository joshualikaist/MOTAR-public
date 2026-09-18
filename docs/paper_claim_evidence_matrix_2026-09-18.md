# Paper claim / evidence matrix — 2026-09-18

Each row pairs a claim axis with the MOTAR number that supports it, the closest published work on
that axis, and what that work reported under **its own** benchmark. The `Benchmark parity` column is
`NO DIRECT MATCH` in every row, which is why the two number columns are never combined.

The `Allowed wording` column is the strongest sentence the evidence supports. Anything stronger is
not licensed by this repository.

MOTAR numbers are rendered from
[`quantitative_positioning_registry.json`](quantitative_positioning_registry.json); external numbers
come from [`literature_quantitative_ledger_2026-09-18.json`](literature_quantitative_ledger_2026-09-18.json).
`tests/test_quantitative_positioning.py` binds each MOTAR value to its result file.

## Matrix

| Claim axis | MOTAR quantitative evidence | Closest external work | External reported number (its own benchmark) | Benchmark parity | Allowed wording | Missing evidence |
|---|---|---|---|---|---|---|
| **Moving-target tracking / close approach** | Capture, crash and timeout rates recorded per cell with receipts; `P2 held-out` **STRICT FAIL** (timeout 5.56 %), `D1 adaptation` **FAIL** | YOPOv2-Tracker | Real forest tracking at a maximum 6 m/s; 8.2 ms perception-to-action onboard. Success rate vs target speed is `NOT_EXTRACTED` (plot only) | NO DIRECT MATCH | MOTAR records close-approach outcomes for a sensor-only policy against a pursuer-independent target. Its own held-out gate is a FAIL and that stands. | No external tracker has been run in MOTAR's contract; YOPOv2-Tracker's code is unreleased |
| **Random obstacle density** | Density sweeps at 70/115/160/205 bars, per-cell receipts, per-outcome strata | NavRL | Success 94.33 → 68.65 % as dynamic obstacles rise 60 → 120, at 350 static, max 2.0 m/s | NO DIRECT MATCH | Both study density as an independent variable. MOTAR sweeps static bar count for a pursuit task; NavRL sweeps dynamic obstacle count for goal navigation. | No shared density definition; "bars" and "obstacles" are not the same unit |
| **Target-motion complexity** | TM-E0…TM-E4 ladder; TM-E2 `IMPLEMENTED; POLICY_COMPARISON_NOT_TESTED`. H/E0/E1/E2 frozen-policy evaluation `RESULT_PENDING` | Elastic Tracker | Cooperative target, position broadcast to chasers, max 2 m/s; escaping targets named as future work | NO DIRECT MATCH | MOTAR's target ladder separates static, constant-velocity and obstacle-aware targets explicitly. The classical trackers evaluate cooperative targets and name escaping targets as future work. | MOTAR's own E0/E1/E2 policy comparison has not produced a verdict yet |
| **Perception-error propagation** | P8 measured error → P9 injection → **−4.57 pp** capture cost on the frozen policy, 95 % CI excludes zero, identical across three training-seed campaigns | NavRL++ | Perception failure is the largest perturbation factor, "more than a 5 % drop" in success across all difficulty levels | NO DIRECT MATCH | Both quantify how perception degradation propagates into policy outcome. MOTAR's injector is driven by a *measured* error distribution from real imagery; NavRL++ perturbs with its own failure model. | No common error model, no common metric, and MOTAR has no live real-image-to-policy connection |
| **Temporal association** | Transformer selector chosen on validation, **+0.00697 utility** over CNN-only; S4 test utility 0.4701 retains its generalization warning | YOPOv2-Tracker / Fast-Tracker | EKF plus spatiotemporal consistency (YOPOv2); EKF/FIFO history with Bézier prediction and 1.82/2.54/3.45 m prediction error under three noise levels (Fast-Tracker) | NO DIRECT MATCH | MOTAR selected a temporal selector on validation utility and did not establish held-out superiority. Published trackers use filter-based prediction with their own error metrics. | MOTAR has no persistent-identity evidence and no prediction-error metric comparable to Fast-Tracker's |
| **Safety-filter geometry** | riskcap vs arc-clearance: crash **−1.4903 pp**, 95 % CI [−1.8981, −1.0826], lower in 15/15 cells. Arc width 0.45→1.2 m at 205 bars: crash **−5.60 pp**, capture **+4.44 pp** | Temporal Barrier | Collisions per 100 s, 0.170 / 0.034 / 0.012 (no CBF / HOCBF / aTTC-CBF) in an obstacle-free domain; under 1.0 ms per call | NO DIRECT MATCH | MOTAR measured how two configured filter geometries change crash rate in dense clutter, with no formal guarantee. Temporal Barrier provides a certificate-style comparator in an obstacle-free setting. | MOTAR's filter is not certified and carries no formal safety guarantee; the C3 explanation is WITHDRAWN |
| **Renderer / visual observation sensitivity** | D8b frozen-policy contrast **−48.967 pp**, 95 % seed-t CI [−50.113, −47.821], `MATERIAL_LOSS`; Renderer Contract v1 frozen | None found | No screened work reports a frozen-policy outcome under a changed observation-rendering treatment | NO DIRECT MATCH | MOTAR measured that a frozen policy scored substantially worse under a mesh-shaded observation treatment. The cause is `NOT_TESTED`. | `causality_vs_d8b = NOT_TESTED`; no renderer result shows that area mismatch, shading or geometry caused the loss |
| **Retraining / readaptation robustness** | P10 replication **INCONCLUSIVE**: mean **+0.73 pp**, 95 % CI [−1.04, +2.50]; residual cost after readaptation −1.90 / −2.71 / −3.84 pp per seed | NavRL++ | Curriculum and perturbation training raise combined success from 63.05 to 94.08 % inside NavRL++'s own framework | NO DIRECT MATCH | MOTAR did not establish a net readaptation benefit and reports that as INCONCLUSIVE. Readaptation never fully recovers the error cost and trades clean performance. | Three training seeds, one density, one arena, one injector |
| **Reproducibility / provenance** | 201 indexed result entries with receipts and source manifests; Record Envelope v2 `producer_unit_validation = PASS`; CPU renderer path installs without a GPU | None found | No screened work publishes per-result receipts, negative-result retention, or a machine-readable result manifest of comparable scope | NO DIRECT MATCH | MOTAR retains preregistrations, receipts, negative outcomes and withdrawn claims alongside positive results. This is a process property, not a performance property. | Hosted CI is unverified; several datasets cannot be redistributed |

## The differentiation sentence, stated in full

MOTAR does not claim a cross-paper performance advantage, and this repository contains no evidence
that would support one.

Its quantitative contribution is a measured chain, where each stage is tied to a recorded effect:

```text
measured perception error          E3-S: 6.2 % median absolute relative range error (3,107 frames)
  -> simulator error injection     P8 distribution, P9 injector
  -> frozen-policy sensitivity     -4.57 pp capture cost, CI excludes zero, 3/3 campaigns
  -> readaptation                  +0.73 pp, CI [-1.04, +2.50], INCONCLUSIVE; residual cost always remains
  -> safety-filter geometry        -1.4903 pp crash, CI [-1.8981, -1.0826], 15/15 cells
  -> observation-rendering shift   -48.967 pp, CI [-50.113, -47.821], MATERIAL_LOSS, causality NOT_TESTED
```

What makes this a contribution is not the size of any single number. It is that the stages are
connected by measurement rather than assumption, and that the chain retains its negative and
inconclusive links instead of reporting only the stages that worked.

## What this matrix does not do

It does not rank MOTAR against any system in the right-hand columns. Those values were produced
under different arenas, sensors, targets, action spaces and success definitions. The matrix places
them side by side to show **which axes the field reports on and under what conditions**, and to make
the empty Class A column visible rather than implicit.
