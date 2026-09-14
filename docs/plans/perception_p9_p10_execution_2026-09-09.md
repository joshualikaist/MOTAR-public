# P9 empirical injector and P10 PPO execution contract

Status: **REGISTERED BEFORE P9 GOODNESS-OF-FIT, TEST OPENING, TRAINING, OR PPO EVALUATION**.

Amendment P9-v2, registered after v1 failed and before its rerun: v1 passed every transition,
offset, latency and seed gate but bin-1 occupancy TV was `0.06048 > 0.05`. Inspection found that
the implementation discarded the observed destination size bin. In bin 1 the within-bin Markov
stationary distribution is approximately `(0.627,0.240,0.133)`, close to observed
`(0.635,0.229,0.137)`, while the destination-pooled row implies `(0.697,0.199,0.104)`.
P9-v2 therefore conditions a transition on **source bin, source state, and destination bin** when
that cell has at least ten observations; otherwise it uses the already-registered nearest
supported source-bin/source-state row. This does not break the sequence at a size change and does
not alter any gate, threshold, bin or P8 input. The failed v1 artifact remains archived.

Amendment P9-v3, registered after v2 failed and before its rerun: destination conditioning reduced
bin-1 occupancy TV to `0.05495` but did not pass `0.05`. The remaining discrepancy is a
finite-sample inconsistency between each raw transition MLE and the separately observed state
occupancy. P9-v3 applies iterative proportional fitting to the empirical transition-flow matrix,
constraining both flow marginals to the measured occupancy and then row-normalizing. This is the
minimum-KL stationary reconciliation, not a threshold change. The largest resulting probability
adjustment is recorded and must still pass the unchanged raw-transition error gate `<=0.06`.
The balanced source-bin row supersedes destination conditioning at runtime because the deployed
simulator can remain in one mapped size bin for long periods. Failed v1/v2 artifacts remain
archived; no sealed test data have been opened.

P9 consumes only the frozen P8 validation artifacts in
`results/perception_p8_2026-09-09/`. P10 changes only the perception-error arm and PPO
checkpoint; reward, arena, target motion, observation schema, tracker, LiDAR association,
speed governor and exact-600 outcome contract stay fixed.

## P9 model and transfer rules

- Condition on `sqrt(GT bbox area)` bins `[0,8), [8,16), [16,32), [32,64), [64,inf)`.
  P8-supported bins are 1, 2 and 3. Simulator bins 0 and 4 map to nearest supported bins 1
  and 3. A state-transition row that lacks ten outgoing observations maps to the nearest
  supported bin for that same source state (bin-3 `NO_LOCK` maps to bin 2). Every fallback is
  serialized; there is no pooled/global silent fallback.
- Sample the three states `HIT`, `FALSE_LOCK`, `NO_LOCK`. Initial state is drawn from the
  mapped bin's empirical occupancy. Later states use the occupancy-balanced source-bin/source-state
  row. Its raw empirical row and the balancing adjustment are both serialized and validated
  row. Convert its observed-interval self probability to the 0.1 s simulator cadence with
  `p_stay(dt) = p_stay(ref_dt) ** (dt/ref_dt)`; divide exit mass among the empirical off-diagonal
  probabilities. `ref_dt` is that row's median raw transition interval.
- Sample paired `(du/width,dv/height)` values, never independent marginals, for `HIT` and
  `FALSE_LOCK`. Apply them to the clean analytic centroid at the detector resolution. The clean
  renderer remains the geometric support: when it has no target pixels, P9 does not invent a
  background proposal. `NO_LOCK` suppresses the measurement. `FALSE_LOCK` clears the target
  mask used by obstacle-map carving.
- P8 has neither metric range truth nor false-lock range truth. P9 therefore keeps the clean
  analytic surface range and confidence in both selected states. This is a pixel/temporal
  P9-lite model, not a metric-range or confidence simulator.
- Draw latency from the three raw P8 timing repeats in the mapped size bin. Convert milliseconds
  to observation steps by nearest integer with half-up rounding. At the fixed 0.1 s policy step,
  sub-50 ms samples impose zero whole-step delay; longer samples use the existing timestamped
  latency ring. JPEG decode is included because P8 measured end-to-end file-pipeline latency;
  it is not claimed to be sensor transport latency.
- P9 is mutually exclusive with the older synthetic detector-noise, image-noise, dropout,
  range-bias and learned-render detector arms. The model path and SHA-256 are mandatory and are
  written into simulator/evaluator receipts.

## P9 validation gates

Use the original eligible P8 size/cadence stream and 64 fixed-seed Monte Carlo replicas. Report
per-supported-bin state occupancy, every supported transition row, paired normalized offsets and
latency. These gates are fixed before seeing generated results:

- state-occupancy total variation distance: <= 0.05 per supported bin;
- supported transition-row maximum absolute probability error: <= 0.06;
- two-sample KS distance for `du_norm`, `dv_norm` and paired radial normalized error: <= 0.10;
- two-sample KS distance for latency: <= 0.05;
- same seed/input trace is byte-identical; a different seed changes the trace;
- clean arm with P9 unset preserves existing behavior and all tests.

Failure freezes P10. Thresholds, bins and fallbacks are not changed after results.

## Sealed test opening

After P9 passes, open the P7c-v2 NPS test exactly once. Before opening, report the already-fixed
P7c metrics: frame hit, selected false-lock rate, no-lock rate, utility, reacquisition count/mean/
median/p95/max, and source-video count. Do not tune selector, checkpoint, threshold, history or
P9 from this result. Record `test_used: true` only in this final-test receipt; prior receipts stay
historical.

## P10 adaptation and held-out evaluation

Warm-start from frozen ep25000/205-bar checkpoint SHA-256
`f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40` and adapt one explicitly
reported training seed for 1,000 additional epochs at fixed 205 bars with P9 enabled. This is a
new perception lineage despite shape-compatible 898-D observations. Keep riskcap, optimizer
safety, target distribution, arena, reward and all non-perception task fields equal to the source
checkpoint. Training seed must not appear in evaluation seeds.

Evaluate 2 previously unused fixed seeds, deterministic actions, 2,049 requested episodes per
cell, 205 bars and exact-600 semantics:

1. source PPO, clean;
2. source PPO, P9;
3. adapted PPO, clean;
4. adapted PPO, P9.

Primary estimand: pooled `adapted-P9 minus source-P9` capture percentage points with a two-sided
normal-approximation 95% CI. Secondary: crash and timeout differences, plus clean retention
(`adapted-clean minus source-clean`). Report counts/rates, actual episodes, one training seed and
two evaluation seeds. This is descriptive with one training seed; it is not a seed-general PPO
claim and has no post-hoc pass margin.
