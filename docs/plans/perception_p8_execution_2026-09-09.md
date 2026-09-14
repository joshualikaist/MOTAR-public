# P8 conditional error measurement contract

Status: REGISTERED BEFORE CONDITIONAL MEASUREMENT.

Use frozen P7c v2 validation predictions/candidates and S1 overlap timings only. Hash-check
manifest, candidates, checkpoint, predictions and all three overlap reports/timing streams.
No detector/selector fit or test access. Retain runtime from measured GPU processes alongside
the runtime of this CPU aggregation. Output is a descriptive validation-calibrated model,
not held-out generalization or a claim that a Markov simulator has passed goodness-of-fit.

## Fixed definitions

- Overall metrics retain existing any-GT IoU >= 0.3 semantics, for exact baseline comparison.
- Conditional population: exactly one GT box. Multi-GT/zero-GT frames are counted and excluded
  from conditional distributions and break temporal segments. A multi-GT annotation exists
  in NPS; the older assertion that every video has one UAV does not justify identity metrics.
- Size is sqrt(GT box area) in original image pixels; bins [0,8), [8,16), [16,32),
  [32,64), [64,infinity). No outcome-dependent bin or selected-box size selection.
- States: HIT (selected IoU >= .3), FALSE_LOCK (selected IoU < .3), NO_LOCK (no selection).
  Report 3-state transition counts/probabilities and their selected/no-lock 2-state projection.
  Lost-target runs combine FALSE_LOCK and NO_LOCK; no-lock runs alone are not reacquisition loss.
- Transitions use adjacent eligible observations within one original sequence, attributed to
  the source frame's size bin. Store every delta-t; a transition probability is per observed
  interval, not per camera frame or arbitrary simulator tick. Sequence changes, excluded rows
  and gaps > .5 seconds break segments; changing size bins does not break an episode.
- State/loss burst duration is first timestamp to the next different state's timestamp.
  At a segment boundary duration is a lower bound ending at the last observed timestamp.
  Record left/right censoring, initial size bin, observation count, completed duration samples.
  Reacquisition requires a preceding HIT, then lost states, then another HIT.
- Store joint (du,dv) errors and normalized (du/image_width,dv/image_height) separately for HIT
  and FALSE_LOCK, with empirical summaries at p0/p5/p25/p50/p75/p95/p99/p100. Keep paired samples;
  sampling independent marginal quantiles would discard spatial covariance.
- Store all three S1 latency values per frame, grouped by frame ID. Three timing repeats are
  three hardware measurements, not three independent perception outcomes. Retain resolution
  and sequence summaries. S1 JPEG decoding time is not sensor transport latency.
- Empty distributions/probability rows are null. Bin support requires >=30 frames; each
  transition row requires >=10 outgoing observations. Sparse/empty bins must not silently
  inherit a global distribution. P9 must explicitly handle unsupported size/cadence regions.

## Deliverables and checks

`error_model.json`, per-observation `frames.jsonl`, `bursts.json`, `receipt.json` with hashes,
input scope, exclusion counts, size-conditioned distributions, temporal parameters and raw
sample provenance. Validate count conservation, timestamp/sequence boundaries, positive GT
geometry, probability row sums, censoring, and exact overall baseline metrics. Repeat CPU
aggregation into a fresh directory; model/frames/bursts must have identical SHA-256.

P9 integration and final test evaluation follow separate implementation steps after P8 is
reviewable. Camera angle/range/identity metrics are unavailable from these annotations.
