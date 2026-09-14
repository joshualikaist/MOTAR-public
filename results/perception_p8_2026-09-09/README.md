# P8 conditional error model

Status: **MEASUREMENT COMPLETE — LIMITED SINGLE-GT SUPPORT**. P9 distribution injection and
goodness-of-fit validation are next. This step did not read a test split or run policy evaluation.

Frozen P7c v2 predictions and all three S1 overlap timing streams were hash-verified before
measurement. Rules were committed before conditional counts were measured in
[the P8 contract](../../docs/plans/perception_p8_execution_2026-09-09.md).

## Population and results

The 2,296 validation frames include **986 multi-GT frames** (466 with two boxes, 271 with three,
140 with four, 109 with five). Older documentation saying every NPS video has only one UAV was
incorrect. These boxes do not have stable target identities. Conditional calibration therefore
uses the **1,310 exactly-one-GT frames**, with exclusions breaking temporal segments.
Overall any-GT baseline metrics still use all 2,296 frames and are reproduced exactly.

Size is sqrt(GT area) in original pixels, with fixed boundaries; it never comes from the selected
candidate. A bin needs 30 frames and each transition row needs 10 observations to be supported.

| Size px | Frames | HIT | FALSE_LOCK | NO_LOCK | HIT center error mean / P95 px |
|---|---:|---:|---:|---:|---:|
| <8 | 0 | 0 | 0 | 0 | unavailable |
| 8–16 | 271 | 172 | 62 | 37 | 0.990 / 2.281 |
| 16–32 | 849 | 648 | 110 | 91 | 2.698 / 7.345 |
| 32–64 | 188 | 145 | 37 | 6 | 3.843 / 7.481 |
| >=64 | 2 | 0 | 2 | 0 | unavailable |

The first and last bins are unsupported. In 32–64 px the NO_LOCK transition row has only five
outgoing observations and is also unsupported. The file retains measurements but does not
replace sparse rows with global estimates. Larger-box absolute error does not establish a size
causal effect: bins also differ in sequence, resolution and scene composition.

The single-GT observations form **27 segments** and **1,283 transitions**. Empirical intervals
range from about **50 to 103 ms**. The transition matrix is per observation interval, not a
30 Hz sensor frame or simulator tick. NO_LOCK means no selected candidate; FALSE_LOCK means
a selected box below IoU .3. A target-loss run can contain both states.

Conditional segments contain 65 completed reacquisitions (mean 0.221 s, max 0.801 s).
This is a different population from the full baseline's 102 reacquisitions (mean 0.470 s,
max 19.933 s): multi-GT exclusions and segment ends censor events. P9 must not interpret the
shorter conditional maximum as proof that long losses no longer occur. All censored runs are
retained with lower-bound duration and flags.

## Artifacts

- [error_model.json](error_model.json): fixed size bins, state counts, empirical joint pixel and
  normalized center-error samples, transition counts/probabilities with interval samples,
  censored-run summaries, three-repeat latency distributions, overall any-GT metrics.
- [frames.jsonl](frames.jsonl): one row per original validation observation, including excluded
  frames, state, geometry, offsets and all three latency repeats. Outcomes are not triplicated.
- [bursts.json](bursts.json): state and target-loss runs, initial size bin, censoring flags,
  timestamps-derived duration and reacquisition indicator.
- [receipt.json](receipt.json): input/output SHA-256, source/contract hash, aggregation runtime
  and actual runtimes from each GPU measurement process. External inputs retain their paths.

The model, frame rows and bursts were independently regenerated into a fresh directory and all
three byte hashes matched. Count conservation, transition conservation, probability row sums,
paired sample correspondence and full baseline metric equality were checked.
The clean-tree regression suite passed **1,237 tests (4 skipped)**, including six P8 tests.

## P9 consumption boundary

Use paired offset samples to preserve u/v covariance. Preserve interval semantics and censoring.
Unsupported sizes/transition rows need an explicit fallback or exclusion contract; they must not
silently acquire invented observations. Seeded simulation and temporal goodness-of-fit checks
remain to be implemented. This is validation calibration, not held-out generalization.

Latency is measured workstation JPEG decode plus inference; camera transport is absent. The
three repeats characterize timing variation only. Degree bearing, metric range and identity
switches cannot be recovered from this model. No policy or camera/ROS integration was added.
The conditional population contains a visible annotated target; false alarms in target-absent
background scenes are not estimated by this calibration.
