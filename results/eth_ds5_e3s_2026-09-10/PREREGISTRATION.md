# E3-S preregistration, fixed 2026-09-10 before any result was computed

Question: how accurately does a target's apparent size in cam0 recover its metric slant range, and how
does that error behave across range?

Everything below was written and committed before the analysis ran. If a result is surprising, the
result is reported, not the criteria changed.

## Inputs

- Measurements: `track_adaptive.json` in this directory. 3174 tracked positions and sizes from cam0,
  seeded by the 41 reviewed centres, produced by `tools/track_eth_ds5_drone.py` with an adaptive search
  window. Ground truth was not consulted while measuring. Image-measurement noise 0.196 px RMS, from
  trajectory smoothness over 15-frame windows.
- Ground truth: `dataset5/pose/fused_pose.txt`, target position only, interpolated to the frame time with
  the existing 0.5 s maximum bracket gap. Slant range is the distance from cam0's surveyed position in
  `camera-locations/campos.txt`.
- Camera: `calibration/sony5100/sony5100.json`, **focal length only**, f = (fx + fy) / 2. No principal
  point, no distortion, no extrinsic. This is what keeps E3-S independent of the unresolved blockers.

## Analysis population and exclusions

Include a tracked frame when all hold:

1. it carries a project timestamp and the pose interpolates at that time;
2. its centre lies at least 40 px from every image border;
3. the tracker accepted it, which already requires contrast at least 25 grey levels, between 4 and 1500
   dark pixels, a step of at most 14 px from the previous frame, a centre above row 740, and a blob that
   does not reach the search-window edge.

No exclusion by range, by residual, or by agreement with any model. Frames closer than about 31 m are
absent because the target's blob could not be measured without truncation there; this is a coverage limit
of the measurement, recorded as such, not a filter on outcomes.

## Size measures

- **Primary: `sqrt(dark pixel count)`.** Chosen before this analysis on the 2026-09-10 exploratory pass,
  which is disclosed rather than hidden: it is the least sensitive of the three to motion blur direction.
- Secondary, reported alongside: vertical extent, horizontal extent.

## Estimator and fit

`Z_hat = f * W_hat / s`, with `W_hat = median over calibration frames of (s_i * Z_i / f)`. The median is
used so a few bad frames cannot drag the fit. `W_hat` carries the units of an effective target size; it
is a fitted constant, not a claim about the airframe's dimensions.

## Splitting, so neighbouring frames are never both fit and test

The unit of analysis is a **15 s block of project time**, not a frame. Neighbouring frames are strongly
correlated and are never treated as independent samples.

- **Primary: leave-one-block-out.** For each block, `W_hat` is fitted on all other blocks and evaluated on
  that block alone. Every evaluated frame is out of sample.
- Secondary: a single alternating split, even-ranked blocks fit, odd-ranked blocks evaluated.
- Reported per block: frame count, range span, median signed relative error, median absolute relative
  error, 90th percentile absolute relative error, and the fitted `W_hat` from its complement.

## Error metrics

Relative error `e = (Z_hat − Z) / Z`, per frame. Aggregated first within a block, then across blocks.
Reported: bias as the median signed error, absolute error as the median and 90th percentile of `|e|`, and
the absolute range error in metres per bin. Frame-level distributions are shown but never used as the
sample size.

## Range bins

30–50, 50–70, 70–90, 90–110 m, fixed. A bin is reported only if at least 3 blocks contribute at least 20
included frames each; otherwise it reports `INSUFFICIENT` and no number is quoted for it.

## Minimum sample

At least 6 evaluation blocks and at least 500 evaluated frames in total.

## Verdict, fixed in advance

`SIZE_RANGE_USABLE` when all four hold:

- **G1 coverage**: at least 6 evaluation blocks and at least 500 evaluated frames.
- **G2 central performance**: the median across blocks of the per-block median `|e|` is below 0.25.
- **G3 stability**: across blocks, the 90th minus the 10th percentile of the per-block median `|e|` is
  below 0.20.
- **G4 range coverage**: at least 3 range bins qualify under the minimum above.

Otherwise `SIZE_RANGE_NOT_USABLE`, naming every gate that failed. No gate is adjusted after seeing the
numbers, and no bin, block or frame is dropped to move a gate.

## Sensitivities, reported and never gating

1. Time alignment: repeat with a 0 frame and a −3.6 frame shift; report how the verdict statistics move.
2. Distortion: repeat with the published model's local radial magnification applied to the size; report
   the same.
3. Secondary size measures: report the same statistics for vertical and horizontal extent.

## Interpretation limits, binding on the write-up

- The result is **size-based range estimation performance** and nothing else.
- It bundles, without separating: the target's apparent size changing with viewing aspect, segmentation
  noise, motion blur, unmodelled distortion, and the unresolved −120 ms timing offset.
- The earlier 6.9 % spread figure is **not** a range error and must not be quoted as one, nor as an
  attitude effect.
- No statement about how much of the error is attitude-driven. That is E3-P, which is blocked.
- `W_hat` is an effective size that absorbs the segmentation threshold. It is not the airframe's width.
- Coverage is roughly 31 to 108 m from one flight and one camera. It is not a general drone result.
