# E3-S: size-based range estimation on ETH ds5 cam0

**Verdict: `SIZE_RANGE_USABLE`.** All four preregistered gates passed. Receipt: `run/e3s_result.json`.
Criteria were fixed in [PREREGISTRATION.md](PREREGISTRATION.md) and committed (572314c) before the
analysis ran; nothing below was adjusted afterwards.

E3-P remains **BLOCKED**. Nothing here says how much of the error is attitude.

## What was measured

3220 tracked measurements of drone0 in cam0, 3107 of them included, spanning 30.7 to 108.4 m over
9 blocks of 15 s. The trajectory-smoothness residual is 0.196 px, not a bound on measurement
accuracy. The estimator is `Z = f · W / s` with
`s = sqrt(dark pixel count)`, `f = 1545.7 px`, and `W` fitted by median on every block except the one
being evaluated, so every evaluated frame is out of sample.

This is block-held-out fitting within a previously explored flight, not an independent-flight test.
There is no temporal embargo at block boundaries; neighbouring boundary frames may belong to fitting
and evaluation blocks. Frame count is not the number of independent experimental repetitions.

## Gates

| gate | requirement | result |
|---|---|---|
| G1 coverage | ≥ 6 blocks, ≥ 500 frames | 9 blocks, 3107 frames |
| G2 central | median across blocks of block median \|e\| < 0.25 | **0.062** |
| G3 stability | p90 − p10 of block medians < 0.20 | **0.093** |
| G4 range coverage | ≥ 3 bins reported | 3 of 4 |

## Per block, leave-one-block-out

| block | t (s) | frames | range (m) | bias | median \|e\| | p90 \|e\| | fitted W (m) |
|---|---|---|---|---|---|---|---|
| 3 | 45–60 | 401 | 30.7–74.8 | +5.6 % | 5.9 % | 12.9 % | 0.241 |
| 4 | 60–75 | 102 | 74.9–85.8 | −2.5 % | 6.0 % | 11.6 % | 0.238 |
| 5 | 75–90 | 450 | 78.1–85.6 | +3.1 % | 5.6 % | 16.7 % | 0.240 |
| 6 | 90–105 | 449 | 85.7–108.4 | −1.1 % | 6.2 % | 17.5 % | 0.238 |
| 7 | 105–120 | 248 | 80.2–100.0 | −7.2 % | 7.2 % | 15.0 % | 0.237 |
| 9 | 135–150 | 429 | 85.2–107.3 | −5.2 % | 5.4 % | 12.1 % | 0.236 |
| 10 | 150–165 | 449 | 97.6–106.7 | −6.9 % | 7.0 % | 14.3 % | 0.236 |
| 11 | 165–180 | 434 | 57.0–97.5 | +14.6 % | 14.6 % | 26.3 % | 0.242 |
| 12 | 180–195 | 145 | 57.2–60.5 | +15.5 % | 15.5 % | 24.3 % | 0.240 |

The fitted effective size is stable at 0.236 to 0.242 m across every held-out fit. That is a property of
the segmentation threshold, not the airframe's dimensions.

## Per range bin

| bin | blocks | frames | bias | median \|e\| | p90 across blocks | median \|error\| |
|---|---|---|---|---|---|---|
| 30–50 m | 1 | 106 | `INSUFFICIENT` | | | |
| 50–70 m | 3 | 596 | +15.5 % | 15.5 % | 20.5 % | 9.0 m |
| 70–90 m | 7 | 951 | +0.4 % | 6.0 % | 13.5 % | 4.7 m |
| 90–110 m | 5 | 1454 | −5.3 % | 6.1 % | 6.8 % | 6.0 m |

**Range and time are confounded in this flight, so the bin column cannot be read as a range effect.**
The 50–70 m bin is carried by blocks 11 and 12, which are also the two blocks with the largest bias
overall: within that bin the per-block medians are 7.7 % (block 3, early) against 21.8 % and 15.5 %
(blocks 11 and 12, late). Block 11 alone spans both bands and shows 21.8 % at 50–70 m against 10.5 % at
70–90 m, which hints at a within-block range dependence, but one block cannot establish it. Whether the
bias tracks range, time, or something that varies with both is not decided by this data.

## Sensitivities, none of which gate

| variant | verdict | block-median \|e\| | stability |
|---|---|---|---|
| primary, published timestamps as-is | usable | 6.2 % | 9.3 % |
| timestamps shifted −3.6 frames | usable | 6.5 % | 8.9 % |
| published distortion's radial magnification removed | usable | 6.2 % | — |
| vertical extent instead of pixel count | usable | 19.7 % | 15.7 % |
| horizontal extent instead of pixel count | usable | 13.0 % | 11.7 % |

The unresolved −120 ms timing offset moves the headline figure by 0.3 percentage points, which is why
E3-S survives a blocker that stops E3-P. The preregistration did not name which shift is primary; the
published timestamps as-is were used, since the offset is unverified, and both are reported.

## Data quality

Pose bracket gaps reach the 0.5 s interpolation limit. 2375 of 3107 included frames sit between pose
samples whose total-station `TrackingStatus` is 1, the upstream warning that the highest accuracy may not
be reached under fast target motion; 705 are status 0 and 27 straddle both. The published position
uncertainty columns are millimetre-level, but the warning flag is not an accuracy figure and was not
used to weight or exclude anything.

## What this number is, and is not

- It is **size-based range estimation performance** on one flight, one camera, 31 to 108 m.
- It bundles, without separating: apparent-size change with viewing aspect, segmentation noise, motion
  blur, unmodelled distortion, and the unresolved timing offset.
- The earlier 6.9 % spread figure was a size-consistency spread, not a range error. It is superseded by
  the 6.2 % here, which is a held-out range error and a different quantity.
- Nothing here attributes any part of the error to attitude. That is E3-P.
- `W = 0.24 m` is an effective size that absorbs the segmentation threshold, not the airframe's width.

## Reproducing

```bash
PY="env PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python"
DS=/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5
V=/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted
R=results/eth_ds5_e3s_2026-09-10
$PY tools/track_eth_ds5_drone.py --video $V/cam0.mp4 \
  --seeds results/eth_ds5_intake_2026-09-10/track_seeds.json \
  --output $R/track_adaptive.json --first 890 --last 5310
$PY tools/measure_eth_ds5_size_range.py --track $R/track_adaptive.json --dataset $DS \
  --calibration $DS/calibration/sony5100/sony5100.json --output $R/run --shift-frames 0
```

## E3-P

`BLOCKED`. Its four preconditions and their status are in
[the E3 plan](../../docs/plans/eth_ds5_e3_2026-09-10.md). None of the numbers above may be used to choose
E3-P's hypotheses or thresholds.
