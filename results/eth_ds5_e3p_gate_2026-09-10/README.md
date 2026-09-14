# E3-P gate: three of four conditions clear, attitude reliability does not

**Verdict: `E3P_GATE_BLOCKED`, on attitude reliability alone.** Receipt: `attitude_gate.json`.

This also **corrects the E3-P preconditions** written on 2026-09-10. Two of the four were overstated. The
correction rests on a geometry and dynamics calculation, not on any E3-S result, and the original wording
is kept in the plan's history.

## What the corrected conditions are

| condition | earlier claim | measured | status |
|---|---|---|---|
| camera orientation | required | the aspect angle needs only two positions and the target's attitude; the camera's rotation never enters an area-based measure | **not required** |
| frame-to-pose timing | strongly required | the −3.6 frame (−120 ms) offset moves the aspect angle by 0.88° at p90, against a 30.8° span: 2.9 % | **not binding** |
| attitude convention | unverified | identified, see below | **clear** |
| attitude reliability | unverified | see below | **fails** |
| aspect and range separable | not considered | correlation 0.21, all four range bins span more than 20° of aspect | **clear** |

The camera's orientation only matters for measures that depend on how the target's silhouette lands on
the sensor, such as horizontal and vertical extent. E3-S already showed the area-based measure is the
best of the three, and its aspect dependence is a property of the viewing direction alone.

## C1, convention: identified with a wide margin

A multirotor accelerates horizontally by tilting its thrust axis, so the published angles can be tested
against accelerations derived from position ground truth. All 32 readings of the angles were scored by
how well each predicts the direction of the measured horizontal acceleration.

The winner is **intrinsic `xyz` Euler, body to NED, thrust along body −z, mapped to ENU**, at a median
direction error of **17.7°**. The runner-up is 16.3° worse. That margin is what identifies the
convention: a wrong reading scrambles the direction, and every wrong reading did.

This confirms from the data what was previously taken on trust from the upstream toolkit's source.

## C2, reliability: fails, and a corrected form still fails

The first form asked whether the tilt magnitude predicts the acceleration magnitude through
`|a_h| = g·tan(tilt)`. Slope 0.885, but correlation **0.18**, far below the 0.5 required. That form is
mis-specified: a multirotor in steady flight tilts to balance aerodynamic drag while not accelerating at
all, and here the median speed is high enough for that to matter.

The corrected form fits thrust and drag together as vectors, with thresholds fixed before it ran:

| quantity | required | measured |
|---|---|---|
| thrust gain | 0.6 to 1.4 | 0.73 ✓ |
| drag coefficient | negative | −0.017 ✓ |
| correlation | ≥ 0.5 | 0.64 ✓ |
| residual improvement from the drag term | ≥ 0.20 | **0.044 ✗** |

It fails on the last one. In hindsight that sub-criterion tests whether drag is large rather than whether
the attitude is reliable, so it is arguably the wrong bar as well. **No third variant was run.** Trying
again until something passes would be selecting the test by its answer, and the gate is more useful
blocked than opened by a criterion chosen after the fact.

What the numbers do say: the attitude's *direction* is informative, at 17.7° median with a 16° margin
over every alternative reading, and the vector model correlates 0.64. Its *magnitude* relation to the
dynamics is weak, and part of that is likely the 1.2 s smoothing window used to differentiate position,
which attenuates a fluctuating acceleration and would bias the thrust gain below one. That is a
hypothesis, not a finding, and it was not tested.

## C3, identifiability: aspect and range are separable

Aspect-to-range correlation **0.21**. Six range-by-aspect cells hold at least 20 frames from at least two
time blocks. Every range bin spans more than 20° of aspect: 21.6°, 24.6°, 25.9°, 23.3°. So unlike range
and time, which E3-S found confounded, aspect and range are not collinear in this flight, and an analysis
that separates them is possible in principle.

## C4, timing: not binding

The −120 ms offset shifts the aspect angle by 0.88° at the 90th percentile, 2.9 % of the observed span,
against a 5 % threshold.

## What would open the gate

One reliability criterion, specified once and in advance, that tests whether the published attitude
describes the airframe rather than whether drag is large. The differentiation window should be a declared
parameter of that criterion rather than an inherited default. It belongs in E3-P's own preregistration,
not in another patch to this gate.

## Reproducing

```bash
PY="env PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python"
DS=/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5
$PY tools/check_eth_ds5_attitude_gate.py --dataset $DS \
  --track results/eth_ds5_e3s_2026-09-10/track_adaptive.json \
  --calibration $DS/calibration/sony5100/sony5100.json \
  --output results/eth_ds5_e3p_gate_2026-09-10
```
