# Can the distortion model be fixed? Investigation, 2026-09-10

**Answer: no, because the distortion is not the fault.** The published calibration is correct for the
images it was made from, and no camera-model change explains cam0's residual against drone0. The
inconsistency is somewhere else, and image evidence alone cannot say where. Separately, the measurement
E3 actually needs does survive.

## 1. The published file is not stale or mis-assigned

The dataset publishes 122 chessboard images for the Sony a5100 alongside the calibration JSON. They were
fetched and hash-verified, boards were detected in 119 of them, and the intrinsics were recomputed from
half the images and scored on the other half.

| model | held-out corner RMS |
|---|---|
| published `sony5100.json` | 1.245 px |
| recomputed from half the boards | 1.262 px |

The published file matches its own images as well as an independent recalibration does, and the
recomputed values agree closely (fx 1542.8 versus 1545.4, k1 −0.0136 versus −0.0112). Receipt:
`../calibration_check/calibration_check.json`. So the file is a faithful calibration of the a5100 in the configuration those
images were shot in. That closes the "the JSON is wrong" hypothesis.

## 2. Against drone0 the residual is 28 times the measurement noise

The 41 reviewed centres were propagated through neighbouring frames, giving 3495 tracked positions over
41 to 186 s. Tracking noise, measured from trajectory smoothness with no ground truth and no camera
model, is **0.198 px**. On 678 of those points the published calibration with a single rotation and a
free time offset leaves **5.6 px RMS**.

The residual is mostly radial (5.0 px radial versus 2.6 px tangential) and grows with radius: +1.4 px
inside 300 px of the principal point, +3.3, +4.3, then +7.9 px beyond 700 px. That looks exactly like a
distortion error, which is why a distortion fit is tempting.

## 3. Every hypothesis was tested on data it had not seen

`tools/audit_eth_ds5_camera_model.py`, receipt `audit/camera_model_audit.json`. Each hypothesis is fitted
on one split and scored on the other, with the rotation refitted on the held-out split so that camera
drift is not mistaken for a lens error.

| split | published | best hypothesis | best held-out |
|---|---|---|---|
| alternating | 5.23 px | radial and focal | 3.04 px |
| early to late | 5.17 px | focal | 5.27 px |
| late to early | 5.20 px | focal | 4.66 px |
| centre to edge | 6.22 px | focal | 6.63 px |

No hypothesis wins everywhere, and the best held-out result is still 15 times the tracking noise. In
particular a radial model fitted on the first half of the flight makes the second half **worse**
(5.17 → 12.24 px), so the residual is not a fixed property of the lens.

Also tested and rejected, each by held-out validation: principal point, camera position (fitted offsets
of 0.7 to 1.3 m, far beyond any plausible survey-to-optical-centre lever arm), a time drift as well as a
time offset, a rotation refitted every 8 s (5.6 → 3.7 px, no further), and the published prism lever arm
(5.61 → 4.59 px with the published 0.216 m; a free lever arm reaches 3.9 px but wants 0.64 m).

An unexplained time offset of −3.6 frames, about −120 ms, is needed by every variant.

## 4. What is left

A residual that is partly radial, partly time-varying, needs a large unexplained time offset, and yields
to no rigid-body or lens parameter is more likely to sit in the published timestamps or the fused pose
than in the camera. The dataset's own history supports that: the frame timestamps carry cam1's
`Time_scale` and predate the 2022 coefficient revision, and ds5 publishes no calibration of its own
despite its README listing one. Settling it needs something this repository does not have: the ds5
calibration session, or the authors.

## 5. What still works

E3 needs apparent size against range, not absolute reprojection. A 5 px position error does not corrupt a
size measurement. Over the 3429 tracked frames from 18 to 108 m, apparent size times range is stable:

| size measure | implied target size | relative spread, median | 90th percentile |
|---|---|---|---|
| square root of dark-pixel count | 0.238 m | 6.9 % | 16.4 % |
| horizontal extent | 0.478 m | 15.6 % | 38.0 % |
| vertical extent | 0.188 m | 16.4 % | 41.4 % |

Vertical extent, the measure least affected by motion blur, gives a consistent implied size across bins
from 30 m to 115 m (0.199, 0.195, 0.199, 0.183 m). So the size-versus-range relation is measurable to
about 7 % even though absolute reprojection is not.

**Consequence for the plan.** Absolute-geometry uses of ds5 cam0 stay blocked. A size-versus-range arm of
E3 is feasible and would have to be preregistered separately, with the 5.6 px position inconsistency and
the unexplained −120 ms offset recorded as known limitations.
