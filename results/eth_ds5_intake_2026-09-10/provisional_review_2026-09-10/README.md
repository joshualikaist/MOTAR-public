> **2026-09-11 정정 안내.** 이 문서의 "six-armed airframe" 서술은 철회됐다. drone0은 쿼드로터다.
> 수치는 고치지 않았다. `../CORRECTION_2026-09-11_rotor_count.md`를 보라.

# Provisional review of the 17 alignment frames, and what it decided

**Reviewer: Claude, an AI, not the independent human reviewer the plan requires.** This does not close
the identity gate. It was run to answer one question: can the alignment set actually discriminate?

## Method

Each frame was searched for moving dark blobs by temporal differencing against frames ±3, without
consulting ground truth for the search. Candidates were inspected as zoomed crops and the airframe was
identified visually. Box centres come from a darkness-weighted centroid in a 27 px window. Ground-truth
positions were never used to place a box. Frame 1778 was marked `unsure` and excluded because the
airframe overlaps a treetop and the centroid is contaminated. Sixteen boxes were recorded.

## Answer: the set discriminates

The pixel RMS against ground truth is a clean single-minimum curve. Around the optimum it rises by about
2 to 3 px per frame of shift, well above the annotation noise. The 17-frame set does what it was built
for; the stride-150 pilot could not have.

| shift, frames | −6 | −5 | −4 | −3 | −2 | −1 | 0 | +1 |
|---|---|---|---|---|---|---|---|---|
| pixel RMS | 11.4 | 8.7 | 6.9 | 7.0 | 9.1 | 12.2 | 15.4 | 19.3 |

## But the answer is not a frame relabelling, and the calibration fails

The refined optimum is at −3.53 frames, that is −117.6 ms. It is 0.47 frames away from any integer, so it
cannot be a frame_id to container-index relabelling. None of the three index candidates explains it. The
original ±3 frame search would have missed it entirely; the search now spans ±8 and reports the
continuous optimum, and refuses to name an index offset for a non-integer result.

Even at that optimum the RMS is 6.9 px against a 4 px gate, so the check returns
`CALIBRATION_CANDIDATE_NOT_VALIDATED`. The residuals are structured: they correlate 0.78 with distance
from the principal point, and the two frames nearest the image edge dominate at 20.1 px and 11.0 px.

Freeing one parameter at a time, with everything else published:

| freed | best RMS | value found | published |
|---|---|---|---|
| nothing | 6.69 px | | |
| focal length | 3.58 px | ×0.985 | |
| radial `k1` | **2.34 px** | **−0.100** | −0.0112 |
| focal and `k1` | 2.34 px | ×1.00, −0.100 | |

A single radial coefficient roughly nine times the published one absorbs almost all of it, and the focal
length is then unchanged. That is the signature of a lens or zoom setting different from the one the
Sony5100 file was calibrated at. **The published calibration does not describe ds5 cam0 as-is.**

## What this does and does not establish

- The tracked object's bearing trace follows drone0's ground truth to within 0.6 degrees over a wide arc,
  and at frame 901 in the pilot set a six-armed airframe is visible where both DJI quadrotors are
  excluded. Identity is well supported but not formally reviewed.
- The −118 ms offset and the distortion mismatch are partly confounded: fitting `k1` moves the optimum to
  −3.35 frames. Sixteen points cannot separate them.
- The fitted `k1` is a diagnostic. Adopting it and then declaring the calibration valid would be
  circular. A calibration for ds5 cam0 has to come from the dataset or from the authors.
- E3 does not start. The blocker is now specific: the distortion model, not the frame indexing.
