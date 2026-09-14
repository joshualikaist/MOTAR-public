# Renderer measurement contract v1

Independent static graphics, not perception or policy evaluation.
`causality_vs_d8b = NOT_TESTED`.

## Chosen measurement resolution

**1280×960** is the smallest tested, non-reference resolution that satisfies the
[preregistered R1b conditions](preregistration_renderer_followup_2026-09-14.md).
This recommendation covers only the three specimens, 24 views each, distance 2 m,
60° horizontal FOV and current point-sampled renderer. It is not a universal accuracy bound.
Do not extrapolate this qualification to distant/subpixel objects or different cameras.

[R1b evidence](../results/renderer_characterization_r1b_2026-09-14/README.md):
288/288 specimen-view-resolution rows measured; no clipping/minimum-area refusals.
The reference is sampled **2048×1536**, the highest legal 4:3 resolution under the unchanged
2048-pixel-per-dimension camera contract. It is not exact GT, and cannot qualify itself.

Worst absolute relative discrepancy across all 72 specimen-views (%):

| Resolution | Width | Height | Area | Centroid / equivalent diameter | Median depth | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 320×240 | 5.391 | 2.476 | 7.021 | 1.267 | 0.195 | No |
| 640×480 | 2.609 | 1.000 | 3.119 | 0.707 | 0.074 | No |
| 1280×960 | 1.217 | 0.483 | 1.143 | 0.299 | 0.056 | Yes |
| Registered tolerance | 2.000 | 2.000 | 2.000 | 1.000 | 1.000 | Every view must pass |

These numbers describe measured finite-resolution discrepancy (a quantization estimate), not a
provable lower floor, absolute measurement accuracy, or a correction to historical RC-R1.
RC-R1 remains `GEOMETRY_DEFECT` under its original experiment and gate.

## Quantities and units

For visible integer pixel coordinates `(u,v)`, focal length `f` in pixels, and hit count `A`:

- Width/height are inclusive: `max - min + 1`; no subpixel contour interpolation.
- Normalize width/height by `f`, area by `f²`. Store raw pixel values as well.
- Centroid is the arithmetic mean of visible pixel coordinates, centered by `(W/2,H/2)`
  and divided by `f`. Sampling is at integer coordinates; do not add half a pixel.
- Optical depth is camera +Z in metres, not Euclidean ray distance. Report its silhouette median.
- For each non-centroid quantity, compare to the same view at reference resolution:
  signed difference, absolute difference, and absolute difference / absolute reference.
- Centroid absolute error is the 2D norm of normalized centroid displacement. Its relative error
  divides by reference equivalent diameter `2 sqrt(A_ref / pi) / f_ref`, not by centroid position.

Stored `reference_in_current_pixels` also expresses reference width/height/area and centroid
displacement in the tested camera's pixel units. Analysis unit is the specimen-view, not pixels.
Fewer than 300 visible pixels or image-border clipping refuses a view. An unmeasured view must
never become zero error. The evidence retains individual rows, not only extrema.

## Multi-view area control: not yet qualified

[R2b](../results/renderer_characterization_r2b_2026-09-14/README.md) used 1280×960 as fixed
before measurement. Six calibration views fit one three-axis box scale by log-area least squares;
`fit.json` and its SHA were frozen before rendering the 24 disjoint held-out views.
No held-out refit, threshold update, or alternate initialization occurred.

| Phase / control | Median absolute relative area error | P90 | Worst view |
| --- | ---: | ---: | ---: |
| Fit / historical isotropic C0 | 22.772% | 25.268% | 25.978% |
| Fit / anisotropic C1 | 7.317% | 11.429% | 11.774% |
| Held-out / historical isotropic C0 | 23.525% | 32.211% | 34.301% |
| Held-out / anisotropic C1 | 8.132% | 11.051% | 11.544% |
| Registered maximum | 5% | 10% | 20% |

C0 retains scale `0.8450780426128714`. C1 scales are
`[0.784449402374687, 0.8725480318093369, 0.6582063554038994]` relative to half extents
`[0.14,0.14,0.06]` m. Optimization succeeded, but the held-out median and P90 gates failed:
**`AREA_CONTROL_FAILED`**. C1 is not adopted as an area-matched control. Smaller error than C0
does not justify a pure-shape causal comparison. These are held-out angles of an already explored
specimen, not independent-object generalization.

## Measurements deliberately left at other resolutions

R3b describes normal/shading statistics at its preregistered 640×480; it is not a ≤2%-accurate
silhouette-area experiment. R5b reports cost at its fixed 320×240×8 and 640×480×32 fixtures;
it is not a benchmark of the newly recommended 1280×960 measurement resolution.
Do not silently change their resolutions after R1b or transport their throughput to 1280×960.

See [renderer contract](renderer_contract_v1_2026-09-14.md) for buffer meanings, supported normals,
timing boundaries, provenance, figures and reproduction commands.
