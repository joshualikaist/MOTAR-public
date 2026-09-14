# RC-R3 — shading, with geometry held fixed

Verdict: **SHADING_GATE_FAILED**

Failed gates: `S3_shading_varies_area_matched_box`, `S3_shading_varies_box_proxy`, `S3_shading_varies_quadrotor_mesh`

A failed S3 is a measurement, not a defect to be tuned away: where only one
surface orientation is visible, lambertian shading is spatially constant by
construction. The per-view variances and visible-triangle counts are recorded
next to each other in `summary.json` so that reading is checkable.

| arm | flat variance (worst) | lambertian variance (worst) | views below S3 |
|---|---|---|---|
| `analytic_sphere` | 0.000e+00 | 1.090e-02 | 0 of 24 |
| `box_proxy` | 0.000e+00 | 0.000e+00 | 4 of 24 |
| `area_matched_box` | 0.000e+00 | 0.000e+00 | 4 of 24 |
| `quadrotor_mesh` | 0.000e+00 | 0.000e+00 | 2 of 24 |

Geometry is identical across shading modes by construction: one G-buffer per cell
is rendered once and shaded repeatedly, and S1 re-hashes it afterwards to confirm
shading did not mutate it.

Reported statistics are ordinary image statistics (generic image statistics only: luminance moments and quantiles, RMS contrast, Sobel gradient magnitude, histogram entropy, within-depth-quantile luminance spread).
No detector, tracking or policy quantity appears here, and no cause of the D8b
result is proposed (`causality_vs_d8b: NOT_TESTED`).
