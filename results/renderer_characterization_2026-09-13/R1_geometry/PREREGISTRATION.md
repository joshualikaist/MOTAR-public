# RC-R1 — preregistration pin

This experiment is governed by the track preregistration:

* path: `results/renderer_characterization_2026-09-13/PREREGISTRATION.md`
* sha256: `603d93c846582735fcebd0792665d3c9fa8f9b6b36e21f23846e0095bd00cc3e`

No threshold was chosen or changed after a result was seen. The values below were read out
of the code at run time, so this file records what was actually applied rather than what
was intended:

```json
{
  "box_bbox_tolerance_px": 1.0,
  "depth_bracket_slack_m": 0.002,
  "inverse_distance_tolerance": 0.02,
  "min_silhouette_pixels": 300,
  "min_visible_triangles": 2,
  "normal_bins": [
    8,
    4
  ],
  "range_difference_max_m": 0.001,
  "resolution_consistency_tolerance": 0.02,
  "silhouette_disagreement_max": 0.005,
  "sphere_centroid_tolerance_px": 0.5,
  "sphere_depth_tolerance_m": 0.002,
  "sphere_radius_tolerance_px": 1.0
}
```



Out of scope, as in the track document: detector behaviour, tracking, policy performance,
and any causal account of the closed D8b result. Causality with respect to D8b is
`NOT_TESTED` here.
