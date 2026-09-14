# RC-R4M — material sensitivity

Verdict: **MATERIAL_CHARACTERIZED**

* cells: 5
* geometry and silhouette identical in every cell: **True**
* silhouette mean luminance spread across cells: 0.4476 (threshold 0.02)
* luminance range: 0.1436 to 0.5912

Changing the material changes the image and nothing else: every geometry buffer hash and
every silhouette pixel count is unchanged across the grid. That separation is the
result; it is not evidence about any detector or policy, and it proposes no cause for
the closed D8b result (`causality_vs_d8b: NOT_TESTED`).
