# RC-R4 — lighting sensitivity

Verdict: **LIGHTING_CHARACTERIZED**

* cells: 36
* geometry and silhouette identical in every cell: **True**
* silhouette mean luminance spread across cells: 0.5170 (threshold 0.02)
* luminance range: 0.0275 to 0.5445

Changing the light changes the image and nothing else: every geometry buffer hash and
every silhouette pixel count is unchanged across the grid. That separation is the
result; it is not evidence about any detector or policy, and it proposes no cause for
the closed D8b result (`causality_vs_d8b: NOT_TESTED`).
