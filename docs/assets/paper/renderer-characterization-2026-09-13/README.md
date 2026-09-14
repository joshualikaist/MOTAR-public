# Renderer characterization figures — 2026-09-13

Seven figures for the RC track: one per experiment, plus the evidence map. Every number plotted is read from a committed `summary.json` under
`results/renderer_characterization_2026-09-13/`; nothing is recomputed at drawing time, so a figure
cannot disagree with the result it illustrates. Rebuild with `python tools/render_rc_figures.py`.

White background, three ink colours, no gradients. Each figure ships as SVG source plus a PNG
export and a vector PDF. These are new files; no existing figure, archive or page was modified.

| figure | what it shows |
|---|---|
| `rc-r1-geometry-representation` | Apparent size across 24 views for the four specimens, beside the entropy of the visible surface normals. The entropy panel is where the zero-variance shading views come from: the box and the mesh both fall to 0 bits at azimuth 0/90/180/270 with elevation 0. |
| `rc-r2-apparent-area` | Mean silhouette area per specimen over the 20 validation views, with the ratio to the quadrotor mesh, and the per-view areas. The isotropic scale was fitted on four other views and frozen. |
| `rc-r3-shading-variation` | Silhouette luminance variance against visible-normal entropy. Every view whose variance is exactly zero sits at exactly zero entropy; every view with any orientation diversity is an order of magnitude above the S3 threshold. |
| `rc-r4-lighting-and-material` | 36 lighting cells and 5 material sets on one fixed geometry. Geometry hashes and silhouette pixel counts are identical in all 41 cells. |
| `rc-r5-render-cost` | Median render time against ray count for six configurations, and the stage breakdown of the largest cell of each. Descriptive only, and not comparable with D6 or D7. |
| `rc-r6-export-invariance` | What an export keeps fixed and what it lets change, across seven arrays and three comparisons. Changing the lighting or the material moves `rgb` alone; two independent processes move nothing. |
| `rc-evidence-map` | What each experiment established, and the explicit `NOT_TESTED` link to the closed D8b result. |

None of these figures is evidence about the detector, the policy, or the cause of the D8b
frozen-policy loss.
