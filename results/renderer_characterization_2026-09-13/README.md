# Renderer characterization track (RC) — 2026-09-13

How this renderer represents geometry, apparent area, material, shading and lighting, measured
generically and reproducibly. Six experiments, all preregistered before any of the code existed.

**Namespace warning.** This track's experiments are `RC-R1 … RC-R6`. An earlier, unrelated
appearance-prototype track from 2026-09-11 also used the names R1–R5
(`results/renderer_r3_*`, `renderer_r4*`, `renderer_r5_benchmark*`). The two sets are different
experiments and their numbers are never pooled.

**What this track does not claim.** Nothing here is evidence about the detector, tracking,
association, the policy, or the cause of the closed D8b frozen-policy result (−48.967 pp, 95 % CI
[−50.113, −47.821] pp, `MATERIAL_LOSS`). Shading, mesh geometry, projected area and the detector
are all `NOT_TESTED` as causes of that loss. Every result file carries
`causality_vs_d8b: NOT_TESTED`.

## Results

| experiment | verdict | one-line reading |
|---|---|---|
| [RC-R1 geometry](R1_geometry/) | `GEOMETRY_DEFECT` | The intersection path is verified (G1, G2, G5, G6 held); G3, G4 and G7 were not met for the polyhedral specimens at 320×240. |
| [RC-R1 diagnostic](R1_quantization_diagnostic/) | `DIAGNOSTIC_NO_VERDICT` | Those three unmet gates shrink roughly as 1/focal length, which is what silhouette quantization does. |
| [RC-R2 apparent area](R2_apparent_area/) | `AREA_MATCH_FAILED` | One isotropic box scale fitted on four views does not transfer to the other twenty: 19.0 % off, against a 5 % tolerance. |
| [RC-R3 shading](R3_shading/) | `SHADING_GATE_FAILED` | Lambertian shading adds spatial variation only where more than one surface orientation is visible; on 10 of 96 arm-views it adds exactly none. |
| [RC-R4 lighting](R4_lighting/) | `LIGHTING_CHARACTERIZED` | 36 lighting cells move silhouette luminance over 0.0275–0.5445 and leave every geometry buffer byte-identical. |
| [RC-R4M material](R4M_material/) | `MATERIAL_CHARACTERIZED` | 5 material sets move luminance over 0.1436–0.5912, again with identical geometry. |
| [RC-R5 cost](R5_performance/) | `DESCRIPTIVE_NO_VERDICT` | 54 of 54 cells measured. Host copy dominates the mesh path; 129× the triangles costs 12 % more time. |
| [RC determinism](determinism/) | `DETERMINISTIC` | Two independent processes produced identical hashes for all seven arrays. |

## What was actually learned

1. **The intersection path is right.** Against an independent CPU Möller–Trumbore intersector, the
   Warp G-buffer disagreed on **zero** silhouette pixels in all 24 views for both mesh specimens,
   with a worst range difference of 1.4e-6 m. The analytic sphere matched its closed form to
   0.03 px in radius and 2.4e-8 m in depth.
2. **Silhouette size at 320×240 is not precise to 2 %.** The measure's per-view error against a
   1280×960 reference is 3.0 % (box), 2.2 % (quadrotor mesh) and 0.14 % (analytic sphere), and it
   halves as the focal length doubles. The preregistered G4 and G7 tolerances of 2 % were set
   without an estimate of this, and they were not met. The tolerance was **not** changed afterwards.
3. **The box proxy looks 67 % larger in area than the quadrotor mesh**, and the analytic sphere
   53 % larger, over the 20 validation views.
4. **An isotropic box scale cannot remove that gap.** Fitted on four axis-aligned views and frozen,
   it left the matched box 19 % too large on the held-out views. The box's apparent area grows with
   view angle differently from the mesh's, and one scalar cannot follow that.
5. **Triangle count is not a measure of shading structure.** At azimuth 0 and 180 with elevation 0,
   the quadrotor mesh shows 98 and 118 visible triangles with a visible-normal entropy of exactly
   **0 bits** and a mean |n·view| of exactly 1.000: every visible face points at the camera. Its
   lambertian luminance variance there is exactly zero. The analytic sphere, with no triangles at
   all, occupies 16 of 32 normal bins at 3.99 bits in every view.
6. **Lighting and material are separable from geometry in this renderer.** Across 36 lighting cells
   and 5 material sets, every one of the six geometry buffers kept a byte-identical hash and every
   silhouette pixel count was unchanged, while mean luminance moved by 0.52 and 0.45 respectively.
7. **The mesh render path is dominated by the host copy, not by ray casting.** At 640×480 with 32
   scenes, the quadrotor G-buffer takes 14.3 ms and copying it to the host takes 175 ms. Lambertian
   shading adds 6.4 ms. Going from 12 triangles to 1548 costs 12 % more time, as a BVH should.
8. **The analytic arm here is a CPU reference implementation**, not a GPU kernel; its cost
   (933 ms at the largest cell) measures that implementation, not analytic intersection in general.

## Layout

Every experiment directory holds `PREREGISTRATION.md` (a sha256 pin to the track preregistration
plus the thresholds read from code at run time), `README.md`, `config.json`, `summary.json`,
`receipt.json` and `source_manifest.json`. The determinism directory also keeps both exports.

Figures are in
[`docs/assets/paper/renderer-characterization-2026-09-13/`](../../docs/assets/paper/renderer-characterization-2026-09-13/),
as SVG source with PNG and vector PDF exports: one per experiment plus the evidence map.

RC-R6 is the export path itself. Its evidence is the `determinism/` directory, which holds both
exports and their comparison, and the exporter validation checks in `tests/test_rc_export.py`.

## Reproducing

```bash
python tools/run_renderer_characterization.py r1  --output <dir> --device cuda:0
python tools/run_renderer_characterization.py r2  --output <dir> --device cuda:0
python tools/run_renderer_characterization.py r3  --output <dir> --device cuda:0
python tools/run_renderer_characterization.py r4  --output <dir> --device cuda:0
python tools/run_renderer_characterization.py r4m --output <dir> --device cuda:0
python tools/run_renderer_characterization.py determinism --output <dir> --device cuda:0
python tools/benchmark_renderer_characterization.py --output <dir> --device cuda:0
python tools/diagnose_rc_silhouette_quantization.py --output <dir> --device cuda:0
python tools/render_rc_figures.py
```

Each run refuses to write a verdict if its own source changed while it was running, and records the
git commit and a sha256 per source file it depended on.
