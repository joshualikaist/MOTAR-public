# RC follow-up: R5b, R1b, R2b, R3b — frozen before implementation/measurement

Independent static renderer characterization only. No detector, temporal selector, policy,
simulator task, deployment or D8b causal diagnosis. `causality_vs_d8b = NOT_TESTED` throughout.
Historical RC-R1 `GEOMETRY_DEFECT`, RC-R2 `AREA_MATCH_FAILED`, RC-R3 `SHADING_GATE_FAILED`,
R4/R4M records and R5 ResourceWarning remain byte-preserved. Their criteria are not changed.
These new studies use the existing camera, geometry kernels, G-buffer and Lambertian shader.

## Common execution and evidence contract

Order: R5b → R1b → R2b → R3b → measurement/renderer contract and new figures.
Use the existing aerialgym Python, CUDA 0, seed 20260914; record process runtime fingerprint,
Warp version, Git revision, this document SHA and dependency file hashes before/after execution.
Preregistration must be an ancestor of clean committed implementation. Output directories are
new and never overwritten. No source edit during a measurement. No model training or physics.
Record raw per-view/per-iteration statistics, config, receipt, source manifest, README and SHA.
Keep full image arrays transient unless needed for a failed invariance case; retain output hashes.
No historic array/log/receipt cleanup. Preserve failures and unavailable fields, never replace by 0.

Camera remains 60-degree horizontal FOV, far range 20 m, +X right/+Y down/+Z forward.
Reference specimens: analytic sphere radius 0.15 m, box half extents (0.14,0.14,0.06) m,
existing v3 URDF visual mesh. Object remains at origin; no scene/controller/association logic.
Existing Camera limits each dimension to 2048; do not expand its contract for this study.
GPU resource stop: predicted single output >1.5 GiB, Torch reserved >5 GiB, or process RSS >8 GiB.
Stop the affected cell with its reason; do not downscale it or weaken a threshold.
Use sequential fresh processes for R5b repeats/arms; no simultaneous GPU experiment.

## RC-R5b — transfer strategy (first)

New path: `results/renderer_characterization_r5b_2026-09-14/`.
Fix RSS reader lifetime only in a new harness; the old benchmark source stays unchanged.

Two fixtures: (320×240,8 views) and (640×480,32 views), v3 specimen, evenly spaced azimuths,
elevation 10 degrees, distance 2 m, fixed camera-relative light and grey 0.55 material.
Same scene/poses/geometry/shader/seed across arms. Three fresh-process repeats per fixture/arm;
20 warmup and 100 measured iterations. Rotate arm order by repeat to reduce order confounding.

- A0: current-style full transfer: each of seven tensors → CPU numpy → owned copy.
- A1: leave buffers on device, transfer only valid-pixel-count and RGB-sum scalars each iteration.
  Full export once outside timing for invariance. Its throughput is **resident/scalar throughput**,
  not full dataset export throughput; full-export validation cost reported separately.
- A2: pack same-dtype buffers on device; deferred grouped blocking transfers, unpack CPU views.
  Include device packing in the transfer-strategy cost. No lossy casting or changed dtypes.
- A3: preallocated pinned CPU buffers, nonblocking copies followed by explicit completion wait.
  No claim of overlap with geometry compute. Report one-time pinned allocation separately.
  Unsupported allocation/stream support → `UNSUPPORTED`, with exception and no replacement arm.

Compute the geometry and shaded tensors once per iteration in every arm; transfer consumes those
exact tensors. Timing boundaries: geometry completion, shading completion, transfer completion,
end-to-end wall time. Record synchronized staged times and separate unstaged headline totals;
do not add medians and call that total. Raw samples, mean/median/P95, scene throughput, RSS,
NVML device memory (via nvidia-smi if needed), Torch allocated/reserved and untracked-Warp-memory
qualification. Allocation/initialization is outside steady-state timing and reported separately.

Every repeat hashes RGB/range/depth/normal/face ID/instance ID/valid against A0 and checks within-run
first/last outputs. A1's terminal full export must also match. Any mismatch gives
`OUTPUT_MISMATCH` and withholds performance comparison; no selection based on invalid times.
If all supported arms match: `TRANSFER_CHARACTERIZED` (descriptive, no speedup acceptance gate).
Call host strategy cost dominant only if its fraction of staged A0 total exceeds 50% in all three
repeats of the large fixture. This includes allocation/CPU copy overhead, not pure PCIe bandwidth.

## RC-R1b — resolution convergence

New path: `results/renderer_characterization_r1b_2026-09-14/`.
Three specimens, primary_grid's same 24 views (azimuth 0:45:315, elevation -20/0/35, distance 2 m).
Resolutions 320×240, 640×480, 1280×960; reference **2048×1536**, highest legal 4:3 resolution.
Render one view at a time to bound memory. Reference is sampled, not exact geometry ground truth.
All 288 rows retained or explicitly refused. Fewer than 300 visible pixels or border clipping →
refused; any refused required row prevents a measurement-resolution recommendation.

Per row: width/height (inclusive pixel extents), area, centroid and median optical depth.
Compare width/f, height/f, area/f² and centered centroid/f to corresponding reference quantities.
Report signed difference, absolute error and relative error; centroid absolute error is a 2D norm,
relative centroid error divides by reference equivalent diameter/f (not distance to image origin).
Also retain raw pixel quantities and scaled-reference values in the current pixel units.

Recommendation: smallest non-reference resolution satisfying **all views of all specimens**:
width/height/area absolute relative error ≤2%, relative centroid error ≤1%, depth relative error ≤1%.
If none qualifies: `NO_QUALIFIED_RESOLUTION`; still report all errors. Reference cannot pass by
comparison to itself and cannot be recommended without a finer reference. No bootstrap over pixels.
The measured finite-resolution discrepancy is a quantization estimate, not an accuracy guarantee
or the historical RC-R1 2% criterion rewritten. R2b uses fixed 1280×960 regardless of this result;
if that resolution is not qualified, its limitation accompanies R2b.

## RC-R2b — anisotropic generic area control

New path: `results/renderer_characterization_r2b_2026-09-14/`.
C0 = historic isotropic box with exact frozen scale 0.8450780426128714, never refitted.
C1 = three-axis-scaled box (sx,sy,sz), same procedural box topology. No ellipsoid arm: the existing
contract supports this simpler family directly. Objective is multi-view area, not shape imitation.
All renders 1280×960, distance 2 m. Fixed **new** fit views (azimuth,elevation):
(0,0),(90,0),(0,35),(90,35),(45,20),(135,20).
Held-out views: azimuth (22.5 + 45*k), k=0..7 × elevation -10/15/40 =24 views.
These exact angles are disjoint from calibration and the first RC primary grid; same specimen
family remains previously explored. Do not call this independent-object generalization.

Fit once: scipy least_squares on log(projected convex-box area / raster mesh area), normalized by
f²; compute convex-box area from the perspective-projected eight corners and 2D convex hull.
Bounds sx,sy,sz in [0.2,2.0], one initialization at C0 scale, max_nfev=200,
ftol/xtol/gtol=1e-10. The fit stage never renders/reads held-out views. Save frozen parameters and
their SHA before evaluating held-out views; no retry or model selection using validation.
Primary validation statistic: median absolute relative area error; secondary P90 and worst view.
`AREA_CONTROL_MATCHED` only if median≤5%, P90≤10%, worst≤20%, no refused view, successful fit.
Otherwise `AREA_CONTROL_FAILED` (or `FIT_FAILED`); never refit. Report fit and held-out separately.

## RC-R3b — normal-field characterization

New path: `results/renderer_characterization_r3b_2026-09-14/`.
640×480, same 24 primary views, three reference specimens, existing fixed grey/light.
For meshes N0 uses current geometric face normals. Analytic sphere normals are its analytic
reference, not interpolated mesh normals. Current MeshScene/source loader has no authored vertex
normal or smoothing-group contract: **N1 interpolated source vertex normals and N2 source smooth
normals are UNSUPPORTED**. Do not synthesize replacement normals or alter geometry to add them.

For each view record camera-frame normal entropy, occupied/unique bins (fixed 8 azimuth ×4 polar),
mean |n·view|, Lambertian silhouette luminance variance, interior gradient and object-boundary edge
contrast. Reuse existing luminance weights; edge contrast is mean absolute luminance difference
over horizontal/vertical valid↔invalid neighbouring pairs, counted once. Describe entropy/variance
relation with per-specimen scatter and Pearson r when both vary; otherwise r=null with reason.
`NORMAL_FIELD_CHARACTERIZED` is a descriptive completion status, no minimum variance/entropy
criterion and no causal effect estimate for unsupported normal interventions. R3 failure preserved.
Do not rerun R4/R4M grids; reference their recorded geometry invariance only.

## Integration of documentation, not perception

After four results: `docs/renderer_measurement_contract_v1.md` and
`docs/renderer_contract_v1_2026-09-14.md`, evidence map, exporter contract tests (no exporter feature
expansion), and five fresh figures under `docs/assets/paper/renderer-contract-v1-2026-09-14/`:
resolution convergence; held-out area control; normal/variance scatter; timing breakdown;
renderer contract diagram. SVG+PNG+vector PDF, no old figure overwrite. Site shows figures and
short captions. Include limits, unsupported arms and `causality_vs_d8b: NOT_TESTED`.
