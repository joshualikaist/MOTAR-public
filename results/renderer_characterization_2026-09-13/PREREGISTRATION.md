# Renderer characterization track (RC) — preregistration

Written 2026-09-13, **before** any of this track's code was written or run. Every threshold below
is fixed by this document. No threshold moves after a result is seen. A failing gate is recorded
as a failure; it is not repaired by changing the gate.

## 0. Name space, and what this track is not

This track's experiments are called **R1–R6**. The repository already contains an *earlier,
unrelated* appearance-prototype track whose experiments were also called R1–R5
(`results/renderer_r2_smoke_2026-09-11`, `renderer_r3_*`, `renderer_r4*`, `renderer_r5_benchmark*`,
`docs/preregistration_renderer_r4_2026-09-11.md`). The two name spaces are **not** the same
experiments and their numbers are never pooled or compared. Whenever ambiguity is possible this
track is written `RC-R1 … RC-R6`.

**Goal.** Quantify, generically and reproducibly, how this renderer represents geometry, apparent
area, material, shading and lighting.

**Explicitly out of scope.** Detector behaviour, tracking, association, policy performance,
training, adaptation, and any explanation of the D8b frozen-policy result
(`results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13`, verdict `MATERIAL_LOSS`, primary
estimand −48.967 pp, 95 % CI [−50.113, −47.821] pp). D8b is closed and preserved. This track
establishes **no** causal link to it: shading, mesh geometry, projected area and the detector are
all `NOT_TESTED` as causes of that loss. An evidence map may place RC measurements next to D8b for
orientation, and must carry `causality: NOT_TESTED` wherever it does.

No detector score, tracking score, learned model, policy, semantic label or ground-truth identity
enters any metric in this track. Labels (face id, instance id) are measured as renderer outputs and
are never inputs to shading.

## 1. Specimens (arms)

All specimens are centred on the world origin with identity orientation. Specimen identity is
geometric only; none of them carries an operational meaning here.

| Arm | Name | Definition |
|---|---|---|
| A0 | `analytic_sphere` | Sphere, radius **0.15 m**, intersected in closed form. No triangles. The value is the repository's documented analytic proxy radius (`camera_target_radius` default), used here purely as a geometric specimen. |
| A1 | `box_proxy` | Axis-aligned box, half-extents **(0.14, 0.14, 0.06) m**, 12 triangles, 1 material. The repository's documented box half-extents, again used only as a specimen. |
| A2 | `area_matched_box` | A1 scaled by one isotropic factor `s`, fitted in RC-R2 on the fit views only and then frozen. |
| A3 | `quadrotor_mesh` | `resources/models/environment_assets/objects/navrl_target_drone_v3.urdf`, urdfpy backend: **1548 triangles, 13 visual links, 4 materials**, bounding-box extent **(0.28256339, 0.28256339, 0.12) m**, centred on the origin. |

## 2. Camera and views

Pinhole camera, horizontal FOV **60.0°**, axes +X right / +Y down / +Z forward, `far_range_m`
**20.0**. Primary resolution **320×240**; resolution-consistency repeat **640×480** at the same
poses.

* `V_primary` — azimuth {0, 45, 90, 135, 180, 225, 270, 315}°, elevation {−20, 0, 35}°, distance
  **2.0 m**, camera aimed at the origin with zero roll: **24 views**.
* `V_dist` — azimuth {0, 45, 90}°, elevation {0, 35}°, distances {1.5, 2.0, 2.5, 3.0, 4.0} m:
  **30 views**.
* RC-R2 split, fixed here and disjoint: **fit views** = azimuth {0, 90, 180, 270}° at elevation 0°,
  distance 2.0 m (4 views); **validation views** = the remaining **20** views of `V_primary`.

## 3. RC-R1 — geometry correctness

Per view and arm: silhouette pixel count; silhouette bounding-box width and height (px);
equivalent circular diameter `2·sqrt(area/π)` (px); silhouette centroid (u, v) in px; depth
(camera +Z) min / median / max; visible triangle count; visible link count; camera-frame normal
histogram (8 azimuth × 4 polar bins) with Shannon entropy in bits and dominant-bin fraction; mean
|n·view direction|.

Gates (all must hold):

* **G1 cross-implementation.** For A1 and A3, the Warp G-buffer and an independent CPU
  Möller–Trumbore reference, given the *same* rays: silhouette-mask disagreement
  **≤ 0.5 %** of the reference silhouette pixels per view, and on pixels where both hit,
  max |range difference| **≤ 1.0e-3 m**.
* **G2 analytic sphere closed form.** Measured equivalent silhouette radius vs the exact projected
  radius `r·f/sqrt(z²−r²)` within **1.0 px**; silhouette centroid within **0.5 px** of the
  projected centre; measured minimum depth within **2.0e-3 m** of `z − r`.
* **G3 box closed form.** Measured silhouette bounding box agrees with the bounding box of the
  eight projected corners to within **1.0 px** on every side.
* **G4 inverse-distance scaling.** For each arm and each (azimuth, elevation) of `V_dist`,
  `equivalent diameter × distance` is constant across the five distances within **2.0 %**
  (max/min − 1 ≤ 0.02).
* **G5 depth bracket.** For every view and arm, minimum depth ≥ `d − R_circ − 2 mm` and maximum
  depth ≤ `d + R_circ + 2 mm`, where `R_circ` is the specimen's circumscribed radius.
* **G6 degenerate-view refusal.** A view with fewer than **300** silhouette pixels, or fewer than
  **3** visible triangles for a mesh arm, is refused by raising. Such a view is never reported as a
  measurement.
* **G7 resolution consistency.** `equivalent diameter / focal length` agrees between 320×240 and
  640×480 within **2.0 %** per view and arm.

Verdict: `GEOMETRY_VERIFIED` if G1–G7 all pass, else `GEOMETRY_DEFECT` naming each failed gate.

## 4. RC-R2 — apparent area

Primary estimand: **mean silhouette area in pixels per arm over the 20 validation views** at
320×240, distance 2.0 m. Reported with every per-view value, and as ratios A0/A3, A1/A3, A2/A3.

The A2 scale is fitted **only** on the 4 fit views as `s = sqrt(mean area(A3, fit) / mean
area(A1, fit))`, then frozen and applied unmodified to the validation views.

* **Area-match criterion.** `|mean area(A2, validation) / mean area(A3, validation) − 1| ≤ 5.0 %`
  → `AREA_MATCHED`; otherwise `AREA_MATCH_FAILED`. No refit after seeing validation views.

RC-R2 reports the apparent-area gap between the arms and whether an isotropic box scale removes it.
It states nothing about any detector, policy or the D8b result.

## 5. RC-R3 — shading, with geometry held fixed

Each cell renders **one** G-buffer and shades it several ways, so camera, pose, geometry and depth
are identical across shading modes by construction rather than by comparison.

* **S1 invariance.** `range`, `depth`, `normal`, `face`, `instance`, `valid` hashes byte-identical
  across all shading modes of a cell.
* **S2** flat shading with one shared colour: silhouette luminance variance **≤ 1e-12**.
* **S3** lambertian shading: silhouette luminance variance **≥ 1e-4**.
* **S4** reported generic image statistics, over silhouette pixels and over the full frame
  separately: luminance mean, std, min, max, p05, p50, p95; RMS contrast; mean Sobel gradient
  magnitude; 64-bin luminance histogram entropy (bits); median within-depth-quintile luminance std.
  ITU-R BT.709 linear-light luminance weights.

## 6. RC-R4 — lighting, and RC-R4M — material, separated

RC-R4: arm A3, the 4 fit views, 320×240, grid of **6** light directions × ambient {0.05, 0.2, 0.35}
× directional {0.4, 0.8} = **36 cells**.

* **L1** `range`/`depth`/`normal`/`face`/`instance`/`valid` hashes and silhouette pixel counts
  identical in all 36 cells.
* **L2** across cells, max − min of silhouette mean luminance **≥ 0.02**.
* **L3** every cell's RGB finite and inside [0, 1].

RC-R4M, run and reported separately: **5** material sets at fixed lighting, same arm and views.
**M1** geometry hashes identical; **M2** silhouette mean-luminance spread **≥ 0.02**;
**M3** RGB finite and in [0, 1].

## 7. Determinism gate

The exporter is run **twice in two independent OS processes** with identical arguments. Required:
identical sha256 for `rgb`, `depth_m`, `range_m`, `normal_world`, `face_id`, `instance_id`, `valid`,
and identical metadata apart from the runtime fingerprint, timestamps and output paths. Verdict
`DETERMINISTIC` or `NONDETERMINISTIC`; a mismatch is reported as found, with the differing fields.

## 8. RC-R5 — performance (descriptive, no verdict)

Six configurations: (1) A0 analytic, (2) A1 geometry only, (3) A1 + flat shading, (4) A2 +
lambertian, (5) A3 geometry only, (6) A3 + lambertian. Matrix: resolutions {160×90, 320×240,
640×480} × scene counts {1, 8, 32}. **Warmup ≥ 20** renders, **measured ≥ 100** renders per cell.

Per-stage medians reported: normal/face pass, range pass, G-buffer finalize, shading, host copy.
The headline total is measured **without** per-stage synchronization; the instrumentation overhead
(synchronized total − unsynchronized total) is reported as its own number.

Memory is reported as five independent fields — `torch_allocated_mib`, `torch_reserved_mib`,
`warp_mib`, `nvml_used_mib`, `rss_mib`. An unavailable field is the string **`UNAVAILABLE`** and is
**never** written as 0.

**Resource stop rule.** A cell is `SKIPPED_RESOURCE_STOP` if projected output bytes exceed
**1.5 GiB**, or a single warmup render exceeds **2.0 s**, or torch reserved memory exceeds
**5.0 GiB**. Remaining cells still run.

RC-R5 is descriptive and carries no pass/fail verdict. It is **not** compared with D6 (1.5338×
ratio, INCONCLUSIVE) or D7 (+0.411 ms, GO): different denominators, different code paths.

## 9. RC-R6 — dataset export

`tools/export_renderer_dataset.py` gains the characterization arms. Its existing
`--geometry {boxes, background}` behaviour is unchanged, byte for byte, when no new option is used.
Five exporter validation checks:

1. identical arguments twice, in separate processes, give identical decoded arrays;
2. a lighting change alters `rgb` and leaves every geometry hash identical;
3. a material change alters `rgb` and leaves every geometry hash identical;
4. instance-label renumbering leaves `rgb` identical and changes `instance_id`;
5. every receipt sha256 matches the file bytes, and the decoded arrays match the receipt.

Plus: arm/view metadata completeness, and refusal when the output directory already exists.

## 10. Result layout, figures, and discipline

Each experiment directory under `results/renderer_characterization_2026-09-13/` holds
`PREREGISTRATION.md` (a pin to this document plus the thresholds actually read from code at run
time), `README.md`, `config.json`, `receipt.json`, `source_manifest.json`, `summary.json`.

Figures RC-R1 … RC-R6 are written to
`docs/assets/paper/renderer-characterization-2026-09-13/` as SVG source plus PNG and vector PDF
exports, on a white background, with at most three colours, no gradients, and **new filenames**.
No existing hash-pinned figure, ZIP or archived HTML is overwritten.

Discipline carried from the repository's standing rules: no threshold changes after results; failed
results preserved; no test deleted or skipped to turn the suite green; no force push; no rewriting
of existing commits.

---

## Amendment 1 — 2026-09-13, before any RC experiment was run

**What changes.** G6's minimum visible-triangle count goes from **3 to 2**. Nothing else changes:
the 300-pixel silhouette minimum, every RC-R1 tolerance, the RC-R2 tolerance and split, and the
RC-R3 variance thresholds S2/S3 are all untouched.

**Why.** The count was written as a proxy for "enough visible structure to measure", with a
tessellated specimen in mind. A box is not tessellated: one of its faces is exactly two triangles.
Measured on the primary grid, the A1 box shows 2 visible triangles at azimuth 0/90/180/270 with
elevation 0 — which are precisely the four RC-R2 fit views — and 4 or 6 triangles everywhere else.
At 3, G6 would refuse the views the RC-R2 scale is fitted on, and the A1/A2 arms could not be
measured on the same view set as A3 at all. Those four views are not degenerate: their silhouettes
hold 697–1521 pixels, far above the 300-pixel minimum, and a fully visible planar face is a
complete measurement of that face.

**Direction of the change, stated so it can be checked.** Lowering the count *admits* views rather
than excluding them, and the views it admits are the least structured ones in the grid. Every
admitted view must still pass G1, G3, G4, G5 and G7. It cannot convert a failing gate into a passing
one, and it cannot affect RC-R3's S2/S3 at all.

**What this amendment explicitly does not do.** It does not touch S3. The four axis-aligned box
views show a single planar face, whose lambertian luminance is spatially constant by construction,
so S3 (silhouette luminance variance ≥ 1e-4) is expected to **fail** on them. That failure is a
measurement of the renderer's behaviour and will be reported as a failure, per view, with its
geometric reason. The threshold stays at 1e-4.

**Status.** Written and committed before any RC experiment was executed. No RC result existed at the
time of this amendment; what existed was the implementation-time observation of visible-triangle
counts recorded above.
