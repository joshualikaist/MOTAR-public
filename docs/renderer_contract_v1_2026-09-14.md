# Independent Renderer Contract v1 — 14 September 2026

Static geometry → G-buffer → linear-RGB appearance → measured/exported arrays.
This contract does not connect a detector, association module, policy or simulator task.
**`causality_vs_d8b = NOT_TESTED`**. It is a characterization contract, not a blanket renderer PASS.

## Evidence and preserved failures

The [first RC lineage](../results/renderer_characterization_2026-09-13/) was published at
`32ee105`; its sources, receipts, arrays, figures and negative verdicts remain unchanged.
The [follow-up preregistration](preregistration_renderer_followup_2026-09-14.md) was committed
at `3e6d074` before implementation and these new measurements.

| Question | Follow-up evidence | Result / boundary |
| --- | --- | --- |
| Resolution convergence | [R1b](../results/renderer_characterization_r1b_2026-09-14/README.md) | `MEASUREMENT_RESOLUTION_QUALIFIED`, 1280×960 on the fixed grid |
| Multi-view area control | [R2b](../results/renderer_characterization_r2b_2026-09-14/README.md) | `AREA_CONTROL_FAILED`; no accepted area-matched control |
| Visible normal field | [R3b](../results/renderer_characterization_r3b_2026-09-14/README.md) | `NORMAL_FIELD_CHARACTERIZED`; N1/N2 unsupported |
| Compute versus host strategy | [R5b](../results/renderer_characterization_r5b_2026-09-14/README.md) | `TRANSFER_CHARACTERIZED`; seven output hashes invariant |
| Lighting and material invariance | [R4](../results/renderer_characterization_2026-09-13/R4_lighting/README.md), [R4M](../results/renderer_characterization_2026-09-13/R4M_material/README.md) | Existing evidence reused; grids not rerun |
| Repeated output determinism | [Historical export](../results/renderer_characterization_2026-09-13/determinism/README.md), R5b | Recorded source/config/environment only |
| Downstream causal contribution | No such experiment in this track | **NOT_TESTED** |

Original RC-R1 `GEOMETRY_DEFECT`, RC-R2 `AREA_MATCH_FAILED`, RC-R3 `SHADING_GATE_FAILED`
remain valid historical outcomes. The new studies do not reinterpret them into PASS.

## Geometry and appearance boundaries

Geometry and camera pose determine surface hits, silhouette, range, optical depth, geometric
normal and surface/instance indices. A geometry/pose change may alter these buffers; symmetric
changes can leave particular projections unchanged. Do not assert that every pose change must
change every hash. The specimen origin stays fixed in the independent multi-view experiments.

Material or light changes are applied only to the existing G-buffer. On the recorded fixed grids,
they change RGB/luminance while all six geometry/debug buffers remain byte-identical.
This is tested by R4/R4M and exporter regression, not assumed for arbitrary future shaders.

The unchanged shader is:

```text
RGB = clamp(material_color × (ambient + kd × directional × max(0, n_world · light_world)), 0, 1)
```

Misses take the configured background. No PBR, cast-shadow solver, authored smooth normals,
model inference, physical dynamics or image-driven control was added.

## Exported buffers and metadata

[Machine-readable buffer contract](renderer_buffer_contract_v1.json), checked by
[`test_renderer_contract_v1.py`](../tests/test_renderer_contract_v1.py) against actual decoded output
from the unchanged [`export_renderer_dataset.py`](../tools/export_renderer_dataset.py).

| Buffer | Layout / dtype | Meaning | Miss |
| --- | --- | --- | --- |
| RGB | NHWC / float32, C=3 | Linear RGB in [0,1], not gamma-encoded display RGB | Configured background |
| range_m | NHW / float32 | Euclidean camera-ray distance, metres | 0 |
| depth_m | NHW / float32 | Camera +Z optical depth, metres | 0 |
| normal_world | NHWC / float32, C=3 | Unit world-space geometric face normal; analytic normal for sphere | Zero vector |
| face_id | NHW / int32 | Scene triangle index; sphere uses analytic surface slot 0 | −1 |
| instance_id | NHW / int32 | Debug/evaluation object label only | −1 |
| valid | NHW / bool | Valid surface hit within ray-range cutoff | false |

Camera axes are +X right, +Y down, +Z forward; pose is camera-to-world, quaternion xyzw.
This independent view grid defines world up as −Y. These conventions must not be silently
equated with simulator or dataset pose conventions. Depth equals ray range times the optical
projection factor; integer pixel samples and principal point `(width/2,height/2)` are explicit.

RC arm exports have black background. Generic procedural exports use `(0.04,0.04,0.04)`.
Do not conflate these two existing export schemas. Metadata records source hashes, runtime,
camera/view grid, appearance, array shape/dtype/SHA and file SHA. Export status remains
`EXPORTED_UNASSESSED`, experiment verdict `NOT_EVALUATED`, training `NOT_SUPPORTED`.

**Historical wording correction:** `CharacterizationCell.description()` and the old shader
docstring overstate label exclusion. `face_id` actually indexes the face-to-material table inside
shading. `instance_id` does not enter shading. The old source/receipt strings are preserved for
reproduction, but must not be quoted as excluding face IDs from material lookup. New tests show
both boundaries explicitly. Neither label is supplied to a learned model by this track.

## Supported normal contract

N0 uses face normals derived from existing mesh geometry. The analytic sphere is a separate
analytic-normal reference, not an interpolated mesh. MeshScene carries vertices, triangles,
face_material and face_instance; the current loader contract carries no authored vertex normals
or smoothing groups. N1/N2 are therefore **UNSUPPORTED**, not zero-effect experimental arms.

R3b, 24 views per specimen at 640×480, fixed camera-relative light/material:

| Specimen | Median normal entropy (bits) | Occupied normal bins, min–max | Median luminance variance | Descriptive Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Analytic sphere | 3.99149 | 16–16 | 0.0109250 | −0.80985 |
| Box | 0.89306 | 1–3 | 0.0071478 | +0.57161 |
| Quadrotor mesh | 1.25934 | 1–14 | 0.0053678 | +0.59242 |

Entropy uses the frozen camera-frame 8 azimuth × 4 polar histogram. The raw rows additionally
record mean |n·view|, boundary contrast and interior gradient. These are pixel-weighted statistics
within each view; views, not pixels, are the reporting units. No independence-based CI is claimed.

The sphere's luminance variance changes by only about **1.64×10⁻⁹** across views. Its calculated
Pearson r is retained under the preregistered formula, but is not a meaningful physical trend.
The mesh has higher median entropy yet lower median luminance variance than the box. Entropy
alone and triangle count alone do not establish a universal shading law. R3b is descriptive;
it does not causally compare face versus smooth normals.

## Measurement contract

Use [measurement contract v1](renderer_measurement_contract_v1.md): 1280×960, only within the
tested coverage. Worst measured width/height/area discrepancies are 1.217/0.483/1.143% versus
sampled 2048×1536. No accepted area control exists: C1 held-out median/P90/worst relative
area errors are 8.132/11.051/11.544%, failing the unchanged 5/10/20% gates.

## Performance contract

R5b used the same quadrotor mesh, fixed appearance and view grid for all four transfer strategies.
Two fixtures × four arms × three fresh processes; 20 warmup, 100 staged and 100 unstaged samples
per cell. All seven output hashes match across arms/repeats and first/last validation exports.

640×480×32: arithmetic means of the three repeat means (ms). Headline totals come from a
separate unstaged pass; they are not sums of component medians or staged totals.

| Arm | Geometry | Shading | Host strategy | Headline total | Mean of repeat P95 totals |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 individual full owned copy | 16.879 | 7.043 | 178.620 | 205.215 | 209.260 |
| A1 resident + scalars only | 15.559 | 6.922 | 2.042 | 24.348 | 25.670 |
| A2 dtype-grouped copy | 15.422 | 6.917 | 109.554 | 131.582 | 136.186 |
| A3 pinned async + completion wait | 15.293 | 6.912 | 17.362 | 39.592 | 41.398 |

Raw samples and per-repeat mean/median/P95/scene throughput for both fixtures are in
[R5b summary.json](../results/renderer_characterization_r5b_2026-09-14/summary.json).
Mean-of-repeat-P95 is explicitly not pooled P95 or a confidence interval.

A0's host strategy is 88.02/88.08/88.46% of staged total: >50% in all three large-fixture repeats.
This supports **host strategy cost dominant**, not a pure PCIe bandwidth bottleneck claim.
A2 includes device packing. A3 includes transfer completion, but no compute/copy overlap.
A3 output views reuse pinned storage: consume/copy them before the next transfer. A0 owns its
arrays. A1 is not full export and its speed must not be advertised as full dataset-export speed.
One-time setup/allocation and terminal validation export are reported separately. Compression
and disk writing are outside these timings. No production renderer/exporter was switched to A3.

Large-fixture memory observations (MiB, across repeats):

| Arm | Process RSS | NVML device used | Torch peak reserved |
| --- | ---: | ---: | ---: |
| A0 | 1228–1247 | 2484–2492 | 1546 |
| A1 | 1227–1236 | 2409–2414 | 1468 |
| A2 | 1237–1254 | 2773–2784 | 1842 |
| A3 | 1370–1371 | 2477–2492 | 1546 |

RSS/NVML are end-of-cell observations, not continuous peaks. NVML and Warp's device-wide
used-memory observation include non-Torch allocations and desktop processes; they are not
per-Warp allocation counters. `warp_only_mib = null` is deliberately unknown, not zero.
Torch allocated/reserved and Torch peak counters are separately retained in each receipt.

Historical R5 contains the known RSS file-handle ResourceWarning. R5b fixes the reader in a
new harness, with lifetime regression tests; the old source and warning remain untouched.
Old R5 host-copy timing also included shading inside `cell.arrays`; it cannot be treated as
identical to the newly separated transfer-stage measurement.

## Reproduce and verify

No new training, detector, policy or R4/R4M grid run is required. Do not overwrite result folders.
Recorded environment: aerialgym Python, Torch 2.4.1+cu121, Warp 1.0.0, RTX 3070;
full environment flags/versions are in each receipt. Historical source commits:

| Study | Source commit | Result commit |
| --- | --- | --- |
| R5b | `dd0fd4f` | `ad414f6` |
| R1b | `901d4d8` | `be631bf` |
| R2b | `be631bf` | `a19d9fe` |
| R3b | `a19d9fe` | `62865ac` |

From the repository root, current evidence can be verified without rerendering:

```bash
PYTHONNOUSERSITE=1 python \
  tools/measure_renderer_contract.py verify --output results/renderer_characterization_r1b_2026-09-14
PYTHONNOUSERSITE=1 python -m unittest discover \
  -s tests -p 'test_renderer_followup_evidence.py' -v
```

The evidence tests independently recompute all R1b errors, R2b fit/held-out statistics, R3b
histogram entropy/correlations, and R5b timing summaries; they also verify all receipt files,
historical Git blobs and preregistration ancestry, and byte-preservation of first-RC evidence.
For an authorized fresh reproduction, use a detached worktree at the recorded source commit,
the recorded environment and a new output directory:

```bash
# At the study's recorded source commit, never in an existing result directory:
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python tools/benchmark_renderer_transfer.py run --output /tmp/rc-r5b-new
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python tools/measure_renderer_contract.py r1b --output /tmp/rc-r1b-new
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python tools/measure_renderer_contract.py r2b --output /tmp/rc-r2b-new
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python tools/measure_renderer_contract.py r3b --output /tmp/rc-r3b-new
```

The four commands belong to their respective source revisions, not one common checkout.
R2b additionally reads the committed R1b summary at its original path (included by `be631bf`),
and pins that evidence hash. Replace `python` only with the matching recorded interpreter.
These are reproduction instructions, not authority to reopen policy experiments.

## Paper/site outputs and next boundary

[Figure page](status/renderer-contract-v1.html): five fresh figures, each SVG + PNG + vector PDF,
under `docs/assets/paper/renderer-contract-v1-2026-09-14/`. The figure manifest pins result hashes,
generator source and rendered assets. Original pinned figures are not overwritten.

Evidence map: geometry, apparent area, normals, shading, lighting and material are characterized
within their stated coverage. **Downstream causal attribution remains NOT_TESTED.**
No finding here establishes that area mismatch, shading or mesh geometry caused D8b loss.

This scope is complete. A richer area-control family or a source-authored normal contract would
need a new preregistration, separate from these held-out results; neither is silently selected or
implemented here. Light/material grids remain frozen, and D8c/D9 are not opened by this contract.
