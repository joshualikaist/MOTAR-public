# D8 preregistration — mesh-derived target observation

Date frozen: 2026-09-13, before D8 treatment code or D8 output exists.

This is a simulation-only renderer/perception experiment. It does not authorize a new PPO run,
controller change, real-flight claim, target-identity claim, or deployment work. D6 remains
`INCONCLUSIVE`; D7 `GO` applies only to shadow cost and output non-interference.

## 1. Question and calculation-first prediction

Can the visual geometry of `navrl_target_drone_v3.urdf` produce a deterministic, occlusion-aware
target mask/depth observation in the existing camera path without violating the D7 cost budget?
What changes when geometry is separated from simple within-object shading?

Existing evidence predicts a non-zero geometry effect. An independent 21-view prototype placed
the v3 projected area at 54–70% of the solid-box area, whereas the old in-simulator probe produced
an area ratio of exactly 1.000 because both arms still used the analytic box. At 160×90, raster
quantisation may make the observed ratio discrete and need not equal 54–70%. No direction or
magnitude is a success gate.

## 2. Geometry contract — option C, explicit separation

Three geometries keep different meanings and are not silently unified:

| Role | Frozen definition |
|---|---|
| Historical collision | v2/v3 collision box `0.283 × 0.283 × 0.12 m`; unchanged |
| Historical sensor baseline | analytic OBB half-extents `(0.14, 0.14, 0.06 m)`; unchanged |
| D8 treatment | v3 URDF **visual** mesh in target-local coordinates |

The v3 URDF SHA-256 at preregistration is
`c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0`.
The target remains excluded from the static Warp scene. D8 transforms each ray into the target
frame and therefore does not rebuild a BVH as the target moves. The static scene remains the
occluder. Debug normal, face and material buffers may be recorded but must not enter actor input.

## 3. Arms and single-axis attribution

The runtime default remains `analytic_flat`. D8 adds two explicitly attached treatments:

1. `analytic_flat`: historical analytic OBB and uniform nominal target colour.
2. `mesh_flat`: v3 visual mesh mask/depth; the same uniform nominal target colour.
3. `mesh_shaded`: the same v3 mesh mask/depth as arm 2; fixed Lambertian intensity and the
   URDF material's relative luminance modulate the same nominal target hue.

Thus arm 1→2 changes geometry only, and arm 2→3 changes shading only. Material colour does not
replace the target with an unrelated gray object. There is no randomization in D8; light and
material coefficients are receipt fields. Random material/light variation belongs to a later
preregistered D9 experiment.

## 4. D8-A technical fixture and metrics

The primary integrated cell is 128 environments at 160×90, with 50 warm-up and 500 measured
steps, seed `20260913`, physical v2 collision geometry, deterministic step-index actions, and no
distractors or appearance perturbations. A smaller controlled pose fixture must cover at least
21 target poses across three distance bands `[4,5)`, `[5,6)`, `[6,7] m` and more than one target
orientation. Runs are separate child processes.

Required outputs:

- source commit/dirty state, runtime fingerprint, URDF and detector source hashes;
- mask/depth/RGB hashes, target and robot pose hashes, and repeat hashes;
- per-arm pixel count, centroid, bounding box, mean surface depth and segmenter-visible count;
- mask IoU, area ratio and centroid/depth differences for both planned contrasts;
- mesh hit, scene-occluded-hit, invalid-depth, invalid-face/material and invalid-normal counts;
- step mean/median/P95, steps/s, torch allocator memory, NVML device memory and process RSS.

No adjacent frame is treated as an independent inferential sample. D8-A is an engineering gate;
the controlled pose is the unit for geometry summaries and the child run is the unit for repeat
identity. It reports descriptive distributions, not p-values.

## 5. Frozen D8-A verdict

`TECHNICAL_GO` requires all of the following:

1. With D8 unset/off, the module is not imported, no treatment buffer is allocated, and the
   historical source/output regression tests pass.
2. Every treatment repeat has identical output hashes; both repeats use the same source and
   runtime fingerprint.
3. Every hit has finite depth strictly inside the far plane, a valid face/material id, and a
   finite unit normal within `|norm-1| ≤ 1e-3`.
4. Scene-occluded target hits never survive in the published target mask.
5. At least one mesh pixel is produced in every preregistered distance band, and
   `mesh_flat` differs from `analytic_flat` in at least one controlled pose. This prevents an
   unwired treatment; it does not require improvement or the predicted area ratio.
6. In `mesh_shaded`, target-pixel RGB variance is greater than zero in at least one pose, while
   `mesh_flat` target-pixel RGB variance is zero apart from floating-point representation.
7. On the primary integrated cell, relative median step increase is ≤10%, absolute median
   increase is ≤5 ms, throughput loss is ≤10%, torch reserved increase is ≤256 MiB, and NVML is
   available and recorded. These are the D7 `GO` limits, reused before seeing D8 results.

`TECHNICAL_NO_GO` is issued if any integrity item 1–6 fails or if relative increase exceeds 30%,
absolute increase exceeds 20 ms, or torch reserved increase exceeds 1024 MiB. Results between the
GO and NO-GO cost regions are `TECHNICAL_INCONCLUSIVE`. A correctness failure cannot be rescued by
fast timing.

## 6. D8-B frozen sensitivity — gated, no adaptation

D8-B may run only after a committed D8-A `TECHNICAL_GO`. It uses the same frozen detector and
policy artifacts in every arm and reports `analytic_flat`, `mesh_flat`, and `mesh_shaded` as
separate treatments. It asks how an existing lineage responds to an observation change; it does
not measure adaptation. Checkpoint SHA, seeds, cell grid, episode count and confidence-interval
unit must be frozen in a separate addendum before launch. Test or historical result thresholds
must not be tuned after opening an arm.

## 7. Adaptation and D9 boundary

No learning is authorized here. Adaptation is considered only if D8-B establishes a practically
material loss or a stated failure mechanism, and then requires a separate multi-seed training
preregistration. D9 shortcut remeasurement is also separate: it must vary distractor count,
compare correct/distractor/ghost/no association, separate initial acquisition from temporal
maintenance, and reserve motion variation for a held-out diagnostic. D8 results cannot be used to
change D9's hypothesis or verdict thresholds after the fact.

## 8. Stop rules and preservation

- Historical masks, results and status files are never overwritten.
- Any unknown treatment value fails closed.
- Detect-resolution decoupling is refused until the mesh treatment implements that resolution;
  silently falling back to the analytic high-resolution path is forbidden.
- A dirty runtime, source/asset hash mismatch, missing NVML primary-cell reading, or debug buffer
  entering the observation invalidates the run.
- Unexpected results are reported under the frozen criteria; the criteria are not edited.
