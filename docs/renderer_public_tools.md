# Independent public graphics tools

Only procedural static boxes and the existing generic floor/wall/column/panel background are
accepted. No arbitrary asset loader flag, UAV assets, detector, policy or simulator is exposed.
These tools do not change the existing camera kernels or historical renderer results.

## Environment doctor

```bash
python -B tools/motar_doctor.py --profile renderer-cpu
python -B tools/motar_doctor.py --profile simulator
```

CPU PASS means checked dependencies/origins, not a rendered frame. The simulator mode is a
read-only inventory: UNAVAILABLE or INVENTORY_ONLY, never an execution certification. It does
not import Isaac Gym, install/repair dependencies or print environment credentials.

## Dataset export and validation

Use the isolated [CPU environment](renderer_cpu_quickstart_2026-09-12.md). Output paths must be new.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B tools/export_renderer_dataset.py \
  --geometry boxes --device cpu --material-seed 0 --light-seed 0 \
  --width 160 --height 90 --num-scenes 1 --frames 1 --output /tmp/motar-generic-export-new
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B tools/validate_renderer_export.py \
  --output /tmp/motar-export-checks-new
```

The exporter also accepts `background`, a camera position and normalized xyzw quaternion.
Output NPZ stores float32 linear RGB, optical depth/ray range in metres, world normals,
face/instance IDs and a validity mask. IDs are separate debug/label data, never model input.
Material and light seeds are independent and fixed throughout an export; frames repeat the
static scene, not independent trajectories. This is not a training dataset split.

Validation uses five actual exporter processes: repeat seed, changed light, changed material
and instance renumbering. It checks decoded hashes, unchanged geometry, changed RGB where
expected, and independence of RGB from debug identity. File SHA/shape/dtype are checked too.
Existing outputs are refused, and failures leave an explicit incomplete record.

## Full-loop static graphics benchmark

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B tools/benchmark_renderer_pipeline.py \
  --device cpu --geometries boxes background --resolutions 160x90 --batches 1 \
  --warmup 3 --iterations 20 --seed 0 --output /tmp/motar-generic-benchmark-new
```

Run from a **committed clean checkout**. Setup (procedural construction, renderer/BVH and
appearance) and first frame are measured separately. Warm steady samples include geometry,
normal/face lookup, material lookup, shading, background and CPU-owned output-image buffers.
They exclude compression/disk IO and all application/simulator work. Mean, median and nearest-rank
P95 use actual per-iteration samples, not sums of separate stage means.

The CLI supports the requested 160×90/320×180/480×270 and 1/32/128 matrix for the **two generic
fixtures** (18 cells), subject to a 1 GiB raw-buffer budget per call. Memory overhead exceeds raw
size. A quadrotor/target-specific benchmark and area-matched target box are BLOCKED_BY_POLICY
in this project context; no replacement with a renamed operational asset is performed.

Torch allocated/reserved peaks cover setup through measured loop. They exclude Warp allocations
and are not total CUDA VRAM. RSS current and process-lifetime high-water are separately labelled;
later cells inherit the lifetime RSS high-water. Total process CUDA VRAM is null when unmeasured.
The CPU implementation has no CUDA allocation figure. JIT/cache effects can enter the first frame.

Standalone full-loop results do **not** close `INTEGRATED_RENDER_COST_UNMEASURED` for the simulator.
No FPS, safety, association, shortcut-reduction or deployment criterion is inferred from these timings.
Execution receipts and campaign status are indexed by [the status manifest](status_manifest.json).
