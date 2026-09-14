#!/usr/bin/env python3
"""Measure what the object-local dynamic-mesh path costs, against baselines measured here.

Three arms over one shared ray set, object pose sequence, static scene and scene count:
STATIC_ONLY, STATIC_PLUS_ANALYTIC_PROXY, STATIC_PLUS_DYNAMIC_LOCAL_MESH. Gates, the resource stop
rule and the verdict thresholds are fixed in
results/dynamic_mesh_raycast_feasibility_2026-09-12/PREREGISTRATION.md.

R5's numbers are not used as an expected result: they come from other fixtures and stages.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = ((160, 90), (320, 180), (480, 270))
SCENES = (1, 32, 128)
WARMUP, MEASURED, REPEATS = 3, 10, 3
RESERVED_LIMIT_GIB = 6.0
ARMS = ("STATIC_ONLY", "STATIC_PLUS_ANALYTIC_PROXY", "STATIC_PLUS_DYNAMIC_LOCAL_MESH")


def tracked_tree_clean():
    return all(subprocess.run(["git", "-C", str(ROOT), "diff"] + f + ["--quiet"]).returncode == 0
               for f in ([], ["--cached"]))


def summarise(samples):
    ordered = sorted(samples)
    return {"n": len(ordered), "mean_ms": 1000 * statistics.fmean(ordered),
            "median_ms": 1000 * statistics.median(ordered),
            "p95_ms": 1000 * ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
            "min_ms": 1000 * ordered[0], "max_ms": 1000 * ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("a new output directory is required")
    if not tracked_tree_clean():
        raise RuntimeError("this benchmark requires a committed, tracked-clean source tree")
    sys.path.insert(0, str(ROOT / "tools"))
    import numpy as np
    import torch
    import warp as wp
    from renderer_validation.dynamic_mesh import DynamicMeshRaycaster
    from renderer_validation.scene import asymmetric_box_fixture, l_shape_fixture, box_fixture
    from runtime_fingerprint import runtime_fingerprint

    static_scene, dynamic_scene = box_fixture(), l_shape_fixture()
    rows, stopped = [], None
    for scenes in SCENES:
        for width, height in RESOLUTIONS:
            if stopped:
                rows.append({"scenes": scenes, "width": width, "height": height,
                             "status": "NOT_RUN_RESOURCE_LIMIT", "reason": stopped})
                continue
            pixels = width * height
            focal = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            directions = np.stack([(x - width / 2.0) / focal, (y - height / 2.0) / focal,
                                   np.ones_like(x, float)], -1).reshape(-1, 3)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            origins = np.tile([0.0, 0.0, -1.4], (pixels, 1))
            origins = np.repeat(origins[None], scenes, 0)
            directions = np.repeat(directions[None], scenes, 0)
            angles = np.linspace(0.0, 1.2, scenes)
            positions = np.stack([0.1 * np.sin(angles), np.zeros(scenes), 0.3 + 0.05 * angles], 1)
            rotations = np.stack([np.zeros(scenes), np.sin(angles / 2), np.zeros(scenes),
                                  np.cos(angles / 2)], 1)
            caster = DynamicMeshRaycaster(static_scene, dynamic_scene, args.device, far_range_m=50.0)
            # Rays, poses and outputs live on the device across every sample, so the timed region
            # is the query alone rather than 22 MiB of host traffic per call.
            rays = caster.upload_rays(origins, directions)
            wp_positions, wp_rotations = caster.upload_poses(positions, rotations, scenes)
            outputs = caster.allocate_outputs(scenes, pixels)
            torch.cuda.reset_peak_memory_stats()
            cell = {"scenes": scenes, "width": width, "height": height,
                    "pixels_per_frame": pixels, "status": "OK", "arms": {}}
            try:
                for arm in ARMS:
                    # All three treatments run inside the one kernel, so the comparison
                    # isolates analytic test against mesh query rather than GPU against host.
                    mode = {"STATIC_ONLY": 0, "STATIC_PLUS_ANALYTIC_PROXY": 1,
                            "STATIC_PLUS_DYNAMIC_LOCAL_MESH": 2}[arm]

                    def once(mode=mode):
                        caster.cast_uploaded(rays, wp_positions, wp_rotations, outputs,
                                             use_static=True, dynamic_mode=mode)
                        wp.synchronize_device(args.device)
                    for _ in range(WARMUP):
                        once()
                    samples = []
                    for _ in range(REPEATS):
                        for _ in range(MEASURED):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            once()
                            torch.cuda.synchronize()
                            samples.append(time.perf_counter() - start)
                    cell["arms"][arm] = summarise(samples)
                    cell["arms"][arm]["frames_per_second"] = scenes / statistics.fmean(samples)
                cell["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
                cell["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 ** 2
                cell["process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if cell["torch_peak_reserved_mib"] / 1024.0 > RESERVED_LIMIT_GIB:
                    stopped = f"reserved {cell['torch_peak_reserved_mib']/1024:.2f} GiB exceeds {RESERVED_LIMIT_GIB} GiB"
            except torch.cuda.OutOfMemoryError as error:
                cell.update(status="OOM", reason=str(error)[:200])
                stopped = "out of memory"
            rows.append(cell)
            base = cell.get("arms", {}).get("STATIC_PLUS_ANALYTIC_PROXY", {}).get("median_ms")
            mesh = cell.get("arms", {}).get("STATIC_PLUS_DYNAMIC_LOCAL_MESH", {}).get("median_ms")
            print(f"  {scenes:3d} scenes {width}x{height}: {cell['status']}"
                  + (f"  proxy {base:.2f} ms  mesh {mesh:.2f} ms  x{mesh/base:.2f}"
                     if base and mesh else ""), flush=True)
            del caster
            torch.cuda.empty_cache()

    payload = {"schema": "dynamic_mesh_raycast_benchmark_v1", "stage": "D6",
               "device": args.device, "arms": list(ARMS),
               "protocol": {"warmup": WARMUP, "measured_per_repeat": MEASURED, "repeats": REPEATS,
                            "synchronised": True, "reserved_limit_gib": RESERVED_LIMIT_GIB},
               "git_commit": subprocess.check_output(
                   ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "tracked_tree_clean": True, "rows": rows, "runtime": runtime_fingerprint(),
               "interpretation": "Independent graphics feasibility. Says nothing about detector "
                                 "accuracy, shortcut reduction, association or policy performance."}
    args.output.mkdir(parents=True)
    (args.output / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output / "benchmark.json")


if __name__ == "__main__":
    main()
