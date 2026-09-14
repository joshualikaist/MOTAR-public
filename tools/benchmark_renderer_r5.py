"""R5: what does shaded appearance cost, against the same geometry pass it replaces?

The question this answers is narrow and stated before any number exists: how much slower is one
render iteration when a flat fill is replaced by normal-based Lambertian shading, and how does that
scale with batch and resolution. Geometry (two ray passes) is timed separately from shading, because
the plan's decision depends on which of the two dominates.

This measures a standalone renderer on a fixed box fixture. It is NOT a simulator step rate, NOT a
policy frame rate, and NOT a training throughput. Nothing here loads aerial_gym, a task, a detector
or a policy, and nothing here renders a UAV asset.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

REPEATS = 3
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 10
BATCHES = (1, 8, 32)
RESOLUTIONS = ((160, 120), (240, 135), (480, 270))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    p.add_argument("--resolutions", type=str, nargs="+", default=["%dx%d" % r for r in RESOLUTIONS])
    return p


def tracked_tree_clean():
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0 and subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0


def parse_resolution(text):
    width, _, height = text.partition("x")
    if not height or not width.isdigit() or not height.isdigit():
        raise ValueError("resolution must be WIDTHxHEIGHT")
    return int(width), int(height)


def timed(function, device, torch, iterations):
    """Wall-clock per iteration, with the device synchronised on both sides of every sample."""
    samples = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return samples


def summarise(samples):
    ordered = sorted(samples)
    return {"n": len(ordered), "mean_ms": 1000 * statistics.fmean(ordered),
            "p50_ms": 1000 * statistics.median(ordered),
            "p95_ms": 1000 * ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
            "min_ms": 1000 * ordered[0], "max_ms": 1000 * ordered[-1]}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new result directory")
    # The first R5 campaign recorded no commit and ran before this file was committed. A timing
    # number nobody can tie to a source tree is not a result; refuse rather than repeat that.
    if not tracked_tree_clean():
        raise RuntimeError("R5 requires a committed, tracked-clean source tree")
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError("Run this standalone benchmark outside any aerial_gym process")
    resolutions = [parse_resolution(text) for text in args.resolutions]
    if any(b < 1 or b > 128 for b in args.batches):
        raise ValueError("batch sizes must lie in [1,128]")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import torch
    from renderer_validation.scene import Camera, box_fixture, sample_appearance
    from renderer_validation.gbuffer import WarpGBufferRenderer
    from renderer_validation.shading import shade
    from runtime_fingerprint import runtime_fingerprint

    scene = box_fixture()
    device = torch.device(args.device)
    rows = []
    for width, height in resolutions:
        for batch in args.batches:
            camera = Camera(width=width, height=height)
            renderer = WarpGBufferRenderer(scene, camera, num_scenes=batch, device=args.device)
            positions = np.zeros((batch, 3), dtype=np.float32)
            orientations = np.tile(np.array([0, 0, 0, 1], np.float32), (batch, 1))
            renderer.set_camera_poses(positions, orientations)
            appearance = sample_appearance(args.seed, batch, scene.material_count)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(WARMUP_ITERATIONS):          # JIT, kernel cache and allocator warm-up
                shade(renderer.render(), scene, appearance, mode="lambertian")
            geometry, flat, lambertian = [], [], []
            for repeat in range(REPEATS):
                geometry += timed(renderer.render, device, torch, MEASURED_ITERATIONS)
                buffer = renderer.render()
                # Order alternates across repeats so a warming trend cannot favour one arm.
                arms = [("flat", flat), ("lambertian", lambertian)]
                for name, sink in (arms if repeat % 2 == 0 else arms[::-1]):
                    sink += timed(lambda n=name: shade(buffer, scene, appearance, mode=n),
                                  device, torch, MEASURED_ITERATIONS)
            total_flat = statistics.fmean(geometry) + statistics.fmean(flat)
            total_lambertian = statistics.fmean(geometry) + statistics.fmean(lambertian)
            row = {"width": width, "height": height, "batch": batch,
                   "pixels_per_iteration": width * height * batch,
                   "triangles": int(len(scene.triangles)),
                   "geometry": summarise(geometry), "shade_flat": summarise(flat),
                   "shade_lambertian": summarise(lambertian),
                   "iteration_flat_ms": 1000 * total_flat,
                   "iteration_lambertian_ms": 1000 * total_lambertian,
                   "lambertian_over_flat": total_lambertian / total_flat if total_flat else None,
                   "shading_share_of_lambertian_iteration":
                       statistics.fmean(lambertian) / total_lambertian if total_lambertian else None,
                   "images_per_second_lambertian": batch / total_lambertian if total_lambertian else None,
                   "iterations_per_second_lambertian": 1.0 / total_lambertian if total_lambertian else None}
            if device.type == "cuda":
                row["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                row["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 2
            rows.append(row)
            print("%4dx%-4d batch %3d  geometry %7.2f ms  flat %6.2f  lambertian %6.2f  ratio %.3f"
                  % (width, height, batch, row["geometry"]["mean_ms"], row["shade_flat"]["mean_ms"],
                     row["shade_lambertian"]["mean_ms"], row["lambertian_over_flat"]), flush=True)
            del renderer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ratios = [r["lambertian_over_flat"] for r in rows if r["lambertian_over_flat"]]
    receipt = {"status": "BENCHMARK_COMPLETE", "stage": "R5",
               "scope": "standalone renderer on a fixed box fixture; not a simulator, policy or training rate",
               "device": args.device, "seed": args.seed,
               "git_commit": subprocess.check_output(
                   ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               "tracked_tree_clean": True,
               "protocol": {"repeats": REPEATS, "warmup_iterations": WARMUP_ITERATIONS,
                            "measured_iterations_per_repeat": MEASURED_ITERATIONS,
                            "arm_order": "alternated across repeats",
                            "synchronised": args.device.startswith("cuda")},
               "slowdown_lambertian_over_flat": {"min": min(ratios), "max": max(ratios),
                                                 "median": statistics.median(ratios)},
               "rows": rows, "runtime": runtime_fingerprint()}
    args.output.mkdir(parents=True)
    (args.output / "benchmark.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("rows", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
