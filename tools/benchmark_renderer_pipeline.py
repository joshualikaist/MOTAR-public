#!/usr/bin/env python3
"""End-to-end STATIC GENERIC renderer timing, not a simulator/target integration benchmark."""
import argparse
import gc
import math
from pathlib import Path
import resource
import statistics
import time


def stats(samples, count):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Positive finite timing samples required")
    ordered = sorted(samples)
    mean = statistics.mean(samples)
    return {"samples_seconds": samples, "mean_ms": mean * 1000,
            "median_ms": statistics.median(samples) * 1000,
            "p95_ms": ordered[math.ceil(.95 * len(samples)) - 1] * 1000,
            "p95_method": "nearest_rank", "images_per_second": count / mean,
            "batches_per_second": 1 / mean}


def rss():
    try:
        import os
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--geometries", nargs="+", choices=("boxes", "background"), default=["boxes", "background"])
    p.add_argument("--resolutions", nargs="+", default=["160x90"])
    p.add_argument("--batches", nargs="+", type=int, default=[1])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Output must not exist")
    if not 1 <= args.warmup <= 100 or not 2 <= args.iterations <= 200 or args.seed < 0:
        raise ValueError("warmup [1,100], iterations [2,200], seed >= 0")
    from renderer_validation.public_pipeline import Pipeline, budget, provenance, verify_source, write_json, isolated
    from runtime_fingerprint import runtime_fingerprint
    import torch
    resolutions = [tuple(map(int, s.split("x"))) for s in args.resolutions]
    cells = [(g, w, h, n) for g in args.geometries for w, h in resolutions for n in args.batches]
    if len(set(cells)) != len(cells) or len(cells) > 18:
        raise ValueError("Unique cells required; at most 18")
    for _, w, h, n in cells:
        budget(w, h, n)
    source = provenance()
    if source["dirty"]:
        raise RuntimeError("Benchmark requires committed clean source; export diagnostics allow dirty with provenance")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "static_generic_pipeline_benchmark_v1", "source": source, "status": "INCOMPLETE",
              "scope": "Procedural static graphics through CPU-owned image buffers; NOT simulator step FPS",
              "integrated_simulator_cost": "INTEGRATED_RENDER_COST_UNMEASURED",
              "protocol": {"warmup": args.warmup, "iterations": args.iterations, "seed": args.seed,
                           "device": args.device, "cells": cells,
                           "steady_loop": "geometry + raycast + normal/face ID + material + shading + background + CPU buffer copy",
                           "setup": "procedural scene construction + renderer/BVH + appearance, separately timed",
                           "excluded": "compression, file IO, simulator, detector, policy",
                           "torch_threads": torch.get_num_threads()}, "rows": []}
    write_json(args.output / "request.json", record)
    def sync():
        if args.device != "cpu":
            torch.cuda.synchronize()
    try:
        for geometry, width, height, count in cells:
            gc.collect()
            if args.device != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            sync()
            start = time.perf_counter()
            pipeline = Pipeline(geometry, width, height, count, args.device, args.seed, args.seed)
            sync()
            setup = time.perf_counter() - start
            start = time.perf_counter()
            image = pipeline.frame()
            sync()
            first = time.perf_counter() - start
            del image
            for _ in range(args.warmup):
                image = pipeline.frame()
                del image
            samples = []
            for _ in range(args.iterations):
                sync()
                start = time.perf_counter()
                image = pipeline.frame()
                sync()
                samples.append(time.perf_counter() - start)
                del image
            memory = {"process_rss_bytes_after_cell": rss(),
                      "process_rss_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "rss_scope": "process lifetime high-water, not incremental/per-cell peak",
                      "torch_peak_allocated_bytes": None, "torch_peak_reserved_bytes": None,
                      "total_process_cuda_vram_bytes": None}
            if args.device != "cpu":
                memory.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            record["rows"].append({"geometry": geometry, "width": width, "height": height, "scenes": count,
                                   "setup_seconds": setup, "first_frame_seconds": first,
                                   "timing": stats(samples, count), "memory": memory})
            record["runtime"] = runtime_fingerprint(include_device=args.device != "cpu")
            record["runtime"]["warp"] = pipeline.renderer.wp.__version__
            print("completed %s %dx%d scenes=%d" % (geometry, width, height, count), flush=True)
            del pipeline
        isolated()
        verify_source(source)
        record["status"] = "GENERIC_PIPELINE_BENCHMARK_COMPLETE"
        write_json(args.output / "receipt.json", record)
        lines = ["# Static generic renderer benchmark", "", record["scope"], "",
                 "Setup/first-frame are separate. No statistical or simulator-integration verdict.", "",
                 "| Scene | Resolution | Batch | Mean ms | Median ms | P95 ms | Images/s |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in record["rows"]:
            t = row["timing"]
            lines.append("| %s | %dx%d | %d | %.3f | %.3f | %.3f | %.2f |" % (
                row["geometry"], row["width"], row["height"], row["scenes"], t["mean_ms"],
                t["median_ms"], t["p95_ms"], t["images_per_second"]))
        with (args.output / "README.md").open("x") as stream:
            stream.write("\n".join(lines) + "\n")
    except BaseException as exc:
        record.update(status="FAILED_INCOMPLETE", error_type=type(exc).__name__)
        write_json(args.output / "failure.json", record)
        raise


if __name__ == "__main__":
    main()
