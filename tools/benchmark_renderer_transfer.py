#!/usr/bin/env python3
"""RC-R5b independent transfer study. No historical renderer/benchmark source edits."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renderer_validation.characterization_pipeline import CharacterizationCell, GEOMETRY_ARRAYS
from renderer_validation.followup_records import ROOT, dump, finish, manifest, verify
from renderer_validation.scene import Camera
from renderer_validation.transfer_strategies import ARMS, TransferPlan, full_host, hashes, rss_mib
from renderer_validation.view_grid import ViewGrid
from runtime_fingerprint import runtime_fingerprint

FIXTURES = ((320, 240, 8), (640, 480, 32))
WARMUP, SAMPLES, REPEATS = 20, 100, 3


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Invalid timing samples")
    return {"n": len(data), "mean_ms": float(data.mean()), "median_ms": float(np.median(data)),
            "p95_ms": float(np.percentile(data, 95))}


def memory():
    import warp
    result = {"rss_mib": rss_mib(), "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
              "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
              "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "warp_only_mib": None, "warp_note": "BVH/native allocations are not all visible to Torch; no per-library counter",
              "nvml_device_used_mib": None, "nvml_source": "nvidia-smi (NVML), device-wide, includes desktop"}
    try:
        result["nvml_device_used_mib"] = float(subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        result["nvml_error"] = str(error)
    device = warp.get_device("cuda:0")
    total, free = getattr(device, "total_memory", None), getattr(device, "free_memory", None)
    result["warp_device_used_mib"] = (total-free)/2**20 if total is not None and free is not None else None
    return result


def resource_check():
    rss = rss_mib()
    if torch.cuda.memory_reserved() > 5 * 2**30 or (rss is not None and rss > 8192):
        raise MemoryError("Preregistered Torch/RSS resource stop")


def run_cell(fixture, arm, repeat, output):
    before = manifest()
    width, height, count = fixture
    if width * height * count * 41 > 1.5 * 2**30:
        raise MemoryError("Projected output >1.5 GiB")
    torch.manual_seed(20260914)
    grid = ViewGrid([(i * 360.0/count, 10.0, 2.0) for i in range(count)])
    start = time.perf_counter()
    cell = CharacterizationCell("quadrotor_mesh", Camera(width, height), grid, "cuda:0")
    appearance = cell.lit_appearance()

    def compute():
        cell._gbuffer = None
        gbuffer = cell.gbuffer()
        rgb = cell.shade(appearance)
        return {"rgb": rgb, **{key: getattr(gbuffer, key) for key in GEOMETRY_ARRAYS}}

    buffers = compute()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter()-start)*1000
    expected = hashes(full_host(buffers))
    allocated = time.perf_counter()
    try:
        plan = TransferPlan(arm, buffers)
    except RuntimeError as error:
        if arm != "A3":
            raise
        dump(output, {"arm": arm, "repeat": repeat, "fixture": list(fixture), "status": "UNSUPPORTED",
                      "reason": str(error), "source_manifest": before, "runtime": runtime_fingerprint()})
        return
    allocation_ms = (time.perf_counter()-allocated)*1000
    first = plan.transfer(buffers)
    first_hashes = hashes(full_host(buffers) if arm == "A1" else first)
    del buffers, first
    for _ in range(WARMUP):
        values = compute()
        plan.transfer(values)
        torch.cuda.synchronize()
        resource_check()
    del values
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    samples = {key: [] for key in ("geometry", "shading", "host_strategy", "staged_total", "headline_total")}
    for index in range(SAMPLES):
        start = time.perf_counter()
        cell._gbuffer = None
        g = cell.gbuffer()
        torch.cuda.synchronize()
        middle = time.perf_counter()
        rgb = cell.shade(appearance)
        torch.cuda.synchronize()
        shaded = time.perf_counter()
        values = {"rgb": rgb, **{key: getattr(g, key) for key in GEOMETRY_ARRAYS}}
        host = plan.transfer(values)
        torch.cuda.synchronize()
        end = time.perf_counter()
        for name, duration in (("geometry", middle-start), ("shading", shaded-middle),
                               ("host_strategy", end-shaded), ("staged_total", end-start)):
            samples[name].append(duration*1000)
        del values, host, rgb, g
        resource_check()
    for index in range(SAMPLES):
        start = time.perf_counter()
        values = compute()
        host = plan.transfer(values)
        torch.cuda.synchronize()
        samples["headline_total"].append((time.perf_counter()-start)*1000)
        del values, host
        resource_check()
    values = compute()
    started = time.perf_counter()
    host = full_host(values) if arm == "A1" else plan.transfer(values)
    torch.cuda.synchronize()
    export_validation_ms = (time.perf_counter()-started)*1000
    actual = hashes(host)
    if before != manifest():
        raise RuntimeError("Source changed during cell")
    invariant = expected == first_hashes == actual
    payload = {"status": "MEASURED" if invariant else "OUTPUT_MISMATCH", "arm": arm,
               "fixture": list(fixture), "repeat": repeat, "setup_ms": setup_ms,
               "transfer_allocation_ms": allocation_ms, "terminal_full_export_ms": export_validation_ms,
               "reference_hashes": expected, "first_hashes": first_hashes, "output_hashes": actual,
               "output_invariance": invariant, "output_bytes": sum(v.nbytes for v in host.values()),
               "memory": memory(), "source_manifest": before, "runtime": runtime_fingerprint(),
               "throughput_scope": "device-resident/scalar; not full export" if arm == "A1" else "full host buffers",
               "warmup": WARMUP, "samples_per_pass": SAMPLES,
               "raw_ms": samples if invariant else None,
               "timing": {k: distribution(v) for k, v in samples.items()} if invariant else None,
               "causality_vs_d8b": "NOT_TESTED"}
    if invariant:
        payload["scenes_per_second"] = count * 1000 / payload["timing"]["headline_total"]["mean_ms"]
    dump(output, payload)
    print("[R5b]", fixture, arm, repeat, payload["status"], flush=True)


def campaign(output):
    before = manifest()
    output.mkdir(parents=True, exist_ok=False)
    cells = output / "cells"
    cells.mkdir()
    rows = []
    for fixture in FIXTURES:
        for repeat in range(REPEATS):
            order = ARMS[repeat:] + ARMS[:repeat]
            for arm in order:
                path = cells / ("%dx%d_n%d_%s_r%d.json" % (*fixture, arm, repeat))
                print("[R5b] start", path.name, flush=True)
                result = subprocess.run([sys.executable, "-B", __file__, "cell", "--output", str(path),
                    "--fixture", *map(str, fixture), "--arm", arm, "--repeat", str(repeat)])
                if result.returncode:
                    raise RuntimeError("Cell failed; partial directory preserved: " + path.name)
                rows.append(json.loads(path.read_text()))
    matching = True
    for fixture in FIXTURES:
        group = [row for row in rows if row["fixture"] == list(fixture) and row["status"] != "UNSUPPORTED"]
        bases = [row for row in group if row["arm"] == "A0"]
        matching &= len(bases) == REPEATS and all(row["status"] == "MEASURED" and
                    row["output_hashes"] == bases[0]["output_hashes"] for row in group)
    large = [r for r in rows if r["fixture"] == list(FIXTURES[-1]) and r["arm"] == "A0"]
    fractions = [r["timing"]["host_strategy"]["mean_ms"] / r["timing"]["staged_total"]["mean_ms"]
                 for r in large] if matching else []
    stopped = any(row["status"] == "SKIPPED_RESOURCE_STOP" for row in rows)
    summary = {"verdict": "INCOMPLETE_RESOURCE_STOP" if stopped else
               "TRANSFER_CHARACTERIZED" if matching else "OUTPUT_MISMATCH",
               "all_output_hashes_match": bool(matching), "cells": rows,
               "a0_large_host_fractions": fractions,
               "host_strategy_dominant": len(fractions) == 3 and all(f > .5 for f in fractions)}
    prose = "# RC-R5b — transfer characterization\n\nVerdict: `%s`.\n\n" % summary["verdict"]
    prose += "R5 historical run contained known ResourceWarning. R5b fixes the harness/resource handling "
    prose += "and is a new measurement lineage. Historical source and receipts are unchanged.\n\n"
    prose += "Historical R5 host_copy included a shading call and an owned NumPy copy. It was not "
    prose += "pure PCIe transfer time; the new stage boundaries must not be treated as identical.\n\n"
    prose += "A1 throughput is resident/scalar throughput, not full image export. A2 includes packing; "
    prose += "A3 includes DMA completion, without compute/copy overlap. Full raw stage/headline timings "
    prose += "and separate setup/validation costs are in summary.json and cells/. Device memory includes "
    prose += "non-Torch allocations and other desktop processes. No per-library Warp zero is invented.\n"
    finish(output, before, {"fixtures": FIXTURES, "arms": ARMS, "repeats": REPEATS,
           "warmup": WARMUP, "samples": SAMPLES}, summary, prose)
    print("[R5b]", verify(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "cell", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=int, nargs=3)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.command == "verify":
        print(verify(args.output))
    elif args.command == "run":
        campaign(args.output)
    else:
        try:
            run_cell(tuple(args.fixture), args.arm, args.repeat, args.output)
        except MemoryError as error:
            dump(args.output, {"status": "SKIPPED_RESOURCE_STOP", "reason": str(error),
                 "fixture": args.fixture, "arm": args.arm, "repeat": args.repeat})
