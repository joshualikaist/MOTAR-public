#!/usr/bin/env python3
"""RC-R5: descriptive performance of the characterization render path.

Descriptive means descriptive. This experiment has no pass/fail verdict, and its numbers are not
comparable with the earlier D6 ray-transform ratio or the D7 integrated-path cost: different
denominators, different code paths, different questions. Anyone who puts them in one table is
comparing three unlike things.

Two totals are reported for every cell. The headline total times the production render call, which
synchronizes once at the end. The staged total inserts a synchronization between pipeline stages so
each stage can be attributed, and the difference between the two is reported as instrumentation
overhead rather than hidden inside the stage numbers. Before any timing is kept, the staged path is
checked against the production path for byte-identical output; a cell whose staged path disagrees
reports no stage timings at all.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

WARMUP_RENDERS = 20
MEASURED_RENDERS = 100
RESOLUTIONS = ((160, 90), (320, 240), (640, 480))
SCENE_COUNTS = (1, 8, 32)
FAR_RANGE_M = 20.0
BENCHMARK_ELEVATION_DEG = 10.0
BENCHMARK_DISTANCE_M = 2.0
OUTPUT_BYTES_MAX = int(1.5 * 1024 ** 3)
WARMUP_SECONDS_MAX = 2.0
TORCH_RESERVED_MAX_BYTES = int(5.0 * 1024 ** 3)
MIB = 1024 ** 2
# (name, arm, shading) - shading None means the G-buffer pass alone, with no image produced.
CONFIGURATIONS = (
    ("analytic_sphere_geometry", "analytic_sphere", None),
    ("box_geometry", "box_proxy", None),
    ("box_flat", "box_proxy", "flat"),
    ("area_matched_box_lambertian", "area_matched_box", "lambertian"),
    ("quadrotor_geometry", "quadrotor_mesh", None),
    ("quadrotor_lambertian", "quadrotor_mesh", "lambertian"),
)


def benchmark_grid(count):
    """`count` distinct views spread evenly in azimuth at one elevation and distance."""
    from renderer_validation.view_grid import ViewGrid
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("Scene count must be an integer in [1,128]")
    azimuths = np.linspace(0.0, 360.0, count, endpoint=False)
    return ViewGrid([(float(a), BENCHMARK_ELEVATION_DEG, BENCHMARK_DISTANCE_M) for a in azimuths])


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return "UNAVAILABLE"


def memory_record(device, warp_module):
    """Five independent fields. An unavailable field says so; it is never reported as zero."""
    record = {"rss_mib": rss_mib(),
              "nvml_used_mib": "UNAVAILABLE",
              "nvml_reason": "pynvml is not installed in this environment; no substitute was "
                             "silently used in its place",
              "torch_allocated_mib": "UNAVAILABLE", "torch_reserved_mib": "UNAVAILABLE",
              "warp_device_used_mib": "UNAVAILABLE",
              "warp_field_note": "Warp exposes whole-device total and free memory, not a per-library "
                                 "figure; this field is device-wide used memory read through Warp"}
    try:
        import pynvml
    except ImportError:
        pass
    else:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            record["nvml_used_mib"] = pynvml.nvmlDeviceGetMemoryInfo(handle).used / MIB
            record.pop("nvml_reason")
        except Exception as error:                       # pragma: no cover - driver dependent
            record["nvml_reason"] = "%s: %s" % (type(error).__name__, error)
    if device != "cpu":
        import torch
        record["torch_allocated_mib"] = torch.cuda.max_memory_allocated(device) / MIB
        record["torch_reserved_mib"] = torch.cuda.max_memory_reserved(device) / MIB
        if warp_module is not None:
            handle = warp_module.get_device(device)
            total = getattr(handle, "total_memory", None)
            free = getattr(handle, "free_memory", None)
            if total is not None and free is not None:
                record["warp_device_used_mib"] = (total - free) / MIB
    return record


def synchronize(device, warp_module):
    if device == "cpu":
        return
    import torch
    torch.cuda.synchronize(device)
    if warp_module is not None:
        warp_module.synchronize_device(device)


def staged_mesh_render(cell):
    """The production kernel launches with a synchronization between stages, for attribution.

    The two launches, their inputs and their order are the ones in
    renderer_validation.gbuffer.WarpGBufferRenderer.render. Equality with that method's output is
    checked before any timing from this function is kept.
    """
    renderer = cell.renderer
    wp, c = renderer.wp, renderer.camera
    stages = {}
    started = time.perf_counter()
    renderer.raw_range.fill_(1000.0)
    renderer.raw_normal.zero_()
    renderer.raw_face.fill_(-1)
    synchronize(renderer.warp_device, wp)
    stages["clear_ms"] = (time.perf_counter() - started) * 1000.0
    common = [renderer.mesh_ids, renderer.wp_positions, renderer.wp_orientations, renderer.k_inv,
              c.far_range_m]
    dim = (renderer.num_scenes, 1, c.width, c.height)
    started = time.perf_counter()
    wp.launch(renderer.kernels.draw_optimized_kernel_normal_faceID, dim=dim,
              inputs=common + [renderer.wp_normal, renderer.wp_face, c.width // 2, c.height // 2, True],
              device=renderer.warp_device)
    wp.synchronize_device(renderer.warp_device)
    stages["normal_face_pass_ms"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    wp.launch(renderer.kernels.draw_optimized_kernel_depth_range, dim=dim,
              inputs=common + [renderer.wp_range, c.width // 2, c.height // 2, False],
              device=renderer.warp_device)
    wp.synchronize_device(renderer.warp_device)
    stages["range_pass_ms"] = (time.perf_counter() - started) * 1000.0
    from renderer_validation.gbuffer import finalize_gbuffer
    started = time.perf_counter()
    gbuffer = finalize_gbuffer(renderer.raw_range[:, 0], renderer.raw_normal[:, 0],
                               renderer.raw_face[:, 0], renderer.scene, c)
    synchronize(renderer.warp_device, wp)
    stages["finalize_ms"] = (time.perf_counter() - started) * 1000.0
    return gbuffer, stages


def staged_analytic_render(cell):
    from renderer_validation.analytic_primitives import analytic_sphere_gbuffer
    started = time.perf_counter()
    gbuffer = analytic_sphere_gbuffer(cell.camera, cell.grid, cell.arm.radius_m, device=cell.device)
    return gbuffer, {"analytic_intersection_ms": (time.perf_counter() - started) * 1000.0}


def buffers_equal(first, second):
    import torch
    names = ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")
    return all(torch.equal(getattr(first, name), getattr(second, name)) for name in names)


def timed(function, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def distribution(samples):
    if not samples or not all(np.isfinite(samples)):
        raise ValueError("Timing samples must be finite and nonempty")
    ordered = sorted(samples)
    return {"count": len(samples), "median_ms": float(statistics.median(ordered)),
            "mean_ms": float(statistics.fmean(ordered)), "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "p95_ms": float(np.quantile(ordered, 0.95))}


def projected_bytes(width, height, count):
    # rgb 12 + normal 12 + range 4 + depth 4 + face 4 + instance 4 + valid 1 bytes per pixel.
    return width * height * count * 41


def run_cell(name, arm, shading, resolution, count, device, scale):
    from renderer_validation.characterization_pipeline import CharacterizationCell, camera_relative_light
    from renderer_validation.scene import Camera
    width, height = resolution
    record = {"configuration": name, "arm": arm, "shading": shading,
              "resolution": [width, height], "scenes": count,
              "projected_output_bytes": projected_bytes(width, height, count)}
    if record["projected_output_bytes"] > OUTPUT_BYTES_MAX:
        record.update(status="SKIPPED_RESOURCE_STOP",
                      reason="projected output %d bytes exceeds the %d byte limit"
                             % (record["projected_output_bytes"], OUTPUT_BYTES_MAX))
        return record
    grid = benchmark_grid(count)
    camera = Camera(width=width, height=height, far_range_m=FAR_RANGE_M)
    cell = CharacterizationCell(arm, camera, grid, device=device,
                               scale=scale if arm == "area_matched_box" else None)
    warp_module = cell.renderer.wp if cell.renderer is not None else None
    if device != "cpu":
        import torch
        torch.cuda.reset_peak_memory_stats(device)

    def production():
        cell._gbuffer = None
        return cell.gbuffer()

    started = time.perf_counter()
    production()
    first = (time.perf_counter() - started)
    if first > WARMUP_SECONDS_MAX:
        record.update(status="SKIPPED_RESOURCE_STOP",
                      reason="first render took %.3f s, above the %.1f s limit" % (first, WARMUP_SECONDS_MAX))
        return record
    reference = cell.gbuffer()
    staged_function = staged_analytic_render if cell.renderer is None else staged_mesh_render
    staged_buffer, _ = staged_function(cell)
    if not buffers_equal(reference, staged_buffer):
        record.update(status="STAGED_PATH_MISMATCH",
                      reason="the staged path did not reproduce the production buffers; no stage "
                             "timings are reported for this cell")
        return record
    appearance = (None if shading is None else
                  cell.appearance(colour=0.55, light_direction=camera_relative_light(grid)))

    def shade_once():
        cell.shade(appearance, shading)

    def copy_once():
        cell.arrays(appearance, shading if shading else "flat")

    for _ in range(WARMUP_RENDERS):
        production()
        if shading is not None:
            shade_once()
    if device != "cpu":
        import torch
        if torch.cuda.max_memory_reserved(device) > TORCH_RESERVED_MAX_BYTES:
            record.update(status="SKIPPED_RESOURCE_STOP",
                          reason="torch reserved %.0f MiB, above the %.0f MiB limit"
                                 % (torch.cuda.max_memory_reserved(device) / MIB,
                                    TORCH_RESERVED_MAX_BYTES / MIB))
            return record
    total = timed(production, MEASURED_RENDERS)
    staged_totals, stage_samples = [], {}
    for _ in range(MEASURED_RENDERS):
        cell._gbuffer = None
        started = time.perf_counter()
        _, stages = staged_function(cell)
        staged_totals.append((time.perf_counter() - started) * 1000.0)
        for stage, value in stages.items():
            stage_samples.setdefault(stage, []).append(value)
        cell._gbuffer = None
    production()
    shading_samples = timed(shade_once, MEASURED_RENDERS) if shading is not None else None
    copy_samples = timed(copy_once, MEASURED_RENDERS)
    record.update(
        status="MEASURED",
        headline_total=distribution(total),
        staged_total=distribution(staged_totals),
        instrumentation_overhead_ms=float(statistics.median(staged_totals) - statistics.median(total)),
        stages={stage: distribution(values) for stage, values in sorted(stage_samples.items())},
        shading=distribution(shading_samples) if shading_samples else None,
        host_copy=distribution(copy_samples),
        images_per_second=float(count * 1000.0 / statistics.median(total)),
        memory=memory_record(device, warp_module),
        warmup_renders=WARMUP_RENDERS, measured_renders=MEASURED_RENDERS)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--scene-counts", type=int, nargs="*", default=list(SCENE_COUNTS))
    args = parser.parse_args(argv)
    from renderer_validation.rc_results import source_manifest, write_experiment
    from run_renderer_characterization import fitted_scale
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    scale, fit_record = fitted_scale(args.device)
    cells = []
    for name, arm, shading in CONFIGURATIONS:
        for resolution in RESOLUTIONS:
            for count in args.scene_counts:
                record = run_cell(name, arm, shading, resolution, int(count), args.device, scale)
                cells.append(record)
                print("%-28s %4dx%-4d n=%-3d %s" % (name, resolution[0], resolution[1], count,
                                                    record["status"]), flush=True)
    measured = [cell for cell in cells if cell["status"] == "MEASURED"]
    summary = {"experiment": "RC-R5", "verdict": "DESCRIPTIVE_NO_VERDICT",
               "configurations": [name for name, _, _ in CONFIGURATIONS],
               "resolutions": [list(r) for r in RESOLUTIONS], "scene_counts": list(args.scene_counts),
               "cells": cells, "measured_cells": len(measured), "total_cells": len(cells),
               "skipped_cells": [cell for cell in cells if cell["status"] != "MEASURED"],
               "area_match_fit": fit_record,
               "comparability": "Not comparable with D6 (1.5338x ray-transform ratio, INCONCLUSIVE) "
                                "or D7 (+0.411 ms integrated shadow cost, GO): different "
                                "denominators and different code paths.",
               "scope": "Render-path cost of this track's own pipeline. No policy, detector or "
                        "training cost is measured.",
               "causality_vs_d8b": "NOT_TESTED"}
    thresholds = {"warmup_renders": WARMUP_RENDERS, "measured_renders": MEASURED_RENDERS,
                  "output_bytes_max": OUTPUT_BYTES_MAX, "warmup_seconds_max": WARMUP_SECONDS_MAX,
                  "torch_reserved_max_bytes": TORCH_RESERVED_MAX_BYTES,
                  "verdict_rule": "descriptive; no pass/fail threshold exists for RC-R5"}
    config = {"command": "r5", "device": args.device, "source_manifest": before,
              "configurations": [{"name": name, "arm": arm, "shading": shading}
                                 for name, arm, shading in CONFIGURATIONS],
              "resolutions": [list(r) for r in RESOLUTIONS],
              "scene_counts": list(args.scene_counts),
              "benchmark_views": {"elevation_deg": BENCHMARK_ELEVATION_DEG,
                                  "distance_m": BENCHMARK_DISTANCE_M,
                                  "azimuths": "evenly spread over 360 degrees per scene count"}}
    readme = r5_readme(summary)
    receipt = write_experiment(args.output, "RC-R5", config, summary, readme, thresholds,
                              runtime_fingerprint(include_device=args.device != "cpu"))
    print("RC-R5 %s -> %s" % (receipt["verdict"], args.output))
    return 0


def r5_readme(summary):
    lines = ["# RC-R5 — render-path cost (descriptive)", "",
             "Verdict: **%s**. RC-R5 has no pass/fail threshold." % summary["verdict"], "",
             "%d of %d cells were measured; the rest hit the preregistered resource stop rule."
             % (summary["measured_cells"], summary["total_cells"]), "",
             "| configuration | resolution | scenes | headline median ms | staged median ms | instrumentation ms |",
             "|---|---|---|---|---|---|"]
    for cell in summary["cells"]:
        if cell["status"] != "MEASURED":
            continue
        lines.append("| `%s` | %dx%d | %d | %.3f | %.3f | %.3f |" % (
            cell["configuration"], cell["resolution"][0], cell["resolution"][1], cell["scenes"],
            cell["headline_total"]["median_ms"], cell["staged_total"]["median_ms"],
            cell["instrumentation_overhead_ms"]))
    lines += ["", summary["comparability"], "",
              "Memory is reported as five independent fields per cell. A field that could not be",
              "read says `UNAVAILABLE` and carries its reason; none is reported as zero."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
