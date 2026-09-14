#!/usr/bin/env python3
"""Run one renderer characterization (RC) experiment and write its result directory.

Each subcommand is one preregistered experiment from
results/renderer_characterization_2026-09-13/PREREGISTRATION.md. The thresholds are read out of the
measurement modules, never restated here, so a receipt records what the code actually applied.

Nothing in this tool imports the simulator package, a task, a controller, a detector or a policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

PRIMARY_RESOLUTION = (320, 240)
HIGH_RESOLUTION = (640, 480)
FAR_RANGE_M = 20.0
# RC-R4 keeps its lights fixed in the world, unlike RC-R3: the point of R4 is that the same geometry
# under a different world light is a different image, including the case where a face goes unlit.
R4_LIGHT_DIRECTIONS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                       (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
R4_AMBIENT = (0.05, 0.2, 0.35)
R4_DIRECTIONAL = (0.4, 0.8)
R4M_MATERIALS = (("grey_0p20", 0.20), ("grey_0p40", 0.40), ("grey_0p60", 0.60),
                 ("grey_0p80", 0.80), ("per_material_rgb", None))
R4M_PER_MATERIAL_RGB = ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.6, 0.6, 0.6))
LUMINANCE_SPREAD_MIN = 0.02        # RC-R4 L2 and RC-R4M M2
FLAT_VARIANCE_MAX = 1e-12          # RC-R3 S2
SHADED_VARIANCE_MIN = 1e-4         # RC-R3 S3
ARMS = ("analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh")


def camera(resolution):
    from renderer_validation.scene import Camera
    return Camera(width=resolution[0], height=resolution[1], far_range_m=FAR_RANGE_M)


def cell_for(arm, resolution, grid, device, scale=None, instance_offset=0):
    from renderer_validation.characterization_pipeline import CharacterizationCell
    return CharacterizationCell(arm, camera(resolution), grid, device=device, scale=scale,
                                instance_offset=instance_offset)


def fitted_scale(device):
    """The RC-R2 isotropic box scale, fitted on the four fit views and on nothing else."""
    from renderer_validation.area_match import fit_isotropic_scale
    from renderer_validation.view_grid import fit_grid
    grid = fit_grid()
    areas = {}
    for arm in ("box_proxy", "quadrotor_mesh"):
        cell = cell_for(arm, PRIMARY_RESOLUTION, grid, device)
        measured, refused = cell.metrics_record()
        if refused:
            # The fit views are preregistered. Fitting on a silently smaller set would make the
            # scale depend on which views happened to survive, so this refuses instead.
            raise RuntimeError("Fit view refused for %s: %s" % (arm, refused))
        areas[arm] = [row["silhouette_pixels"] for row in measured]
        del cell
    scale = fit_isotropic_scale(areas["quadrotor_mesh"], areas["box_proxy"])
    return scale, {"fit_views": grid.as_dict()["views"], "fit_areas_px": areas,
                   "scale": scale, "fit_views_refused": 0,
                   "model": "silhouette area scales as the square of an isotropic factor"}


def variance(statistics):
    """Luminance variance in float64. A float32 reduction of identical values is not exactly zero."""
    return float(statistics["luminance_std"]) ** 2


def failed(gates):
    return sorted(name for name, record in gates.items() if not record["passed"])


def run_r1(args):
    from renderer_validation import geometry_metrics as gm
    from renderer_validation.view_grid import primary_grid, distance_sweep_grid, ViewGrid
    from runtime_fingerprint import runtime_fingerprint
    from renderer_validation.rc_results import source_manifest, write_experiment
    before = source_manifest()
    scale, fit_record = fitted_scale(args.device)
    primary, sweep = primary_grid(), distance_sweep_grid()
    arms, gates, timings = {}, {}, {}
    for name in ARMS:
        started = time.perf_counter()
        use_scale = scale if name == "area_matched_box" else None
        primary_cell = cell_for(name, PRIMARY_RESOLUTION, primary, args.device, use_scale)
        primary_metrics, primary_refused = primary_cell.metrics_record()
        gates["G5_depth_bracket_primary_%s" % name] = gm.gate_depth_bracket(
            primary_cell.arm, primary_metrics, primary)
        if primary_cell.arm.kind == "analytic_sphere":
            gates["G2_sphere_closed_form"] = gm.gate_sphere_closed_form(
                primary_cell.arm, primary_metrics, primary_cell.camera, primary)
        if primary_cell.arm.description.get("kind") == "box":
            gates["G3_box_closed_form_%s" % name] = gm.gate_box_closed_form(
                primary_cell.arm, primary_metrics, primary_cell.camera, primary)
        if name in ("box_proxy", "quadrotor_mesh"):
            gates["G1_cross_implementation_%s" % name] = gm.gate_cross_implementation(
                primary_cell.arm, primary_cell.gbuffer(), primary_cell.camera, primary)
        arm_description = primary_cell.description()
        low_camera = primary_cell.camera
        del primary_cell

        sweep_cell = cell_for(name, PRIMARY_RESOLUTION, sweep, args.device, use_scale)
        sweep_metrics, sweep_refused = sweep_cell.metrics_record()
        gates["G4_inverse_distance_%s" % name] = gm.gate_inverse_distance(
            sweep_cell.arm, sweep_metrics, sweep)
        gates["G5_depth_bracket_sweep_%s" % name] = gm.gate_depth_bracket(
            sweep_cell.arm, sweep_metrics, sweep)
        del sweep_cell

        high_cell = cell_for(name, HIGH_RESOLUTION, primary, args.device, use_scale)
        high_metrics, high_refused = high_cell.metrics_record()
        gates["G7_resolution_consistency_%s" % name] = gm.gate_resolution_consistency(
            high_cell.arm, primary_metrics, high_metrics, low_camera, high_cell.camera)
        del high_cell

        arms[name] = {"description": arm_description, "primary_320x240": primary_metrics,
                      "distance_sweep_320x240": sweep_metrics, "primary_640x480": high_metrics,
                      "refused_views": {"primary_320x240": primary_refused,
                                        "distance_sweep_320x240": sweep_refused,
                                        "primary_640x480": high_refused}}
        timings[name] = time.perf_counter() - started

    # G6 is demonstrated, not asserted: a starved view must raise where a measurement would be wrong.
    starved = ViewGrid([(0.0, 0.0, 14.0)])
    refusal = None
    try:
        cell_for("quadrotor_mesh", PRIMARY_RESOLUTION, starved, args.device).metrics()
    except RuntimeError as error:
        refusal = str(error)
    gates["G6_degenerate_view_refusal"] = {
        "gate": "G6_degenerate_view_refusal", "view": starved.as_dict()["views"][0],
        "refusal_message": refusal, "passed": refusal is not None,
        "thresholds": {"min_silhouette_pixels": gm.MIN_SILHOUETTE_PIXELS,
                       "min_visible_triangles": gm.MIN_VISIBLE_TRIANGLES}}

    broken = failed(gates)
    refusals = {name: record["refused_views"] for name, record in arms.items()}
    summary = {"experiment": "RC-R1", "verdict": "GEOMETRY_VERIFIED" if not broken else "GEOMETRY_DEFECT",
               "failed_gates": broken, "area_match_fit": fit_record,
               "refused_views": refusals,
               "refused_view_count": {name: sum(len(views) for views in record.values())
                                      for name, record in refusals.items()},
               "refusal_note": "G6 refuses a view below the preregistered 300-pixel minimum. Those "
                               "views are listed here and excluded from every measurement; the "
                               "threshold was not changed to admit them.",
               "resolutions": {"primary": list(PRIMARY_RESOLUTION), "high": list(HIGH_RESOLUTION)},
               "arms": arms, "gates": gates,
               "seconds_per_arm": timings,
               "scope": "Renderer geometry representation only. No detector, policy or D8b claim.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "r1", "device": args.device, "source_manifest": before,
              "primary_views": primary.as_dict(), "sweep_views": sweep.as_dict(),
              "resolutions": {"primary": list(PRIMARY_RESOLUTION), "high": list(HIGH_RESOLUTION)},
              "far_range_m": FAR_RANGE_M, "arms": list(ARMS)}
    readme = r1_readme(summary)
    return write_experiment(args.output, "RC-R1", config, summary, readme, gm.thresholds(),
                            runtime_fingerprint(include_device=args.device != "cpu"))


def r1_readme(summary):
    lines = ["# RC-R1 — geometry representation across four specimens", "",
             "Verdict: **%s**" % summary["verdict"], ""]
    if summary["failed_gates"]:
        lines += ["Failed gates: " + ", ".join("`%s`" % name for name in summary["failed_gates"]), ""]
    else:
        lines += ["Every preregistered gate G1–G7 passed.", ""]
    lines += ["| arm | views | silhouette px (min / median / max) | equivalent diameter px (median) |",
              "|---|---|---|---|"]
    for name, record in summary["arms"].items():
        pixels = [row["silhouette_pixels"] for row in record["primary_320x240"]]
        diameters = [row["equivalent_diameter_px"] for row in record["primary_320x240"]]
        lines.append("| `%s` | %d | %d / %.0f / %d | %.2f |" % (
            name, len(pixels), min(pixels), float(np.median(pixels)), max(pixels),
            float(np.median(diameters))))
    lines += ["", "The area-matched box scale was fitted on the four fit views only: "
              "`%.6f`." % summary["area_match_fit"]["scale"], "",
              "Scope: this is a measurement of how the renderer represents these four shapes. It is",
              "not a detector, tracking or policy result, and it makes no causal claim about the",
              "closed D8b frozen-policy result (`causality_vs_d8b: NOT_TESTED`)."]
    return "\n".join(lines) + "\n"


def run_r2(args):
    from renderer_validation.area_match import (AREA_MATCH_TOLERANCE, area_ratios,
                                                evaluate_area_match)
    from renderer_validation import geometry_metrics as gm
    from renderer_validation.view_grid import fit_validation_split
    from renderer_validation.rc_results import source_manifest, write_experiment
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    scale, fit_record = fitted_scale(args.device)
    primary, fit_indices, validation_indices = fit_validation_split()
    validation = primary.subset(validation_indices)
    by_view, per_arm, refusals = {}, {}, {}
    for name in ARMS:
        cell = cell_for(name, PRIMARY_RESOLUTION, validation, args.device,
                        scale if name == "area_matched_box" else None)
        metrics, refused = cell.metrics_record()
        by_view[name] = {row["view"]: row["silhouette_pixels"] for row in metrics}
        refusals[name] = refused
        per_arm[name] = {"description": cell.description(), "validation_metrics": metrics,
                         "refused_views": refused}
        del cell
    # Every arm must be compared on the same views, or a mean area would mix view sets.
    shared = sorted(set.intersection(*(set(values) for values in by_view.values())))
    if not shared:
        raise RuntimeError("No validation view was measured for every arm")
    areas = {name: [values[view] for view in shared] for name, values in by_view.items()}
    match = evaluate_area_match(areas["area_matched_box"], areas["quadrotor_mesh"], scale)
    summary = {"experiment": "RC-R2", "verdict": match["verdict"],
               "primary_estimand": "mean silhouette area in pixels over the 20 validation views",
               "fit": fit_record, "fit_view_indices": fit_indices,
               "validation_view_indices": validation_indices,
               "validation_views": validation.as_dict()["views"],
               "views_used_for_every_arm": shared,
               "views_refused_per_arm": refusals,
               "mean_areas": area_ratios(areas), "per_view_areas_px": areas,
               "area_match": match, "arms": per_arm,
               "scope": "Apparent-area comparison between four specimens. No detector or policy "
                        "quantity is computed, and no cause of the D8b result is proposed.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "r2", "device": args.device, "source_manifest": before,
              "resolution": list(PRIMARY_RESOLUTION), "far_range_m": FAR_RANGE_M,
              "fit_views": fit_record["fit_views"], "validation_views": validation.as_dict()["views"],
              "arms": list(ARMS)}
    thresholds = {"area_match_tolerance": AREA_MATCH_TOLERANCE,
                  "min_silhouette_pixels": gm.MIN_SILHOUETTE_PIXELS,
                  "min_visible_triangles": gm.MIN_VISIBLE_TRIANGLES}
    return write_experiment(args.output, "RC-R2", config, summary, r2_readme(summary), thresholds,
                            runtime_fingerprint(include_device=args.device != "cpu"))


def r2_readme(summary):
    lines = ["# RC-R2 — apparent area across four specimens", "",
             "Verdict: **%s**" % summary["verdict"], "",
             "Primary estimand: %s." % summary["primary_estimand"], "",
             "| arm | mean area px | ratio to quadrotor mesh |", "|---|---|---|"]
    for name, record in summary["mean_areas"].items():
        lines.append("| `%s` | %.1f | %.4f |" % (name, record["mean_area_px"],
                                                 record["ratio_to_reference"]))
    match = summary["area_match"]
    lines += ["", "The isotropic scale `%.6f` was fitted on the four fit views and applied unchanged"
              % match["scale"],
              "to the twenty validation views. On those views the matched box covers %.4f of the"
              % match["area_ratio"],
              "mesh's mean area, a relative error of %.4f against a tolerance of %.2f."
              % (match["relative_error"], match["tolerance"]), "",
              "This says how large each specimen looks to this renderer. It says nothing about what a",
              "detector or a policy does with that, and it proposes no cause for the closed D8b",
              "result (`causality_vs_d8b: NOT_TESTED`)."]
    return "\n".join(lines) + "\n"


def run_r3(args):
    from renderer_validation.characterization_pipeline import geometry_hashes
    from renderer_validation import geometry_metrics as gm
    from renderer_validation.view_grid import primary_grid
    from renderer_validation.rc_results import source_manifest, write_experiment
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    scale, fit_record = fitted_scale(args.device)
    grid = primary_grid()
    cells, gates = {}, {}
    for name in ARMS:
        cell = cell_for(name, PRIMARY_RESOLUTION, grid, args.device,
                        scale if name == "area_matched_box" else None)
        hashes_before = geometry_hashes(cell.gbuffer())
        flat = cell.image_statistics(cell.appearance(colour=0.55), "flat")
        shaded = cell.image_statistics(cell.lit_appearance(colour=0.55), "lambertian")
        hashes_after = geometry_hashes(cell.gbuffer())
        flat_variance = [variance(row["silhouette"]) for row in flat]
        shaded_variance = [variance(row["silhouette"]) for row in shaded]
        gates["S1_geometry_invariant_%s" % name] = {
            "gate": "S1_geometry_invariant_under_shading", "arm": name,
            "hashes": hashes_before, "passed": hashes_before == hashes_after,
            "thresholds": {"rule": "byte-identical geometry hashes across shading modes"}}
        gates["S2_flat_is_flat_%s" % name] = {
            "gate": "S2_flat_luminance_variance", "arm": name,
            "per_view_variance": flat_variance, "worst": max(flat_variance),
            "thresholds": {"variance_max": FLAT_VARIANCE_MAX},
            "passed": max(flat_variance) <= FLAT_VARIANCE_MAX}
        worst_shaded = min(shaded_variance)
        gates["S3_shading_varies_%s" % name] = {
            "gate": "S3_lambertian_luminance_variance", "arm": name,
            "per_view_variance": shaded_variance, "worst": worst_shaded,
            "views_below_threshold": [index for index, value in enumerate(shaded_variance)
                                      if value < SHADED_VARIANCE_MIN],
            "visible_triangles_by_view": {str(row["view"]): row["visible_triangles"]
                                          for row in cell.metrics()},
            "thresholds": {"variance_min": SHADED_VARIANCE_MIN},
            "passed": worst_shaded >= SHADED_VARIANCE_MIN}
        cells[name] = {"description": cell.description(),
                       "refused_views": cell.metrics_record()[1],
                       "flat_statistics": flat, "lambertian_statistics": shaded,
                       "light": "camera-relative, see characterization_pipeline.camera_relative_light"}
        del cell
    broken = failed(gates)
    summary = {"experiment": "RC-R3", "verdict": "SHADING_CHARACTERIZED" if not broken
               else "SHADING_GATE_FAILED", "failed_gates": broken, "gates": gates, "arms": cells,
               "area_match_fit": fit_record,
               "metric_scope": "generic image statistics only: luminance moments and quantiles, "
                               "RMS contrast, Sobel gradient magnitude, histogram entropy, "
                               "within-depth-quantile luminance spread",
               "scope": "Shading behaviour with geometry, camera, pose and depth held fixed by "
                        "construction. No detector score, tracking measure or learned model.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "r3", "device": args.device, "source_manifest": before,
              "resolution": list(PRIMARY_RESOLUTION), "views": grid.as_dict(),
              "shading_modes": ["flat", "lambertian"], "material_colour": 0.55,
              "arms": list(ARMS)}
    thresholds = {"flat_variance_max": FLAT_VARIANCE_MAX, "shaded_variance_min": SHADED_VARIANCE_MIN,
                  "min_silhouette_pixels": gm.MIN_SILHOUETTE_PIXELS,
                  "min_visible_triangles": gm.MIN_VISIBLE_TRIANGLES}
    return write_experiment(args.output, "RC-R3", config, summary, r3_readme(summary), thresholds,
                            runtime_fingerprint(include_device=args.device != "cpu"))


def r3_readme(summary):
    lines = ["# RC-R3 — shading, with geometry held fixed", "",
             "Verdict: **%s**" % summary["verdict"], ""]
    if summary["failed_gates"]:
        lines += ["Failed gates: " + ", ".join("`%s`" % name for name in summary["failed_gates"]),
                  "",
                  "A failed S3 is a measurement, not a defect to be tuned away: where only one",
                  "surface orientation is visible, lambertian shading is spatially constant by",
                  "construction. The per-view variances and visible-triangle counts are recorded",
                  "next to each other in `summary.json` so that reading is checkable.", ""]
    else:
        lines += ["Every preregistered gate S1–S3 passed.", ""]
    lines += ["| arm | flat variance (worst) | lambertian variance (worst) | views below S3 |",
              "|---|---|---|---|"]
    for name in summary["arms"]:
        flat = summary["gates"]["S2_flat_is_flat_%s" % name]
        shaded = summary["gates"]["S3_shading_varies_%s" % name]
        lines.append("| `%s` | %.3e | %.3e | %d of %d |" % (
            name, flat["worst"], shaded["worst"], len(shaded["views_below_threshold"]),
            len(shaded["per_view_variance"])))
    lines += ["", "Geometry is identical across shading modes by construction: one G-buffer per cell",
              "is rendered once and shaded repeatedly, and S1 re-hashes it afterwards to confirm",
              "shading did not mutate it.", "",
              "Reported statistics are ordinary image statistics (%s)." % summary["metric_scope"],
              "No detector, tracking or policy quantity appears here, and no cause of the D8b",
              "result is proposed (`causality_vs_d8b: NOT_TESTED`)."]
    return "\n".join(lines) + "\n"


def lighting_cells():
    for direction in R4_LIGHT_DIRECTIONS:
        for ambient in R4_AMBIENT:
            for directional in R4_DIRECTIONAL:
                yield direction, ambient, directional


def run_r4(args):
    from renderer_validation.characterization_pipeline import geometry_hashes
    from renderer_validation import geometry_metrics as gm
    from renderer_validation.view_grid import fit_grid
    from renderer_validation.rc_results import source_manifest, write_experiment
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    grid = fit_grid()
    cell = cell_for("quadrotor_mesh", PRIMARY_RESOLUTION, grid, args.device)
    reference_hashes = geometry_hashes(cell.gbuffer())
    reference_pixels = [row["silhouette_pixels"] for row in cell.metrics()]
    rows, means = [], []
    for direction, ambient, directional in lighting_cells():
        appearance = cell.appearance(colour=0.55, light_direction=direction, ambient=ambient,
                                     directional=directional)
        statistics = cell.image_statistics(appearance, "lambertian")
        pixels = [row["silhouette"]["pixels"] for row in statistics]
        cell_means = [row["silhouette"]["luminance_mean"] for row in statistics]
        means.extend(cell_means)
        rows.append({"light_direction": list(direction), "ambient": ambient,
                     "directional": directional,
                     "geometry_hashes_match": geometry_hashes(cell.gbuffer()) == reference_hashes,
                     "silhouette_pixels": pixels, "pixels_match": pixels == reference_pixels,
                     "silhouette_luminance_mean": cell_means,
                     "rgb_in_range": True, "statistics": statistics})
    spread = max(means) - min(means)
    gates = {
        "L1_geometry_and_silhouette_invariant": {
            "gate": "L1_geometry_and_silhouette_invariant",
            "cells": len(rows), "reference_hashes": reference_hashes,
            "passed": all(row["geometry_hashes_match"] and row["pixels_match"] for row in rows),
            "thresholds": {"rule": "identical geometry hashes and silhouette pixel counts"}},
        "L2_lighting_changes_luminance": {
            "gate": "L2_lighting_changes_luminance", "spread": spread,
            "minimum_mean": min(means), "maximum_mean": max(means),
            "thresholds": {"spread_min": LUMINANCE_SPREAD_MIN},
            "passed": spread >= LUMINANCE_SPREAD_MIN},
        "L3_rgb_finite_and_in_range": {
            "gate": "L3_rgb_finite_and_in_range", "cells": len(rows),
            "thresholds": {"rule": "image_statistics refuses RGB outside [0,1] or non-finite"},
            "passed": all(row["rgb_in_range"] for row in rows)},
    }
    broken = failed(gates)
    summary = {"experiment": "RC-R4", "verdict": "LIGHTING_CHARACTERIZED" if not broken
               else "LIGHTING_GATE_FAILED", "failed_gates": broken, "gates": gates,
               "arm": cell.description(), "views": grid.as_dict()["views"],
               "grid": {"light_directions": [list(d) for d in R4_LIGHT_DIRECTIONS],
                        "ambient": list(R4_AMBIENT), "directional": list(R4_DIRECTIONAL),
                        "cells": len(rows)},
               "cells": rows,
               "scope": "Lighting sensitivity of a fixed geometry. Material is varied separately in "
                        "RC-R4M so the two are never confounded.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "r4", "device": args.device, "source_manifest": before,
              "resolution": list(PRIMARY_RESOLUTION), "arm": "quadrotor_mesh",
              "views": grid.as_dict(), "material_colour": 0.55,
              "light_directions": [list(d) for d in R4_LIGHT_DIRECTIONS],
              "ambient": list(R4_AMBIENT), "directional": list(R4_DIRECTIONAL)}
    thresholds = {"luminance_spread_min": LUMINANCE_SPREAD_MIN,
                  "min_silhouette_pixels": gm.MIN_SILHOUETTE_PIXELS,
                  "min_visible_triangles": gm.MIN_VISIBLE_TRIANGLES}
    readme = grid_readme("RC-R4 — lighting sensitivity", summary, "light")
    return write_experiment(args.output, "RC-R4", config, summary, readme, thresholds,
                            runtime_fingerprint(include_device=args.device != "cpu"))


def run_r4m(args):
    from renderer_validation.characterization_pipeline import geometry_hashes
    from renderer_validation import geometry_metrics as gm
    from renderer_validation.view_grid import fit_grid
    from renderer_validation.rc_results import source_manifest, write_experiment
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    grid = fit_grid()
    cell = cell_for("quadrotor_mesh", PRIMARY_RESOLUTION, grid, args.device)
    reference_hashes = geometry_hashes(cell.gbuffer())
    reference_pixels = [row["silhouette_pixels"] for row in cell.metrics()]
    light = None
    rows, means = [], []
    for name, grey in R4M_MATERIALS:
        colour = np.asarray(R4M_PER_MATERIAL_RGB) if grey is None else grey
        appearance = cell.lit_appearance(colour=colour)
        if light is None:
            light = appearance.light_direction.tolist()
        statistics = cell.image_statistics(appearance, "lambertian")
        pixels = [row["silhouette"]["pixels"] for row in statistics]
        cell_means = [row["silhouette"]["luminance_mean"] for row in statistics]
        means.extend(cell_means)
        rows.append({"material_set": name,
                     "colour": colour.tolist() if grey is None else grey,
                     "geometry_hashes_match": geometry_hashes(cell.gbuffer()) == reference_hashes,
                     "silhouette_pixels": pixels, "pixels_match": pixels == reference_pixels,
                     "silhouette_luminance_mean": cell_means, "rgb_in_range": True,
                     "statistics": statistics})
    spread = max(means) - min(means)
    gates = {
        "M1_geometry_and_silhouette_invariant": {
            "gate": "M1_geometry_and_silhouette_invariant", "cells": len(rows),
            "reference_hashes": reference_hashes,
            "passed": all(row["geometry_hashes_match"] and row["pixels_match"] for row in rows),
            "thresholds": {"rule": "identical geometry hashes and silhouette pixel counts"}},
        "M2_material_changes_luminance": {
            "gate": "M2_material_changes_luminance", "spread": spread,
            "minimum_mean": min(means), "maximum_mean": max(means),
            "thresholds": {"spread_min": LUMINANCE_SPREAD_MIN},
            "passed": spread >= LUMINANCE_SPREAD_MIN},
        "M3_rgb_finite_and_in_range": {
            "gate": "M3_rgb_finite_and_in_range", "cells": len(rows),
            "thresholds": {"rule": "image_statistics refuses RGB outside [0,1] or non-finite"},
            "passed": all(row["rgb_in_range"] for row in rows)},
    }
    broken = failed(gates)
    summary = {"experiment": "RC-R4M", "verdict": "MATERIAL_CHARACTERIZED" if not broken
               else "MATERIAL_GATE_FAILED", "failed_gates": broken, "gates": gates,
               "arm": cell.description(), "views": grid.as_dict()["views"],
               "lighting_held_fixed": {"mode": "camera_relative", "world_directions": light,
                                       "ambient": 0.2, "directional": 0.8},
               "cells": rows,
               "scope": "Material sensitivity at fixed lighting, reported separately from RC-R4.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "r4m", "device": args.device, "source_manifest": before,
              "resolution": list(PRIMARY_RESOLUTION), "arm": "quadrotor_mesh",
              "views": grid.as_dict(),
              "material_sets": [name for name, _ in R4M_MATERIALS],
              "per_material_rgb": [list(c) for c in R4M_PER_MATERIAL_RGB]}
    thresholds = {"luminance_spread_min": LUMINANCE_SPREAD_MIN,
                  "min_silhouette_pixels": gm.MIN_SILHOUETTE_PIXELS,
                  "min_visible_triangles": gm.MIN_VISIBLE_TRIANGLES}
    readme = grid_readme("RC-R4M — material sensitivity", summary, "material")
    return write_experiment(args.output, "RC-R4M", config, summary, readme, thresholds,
                            runtime_fingerprint(include_device=args.device != "cpu"))


def grid_readme(title, summary, kind):
    gates = summary["gates"]
    invariant = next(record for name, record in gates.items() if name.endswith("invariant"))
    changes = next(record for name, record in gates.items() if "changes_luminance" in name)
    lines = ["# %s" % title, "", "Verdict: **%s**" % summary["verdict"], ""]
    if summary["failed_gates"]:
        lines += ["Failed gates: " + ", ".join("`%s`" % n for n in summary["failed_gates"]), ""]
    lines += ["* cells: %d" % invariant["cells"],
              "* geometry and silhouette identical in every cell: **%s**" % invariant["passed"],
              "* silhouette mean luminance spread across cells: %.4f (threshold %.2f)"
              % (changes["spread"], changes["thresholds"]["spread_min"]),
              "* luminance range: %.4f to %.4f" % (changes["minimum_mean"], changes["maximum_mean"]),
              "",
              "Changing the %s changes the image and nothing else: every geometry buffer hash and" % kind,
              "every silhouette pixel count is unchanged across the grid. That separation is the",
              "result; it is not evidence about any detector or policy, and it proposes no cause for",
              "the closed D8b result (`causality_vs_d8b: NOT_TESTED`)."]
    return "\n".join(lines) + "\n"


def run_determinism(args):
    """Export the same arm twice in two independent processes and compare every hash."""
    from renderer_validation.public_pipeline import verify_export
    from renderer_validation.rc_results import source_manifest, write_experiment
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    arguments = ["--arm", "quadrotor_mesh", "--views", "fit", "--width", "160", "--height", "120",
                 "--device", args.device, "--shading", "lambertian"]
    records = {}
    for run in ("run_a", "run_b"):
        subprocess.run([sys.executable, "-B", str(ROOT / "tools/export_renderer_dataset.py"),
                        "--output", str(output / run), *arguments],
                       check=True, cwd=str(ROOT), env=dict(os.environ, PYTHONNOUSERSITE="1"))
        records[run] = verify_export(output / run)
    first, second = records["run_a"], records["run_b"]
    arrays = {run: record["files"][0]["arrays"] for run, record in records.items()}
    names = sorted(arrays["run_a"])
    differing = [name for name in names if arrays["run_a"][name] != arrays["run_b"][name]]
    metadata_fields = ("schema", "description", "config")
    metadata_differences = {}
    for field in metadata_fields:
        left = json.loads(json.dumps(first.get(field), sort_keys=True))
        right = json.loads(json.dumps(second.get(field), sort_keys=True))
        if field == "config":
            left, right = dict(left), dict(right)
            left.pop("output"), right.pop("output")
        if left != right:
            metadata_differences[field] = {"run_a": left, "run_b": right}
    gates = {"D_identical_arrays": {"gate": "D_identical_arrays", "arrays": names,
                                    "differing": differing, "passed": not differing,
                                    "thresholds": {"rule": "identical sha256 per array"}},
             "D_identical_metadata": {"gate": "D_identical_metadata",
                                      "compared_fields": list(metadata_fields),
                                      "excluded": ["source", "runtime", "output path"],
                                      "differences": metadata_differences,
                                      "passed": not metadata_differences,
                                      "thresholds": {"rule": "identical apart from runtime and path"}}}
    broken = failed(gates)
    summary = {"experiment": "RC-determinism",
               "verdict": "DETERMINISTIC" if not broken else "NONDETERMINISTIC",
               "failed_gates": broken, "gates": gates,
               "array_sha256": {name: arrays["run_a"][name]["sha256"] for name in names},
               "processes": 2, "arguments": arguments,
               "scope": "Cross-process reproducibility of the export path.",
               "causality_vs_d8b": "NOT_TESTED"}
    config = {"command": "determinism", "device": args.device, "source_manifest": before,
              "arguments": arguments}
    readme = "\n".join([
        "# RC determinism gate", "", "Verdict: **%s**" % summary["verdict"], "",
        "The exporter ran twice in two independent OS processes with identical arguments. Every",
        "array sha256 is compared, and the metadata is compared apart from the runtime fingerprint",
        "and the output path.", "",
        "Arrays compared: " + ", ".join("`%s`" % name for name in summary["array_sha256"]), "",
        "Differing arrays: " + (", ".join(gates["D_identical_arrays"]["differing"]) or "none"), "",
    ]) + "\n"
    thresholds = {"rule": "byte-identical arrays and metadata across processes"}
    return write_experiment(output, "RC-determinism", config, summary, readme, thresholds,
                            runtime_fingerprint(include_device=args.device != "cpu"),
                            exist_ok=True)


COMMANDS = {"r1": run_r1, "r2": run_r2, "r3": run_r3, "r4": run_r4, "r4m": run_r4m,
            "determinism": run_determinism}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    receipt = COMMANDS[args.command](args)
    print("%s %s in %.1f s -> %s" % (receipt["experiment"], receipt["verdict"],
                                     time.perf_counter() - started, args.output))
    return 0 if "FAIL" not in receipt["verdict"] and "DEFECT" not in receipt["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
