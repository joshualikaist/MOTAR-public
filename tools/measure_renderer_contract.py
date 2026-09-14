#!/usr/bin/env python3
"""Preregistered R1b/R2b/R3b independent static graphics measurements."""
import argparse
from dataclasses import fields, replace
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renderer_validation.characterization_pipeline import (
    AnalyticSurface, appearance_for, camera_relative_light, geometry_hashes)
from renderer_validation.followup_records import ROOT, dump, finish, manifest, sha, verify
from renderer_validation.geometry_metrics import normal_histogram
from renderer_validation.image_statistics import luminance, convolve3, erode3, SOBEL_X, SOBEL_Y
from renderer_validation.scene import Camera, MeshScene
from renderer_validation.shading import shade
from renderer_validation.target_arms import build_arm, box_scene, BOX_HALF_EXTENTS_M, circumscribed_radius
from renderer_validation.transfer_strategies import rss_mib
from renderer_validation.view_grid import ViewGrid, primary_grid

SPECIMENS = ("analytic_sphere", "box_proxy", "quadrotor_mesh")
RESOLUTIONS = ((320, 240), (640, 480), (1280, 960), (2048, 1536))
OLD_SCALE = 0.8450780426128714
FIT_VIEWS = ((0., 0., 2.), (90., 0., 2.), (0., 35., 2.), (90., 35., 2.),
             (45., 20., 2.), (135., 20., 2.))
VALIDATION_VIEWS = tuple((22.5 + 45*k, e, 2.) for e in (-10., 15., 40.) for k in range(8))


def resource_check():
    resident = rss_mib()
    if (resident is not None and resident > 8192) or torch.cuda.memory_reserved() > 5*2**30:
        raise MemoryError("Preregistered renderer resource stop")


def render(specimen, resolution, view, scales=None):
    if resolution[0] * resolution[1] * 41 > 1.5 * 2**30:
        raise MemoryError("Predicted single output >1.5 GiB: %s %s %s" % (specimen, resolution, view))
    camera, grid = Camera(*resolution), ViewGrid([view])
    arm = build_arm(specimen)
    if scales is not None:
        if specimen != "box_proxy":
            raise ValueError("Scale vector is only supported for the new box control")
        mesh = box_scene(np.asarray(BOX_HALF_EXTENTS_M) * np.asarray(scales))
        arm = replace(arm, mesh=mesh, circumscribed_radius_m=circumscribed_radius(mesh))
    renderer = arm.renderer(camera, 1, "cuda:0") if arm.mesh is not None else None
    buffer = arm.gbuffer(camera, grid, "cuda:0", renderer)
    try:
        resource_check()
    except MemoryError as error:
        raise MemoryError("%s: %s %s %s" % (error, specimen, resolution, view)) from error
    return buffer, arm, camera, grid


def measure(buffer, camera):
    mask = buffer.valid[0].detach().cpu().numpy()
    row, col = np.nonzero(mask)
    if len(row) < 300:
        return {"status": "REFUSED", "reason": "fewer than 300 silhouette pixels", "area_px": len(row)}
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return {"status": "REFUSED", "reason": "border clipped", "area_px": len(row)}
    depth = buffer.depth_m[0].detach().cpu().numpy()[mask]
    if not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("Invalid visible optical depth")
    width, height = int(col.max()-col.min()+1), int(row.max()-row.min()+1)
    centre = np.array([col.mean(), row.mean()])
    f = camera.focal_px
    return {"status": "MEASURED", "area_px": int(len(row)), "width_px": width, "height_px": height,
            "centroid_px": centre.tolist(), "depth_median_m": float(np.median(depth)),
            "focal_px": f, "width_over_f": width/f, "height_over_f": height/f,
            "area_over_f2": len(row)/f**2,
            "centroid_over_f": ((centre-np.array([camera.width, camera.height])/2)/f).tolist()}


def errors(row, reference):
    if row["status"] != "MEASURED" or reference["status"] != "MEASURED":
        return None
    result = {}
    for key in ("width_over_f", "height_over_f", "area_over_f2", "depth_median_m"):
        delta = row[key]-reference[key]
        result[key] = {"signed": delta, "absolute": abs(delta), "relative": abs(delta)/abs(reference[key])}
    shift = np.asarray(row["centroid_over_f"])-reference["centroid_over_f"]
    distance = float(np.linalg.norm(shift))
    result["centroid_over_f"] = {"signed": shift.tolist(), "absolute": distance,
                                 "relative": distance/(2*math.sqrt(reference["area_over_f2"]/math.pi))}
    result["reference_in_current_pixels"] = {
        "width_px": reference["width_over_f"]*row["focal_px"],
        "height_px": reference["height_over_f"]*row["focal_px"],
        "area_px": reference["area_over_f2"]*row["focal_px"]**2,
        "centroid_shift_px": distance*row["focal_px"]}
    return result


def qualified(rows):
    for row in rows:
        error = row.get("errors")
        if not error:
            return False
        for name, tolerance in (("width_over_f", .02), ("height_over_f", .02), ("area_over_f2", .02),
                                ("centroid_over_f", .01), ("depth_median_m", .01)):
            if error[name]["relative"] > tolerance:
                return False
    return bool(rows)


def run_r1b(output, before):
    records = []
    views = primary_grid().views
    for specimen in SPECIMENS:
        for resolution in RESOLUTIONS:
            print("[R1b]", specimen, resolution, flush=True)
            for index, view in enumerate(views):
                buffer, _, camera, _ = render(specimen, resolution, view)
                row = {"specimen": specimen, "resolution": list(resolution), "view_index": index,
                       "view": list(view), **measure(buffer, camera), "geometry_hashes": geometry_hashes(buffer)}
                records.append(row)
                del buffer
            gc.collect()
            torch.cuda.empty_cache()
    by_key = {(r["specimen"], tuple(r["resolution"]), r["view_index"]): r for r in records}
    for row in records:
        reference = by_key[(row["specimen"], RESOLUTIONS[-1], row["view_index"])]
        row["errors"] = errors(row, reference)
    levels = []
    selected = None
    for resolution in RESOLUTIONS[:-1]:
        group = [r for r in records if r["resolution"] == list(resolution)]
        passes = len(group) == 72 and qualified(group)
        if selected is None and passes:
            selected = list(resolution)
        worst = {key: max((r["errors"][key]["relative"] for r in group if r["errors"]), default=None)
                 for key in ("width_over_f", "height_over_f", "area_over_f2", "centroid_over_f", "depth_median_m")}
        levels.append({"resolution": list(resolution), "qualified": passes, "worst_relative": worst})
    summary = {"verdict": "MEASUREMENT_RESOLUTION_QUALIFIED" if selected else "NO_QUALIFIED_RESOLUTION",
               "reference_resolution": list(RESOLUTIONS[-1]), "recommended_resolution": selected,
               "levels": levels, "rows": records, "analysis_unit": "view within specimen; no independent pixel inference"}
    prose = "# RC-R1b — resolution convergence\n\nVerdict: `%s`. Recommended resolution: `%s`.\n\n" % (summary["verdict"], selected)
    prose += "Reference is sampled 2048×1536, not exact GT and not eligible to recommend itself. "
    prose += "Width/height are inclusive; area and centered centroid are normalized by focal length. "
    prose += "Measured discrepancies are not a universal accuracy guarantee. RC-R1 remains GEOMETRY_DEFECT.\n"
    finish(output, before, {"resolutions": RESOLUTIONS, "views": list(views), "specimens": SPECIMENS}, summary, prose)


def projected_box_area(scales, views):
    """Exact perspective convex silhouette area in normalized image-plane units (not pixels)."""
    from scipy.spatial import ConvexHull
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Three positive finite scale factors required")
    vertices = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = vertices * np.asarray(BOX_HALF_EXTENTS_M) * scales
    grid = ViewGrid(views)
    areas = []
    for position, rotation in zip(grid.positions, grid.rotations()):
        points = (vertices-position) @ rotation
        if (points[:, 2] <= 0).any():
            raise ValueError("Box crosses camera plane")
        projected = points[:, :2]/points[:, 2:3]
        areas.append(float(ConvexHull(projected).volume))
    return np.asarray(areas)


def fit_scales(areas):
    from scipy.optimize import least_squares
    areas = np.asarray(areas, dtype=float)
    if areas.shape != (len(FIT_VIEWS),) or not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("Calibration areas invalid")
    result = least_squares(lambda x: np.log(projected_box_area(x, FIT_VIEWS)/areas),
        np.full(3, OLD_SCALE), bounds=(.2, 2.), max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return {"scales": result.x.tolist(), "success": bool(result.success), "message": str(result.message),
            "nfev": result.nfev, "cost": float(result.cost), "frozen_calibration_views": FIT_VIEWS}


def area_statistics(observed, reference):
    observed, reference = np.asarray(observed, dtype=float), np.asarray(reference, dtype=float)
    if (observed.shape != reference.shape or observed.ndim != 1 or not len(reference)
            or not np.isfinite(observed).all() or not np.isfinite(reference).all()
            or (reference <= 0).any() or (observed < 0).any()):
        raise ValueError("Invalid area comparisons")
    error = np.abs(observed/reference-1)
    return {"median_absolute_relative_error": float(np.median(error)),
            "p90_absolute_relative_error": float(np.percentile(error, 90)),
            "worst_absolute_relative_error": float(error.max()), "per_view_absolute_relative_error": error.tolist()}


def area_pass(stats):
    return (stats["median_absolute_relative_error"] <= .05 and
            stats["p90_absolute_relative_error"] <= .10 and stats["worst_absolute_relative_error"] <= .20)


def run_r2b(output, before):
    import scipy
    assert not set(FIT_VIEWS) & set(VALIDATION_VIEWS)
    calibration = []
    for view in FIT_VIEWS:
        buffer, _, camera, _ = render("quadrotor_mesh", (1280, 960), view)
        calibration.append(measure(buffer, camera))
        del buffer
    if any(r["status"] != "MEASURED" for r in calibration):
        finish(output, before, {"fit_views": FIT_VIEWS}, {"verdict": "FIT_FAILED", "calibration": calibration},
               "# RC-R2b\n\nCalibration views refused; no parameters or held-out measurements selected.")
        return
    fitted = fit_scales([r["area_over_f2"] for r in calibration])
    dump(output / "fit.json", {"fit": fitted, "calibration": calibration, "scipy": scipy.__version__})
    frozen_sha = sha(output / "fit.json")
    if not fitted["success"]:
        finish(output, before, {"fit_views": FIT_VIEWS}, {"verdict": "FIT_FAILED", "fit": fitted},
               "# RC-R2b\n\nOptimizer failed; no refit or held-out model selection.")
        return
    phases = {}
    for phase, views in (("fit", FIT_VIEWS), ("validation", VALIDATION_VIEWS)):
        print("[R2b]", phase, "frozen scales", fitted["scales"], flush=True)
        data = {"mesh": [], "C0": [], "C1": []}
        for view in views:
            for key, specimen, scales in (("mesh", "quadrotor_mesh", None),
                    ("C0", "box_proxy", [OLD_SCALE]*3), ("C1", "box_proxy", fitted["scales"])):
                buffer, _, camera, _ = render(specimen, (1280, 960), view, scales)
                data[key].append({"view": view, **measure(buffer, camera)})
                del buffer
        refused = any(row["status"] != "MEASURED" for rows in data.values() for row in rows)
        stats = {} if refused else {key: area_statistics([r["area_px"] for r in data[key]],
                   [r["area_px"] for r in data["mesh"]]) for key in ("C0", "C1")}
        phases[phase] = {"rows": data, "statistics": stats, "refused": refused}
    if sha(output / "fit.json") != frozen_sha:
        raise RuntimeError("Frozen fit changed during evaluation")
    validation = phases["validation"]
    passed = not validation["refused"] and area_pass(validation["statistics"]["C1"])
    r1 = ROOT / "results/renderer_characterization_r1b_2026-09-14/summary.json"
    resolution_record = json.loads(r1.read_text())
    supported = next(r["qualified"] for r in resolution_record["levels"] if r["resolution"] == [1280, 960])
    summary = {"verdict": "AREA_CONTROL_MATCHED" if passed else "AREA_CONTROL_FAILED",
               "fit": fitted, "frozen_fit_sha256": frozen_sha, "phases": phases,
               "r1b_summary_sha256": sha(r1), "measurement_resolution_qualified": supported,
               "historical_c0_scale": OLD_SCALE}
    prose = "# RC-R2b — anisotropic area-control family\n\nVerdict: `%s`.\n\n" % summary["verdict"]
    prose += "Six fit views; parameters frozen in fit.json before 24 disjoint held-out angles. "
    prose += "C0 is the original frozen isotropic box, never refitted. C1 has three scale parameters. "
    prose += "The same previously explored specimen is used: no independent-object generalization claim. "
    prose += "Finite 1280×960 measurement support from R1b: `%s`.\n" % supported
    finish(output, before, {"resolution": [1280, 960], "fit_views": FIT_VIEWS,
            "validation_views": VALIDATION_VIEWS, "scipy": scipy.__version__}, summary, prose)


def boundary_contrast(rgb, mask):
    y = luminance(rgb)
    pieces = []
    for left, right, transition in ((y[:, :-1], y[:, 1:], mask[:, :-1] != mask[:, 1:]),
                                    (y[:-1, :], y[1:, :], mask[:-1, :] != mask[1:, :])):
        pieces.append(np.abs(left-right)[transition])
    values = np.concatenate(pieces)
    return float(values.mean()) if len(values) else None


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return {"pearson_r": None, "reason": "constant within numerical precision (1e-12)"}
    return {"pearson_r": float(np.corrcoef(x, y)[0, 1]), "reason": None}


def run_r3b(output, before):
    rows = []
    fields_present = [f.name for f in fields(MeshScene)]
    for specimen in SPECIMENS:
        print("[R3b]", specimen, flush=True)
        for index, view in enumerate(primary_grid().views):
            buffer, arm, camera, grid = render(specimen, (640, 480), view)
            base = measure(buffer, camera)
            if base["status"] != "MEASURED":
                rows.append({"specimen": specimen, "view": view, **base})
                del buffer
                continue
            before_hashes = geometry_hashes(buffer)
            surface = arm.mesh if arm.mesh is not None else AnalyticSurface()
            light = appearance_for(1, surface.material_count, light_direction=camera_relative_light(grid))
            rgb = shade(buffer, surface, light).detach().cpu().numpy()[0]
            mask = buffer.valid[0].detach().cpu().numpy()
            normals = buffer.normal_world[0].detach().cpu().numpy()[mask].astype(float) @ grid.rotations()[0]
            histogram = normal_histogram(normals)
            y = luminance(rgb)
            gradient = np.hypot(convolve3(y, SOBEL_X), convolve3(y, SOBEL_Y))
            interior = erode3(mask)
            unchanged = before_hashes == geometry_hashes(buffer)
            if not unchanged:
                raise RuntimeError("Shading mutated geometry")
            rows.append({"specimen": specimen, "view": view, "view_index": index, **base,
                         "normal_representation": "N0_face" if arm.mesh is not None else "analytic_reference",
                         "normal_histogram": histogram, "mean_abs_n_dot_view": float(np.abs(normals[:, 2]).mean()),
                         "luminance_variance": float(y[mask].var()), "edge_contrast": boundary_contrast(rgb, mask),
                         "interior_gradient_mean": float(gradient[interior].mean()) if interior.any() else None,
                         "geometry_hashes": before_hashes, "geometry_unchanged_by_shading": unchanged})
            del buffer
    pairs = {}
    for specimen in SPECIMENS:
        group = [r for r in rows if r["specimen"] == specimen and r["status"] == "MEASURED"]
        pairs[specimen] = correlation([r["normal_histogram"]["entropy_bits"] for r in group],
                                      [r["luminance_variance"] for r in group]) if group else None
    summary = {"verdict": "NORMAL_FIELD_CHARACTERIZED" if all(r["status"] == "MEASURED" for r in rows)
               else "INCOMPLETE_REFUSED_VIEWS", "rows": rows, "descriptive_correlations": pairs,
               "mesh_scene_fields": fields_present,
               "N1": {"status": "UNSUPPORTED", "reason": "No authored vertex normals in source/loader contract"},
               "N2": {"status": "UNSUPPORTED", "reason": "No authored smooth-normal/smoothing-group contract"}}
    prose = "# RC-R3b — visible normal field\n\nVerdict: `%s`.\n\n" % summary["verdict"]
    prose += "N0 uses source geometry face normals; sphere is an analytic reference. N1/N2 are UNSUPPORTED, "
    prose += "not silently synthesized or treated as zero-effect experiments. Scatter/correlation describes "
    prose += "views, not a causal intervention or triangle-count effect. Historical RC-R3 remains SHADING_GATE_FAILED.\n"
    finish(output, before, {"resolution": [640, 480], "views": primary_grid().as_dict(),
                           "specimens": SPECIMENS, "normal_bins": [8, 4]}, summary, prose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("r1b", "r2b", "r3b", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "verify":
        print(verify(args.output))
    else:
        before = manifest()
        args.output.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(20260914)
        try:
            {"r1b": run_r1b, "r2b": run_r2b, "r3b": run_r3b}[args.stage](args.output, before)
        except MemoryError as error:
            finish(args.output, before, {"stage": args.stage},
                   {"verdict": "INCOMPLETE_RESOURCE_STOP", "reason": str(error)},
                   "# Resource stop\n\nNo downscaling, threshold change or completion claim. " + str(error))
        print("[RC]", verify(args.output), flush=True)
