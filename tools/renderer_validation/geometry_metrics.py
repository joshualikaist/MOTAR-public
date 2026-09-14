"""RC-R1 geometry measurements and gates.

Every threshold in this module is the one written in
results/renderer_characterization_2026-09-13/PREREGISTRATION.md sections 2-3, before any of this
code existed. They are module constants so a receipt can record what was actually applied, and so
a change to one of them shows up as a source change in the provenance hash.

A degenerate view raises. That is deliberate: a 40-pixel silhouette still produces a centroid and a
depth range, and those numbers would look like measurements.
"""
import math

import numpy as np
import torch

MIN_SILHOUETTE_PIXELS = 300
MIN_VISIBLE_TRIANGLES = 2  # Preregistration amendment 1: a box face is exactly two triangles.
NORMAL_AZIMUTH_BINS = 8
NORMAL_POLAR_BINS = 4
SILHOUETTE_DISAGREEMENT_MAX = 0.005          # G1
RANGE_DIFFERENCE_MAX_M = 1.0e-3              # G1
SPHERE_RADIUS_TOLERANCE_PX = 1.0             # G2
SPHERE_CENTROID_TOLERANCE_PX = 0.5           # G2
SPHERE_DEPTH_TOLERANCE_M = 2.0e-3            # G2
BOX_BBOX_TOLERANCE_PX = 1.0                  # G3
INVERSE_DISTANCE_TOLERANCE = 0.02            # G4
DEPTH_BRACKET_SLACK_M = 2.0e-3               # G5
RESOLUTION_CONSISTENCY_TOLERANCE = 0.02      # G7


def equivalent_diameter_px(area_px):
    return 2.0 * math.sqrt(float(area_px) / math.pi)


def normal_histogram(normals_camera):
    """Counts over a fixed 8x4 direction grid, with the entropy and the dominant-bin share.

    The binning is fixed here rather than derived from the data, so two specimens are always
    described on the same grid.
    """
    normals = np.asarray(normals_camera, dtype=np.float64).reshape(-1, 3)
    if not len(normals):
        raise ValueError("No normals to bin")
    azimuth = np.arctan2(normals[:, 1], normals[:, 0])
    polar = np.arccos(np.clip(normals[:, 2], -1.0, 1.0))
    azimuth_bin = np.clip(((azimuth + math.pi) / (2 * math.pi) * NORMAL_AZIMUTH_BINS).astype(int),
                          0, NORMAL_AZIMUTH_BINS - 1)
    polar_bin = np.clip((polar / math.pi * NORMAL_POLAR_BINS).astype(int), 0, NORMAL_POLAR_BINS - 1)
    counts = np.zeros(NORMAL_AZIMUTH_BINS * NORMAL_POLAR_BINS, dtype=np.int64)
    np.add.at(counts, azimuth_bin * NORMAL_POLAR_BINS + polar_bin, 1)
    share = counts / counts.sum()
    nonzero = share[share > 0.0]
    return {"counts": counts.tolist(),
            "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
            "dominant_bin_fraction": float(share.max()),
            "occupied_bins": int((counts > 0).sum()),
            "bins": {"azimuth": NORMAL_AZIMUTH_BINS, "polar": NORMAL_POLAR_BINS}}


def view_metrics(gbuffer, index, camera, rotation, has_triangles,
                 min_silhouette_pixels=MIN_SILHOUETTE_PIXELS,
                 min_visible_triangles=MIN_VISIBLE_TRIANGLES):
    """Generic silhouette, depth and normal statistics for one rendered view.

    G6 lives here: a view too small to measure raises instead of returning numbers.
    """
    valid = gbuffer.valid[index]
    if valid.ndim != 2 or valid.shape != (camera.height, camera.width):
        raise ValueError("Validity mask must be [H,W] matching the camera")
    # Integer count, never mean(): an all-true CUDA mean can fall an ulp short of 1.0.
    area = int(valid.sum().item())
    face = gbuffer.face_id[index]
    visible_triangles = int(torch.unique(face[valid]).numel()) if area else 0
    if area < int(min_silhouette_pixels):
        raise RuntimeError(f"Degenerate view {index}: {area} silhouette pixels is below the "
                           f"preregistered minimum of {min_silhouette_pixels}")
    if has_triangles and visible_triangles < int(min_visible_triangles):
        raise RuntimeError(f"Degenerate view {index}: {visible_triangles} visible triangles is "
                           f"below the preregistered minimum of {min_visible_triangles}")
    mask = valid.detach().cpu().numpy()
    rows, columns = np.nonzero(mask)
    depth = gbuffer.depth_m[index].detach().cpu().numpy().astype(np.float64)[mask]
    ranges = gbuffer.range_m[index].detach().cpu().numpy().astype(np.float64)[mask]
    normals_world = gbuffer.normal_world[index].detach().cpu().numpy().astype(np.float64)[mask]
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    normals_camera = normals_world @ rotation          # world -> camera is R^T, applied on the right
    forward = rotation[:, 2]
    instance = gbuffer.instance_id[index]
    return {
        "view": int(index),
        "silhouette_pixels": area,
        "silhouette_fraction": area / float(camera.width * camera.height),
        "bbox_u_min": int(columns.min()), "bbox_u_max": int(columns.max()),
        "bbox_v_min": int(rows.min()), "bbox_v_max": int(rows.max()),
        "bbox_width_px": int(columns.max() - columns.min() + 1),
        "bbox_height_px": int(rows.max() - rows.min() + 1),
        "equivalent_diameter_px": equivalent_diameter_px(area),
        "centroid_u_px": float(columns.mean()), "centroid_v_px": float(rows.mean()),
        "depth_min_m": float(depth.min()), "depth_median_m": float(np.median(depth)),
        "depth_max_m": float(depth.max()),
        "range_min_m": float(ranges.min()), "range_median_m": float(np.median(ranges)),
        "range_max_m": float(ranges.max()),
        "visible_triangles": visible_triangles,
        "visible_instances": int(torch.unique(instance[valid]).numel()) if area else 0,
        "mean_abs_normal_dot_view": float(np.abs(normals_world @ forward).mean()),
        "normal_histogram": normal_histogram(normals_camera),
    }


def arm_metrics(arm, gbuffer, camera, grid):
    """view_metrics for every view of a grid, in grid order. Raises on any degenerate view."""
    rotations = grid.rotations()
    if gbuffer.valid.shape[0] != len(grid):
        raise ValueError("One rendered view per grid entry is required")
    return [view_metrics(gbuffer, index, camera, rotations[index], arm.triangles > 0)
            for index in range(len(grid))]


def measured_arm_metrics(arm, gbuffer, camera, grid):
    """Metrics for the views that can be measured, plus the refusals, kept rather than swallowed.

    G6 refuses a view that is too small to measure. The preregistration's words are that such a view
    "is never reported as a measurement" - not that the experiment ends. A specimen that is thin from
    some direction genuinely disappears at distance, and that is a fact about the specimen and the
    resolution, so it is recorded as a refusal and every gate is then evaluated on the views that
    were measured. The threshold itself is untouched.
    """
    rotations = grid.rotations()
    if gbuffer.valid.shape[0] != len(grid):
        raise ValueError("One rendered view per grid entry is required")
    measured, refused = [], []
    for index in range(len(grid)):
        try:
            measured.append(view_metrics(gbuffer, index, camera, rotations[index],
                                         arm.triangles > 0))
        except RuntimeError as error:
            azimuth, elevation, distance = grid.views[index]
            refused.append({"view": index, "azimuth_deg": azimuth, "elevation_deg": elevation,
                            "distance_m": distance, "reason": str(error),
                            "silhouette_pixels": int(gbuffer.valid[index].sum().item())})
    if not measured:
        raise RuntimeError("Every view of this grid was refused; there is nothing to measure")
    return measured, refused


def _column(rows, name):
    return [row[name] for row in rows]


def gate_cross_implementation(arm, gbuffer, camera, grid):
    """G1: the Warp G-buffer against an independent CPU Moller-Trumbore intersector.

    The reference shares no intersection code with the GPU path; only the camera ray expression is
    shared, and that is the thing both are being asked about.
    """
    from .analytic_primitives import world_rays
    from .reference_raycast import intersect
    if arm.mesh is None:
        raise TypeError("G1 applies to mesh arms; the analytic arm is checked by G2")
    rotations = grid.rotations()
    per_view = []
    for index in range(len(grid)):
        origin, directions = world_rays(camera, grid.positions[index], rotations[index])
        flat = directions.reshape(-1, 3)
        origins = np.repeat(origin[None, :], len(flat), axis=0)
        distance, _, _, hit = intersect(arm.mesh, origins, flat, far=camera.far_range_m)
        reference_hit = hit.reshape(camera.height, camera.width)
        measured_hit = gbuffer.valid[index].detach().cpu().numpy()
        reference_pixels = int(reference_hit.sum())
        if reference_pixels == 0:
            raise RuntimeError(f"Reference intersector sees nothing in view {index}")
        disagreement = int(np.logical_xor(reference_hit, measured_hit).sum())
        both = reference_hit & measured_hit
        measured_range = gbuffer.range_m[index].detach().cpu().numpy().astype(np.float64)
        reference_range = distance.reshape(camera.height, camera.width)
        difference = float(np.abs(measured_range[both] - reference_range[both]).max()) if both.any() else 0.0
        per_view.append({"view": index, "reference_pixels": reference_pixels,
                         "measured_pixels": int(measured_hit.sum()),
                         "disagreeing_pixels": disagreement,
                         "disagreement_fraction": disagreement / reference_pixels,
                         "agreed_pixels": int(both.sum()),
                         "max_range_difference_m": difference})
    worst_mask = max(_column(per_view, "disagreement_fraction"))
    worst_range = max(_column(per_view, "max_range_difference_m"))
    return {"gate": "G1_cross_implementation", "arm": arm.name, "per_view": per_view,
            "worst_disagreement_fraction": worst_mask, "worst_range_difference_m": worst_range,
            "thresholds": {"disagreement_fraction_max": SILHOUETTE_DISAGREEMENT_MAX,
                           "range_difference_max_m": RANGE_DIFFERENCE_MAX_M},
            "passed": bool(worst_mask <= SILHOUETTE_DISAGREEMENT_MAX
                           and worst_range <= RANGE_DIFFERENCE_MAX_M)}


def gate_sphere_closed_form(arm, metrics, camera, grid):
    """G2: the analytic arm against the sphere's exact projected radius, centre and near point."""
    from .analytic_primitives import project_points, sphere_projected_radius_px
    if arm.kind != "analytic_sphere":
        raise TypeError("G2 applies to the analytic sphere arm")
    rotations = grid.rotations()
    per_view = []
    for row in metrics:
        index = row["view"]
        distance = grid.views[index][2]
        expected_radius = sphere_projected_radius_px(camera, distance, arm.radius_m)
        centre, _ = project_points(camera, [[0.0, 0.0, 0.0]], grid.positions[index], rotations[index])
        measured_radius = row["equivalent_diameter_px"] / 2.0
        per_view.append({
            "view": index,
            "radius_error_px": abs(measured_radius - expected_radius),
            "centroid_error_px": float(np.hypot(row["centroid_u_px"] - centre[0, 0],
                                                row["centroid_v_px"] - centre[0, 1])),
            "depth_error_m": abs(row["depth_min_m"] - (distance - arm.radius_m)),
            "expected_radius_px": expected_radius, "measured_radius_px": measured_radius})
    return {"gate": "G2_sphere_closed_form", "arm": arm.name, "per_view": per_view,
            "worst_radius_error_px": max(_column(per_view, "radius_error_px")),
            "worst_centroid_error_px": max(_column(per_view, "centroid_error_px")),
            "worst_depth_error_m": max(_column(per_view, "depth_error_m")),
            "thresholds": {"radius_px": SPHERE_RADIUS_TOLERANCE_PX,
                           "centroid_px": SPHERE_CENTROID_TOLERANCE_PX,
                           "depth_m": SPHERE_DEPTH_TOLERANCE_M},
            "passed": bool(max(_column(per_view, "radius_error_px")) <= SPHERE_RADIUS_TOLERANCE_PX
                           and max(_column(per_view, "centroid_error_px")) <= SPHERE_CENTROID_TOLERANCE_PX
                           and max(_column(per_view, "depth_error_m")) <= SPHERE_DEPTH_TOLERANCE_M)}


def gate_box_closed_form(arm, metrics, camera, grid):
    """G3: the measured silhouette box against the projection of the eight corners.

    A pixel index is sampled at its integer coordinate, so a measured edge may sit up to one pixel
    inside the analytic edge; the preregistered tolerance is one pixel on every side.
    """
    from .analytic_primitives import project_points
    if arm.mesh is None or arm.description.get("kind") != "box":
        raise TypeError("G3 applies to the box arms")
    vertices = np.asarray(arm.mesh.vertices, dtype=np.float64)
    corners = np.unique(vertices, axis=0)
    if len(corners) != 8:
        raise ValueError("A box arm must have eight distinct corners")
    rotations = grid.rotations()
    per_view = []
    for row in metrics:
        index = row["view"]
        uv, _ = project_points(camera, corners, grid.positions[index], rotations[index])
        expected = {"u_min": float(uv[:, 0].min()), "u_max": float(uv[:, 0].max()),
                    "v_min": float(uv[:, 1].min()), "v_max": float(uv[:, 1].max())}
        errors = {"u_min": abs(row["bbox_u_min"] - expected["u_min"]),
                  "u_max": abs(row["bbox_u_max"] - expected["u_max"]),
                  "v_min": abs(row["bbox_v_min"] - expected["v_min"]),
                  "v_max": abs(row["bbox_v_max"] - expected["v_max"])}
        per_view.append({"view": index, "expected": expected, "errors_px": errors,
                         "worst_error_px": max(errors.values())})
    worst = max(_column(per_view, "worst_error_px"))
    return {"gate": "G3_box_closed_form", "arm": arm.name, "per_view": per_view,
            "worst_error_px": worst, "thresholds": {"bbox_px": BOX_BBOX_TOLERANCE_PX},
            "passed": bool(worst <= BOX_BBOX_TOLERANCE_PX)}


def gate_inverse_distance(arm, metrics, grid):
    """G4: equivalent diameter times distance is constant along each line of sight."""
    groups = {}
    for row in metrics:
        azimuth, elevation, distance = grid.views[row["view"]]
        groups.setdefault((azimuth, elevation), []).append(
            (distance, row["equivalent_diameter_px"] * distance))
    per_line, unusable = [], []
    for (azimuth, elevation), rows in sorted(groups.items()):
        if len(rows) < 2:
            # One surviving distance carries no scaling information. Recorded, never silently
            # dropped and never padded with a refused view's numbers.
            unusable.append({"azimuth_deg": azimuth, "elevation_deg": elevation,
                             "measured_distances_m": [distance for distance, _ in sorted(rows)]})
            continue
        products = [product for _, product in sorted(rows)]
        spread = max(products) / min(products) - 1.0
        per_line.append({"azimuth_deg": azimuth, "elevation_deg": elevation,
                         "distances_m": [distance for distance, _ in sorted(rows)],
                         "diameter_times_distance_px_m": products, "relative_spread": spread})
    if not per_line:
        raise RuntimeError("No line of sight kept two distances; G4 cannot be evaluated")
    worst = max(_column(per_line, "relative_spread"))
    return {"gate": "G4_inverse_distance_scaling", "arm": arm.name, "per_line_of_sight": per_line,
            "lines_without_two_distances": unusable, "worst_relative_spread": worst,
            "thresholds": {"relative_spread_max": INVERSE_DISTANCE_TOLERANCE},
            "passed": bool(worst <= INVERSE_DISTANCE_TOLERANCE)}


def gate_depth_bracket(arm, metrics, grid):
    """G5: no surface is nearer or farther than the specimen's circumscribed sphere allows."""
    per_view = []
    for row in metrics:
        index = row["view"]
        distance = grid.views[index][2]
        radius = arm.circumscribed_radius_m
        lower = distance - radius - DEPTH_BRACKET_SLACK_M
        upper = distance + radius + DEPTH_BRACKET_SLACK_M
        per_view.append({"view": index, "depth_min_m": row["depth_min_m"],
                         "depth_max_m": row["depth_max_m"], "lower_bound_m": lower,
                         "upper_bound_m": upper,
                         "inside": bool(row["depth_min_m"] >= lower and row["depth_max_m"] <= upper)})
    return {"gate": "G5_depth_bracket", "arm": arm.name, "per_view": per_view,
            "circumscribed_radius_m": arm.circumscribed_radius_m,
            "thresholds": {"slack_m": DEPTH_BRACKET_SLACK_M},
            "passed": bool(all(row["inside"] for row in per_view))}


def gate_resolution_consistency(arm, low_metrics, high_metrics, low_camera, high_camera):
    """G7: silhouette size scales with focal length, so the measure is not a resolution artefact.

    A view measured at one resolution and refused at the other carries no comparison, so it is
    listed rather than matched up by position against a different view.
    """
    low_by_view = {row["view"]: row for row in low_metrics}
    high_by_view = {row["view"]: row for row in high_metrics}
    shared = sorted(set(low_by_view) & set(high_by_view))
    if not shared:
        raise RuntimeError("No view was measured at both resolutions; G7 cannot be evaluated")
    per_view = []
    for index in shared:
        ratio_low = low_by_view[index]["equivalent_diameter_px"] / low_camera.focal_px
        ratio_high = high_by_view[index]["equivalent_diameter_px"] / high_camera.focal_px
        per_view.append({"view": index, "low": ratio_low, "high": ratio_high,
                         "relative_difference": abs(ratio_high / ratio_low - 1.0)})
    worst = max(_column(per_view, "relative_difference"))
    return {"gate": "G7_resolution_consistency", "arm": arm.name, "per_view": per_view,
            "views_measured_at_one_resolution_only":
                sorted(set(low_by_view) ^ set(high_by_view)),
            "worst_relative_difference": worst,
            "thresholds": {"relative_difference_max": RESOLUTION_CONSISTENCY_TOLERANCE},
            "passed": bool(worst <= RESOLUTION_CONSISTENCY_TOLERANCE)}


def thresholds():
    """Exactly what this module applied, for the receipt."""
    return {"min_silhouette_pixels": MIN_SILHOUETTE_PIXELS,
            "min_visible_triangles": MIN_VISIBLE_TRIANGLES,
            "silhouette_disagreement_max": SILHOUETTE_DISAGREEMENT_MAX,
            "range_difference_max_m": RANGE_DIFFERENCE_MAX_M,
            "sphere_radius_tolerance_px": SPHERE_RADIUS_TOLERANCE_PX,
            "sphere_centroid_tolerance_px": SPHERE_CENTROID_TOLERANCE_PX,
            "sphere_depth_tolerance_m": SPHERE_DEPTH_TOLERANCE_M,
            "box_bbox_tolerance_px": BOX_BBOX_TOLERANCE_PX,
            "inverse_distance_tolerance": INVERSE_DISTANCE_TOLERANCE,
            "depth_bracket_slack_m": DEPTH_BRACKET_SLACK_M,
            "resolution_consistency_tolerance": RESOLUTION_CONSISTENCY_TOLERANCE,
            "normal_bins": [NORMAL_AZIMUTH_BINS, NORMAL_POLAR_BINS]}
