"""Preregistered R4 appearance statistics.

Thresholds live only here and in docs/preregistration_renderer_r4_2026-09-11.md. The
preregistration fixed each criterion but not how to aggregate it over scenes; this module takes
the strict reading in each direction (worst scene must pass) and records every per-scene value
so any other rule can be applied to the receipt afterwards.

R-squared is a diagnostic. A low value alone passes nothing: criterion D has to pass with it.
"""
import numpy as np
import torch

from .appearance_models import BT709, render_arms
from .validation import tensor_hash

PREREGISTERED_ARMS = ("flat", "depth_gradient", "lambertian")
RANGE_QUANTILE_BINS = 20
FLAT_VARIANCE_MAX = 1e-12
DEPTH_R2_MIN = 0.99
LAMBERTIAN_R2_MAX = 0.5
WITHIN_BIN_RATIO_MIN = 3.0
MIN_COVERAGE = 0.05
MIN_VISIBLE_FACES = 4
MIN_VALID_PIXELS = 500


def luminance(rgb):
    weights = torch.tensor(BT709, dtype=torch.float64, device=rgb.device)
    return (rgb.double() * weights).sum(dim=-1)


def determination(y, x):
    """R^2 of the simple linear regression of luminance on range, in float64."""
    if y.numel() < 3:
        raise ValueError("Too few pixels to regress")
    total = ((y - y.mean()) ** 2).sum()
    spread = ((x - x.mean()) ** 2).sum()
    if float(total) <= 0.0:
        return 0.0 if float(spread) > 0.0 else float("nan")
    if float(spread) <= 0.0:
        return 0.0
    covariance = ((x - x.mean()) * (y - y.mean())).sum()
    return float((covariance ** 2 / (spread * total)).clamp(0.0, 1.0))


def equal_count_bins(values, count):
    """Rank-based bins, so every arm is compared inside the same geometric grouping."""
    order = torch.argsort(values, stable=True)
    return [chunk for chunk in torch.tensor_split(order, count) if chunk.numel() >= 2]


def scene_statistics(rgb, gbuffer, materials, index):
    mask = gbuffer.valid[index]
    y = luminance(rgb[index])[mask]
    ranges = gbuffer.range_m[index][mask].double()
    material = materials[index][mask]
    per_material = []
    for value in sorted(set(material.tolist())):
        selected = y[material == value]
        per_material.append({"material": int(value), "pixels": int(selected.numel()),
                             "luminance_mean": float(selected.mean()),
                             "luminance_variance": float(selected.var(unbiased=False))})
    bins = equal_count_bins(ranges, RANGE_QUANTILE_BINS)
    within = [float(y[chunk].std(unbiased=False)) for chunk in bins]
    if not within:
        raise ValueError("No range bin holds two pixels; there is no within-bin spread to report")
    channels = rgb[index][mask]
    return {
        "valid_pixels": int(mask.sum()),
        "luminance_mean": float(y.mean()),
        "luminance_variance": float(y.var(unbiased=False)),
        "luminance_min": float(y.min()),
        "luminance_max": float(y.max()),
        "range_luminance_r_squared": determination(y, ranges),
        "within_range_bin_luminance_std_median": float(np.median(within)),
        "within_range_bin_count": len(bins),
        "saturated_pixel_fraction": float((channels >= 1.0 - 1e-6).any(dim=-1).double().mean()),
        "black_pixel_fraction": float((channels <= 1e-6).all(dim=-1).double().mean()),
        "per_material": per_material,
    }


def column(statistics, name):
    return [scene[name] for scene in statistics]


def evaluate_r4(scene, gbuffer, appearance, camera):
    count = gbuffer.valid.shape[0]
    table = torch.tensor(scene.face_material.copy(), dtype=torch.long, device=gbuffer.face_id.device)
    materials = table[gbuffer.face_id.clamp_min(0).long()]
    coverage = [float(gbuffer.valid[i].double().mean()) for i in range(count)]
    faces = [int(torch.unique(gbuffer.face_id[i][gbuffer.valid[i]]).numel()) for i in range(count)]
    pixels = [int(gbuffer.valid[i].sum()) for i in range(count)]
    # Fixture guard, not a criterion: a degenerate view must refuse rather than produce statistics.
    if min(coverage) < MIN_COVERAGE or min(faces) < MIN_VISIBLE_FACES or min(pixels) < MIN_VALID_PIXELS:
        raise RuntimeError(f"Degenerate fixture view: coverage {min(coverage):.4f}, "
                           f"faces {min(faces)}, pixels {min(pixels)}")
    arms = render_arms(gbuffer, scene, appearance, camera)
    statistics = {name: [scene_statistics(rgb, gbuffer, materials, i) for i in range(count)]
                  for name, rgb in arms.items()}
    within = {name: column(values, "within_range_bin_luminance_std_median")
              for name, values in statistics.items()}
    r_squared = {name: column(values, "range_luminance_r_squared") for name, values in statistics.items()}
    ratios = [within["lambertian"][i] / within["depth_gradient"][i]
              if within["depth_gradient"][i] > 0 else float("inf") for i in range(count)]
    diagnostic_ratios = [within["lambertian_uniform_color"][i] / within["depth_gradient"][i]
                         if within["depth_gradient"][i] > 0 else float("inf") for i in range(count)]
    checks = {
        "A_flat_is_flat_in_every_scene":
            max(column(statistics["flat"], "luminance_variance")) <= FLAT_VARIANCE_MAX,
        "B_depth_arm_is_a_function_of_depth_in_every_scene":
            min(r_squared["depth_gradient"]) >= DEPTH_R2_MIN,
        "C_lambertian_is_not_depth_alone_in_every_scene":
            max(r_squared["lambertian"]) < LAMBERTIAN_R2_MAX,
        "D_within_depth_bin_spread_exceeds_depth_arm_in_every_scene":
            min(ratios) > WITHIN_BIN_RATIO_MIN,
    }
    return {
        "scenes": count,
        "aggregation_rule": "worst scene per criterion; every per-scene value is recorded",
        "preregistered_arms": list(PREREGISTERED_ARMS),
        "diagnostic_arms": [name for name in arms if name not in PREREGISTERED_ARMS],
        "fixture": {"coverage": coverage, "visible_faces": faces, "valid_pixels": pixels},
        "per_scene": statistics,
        "summary": {name: {"within_bin_std_median_over_scenes": float(np.median(within[name])),
                           "r_squared_min": min(r_squared[name]), "r_squared_max": max(r_squared[name]),
                           "luminance_mean_over_scenes": float(np.mean(column(values, "luminance_mean")))}
                    for name, values in statistics.items()},
        "within_bin_ratio_lambertian_over_depth": ratios,
        "within_bin_ratio_uniform_color_lambertian_over_depth": diagnostic_ratios,
        "thresholds": {"flat_variance_max": FLAT_VARIANCE_MAX, "depth_r_squared_min": DEPTH_R2_MIN,
                       "lambertian_r_squared_max": LAMBERTIAN_R2_MAX,
                       "within_bin_ratio_min": WITHIN_BIN_RATIO_MIN,
                       "range_quantile_bins": RANGE_QUANTILE_BINS},
        "checks": checks,
        "array_sha256": {**{f"{name}_rgb": tensor_hash(rgb) for name, rgb in arms.items()},
                         "range": tensor_hash(gbuffer.range_m), "normal": tensor_hash(gbuffer.normal_world),
                         "face": tensor_hash(gbuffer.face_id), "valid": tensor_hash(gbuffer.valid)},
        "run_verdict": "PASS" if all(checks.values()) else "FAIL",
    }
