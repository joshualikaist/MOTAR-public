"""Preregistered R3 renderer-only checks; deterministic functions, no learned model."""
from dataclasses import replace
import hashlib

import numpy as np
import torch

from .scene import Appearance
from .shading import shade


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def geometry_equal(a, b):
    names = ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")
    return all(torch.equal(getattr(a, name), getattr(b, name)) for name in names)


def uniform_appearance(count, materials, light):
    return Appearance(
        np.full((count, materials, 3), 0.55, dtype=np.float32),
        np.full((count, materials), 0.8, dtype=np.float32),
        np.tile(np.asarray(light, dtype=np.float32), (count, 1)),
        np.full(count, 0.2, dtype=np.float32),
        np.full(count, 0.8, dtype=np.float32),
    )


def evaluate(scene, first, second):
    """Return fixed R3 metrics/checks. Thresholds live only in this function and preregistration."""
    if first.valid.shape[0] != 1 or second.valid.shape[0] != 1:
        raise ValueError("R3 formal fixture requires exactly one scene")
    valid = first.valid
    if not valid.any():
        raise RuntimeError("No visible geometry")
    base = uniform_appearance(1, scene.material_count, [-1.0, 0.0, -1.0])
    other_light = uniform_appearance(1, scene.material_count, [1.0, 0.0, -1.0])
    flat = shade(first, scene, base, "flat")
    shaded_a = shade(first, scene, base, "lambertian")
    shaded_b = shade(first, scene, other_light, "lambertian")
    # ITU-R BT.709 linear-light luminance weights. Uniform RGB makes this choice immaterial for A.
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=flat.device)
    flat_y = (flat * weights).sum(-1)[valid]
    shaded_y = (shaded_a * weights).sum(-1)[valid]
    light_mae = torch.abs(shaded_a - shaded_b)[valid].mean()

    material_color = base.base_color.copy()
    material_color[:, 0] = [0.8, 0.2, 0.2]
    changed_appearance = replace(base, base_color=material_color)
    material_rgb = shade(first, scene, changed_appearance, "flat")
    face_material = torch.tensor(scene.face_material.copy(), device=first.face_id.device)
    per_pixel_material = face_material[first.face_id.clamp_min(0).long()]
    selected = valid & (per_pixel_material == 0)
    other = valid & ~selected
    delta = torch.abs(material_rgb - flat).amax(-1)

    renumbered = replace(first, instance_id=torch.where(valid, first.instance_id + 100, -1))
    id_rgb = shade(renumbered, scene, base, "lambertian")
    visible_faces = int(torch.unique(first.face_id[valid]).numel())
    visible_instances = int(torch.unique(first.instance_id[valid]).numel())
    metrics = {
        "coverage": float(valid.float().mean()),
        "valid_pixels": int(valid.sum()),
        "visible_faces": visible_faces,
        "visible_instances": visible_instances,
        "flat_luminance_variance": float(flat_y.var(unbiased=False)),
        "shaded_luminance_variance": float(shaded_y.var(unbiased=False)),
        "light_change_rgb_mae": float(light_mae),
        "selected_material_pixels": int(selected.sum()),
        # Integer count, never mean(): see true_fraction in background_validation. This value is
        # compared to exactly 1.0 below, and on CUDA an all-true mean falls one ulp short for
        # about 13% of pixel counts. R3's 2,196 is not one of them, so this changes no result.
        "material_changed_fraction_selected": (int((delta[selected] > 0).sum()) / int(selected.sum())
                                               if selected.any() else 0.0),
        "material_max_change_outside_selected": float(delta[other].max()) if other.any() else 0.0,
        "depth_min_m": float(first.depth_m[valid].min()),
        "depth_max_m": float(first.depth_m[valid].max()),
    }
    checks = {
        "fixture_coverage_at_least_0p05": metrics["coverage"] >= 0.05,
        "fixture_has_two_instances": visible_instances == 2,
        "fixture_has_four_faces": visible_faces >= 4,
        "flat_variance_at_most_1e_12": metrics["flat_luminance_variance"] <= 1e-12,
        "shaded_variance_at_least_1e_4": metrics["shaded_luminance_variance"] >= 1e-4,
        "light_mae_at_least_0p02": metrics["light_change_rgb_mae"] >= 0.02,
        "selected_material_has_50_pixels": metrics["selected_material_pixels"] >= 50,
        "selected_material_all_changed": metrics["material_changed_fraction_selected"] == 1.0,
        "unselected_material_exactly_unchanged": metrics["material_max_change_outside_selected"] == 0.0,
        "rerender_geometry_exact": geometry_equal(first, second),
        "instance_renumber_rgb_exact": torch.equal(shaded_a, id_rgb),
        "rgb_exactly_three_channels": all(x.ndim == 4 and x.shape[-1] == 3 for x in
                                          (flat, shaded_a, shaded_b, material_rgb, id_rgb)),
    }
    hashes = {name: tensor_hash(value) for name, value in {
        "range": first.range_m, "depth": first.depth_m, "normal": first.normal_world,
        "face": first.face_id, "instance": first.instance_id, "valid": first.valid,
        "flat_rgb": flat, "shaded_light_a_rgb": shaded_a, "shaded_light_b_rgb": shaded_b,
        "material_changed_rgb": material_rgb, "renumbered_instance_rgb": id_rgb,
    }.items()}
    return {"metrics": metrics, "checks": checks, "array_sha256": hashes,
            "run_verdict": "PASS" if all(checks.values()) else "FAIL"}
