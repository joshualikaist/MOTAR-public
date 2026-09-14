"""R4 background-v1 engineering checks; historical R4/R4b verdicts are untouched."""
from dataclasses import replace
import numpy as np
import torch

from .appearance_models import render_arms
from .background_scene import cycle_materials
from .r4_metrics import scene_statistics
from .scene import frozen_array, sample_appearance
from .shading import shade
from .validation import geometry_equal, tensor_hash

MIN_INSTANCE_PIXELS = 32
MIN_MATERIAL_PIXELS = 64
RGB_CHANGE_EPS = 1e-7
LIGHT_MAE_MIN = 1e-6


def true_fraction(values):
    """Count booleans as integers, because torch.mean cannot be trusted to return exactly 1.

    On CUDA, mean() of an all-true mask divides by n instead of counting, and for about 13% of
    sizes the round trip lands one ulp low: float32 does it at n=7600, float64 at n=7898 and
    n=25815. Both of those counts occurred in the first background run and turned two checks that
    had actually passed into failures. CPU is exact for every size tested, so a CPU unit test
    cannot reproduce this; only the integer form is safe on both.
    """
    if values.dtype != torch.bool:
        raise ValueError("Boolean event indicators required")
    count = values.numel()
    return int(values.sum()) / count if count else None


def background_appearance(count):
    base = sample_appearance(619, count, 4)
    return replace(base, light_direction=np.tile([-.7,-.6,-1.], (count,1)))


def appearance_variants(appearance):
    """Validate edits while preserving unchanged float32 lighting bits.

    Appearance.__post_init__ normalizes again even for an albedo-only dataclass replace.
    Reuse the already validated immutable direction in that case. Reflection preserves its
    norm, so retain the reflected bits too: the intervention must change X and nothing else.
    """
    colors = appearance.base_color.copy()
    colors[:,0] *= .5
    direction = appearance.light_direction.copy()
    direction[:,0] *= -1
    changed_material = replace(appearance, base_color=colors)
    changed_light = replace(appearance, light_direction=direction)
    object.__setattr__(changed_material, "light_direction", appearance.light_direction)
    object.__setattr__(changed_light, "light_direction", frozen_array(direction, np.float32))
    return changed_material, changed_light


def interventions(gbuffer, fixture, appearance):
    scene = fixture.mesh
    changed_material, changed_light = appearance_variants(appearance)
    cycled = cycle_materials(scene)
    renumbered = replace(gbuffer, instance_id=torch.where(gbuffer.valid, gbuffer.instance_id+100, -1))
    return {"material0_half_albedo": shade(gbuffer, scene, changed_material),
            "material_cycle": shade(gbuffer, cycled, appearance),
            "light_x_reflected": shade(gbuffer, scene, changed_light),
            "instance_debug_renumbered": shade(renumbered, scene, appearance)}, cycled


def evaluate_background(fixture, camera, appearance, first, second):
    scene, valid = fixture.mesh, first.valid
    count = valid.shape[0]
    table = torch.tensor(scene.face_material.copy(), device=first.face_id.device, dtype=torch.long)
    materials = table[first.face_id.clamp_min(0).long()]
    arms = render_arms(first, scene, appearance, camera)
    changed, cycled = interventions(first, fixture, appearance)
    all_rgb = dict(arms, **changed)
    original = arms["lambertian"]
    stats, checks = [], {}
    cycle_table = torch.tensor(cycled.face_material.copy(), device=first.face_id.device, dtype=torch.long)
    cycle_ids = cycle_table[first.face_id.clamp_min(0).long()]
    visibility_ok = True
    for i in range(count):
        selected = valid[i] & (materials[i] == 0)
        instance_pixels = [int((valid[i] & (first.instance_id[i] == n)).sum()) for n in range(4)]
        selected_pixels = int(selected.sum())
        visible = min(instance_pixels) >= MIN_INSTANCE_PIXELS and selected_pixels >= MIN_MATERIAL_PIXELS
        visibility_ok = visibility_ok and visible
        delta = (changed["material0_half_albedo"][i] - original[i]).abs().amax(-1)
        cycle_delta = (changed["material_cycle"][i] - original[i]).abs().amax(-1)
        light_mae = float((changed["light_x_reflected"][i] - original[i])[valid[i]].abs().mean()) if valid[i].any() else None
        per_arm = {name: scene_statistics(rgb, first, materials, i) for name, rgb in arms.items()} if int(valid[i].sum()) >= 40 else {}
        for s in per_arm.values():
            s["within_bin_std_over_mean_luminance"] = (
                s["within_range_bin_luminance_std_median"] / s["luminance_mean"] if s["luminance_mean"] > 0 else None)
        fraction = true_fraction(delta[selected] > RGB_CHANGE_EPS)
        cycle_fraction = true_fraction(cycle_delta[valid[i]] > RGB_CHANGE_EPS)
        id_fraction = true_fraction(cycle_ids[i][valid[i]] != materials[i][valid[i]])
        outside = bool(torch.equal(changed["material0_half_albedo"][i][~selected], original[i][~selected]))
        checks[str(i)] = {"sufficient_visibility": visible,
                          "material0_all_selected_changed": fraction == 1.,
                          "material0_outside_exact": outside,
                          "cycle_all_visible_ids_changed": id_fraction == 1.,
                          "cycle_all_visible_rgb_changed": cycle_fraction == 1.,
                          "lighting_sensitive": light_mae is not None and light_mae > LIGHT_MAE_MIN}
        stats.append({"instance_pixels": instance_pixels, "selected_material_pixels": selected_pixels,
                      "material_changed_fraction": fraction, "cycle_id_changed_fraction": id_fraction,
                      "cycle_rgb_changed_fraction": cycle_fraction, "light_rgb_mae": light_mae,
                      "appearance_statistics": per_arm,
                      "historical_C_diagnostic_only": per_arm.get("lambertian", {}).get("range_luminance_r_squared")})
    global_checks = {"geometry_rerender_exact": geometry_equal(first, second),
                     "instance_debug_does_not_change_rgb": torch.equal(original, changed["instance_debug_renumbered"]),
                     "rgb_contract": all(rgb.shape == valid.shape+(3,) and bool(torch.isfinite(rgb).all())
                                         and bool(((rgb >= 0) & (rgb <= 1)).all()) for rgb in all_rgb.values()),
                     "cycling_does_not_change_geometry": all(np.array_equal(getattr(scene, name), getattr(cycled, name))
                                                               for name in ("vertices", "triangles", "face_instance"))}
    passed = all(global_checks.values()) and all(all(v.values()) for v in checks.values())
    status = "TECHNICAL_PASS" if passed else ("INSUFFICIENT_VISIBILITY" if not visibility_ok else "TECHNICAL_FAIL")
    arrays = {name+"_rgb": value for name, value in all_rgb.items()}
    arrays.update({name: getattr(first, name) for name in ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")})
    return {"stage": "R4_BACKGROUND_V1", "status": status, "per_view": stats,
            "checks": checks, "global_checks": global_checks,
            "array_sha256": {name: tensor_hash(value) for name, value in arrays.items()},
            "historical_R4_R4b_status": "FAIL_UNCHANGED",
            "interpretation": "Engineering validation only; no claim about shortcut reduction or R4 C acceptance."}, arrays
