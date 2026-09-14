"""RC-R2 apparent-area matching: fit one isotropic scale on the fit views, then freeze it.

The scale is a single number fitted to the fit views only, and applied to the validation views
without modification. Re-fitting after looking at validation areas would make the match criterion
unfalsifiable, so this module cannot do it: the fit takes fit areas and the evaluation takes a scale
it is given, and the two never see each other's views.

Threshold from results/renderer_characterization_2026-09-13/PREREGISTRATION.md section 4.
"""
import numpy as np

AREA_MATCH_TOLERANCE = 0.05


def fit_isotropic_scale(reference_areas, base_areas):
    """s such that scaling the base specimen by s matches the reference mean silhouette area.

    Silhouette area grows as the square of an isotropic scale, so the fitted factor is the square
    root of the area ratio. This is the whole model; there is no free offset to absorb a mismatch.
    """
    reference = np.asarray(reference_areas, dtype=np.float64).reshape(-1)
    base = np.asarray(base_areas, dtype=np.float64).reshape(-1)
    if not reference.size or reference.shape != base.shape:
        raise ValueError("Fit needs one reference area per base area, and at least one view")
    if (reference <= 0).any() or (base <= 0).any():
        raise ValueError("A fit view has no silhouette; the scale is not identifiable there")
    scale = float(np.sqrt(reference.mean() / base.mean()))
    if not np.isfinite(scale) or not 0.01 <= scale <= 100.0:
        raise ValueError("Fitted scale is outside the range this study accepts")
    return scale


def evaluate_area_match(matched_areas, reference_areas, scale, tolerance=AREA_MATCH_TOLERANCE):
    """Did the frozen scale match the reference mean area on views it was not fitted on?"""
    matched = np.asarray(matched_areas, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_areas, dtype=np.float64).reshape(-1)
    if not matched.size or matched.shape != reference.shape:
        raise ValueError("One matched area per reference area is required")
    ratio = float(matched.mean() / reference.mean())
    return {"scale": float(scale), "mean_matched_area_px": float(matched.mean()),
            "mean_reference_area_px": float(reference.mean()), "area_ratio": ratio,
            "relative_error": abs(ratio - 1.0), "tolerance": float(tolerance),
            "verdict": "AREA_MATCHED" if abs(ratio - 1.0) <= float(tolerance) else "AREA_MATCH_FAILED"}


def area_ratios(areas_by_arm, reference_arm="quadrotor_mesh"):
    """Mean silhouette area of every arm, as itself and relative to the reference specimen."""
    if reference_arm not in areas_by_arm:
        raise KeyError(f"Reference arm {reference_arm!r} was not measured")
    reference = float(np.mean(areas_by_arm[reference_arm]))
    if reference <= 0.0:
        raise ValueError("Reference arm has no silhouette")
    return {arm: {"mean_area_px": float(np.mean(values)),
                  "ratio_to_reference": float(np.mean(values)) / reference,
                  "views": len(values)}
            for arm, values in sorted(areas_by_arm.items())}
