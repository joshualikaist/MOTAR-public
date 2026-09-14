"""Fail-closed receipt comparison; no rendering, model, simulator or learning runs.

Malformed/missing evidence exits 2; valid unequal/failed runs exit 1. Default is
read-only. Hash equality verifies retained bytes, not authenticity or the original
scientific hypothesis. Historical R4/R4b verdicts remain FAIL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXACT = ("schema", "stage", "seed", "views", "camera", "fixture", "appearance",
         "camera_poses", "checks", "global_checks", "per_view", "array_sha256",
         "arrays_npz_sha256", "historical_R4_R4b_status", "status", "source", "runtime")
RUNTIME = ("python", "executable", "platform", "torch", "cuda", "cudnn", "numpy",
           "device_name", "device_capability", "warp", "matmul_allow_tf32",
           "cudnn_allow_tf32", "cudnn_benchmark", "deterministic_algorithms")
CHECKS = ("cycle_all_visible_ids_changed", "cycle_all_visible_rgb_changed", "lighting_sensitive",
          "material0_all_selected_changed", "material0_outside_exact", "sufficient_visibility")
GLOBAL_CHECKS = ("cycling_does_not_change_geometry", "geometry_rerender_exact",
                 "instance_debug_does_not_change_rgb", "rgb_contract")
ARRAYS = ("depth_gradient_rgb", "depth_m", "face_id", "flat_rgb", "instance_debug_renumbered_rgb",
          "instance_id", "lambertian_rgb", "lambertian_uniform_color_rgb", "light_x_reflected_rgb",
          "material0_half_albedo_rgb", "material_cycle_rgb", "normal_world", "range_m", "valid")
PER_VIEW = ("appearance_statistics", "cycle_id_changed_fraction", "cycle_rgb_changed_fraction",
            "historical_C_diagnostic_only", "instance_pixels", "light_rgb_mae",
            "material_changed_fraction", "selected_material_pixels")
STATS = ("black_pixel_fraction", "luminance_max", "luminance_mean", "luminance_min",
         "luminance_variance", "per_material", "range_luminance_r_squared",
         "saturated_pixel_fraction", "valid_pixels", "within_bin_std_over_mean_luminance",
         "within_range_bin_count", "within_range_bin_luminance_std_median")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, path):
    require(isinstance(value, dict), path + " must be an object")
    missing = set(names) - set(value)
    require(not missing, path + " missing: " + ", ".join(sorted(missing)))
    return value


def finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite JSON number")


def digest(value, length=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None,
            "Invalid digest: expected %d lower-case hexadecimal characters" % length)


def validate_receipt(record):
    fields(record, EXACT, "receipt")
    finite_json(record)
    require(record["schema"] == "renderer_background_validation_v1", "Unsupported schema")
    require(record["stage"] == "R4_BACKGROUND_V1", "Wrong stage")
    require(record["historical_R4_R4b_status"] == "FAIL_UNCHANGED", "Historical verdict changed")
    require(record["status"] in ("TECHNICAL_PASS", "TECHNICAL_FAIL", "INSUFFICIENT_VISIBILITY"),
            "Invalid technical status")
    for key in ("seed", "views"):
        require(type(record[key]) is int and record[key] >= (1 if key == "views" else 0),
                key + " must be a valid integer")
    contracts = {
        "camera": ("width", "height", "far_range_m", "horizontal_fov_deg"),
        "camera_poses": ("positions", "orientations"),
        "appearance": ("ambient", "base_color", "directional", "kd", "light_direction"),
        "fixture": ("face_part", "face_patch", "instance_names", "joint_angle_deg", "mesh", "part_names"),
    }
    for key, names in contracts.items():
        fields(record[key], names, key)
        require(all(record[key][name] is not None for name in names), key + " has null fields")
    for key in ("width", "height"):
        require(type(record["camera"][key]) is int and record["camera"][key] > 0,
                "Camera dimensions must be positive integers")
    for key in ("far_range_m", "horizontal_fov_deg"):
        require(type(record["camera"][key]) in (int, float) and record["camera"][key] > 0,
                "Camera scalar must be positive")
    for key in contracts["appearance"]:
        require(isinstance(record["appearance"][key], list) and bool(record["appearance"][key]),
                "Empty/malformed appearance: " + key)
    for key, width in (("positions", 3), ("orientations", 4)):
        rows = record["camera_poses"][key]
        require(isinstance(rows, list) and len(rows) == record["views"] and
                all(isinstance(row, list) and len(row) == width and
                    all(type(x) in (int, float) for x in row) for row in rows), "Invalid camera poses")
    fields(record["fixture"]["mesh"], ("vertices", "triangles", "face_material", "face_instance"), "mesh")
    for key in ("vertices", "triangles", "face_material", "face_instance"):
        require(isinstance(record["fixture"]["mesh"][key], list) and bool(record["fixture"]["mesh"][key]),
                "Empty/malformed mesh: " + key)
    fields(record["runtime"], RUNTIME, "runtime")
    for key in RUNTIME:
        require(record["runtime"][key] is not None, "Null runtime field: " + key)
    for key in RUNTIME[:5] + ("numpy", "device_name", "device_capability", "warp"):
        require(isinstance(record["runtime"][key], str) and bool(record["runtime"][key]),
                "Runtime value must be a nonempty string: " + key)
    for key in RUNTIME[-4:]:
        require(type(record["runtime"][key]) is bool, "Runtime flag must be boolean: " + key)
    fields(record["checks"], map(str, range(record["views"])), "checks")
    require(len(record["checks"]) == record["views"], "Check/view count mismatch")
    groups = [(record["global_checks"], GLOBAL_CHECKS)]
    groups += [(item, CHECKS) for item in record["checks"].values()]
    for checks, names in groups:
        fields(checks, names, "checks")
        require(all(type(v) is bool for v in checks.values()), "Checks must be booleans")
    views = record["per_view"]
    require(isinstance(views, list) and len(views) == record["views"], "per_view count mismatch")
    for view in views:
        fields(view, PER_VIEW, "per_view")
        stats = fields(view["appearance_statistics"],
                       ("depth_gradient", "flat", "lambertian", "lambertian_uniform_color"), "statistics")
        for values in stats.values():
            fields(values, STATS, "appearance statistics")
            for key in STATS:
                if key != "per_material":
                    require(type(values[key]) in (int, float), "Malformed statistic: " + key)
            materials = values["per_material"]
            require(isinstance(materials, list) and bool(materials), "Missing material statistics")
            for material in materials:
                fields(material, ("material", "pixels", "luminance_mean", "luminance_variance"), "material")
                require(all(type(material[k]) in (int, float) for k in
                            ("material", "pixels", "luminance_mean", "luminance_variance")),
                        "Malformed material statistic")
        require(isinstance(view["instance_pixels"], list) and bool(view["instance_pixels"]),
                "Missing instance pixel counts")
        require(all(type(x) is int and x >= 0 for x in view["instance_pixels"]), "Invalid pixel counts")
        require(type(view["selected_material_pixels"]) is int and view["selected_material_pixels"] >= 0,
                "Invalid selected pixel count")
        for key in ("light_rgb_mae", "historical_C_diagnostic_only"):
            require(type(view[key]) in (int, float), "Malformed per-view metric: " + key)
        for key in ("cycle_id_changed_fraction", "cycle_rgb_changed_fraction", "material_changed_fraction"):
            require(type(view[key]) in (int, float) and 0 <= view[key] <= 1, "Invalid fraction")
    fields(record["array_sha256"], ARRAYS, "array_sha256")
    for value in record["array_sha256"].values():
        digest(value)
    digest(record["arrays_npz_sha256"])
    source = fields(record["source"],
                    ("base_commit", "files", "tracked_tree_clean", "source_unchanged_during_run"), "source")
    digest(source["base_commit"], 40)
    fields(source["files"], ("tools/run_renderer_background_validation.py",
                             "docs/renderer_background_v1_contract.md"), "source files")
    for path, info in source["files"].items():
        require(not Path(path).is_absolute() and ".." not in Path(path).parts, "Unsafe source path")
        fields(info, ("sha256", "matches_head", "tracked_at_head"), "source file")
        digest(info["sha256"])
        require(type(info["matches_head"]) is bool and type(info["tracked_at_head"]) is bool,
                "Source flags must be boolean")
    for key in ("tracked_tree_clean", "source_unchanged_during_run"):
        require(type(source[key]) is bool, "Source flag must be boolean")
    return record


def load_receipt(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return validate_receipt(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs))


def compare_receipts(a, b):
    for record in (a, b):
        validate_receipt(record)
    exact = {key: a[key] == b[key] for key in EXACT}
    checks_ok = all(all(v is True for v in c.values()) for r in (a, b)
                    for c in list(r["checks"].values()) + [r["global_checks"]])
    source_ok = all(r["source"]["tracked_tree_clean"] and r["source"]["source_unchanged_during_run"] and
                    all(x["matches_head"] and x["tracked_at_head"] for x in r["source"]["files"].values())
                    for r in (a, b))
    passed = all(exact.values()) and checks_ok and source_ok and a["status"] == b["status"] == "TECHNICAL_PASS"
    return {"schema": "renderer_background_comparison_v2", "exact_comparisons": exact,
            "checks_passed": checks_ok, "source_claims_clean": source_ok,
            "verdict": "PASS" if passed else "FAIL",
            "interpretation": "Technical reproducibility only; historical R4/R4b remain FAIL."}


def verify_evidence(path, record, repo=ROOT):
    """Check archive bytes, decoded array hashes, and source Git objects, not current files."""
    import numpy as np
    raw = Path(path).parent / "arrays.npz"
    require(raw.is_file(), "Missing raw arrays: " + str(raw))
    require(hashlib.sha256(raw.read_bytes()).hexdigest() == record["arrays_npz_sha256"],
            "Raw archive SHA mismatch")
    with np.load(raw, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(record["array_sha256"]), "Raw array keys mismatch")
        for key, expected in record["array_sha256"].items():
            require(hashlib.sha256(arrays[key].tobytes(order="C")).hexdigest() == expected,
                    "Decoded array SHA mismatch: " + key)
    for name, info in record["source"]["files"].items():
        obj = record["source"]["base_commit"] + ":" + name
        data = subprocess.run(["git", "-C", str(repo), "show", obj], capture_output=True, timeout=15)
        require(data.returncode == 0, "Missing source Git object: " + obj)
        require(hashlib.sha256(data.stdout).hexdigest() == info["sha256"], "Source SHA mismatch: " + name)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, help="Optional new directory; omitted means read-only")
    args = parser.parse_args(argv)
    try:
        require(len(args.run) == 2 and args.run[0].resolve() != args.run[1].resolve(),
                "Exactly two distinct receipts are required")
        require(args.output is None or not args.output.exists(), "Use a new comparison directory")
        records = [load_receipt(path) for path in args.run]
        result = compare_receipts(*records)
        result["runs"] = [str(path) for path in args.run]
        result["evidence_verified"] = [verify_evidence(path, record) for path, record in zip(args.run, records)]
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["verdict"] == "PASS" else 1
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, "INVALID_EVIDENCE: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
