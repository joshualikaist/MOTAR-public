#!/usr/bin/env python3
"""Render real URDF assets through the independent prototype and check what reached the image.

This is the URDF loader's technical smoke, not an experiment. It asserts that every link and
every material a URDF declares actually becomes visible pixels, which is the property V1 needs
before any asset can be swapped. It claims nothing about appearance quality or shortcut use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/renderer_urdf_loader_v1_contract.md"
ASSETS = {
    "interceptor": "resources/robots/quad/quad_navrl_ref5in_v2.urdf",
    "target": "resources/models/environment_assets/objects/navrl_target_drone_v2.urdf",
    "distractor_sphere": "resources/models/environment_assets/objects/navrl_distractor_sphere.urdf",
    "distractor_box": "resources/models/environment_assets/objects/navrl_distractor_box.urdf",
    "distractor_pole": "resources/models/environment_assets/objects/navrl_distractor_pole.urdf",
}
MIN_COVERAGE = 0.01
MIN_PIXELS_PER_LINK = 8
VIEWS = 4
SEED = 811


def isolation_guard(stage):
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError(f"URDF smoke isolation violation {stage}")


def source_bytes():
    paths = [Path(__file__).resolve(), CONTRACT, ROOT / "tools/runtime_fingerprint.py",
             ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"]
    paths += sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    return {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}


FRAME_FILL = 0.85


def framing_distance(asset, camera, rotation, fill=FRAME_FILL):
    """Fit the asset's projected extent to the frame, per view.

    Framing by bounding sphere wastes the frame for anything anisotropic: a 1.6 m pole 0.12 m
    across then occupies a strip a few pixels wide, and the coverage guard fires on the framing
    rather than on anything about the asset. Projecting onto the camera axes and fitting the
    larger of the two extents is correct for any aspect ratio and needs no per-asset tuning.
    """
    import numpy as np
    vertices = asset.mesh.vertices.astype(np.float64)
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(vertices - centre, axis=1).max())
    camera_frame = (vertices - centre) @ rotation          # rows of rotation^T are the camera axes
    half = np.abs(camera_frame).max(axis=0)
    tangent = np.array([camera.width, camera.height]) / (2.0 * camera.focal_px)
    distance = float(np.max(half[:2] / (fill * tangent)) + half[2])
    return centre, radius, distance


def evaluate(asset, camera, device, views=VIEWS):
    """Orbit the asset and record which links and materials produced pixels in each view."""
    import numpy as np
    import torch
    from renderer_validation.gbuffer import WarpGBufferRenderer
    from renderer_validation.scene import quaternion, quaternion_product
    from renderer_validation.shading import shade
    from renderer_validation.scene import Appearance

    positions, orientations, distances = [], [], []
    centre = radius = None
    for index in range(views):
        azimuth = 2.0 * np.pi * index / views
        elevation = np.radians(15.0 if index % 2 == 0 else -20.0)
        # Camera +Z forward: yaw about world Y then pitch about world X brings the asset into view.
        quaternion_xyzw = quaternion_product(quaternion([0, 1, 0], azimuth),
                                             quaternion([1, 0, 0], elevation))
        x, y, z, w = quaternion_xyzw
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        centre, radius, distance = framing_distance(asset, camera, rotation)
        distances.append(distance)
        positions.append(centre - rotation[:, 2] * distance)
        orientations.append(quaternion_xyzw)
    positions = np.asarray(positions, dtype=np.float32)
    orientations = np.asarray(orientations, dtype=np.float64)
    orientations = (orientations / np.linalg.norm(orientations, axis=1, keepdims=True)).astype(np.float32)

    renderer = WarpGBufferRenderer(asset.mesh, camera, views, device)
    renderer.set_camera_poses(positions, orientations)
    gbuffer = renderer.render()
    second = renderer.render()

    count = asset.mesh.material_count
    colours = np.repeat(asset.material_rgba[None, :, :3], views, axis=0).astype(np.float32)
    appearance = Appearance(colours, np.full((views, count), 0.8, dtype=np.float32),
                            np.tile([-0.6, -0.7, -1.0], (views, 1)),
                            np.full(views, 0.25, dtype=np.float32),
                            np.full(views, 0.75, dtype=np.float32))
    rgb = shade(gbuffer, asset.mesh, appearance, "lambertian")

    table = torch.tensor(asset.mesh.face_material.copy(), device=gbuffer.face_id.device, dtype=torch.long)
    per_pixel_material = table[gbuffer.face_id.clamp_min(0).long()]
    per_view, link_totals, material_totals = [], {}, {}
    for index in range(views):
        valid = gbuffer.valid[index]
        links = {asset.link_names[n]: int((valid & (gbuffer.instance_id[index] == n)).sum())
                 for n in range(len(asset.link_names))}
        materials = {asset.material_names[m]: int((valid & (per_pixel_material[index] == m)).sum())
                     for m in range(count)}
        for name, value in links.items():
            link_totals[name] = link_totals.get(name, 0) + value
        for name, value in materials.items():
            material_totals[name] = material_totals.get(name, 0) + value
        per_view.append({"coverage": float(int(valid.sum()) / valid.numel()),
                         "valid_pixels": int(valid.sum()),
                         "visible_triangles": int(torch.unique(gbuffer.face_id[index][valid]).numel()),
                         "link_pixels": links, "material_pixels": materials,
                         "range_min_m": float(gbuffer.range_m[index][valid].min()) if valid.any() else None,
                         "range_max_m": float(gbuffer.range_m[index][valid].max()) if valid.any() else None})
    checks = {
        "geometry_rerender_exact": all(torch.equal(getattr(gbuffer, f), getattr(second, f)) for f in
                                       ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")),
        "every_view_has_coverage": all(v["coverage"] >= MIN_COVERAGE for v in per_view),
        "every_link_reaches_the_image": min(link_totals.values()) >= MIN_PIXELS_PER_LINK,
        "every_material_reaches_the_image": min(material_totals.values()) >= MIN_PIXELS_PER_LINK,
        "rgb_contract": bool(rgb.shape == gbuffer.valid.shape + (3,) and torch.isfinite(rgb).all()
                             and ((rgb >= 0) & (rgb <= 1)).all()),
    }
    from renderer_validation.validation import tensor_hash
    return {"asset": asset.as_dict(), "framing": {"bounding_radius_m": radius,
                                                  "camera_distance_m": distances,
                                                  "frame_fill": FRAME_FILL,
                                                  "centre_m": centre.tolist()},
            "per_view": per_view, "link_pixels_total": link_totals,
            "material_pixels_total": material_totals, "checks": checks,
            "status": "TECHNICAL_PASS" if all(checks.values()) else "TECHNICAL_FAIL",
            "array_sha256": {name: tensor_hash(getattr(gbuffer, name)) for name in
                             ("range_m", "normal_world", "face_id", "instance_id", "valid")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=135)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("A new output directory is required")
    isolation_guard("before rendering")
    before = source_bytes()
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    clean = all(subprocess.run(["git", "-C", str(ROOT), "diff"] + flags + ["--quiet"]).returncode == 0
                for flags in ([], ["--cached"]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from renderer_validation.scene import Camera
    from renderer_validation.urdf_asset import (load_urdf_asset, convex_outward_violations,
                                                orientation_report)
    from runtime_fingerprint import runtime_fingerprint

    camera = Camera(width=args.width, height=args.height)
    results = {}
    for name, relative in ASSETS.items():
        asset = load_urdf_asset(ROOT / relative)
        record = evaluate(asset, camera, args.device)
        report = orientation_report(asset.mesh)
        record["orientation"] = report
        record["orientation"]["convex_centroid_violations"] = convex_outward_violations(asset.mesh)
        record["checks"]["closed_and_wound_outward"] = report["closed_and_consistently_wound_outward"]
        record["status"] = "TECHNICAL_PASS" if all(record["checks"].values()) else "TECHNICAL_FAIL"
        results[name] = record
        print(record["status"], name, "| triangles", record["asset"]["triangles"],
              "| links", len(record["asset"]["link_names"]), flush=True)
    isolation_guard("after rendering")
    if before != source_bytes():
        raise RuntimeError("Source changed during execution")

    runtime = runtime_fingerprint()
    if args.device.startswith("cuda"):
        runtime["nvidia_driver_version"] = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    status = "TECHNICAL_PASS" if all(r["status"] == "TECHNICAL_PASS" for r in results.values()) else "TECHNICAL_FAIL"
    payload = {"schema": "renderer_urdf_smoke_v1", "stage": "URDF_LOADER_V1", "status": status,
               "camera": {"width": camera.width, "height": camera.height,
                          "horizontal_fov_deg": camera.horizontal_fov_deg, "far_range_m": camera.far_range_m},
               "views": VIEWS, "seed": SEED, "device": args.device, "assets": results,
               "thresholds": {"min_coverage": MIN_COVERAGE, "min_pixels_per_link": MIN_PIXELS_PER_LINK},
               "runtime": runtime,
               "source": {"base_commit": commit, "tracked_tree_clean": clean,
                          "files": {name: hashlib.sha256(value).hexdigest() for name, value in before.items()}},
               "interpretation": "Loader technical smoke only. No appearance-quality, shortcut or "
                                 "training claim, and no asset has been swapped in the simulator."}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.json").open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(status, args.output)
    if status != "TECHNICAL_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
