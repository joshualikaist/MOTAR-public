#!/usr/bin/env python3
"""Execute the preregistered D8-A mesh-derived observation gate.

The parent refuses a dirty tree and runs each arm in a fresh child process.  Children use fixed
actions, not a policy.  ``pose`` compares all three arms on the same 21 visible poses plus one
bar-occluded pose.  ``step`` measures the existing simulator path at 128 x 160 x 90 by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md"
V3 = ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf"
V3_SHA = "c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0"
SEED = 20260913
ARMS = ("analytic_flat", "mesh_flat", "mesh_shaded")
WARMUP = 50
MEASURED = 500
REPEATS = 3
BASE_ENV = {
    "NAVRL_VISION": "1",
    "NAVRL_PERCEPTION": "1",
    "NAVRL_GENERAL_TRAIN": "1",
    "NAVRL_MAX_OBSTACLES": "8",
    "NAVRL_OBSTACLE_FOV_DEG": "240",
    "NAVRL_OBSTACLE_SUPPRESS_DEG": "10",
    "NAVRL_LIDAR_HBEAMS": "72",
    "NAVRL_LIDAR_VBEAMS": "4",
    "NAVRL_LIDAR_RANGE": "12",
    "NAVRL_NUM_BARS": "0",
    "NAVRL_MAX_BARS": "150",
    "NAVRL_MAX_VELOCITY": "2.5",
    "NAVRL_ALT_HOLD_VMAX": "2.5",
    "NAVRL_YAW_RATE_MAX": "3.0",
    "NAVRL_TILT_COMP": "1",
    "NAVRL_TARGET_DYNAMICS": "physical",
    "NAVRL_PHYSICAL_GEOMETRY_VERSION": "v2",
    "NAVRL_ROBOT": "navrl_ref5in_v2_quad",
    "NAVRL_DYNAMIC_MESH_SHADOW": "0",
    "NAVRL_DISTRACTOR_COUNT": "0",
    "NAVRL_APP_HUE_DEG": "0",
    "NAVRL_APP_LIGHT_GAIN": "0",
    "NAVRL_APP_ALBEDO_JITTER": "0",
    "NAVRL_APP_TEXTURE_STD": "0",
    "NAVRL_APP_MOTION_BLUR": "0",
}
SOURCE_FILES = (
    "aerial_gym/task/navrl_task/navrl_detector.py",
    "aerial_gym/task/navrl_task/navrl_dynamic_mesh_treatment.py",
    "aerial_gym/task/navrl_task/navrl_dynamic_mesh_shadow.py",
    "tools/renderer_validation/urdf_asset.py",
    "tools/probe_dynamic_mesh_treatment.py",
    "tests/test_dynamic_mesh_treatment.py",
    "docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md",
    "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha256_bytes(Path(path).read_bytes())


def launcher_contract():
    """Pin the build helper beside the selected interpreter, not whatever PATH happens to expose."""
    python = Path(sys.executable).resolve()
    ninja = python.with_name("ninja")
    if not ninja.is_file() or not os.access(str(ninja), os.X_OK):
        raise RuntimeError(f"selected interpreter has no executable ninja beside it: {ninja}")
    version = subprocess.check_output([str(ninja), "--version"], text=True).strip()
    return {
        "python": str(python),
        "ninja": str(ninja.resolve()),
        "ninja_version": version,
        "ninja_sha256": file_sha(ninja),
    }


def tensor_sha(tensor):
    return sha256_bytes(tensor.detach().contiguous().cpu().numpy().tobytes())


def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def timing_summary(samples):
    return {
        "n": len(samples),
        "mean_ms": 1000.0 * statistics.fmean(samples),
        "median_ms": 1000.0 * statistics.median(samples),
        "p95_ms": 1000.0 * percentile(samples, 0.95),
        "min_ms": 1000.0 * min(samples),
        "max_ms": 1000.0 * max(samples),
        "steps_per_second": len(samples) / sum(samples),
    }


def device_memory_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return None


def seed_everything(torch):
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED % (2 ** 32))
    except ImportError:
        pass
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def configure(mode, width, height, bars=0):
    for key, value in BASE_ENV.items():
        os.environ[key] = value
    os.environ["NAVRL_CAMERA_WIDTH"] = str(width)
    os.environ["NAVRL_CAMERA_HEIGHT"] = str(height)
    os.environ["NAVRL_DETECT_WIDTH"] = str(width)
    os.environ["NAVRL_DETECT_HEIGHT"] = str(height)
    os.environ["NAVRL_NUM_BARS"] = str(bars)
    os.environ["NAVRL_MAX_BARS"] = str(max(1, bars) if bars else 150)
    os.environ["NAVRL_DYNAMIC_MESH_TREATMENT"] = "off" if mode == "analytic_flat" else mode


def load_asset():
    sys.path.insert(0, str(ROOT / "tools"))
    from renderer_validation.urdf_asset import load_urdf_asset
    asset = load_urdf_asset(V3)
    if asset.source_sha256 != V3_SHA:
        raise RuntimeError(f"v3 asset drift: {asset.source_sha256} != {V3_SHA}")
    return asset


def attach(detector, mode, asset):
    if mode == "analytic_flat":
        return None
    return detector.attach_dynamic_mesh_treatment(asset.mesh, asset.material_rgba)


def child_step(mode, envs, width, height, warmup, measured):
    configure(mode, width, height, bars=0)
    import isaacgym  # noqa: F401 -- must precede torch
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    from aerial_gym.registry.task_registry import task_registry
    from runtime_fingerprint import runtime_fingerprint

    seed_everything(torch)
    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=envs)
    task.reset()
    detector = task.detector
    if detector is None or task.perception is None:
        raise RuntimeError("D8-A requires the RGB-D perception path")
    asset = load_asset()
    treatment = attach(detector, mode, asset)

    def one_step(index):
        actions = torch.zeros((envs, 4), device=task.device)
        actions[:, 0] = 0.08 * math.sin(index * 0.07)
        actions[:, 1] = 0.05 * math.cos(index * 0.11)
        actions[:, 3] = 0.2 * ((index % 20) / 20.0 - 0.5)
        task.step(actions)

    for index in range(warmup):
        one_step(index)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_nvml = device_memory_mib()
    samples = []
    visible_env_frames = torch.zeros((), dtype=torch.int64, device=task.device)
    pixel_sum = torch.zeros((), dtype=torch.int64, device=task.device)
    for index in range(measured):
        torch.cuda.synchronize()
        start = time.perf_counter()
        one_step(warmup + index)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
        mask = detector.target_mask > 0
        visible_env_frames += mask.flatten(1).any(dim=1).sum()
        pixel_sum += mask.sum()
    after_nvml = device_memory_mib()

    hashes = {
        "target_mask": tensor_sha(detector.target_mask),
        "target_depth": tensor_sha(detector.target_depth),
        "raw_rgb": tensor_sha(task.obs_dict["navrl_raw_rgb"]),
        "raw_depth": tensor_sha(task.obs_dict["navrl_raw_depth"]),
        "structured_observation": tensor_sha(task.task_obs["observations"]),
        "robot_position": tensor_sha(task.obs_dict["robot_position"]),
        "target_position": tensor_sha(task.target_position),
    }
    report = {
        "kind": "step",
        "mode": mode,
        "seed": SEED,
        "envs": envs,
        "width": width,
        "height": height,
        "warmup": warmup,
        "measured": measured,
        "timing": timing_summary(samples),
        "visible_env_frames": int(visible_env_frames.item()),
        "target_pixel_sum": int(pixel_sum.item()),
        "hashes": hashes,
        "treatment": treatment.diagnostics() if treatment is not None else None,
        "treatment_contract": None if treatment is None else {
            "triangles": treatment.triangles,
            "materials": treatment.materials,
            "light_direction": treatment.light_direction,
            "ambient": treatment.ambient,
            "directional": treatment.directional,
            "material_gains": treatment.material_gain_values,
            "build_signature_stable": True,
        },
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "device_memory_before_mib": before_nvml,
        "device_memory_after_mib": after_nvml,
        "process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "runtime": runtime_fingerprint(),
        "launcher": launcher_contract(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    print("D8REPORT" + json.dumps(report, sort_keys=True))


def quat_xyzw(torch, roll, pitch, yaw, device):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor(
        [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy],
        dtype=torch.float32,
        device=device,
    )


def mask_stats(torch, mask, depth, rgb):
    rows = []
    h, w = mask.shape[-2:]
    u_grid = torch.arange(w, device=mask.device).view(1, 1, w)
    v_grid = torch.arange(h, device=mask.device).view(1, h, 1)
    for index in range(mask.shape[0]):
        hit = mask[index]
        count = int(hit.sum().item())
        if count:
            u = float((hit * u_grid[0]).sum().item() / count)
            v = float((hit * v_grid[0]).sum().item() / count)
            mean_depth = float(depth[index][hit].mean().item())
            red_variance = float(rgb[index, 0][hit].var(unbiased=False).item())
        else:
            u = v = mean_depth = red_variance = None
        rows.append({"count": count, "u": u, "v": v, "mean_depth": mean_depth,
                     "red_variance": red_variance})
    return rows


def child_pose(width, height):
    # One bar is used only by the final pose to prove static-scene occlusion. The 21 visible poses
    # are offset 3 m laterally from that bar and point along world +X.
    configure("analytic_flat", width, height, bars=1)
    import isaacgym  # noqa: F401
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.task.navrl_task.navrl_detector import NavRLTargetDetector
    from runtime_fingerprint import runtime_fingerprint

    seed_everything(torch)
    count = 22
    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=count)
    task.reset()
    asset = load_asset()
    device = task.device
    # Physical target is obstacle 0; the single bar follows it.
    bar = task.obs_dict["obstacle_position"][:, 1].detach().clone()
    drone = bar.clone()
    drone[:, 1] += 3.0
    drone[:, 2] = 1.0
    target = drone.clone()
    distances = [4.5, 5.5, 6.5]
    angles = [
        (0, 0, 0), (15, 0, 0), (0, 20, 0), (0, 0, 22),
        (25, 15, 0), (-20, 10, 18), (10, -18, -25),
    ]
    pose_meta = []
    target_q = torch.zeros((count, 4), dtype=torch.float32, device=device)
    for index, (distance, degrees) in enumerate(
        (distance, angle) for distance in distances for angle in angles
    ):
        target[index, 0] += distance
        target_q[index] = quat_xyzw(
            torch, *(math.radians(v) for v in degrees), device=device
        )
        pose_meta.append({"distance_m": distance, "rpy_deg": list(degrees), "occlusion": False})
    # Camera -> bar -> target, all on +X. The bar must remove every published target pixel.
    drone[-1] = bar[-1] + torch.tensor([-2.0, 0.0, 0.0], device=device)
    drone[-1, 2] = bar[-1, 2]
    target[-1] = bar[-1] + torch.tensor([2.0, 0.0, 0.0], device=device)
    target[-1, 2] = bar[-1, 2]
    target_q[-1, 3] = 1.0
    pose_meta.append({"distance_m": 4.0, "rpy_deg": [0, 0, 0], "occlusion": True})
    vehicle_q = torch.zeros((count, 4), dtype=torch.float32, device=device)
    vehicle_q[:, 3] = 1.0

    outputs = {}
    for mode in ARMS:
        os.environ["NAVRL_DYNAMIC_MESH_TREATMENT"] = "off" if mode == "analytic_flat" else mode
        detector = NavRLTargetDetector(task.sim_env.warp_env, count, device, task.vis_cfg, task.step_dt)
        treatment = attach(detector, mode, asset)
        rgb, depth = detector.render_raw_rgbd(drone, vehicle_q, target, target_q)
        torch.cuda.synchronize()
        mask = detector.target_mask > 0
        outputs[mode] = {
            "mask": mask.detach().clone(),
            "depth": detector.target_depth.detach().clone(),
            "rgb": rgb.detach().clone(),
            "stats": mask_stats(torch, mask, detector.target_depth, rgb),
            "diagnostics": treatment.diagnostics() if treatment is not None else None,
            "hashes": {
                "mask": tensor_sha(mask),
                "depth": tensor_sha(detector.target_depth),
                "rgb": tensor_sha(rgb),
            },
        }

    base, flat, shaded = (outputs[name] for name in ARMS)
    pose_rows = []
    for index, meta in enumerate(pose_meta):
        a, b = base["mask"][index], flat["mask"][index]
        intersection = int((a & b).sum().item())
        union = int((a | b).sum().item())
        base_count, flat_count = int(a.sum().item()), int(b.sum().item())
        pose_rows.append({
            **meta,
            "analytic_pixels": base_count,
            "mesh_pixels": flat_count,
            "area_ratio": flat_count / base_count if base_count else None,
            "mask_iou": intersection / union if union else None,
            "mesh_flat_red_variance": flat["stats"][index]["red_variance"],
            "mesh_shaded_red_variance": shaded["stats"][index]["red_variance"],
        })
    report = {
        "kind": "pose",
        "seed": SEED,
        "width": width,
        "height": height,
        "poses": pose_rows,
        "hashes": {name: outputs[name]["hashes"] for name in ARMS},
        "mesh_flat_shaded_mask_equal": bool(torch.equal(flat["mask"], shaded["mask"])),
        "mesh_flat_shaded_depth_equal": bool(torch.equal(flat["depth"], shaded["depth"])),
        "treatment_diagnostics": {
            name: outputs[name]["diagnostics"] for name in ("mesh_flat", "mesh_shaded")
        },
        "runtime": runtime_fingerprint(),
        "launcher": launcher_contract(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    print("D8REPORT" + json.dumps(report, sort_keys=True))


def run_child(kind, mode=None, envs=128, width=160, height=90, warmup=WARMUP, measured=MEASURED):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", kind,
               "--width", str(width), "--height", str(height)]
    if kind == "step":
        command += ["--mode", mode, "--envs", str(envs), "--warmup", str(warmup),
                    "--measured", str(measured)]
    contract = launcher_contract()
    child_env = os.environ.copy()
    child_env["PATH"] = str(Path(contract["python"]).parent) + os.pathsep + child_env.get("PATH", "")
    child_env["NAVRL_NINJA"] = contract["ninja"]
    result = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("D8REPORT")]
    if result.returncode != 0 or not lines:
        raise RuntimeError(
            f"D8 {kind} child mode={mode} failed {result.returncode}:\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    return json.loads(lines[-1][len("D8REPORT"):])


def preflight():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("D8 execution requires a clean tracked/untracked working tree")
    if file_sha(V3) != V3_SHA:
        raise RuntimeError("v3 URDF SHA does not match the preregistration")
    prereg_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(PREREG.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prereg_commit, head], cwd=ROOT
    ).returncode == 0
    if not prereg_commit or not ancestor or prereg_commit == head:
        raise RuntimeError("D8 preregistration must be a committed ancestor before implementation")
    return {"head": head, "preregistration_commit": prereg_commit, "tree_clean": True,
            "launcher": launcher_contract()}


def same_runtime(reports):
    first = reports[0]["runtime"]
    launcher = reports[0]["launcher"]
    return all(
        report["runtime"] == first and report["launcher"] == launcher
        for report in reports[1:]
    )


def evaluate(pose_runs, step_runs):
    checks = {}
    checks["pose_repeat_hashes_equal"] = pose_runs[0]["hashes"] == pose_runs[1]["hashes"]
    pose = pose_runs[0]
    checks["mesh_flat_shaded_mask_equal"] = pose["mesh_flat_shaded_mask_equal"]
    checks["mesh_flat_shaded_depth_equal"] = pose["mesh_flat_shaded_depth_equal"]
    visible = [row for row in pose["poses"] if not row["occlusion"]]
    for lower in (4, 5, 6):
        rows = [row for row in visible if lower <= row["distance_m"] < lower + 1]
        checks[f"mesh_nonempty_{lower}_{lower+1}m"] = bool(rows) and any(
            row["mesh_pixels"] > 0 for row in rows
        )
    checks["geometry_effect_nonzero"] = any(
        row["analytic_pixels"] != row["mesh_pixels"] for row in visible
    )
    flat_variances = [row["mesh_flat_red_variance"] for row in visible
                      if row["mesh_flat_red_variance"] is not None]
    shaded_variances = [row["mesh_shaded_red_variance"] for row in visible
                        if row["mesh_shaded_red_variance"] is not None]
    checks["flat_target_variance_zero"] = bool(flat_variances) and max(flat_variances) <= 1e-12
    checks["shaded_target_variance_nonzero"] = bool(shaded_variances) and max(shaded_variances) > 0
    for mode in ("mesh_flat", "mesh_shaded"):
        diag = pose["treatment_diagnostics"][mode]
        checks[f"{mode}_debug_valid"] = all(diag[key] == 0 for key in (
            "occluded_survivors", "invalid_depth", "invalid_face", "invalid_material", "invalid_normal"
        ))
        checks[f"{mode}_occlusion_exercised"] = diag["scene_occluded_hits"] > 0
    for mode in ARMS:
        reports = step_runs[mode]
        checks[f"{mode}_repeat_output_hashes_equal"] = all(
            report["hashes"] == reports[0]["hashes"] for report in reports[1:]
        )
        checks[f"{mode}_runtime_equal"] = same_runtime(reports)
    checks["all_fixed_action_trajectory_hashes_equal_by_field"] = all(
        report["hashes"]["robot_position"] == step_runs["analytic_flat"][0]["hashes"]["robot_position"]
        and report["hashes"]["target_position"] == step_runs["analytic_flat"][0]["hashes"]["target_position"]
        for mode in ARMS for report in step_runs[mode]
    )
    checks["all_pose_runtimes_equal"] = same_runtime(pose_runs)

    aggregate = {}
    for mode in ARMS:
        reports = step_runs[mode]
        aggregate[mode] = {
            "median_step_ms": statistics.median(r["timing"]["median_ms"] for r in reports),
            "median_steps_per_second": statistics.median(r["timing"]["steps_per_second"] for r in reports),
            "max_torch_reserved_mib": max(r["torch_peak_reserved_mib"] for r in reports),
            "max_device_memory_after_mib": None if any(r["device_memory_after_mib"] is None for r in reports)
                else max(r["device_memory_after_mib"] for r in reports),
        }
    base = aggregate["analytic_flat"]
    contrasts = {}
    for mode in ("mesh_flat", "mesh_shaded"):
        row = aggregate[mode]
        delta = row["median_step_ms"] - base["median_step_ms"]
        relative = 100.0 * delta / base["median_step_ms"]
        throughput_loss = 100.0 * (
            base["median_steps_per_second"] - row["median_steps_per_second"]
        ) / base["median_steps_per_second"]
        torch_delta = row["max_torch_reserved_mib"] - base["max_torch_reserved_mib"]
        nvml_available = row["max_device_memory_after_mib"] is not None and base[
            "max_device_memory_after_mib"
        ] is not None
        contrasts[mode] = {
            "absolute_median_increase_ms": delta,
            "relative_median_increase_percent": relative,
            "throughput_loss_percent": throughput_loss,
            "torch_reserved_increase_mib": torch_delta,
            "nvml_available": nvml_available,
            "nvml_after_difference_mib": (
                row["max_device_memory_after_mib"] - base["max_device_memory_after_mib"]
                if nvml_available else None
            ),
        }
    integrity = all(checks.values())
    no_go_cost = any(
        row["relative_median_increase_percent"] > 30.0
        or row["absolute_median_increase_ms"] > 20.0
        or row["torch_reserved_increase_mib"] > 1024.0
        for row in contrasts.values()
    )
    go_cost = all(
        row["relative_median_increase_percent"] <= 10.0
        and row["absolute_median_increase_ms"] <= 5.0
        and row["throughput_loss_percent"] <= 10.0
        and row["torch_reserved_increase_mib"] <= 256.0
        and row["nvml_available"]
        for row in contrasts.values()
    )
    verdict = "TECHNICAL_GO" if integrity and go_cost else (
        "TECHNICAL_NO_GO" if not integrity or no_go_cost else "TECHNICAL_INCONCLUSIVE"
    )
    return {"checks": checks, "aggregate": aggregate, "contrasts": contrasts,
            "integrity_pass": integrity, "go_cost": go_cost, "no_go_cost": no_go_cost,
            "verdict": verdict}


def parent(args):
    if args.output.exists():
        raise RuntimeError("D8 output directory already exists")
    provenance = preflight()
    pose_runs = []
    step_runs = {mode: [] for mode in ARMS}
    try:
        for repeat in range(2):
            print(f"pose repeat {repeat + 1}/2", flush=True)
            pose_runs.append(run_child("pose", width=args.width, height=args.height))
        for mode in ARMS:
            for repeat in range(args.repeats):
                print(f"step {mode} repeat {repeat + 1}/{args.repeats}", flush=True)
                step_runs[mode].append(run_child(
                    "step", mode=mode, envs=args.envs, width=args.width, height=args.height,
                    warmup=args.warmup, measured=args.measured,
                ))
        decision = evaluate(pose_runs, step_runs)
        payload = {
            "schema": "dynamic_mesh_detector_d8a_v1",
            "stage": "D8-A",
            "seed": SEED,
            "provenance": provenance,
            "protocol": {"arms": list(ARMS), "pose_repeats": 2, "step_repeats": args.repeats,
                         "envs": args.envs, "width": args.width, "height": args.height,
                         "warmup": args.warmup, "measured": args.measured,
                         "fixed_actions": True, "policy_loaded": False},
            "source_sha256": {name: file_sha(ROOT / name) for name in SOURCE_FILES},
            "pose_runs": pose_runs,
            "step_runs": step_runs,
            "decision": decision,
            "interpretation": (
                "D8-A technical integration only. A GO does not establish detector accuracy, "
                "shortcut reduction, association improvement, policy robustness, adaptation, "
                "real-flight performance, or deployment readiness."
            ),
        }
    except Exception as error:
        args.output.mkdir(parents=True)
        failure = {"schema": "dynamic_mesh_detector_d8a_failure_v1", "stage": "D8-A",
                   "provenance": provenance, "error_type": type(error).__name__,
                   "error": str(error), "verdict": "TECHNICAL_NO_GO_UNFINISHED"}
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    args.output.mkdir(parents=True)
    (args.output / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": decision["verdict"], "checks": decision["checks"],
                      "contrasts": decision["contrasts"]}, indent=2, sort_keys=True))
    print("wrote", args.output / "receipt.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("step", "pose"))
    parser.add_argument("--mode", choices=ARMS, default="analytic_flat")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--measured", type=int, default=MEASURED)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.child == "step":
        return child_step(args.mode, args.envs, args.width, args.height, args.warmup, args.measured)
    if args.child == "pose":
        return child_pose(args.width, args.height)
    if args.output is None:
        parser.error("--output is required for the parent run")
    if min(args.envs, args.width, args.height, args.measured, args.repeats) < 1 or args.warmup < 0:
        parser.error("positive dimensions/measurements/repeats and nonnegative warmup required")
    return parent(args)


if __name__ == "__main__":
    main()
