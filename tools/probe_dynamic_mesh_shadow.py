#!/usr/bin/env python3
"""D7: what the dynamic-mesh shadow query costs in the production render path, and that it changes nothing.

Gates and thresholds are fixed in
results/dynamic_mesh_integrated_cost_2026-09-12/PREREGISTRATION.md.

Shadow OFF and shadow ON are run as two child processes over an identical seed, env count,
resolution and pose sequence. Output invariance is checked by hashing the detector's own buffers;
if any hash moves, no cost verdict is issued.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = dict(
    NAVRL_VISION="1", NAVRL_PERCEPTION="1", NAVRL_GENERAL_TRAIN="1",
    NAVRL_MAX_OBSTACLES="8", NAVRL_OBSTACLE_FOV_DEG="240", NAVRL_OBSTACLE_SUPPRESS_DEG="10",
    NAVRL_LIDAR_HBEAMS="72", NAVRL_LIDAR_VBEAMS="4", NAVRL_LIDAR_RANGE="12",
    NAVRL_NUM_BARS="0", NAVRL_MAX_BARS="150", NAVRL_MAX_VELOCITY="2.5",
    NAVRL_ALT_HOLD_VMAX="2.5", NAVRL_YAW_RATE_MAX="3.0", NAVRL_TILT_COMP="1",
    NAVRL_TARGET_DYNAMICS="physical", NAVRL_PHYSICAL_GEOMETRY_VERSION="v2",
    NAVRL_ROBOT="navrl_ref5in_v2_quad",
)
WARMUP, MEASURED = 50, 500
SEED = 20260912


def device_memory_mib():
    """NVML device-level used memory, or UNAVAILABLE. Never 0 as a stand-in."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return None


def summarise(samples):
    ordered = sorted(samples)
    return {"n": len(ordered), "mean_ms": 1000 * statistics.fmean(ordered),
            "median_ms": 1000 * statistics.median(ordered),
            "p95_ms": 1000 * ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
            "min_ms": 1000 * ordered[0], "max_ms": 1000 * ordered[-1]}


def child(shadow, envs, width, height, warmup, measured, timed):
    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["NAVRL_CAMERA_WIDTH"] = str(width)
    os.environ["NAVRL_CAMERA_HEIGHT"] = str(height)
    os.environ["NAVRL_DYNAMIC_MESH_SHADOW"] = "1" if shadow else "0"
    import isaacgym  # noqa: F401  (must precede torch)
    import torch
    sys.path.insert(0, str(ROOT / "tools"))
    from aerial_gym.registry.task_registry import task_registry
    from renderer_validation.urdf_asset import load_urdf_asset

    torch.manual_seed(SEED)
    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=envs)
    task.reset()
    detector = task.detector
    if detector is None:
        raise RuntimeError("no detector; D7 measures the detector's render path")

    shadow_object = None
    if shadow:
        asset = load_urdf_asset(
            ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf")
        shadow_object = detector.attach_dynamic_mesh_shadow(asset.mesh)

    def one_step(step_index):
        # A fixed action sequence, derived from the step index alone, so both arms drive the
        # simulation identically. No RNG is consumed here.
        actions = torch.zeros((envs, 4), device=task.device)
        actions[:, 3] = 0.2 * ((step_index % 20) / 20.0 - 0.5)
        task.step(actions)

    for index in range(warmup):
        one_step(index)

    torch.cuda.reset_peak_memory_stats()
    samples = []
    for index in range(measured):
        if timed:
            torch.cuda.synchronize()
            start = time.perf_counter()
            one_step(warmup + index)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        else:
            one_step(warmup + index)

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    report = {
        "shadow": bool(shadow), "envs": envs, "width": width, "height": height,
        "timed": bool(timed), "warmup": warmup, "measured": measured,
        "step": summarise(samples) if samples else None,
        "steps_per_second": (measured / sum(samples)) if samples else None,
        "hashes": {
            "target_mask": digest(detector.target_mask),
            "target_depth": digest(detector.target_depth),
            "robot_position": digest(task.obs_dict["robot_position"]),
            "target_position": digest(task.target_position),
        },
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "device_memory_used_mib": device_memory_mib(),
        "shadow_triangles": shadow_object.triangles if shadow_object else None,
        "shadow_hit_pixels": shadow_object.hit_count() if shadow_object else None,
        "shadow_build_signature_stable": bool(shadow_object) or None,
    }
    print("D7REPORT" + json.dumps(report))


def run_child(shadow, envs, width, height, warmup, measured, timed):
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child",
         "--shadow" if shadow else "--no-shadow", "--envs", str(envs),
         "--width", str(width), "--height", str(height), "--warmup", str(warmup),
         "--measured", str(measured)] + ([] if timed else ["--untimed"]),
        capture_output=True, text=True, cwd=str(ROOT))
    line = [l for l in result.stdout.splitlines() if l.startswith("D7REPORT")]
    if result.returncode != 0 or not line:
        raise RuntimeError(f"child (shadow={shadow}) failed {result.returncode}:\n"
                           f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return json.loads(line[-1][len("D7REPORT"):])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--child", action="store_true")
    p.add_argument("--shadow", dest="shadow", action="store_true")
    p.add_argument("--no-shadow", dest="shadow", action="store_false")
    p.add_argument("--untimed", dest="timed", action="store_false")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=90)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--measured", type=int, default=MEASURED)
    p.add_argument("--cells", type=str, default="128x160x90,32x160x90,128x320x180,128x480x270")
    p.set_defaults(shadow=False, timed=True)
    args = p.parse_args()
    if args.child:
        return child(args.shadow, args.envs, args.width, args.height,
                     args.warmup, args.measured, args.timed)
    if args.output is None or args.output.exists():
        raise SystemExit("a new --output directory is required")

    cells, rows = [], []
    for spec in args.cells.split(","):
        envs, width, height = (int(v) for v in spec.split("x"))
        cells.append((envs, width, height))
    primary = cells[0]

    # Instrumentation overhead: the same arm with and without the timing calls.
    overhead = {}
    envs, width, height = primary
    timed_run = run_child(False, envs, width, height, 10, 60, True)
    untimed_start = time.perf_counter()
    run_child(False, envs, width, height, 10, 60, False)
    untimed_wall = time.perf_counter() - untimed_start
    overhead = {"timed_median_ms": timed_run["step"]["median_ms"],
                "untimed_wall_s_for_60_steps": untimed_wall,
                "note": "wall clock includes process start; used only to show the timing calls "
                        "are not the dominant term"}

    for envs, width, height in cells:
        print(f"  cell {envs} env {width}x{height} ...", flush=True)
        off = run_child(False, envs, width, height, args.warmup, args.measured, True)
        on = run_child(True, envs, width, height, args.warmup, args.measured, True)
        invariant = {k: off["hashes"][k] == on["hashes"][k] for k in off["hashes"]}
        rows.append({"envs": envs, "width": width, "height": height,
                     "baseline": off, "shadow": on, "output_invariant": invariant,
                     "all_hashes_equal": all(invariant.values())})
        d = on["step"]["median_ms"] - off["step"]["median_ms"]
        print(f"    baseline {off['step']['median_ms']:.2f} ms  shadow {on['step']['median_ms']:.2f} ms"
              f"  delta {d:+.2f} ms ({100*d/off['step']['median_ms']:+.1f}%)  invariant "
              f"{all(invariant.values())}", flush=True)

    payload = {
        "schema": "dynamic_mesh_integrated_cost_v1", "stage": "D7", "seed": SEED,
        "primary_cell": {"envs": primary[0], "width": primary[1], "height": primary[2]},
        "protocol": {"warmup": args.warmup, "measured": args.measured,
                     "paired": "same seed, env count, resolution and action sequence",
                     "rng": "shadow path consumes no random numbers; actions derive from the step index"},
        "instrumentation_overhead": overhead,
        "geometry_contract": {
            "status": "GEOMETRY_CONTRACT_DECISION_PENDING",
            "shadow_fixture": "navrl_target_drone_v3.urdf visual mesh, target-local",
            "historical_collision": "navrl_target_drone_v2.urdf collision box 0.283 x 0.283 x 0.12",
            "historical_sensor_proxy": "navrl_detector target_half_extents wp.vec3(0.14, 0.14, 0.06)",
            "note": "D7 unifies nothing; these are recorded, not reconciled."},
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "rows": rows,
        "interpretation": "Cost and non-interference only. No detector output, observation "
                          "contract, association, reward or policy was changed.",
    }
    args.output.mkdir(parents=True)
    (args.output / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output / "benchmark.json")


if __name__ == "__main__":
    main()
