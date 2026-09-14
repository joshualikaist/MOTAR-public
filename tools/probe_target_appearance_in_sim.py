#!/usr/bin/env python3
"""Does the v3 target silhouette survive the real simulator, and what does the detector see?

V1 was verified in the isolated renderer prototype. That says nothing about Isaac Gym, which has
to load thirteen <visual> elements in one link, nor about the detector's own Warp camera, which
renders at its own resolution through its own target-pixel path. This measures both, by building
the same task twice and comparing what the detector reports at matched target ranges.

Measurement only: no preregistered hypothesis, no threshold, no training, no policy update. It
reports pixel counts and refuses rather than guessing when a range bin is empty.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Must be set BEFORE any aerial_gym import: several modules read these at import time.
BASE_ENV = dict(
    NAVRL_VISION="1", NAVRL_PERCEPTION="1", NAVRL_GENERAL_TRAIN="1",
    NAVRL_MAX_OBSTACLES="8", NAVRL_OBSTACLE_FOV_DEG="240", NAVRL_OBSTACLE_SUPPRESS_DEG="10",
    NAVRL_LIDAR_HBEAMS="72", NAVRL_LIDAR_VBEAMS="4", NAVRL_LIDAR_RANGE="12",
    NAVRL_NUM_BARS="0", NAVRL_MAX_BARS="150", NAVRL_MAX_VELOCITY="2.5",
    NAVRL_ALT_HOLD_VMAX="2.5", NAVRL_YAW_RATE_MAX="3.0", NAVRL_TILT_COMP="1",
    NAVRL_TARGET_DYNAMICS="physical", NAVRL_PHYSICAL_GEOMETRY_VERSION="v2",
    # A physical ref5in target is only a valid same-platform experiment against a ref5in pursuer;
    # the task refuses the legacy 0.25 kg pursuer outright rather than mixing them.
    NAVRL_ROBOT="navrl_ref5in_v2_quad",
)
RANGE_EDGES = (2.0, 4.0, 6.0, 8.0, 12.0, 20.0)
MIN_SAMPLES_PER_BIN = 30


def child(appearance, envs, steps, seed):
    """One appearance per process: the asset is chosen at import time and cannot be swapped."""
    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["NAVRL_TARGET_APPEARANCE"] = appearance
    import isaacgym  # noqa: F401  (must precede torch)
    import torch
    from aerial_gym.registry.task_registry import task_registry

    task = task_registry.make_task("navrl_task", headless=True, use_warp=True, num_envs=envs)
    task.reset()
    torch.manual_seed(seed)
    detector = getattr(task, "detector", None)
    if detector is None:
        raise RuntimeError("the task exposes no detector; this probe measures the detector's view")
    ranges, pixels = [], []
    for _ in range(steps):
        # Zero actions left the target outside the camera in every sample of a first run: the
        # camera is forward-facing and nothing turned the vehicle toward the target. Command yaw
        # toward the target bearing so it enters the frame. Appearance does not affect physics, so
        # the two arms still follow the same trajectory and the comparison stays paired.
        actions = torch.zeros((envs, 4), device=task.device)
        offset = task.target_position - task.obs_dict["robot_position"]
        quaternion = task.obs_dict["robot_vehicle_orientation"]
        x, y, z, w = (quaternion[:, i] for i in range(4))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        bearing = torch.atan2(offset[:, 1], offset[:, 0]) - yaw
        bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
        actions[:, 3] = torch.clamp(bearing * 1.5, -1.0, 1.0)
        task.step(actions)
        # Measure the silhouette from target_mask, which is [envs, height, width] and is what the
        # renderer actually wrote. last_pixel_count was the obvious-looking field and is the wrong
        # one here: it stayed 0 for every sample of two full runs while target_mask held 48 to 55
        # pixels, because it is the count the detection path keeps after its own gating, not the
        # number of pixels the target occupies.
        mask = getattr(detector, "target_mask", None)
        if mask is None:
            raise RuntimeError("NavRLTargetDetector no longer exposes target_mask")
        count = (mask > 0).flatten(1).sum(dim=1)
        robot = task.obs_dict.get("robot_position")
        if robot is None or not hasattr(task, "target_position"):
            raise RuntimeError("cannot read robot_position or target_position from the task")
        distance = torch.linalg.vector_norm(task.target_position - robot, dim=1)
        ranges.extend(distance.detach().cpu().tolist())
        pixels.extend([float(v) for v in count.detach().cpu().tolist()])
    print(json.dumps({"appearance": appearance, "ranges": ranges, "pixels": pixels}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--child", choices=("v2", "v3"))
    args = parser.parse_args()
    if args.child:
        return child(args.child, args.envs, args.steps, args.seed)
    if args.output.exists():
        raise FileExistsError("a new output directory is required")
    samples = {}
    for appearance in ("v2", "v3"):
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child", appearance,
             "--output", str(args.output), "--envs", str(args.envs), "--steps", str(args.steps),
             "--seed", str(args.seed)],
            capture_output=True, text=True, cwd=str(ROOT))
        line = [l for l in result.stdout.splitlines() if l.startswith('{"appearance"')]
        if result.returncode != 0 or not line:
            raise RuntimeError(f"{appearance} child failed ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        samples[appearance] = json.loads(line[-1])
        print(f"  {appearance}: {len(samples[appearance]['pixels'])} samples", flush=True)
    args.output.mkdir(parents=True)
    (args.output / "raw.json").write_text(json.dumps(samples, indent=2))
    print("wrote", args.output / "raw.json")


if __name__ == "__main__":
    main()
