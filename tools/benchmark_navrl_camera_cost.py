"""S0-a: measure render cost of the NavRL vision stack vs camera resolution and env count.

Engineering measurement only -- no research claim, no preregistration required
(docs/plans/perception_shape_temporal_redesign_2026-09-03.md section 3). One (resolution,
num_envs) cell per process because Isaac Gym cannot re-initialise in-process; the sweep driver
is tools/run_navrl_camera_cost_sweep.sh.

Reads the cell from the environment (NAVRL_CAMERA_WIDTH/HEIGHT, BENCH_NUM_ENVS), steps the task
with a constant command, and appends one JSON line to BENCH_OUT with steps/s and VRAM. The first
BENCH_WARMUP steps are excluded from timing (JIT, allocator growth, curriculum reset).
"""

import json
import os
import subprocess
import time


def _int_env(name, default):
    return int(os.environ.get(name, "").strip() or default)


def nvidia_smi_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            universal_newlines=True,
        )
        return int(out.splitlines()[0].strip())
    except Exception:
        return -1


def main():
    num_envs = _int_env("BENCH_NUM_ENVS", 64)
    steps = _int_env("BENCH_STEPS", 300)
    warmup = _int_env("BENCH_WARMUP", 50)
    out_path = os.environ.get("BENCH_OUT", "").strip() or "results/navrl_camera_cost/cells.jsonl"

    cam_w = _int_env("NAVRL_CAMERA_WIDTH", 160)
    cam_h = _int_env("NAVRL_CAMERA_HEIGHT", 90)

    # vision=0 cells are the deliberate baseline that isolates the camera share.
    vision_on = os.environ.get("NAVRL_VISION", "0").strip() == "1"
    if vision_on:
        assert os.environ.get("NAVRL_PERCEPTION") == "1", "set NAVRL_PERCEPTION=1"

    # isaacgym modules must be imported before torch (gymdeps enforces the order)
    from aerial_gym.registry.task_registry import task_registry
    import torch

    smi_before = nvidia_smi_used_mib()
    t_build0 = time.time()
    env = task_registry.make_task("navrl_task", headless=True, num_envs=num_envs)
    build_s = time.time() - t_build0
    env.reset()

    actions = torch.zeros(
        (env.task_config.num_envs, env.task_config.action_space_dim), device="cuda:0"
    )
    actions[:, 0] = 0.5  # constant forward command; content does not matter for render cost

    torch.cuda.reset_peak_memory_stats()
    start = None
    with torch.no_grad():
        for i in range(steps):
            if i == warmup:
                torch.cuda.synchronize()
                start = time.time()
            env.step(actions=actions)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    timed_steps = steps - warmup

    row = {
        "vision": "on" if vision_on else "off",
        "camera_width": cam_w,
        "camera_height": cam_h,
        "pixels": cam_w * cam_h,
        "num_envs": num_envs,
        "timed_steps": timed_steps,
        "steps_per_s": timed_steps / elapsed,
        "env_steps_per_s": timed_steps * num_envs / elapsed,
        "ms_per_step": 1000.0 * elapsed / timed_steps,
        "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        "smi_used_before_mib": smi_before,
        "smi_used_after_mib": nvidia_smi_used_mib(),
        "build_s": build_s,
        "detector_max_range": os.environ.get("NAVRL_DETECTOR_MAX_RANGE", "default"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print("[camera-cost] " + json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
