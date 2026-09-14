#!/usr/bin/env python3
"""Run one preregistered independent R3 process; never trains or loads a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def tracked_tree_clean():
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0 and subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(173,), default=173)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("R3 output must not exist")
    if not tracked_tree_clean():
        raise RuntimeError("R3 requires a committed, tracked-clean source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    from renderer_validation.scene import Camera, box_fixture
    from renderer_validation.gbuffer import WarpGBufferRenderer, KERNEL_SHA256
    from renderer_validation.validation import evaluate
    from runtime_fingerprint import runtime_fingerprint
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError("R3 isolation violation before rendering")
    scene, camera = box_fixture(), Camera(width=160, height=120)
    renderer = WarpGBufferRenderer(scene, camera, 1, args.device)
    first, second = renderer.render(), renderer.render()
    result = evaluate(scene, first, second)
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError("R3 isolation violation after rendering")
    result.update({
        "schema": "independent_renderer_r3_run_v1",
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "tracked_tree_clean": True,
        "seed": args.seed,
        "device": args.device,
        "camera": {"width": camera.width, "height": camera.height,
                   "horizontal_fov_deg": camera.horizontal_fov_deg, "far_range_m": camera.far_range_m},
        "generic_kernel_sha256": KERNEL_SHA256,
        "runtime": runtime_fingerprint(),
        "experiment_scope": "renderer-only; no task, detector, model, training or policy",
    })
    result["runtime"]["warp"] = renderer.wp.__version__
    result["runtime"]["warp_kernel_cache_dir"] = str(renderer.wp.config.kernel_cache_dir)
    with (args.output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["run_verdict"], args.output)


if __name__ == "__main__":
    main()
