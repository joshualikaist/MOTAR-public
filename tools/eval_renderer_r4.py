#!/usr/bin/env python3
"""Run one preregistered independent R4 process; never trains or loads a model.

Compares three appearance models on ONE shared G-buffer: geometry is cast once, so a difference
between arms cannot be a difference in geometry. The depth arm is an independent implementation
of a depth-driven appearance, not a re-run of the NavRL renderer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = 8


def tracked_tree_clean():
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0 and subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0


def isolation_guard(stage):
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError(f"R4 isolation violation {stage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(409,), default=409)
    parser.add_argument("--fixture", choices=("box", "mixed_material_box"), default="box",
                        help="mixed_material_box is R4b: same geometry, materials on both boxes")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("R4 output must not exist")
    if not tracked_tree_clean():
        raise RuntimeError("R4 requires a committed, tracked-clean source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    from renderer_validation.scene import (Camera, box_fixture, mixed_material_box_fixture,
                                           sample_appearance, sample_camera_poses)
    from renderer_validation.gbuffer import WarpGBufferRenderer, KERNEL_SHA256
    from renderer_validation.r4_metrics import evaluate_r4
    from runtime_fingerprint import runtime_fingerprint
    isolation_guard("before rendering")
    fixtures = {"box": box_fixture, "mixed_material_box": mixed_material_box_fixture}
    scene, camera = fixtures[args.fixture](), Camera(width=480, height=270)
    appearance = sample_appearance(args.seed, SCENES, scene.material_count)
    positions, orientations = sample_camera_poses(args.seed, SCENES)
    renderer = WarpGBufferRenderer(scene, camera, SCENES, args.device)
    renderer.set_camera_poses(positions, orientations)
    result = evaluate_r4(scene, renderer.render(), appearance, camera)
    isolation_guard("after rendering")
    result.update({
        "schema": "independent_renderer_r4_run_v1",
        "fixture_name": args.fixture,
        "stage": "R4" if args.fixture == "box" else "R4b",
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "tracked_tree_clean": True,
        "seed": args.seed,
        "device": args.device,
        "camera": {"width": camera.width, "height": camera.height,
                   "horizontal_fov_deg": camera.horizontal_fov_deg, "far_range_m": camera.far_range_m},
        "camera_poses": {"positions": positions.tolist(), "orientations": orientations.tolist(),
                         "note": "pose variation per scene was chosen at implementation time; the "
                                 "preregistration fixed the fixture and camera, not the pose"},
        "generic_kernel_sha256": KERNEL_SHA256,
        "runtime": runtime_fingerprint(),
        "experiment_scope": "renderer-only; no task, detector, model, training or policy. Says nothing "
                            "about any model's shortcut dependence.",
    })
    result["runtime"]["warp"] = renderer.wp.__version__
    with (args.output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["run_verdict"], args.output)


if __name__ == "__main__":
    main()
