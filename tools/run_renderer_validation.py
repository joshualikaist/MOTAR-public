#!/usr/bin/env python3
"""R2 frame producer only: independent box scene, no experiment verdict or learning.

--help is stdlib-only. Rendering happens only after explicit CLI invocation with --output.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-scenes", type=int, default=1)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--frames", type=int, default=2)
    return p


def check_budget(args):
    if args.seed < 0 or not 1 <= args.num_scenes <= 128 or not 1 <= args.frames <= 16:
        raise ValueError("Require seed >= 0, scenes in [1,128], frames in [1,16]")
    if not 1 <= args.width <= 2048 or not 1 <= args.height <= 2048:
        raise ValueError("Image dimensions must be in [1,2048]")
    # Both RGB modes + depth/range + normal + two IDs + valid mask = 53 bytes/pixel.
    size = args.frames * args.num_scenes * args.width * args.height * 53
    if size > 256 * 1024 ** 2:
        raise ValueError("R2 output exceeds 256 MiB raw-data budget; reduce frames/batch/resolution")
    return size


def git_value(*args):
    return subprocess.check_output(["git", "-C", str(ROOT)] + list(args),
                                   text=True, timeout=15).strip()


def source_record():
    paths = sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "tools/runtime_fingerprint.py",
              ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"]
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty_before_output": bool(git_value("status", "--porcelain", "--untracked-files=all")),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
    }


def write_json(path, value):
    # 'x' never replaces a historical receipt, including a failed run.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def export_frame(path, gbuffer, flat, lambertian):
    import numpy as np
    arrays = {"rgb_flat": flat, "rgb_lambertian": lambertian,
              "depth_m": gbuffer.depth_m, "range_m": gbuffer.range_m,
              "normal_world": gbuffer.normal_world, "face_id": gbuffer.face_id,
              "instance_id": gbuffer.instance_id, "valid": gbuffer.valid}
    arrays = {key: value.detach().cpu().numpy() for key, value in arrays.items()}
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "file": Path(path).name,
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                   for key, value in arrays.items()},
    }


def main(argv=None):
    args = parser().parse_args(argv)
    check_budget(args)
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new result directory")
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError("Run this standalone producer outside any aerial_gym process")
    source = source_record()
    # No heavy imports before argument validation. None of these imports loads aerial_gym.
    import numpy as np
    import torch
    from renderer_validation.scene import Camera, box_fixture, sample_appearance
    from renderer_validation.gbuffer import WarpGBufferRenderer, checked_kernel_source, KERNEL_SHA256
    from renderer_validation.shading import shade
    from runtime_fingerprint import runtime_fingerprint

    checked_kernel_source()
    scene = box_fixture()
    camera = Camera(width=args.width, height=args.height)
    appearance = sample_appearance(args.seed, args.num_scenes, scene.material_count)
    record = {
        "schema": "independent_renderer_r2_v1",
        "status": "INCOMPLETE",
        "source": source,
        "generic_kernel_sha256": KERNEL_SHA256,
        "seed": args.seed,
        "device": args.device,
        "num_scenes": args.num_scenes,
        "requested_frames": args.frames,
        "scene": scene.as_dict(),
        "scene_kind": "static_generic_box_fixture",
        "camera": asdict(camera),
        "camera_axes": "+X right, +Y down, +Z forward; world pose xyzw",
        "camera_positions": np.zeros((args.num_scenes, 3)).tolist(),
        "camera_quaternions_xyzw": np.tile([0, 0, 0, 1], (args.num_scenes, 1)).tolist(),
        "appearance": appearance.as_dict(),
        "appearance_sampling": "once_per_sequence; numpy SeedSequence([seed, scene_index])",
        "rgb_contract": "NHWC float32 linear RGB [0,1]; black background; no hidden channels",
        "light_direction_convention": "world surface-to-light",
        "depth_contract": "camera +Z metres; ray-distance far cutoff; misses zero with valid=false",
        "debug_contract": "normal_world, face_id, instance_id, valid are separate arrays; IDs miss=-1",
        "benchmark": "NOT_RUN",
        "experiment_verdict": "NOT_EVALUATED",
        "training": "NOT_SUPPORTED",
        "environment_step_latency": None,
        "files": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "request.json", record)
    try:
        renderer = WarpGBufferRenderer(scene, camera, args.num_scenes, args.device)
        record["runtime"] = runtime_fingerprint(include_device=args.device != "cpu")
        record["runtime"]["warp"] = renderer.wp.__version__
        record["runtime"]["torch_threads"] = torch.get_num_threads()
        record["runtime"]["warp_kernel_cache_dir"] = str(renderer.wp.config.kernel_cache_dir)
        if args.device != "cpu":
            record["runtime"]["cuda_device_index"] = 0
            record["runtime"]["cuda_device_name"] = torch.cuda.get_device_name(0)
            try:
                record["runtime"]["nvidia_smi_inventory"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                     "--format=csv,noheader"], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                record["runtime"]["nvidia_smi_inventory"] = None
        for index in range(args.frames):
            geometry = renderer.render()
            flat = shade(geometry, scene, appearance, "flat")
            shaded = shade(geometry, scene, appearance, "lambertian")
            record["files"].append(export_frame(args.output / ("frame_%04d.npz" % index),
                                                 geometry, flat, shaded))
        if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
            raise RuntimeError("Isolation violation: aerial_gym was imported")
        after = source_record()
        if after["source_sha256"] != source["source_sha256"] or after["git_commit"] != source["git_commit"]:
            raise RuntimeError("Source changed while producing frames; run is incomplete")
        record["status"] = "RENDERED_UNASSESSED"
        write_json(args.output / "receipt.json", record)
    except BaseException as exc:
        record["status"] = "FAILED_INCOMPLETE"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.output / "failure.json", record)
        raise
    print("R2 frames saved (no experiment verdict): %s" % args.output)


if __name__ == "__main__":
    main()
