#!/usr/bin/env python3
"""Deterministic static graphics export.

Two paths, kept apart on purpose. Without `--arm` this is the generic box/background export it has
always been, and the public pipeline it uses still refuses to load any specimen asset. With `--arm`
it exports one renderer-characterization specimen over a preregistered view grid; that path is the
RC track's data production step, and it is the only path that reads an asset from `resources/`.
"""
import argparse
import hashlib
from pathlib import Path

ARMS = ("analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh")
# Arm-only options are stripped from a generic receipt, so a generic export made before the
# RC track existed still reproduces its receipt exactly, not merely its arrays.
ARM_ONLY = ("arm", "views", "shading", "scale", "colour", "kd", "light_mode",
            "light_direction", "ambient", "directional")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--geometry", choices=("boxes", "background"), default="boxes")
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=90)
    p.add_argument("--num-scenes", type=int, default=1)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--material-seed", type=int, default=0)
    p.add_argument("--light-seed", type=int, default=0)
    p.add_argument("--position", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--quaternion", type=float, nargs=4, default=[0, 0, 0, 1])
    p.add_argument("--instance-offset", type=int, default=0, help="Debug label renumbering only")
    # Renderer characterization arms (RC track). Absent --arm, every default below is inert and the
    # generic box/background export above is unchanged, byte for byte.
    p.add_argument("--arm", choices=ARMS, default=None,
                   help="Export one RC specimen over a view grid instead of the generic geometry")
    p.add_argument("--views", choices=("primary", "sweep", "fit"), default="primary")
    p.add_argument("--shading", choices=("flat", "lambertian"), default="lambertian")
    p.add_argument("--scale", type=float, default=None, help="Isotropic scale; area_matched_box only")
    p.add_argument("--colour", type=float, default=0.55, help="Grey material value in [0,1]")
    p.add_argument("--kd", type=float, default=0.8)
    p.add_argument("--light-mode", choices=("camera_relative", "world"), default="camera_relative")
    p.add_argument("--light-direction", type=float, nargs=3, default=[-1.0, -1.0, -1.0])
    p.add_argument("--ambient", type=float, default=0.2)
    p.add_argument("--directional", type=float, default=0.8)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Output must not exist")
    if args.arm is None:
        for name in ARM_ONLY[1:]:
            if getattr(args, name) != p.get_default(name):
                raise ValueError(f"--{name.replace('_', '-')} applies only together with --arm")
        return export_generic(args, p)
    return export_arm(args, p)


def export_generic(args, p):
    from renderer_validation.public_pipeline import (Pipeline, budget, provenance, verify_source,
                                                     write_json, array_record, isolated)
    from runtime_fingerprint import runtime_fingerprint
    import numpy as np
    budget(args.width, args.height, args.num_scenes, args.frames)
    if min(args.material_seed, args.light_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    source = provenance()
    config = {key: value for key, value in vars(args).items() if key not in ARM_ONLY}
    config["output"] = str(config["output"])
    record = {"schema": "generic_renderer_export_v1", "source": source, "config": config,
              "status": "INCOMPLETE", "experiment_verdict": "NOT_EVALUATED",
              "training": "NOT_SUPPORTED", "files": []}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "request.json", record)
    try:
        pipeline = Pipeline(args.geometry, args.width, args.height, args.num_scenes, args.device,
                            args.material_seed, args.light_seed, args.position, args.quaternion,
                            args.instance_offset)
        record["description"] = pipeline.description()
        record["runtime"] = runtime_fingerprint(include_device=args.device != "cpu")
        record["runtime"]["warp"] = pipeline.renderer.wp.__version__
        for i in range(args.frames):
            arrays = pipeline.frame()
            path = args.output / ("frame_%04d.npz" % i)
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            record["files"].append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "arrays": array_record(arrays)})
        isolated()
        verify_source(source)
        record["status"] = "EXPORTED_UNASSESSED"
        write_json(args.output / "receipt.json", record)
    except BaseException as exc:
        record.update(status="FAILED_INCOMPLETE", error_type=type(exc).__name__)
        write_json(args.output / "failure.json", record)
        raise
    print("EXPORTED_UNASSESSED " + str(args.output))


def arm_grid(name):
    from renderer_validation import view_grid
    return {"primary": view_grid.primary_grid, "sweep": view_grid.distance_sweep_grid,
            "fit": view_grid.fit_grid}[name]()


def export_arm(args, p):
    """Export one RC specimen: all views of one grid, one shading mode, in a single npz."""
    from renderer_validation.public_pipeline import (budget, provenance, verify_source, write_json,
                                                     array_record, isolated)
    from renderer_validation.characterization_pipeline import (CharacterizationCell,
                                                               camera_relative_light)
    from renderer_validation.scene import Camera
    from runtime_fingerprint import runtime_fingerprint
    import numpy as np
    if args.frames != 1:
        raise ValueError("An arm export is one static render of every view; --frames must be 1")
    if args.num_scenes != 1:
        raise ValueError("The view grid sets the batch size; --num-scenes does not apply to --arm")
    if list(args.position) != [0, 0, 0] or list(args.quaternion) != [0, 0, 0, 1]:
        raise ValueError("Arm camera poses come from the preregistered view grid, not --position")
    if (args.scale is None) != (args.arm != "area_matched_box"):
        raise ValueError("--scale is required by area_matched_box and rejected by every other arm")
    if not 0.0 <= args.colour <= 1.0 or not 0.0 <= args.kd <= 1.0:
        raise ValueError("--colour and --kd must lie in [0,1]")
    if args.ambient < 0.0 or args.directional < 0.0:
        raise ValueError("Lighting gains must be nonnegative")
    grid = arm_grid(args.views)
    budget(args.width, args.height, len(grid), args.frames)
    source = provenance()
    config = vars(args).copy()
    config["output"] = str(config["output"])
    record = {"schema": "renderer_characterization_export_v1", "source": source, "config": config,
              "status": "INCOMPLETE", "experiment_verdict": "NOT_EVALUATED",
              "training": "NOT_SUPPORTED", "files": []}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "request.json", record)
    try:
        camera = Camera(width=args.width, height=args.height, far_range_m=20.0)
        cell = CharacterizationCell(args.arm, camera, grid, device=args.device, scale=args.scale,
                                    instance_offset=args.instance_offset)
        light = (camera_relative_light(grid) if args.light_mode == "camera_relative"
                 else np.asarray(args.light_direction, dtype=np.float64))
        appearance = cell.appearance(colour=args.colour, kd=args.kd, light_direction=light,
                                     ambient=args.ambient, directional=args.directional)
        arrays = cell.arrays(appearance, args.shading)
        record["description"] = cell.description()
        record["description"]["appearance"] = appearance.as_dict()
        record["description"]["light_mode"] = args.light_mode
        record["runtime"] = runtime_fingerprint(include_device=args.device != "cpu")
        record["runtime"]["warp"] = (cell.renderer.wp.__version__ if cell.renderer is not None
                                     else "NOT_USED")
        path = args.output / "frame_0000.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record["files"].append({"file": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "arrays": array_record(arrays)})
        isolated()
        verify_source(source)
        record["status"] = "EXPORTED_UNASSESSED"
        write_json(args.output / "receipt.json", record)
    except BaseException as exc:
        record.update(status="FAILED_INCOMPLETE", error_type=type(exc).__name__)
        write_json(args.output / "failure.json", record)
        raise
    print("EXPORTED_UNASSESSED " + str(args.output))


if __name__ == "__main__":
    main()
