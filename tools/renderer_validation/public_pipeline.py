"""Public static graphics pipeline; only procedural boxes and a generic background.

No UAV assets, learned models, trajectories, physics or operational object identities.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .scene import Appearance, Camera, box_fixture, sample_appearance
from .background_scene import background_fixture

ROOT = Path(__file__).resolve().parents[2]
GEOMETRIES = ("boxes", "background")


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def provenance():
    paths = sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    paths += [ROOT / "tools" / name for name in ("export_renderer_dataset.py",
              "benchmark_renderer_pipeline.py", "validate_renderer_export.py", "runtime_fingerprint.py")]
    paths += [ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}}


def verify_source(before):
    after = provenance()
    if before["commit"] != after["commit"] or before["source_sha256"] != after["source_sha256"]:
        raise RuntimeError("Source changed during execution")


def isolated():
    if any(m == "aerial_gym" or m.startswith("aerial_gym.") for m in sys.modules):
        raise RuntimeError("Public renderer must not import the simulator package")


def scene_for(name):
    if name == "boxes":
        return box_fixture()
    if name == "background":
        return background_fixture().mesh
    raise ValueError("Only procedural boxes/background are supported")


def separate_appearance(material_seed, light_seed, count, materials):
    mat = sample_appearance(material_seed, count, materials)
    light = sample_appearance(light_seed, count, materials)
    return Appearance(mat.base_color, mat.kd, light.light_direction, light.ambient, light.directional)


def budget(width, height, count, frames=1):
    Camera(width=width, height=height)
    if not 1 <= count <= 128 or not 1 <= frames <= 16:
        raise ValueError("Scene count [1,128], frames [1,16] required")
    size = width * height * count * frames * 41
    if size > 1024 ** 3:
        raise ValueError("Raw output budget exceeds 1 GiB")
    return size


class Pipeline:
    def __init__(self, geometry, width, height, count, device, material_seed, light_seed,
                 position=(0., 0., 0.), quaternion=(0., 0., 0., 1.), instance_offset=0):
        isolated()
        from dataclasses import replace
        from .gbuffer import WarpGBufferRenderer
        if not 0 <= instance_offset <= 1000000:
            raise ValueError("Debug instance offset outside [0,1000000]")
        budget(width, height, count)
        self.scene = scene_for(geometry)
        self.scene = replace(self.scene, face_instance=self.scene.face_instance + instance_offset)
        self.camera = Camera(width=width, height=height)
        self.appearance = separate_appearance(material_seed, light_seed, count, self.scene.material_count)
        self.renderer = WarpGBufferRenderer(self.scene, self.camera, count, device)
        self.renderer.set_camera_poses(np.tile(position, (count, 1)), np.tile(quaternion, (count, 1)))

    def frame(self):
        from .shading import shade
        g = self.renderer.render()
        rgb = shade(g, self.scene, self.appearance, background=(0.04, 0.04, 0.04))
        # CPU-owned image output is part of this pipeline; compression/disk writing is not.
        return {key: tensor.detach().cpu().numpy().copy() for key, tensor in {
            "rgb": rgb, "depth_m": g.depth_m, "range_m": g.range_m,
            "normal_world": g.normal_world, "face_id": g.face_id,
            "instance_id": g.instance_id, "valid": g.valid}.items()}

    def description(self):
        return {"scene": self.scene.as_dict(), "camera": asdict(self.camera),
                "appearance": self.appearance.as_dict(),
                "camera_axes": "+X right, +Y down, +Z forward; world pose quaternion xyzw",
                "rgb": "NHWC float32 linear RGB [0,1]", "depth": "camera +Z metres; misses zero",
                "labels": "Separate debug/evaluation arrays; never model inputs"}


def array_record(arrays):
    return {key: {"shape": list(a.shape), "dtype": str(a.dtype),
                  "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()} for key, a in arrays.items()}


def verify_export(folder):
    folder = Path(folder)
    record = json.loads((folder / "receipt.json").read_text())
    if record.get("status") != "EXPORTED_UNASSESSED" or not record.get("files"):
        raise ValueError("Export is incomplete")
    for frame in record["files"]:
        path = folder / frame["file"]
        if path.parent.resolve() != folder.resolve() or path.suffix != ".npz":
            raise ValueError("Invalid exported file path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError("File SHA mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if array_record({name: archive[name] for name in archive.files}) != frame["arrays"]:
                raise ValueError("Decoded arrays differ")
    return record
