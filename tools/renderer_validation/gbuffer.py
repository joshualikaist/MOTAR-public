"""Reuse audited generic ray kernels without importing aerial_gym or any simulator task.

Importing this module does not import/initialize Warp. Constructing the renderer does.
Two ray passes use the SAME ray-distance cutoff; optical depth is derived afterwards.
"""
from dataclasses import dataclass
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from .scene import Camera, MeshScene

ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"
KERNEL_SHA256 = "e3f2080c715cd641c5ec156323dbdaa125b7f62c17b30ade14cbcc51777b65fc"


def checked_kernel_source(path=KERNEL_PATH):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Generic camera source changed since R1; re-audit before reuse")
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    if len(imports) != 1 or not isinstance(imports[0], ast.Import) or (
            [(alias.name, alias.asname) for alias in imports[0].names] != [("warp", "wp")]):
        raise RuntimeError("Unexpected dependency in generic kernel source")
    return source


def load_camera_kernels():
    source = checked_kernel_source()
    name = "_renderer_validation_camera_" + KERNEL_SHA256[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, KERNEL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # Warp resolves the function's source module.
        try:
            # Keep the real source filename for Warp codegen, without package init or pyc writes.
            exec(compile(source, str(KERNEL_PATH), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].DepthCameraWarpKernels


@dataclass(frozen=True)
class GBuffer:
    range_m: torch.Tensor
    depth_m: torch.Tensor
    normal_world: torch.Tensor
    face_id: torch.Tensor
    instance_id: torch.Tensor
    valid: torch.Tensor


def finalize_gbuffer(raw_range, raw_normal, raw_face, scene, camera):
    """Validate two-pass consistency; return owned tensors, independent of reusable buffers."""
    shape = raw_range.shape
    if len(shape) != 3 or tuple(shape[1:]) != (camera.height, camera.width):
        raise ValueError("range must be [N,H,W]")
    if raw_normal.shape != shape + (3,) or raw_face.shape != shape:
        raise ValueError("G-buffer shape mismatch")
    if raw_range.dtype != torch.float32 or raw_normal.dtype != torch.float32 or raw_face.dtype != torch.int32:
        raise ValueError("Expected float32 range/normal and int32 face index")
    if raw_face.device != raw_range.device or raw_normal.device != raw_range.device:
        raise ValueError("G-buffer device mismatch")
    if not torch.isfinite(raw_range).all() or not torch.isfinite(raw_normal).all():
        raise ValueError("Non-finite raw geometry buffer")
    if ((raw_face < -1) | (raw_face >= len(scene.triangles))).any():
        raise ValueError("Invalid face index")
    hit = raw_face >= 0
    range_hit = (raw_range > 0) & (raw_range <= camera.far_range_m)
    if not torch.equal(hit, range_hit) or ((~hit) & (raw_range != 1000.0)).any():
        raise RuntimeError("Normal/face and range ray passes disagree; no image is accepted")
    lengths = torch.linalg.vector_norm(raw_normal, dim=-1)
    if (hit & (lengths < 1e-8)).any():
        raise RuntimeError("Hit with zero normal")
    normal = torch.where(hit[..., None], raw_normal / lengths.clamp_min(1e-8)[..., None], 0.0)
    ranges = torch.where(hit, raw_range, 0.0)
    projection = torch.tensor(camera.optical_projection(), device=raw_range.device)
    table = torch.tensor(scene.face_instance.copy(), device=raw_face.device, dtype=torch.int32)
    instances = torch.where(hit, table[raw_face.clamp_min(0).long()], -1)
    return GBuffer(ranges, ranges * projection, normal, raw_face.clone(), instances, hit)


def validated_poses(positions, orientations, count):
    p, q = np.asarray(positions, dtype=np.float32), np.asarray(orientations, dtype=np.float32)
    if p.shape != (count, 3) or q.shape != (count, 4) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Expected finite [N,3] positions and [N,4] xyzw quaternions")
    if not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Camera quaternion must have unit length")
    return p.copy(), q.copy()


class WarpGBufferRenderer:
    """Static triangle scene, one camera per batch entry; no physics or graph optimization."""
    def __init__(self, scene: MeshScene, camera: Camera, num_scenes=1, device="cuda:0"):
        if type(num_scenes) is not int or not 1 <= num_scenes <= 128:
            raise ValueError("num_scenes must be an integer in [1,128]")
        self.scene, self.camera, self.num_scenes = scene, camera, num_scenes
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU or CUDA devices are supported")
        checked_kernel_source()  # Fail before Warp initialization on source drift.
        import warp as wp
        self.wp = wp
        wp.init()
        self.kernels = load_camera_kernels()
        self.warp_device = str(self.device)
        # Retain all backing arrays and meshes for the lifetime of Warp views.
        self.vertices = torch.tensor(scene.vertices.copy(), device=self.device)
        self.indices = torch.tensor(scene.triangles.copy().reshape(-1), device=self.device, dtype=torch.int32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.meshes = [wp.Mesh(points=wp.from_torch(self.vertices, dtype=wp.vec3),
                               indices=wp.from_torch(self.indices, dtype=wp.int32)) for _ in range(num_scenes)]
        self.mesh_ids = wp.array([mesh.id for mesh in self.meshes], dtype=wp.uint64, device=self.warp_device)
        self.positions = torch.zeros((num_scenes, 1, 3), dtype=torch.float32, device=self.device)
        self.orientations = torch.zeros((num_scenes, 1, 4), dtype=torch.float32, device=self.device)
        self.orientations[..., 3] = 1.0
        shape = (num_scenes, 1, camera.height, camera.width)
        self.raw_range = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.raw_normal = torch.empty(shape + (3,), dtype=torch.float32, device=self.device)
        self.raw_face = torch.empty(shape, dtype=torch.int32, device=self.device)
        self.wp_positions = wp.from_torch(self.positions, dtype=wp.vec3)
        self.wp_orientations = wp.from_torch(self.orientations, dtype=wp.quat)
        self.wp_range = wp.from_torch(self.raw_range, dtype=wp.float32)
        self.wp_normal = wp.from_torch(self.raw_normal, dtype=wp.vec3)
        self.wp_face = wp.from_torch(self.raw_face, dtype=wp.int32)
        self.k_inv = wp.mat44(*camera.inverse_intrinsics().reshape(-1).tolist())

    def set_camera_poses(self, positions, orientations):
        """Camera axes: +X right, +Y down, +Z forward; world pose quaternion xyzw."""
        p, q = validated_poses(positions, orientations, self.num_scenes)
        self.positions[:, 0].copy_(torch.tensor(p, device=self.device))
        self.orientations[:, 0].copy_(torch.tensor(q, device=self.device))

    @torch.no_grad()
    def render(self):
        wp, c = self.wp, self.camera
        self.raw_range.fill_(1000.0)
        self.raw_normal.zero_()
        self.raw_face.fill_(-1)
        # Conservative, explicit Torch/Warp stream boundary. Benchmark/graph work is R5.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        common = [self.mesh_ids, self.wp_positions, self.wp_orientations, self.k_inv, c.far_range_m]
        dim = (self.num_scenes, 1, c.width, c.height)
        wp.launch(self.kernels.draw_optimized_kernel_normal_faceID, dim=dim,
                  inputs=common + [self.wp_normal, self.wp_face, c.width // 2, c.height // 2, True],
                  device=self.warp_device)
        # calculate_depth=False is essential: both passes then have the same ray-distance cutoff.
        wp.launch(self.kernels.draw_optimized_kernel_depth_range, dim=dim,
                  inputs=common + [self.wp_range, c.width // 2, c.height // 2, False],
                  device=self.warp_device)
        wp.synchronize_device(self.warp_device)
        return finalize_gbuffer(self.raw_range[:, 0], self.raw_normal[:, 0], self.raw_face[:, 0],
                                self.scene, c)
