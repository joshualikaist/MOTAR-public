"""D8 opt-in target observation rendered from a rigid target-local mesh.

The historical detector path remains an analytic oriented box.  This module is imported only
when a D8 treatment is explicitly attached.  It keeps the target mesh in local coordinates,
transforms camera rays into that frame, and checks the static scene separately for occlusion.
No BVH is rebuilt when the target moves.

The published outputs are the detector's existing target mask and target depth tensors.  Normal,
face, material, raw-hit and occlusion buffers are diagnostic-only and are never placed in the
task observation.  ``mesh_flat`` changes geometry only; ``mesh_shaded`` uses the same mask/depth
and adds deterministic within-object intensity variation to the RGB paint path.
"""

import os
from pathlib import Path

import numpy as np
import torch
import warp as wp


FLAG = "NAVRL_DYNAMIC_MESH_TREATMENT"
OFF = "off"
MESH_FLAT = "mesh_flat"
MESH_SHADED = "mesh_shaded"
MODES = (OFF, MESH_FLAT, MESH_SHADED)
DEFAULT_LIGHT_DIRECTION = (-0.45, -0.30, 0.84)
DEFAULT_AMBIENT = 0.45
DEFAULT_DIRECTIONAL = 0.55
V3_ASSET_RELATIVE = Path(
    "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf"
)
V3_ASSET_SHA256 = "c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0"


def treatment_mode():
    """Return the explicit D8 mode; unknown values fail closed."""
    raw = os.environ.get(FLAG, OFF).strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return OFF
    if raw in (MESH_FLAT, MESH_SHADED):
        return raw
    raise ValueError(
        f"{FLAG}={raw!r} is invalid; use off, {MESH_FLAT}, or {MESH_SHADED}."
    )


def material_gains(material_rgba):
    """Map URDF colours to a bounded relative luminance while retaining the nominal red hue."""
    rgba = np.asarray(material_rgba, dtype=np.float32)
    if rgba.ndim != 2 or rgba.shape[1] != 4 or len(rgba) < 1:
        raise ValueError("material_rgba must be a nonempty [M,4] array")
    if not np.isfinite(rgba).all() or (rgba < 0.0).any() or (rgba > 1.0).any():
        raise ValueError("material_rgba must be finite and lie in [0,1]")
    luma = rgba[:, :3] @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    span = float(luma.max() - luma.min())
    if span <= 1.0e-8:
        return np.ones(len(luma), dtype=np.float32)
    # The darkest declared material retains 55% of the nominal response; the lightest retains
    # 100%.  This isolates shading from a wholesale hue/class change.
    return np.ascontiguousarray(0.55 + 0.45 * (luma - luma.min()) / span, dtype=np.float32)


def load_pinned_v3_asset(repository_root):
    """Load the audited URDF only for an explicitly enabled D8 treatment.

    The parser is the renderer-validation loader already checked with two backends. D8-B's source
    bundle includes that loader and the object asset as additional runtime roots. Default/off
    never calls this function and therefore carries no urdfpy/trimesh dependency at runtime.
    """
    root = Path(repository_root).resolve()
    path = (root / V3_ASSET_RELATIVE).resolve()
    if root not in path.parents or not path.is_file():
        raise RuntimeError(f"pinned D8 v3 asset is missing or escapes the repository: {path}")
    from tools.renderer_validation.urdf_asset import load_urdf_asset
    asset = load_urdf_asset(path)
    if asset.source_sha256 != V3_ASSET_SHA256:
        raise RuntimeError(
            f"D8 v3 asset drift: {asset.source_sha256} != {V3_ASSET_SHA256}"
        )
    return asset


@wp.kernel
def render_dynamic_mesh_target_kernel(
    static_mesh_ids: wp.array(dtype=wp.uint64),
    target_mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    ray_vectors: wp.array2d(dtype=wp.vec3),
    target_positions: wp.array(dtype=wp.vec3),
    target_orientations: wp.array(dtype=wp.quat),
    face_material: wp.array(dtype=wp.int32),
    material_gain: wp.array(dtype=wp.float32),
    light_direction: wp.vec3,
    ambient: float,
    directional: float,
    far_plane: float,
    target_mask: wp.array(dtype=wp.int32, ndim=3),
    target_depth: wp.array(dtype=wp.float32, ndim=3),
    target_normal: wp.array(dtype=wp.vec3, ndim=3),
    target_face: wp.array(dtype=wp.int32, ndim=3),
    target_material: wp.array(dtype=wp.int32, ndim=3),
    target_shade: wp.array(dtype=wp.float32, ndim=3),
    raw_mesh_hit: wp.array(dtype=wp.int32, ndim=3),
    scene_occluded: wp.array(dtype=wp.int32, ndim=3),
):
    """Intersect a moving target mesh and publish it only when the static scene is farther."""
    env_id, row, col = wp.tid()
    ro = origins[env_id]
    rd = wp.normalize(wp.quat_rotate(orientations[env_id], ray_vectors[row, col]))

    target_mask[env_id, row, col] = wp.int32(0)
    target_depth[env_id, row, col] = far_plane
    target_normal[env_id, row, col] = wp.vec3(0.0, 0.0, 0.0)
    target_face[env_id, row, col] = wp.int32(-1)
    target_material[env_id, row, col] = wp.int32(-1)
    target_shade[env_id, row, col] = 0.0
    raw_mesh_hit[env_id, row, col] = wp.int32(0)
    scene_occluded[env_id, row, col] = wp.int32(0)

    rotation = target_orientations[env_id]
    local_origin = wp.quat_rotate_inv(rotation, ro - target_positions[env_id])
    local_direction = wp.quat_rotate_inv(rotation, rd)
    target_t = float(0.0)
    target_u = float(0.0)
    target_v = float(0.0)
    target_sign = float(0.0)
    local_normal = wp.vec3()
    face = int(0)
    hit = wp.mesh_query_ray(
        target_mesh,
        local_origin,
        local_direction,
        far_plane,
        target_t,
        target_u,
        target_v,
        target_sign,
        local_normal,
        face,
    )
    if hit:
        raw_mesh_hit[env_id, row, col] = wp.int32(1)

        scene_t = float(0.0)
        scene_u = float(0.0)
        scene_v = float(0.0)
        scene_sign = float(0.0)
        scene_normal = wp.vec3()
        scene_face = int(0)
        blocked = wp.mesh_query_ray(
            static_mesh_ids[env_id],
            ro,
            rd,
            target_t,
            scene_t,
            scene_u,
            scene_v,
            scene_sign,
            scene_normal,
            scene_face,
        )
        if blocked:
            scene_occluded[env_id, row, col] = wp.int32(1)
        else:
            world_normal = wp.normalize(wp.quat_rotate(rotation, local_normal))
            material = face_material[face]
            ndotl = wp.max(0.0, wp.dot(world_normal, light_direction))
            intensity = material_gain[material] * (ambient + directional * ndotl)
            intensity = wp.min(1.0, wp.max(0.0, intensity))
            target_mask[env_id, row, col] = wp.int32(1)
            target_depth[env_id, row, col] = target_t
            target_normal[env_id, row, col] = world_normal
            target_face[env_id, row, col] = wp.int32(face)
            target_material[env_id, row, col] = material
            target_shade[env_id, row, col] = intensity


class DynamicMeshTargetTreatment:
    """Own a target-local mesh and D8 diagnostic buffers; overwrite existing mask/depth."""

    def __init__(
        self,
        mesh_scene,
        material_rgba,
        mode,
        static_mesh_ids,
        target_mask,
        target_depth,
        device,
        far_plane,
        light_direction=DEFAULT_LIGHT_DIRECTION,
        ambient=DEFAULT_AMBIENT,
        directional=DEFAULT_DIRECTIONAL,
    ):
        if mode not in (MESH_FLAT, MESH_SHADED):
            raise ValueError(f"D8 treatment mode must be mesh_flat or mesh_shaded, got {mode!r}")
        if target_mask.ndim != 3 or tuple(target_depth.shape) != tuple(target_mask.shape):
            raise ValueError("target mask/depth must have the same [N,H,W] shape")
        if target_mask.dtype != torch.int32 or target_depth.dtype != torch.float32:
            raise ValueError("target mask/depth must be int32/float32")
        vertices = np.ascontiguousarray(mesh_scene.vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(mesh_scene.triangles, dtype=np.int32)
        face_material = np.ascontiguousarray(mesh_scene.face_material, dtype=np.int32)
        rgba = np.ascontiguousarray(material_rgba, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("mesh vertices must be finite [V,3]")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
            raise ValueError("mesh triangles must be nonempty [F,3]")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("mesh triangle index outside vertex array")
        if face_material.shape != (len(triangles),) or face_material.min() < 0:
            raise ValueError("one nonnegative material id per face is required")
        gains = material_gains(rgba)
        if face_material.max() >= len(gains):
            raise ValueError("mesh face references an unavailable material")
        light = np.asarray(light_direction, dtype=np.float32)
        if light.shape != (3,) or not np.isfinite(light).all() or np.linalg.norm(light) < 1e-8:
            raise ValueError("light_direction must be a finite nonzero 3-vector")
        light = light / np.linalg.norm(light)
        if not np.isfinite(far_plane) or float(far_plane) <= 0.0:
            raise ValueError("far_plane must be positive and finite")
        if min(float(ambient), float(directional)) < 0.0:
            raise ValueError("ambient and directional gains must be nonnegative")

        self.mode = mode
        self.device = str(device)
        self.far_plane = float(far_plane)
        self.num_envs, self.height, self.width = map(int, target_mask.shape)
        self.triangles = int(len(triangles))
        self.materials = int(len(gains))
        self.ambient = float(ambient)
        self.directional = float(directional)
        self.light_direction = tuple(float(v) for v in light)
        self.material_gain_values = tuple(float(v) for v in gains)
        self._static_mesh_ids = static_mesh_ids

        self._points = wp.array(vertices, dtype=wp.vec3, device=self.device)
        self._indices = wp.array(triangles.reshape(-1), dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self._points, indices=self._indices)
        self._face_material = wp.array(face_material, dtype=wp.int32, device=self.device)
        self._material_gain = wp.array(gains, dtype=wp.float32, device=self.device)

        self.target_mask = target_mask
        self.target_depth = target_depth
        shape = tuple(target_mask.shape)
        self.target_normal = torch.zeros(shape + (3,), dtype=torch.float32, device=device)
        self.target_face = torch.full(shape, -1, dtype=torch.int32, device=device)
        self.target_material = torch.full(shape, -1, dtype=torch.int32, device=device)
        self.target_shade = torch.zeros(shape, dtype=torch.float32, device=device)
        self.raw_mesh_hit = torch.zeros(shape, dtype=torch.int32, device=device)
        self.scene_occluded = torch.zeros(shape, dtype=torch.int32, device=device)

        self._target_mask_wp = wp.from_torch(target_mask, dtype=wp.int32)
        self._target_depth_wp = wp.from_torch(target_depth, dtype=wp.float32)
        self._target_normal_wp = wp.from_torch(self.target_normal, dtype=wp.vec3)
        self._target_face_wp = wp.from_torch(self.target_face, dtype=wp.int32)
        self._target_material_wp = wp.from_torch(self.target_material, dtype=wp.int32)
        self._target_shade_wp = wp.from_torch(self.target_shade, dtype=wp.float32)
        self._raw_mesh_hit_wp = wp.from_torch(self.raw_mesh_hit, dtype=wp.int32)
        self._scene_occluded_wp = wp.from_torch(self.scene_occluded, dtype=wp.int32)
        self._light = wp.vec3(*self.light_direction)
        self.build_signature = (
            self.mesh.id,
            self._points.ptr,
            self._indices.ptr,
            self._face_material.ptr,
            self._material_gain.ptr,
        )

    def run(self, origins, orientations, ray_vectors, target_positions, target_orientations):
        wp.launch(
            render_dynamic_mesh_target_kernel,
            dim=(self.num_envs, self.height, self.width),
            inputs=[
                self._static_mesh_ids,
                self.mesh.id,
                origins,
                orientations,
                ray_vectors,
                target_positions,
                target_orientations,
                self._face_material,
                self._material_gain,
                self._light,
                self.ambient,
                self.directional,
                self.far_plane,
                self._target_mask_wp,
                self._target_depth_wp,
                self._target_normal_wp,
                self._target_face_wp,
                self._target_material_wp,
                self._target_shade_wp,
                self._raw_mesh_hit_wp,
                self._scene_occluded_wp,
            ],
            device=self.device,
        )

    def target_rgb(self, nominal_rgb):
        """Return [N,3,H,W] paint; mask/depth are identical across the two mesh modes."""
        flat = nominal_rgb.view(self.num_envs, 3, 1, 1)
        if self.mode == MESH_FLAT:
            return flat
        return flat * self.target_shade.unsqueeze(1)

    def contract(self):
        """JSON-safe treatment identity without synchronising diagnostic tensors."""
        return {
            "mode": self.mode,
            "triangles": self.triangles,
            "materials": self.materials,
            "light_direction": list(self.light_direction),
            "ambient": self.ambient,
            "directional": self.directional,
            "material_gains": list(self.material_gain_values),
        }

    def diagnostics(self):
        """Synchronising post-run summary for receipts; never call on the hot path."""
        hit = self.target_mask > 0
        raw = self.raw_mesh_hit > 0
        occluded = self.scene_occluded > 0
        normals = torch.linalg.vector_norm(self.target_normal, dim=-1)
        visible_depth = self.target_depth[hit]
        visible_face = self.target_face[hit]
        visible_material = self.target_material[hit]
        invalid_depth = (~torch.isfinite(visible_depth)) | (visible_depth <= 0.0) | (
            visible_depth >= self.far_plane
        )
        invalid_face = (visible_face < 0) | (visible_face >= self.triangles)
        invalid_material = (visible_material < 0) | (visible_material >= self.materials)
        invalid_normal = (~torch.isfinite(normals[hit])) | ((normals[hit] - 1.0).abs() > 1e-3)
        return {
            "raw_mesh_hits": int(raw.sum().item()),
            "visible_hits": int(hit.sum().item()),
            "scene_occluded_hits": int(occluded.sum().item()),
            "occluded_survivors": int((occluded & hit).sum().item()),
            "invalid_depth": int(invalid_depth.sum().item()),
            "invalid_face": int(invalid_face.sum().item()),
            "invalid_material": int(invalid_material.sum().item()),
            "invalid_normal": int(invalid_normal.sum().item()),
        }
