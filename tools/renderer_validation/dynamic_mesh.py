"""Ray-intersect a rigid mesh whose pose changes per frame, without rebuilding its BVH.

The mesh stays in object-local coordinates and is built once. Each frame the camera ray is
transformed world -> object by the inverse of the object's pose, intersected against that static
BVH, and the hit distance and normal are transformed back. Nothing about the acceleration
structure depends on the pose, so no refit is needed.

Static scene geometry and the dynamic object are kept in SEPARATE meshes on purpose. Putting a
moving object into the shared scene mesh would make it occlude itself in any kernel that uses one
mesh id for both the primary hit and the occlusion test.

This module imports no aerial_gym, no task, no detector and no policy, and it writes its own Warp
kernel rather than reusing the production camera kernels, because the production ones take a
camera pose and have nowhere to put an object pose.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from .scene import MeshScene

MISS = 1.0e30


@wp.kernel
def dynamic_and_static_raycast(
    static_mesh: wp.uint64,
    dynamic_mesh: wp.uint64,
    ray_origin: wp.array(dtype=wp.vec3, ndim=2),
    ray_direction: wp.array(dtype=wp.vec3, ndim=2),
    object_position: wp.array(dtype=wp.vec3),
    object_rotation: wp.array(dtype=wp.quat),
    far: float,
    out_range: wp.array(dtype=wp.float32, ndim=2),
    out_normal: wp.array(dtype=wp.vec3, ndim=2),
    out_face: wp.array(dtype=wp.int32, ndim=2),
    out_source: wp.array(dtype=wp.int32, ndim=2),
    use_static: int,
    dynamic_mode: int,
    proxy_radius: float,
):
    """One pixel: nearer of the static hit and the object-local dynamic hit.

    out_source is 0 for a miss, 1 for the static scene, 2 for the dynamic object, so the two
    contributions stay separable in the receipt instead of being merged into one depth image.
    """
    frame, pixel = wp.tid()
    origin = ray_origin[frame, pixel]
    direction = ray_direction[frame, pixel]

    best = float(far)
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    best_face = int(-1)
    best_source = int(0)

    if use_static == 1:
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(static_mesh, origin, direction, far, t, u, v, sign, normal, face):
            if t < best:
                best = t
                best_normal = normal
                best_face = face
                best_source = 1

    if dynamic_mode == 1:
        # Analytic sphere, the treatment the production detector uses for the moving target. Run
        # in the SAME kernel so the comparison isolates mesh query against analytic test rather
        # than GPU against CPU: an earlier version did this part in numpy and made the proxy look
        # slower than the mesh, which measured the host round trip and nothing else.
        oc = origin - object_position[frame]
        b = wp.dot(oc, direction)
        c = wp.dot(oc, oc) - proxy_radius * proxy_radius
        disc = b * b - c
        if disc >= 0.0:
            root = wp.sqrt(disc)
            t = -b - root
            if t < 0.0:
                t = -b + root
            if t >= 0.0 and t < best:
                best = t
                best_normal = wp.normalize(origin + direction * t - object_position[frame])
                best_face = 0
                best_source = 2

    if dynamic_mode == 2:
        # world -> object: subtract the translation, then rotate by the inverse rotation. The
        # rotation is a unit quaternion, so the direction keeps unit length and the ray parameter
        # t is the same number in both frames; no rescaling is needed.
        rotation = object_rotation[frame]
        local_origin = wp.quat_rotate_inv(rotation, origin - object_position[frame])
        local_direction = wp.quat_rotate_inv(rotation, direction)
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(dynamic_mesh, local_origin, local_direction, far,
                             t, u, v, sign, normal, face):
            if t < best:
                best = t
                # object -> world for the normal. A rotation is orthonormal, so rotating the
                # normal is correct here; an anisotropic scale would need the inverse transpose.
                best_normal = wp.quat_rotate(rotation, normal)
                best_face = face
                best_source = 2

    if best_source == 0:
        out_range[frame, pixel] = MISS
        out_normal[frame, pixel] = wp.vec3(0.0, 0.0, 0.0)
        out_face[frame, pixel] = -1
        out_source[frame, pixel] = 0
    else:
        out_range[frame, pixel] = best
        out_normal[frame, pixel] = best_normal
        out_face[frame, pixel] = best_face
        out_source[frame, pixel] = best_source


@dataclass(frozen=True)
class RaycastResult:
    range_m: np.ndarray
    normal_world: np.ndarray
    face_id: np.ndarray
    source: np.ndarray          # 0 miss, 1 static, 2 dynamic

    @property
    def hit(self):
        return self.source > 0


def unit_quaternion(array):
    q = np.asarray(array, dtype=np.float64).reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-8).any():
        raise ValueError("Object rotations must be non-degenerate quaternions")
    return (q / norm).astype(np.float32)


class DynamicMeshRaycaster:
    """Owns one static scene mesh and one object-local dynamic mesh, both built once."""

    def __init__(self, static_scene: MeshScene, dynamic_scene: MeshScene, device="cuda:0",
                 far_range_m=50.0):
        if not 0.0 < far_range_m < MISS:
            raise ValueError("far_range_m must be positive and below the miss sentinel")
        wp.init()
        self.device = str(device)
        self.far = float(far_range_m)
        self._static_arrays = self._mesh(static_scene)
        self._dynamic_arrays = self._mesh(dynamic_scene)
        self.static_mesh, self.dynamic_mesh = self._static_arrays[0], self._dynamic_arrays[0]
        # Recorded so a test can prove the pose path never rebuilds them.
        self.build_signature = (self.static_mesh.id, self.dynamic_mesh.id,
                                self._static_arrays[1].ptr, self._dynamic_arrays[1].ptr)

    def _mesh(self, scene: MeshScene):
        points = wp.array(np.ascontiguousarray(scene.vertices, dtype=np.float32),
                          dtype=wp.vec3, device=self.device)
        indices = wp.array(np.ascontiguousarray(scene.triangles.reshape(-1), dtype=np.int32),
                           dtype=wp.int32, device=self.device)
        return wp.Mesh(points=points, indices=indices), points, indices

    def upload_rays(self, origins, directions):
        """Put the ray set on the device once and keep it.

        cast() uploads its rays every call, which is right for a correctness test and wrong for a
        timing one: at 128 frames and 160x90 that is 22 MiB of host-to-device traffic per call,
        and a first benchmark measured mostly that. Hoist the rays, then time only the query.
        """
        origins, directions = self._validated_rays(origins, directions)
        return (wp.array(origins, dtype=wp.vec3, device=self.device),
                wp.array(directions, dtype=wp.vec3, device=self.device),
                origins.shape[0], origins.shape[1])

    def _validated_rays(self, origins, directions):
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        if origins.ndim != 3 or origins.shape != directions.shape or origins.shape[-1] != 3:
            raise ValueError("origins and directions must both be [frames, pixels, 3]")
        lengths = np.linalg.norm(directions, axis=-1)
        if not np.isfinite(lengths).all() or (np.abs(lengths - 1.0) > 1e-4).any():
            raise ValueError("ray directions must be unit length")
        return origins, directions

    def cast_uploaded(self, rays, positions, rotations, out, use_static=True,
                      dynamic_mode=2, proxy_radius=0.25):
        """Launch against already-uploaded rays and already-allocated outputs. Nothing else."""
        ray_origin, ray_direction, frames, pixels = rays
        wp.launch(
            dynamic_and_static_raycast, dim=(frames, pixels),
            inputs=[self.static_mesh.id, self.dynamic_mesh.id, ray_origin, ray_direction,
                    positions, rotations, self.far, *out,
                    int(bool(use_static)), int(dynamic_mode), float(proxy_radius)],
            device=self.device)

    def allocate_outputs(self, frames, pixels):
        return (wp.zeros((frames, pixels), dtype=wp.float32, device=self.device),
                wp.zeros((frames, pixels), dtype=wp.vec3, device=self.device),
                wp.zeros((frames, pixels), dtype=wp.int32, device=self.device),
                wp.zeros((frames, pixels), dtype=wp.int32, device=self.device))

    def upload_poses(self, positions, rotations, frames):
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(frames, 3)
        rotations = unit_quaternion(rotations).reshape(frames, 4)
        return (wp.array(positions, dtype=wp.vec3, device=self.device),
                wp.array(rotations, dtype=wp.quat, device=self.device))

    def cast(self, origins, directions, positions, rotations, use_static=True,
             dynamic_mode=2, proxy_radius=0.25):
        """origins/directions are [frames, pixels, 3]; positions/rotations are per frame.

        dynamic_mode: 0 none, 1 analytic sphere proxy, 2 object-local mesh.
        """
        if dynamic_mode not in (0, 1, 2):
            raise ValueError("dynamic_mode must be 0 (none), 1 (analytic proxy) or 2 (mesh)")
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        if origins.ndim != 3 or origins.shape != directions.shape or origins.shape[-1] != 3:
            raise ValueError("origins and directions must both be [frames, pixels, 3]")
        frames, pixels = origins.shape[:2]
        lengths = np.linalg.norm(directions, axis=-1)
        if not np.isfinite(lengths).all() or (np.abs(lengths - 1.0) > 1e-4).any():
            raise ValueError("ray directions must be unit length")
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(frames, 3)
        rotations = unit_quaternion(rotations).reshape(frames, 4)

        out_range = wp.zeros((frames, pixels), dtype=wp.float32, device=self.device)
        out_normal = wp.zeros((frames, pixels), dtype=wp.vec3, device=self.device)
        out_face = wp.zeros((frames, pixels), dtype=wp.int32, device=self.device)
        out_source = wp.zeros((frames, pixels), dtype=wp.int32, device=self.device)
        wp.launch(
            dynamic_and_static_raycast, dim=(frames, pixels),
            inputs=[self.static_mesh.id, self.dynamic_mesh.id,
                    wp.array(origins, dtype=wp.vec3, device=self.device),
                    wp.array(directions, dtype=wp.vec3, device=self.device),
                    wp.array(positions, dtype=wp.vec3, device=self.device),
                    wp.array(rotations, dtype=wp.quat, device=self.device),
                    self.far, out_range, out_normal, out_face, out_source,
                    int(bool(use_static)), int(dynamic_mode), float(proxy_radius)],
            device=self.device)
        wp.synchronize_device(self.device)
        return RaycastResult(out_range.numpy(), out_normal.numpy(),
                             out_face.numpy(), out_source.numpy())
