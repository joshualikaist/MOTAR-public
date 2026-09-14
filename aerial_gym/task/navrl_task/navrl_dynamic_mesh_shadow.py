"""Shadow-only dynamic-mesh ray query, for measuring cost in the production render path.

Nothing here reaches the detector's output. It runs beside the existing render, writes into its
own buffers, and is read only by the D7 benchmark. The feature is off unless
NAVRL_DYNAMIC_MESH_SHADOW is set, and with it off this module is never imported by the detector.

Why it exists: navrl_physical_target_params sets include_in_warp False, so the moving target is
absent from the static Warp scene and the detector ray-tests an analytic sphere or box instead.
D6 showed that a rigid mesh can be intersected in its own frame without rebuilding a BVH. D7 asks
what that costs where it would actually run. It does not change what the detector sees, and it is
not a step toward doing so without its own preregistration.

Ray construction mirrors the detector's own kernel: the same origins, orientations and
ray_vectors, so the measured work is the work an integration would do.
"""
import os

import numpy as np
import warp as wp

FLAG = "NAVRL_DYNAMIC_MESH_SHADOW"
MISS = 1.0e30


def shadow_enabled():
    """Default off. An unrecognised value is refused rather than read as off."""
    raw = os.environ.get(FLAG, "0").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{FLAG}={raw!r} is not a recognised boolean; use 1/0, true/false, yes/no or on/off.")


@wp.kernel
def shadow_dynamic_mesh_kernel(
    dynamic_mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    ray_vectors: wp.array2d(dtype=wp.vec3),
    target_positions: wp.array(dtype=wp.vec3),
    target_orientations: wp.array(dtype=wp.quat),
    far_plane: float,
    shadow_depth: wp.array(dtype=wp.float32, ndim=3),
    shadow_hit: wp.array(dtype=wp.int32, ndim=3),
):
    """Per pixel: intersect the target's own mesh in the target's frame. Writes shadow buffers only."""
    env_id, row, col = wp.tid()
    ro = origins[env_id]
    rd = wp.normalize(wp.quat_rotate(orientations[env_id], ray_vectors[row, col]))

    shadow_depth[env_id, row, col] = far_plane
    shadow_hit[env_id, row, col] = wp.int32(0)

    # world -> target local. The rotation is a unit quaternion, so the ray parameter t means the
    # same distance in both frames and needs no rescaling.
    rotation = target_orientations[env_id]
    local_origin = wp.quat_rotate_inv(rotation, ro - target_positions[env_id])
    local_direction = wp.quat_rotate_inv(rotation, rd)

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)
    if wp.mesh_query_ray(dynamic_mesh, local_origin, local_direction, far_plane,
                         t, u, v, sign, normal, face):
        shadow_depth[env_id, row, col] = t
        shadow_hit[env_id, row, col] = wp.int32(1)


class DynamicMeshShadow:
    """Owns the target-local mesh and the shadow buffers. Built once; poses never rebuild it."""

    def __init__(self, mesh_scene, num_envs, height, width, device, far_plane):
        self.num_envs, self.height, self.width = int(num_envs), int(height), int(width)
        self.far_plane = float(far_plane)
        self.device = str(device)
        vertices = np.ascontiguousarray(mesh_scene.vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(mesh_scene.triangles.reshape(-1), dtype=np.int32)
        self._points = wp.array(vertices, dtype=wp.vec3, device=self.device)
        self._indices = wp.array(triangles, dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self._points, indices=self._indices)
        self.triangles = int(len(mesh_scene.triangles))
        shape = (self.num_envs, self.height, self.width)
        self.shadow_depth = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.shadow_hit = wp.zeros(shape, dtype=wp.int32, device=self.device)
        # Recorded so a test can show pose changes rebuild nothing.
        self.build_signature = (self.mesh.id, self._points.ptr, self._indices.ptr)

    def run(self, origins, orientations, ray_vectors, target_positions, target_orientations):
        """Launch the shadow query. Returns nothing: the detector must not read these buffers."""
        wp.launch(
            shadow_dynamic_mesh_kernel,
            dim=(self.num_envs, self.height, self.width),
            inputs=[self.mesh.id, origins, orientations, ray_vectors,
                    target_positions, target_orientations, self.far_plane,
                    self.shadow_depth, self.shadow_hit],
            device=self.device)

    def hit_count(self):
        """Diagnostics only, for the benchmark receipt; never an observation."""
        return int(self.shadow_hit.numpy().sum())
