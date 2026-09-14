"""Closed-form camera rays and an analytic sphere, sharing the renderer's exact ray construction.

The Warp kernel builds a ray direction as normalize(((x - W/2)/f, (y - H/2)/f, 1)) in the camera
frame and rotates it into the world with the pose quaternion. That expression is reproduced here
once, in float64, and everything analytic in this track is built on it: the analytic sphere arm,
the closed-form projections used by the RC-R1 gates, and the rays handed to the independent CPU
reference intersector. Reproducing it is the point - if this expression and the kernel's disagree,
the RC-R1 cross-implementation gate is what finds out.
"""
import numpy as np
import torch

from .gbuffer import GBuffer
from .scene import Camera


def camera_directions(camera: Camera):
    """[H,W,3] unit ray directions in the camera frame, pixel (x, y) sampled at its integer index."""
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera")
    x, y = np.meshgrid(np.arange(camera.width, dtype=np.float64),
                       np.arange(camera.height, dtype=np.float64))
    focal = camera.focal_px
    directions = np.stack([(x - camera.width / 2.0) / focal,
                           (y - camera.height / 2.0) / focal,
                           np.ones_like(x)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def world_rays(camera: Camera, position, rotation):
    """Ray origin and [H,W,3] world directions for one camera pose (camera->world rotation)."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    directions = camera_directions(camera) @ rotation.T
    return position, directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def project_points(camera: Camera, points_world, position, rotation):
    """World points to continuous pixel coordinates (u, v) and camera +Z depth.

    Exactly inverts camera_directions: a point at pixel coordinate u lies on the ray of pixel
    index u. Points at or behind the camera plane are refused rather than wrapped around.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    position = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    local = (points - position) @ rotation
    if (local[:, 2] <= 1e-9).any():
        raise ValueError("A point is not in front of the camera; its projection is undefined")
    focal = camera.focal_px
    u = focal * local[:, 0] / local[:, 2] + camera.width / 2.0
    v = focal * local[:, 1] / local[:, 2] + camera.height / 2.0
    return np.stack([u, v], axis=-1), local[:, 2]


def sphere_projected_radius_px(camera: Camera, distance_m, radius_m):
    """Exact silhouette radius of a sphere: the tangent cone's half angle, not r*f/z."""
    distance, radius = float(distance_m), float(radius_m)
    if radius <= 0.0 or distance <= radius:
        raise ValueError("Sphere radius must be positive and smaller than the view distance")
    return camera.focal_px * radius / np.sqrt(distance * distance - radius * radius)


def sphere_hits(camera: Camera, position, rotation, radius_m, centre=(0.0, 0.0, 0.0)):
    """Nearest sphere intersection per pixel in float64: (range, normal_world, hit)."""
    radius = float(radius_m)
    if radius <= 0.0:
        raise ValueError("Sphere radius must be positive")
    origin, directions = world_rays(camera, position, rotation)
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    oc = centre - origin
    b = directions @ oc
    discriminant = b * b - (float(oc @ oc) - radius * radius)
    positive = discriminant > 0.0
    root = np.sqrt(np.where(positive, discriminant, 0.0))
    near = b - root
    far = b + root
    distance = np.where(near > 1e-9, near, far)
    hit = positive & (distance > 1e-9) & (distance <= camera.far_range_m)
    points = origin + distance[..., None] * directions
    normal = (points - centre) / radius
    return (np.where(hit, distance, 0.0), np.where(hit[..., None], normal, 0.0), hit)


def gbuffer_from_analytic(camera: Camera, ranges, normals, hits, device):
    """Pack per-view analytic results into the same G-buffer container the mesh path returns.

    face_id and instance_id are 0 on a hit and -1 on a miss: an analytic primitive has no triangles,
    and pretending it has several would invent structure the specimen does not have.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    hits = np.asarray(hits, dtype=bool)
    count = ranges.shape[0]
    if ranges.shape != (count, camera.height, camera.width) or normals.shape != ranges.shape + (3,):
        raise ValueError("Analytic buffers must be [N,H,W] and [N,H,W,3]")
    if hits.shape != ranges.shape:
        raise ValueError("Hit mask shape mismatch")
    if not np.isfinite(ranges).all() or not np.isfinite(normals).all():
        raise ValueError("Non-finite analytic buffer")
    lengths = np.linalg.norm(normals, axis=-1)
    if (hits & (np.abs(lengths - 1.0) > 1e-9)).any():
        raise ValueError("Analytic hit normal is not unit length")
    projection = camera.optical_projection().astype(np.float64)
    torch_device = torch.device(device)
    valid = torch.tensor(hits, device=torch_device)
    range_m = torch.tensor(ranges.astype(np.float32), device=torch_device)
    depth_m = torch.tensor((ranges * projection).astype(np.float32), device=torch_device)
    normal_world = torch.tensor(normals.astype(np.float32), device=torch_device)
    label = torch.where(valid, torch.zeros_like(valid, dtype=torch.int32),
                        torch.full_like(valid, -1, dtype=torch.int32))
    return GBuffer(range_m, depth_m, normal_world, label.clone(), label.clone(), valid)


def analytic_sphere_gbuffer(camera: Camera, grid, radius_m, centre=(0.0, 0.0, 0.0), device="cpu"):
    """The A0 arm: one analytic sphere seen from every view of a grid."""
    rotations = grid.rotations()
    ranges, normals, hits = [], [], []
    for index in range(len(grid)):
        one_range, one_normal, one_hit = sphere_hits(
            camera, grid.positions[index], rotations[index], radius_m, centre)
        ranges.append(one_range)
        normals.append(one_normal)
        hits.append(one_hit)
    return gbuffer_from_analytic(camera, np.stack(ranges), np.stack(normals), np.stack(hits), device)
