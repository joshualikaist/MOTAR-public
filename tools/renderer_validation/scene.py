"""Static triangle geometry and sequence-fixed appearance contracts (metres, linear RGB)."""
from dataclasses import dataclass
import math

import numpy as np


def frozen_array(value, dtype):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite scene parameter")
    result.setflags(write=False)
    return result


def integer_array(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or (raw.size and (
            raw.min() < np.iinfo(np.int32).min or raw.max() > np.iinfo(np.int32).max)):
        raise ValueError("Indices must be int32-compatible integers")
    return frozen_array(raw, np.int32)


@dataclass(frozen=True)
class Camera:
    width: int = 160
    height: int = 120
    horizontal_fov_deg: float = 60.0
    far_range_m: float = 20.0

    def __post_init__(self):
        if any(type(x) is not int or not 1 <= x <= 2048 for x in (self.width, self.height)):
            raise ValueError("Camera dimensions must be integers in [1, 2048]")
        if not 1.0 <= self.horizontal_fov_deg < 179.0:
            raise ValueError("Invalid horizontal FOV")
        # Existing generic range kernel uses 1000 as its miss sentinel.
        if not 0.0 < self.far_range_m < 1000.0:
            raise ValueError("far_range_m must be positive and below the kernel miss sentinel")

    @property
    def focal_px(self):
        return self.width / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))

    def inverse_intrinsics(self):
        k = np.eye(4, dtype=np.float32)
        k[0, 0] = k[1, 1] = 1.0 / self.focal_px
        k[0, 2] = -self.width / (2.0 * self.focal_px)
        k[1, 2] = -self.height / (2.0 * self.focal_px)
        return k

    def optical_projection(self):
        """Ray range -> camera +Z depth, including odd image dimensions."""
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return (1.0 / np.sqrt(1.0 + ((x - self.width / 2.0) / self.focal_px) ** 2
                             + ((y - self.height / 2.0) / self.focal_px) ** 2)).astype(np.float32)


@dataclass(frozen=True)
class MeshScene:
    vertices: np.ndarray
    triangles: np.ndarray
    face_material: np.ndarray
    face_instance: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "vertices", frozen_array(self.vertices, np.float32))
        for name in ("triangles", "face_material", "face_instance"):
            object.__setattr__(self, name, integer_array(getattr(self, name)))
        v, f = self.vertices, self.triangles
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
            raise ValueError("vertices must be [V,3], V >= 3")
        if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.min() < 0 or f.max() >= len(v):
            raise ValueError("triangles must be nonempty [F,3] valid vertex indices")
        for a in (self.face_material, self.face_instance):
            if a.shape != (len(f),) or a.min() < 0:
                raise ValueError("One nonnegative material/instance index per triangle is required")
        cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        areas = np.linalg.norm(cross, axis=1)
        if not np.isfinite(areas).all() or (areas <= 1e-10).any():
            raise ValueError("Degenerate triangle")

    @property
    def material_count(self):
        return int(self.face_material.max()) + 1

    def as_dict(self):
        return {name: getattr(self, name).tolist() for name in
                ("vertices", "triangles", "face_material", "face_instance")}


@dataclass(frozen=True)
class Appearance:
    base_color: np.ndarray       # [N,M,3], linear RGB
    kd: np.ndarray               # [N,M]
    light_direction: np.ndarray  # [N,3], world surface-to-light direction
    ambient: np.ndarray          # [N]
    directional: np.ndarray      # [N]

    def __post_init__(self):
        for name in ("base_color", "kd", "light_direction", "ambient", "directional"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), np.float32))
        c = self.base_color
        if c.ndim != 3 or c.shape[-1] != 3 or min(c.shape[:2]) < 1:
            raise ValueError("base_color must be [N,M,3]")
        n, m = c.shape[:2]
        if self.kd.shape != (n, m) or self.light_direction.shape != (n, 3):
            raise ValueError("Material/light shape mismatch")
        if any(a.shape != (n,) for a in (self.ambient, self.directional)):
            raise ValueError("Lighting gain shape mismatch")
        if (c < 0).any() or (c > 1).any() or (self.kd < 0).any() or (self.kd > 1).any():
            raise ValueError("Color and kd must lie in [0,1]")
        if (self.ambient < 0).any() or (self.directional < 0).any():
            raise ValueError("Negative lighting gain")
        lengths = np.linalg.norm(self.light_direction, axis=1, keepdims=True)
        if not np.isfinite(lengths).all() or (lengths < 1e-8).any():
            raise ValueError("Zero light direction")
        object.__setattr__(self, "light_direction", frozen_array(self.light_direction / lengths, np.float32))

    def as_dict(self):
        return {name: getattr(self, name).tolist() for name in
                ("base_color", "kd", "light_direction", "ambient", "directional")}


def sample_appearance(seed, num_scenes, material_count):
    """Sample once per sequence. Prefix stable when batch size changes; no global RNG."""
    if type(seed) is not int or seed < 0 or type(num_scenes) is not int or num_scenes < 1:
        raise ValueError("Nonnegative integer seed and positive scene count required")
    if type(material_count) is not int or material_count < 1:
        raise ValueError("Positive material count required")
    colors, lights = [], []
    for i in range(num_scenes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        colors.append(rng.uniform(0.15, 0.8, (material_count, 3)))
        lights.append(rng.normal(size=3) + np.array([0.0, 0.0, -1.5]))
    return Appearance(colors, np.full((num_scenes, material_count), 0.8), lights,
                      np.full(num_scenes, 0.2), np.full(num_scenes, 0.8))


def box_fixture():
    """Two static, ordinary boxes; no UAV assets, physics, identity or task semantics."""
    corners = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=np.float32)
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
                      [3,7,6], [3,6,2], [0,4,7], [0,7,3], [1,2,6], [1,6,5]], dtype=np.int32)
    angle = math.radians(25)
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0,1,0],
                         [-math.sin(angle), 0, math.cos(angle)]], dtype=np.float32)
    a = (corners * [0.55, 0.6, 0.5]) @ rotation.T + [-0.6, 0.0, 3.5]
    b = corners * [0.35, 0.4, 0.35] + [0.8, 0.2, 4.0]
    return MeshScene(np.concatenate([a, b]), np.concatenate([faces, faces + 8]),
                     np.repeat(np.arange(4, dtype=np.int32), 6),
                     np.repeat(np.arange(2, dtype=np.int32), 12))


def mixed_material_box_fixture():
    """box_fixture geometry with materials spread over both boxes, so material cannot mean depth.

    R4 showed that assigning material by box confounds it with range: the two boxes occupy nearly
    disjoint range intervals. Only the face-to-material map changes here; box_fixture itself must
    stay untouched so R3's array hashes remain valid.
    """
    base = box_fixture()
    return MeshScene(base.vertices, base.triangles,
                     (np.arange(len(base.triangles), dtype=np.int32) // 2) % 4, base.face_instance)


def quaternion(axis, radians):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = radians / 2.0
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def quaternion_product(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def sample_camera_poses(seed, num_scenes, position_jitter_m=0.1, angle_jitter_deg=3.0):
    """Small deterministic pose variation so range and normal distributions differ per scene.

    Prefix stable when the scene count changes, as in sample_appearance; no global RNG.
    """
    if type(seed) is not int or seed < 0 or type(num_scenes) is not int or num_scenes < 1:
        raise ValueError("Nonnegative integer seed and positive scene count required")
    if not 0.0 <= position_jitter_m <= 1.0 or not 0.0 <= angle_jitter_deg <= 15.0:
        raise ValueError("Pose jitter outside the range this fixture was checked for")
    positions, orientations = [], []
    for i in range(num_scenes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE, i]))
        positions.append(rng.uniform(-position_jitter_m, position_jitter_m, 3))
        yaw, pitch = np.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg, 2))
        orientations.append(quaternion_product(quaternion([0, 1, 0], yaw), quaternion([1, 0, 0], pitch)))
    q = np.asarray(orientations, dtype=np.float64)
    return (np.asarray(positions, dtype=np.float32),
            (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32))


def asymmetric_box_fixture(size=(0.6, 0.25, 0.12), offset=(0.12, -0.05, 0.03)):
    """One box whose extents all differ and whose centre is off the origin.

    Asymmetry is the point: a cube centred on the origin looks the same under many rotations, so
    it cannot show that a rotation reached the intersection at all.
    """
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    faces = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]],
                     dtype=np.int32)
    points = corners * (np.asarray(size, dtype=np.float64) / 2.0) + np.asarray(offset, dtype=np.float64)
    return MeshScene(points, faces, np.zeros(len(faces), np.int32), np.zeros(len(faces), np.int32))


def l_shape_fixture(arm=0.5, thickness=0.14, depth=0.12):
    """Two boxes joined at a corner: chiral, so a rotation cannot be mistaken for the identity."""
    first = asymmetric_box_fixture((arm, thickness, depth), (arm / 2.0, 0.0, 0.0))
    second = asymmetric_box_fixture((thickness, arm, depth), (0.0, arm / 2.0, 0.0))
    vertices = np.concatenate([first.vertices, second.vertices])
    triangles = np.concatenate([first.triangles, second.triangles + len(first.vertices)])
    parts = np.concatenate([np.zeros(len(first.triangles), np.int32),
                            np.ones(len(second.triangles), np.int32)])
    return MeshScene(vertices, triangles, parts, parts)
