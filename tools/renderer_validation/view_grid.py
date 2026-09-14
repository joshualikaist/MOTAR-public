"""Deterministic multi-view camera poses aimed at the world origin.

The renderer's camera frame is +X right, +Y down, +Z forward, and the Warp kernel rotates
camera-frame ray directions into the world with the pose quaternion. So a view is fully described
by the rotation whose columns are the camera axes expressed in world coordinates.

World convention here, stated once and never inferred: the up direction is world -Y, which is the
direction an identity-pose camera calls "up". Azimuth turns about that axis, elevation lifts from
the horizontal plane, and distance is measured from the origin.
"""
import math

import numpy as np

WORLD_UP = np.array([0.0, -1.0, 0.0])
MAX_ELEVATION_DEG = 80.0


def spherical_position(azimuth_deg, elevation_deg, distance_m):
    """Camera centre for one view. az=0, el=0 puts the camera on +Z looking back at the origin."""
    if not 0.05 <= float(distance_m) <= 100.0:
        raise ValueError("distance_m must lie in [0.05, 100] m")
    if abs(float(elevation_deg)) > MAX_ELEVATION_DEG:
        raise ValueError("elevation must stay below the pole; the up axis would be degenerate")
    azimuth, elevation = math.radians(float(azimuth_deg)), math.radians(float(elevation_deg))
    return float(distance_m) * np.array([
        math.sin(azimuth) * math.cos(elevation),
        -math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation)])


def look_at_rotation(position, target=(0.0, 0.0, 0.0)):
    """Camera->world rotation with columns [right, down, forward], right-handed."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("Camera sits on its target; there is no view direction")
    forward = forward / norm
    down = -WORLD_UP - np.dot(-WORLD_UP, forward) * forward
    length = np.linalg.norm(down)
    if length < 1e-6:
        raise ValueError("View direction is parallel to the up axis; the frame is degenerate")
    down = down / length
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    rotation = np.stack([right, down, forward], axis=1)
    if not np.isfinite(rotation).all() or abs(np.linalg.det(rotation) - 1.0) > 1e-9:
        raise RuntimeError("look_at produced a non-rotation")
    return rotation


def rotation_to_quaternion(rotation):
    """xyzw quaternion of a 3x3 rotation, by the branch with the largest pivot."""
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[2, 1] - m[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                          (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                          (m[1, 0] - m[0, 1]) / s])
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def quaternion_to_rotation(quaternion):
    """Inverse of rotation_to_quaternion, written out so a round trip is a real check."""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=np.float64).reshape(4))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ViewGrid:
    """An ordered, hashable set of views. The order is part of the measurement's identity."""

    def __init__(self, views):
        rows = []
        for view in views:
            azimuth, elevation, distance = (float(view[0]), float(view[1]), float(view[2]))
            rows.append((azimuth, elevation, distance))
        if not rows:
            raise ValueError("A view grid needs at least one view")
        if len(set(rows)) != len(rows):
            raise ValueError("Duplicate view in grid; per-view results would be ambiguous")
        self.views = tuple(rows)
        positions, quaternions = [], []
        for azimuth, elevation, distance in self.views:
            position = spherical_position(azimuth, elevation, distance)
            rotation = look_at_rotation(position)
            quaternion = rotation_to_quaternion(rotation)
            if not np.allclose(quaternion_to_rotation(quaternion), rotation, atol=1e-9, rtol=0):
                raise RuntimeError("Quaternion round trip disagrees with the rotation")
            positions.append(position)
            quaternions.append(quaternion)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.quaternions = np.asarray(quaternions, dtype=np.float64)

    def __len__(self):
        return len(self.views)

    def float32(self):
        return self.positions.astype(np.float32), self.quaternions.astype(np.float32)

    def rotations(self):
        return np.stack([quaternion_to_rotation(q) for q in self.quaternions])

    def subset(self, indices):
        return ViewGrid([self.views[i] for i in indices])

    def index_of(self, view):
        key = (float(view[0]), float(view[1]), float(view[2]))
        if key not in self.views:
            raise KeyError(f"View {key} is not in this grid")
        return self.views.index(key)

    def as_dict(self):
        return {"views": [{"azimuth_deg": a, "elevation_deg": e, "distance_m": d}
                          for a, e, d in self.views],
                "positions": self.positions.tolist(),
                "quaternions_xyzw": self.quaternions.tolist(),
                "world_up": WORLD_UP.tolist(),
                "camera_axes": "+X right, +Y down, +Z forward; quaternion rotates camera to world"}


# Preregistered grids (results/renderer_characterization_2026-09-13/PREREGISTRATION.md, section 2).
PRIMARY_AZIMUTHS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
PRIMARY_ELEVATIONS = (-20.0, 0.0, 35.0)
PRIMARY_DISTANCE_M = 2.0
SWEEP_AZIMUTHS = (0.0, 45.0, 90.0)
SWEEP_ELEVATIONS = (0.0, 35.0)
SWEEP_DISTANCES_M = (1.5, 2.0, 2.5, 3.0, 4.0)
FIT_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
FIT_ELEVATION = 0.0


def primary_grid():
    return ViewGrid([(a, e, PRIMARY_DISTANCE_M)
                     for e in PRIMARY_ELEVATIONS for a in PRIMARY_AZIMUTHS])


def distance_sweep_grid():
    return ViewGrid([(a, e, d) for d in SWEEP_DISTANCES_M
                     for e in SWEEP_ELEVATIONS for a in SWEEP_AZIMUTHS])


def fit_grid():
    return ViewGrid([(a, FIT_ELEVATION, PRIMARY_DISTANCE_M) for a in FIT_AZIMUTHS])


def fit_validation_split():
    """The preregistered disjoint split of the primary grid: 4 fit views, 20 validation views."""
    primary = primary_grid()
    fit = [primary.index_of(view) for view in fit_grid().views]
    validation = [i for i in range(len(primary)) if i not in set(fit)]
    if len(fit) != 4 or len(validation) != 20 or set(fit) & set(validation):
        raise RuntimeError("The preregistered fit/validation split no longer holds")
    return primary, fit, validation
