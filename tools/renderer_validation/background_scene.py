"""Procedural general graphics fixture: floor, wall, column, and a hinged display panel.

No asset loader, simulator, detector, object identity model or physics dependency.
Patch IDs describe surfaces for diagnostics, not independent statistical samples.
"""
from dataclasses import dataclass, replace
import math
import numpy as np

from .scene import MeshScene, integer_array

PART_NAMES = ("floor", "wall", "column", "hinged_base", "hinged_leaf")


@dataclass(frozen=True)
class BackgroundFixture:
    mesh: MeshScene
    face_part: np.ndarray
    face_patch: np.ndarray
    joint_angle_deg: float

    def __post_init__(self):
        for name in ("face_part", "face_patch"):
            value = integer_array(getattr(self, name))
            if value.shape != (len(self.mesh.triangles),) or value.min() < 0:
                raise ValueError("One nonnegative metadata value per triangle required")
            object.__setattr__(self, name, value)
        if self.face_part.max() >= len(PART_NAMES):
            raise ValueError("Unknown part")

    def as_dict(self):
        return {"mesh": self.mesh.as_dict(), "face_part": self.face_part.tolist(),
                "face_patch": self.face_patch.tolist(), "part_names": list(PART_NAMES),
                "joint_angle_deg": self.joint_angle_deg,
                "instance_names": ["floor", "wall", "column", "hinged_panel"]}


class _Builder:
    def __init__(self):
        self.vertices, self.triangles, self.materials = [], [], []
        self.instances, self.parts, self.patches = [], [], []
        self.next_patch = 0

    def polygon(self, points, part, instance, patch=None):
        if patch is None:
            patch = self.next_patch
            self.next_patch += 1
        start = len(self.vertices)
        self.vertices.extend(points)
        for i in range(1, len(points)-1):
            self.triangles.append([start, start+i, start+i+1])
            self.materials.append(patch % 4)
            self.instances.append(instance)
            self.parts.append(part)
            self.patches.append(patch)

    def box(self, center, half_extents, rotation, part, instance):
        points = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                           [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
        points = (points * half_extents) @ rotation.T + center
        for ids in [[0,3,2,1], [4,5,6,7], [0,1,5,4], [3,7,6,2], [0,4,7,3], [1,2,6,5]]:
            self.polygon(points[ids], part, instance)


def background_fixture(joint_angle_deg=30.0):
    if not np.isfinite(joint_angle_deg) or not -60 <= joint_angle_deg <= 60:
        raise ValueError("Joint angle must be finite and in [-60,60] degrees")
    b = _Builder()
    xs, zs = np.linspace(-4, 4, 5), np.linspace(1, 9, 7)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for z0, z1 in zip(zs[:-1], zs[1:]):
            b.polygon([[x0,2,z0], [x1,2,z0], [x1,2,z1], [x0,2,z1]], 0, 0)
    ys = np.linspace(-2.5, 2, 4)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            b.polygon([[x0,y0,9], [x0,y1,9], [x1,y1,9], [x1,y0,9]], 1, 1)
    # Outward normals on the sides and caps; +Y points down.
    angles = np.linspace(0, 2*math.pi, 13)[:-1]
    top = np.column_stack([-1.6 + .45*np.cos(angles), np.full(12,-.7), 4.5 + .45*np.sin(angles)])
    bottom = top.copy()
    bottom[:,1] = 2
    for i in range(12):
        nxt = (i+1) % 12
        b.polygon([top[i], bottom[i], bottom[nxt], top[nxt]], 2, 2)
    # Each whole cap is one patch despite its twelve triangles.
    for ring, center, reverse in [(top, [-1.6,-.7,4.5], False), (bottom, [-1.6,2,4.5], True)]:
        patch = b.next_patch
        b.next_patch += 1
        for i in range(12):
            pair = [ring[i], ring[(i+1) % 12]]
            b.polygon([center] + (pair[::-1] if reverse else pair), 2, 2, patch)
    b.box([1.4,1.35,4.8], [.5,.65,.16], np.eye(3), 3, 3)
    angle = math.radians(joint_angle_deg)
    rotation = np.array([[1,0,0], [0,math.cos(angle),-math.sin(angle)], [0,math.sin(angle),math.cos(angle)]])
    hinge = np.array([1.4,.7,4.8])
    center = hinge + rotation @ np.array([0,-.65,0])
    b.box(center, [.5,.65,.12], rotation, 4, 3)
    return BackgroundFixture(MeshScene(b.vertices, b.triangles, b.materials, b.instances),
                             b.parts, b.patches, float(joint_angle_deg))


def cycle_materials(scene):
    """A declared intervention: every surface changes material, without inspecting the image.

    This guarantees changed IDs, not lower R² or statistical independence from depth.
    """
    if scene.material_count != 4:
        raise ValueError("This fixed intervention requires four materials")
    return replace(scene, face_material=(scene.face_material + 1) % 4)
