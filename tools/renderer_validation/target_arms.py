"""The four RC specimens, behind one interface: analytic sphere, box, scaled box, quadrotor mesh.

Specimen identity here is geometric only. These are shapes to measure a renderer with; nothing in
this module knows about detectors, policies, tracking or the meaning of any of them in a task.

Preregistered definitions: results/renderer_characterization_2026-09-13/PREREGISTRATION.md §1.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .scene import MeshScene, asymmetric_box_fixture
from .analytic_primitives import analytic_sphere_gbuffer

ROOT = Path(__file__).resolve().parents[2]
SPHERE_RADIUS_M = 0.15
BOX_HALF_EXTENTS_M = (0.14, 0.14, 0.06)
QUADROTOR_URDF = "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf"
ARM_NAMES = ("analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh")
ARM_CODES = {"analytic_sphere": "A0", "box_proxy": "A1",
             "area_matched_box": "A2", "quadrotor_mesh": "A3"}


def box_scene(half_extents=BOX_HALF_EXTENTS_M, scale=1.0):
    """A box centred on the origin. scale is isotropic and multiplies every half extent."""
    scale = float(scale)
    if not 0.01 <= scale <= 100.0:
        raise ValueError("Isotropic scale must lie in [0.01, 100]")
    extents = np.asarray(half_extents, dtype=np.float64) * 2.0 * scale
    if extents.shape != (3,) or (extents <= 0.0).any():
        raise ValueError("Box half extents must be three positive numbers")
    return asymmetric_box_fixture(tuple(extents.tolist()), (0.0, 0.0, 0.0))


def quadrotor_scene(path=None):
    """The generated quadrotor airframe URDF, read through the audited URDF loader."""
    from .urdf_asset import load_urdf_asset
    asset = load_urdf_asset(str(ROOT / (path or QUADROTOR_URDF)))
    return asset.mesh, asset


@dataclass(frozen=True)
class Arm:
    """One specimen: either a triangle mesh or an analytic primitive, never both."""
    name: str
    code: str
    kind: str                      # "mesh" or "analytic_sphere"
    mesh: Optional[MeshScene]
    radius_m: Optional[float]
    scale: float
    circumscribed_radius_m: float
    description: dict

    def gbuffer(self, camera, grid, device="cpu", renderer=None):
        if self.kind == "analytic_sphere":
            return analytic_sphere_gbuffer(camera, grid, self.radius_m, device=device)
        if renderer is None:
            renderer = self.renderer(camera, len(grid), device)
        positions, quaternions = grid.float32()
        renderer.set_camera_poses(positions, quaternions)
        return renderer.render()

    def renderer(self, camera, views, device):
        if self.kind != "mesh":
            raise TypeError("An analytic arm has no mesh renderer")
        from .gbuffer import WarpGBufferRenderer
        return WarpGBufferRenderer(self.mesh, camera, views, device)

    @property
    def material_count(self):
        return 1 if self.mesh is None else self.mesh.material_count

    @property
    def triangles(self):
        return 0 if self.mesh is None else int(len(self.mesh.triangles))


def circumscribed_radius(mesh: MeshScene):
    return float(np.linalg.norm(np.asarray(mesh.vertices, dtype=np.float64), axis=1).max())


def build_arm(name, scale=None, urdf_path=None):
    """Construct one arm. `scale` is accepted only by the area-matched box, and required by it."""
    if name not in ARM_NAMES:
        raise ValueError(f"Unknown arm {name!r}; expected one of {ARM_NAMES}")
    if name == "area_matched_box":
        if scale is None:
            raise ValueError("area_matched_box requires the scale fitted in RC-R2; it has no default")
    elif scale is not None:
        raise ValueError(f"Arm {name!r} has no scale parameter; only area_matched_box is scaled")
    code = ARM_CODES[name]
    if name == "analytic_sphere":
        return Arm(name, code, "analytic_sphere", None, SPHERE_RADIUS_M, 1.0, SPHERE_RADIUS_M,
                   {"kind": "analytic_sphere", "radius_m": SPHERE_RADIUS_M,
                    "triangles": 0, "materials": 1,
                    "source": "closed-form sphere intersection, no triangles"})
    if name in ("box_proxy", "area_matched_box"):
        factor = 1.0 if scale is None else float(scale)
        mesh = box_scene(BOX_HALF_EXTENTS_M, factor)
        return Arm(name, code, "mesh", mesh, None, factor, circumscribed_radius(mesh),
                   {"kind": "box", "half_extents_m": [h * factor for h in BOX_HALF_EXTENTS_M],
                    "isotropic_scale": factor, "triangles": int(len(mesh.triangles)),
                    "materials": mesh.material_count, "source": "procedural axis-aligned box"})
    mesh, asset = quadrotor_scene(urdf_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return Arm(name, code, "mesh", mesh, None, 1.0, circumscribed_radius(mesh),
               {"kind": "urdf_mesh", "triangles": int(len(mesh.triangles)),
                "materials": mesh.material_count,
                "visual_links": int(len(set(mesh.face_instance.tolist()))),
                "bounding_box_extent_m": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
                "bounding_box_centre_m": ((vertices.max(axis=0) + vertices.min(axis=0)) / 2).tolist(),
                "source_path": asset.source_path, "source_sha256": asset.source_sha256,
                "backend": asset.backend, "library_versions": dict(asset.library_versions)})
