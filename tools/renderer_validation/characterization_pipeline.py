"""One arm, one view grid, one appearance: the render path every RC experiment shares.

A cell renders the G-buffer once and shades it as many times as the experiment needs. That is why
RC-R3 and RC-R4 can hold geometry fixed by construction instead of by comparing two renders and
hoping they agree: there is only ever one geometry per cell.

Nothing in this module imports the simulator package, a task, a controller, a detector or a model.
"""
from dataclasses import asdict, replace
import hashlib

import numpy as np
import torch

from .scene import Appearance, Camera
from .shading import shade
from .target_arms import build_arm
from .view_grid import ViewGrid

SHADING_MODES = ("flat", "lambertian")
BACKGROUND = (0.0, 0.0, 0.0)
# Camera-frame light for RC-R3: above, to the side, and in front of the specimen. Camera axes are
# +X right, +Y down, +Z forward, so -Y is up and -Z points back towards the camera.
LIGHT_DIRECTION_CAMERA = (0.4, -0.5, -1.0)
GEOMETRY_ARRAYS = ("range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")


class AnalyticSurface:
    """The material table an analytic primitive needs to be shaded like a mesh.

    The analytic sphere has no triangles, so it has exactly one surface and one material. Saying
    that explicitly is better than inventing a triangle list to satisfy the shader's signature.
    """
    def __init__(self):
        self.face_material = np.zeros(1, dtype=np.int32)
        self.face_instance = np.zeros(1, dtype=np.int32)

    @property
    def material_count(self):
        return 1


def camera_relative_light(grid, direction_camera=LIGHT_DIRECTION_CAMERA):
    """One world light direction per view, fixed relative to each camera.

    A world-fixed light leaves the camera-facing side of the specimen in ambient light from about
    half the views, and an image that is uniformly ambient says nothing about shading. RC-R3 asks
    what shading does to a lit surface, so it lights each view the same way relative to its camera:
    slightly above, slightly to the side, and from the camera's side of the object. RC-R4 is the
    opposite case and deliberately keeps its lights fixed in the world.
    """
    direction = np.asarray(direction_camera, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        raise ValueError("Light direction must be a nonzero camera-frame vector")
    return grid.rotations() @ (direction / norm)


def appearance_for(count, materials, colour=0.55, kd=0.8, light_direction=(-1.0, -1.0, -1.0),
                   ambient=0.2, directional=0.8):
    """An Appearance with one explicit value per field; no sampling, no hidden seed.

    `colour` is either a scalar grey, one RGB triple shared by every material, or one triple per
    material. `light_direction` is one world direction shared by every view, or one per view.
    RC-R3/R4 vary these fields one at a time, so they must be stated, not drawn.
    """
    if type(count) is not int or count < 1 or type(materials) is not int or materials < 1:
        raise ValueError("Positive integer scene and material counts are required")
    values = np.asarray(colour, dtype=np.float64)
    if values.ndim == 0:
        colours = np.full((materials, 3), float(values))
    elif values.shape == (3,):
        colours = np.tile(values, (materials, 1))
    elif values.shape == (materials, 3):
        colours = values
    else:
        raise ValueError("colour must be a scalar, one RGB triple, or one triple per material")
    kd_values = np.asarray(kd, dtype=np.float64)
    if kd_values.ndim == 0:
        kd_values = np.full((materials,), float(kd_values))
    elif kd_values.shape != (materials,):
        raise ValueError("kd must be a scalar or one value per material")
    light = np.asarray(light_direction, dtype=np.float64)
    if light.shape == (3,):
        light = np.tile(light, (count, 1))
    elif light.shape != (count, 3):
        raise ValueError("light_direction must be one world direction, or one per view")
    return Appearance(np.tile(colours[None, :, :], (count, 1, 1)),
                      np.tile(kd_values[None, :], (count, 1)), light,
                      np.full(count, float(ambient)), np.full(count, float(directional)))


def array_hashes(arrays):
    return {name: hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
            for name, value in sorted(arrays.items())}


def geometry_hashes(gbuffer):
    """Hashes of exactly the six geometry buffers, for the RC-R3/R4 invariance gates."""
    return array_hashes({name: getattr(gbuffer, name).detach().cpu().numpy()
                         for name in GEOMETRY_ARRAYS})


class CharacterizationCell:
    """One arm rendered over one view grid at one resolution, shaded on demand."""

    def __init__(self, arm_name, camera: Camera, grid: ViewGrid, device="cpu", scale=None,
                 urdf_path=None, instance_offset=0):
        if not isinstance(camera, Camera) or not isinstance(grid, ViewGrid):
            raise TypeError("A cell needs a Camera and a ViewGrid")
        if type(instance_offset) is not int or not 0 <= instance_offset <= 1000000:
            raise ValueError("instance_offset must be an integer in [0, 1000000]")
        self.arm = build_arm(arm_name, scale=scale, urdf_path=urdf_path)
        self.instance_offset = instance_offset
        if instance_offset and self.arm.mesh is not None:
            # Renumbering the instance label must not move a vertex: only the face->instance table
            # changes, and the renderer derives instance_id from it.
            self.arm = replace(self.arm, mesh=replace(
                self.arm.mesh, face_instance=self.arm.mesh.face_instance + instance_offset))
        self.camera, self.grid, self.device = camera, grid, str(device)
        self.renderer = (None if self.arm.kind != "mesh"
                         else self.arm.renderer(camera, len(grid), self.device))
        self._gbuffer = None

    @property
    def surface(self):
        return AnalyticSurface() if self.arm.mesh is None else self.arm.mesh

    def gbuffer(self):
        """Render once and keep it: every shading mode in this cell must see the same geometry."""
        if self._gbuffer is None:
            buffer = self.arm.gbuffer(self.camera, self.grid, self.device, self.renderer)
            if buffer.valid.shape[0] != len(self.grid):
                raise RuntimeError("Renderer returned the wrong number of views")
            if self.instance_offset and self.arm.mesh is None:
                # The analytic arm has no face->instance table to renumber, so the label is offset
                # on the buffer itself. Misses stay -1; a renumbered miss would be a new object.
                buffer = replace(buffer, instance_id=torch.where(
                    buffer.valid, buffer.instance_id + self.instance_offset, -1))
            self._gbuffer = buffer
        return self._gbuffer

    def appearance(self, **overrides):
        return appearance_for(len(self.grid), self.surface.material_count, **overrides)

    def lit_appearance(self, **overrides):
        """The RC-R3 appearance: the same material everywhere, lit relative to each camera."""
        overrides.setdefault("light_direction", camera_relative_light(self.grid))
        return self.appearance(**overrides)

    def shade(self, appearance=None, mode="lambertian"):
        if mode not in SHADING_MODES:
            raise ValueError(f"mode must be one of {SHADING_MODES}")
        appearance = self.appearance() if appearance is None else appearance
        return shade(self.gbuffer(), self.surface, appearance, mode, BACKGROUND)

    def arrays(self, appearance=None, mode="lambertian"):
        """Host-owned copies of the shaded image and every geometry buffer."""
        gbuffer = self.gbuffer()
        rgb = self.shade(appearance, mode)
        record = {"rgb": rgb}
        record.update({name: getattr(gbuffer, name) for name in GEOMETRY_ARRAYS})
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in record.items()}

    def metrics(self):
        """Measured views only. Refused views are available from metrics_record()."""
        return self.metrics_record()[0]

    def metrics_record(self):
        from .geometry_metrics import measured_arm_metrics
        return measured_arm_metrics(self.arm, self.gbuffer(), self.camera, self.grid)

    def image_statistics(self, appearance=None, mode="lambertian"):
        from .image_statistics import image_statistics
        gbuffer = self.gbuffer()
        rgb = self.shade(appearance, mode).detach().cpu().numpy()
        valid = gbuffer.valid.detach().cpu().numpy()
        depth = gbuffer.depth_m.detach().cpu().numpy()
        rows = []
        for index in range(len(self.grid)):
            rows.append({
                "view": index,
                "silhouette": image_statistics(rgb[index], valid[index], depth[index]),
                "full_frame": image_statistics(rgb[index]),
            })
        return rows

    def description(self):
        return {"arm": self.arm.name, "arm_code": self.arm.code, "arm_kind": self.arm.kind,
                "instance_offset": self.instance_offset,
                "arm_detail": self.arm.description, "isotropic_scale": self.arm.scale,
                "circumscribed_radius_m": self.arm.circumscribed_radius_m,
                "camera": asdict(self.camera), "focal_px": self.camera.focal_px,
                "device": self.device, "views": self.grid.as_dict(),
                "background_rgb": list(BACKGROUND),
                "shading_modes": list(SHADING_MODES),
                "rgb": "NHWC float32 linear RGB [0,1]; misses take the background",
                "depth": "camera +Z metres; misses zero",
                "labels": "face_id/instance_id are renderer outputs, never shading inputs"}


def recoloured(appearance, colour):
    """The same lighting with a different material colour; one field changes, nothing else."""
    count, materials = appearance.base_color.shape[:2]
    values = np.asarray(colour, dtype=np.float64)
    if values.ndim == 0:
        colours = np.full((materials, 3), float(values))
    elif values.shape == (3,):
        colours = np.tile(values, (materials, 1))
    elif values.shape == (materials, 3):
        colours = values
    else:
        raise ValueError("colour must be a scalar, one RGB triple, or one triple per material")
    return replace(appearance, base_color=np.tile(colours[None, :, :], (count, 1, 1)))


def relit(appearance, light_direction=None, ambient=None, directional=None):
    """The same materials under different lighting; material colour and kd are untouched."""
    count = appearance.base_color.shape[0]
    fields = {}
    if light_direction is not None:
        fields["light_direction"] = np.tile(np.asarray(light_direction, dtype=np.float64), (count, 1))
    if ambient is not None:
        fields["ambient"] = np.full(count, float(ambient))
    if directional is not None:
        fields["directional"] = np.full(count, float(directional))
    if not fields:
        raise ValueError("relit was asked to change nothing")
    return replace(appearance, **fields)
