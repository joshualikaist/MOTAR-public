"""Turn a URDF's visual geometry into a MeshScene.

The default backend is urdfpy, which owns URDF semantics: the link tree, forward kinematics,
materials, mesh files and their scaling. Isolation in this prototype means not importing
aerial_gym, the simulator, a task, a detector or a policy. A general-purpose URDF parser is not
part of the system under study, and hand-rolling one only moves well-tested behaviour into code
nobody has tested.

A second backend parses the same file with the standard library alone. It is not a fallback: it
exists so the two can be compared, and compare_backends turns any disagreement into a listed
discrepancy rather than a silent difference. On the five assets of the appearance path they agree
exactly on link order, forward kinematics, geometry parameters and material colours.

Scope and conventions are fixed in docs/renderer_urdf_loader_v1_contract.md. URDF rgba is carried
through unconverted: a declared choice, not a claim that a display colour equals a linear one.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

from .scene import MeshScene, frozen_array, integer_array

SPHERE_SEGMENTS = 16
SPHERE_RINGS = 8
CYLINDER_SEGMENTS = 16
DEFAULT_MATERIAL_NAME = "__urdf_default__"
DEFAULT_MATERIAL_RGBA = (0.5, 0.5, 0.5, 1.0)
SUPPORTED_GEOMETRY = ("box", "sphere", "cylinder")
BACKENDS = ("urdfpy", "builtin")
DEFAULT_BACKEND = "urdfpy"


class UrdfError(ValueError):
    """Every refusal in this module. Callers should not have to guess which failure occurred."""


def _floats(text, count, what):
    parts = (text or "").split()
    if len(parts) != count:
        raise UrdfError(f"{what} needs {count} numbers, got {len(parts)}: {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError as error:
        raise UrdfError(f"{what} is not numeric: {text!r}") from error
    if not all(math.isfinite(v) for v in values):
        raise UrdfError(f"{what} is not finite: {text!r}")
    return values


def rotation_from_rpy(roll, pitch, yaw):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]], dtype=np.float64)


def origin_transform(element):
    """A URDF <origin> as a 4x4 homogeneous transform; a missing origin is the identity."""
    if element is None:
        return np.eye(4)
    node = element if element.tag == "origin" else element.find("origin")
    if node is None:
        return np.eye(4)
    transform = np.eye(4)
    transform[:3, :3] = rotation_from_rpy(*_floats(node.get("rpy", "0 0 0"), 3, "origin rpy"))
    transform[:3, 3] = _floats(node.get("xyz", "0 0 0"), 3, "origin xyz")
    return transform


def _quad(indices, a, b, c, d):
    """Two triangles for a planar quad whose corners run counter-clockwise seen from outside."""
    indices.append((a, b, c))
    indices.append((a, c, d))


def box_mesh(size):
    sx, sy, sz = (v / 2.0 for v in size)
    points = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
              (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
    faces = []
    _quad(faces, 0, 3, 2, 1)   # -Z
    _quad(faces, 4, 5, 6, 7)   # +Z
    _quad(faces, 0, 1, 5, 4)   # -Y
    _quad(faces, 2, 3, 7, 6)   # +Y
    _quad(faces, 0, 4, 7, 3)   # -X
    _quad(faces, 1, 2, 6, 5)   # +X
    return np.array(points, dtype=np.float64), faces


def sphere_mesh(radius, segments=SPHERE_SEGMENTS, rings=SPHERE_RINGS):
    """A UV sphere with triangle fans at the poles, so no degenerate triangle is emitted."""
    points = [(0.0, 0.0, radius)]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for segment in range(segments):
            azimuth = 2.0 * math.pi * segment / segments
            points.append((radius * math.sin(polar) * math.cos(azimuth),
                           radius * math.sin(polar) * math.sin(azimuth),
                           radius * math.cos(polar)))
    south = len(points)
    points.append((0.0, 0.0, -radius))
    index = lambda ring, segment: 1 + (ring - 1) * segments + segment % segments
    faces = [(0, index(1, s), index(1, s + 1)) for s in range(segments)]
    for ring in range(1, rings - 1):
        for s in range(segments):
            _quad(faces, index(ring, s), index(ring + 1, s),
                  index(ring + 1, s + 1), index(ring, s + 1))
    faces += [(south, index(rings - 1, s + 1), index(rings - 1, s)) for s in range(segments)]
    return np.array(points, dtype=np.float64), faces


def cylinder_mesh(radius, length, segments=CYLINDER_SEGMENTS):
    """URDF convention: the axis is +Z and the origin is the centre of the cylinder."""
    half = length / 2.0
    ring = [(radius * math.cos(2.0 * math.pi * s / segments),
             radius * math.sin(2.0 * math.pi * s / segments)) for s in range(segments)]
    points = [(x, y, half) for x, y in ring] + [(x, y, -half) for x, y in ring]
    top, bottom = len(points), len(points) + 1
    points += [(0.0, 0.0, half), (0.0, 0.0, -half)]
    faces = []
    for s in range(segments):
        nxt = (s + 1) % segments
        _quad(faces, s, segments + s, segments + nxt, nxt)
        faces.append((top, s, nxt))
        faces.append((bottom, segments + nxt, segments + s))
    return np.array(points, dtype=np.float64), faces


def geometry_mesh(node):
    children = [child for child in node if child.tag in ("box", "sphere", "cylinder", "mesh")]
    if len(children) != 1:
        tags = [child.tag for child in node]
        raise UrdfError(f"<geometry> needs exactly one supported shape, found {tags}")
    shape = children[0]
    if shape.tag == "mesh":
        raise UrdfError("<mesh> is out of scope for URDF loader v1; see the contract document")
    if shape.tag == "box":
        size = _floats(shape.get("size"), 3, "box size")
        if min(size) <= 0.0:
            raise UrdfError(f"box size must be positive: {size}")
        return box_mesh(size), ("box", {"size": size})
    if shape.tag == "sphere":
        radius = _floats(shape.get("radius"), 1, "sphere radius")[0]
        if radius <= 0.0:
            raise UrdfError(f"sphere radius must be positive: {radius}")
        return sphere_mesh(radius), ("sphere", {"radius": radius})
    radius = _floats(shape.get("radius"), 1, "cylinder radius")[0]
    length = _floats(shape.get("length"), 1, "cylinder length")[0]
    if radius <= 0.0 or length <= 0.0:
        raise UrdfError(f"cylinder radius and length must be positive: {radius}, {length}")
    return cylinder_mesh(radius, length), ("cylinder", {"radius": radius, "length": length})


def link_world_transforms(links, joints):
    """Chain fixed-joint origins from the single root. Refuses cycles, orphans and extra roots."""
    children = {}
    for joint in joints:
        parent, child = joint["parent"], joint["child"]
        for name in (parent, child):
            if name not in links:
                raise UrdfError(f"joint {joint['name']!r} refers to unknown link {name!r}")
        if child in children:
            raise UrdfError(f"link {child!r} has more than one parent joint")
        children[child] = joint
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise UrdfError(f"exactly one root link is required, found {sorted(roots)}")
    transforms, order = {roots[0]: np.eye(4)}, [roots[0]]
    remaining = [joint for joint in joints]
    while remaining:
        progressed = [joint for joint in remaining if joint["parent"] in transforms]
        if not progressed:
            unreached = sorted(joint["child"] for joint in remaining)
            raise UrdfError(f"links unreachable from the root (cycle or split tree): {unreached}")
        for joint in progressed:
            transforms[joint["child"]] = transforms[joint["parent"]] @ joint["origin"]
            order.append(joint["child"])
            remaining.remove(joint)
    return transforms, order


@dataclass(frozen=True)
class UrdfAsset:
    mesh: MeshScene
    robot_name: str
    link_names: tuple
    material_names: tuple
    material_rgba: np.ndarray
    shapes: tuple
    source_path: str
    source_sha256: str
    default_material_visuals: int
    backend: str = DEFAULT_BACKEND
    library_versions: tuple = ()
    link_transforms: np.ndarray = None      # [L,4,4] world transform, aligned with link_names

    def __post_init__(self):
        object.__setattr__(self, "material_rgba", frozen_array(self.material_rgba, np.float64))
        if self.link_transforms is None:
            raise UrdfError("Link world transforms are required; they are what the backends compare")
        object.__setattr__(self, "link_transforms", frozen_array(self.link_transforms, np.float64))
        if self.link_transforms.shape != (len(self.link_names), 4, 4):
            raise UrdfError("One 4x4 world transform per link is required")
        if self.material_rgba.shape != (len(self.material_names), 4):
            raise UrdfError("One rgba row per material is required")
        if self.mesh.material_count > len(self.material_names):
            raise UrdfError("A face references a material that was never declared")

    @property
    def face_link(self):
        return self.mesh.face_instance

    def as_dict(self):
        return {"backend": self.backend, "library_versions": dict(self.library_versions),
                "robot_name": self.robot_name, "link_names": list(self.link_names),
                "material_names": list(self.material_names),
                "material_rgba": self.material_rgba.tolist(),
                "shapes": [{"link": link, "kind": kind, "parameters": parameters,
                            "triangles": count} for link, kind, parameters, count in self.shapes],
                "link_transforms": self.link_transforms.tolist(),
                "source_path": self.source_path, "source_sha256": self.source_sha256,
                "default_material_visuals": self.default_material_visuals,
                "tessellation": {"sphere_segments": SPHERE_SEGMENTS, "sphere_rings": SPHERE_RINGS,
                                 "cylinder_segments": CYLINDER_SEGMENTS},
                "colour_note": "URDF rgba carried through unconverted; alpha ignored",
                "triangles": int(len(self.mesh.triangles)),
                "links_with_visual_geometry": int(len(set(self.mesh.face_instance.tolist())))}


def load_urdf_asset(path, backend=DEFAULT_BACKEND):
    """Read one URDF file into a MeshScene using the named backend."""
    if backend not in BACKENDS:
        raise UrdfError(f"backend must be one of {BACKENDS}, got {backend!r}")
    return (load_urdf_asset_urdfpy if backend == "urdfpy" else load_urdf_asset_builtin)(path)


def load_urdf_asset_urdfpy(path):
    """Library-backed load. urdfpy owns the link tree, kinematics, materials and mesh files."""
    import trimesh
    import urdfpy
    path = Path(path)
    raw = path.read_bytes()
    try:
        robot = urdfpy.URDF.load(str(path))
    except Exception as error:                      # urdfpy raises many types for bad input
        raise UrdfError(f"urdfpy could not load {path}: {type(error).__name__}: {error}") from error
    # urdfpy accepts an articulated robot and silently evaluates it at the zero configuration.
    # The contract refuses non-fixed joints, so the refusal has to be made here rather than relied
    # on: without it a revolute joint that declares limits loads as a pose nobody chose.
    articulated = [(joint.name, joint.joint_type) for joint in robot.joints if joint.joint_type != "fixed"]
    if articulated:
        raise UrdfError(f"loader v1 supports fixed joints only; {path} declares {articulated}")
    kinematics = robot.link_fk()
    materials, rgba = {}, []

    def material_index(name, values):
        if name not in materials:
            materials[name] = len(rgba)
            rgba.append([float(v) for v in values])
        return materials[name]

    vertices, triangles, face_material, face_link, shapes = [], [], [], [], []
    link_names, defaulted = [], 0
    for instance, link in enumerate(robot.links):
        link_names.append(link.name)
        for visual in link.visuals:
            meshes = visual.geometry.meshes
            if not meshes:
                raise UrdfError(f"link {link.name!r} has a <visual> urdfpy produced no mesh for")
            combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            points = np.asarray(combined.vertices, dtype=np.float64)
            faces = np.asarray(combined.faces, dtype=np.int64)
            if points.size == 0 or faces.size == 0:
                raise UrdfError(f"link {link.name!r} produced empty geometry")
            material = visual.material
            if material is not None and material.color is None and material.name:
                # urdfpy resolves robot-level materials itself, so a named material that still has
                # no colour was never declared. It would otherwise become the default silently.
                raise UrdfError(f"link {link.name!r} references undeclared material {material.name!r}")
            if material is None or material.color is None:
                defaulted += 1
                index = material_index(DEFAULT_MATERIAL_NAME, DEFAULT_MATERIAL_RGBA)
            else:
                index = material_index(material.name or f"__anonymous_{len(rgba)}__", material.color)
            transform = kinematics[link] @ (np.eye(4) if visual.origin is None else np.asarray(visual.origin))
            if not np.isfinite(transform).all():
                raise UrdfError(f"link {link.name!r} has a non-finite visual transform")
            placed = points @ transform[:3, :3].T + transform[:3, 3]
            offset = len(vertices)
            vertices.extend(placed.tolist())
            triangles.extend((faces + offset).tolist())
            face_material.extend([index] * len(faces))
            face_link.extend([instance] * len(faces))
            geometry = visual.geometry
            if geometry.box is not None:
                kind, parameters = "box", {"size": [float(v) for v in geometry.box.size]}
                dimensions = parameters["size"]
            elif geometry.sphere is not None:
                kind, parameters = "sphere", {"radius": float(geometry.sphere.radius)}
                dimensions = [parameters["radius"]]
            elif geometry.cylinder is not None:
                kind, parameters = "cylinder", {"radius": float(geometry.cylinder.radius),
                                                "length": float(geometry.cylinder.length)}
                dimensions = [parameters["radius"], parameters["length"]]
            else:
                kind, parameters = "mesh", {"filename": str(getattr(geometry.mesh, "filename", ""))}
                dimensions = []
            # Check the declared dimensions rather than waiting for a degenerate triangle: the
            # message should name the shape that is wrong, not the first triangle that collapsed.
            if dimensions and min(dimensions) <= 0.0:
                raise UrdfError(f"link {link.name!r} {kind} has a non-positive dimension: {parameters}")
            shapes.append((link.name, kind, parameters, int(len(faces))))
    if not triangles:
        raise UrdfError(f"{path} declares no visual geometry")
    try:
        mesh = MeshScene(vertices, triangles, face_material, face_link)
    except ValueError as error:
        raise UrdfError(f"{path} produced geometry this renderer refuses: {error}") from error
    return UrdfAsset(mesh,
                     robot.name or path.stem, tuple(link_names), tuple(materials),
                     np.array(rgba, dtype=np.float64), tuple(shapes), str(path),
                     hashlib.sha256(raw).hexdigest(), defaulted, "urdfpy",
                     (("urdfpy", urdfpy.__version__), ("trimesh", trimesh.__version__)),
                     np.array([kinematics[link] for link in robot.links], dtype=np.float64))


def load_urdf_asset_builtin(path):
    """Standard-library parse, kept as an independent cross-check of the library backend."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as error:
        raise UrdfError(f"{path} is not well-formed XML: {error}") from error
    if root.tag != "robot":
        raise UrdfError(f"{path} root element is <{root.tag}>, expected <robot>")

    materials, rgba = {}, []

    def material_index(name, values):
        if name not in materials:
            materials[name] = len(rgba)
            rgba.append(values)
        return materials[name]

    for node in root.findall("material"):
        name = node.get("name")
        colour = node.find("color")
        if name and colour is not None:
            material_index(name, _floats(colour.get("rgba"), 4, f"material {name} rgba"))

    links = {}
    for node in root.findall("link"):
        name = node.get("name")
        if not name:
            raise UrdfError("every <link> needs a name")
        if name in links:
            raise UrdfError(f"duplicate link name {name!r}")
        links[name] = node

    joints = []
    for node in root.findall("joint"):
        kind = node.get("type")
        name = node.get("name") or "<unnamed>"
        if kind != "fixed":
            raise UrdfError(f"joint {name!r} is {kind!r}; loader v1 supports fixed joints only")
        parent, child = node.find("parent"), node.find("child")
        if parent is None or child is None:
            raise UrdfError(f"joint {name!r} needs <parent> and <child>")
        joints.append({"name": name, "parent": parent.get("link"), "child": child.get("link"),
                       "origin": origin_transform(node)})

    transforms, order = link_world_transforms(links, joints)
    vertices, triangles, face_material, face_link, shapes = [], [], [], [], []
    defaulted = 0
    for instance, link_name in enumerate(order):
        for visual in links[link_name].findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                raise UrdfError(f"link {link_name!r} has a <visual> with no <geometry>")
            (points, faces), (kind, parameters) = geometry_mesh(geometry)
            node = visual.find("material")
            if node is None:
                defaulted += 1
                index = material_index(DEFAULT_MATERIAL_NAME, list(DEFAULT_MATERIAL_RGBA))
            else:
                name = node.get("name") or ""
                colour = node.find("color")
                if colour is not None:
                    index = material_index(name or f"__anonymous_{len(rgba)}__",
                                           _floats(colour.get("rgba"), 4, f"material {name} rgba"))
                elif name in materials:
                    index = materials[name]
                else:
                    raise UrdfError(f"link {link_name!r} references undeclared material {name!r}")
            transform = transforms[link_name] @ origin_transform(visual)
            placed = points @ transform[:3, :3].T + transform[:3, 3]
            offset = len(vertices)
            vertices.extend(placed.tolist())
            triangles.extend([[offset + a, offset + b, offset + c] for a, b, c in faces])
            face_material.extend([index] * len(faces))
            face_link.extend([instance] * len(faces))
            shapes.append((link_name, kind, parameters, len(faces)))
    if not triangles:
        raise UrdfError(f"{path} declares no visual geometry this loader can use")
    return UrdfAsset(MeshScene(vertices, triangles, face_material, face_link),
                     root.get("name") or path.stem, tuple(order),
                     tuple(materials), np.array(rgba, dtype=np.float64), tuple(shapes),
                     str(path), hashlib.sha256(raw).hexdigest(), defaulted, "builtin",
                     (("tessellation", "declared in this module"),),
                     np.array([transforms[name] for name in order], dtype=np.float64))


def convex_outward_violations(mesh, asset=None):
    """Triangles winding into their own shape, judged against each link's centroid.

    ONLY VALID FOR CONVEX SHAPES. On a concave surface a face legitimately points towards the
    centroid, so this counts thousands of "violations" on a real hull that is correctly oriented:
    the shipped BlueROV mesh reports 88,721 of them while its signed volume is positive and its
    winding is almost entirely consistent. Use orientation_report for anything that is not a box,
    a sphere or a cylinder.
    """
    points = mesh.vertices[mesh.triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    centroids = points.mean(axis=1)
    violations = 0
    for instance in sorted(set(mesh.face_instance.tolist())):
        selected = mesh.face_instance == instance
        centre = mesh.vertices[np.unique(mesh.triangles[selected])].mean(axis=0)
        violations += int((np.sum(normals[selected] * (centroids[selected] - centre), axis=1) <= 0).sum())
    return violations


def open_edges(mesh):
    """Edges not shared by exactly two triangles. A closed surface has none."""
    counts = {}
    for triangle in mesh.triangles.tolist():
        for i in range(3):
            edge = (triangle[i], triangle[(i + 1) % 3])
            counts[tuple(sorted(edge))] = counts.get(tuple(sorted(edge)), 0) + 1
    return sorted(edge for edge, count in counts.items() if count != 2)


def signed_volume(mesh, instance=None):
    """Divergence-theorem volume of one link's closed surface, for a convergence check."""
    selected = slice(None) if instance is None else (mesh.face_instance == instance)
    points = mesh.vertices[mesh.triangles[selected]].astype(np.float64)
    return float(np.sum(np.einsum("ij,ij->i", points[:, 0],
                                  np.cross(points[:, 1], points[:, 2]))) / 6.0)


def compare_backends(path, position_tolerance_m=1e-9, colour_tolerance=0.0):
    """List every way the two backends disagree about the same file.

    Tessellation is expected to differ: trimesh subdivides curved surfaces more finely than the
    declared segment counts here, so triangle counts and vertex positions are not compared. What
    must agree is everything the URDF actually states: which links exist and in what order, where
    each visual is placed, which geometry it is and with what parameters, and what colour it has.
    Placement is compared through each link's world transform directly. An earlier version used
    the mean of each link's vertices, which is not a kinematic quantity, and the comment below
    records why that was wrong.
    """
    library, builtin = load_urdf_asset_urdfpy(path), load_urdf_asset_builtin(path)
    issues = []
    if list(library.link_names) != list(builtin.link_names):
        issues.append(f"link order: urdfpy {list(library.link_names)} vs builtin {list(builtin.link_names)}")
    if library.robot_name != builtin.robot_name:
        issues.append(f"robot name: {library.robot_name!r} vs {builtin.robot_name!r}")
    if library.source_sha256 != builtin.source_sha256:
        issues.append("source sha256 differs, which means the file changed between reads")

    # Compare the kinematics themselves. An earlier version compared the mean of each link's
    # vertices, which is not a kinematic quantity: a UV sphere and an icosphere inscribe the same
    # sphere but average to slightly different points, so tessellation leaked into a test that was
    # supposed to be about placement and produced nanometre "disagreements" that meant nothing.
    if list(library.link_names) == list(builtin.link_names):
        difference = np.abs(library.link_transforms - builtin.link_transforms)
        for index, name in enumerate(library.link_names):
            worst = float(difference[index].max())
            if worst > position_tolerance_m:
                issues.append(f"link {name!r} world transform differs by {worst:.3e}")

    def visible_links(asset):
        return {asset.link_names[i] for i in set(asset.mesh.face_instance.tolist())}

    left, right = visible_links(library), visible_links(builtin)
    if left != right:
        issues.append(f"links with visual geometry differ: {sorted(left ^ right)}")

    def shapes(asset):
        result = {}
        for link, kind, parameters, _ in asset.shapes:
            result.setdefault(link, []).append((kind, parameters))
        return result

    left, right = shapes(library), shapes(builtin)
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            issues.append(f"link {name!r} geometry: urdfpy {left.get(name)} vs builtin {right.get(name)}")

    for name in sorted(set(library.material_names) | set(builtin.material_names)):
        if name not in library.material_names or name not in builtin.material_names:
            issues.append(f"material {name!r} is present in only one backend")
            continue
        a = library.material_rgba[library.material_names.index(name)]
        b = builtin.material_rgba[builtin.material_names.index(name)]
        if float(np.abs(a - b).max()) > colour_tolerance:
            issues.append(f"material {name!r} rgba {a.tolist()} vs {b.tolist()}")
    return issues


def winding_inconsistent_edges(mesh):
    """Edges whose two triangles traverse them the same way, which means opposed normals.

    Unlike the convex test this is valid for any surface: on a consistently oriented mesh every
    interior edge is traversed once in each direction.
    """
    seen, inconsistent = {}, 0
    for triangle in mesh.triangles.tolist():
        for i in range(3):
            first, second = triangle[i], triangle[(i + 1) % 3]
            edge = (min(first, second), max(first, second))
            direction = 1 if first < second else -1
            if edge in seen:
                inconsistent += int(seen[edge] == direction)
            else:
                seen[edge] = direction
    return inconsistent


def orientation_report(mesh):
    """The checks that hold for any closed orientable surface, convex or not.

    Signed volume is taken PER SHELL, not over the whole mesh. A total is not sufficient once the
    mesh has more than one connected component, which every multi-link asset does: inverting one
    link keeps open_edges and winding_inconsistent_edges at zero, because a reversed shell is
    still consistently wound with itself, and merely subtracts its volume from the sum. Inverting
    the interceptor's motor_0 leaves a total of +6.88e-4 while that link sits at -1.56e-5, and an
    earlier version of this function called that outward-facing. The convex test it was meant to
    generalise caught it; this one has to as well.

    Reported rather than asserted, because a shipped mesh asset can fail these and that is a fact
    about the asset, not about this loader.
    """
    shells = sorted(set(mesh.face_instance.tolist()))
    volumes = {int(shell): signed_volume(mesh, shell) for shell in shells}
    inverted = sorted(shell for shell, volume in volumes.items() if volume <= 0.0)
    return {"triangles": int(len(mesh.triangles)),
            "open_edges": len(open_edges(mesh)),
            "winding_inconsistent_edges": winding_inconsistent_edges(mesh),
            "signed_volume_m3": signed_volume(mesh),
            "shells": len(shells),
            "shell_signed_volume_m3": volumes,
            "inverted_shells": inverted,
            "closed_and_consistently_wound_outward":
                bool(not open_edges(mesh) and not winding_inconsistent_edges(mesh)
                     and not inverted)}
