"""A slow CPU ray-triangle intersector, written to disagree with the GPU path if it is wrong.

Deliberately independent: plain Moller-Trumbore over every triangle in numpy, no BVH, no Warp,
and no shared helper with dynamic_mesh.py beyond the MeshScene container. Validating the object-
local transform against the same transform code would establish nothing.
"""
import numpy as np

from .scene import MeshScene

EPSILON = 1e-12


def quaternion_matrix(q):
    """xyzw unit quaternion to a 3x3 rotation, written out rather than imported."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def intersect(scene: MeshScene, origins, directions, far=1.0e30):
    """Nearest hit per ray, brute force. Returns (distance, normal, face index, hit mask)."""
    vertices = np.asarray(scene.vertices, dtype=np.float64)
    triangles = np.asarray(scene.triangles, dtype=np.int64)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    edge1, edge2 = b - a, c - a
    face_normal = np.cross(edge1, edge2)
    face_normal /= np.linalg.norm(face_normal, axis=1, keepdims=True)

    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    best = np.full(len(origins), far)
    best_face = np.full(len(origins), -1, dtype=np.int64)
    for index in range(len(triangles)):
        pvec = np.cross(directions, edge2[index])
        det = pvec @ edge1[index]
        parallel = np.abs(det) < EPSILON
        safe = np.where(parallel, 1.0, det)
        tvec = origins - a[index]
        u = np.einsum("ij,ij->i", tvec, pvec) / safe
        qvec = np.cross(tvec, edge1[index])
        v = np.einsum("ij,ij->i", directions, qvec) / safe
        t = (qvec @ edge2[index]) / safe
        ok = (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9) & (t < best)
        best = np.where(ok, t, best)
        best_face = np.where(ok, index, best_face)
    hit = best_face >= 0
    normal = np.where(hit[:, None], face_normal[np.clip(best_face, 0, None)], 0.0)
    return np.where(hit, best, far), normal, best_face, hit


def intersect_posed(scene: MeshScene, origins, directions, position, rotation, far=1.0e30):
    """Same query with the MESH moved into world space, which is the thing being compared to.

    The GPU path leaves the mesh alone and moves the ray. This one moves the mesh, so agreement
    between them is evidence about the transform rather than about one implementation.
    """
    matrix = quaternion_matrix(rotation)
    moved = MeshScene(np.asarray(scene.vertices, dtype=np.float64) @ matrix.T
                      + np.asarray(position, dtype=np.float64),
                      scene.triangles, scene.face_material, scene.face_instance)
    return intersect(moved, origins, directions, far)
