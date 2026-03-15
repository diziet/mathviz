"""Geodesic sphere generator.

Generates triangulated geodesic spheres at various subdivision frequencies,
starting from an icosahedron. Optionally produces the dual polyhedron
(Goldberg polyhedron) with pentagonal and hexagonal faces.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Mesh, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_PHI = (1.0 + np.sqrt(5.0)) / 2.0
_DEFAULT_FREQUENCY = 4
_MAX_FREQUENCY = 32
_FALSY_STRINGS = frozenset({"false", "0", "no", "off"})


def _parse_bool(value: Any) -> bool:
    """Parse a boolean value, handling string representations from CLI."""
    if isinstance(value, str):
        return value.lower() not in _FALSY_STRINGS
    return bool(value)


def _validate_params(frequency: int, radius: float) -> None:
    """Validate generator parameters."""
    if frequency < 1:
        raise ValueError(f"frequency must be >= 1, got {frequency}")
    if frequency > _MAX_FREQUENCY:
        raise ValueError(
            f"frequency must be <= {_MAX_FREQUENCY}, got {frequency}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")


def _build_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Build the base icosahedron vertices and faces."""
    verts = np.array([
        [-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
        [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
        [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
    ], dtype=np.float64)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return verts, faces


def _subdivide_and_project(
    vertices: np.ndarray, faces: np.ndarray, frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide icosahedron faces and project onto unit sphere."""
    if frequency == 1:
        return vertices, faces

    vertex_map: dict[tuple[float, float, float], int] = {}
    new_verts: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    def _get_or_add(point: np.ndarray) -> int:
        """Get index for a vertex, projecting onto unit sphere."""
        norm = np.linalg.norm(point)
        if norm > 0:
            point = point / norm
        key = (round(point[0], 10), round(point[1], 10), round(point[2], 10))
        if key in vertex_map:
            return vertex_map[key]
        idx = len(new_verts)
        vertex_map[key] = idx
        new_verts.append(point)
        return idx

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        _subdivide_face(v0, v1, v2, frequency, _get_or_add, new_faces)

    result_verts = np.array(new_verts, dtype=np.float64)
    result_faces = np.array(new_faces, dtype=np.int64)
    return result_verts, result_faces


def _subdivide_face(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    freq: int,
    get_or_add: Any,
    out_faces: list[list[int]],
) -> None:
    """Subdivide a single triangle face into freq^2 sub-triangles."""
    # Build grid of points using barycentric coordinates
    grid: list[list[int]] = []
    for i in range(freq + 1):
        row: list[int] = []
        for j in range(freq + 1 - i):
            # Barycentric interpolation
            a = i / freq
            b = j / freq
            c = 1.0 - a - b
            point = c * v0 + a * v1 + b * v2
            row.append(get_or_add(point))
        grid.append(row)

    # Connect grid points into triangles
    for i in range(freq):
        for j in range(freq - i):
            # Upward triangle
            out_faces.append([grid[i][j], grid[i + 1][j], grid[i][j + 1]])
            # Downward triangle (if exists)
            if j + 1 < freq - i:
                out_faces.append(
                    [grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]]
                )


def _build_dual(
    vertices: np.ndarray, faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the dual polyhedron (Goldberg polyhedron)."""
    # Face centroids become dual vertices
    dual_verts = vertices[faces].mean(axis=1)
    # Normalize onto unit sphere
    norms = np.linalg.norm(dual_verts, axis=1, keepdims=True)
    dual_verts = dual_verts / norms

    # Build vertex-to-face adjacency
    vert_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(faces):
        for vi in face:
            vert_faces.setdefault(int(vi), []).append(fi)

    dual_triangles: list[list[int]] = []
    for vi, adj_faces in vert_faces.items():
        if len(adj_faces) < 3:
            continue
        # Sort adjacent faces by angle around the vertex
        center = dual_verts[adj_faces].mean(axis=0)
        relative = dual_verts[adj_faces] - center
        normal = vertices[vi]
        normal = normal / np.linalg.norm(normal)
        # Build local coordinate frame
        u_axis = relative[0] - np.dot(relative[0], normal) * normal
        u_len = np.linalg.norm(u_axis)
        if u_len < 1e-12:
            continue
        u_axis = u_axis / u_len
        v_axis = np.cross(normal, u_axis)
        angles = np.arctan2(
            np.dot(relative, v_axis), np.dot(relative, u_axis)
        )
        order = np.argsort(angles)
        sorted_faces = [adj_faces[o] for o in order]
        # Fan-triangulate the dual face
        for i in range(1, len(sorted_faces) - 1):
            dual_triangles.append(
                [sorted_faces[0], sorted_faces[i], sorted_faces[i + 1]]
            )

    dual_faces = np.array(dual_triangles, dtype=np.int64)
    return dual_verts.astype(np.float64), dual_faces


def _count_dual_face_sides(
    vertices: np.ndarray, faces: np.ndarray,
) -> dict[int, int]:
    """Count faces by number of sides in the dual polyhedron."""
    vert_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(faces):
        for vi in face:
            vert_faces.setdefault(int(vi), []).append(fi)

    counts: dict[int, int] = {}
    for adj in vert_faces.values():
        n_sides = len(adj)
        counts[n_sides] = counts.get(n_sides, 0) + 1
    return counts


@register
class GeodesicSphereGenerator(GeneratorBase):
    """Geodesic sphere generator."""

    name = "geodesic_sphere"
    category = "geometry"
    description = (
        "Geodesic sphere: triangulated sphere at various subdivision "
        "frequencies, with optional dual (Goldberg) polyhedron mode"
    )
    resolution_params: dict[str, str] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "frequency": _DEFAULT_FREQUENCY,
            "radius": 1.0,
            "dual": False,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate geodesic sphere geometry."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        frequency = int(merged["frequency"])
        radius = float(merged["radius"])
        is_dual = _parse_bool(merged["dual"])
        _validate_params(frequency, radius)

        vertices, faces = _build_icosahedron()
        vertices, faces = _subdivide_and_project(vertices, faces, frequency)

        if is_dual:
            vertices, faces = _build_dual(vertices, faces)

        # Scale to desired radius
        vertices = vertices * radius

        mesh = Mesh(vertices=vertices, faces=faces)
        logger.info(
            "Generated geodesic_sphere: freq=%d, dual=%s, verts=%d, faces=%d",
            frequency, is_dual, len(vertices), len(faces),
        )
        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=BoundingBox.from_points(vertices),
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
