"""Apollonian gasket 3D generator.

Generates a 3D Apollonian gasket — recursive sphere packing where each gap
between tangent spheres is filled with the largest fitting sphere. Starts
from a Soddy configuration of 4 mutually tangent spheres inside an outer
bounding sphere. Uses Descartes' circle theorem extended to 3D to compute
new sphere curvatures and centers. Default representation: SURFACE_SHELL.

Seed controls a random rotation of the initial tetrahedron configuration,
producing visually distinct gaskets for different seeds.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

# --- Defaults and limits ---------------------------------------------------

DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_RADIUS = 0.01
DEFAULT_ICOSPHERE_SUBDIVISIONS = 1
_MAX_DEPTH_LIMIT = 8
_MIN_RADIUS_FLOOR = 1e-8

# --- Soddy configuration constants ------------------------------------------

_SQRT6 = float(np.sqrt(6.0))
_SQRT3 = float(np.sqrt(3.0))
_K_OUTER = -1.0
_K_INNER = (2.0 + _SQRT6) / 2.0
_R_INNER = 1.0 / _K_INNER
_D_INNER = 1.0 - _R_INNER  # distance from origin to inner centers

# Regular tetrahedron unit direction vectors (length 1)
_TETRA_DIRS = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
], dtype=np.float64) / _SQRT3

# Type alias: (curvature, center)
_Sphere = tuple[float, np.ndarray]


# --- Icosphere mesh ---------------------------------------------------------


def _build_unit_icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a unit icosphere with given subdivision level."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=np.float64)
    vertices = raw / np.linalg.norm(raw[0])

    faces = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int64)

    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)
    return vertices, faces


def _subdivide_icosphere(
    vertices: np.ndarray, faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4, projecting midpoints onto unit sphere."""
    edge_midpoints: dict[tuple[int, int], int] = {}
    new_verts = list(vertices)

    def _get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (vertices[i] + vertices[j]) / 2.0
        mid /= np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = _get_midpoint(a, b)
        bc = _get_midpoint(b, c)
        ca = _get_midpoint(c, a)
        new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])

    return (
        np.array(new_verts, dtype=np.float64),
        np.array(new_faces, dtype=np.int64),
    )


# --- Soddy configuration and Descartes theorem ------------------------------


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3D rotation matrix via QR decomposition."""
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _initial_config(rotation: np.ndarray) -> list[_Sphere]:
    """Return 5 mutually tangent spheres: [outer, inner0..inner3]."""
    outer: _Sphere = (_K_OUTER, np.zeros(3, dtype=np.float64))
    rotated_dirs = _TETRA_DIRS @ rotation.T
    centers = _D_INNER * rotated_dirs
    return [outer] + [(_K_INNER, centers[i].copy()) for i in range(4)]


def _descartes_curvature(group: list[_Sphere], old: _Sphere) -> float:
    """Compute new sphere curvature via 3D Descartes theorem.

    In 3D, given 5 mutually tangent spheres satisfying
    (sum k_i)^2 = 3 * sum(k_i^2), the two solutions for the 5th
    curvature satisfy k + k' = sum(k_other_4). So k_new = S4 - k_old.
    """
    return sum(s[0] for s in group) - old[0]


def _solve_center(group: list[_Sphere], k_new: float) -> np.ndarray:
    """Find center of a sphere with curvature k_new tangent to 4 spheres.

    Uses the tangency condition |c_new - c_i|^2 = (1/k_new + 1/k_i)^2
    for signed curvatures, then takes pairwise differences to obtain
    a 3x3 linear system.
    """
    ds = [1.0 / k_new + 1.0 / s[0] for s in group]
    c0 = group[0][1]
    norm_c0_sq = float(np.dot(c0, c0))
    d0_sq = ds[0] ** 2

    a_mat = np.empty((3, 3), dtype=np.float64)
    b_vec = np.empty(3, dtype=np.float64)

    for j in range(3):
        ci = group[j + 1][1]
        a_mat[j] = 2.0 * (ci - c0)
        b_vec[j] = float(np.dot(ci, ci)) - norm_c0_sq - ds[j + 1] ** 2 + d0_sq

    return np.linalg.solve(a_mat, b_vec)


def _descartes_new(group: list[_Sphere], old: _Sphere) -> _Sphere:
    """Compute the new sphere tangent to 4 given spheres via 3D Descartes."""
    k_new = _descartes_curvature(group, old)
    if abs(k_new) < 1e-12:
        return (0.0, np.zeros(3, dtype=np.float64))
    try:
        c_new = _solve_center(group, k_new)
    except np.linalg.LinAlgError:
        return (0.0, np.zeros(3, dtype=np.float64))
    return (k_new, c_new)


def _fill_gasket(
    initial: list[_Sphere],
    max_depth: int,
    min_radius: float,
) -> list[_Sphere]:
    """Recursively fill Apollonian gasket gaps via iterative DFS."""
    result: list[_Sphere] = []
    # Stack items: (group_of_4, old_sphere, depth)
    stack: list[tuple[list[_Sphere], _Sphere, int]] = []

    for i in range(5):
        group = [initial[j] for j in range(5) if j != i]
        stack.append((group, initial[i], 0))

    while stack:
        group, old, depth = stack.pop()
        if depth >= max_depth:
            continue

        new = _descartes_new(group, old)
        k_new = new[0]

        if k_new <= 0:
            continue

        r_new = 1.0 / k_new
        if r_new < min_radius:
            continue

        result.append(new)

        for i in range(4):
            sub_group = list(group)
            sub_old = group[i]
            sub_group[i] = new
            stack.append((sub_group, sub_old, depth + 1))

    return result


# --- Mesh assembly -----------------------------------------------------------


def _spheres_to_mesh(
    spheres: list[_Sphere], subdivisions: int,
) -> Mesh:
    """Convert spheres to a combined triangle mesh of icospheres."""
    unit_verts, unit_faces = _build_unit_icosphere(subdivisions)
    nv = len(unit_verts)
    nf = len(unit_faces)
    n_spheres = len(spheres)

    all_verts = np.empty((n_spheres * nv, 3), dtype=np.float64)
    all_faces = np.empty((n_spheres * nf, 3), dtype=np.int64)

    for i, (k, center) in enumerate(spheres):
        r = 1.0 / k
        v_off = i * nv
        all_verts[v_off:v_off + nv] = unit_verts * r + center
        all_faces[i * nf:(i + 1) * nf] = unit_faces + v_off

    return Mesh(vertices=all_verts, faces=all_faces)


# --- Validation --------------------------------------------------------------


def _validate_params(max_depth: int, min_radius: float) -> None:
    """Validate Apollonian gasket parameters."""
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if max_depth > _MAX_DEPTH_LIMIT:
        raise ValueError(
            f"max_depth must be <= {_MAX_DEPTH_LIMIT}, got {max_depth}"
        )
    if min_radius < _MIN_RADIUS_FLOOR:
        raise ValueError(
            f"min_radius must be >= {_MIN_RADIUS_FLOOR}, got {min_radius}"
        )
    if min_radius >= _R_INNER:
        raise ValueError(
            f"min_radius must be < {_R_INNER:.6f} (inner sphere radius), "
            f"got {min_radius}"
        )


# --- Generator ---------------------------------------------------------------


@register
class Apollonian3DGenerator(GeneratorBase):
    """Apollonian gasket 3D — recursive tangent sphere packing."""

    name = "apollonian_3d"
    category = "fractals"
    aliases = ("apollonian_gasket_3d",)
    description = "3D Apollonian gasket via recursive Soddy sphere packing"
    resolution_params = {
        "icosphere_subdivisions": "Subdivision level for each sphere mesh",
    }
    _resolution_defaults = {
        "icosphere_subdivisions": DEFAULT_ICOSPHERE_SUBDIVISIONS,
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Apollonian gasket."""
        return {
            "max_depth": DEFAULT_MAX_DEPTH,
            "min_radius": DEFAULT_MIN_RADIUS,
        }

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for parameters."""
        return {
            "max_depth": {"min": 0, "max": _MAX_DEPTH_LIMIT, "step": 1},
            "min_radius": {"min": 0.001, "max": 0.1, "step": 0.005},
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 3D Apollonian gasket mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        max_depth = int(merged["max_depth"])
        min_radius = float(merged["min_radius"])
        icosphere_subdivisions = int(
            resolution_kwargs.get(
                "icosphere_subdivisions", DEFAULT_ICOSPHERE_SUBDIVISIONS,
            )
        )

        _validate_params(max_depth, min_radius)
        merged["icosphere_subdivisions"] = icosphere_subdivisions

        rng = np.random.default_rng(seed)
        rotation = _random_rotation_matrix(rng)
        initial = _initial_config(rotation)

        inner_spheres = list(initial[1:])  # 4 initial inner spheres
        filled = _fill_gasket(initial, max_depth, min_radius)
        all_spheres = inner_spheres + filled

        mesh = _spheres_to_mesh(all_spheres, icosphere_subdivisions)

        centers = np.array([s[1] for s in all_spheres], dtype=np.float64)
        radii = np.array([1.0 / s[0] for s in all_spheres], dtype=np.float64)
        merged["_sphere_centers"] = centers
        merged["_sphere_radii"] = radii

        bbox = BoundingBox.from_points(mesh.vertices)

        logger.info(
            "Generated apollonian_3d: max_depth=%d, min_radius=%.4f, "
            "spheres=%d, vertices=%d, faces=%d",
            max_depth, min_radius, len(all_spheres),
            len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
