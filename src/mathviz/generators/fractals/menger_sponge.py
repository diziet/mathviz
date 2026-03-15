"""Menger sponge fractal generator.

Recursively subdivides a cube into a 3x3x3 grid and removes the center
of each face plus the center cube (7 removals per level), keeping 20
sub-cubes. Generates a triangle mesh with internal faces culled for
efficiency. Default representation: SURFACE_SHELL.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_LEVEL = 3
_MAX_LEVEL = 4
_DEFAULT_SIZE = 1.0
_MIN_SIZE = 1e-6

# Sub-cube offsets removed at each recursion step (center of each face + center)
_REMOVED_OFFSETS: frozenset[tuple[int, int, int]] = frozenset([
    (1, 1, 0), (1, 1, 2),  # z-faces center
    (1, 0, 1), (1, 2, 1),  # y-faces center
    (0, 1, 1), (2, 1, 1),  # x-faces center
    (1, 1, 1),              # cube center
])

# The 20 kept offsets (precomputed)
_KEPT_OFFSETS: list[tuple[int, int, int]] = [
    (x, y, z)
    for x in range(3) for y in range(3) for z in range(3)
    if (x, y, z) not in _REMOVED_OFFSETS
]


def _compute_cube_centers(level: int, size: float) -> np.ndarray:
    """Compute centers of all sub-cubes at the given recursion level."""
    if level == 0:
        return np.array([[0.0, 0.0, 0.0]])

    half = size / 2.0
    sub_size = size / 3.0
    offset_base = -half + sub_size / 2.0

    # Start with level-0 center
    centers = np.array([[0.0, 0.0, 0.0]])

    for _ in range(level):
        new_centers = []
        for cx, cy, cz in centers:
            for ox, oy, oz in _KEPT_OFFSETS:
                nx = cx + offset_base + ox * sub_size
                ny = cy + offset_base + oy * sub_size
                nz = cz + offset_base + oz * sub_size
                new_centers.append((nx, ny, nz))
        centers = np.array(new_centers)
        # Next level operates on smaller cubes
        sub_size /= 3.0
        offset_base = -sub_size * 3.0 / 2.0 + sub_size / 2.0

    return centers


def _build_cube_set(level: int, size: float) -> set[tuple[int, int, int]]:
    """Build set of occupied grid positions at given level.

    Returns integer grid coordinates at the finest subdivision level.
    Grid side length is 3**level.
    """
    if level == 0:
        return {(0, 0, 0)}

    occupied: set[tuple[int, int, int]] = {(0, 0, 0)}
    for _ in range(level):
        new_occupied: set[tuple[int, int, int]] = set()
        for bx, by, bz in occupied:
            for ox, oy, oz in _KEPT_OFFSETS:
                new_occupied.add((bx * 3 + ox, by * 3 + oy, bz * 3 + oz))
        occupied = new_occupied
    return occupied


def _build_menger_mesh(level: int, size: float) -> Mesh:
    """Build triangle mesh for a Menger sponge at the given level."""
    occupied = _build_cube_set(level, size)
    grid_side = 3 ** level
    cube_size = size / grid_side

    # Six face directions: (axis, sign) -> normal direction
    # For each cube face, check if the neighbor in that direction exists
    face_normals = [
        (0, -1), (0, 1),   # -x, +x
        (1, -1), (1, 1),   # -y, +y
        (2, -1), (2, 1),   # -z, +z
    ]

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    half_size = size / 2.0

    for gx, gy, gz in occupied:
        for axis, sign in face_normals:
            # Check neighbor
            neighbor = [gx, gy, gz]
            neighbor[axis] += sign
            nx, ny, nz = neighbor

            # If neighbor is occupied, skip this face (internal)
            if (nx, ny, nz) in occupied:
                continue

            # Build a quad face (2 triangles) for this exposed face
            center = np.array([
                -half_size + (gx + 0.5) * cube_size,
                -half_size + (gy + 0.5) * cube_size,
                -half_size + (gz + 0.5) * cube_size,
            ])

            quad_verts = _make_face_quad(center, cube_size, axis, sign)
            all_vertices.append(quad_verts)

            base = vertex_offset
            all_faces.append(np.array([
                [base, base + 1, base + 2],
                [base, base + 2, base + 3],
            ], dtype=np.int64))
            vertex_offset += 4

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float64)
    faces = np.concatenate(all_faces, axis=0).astype(np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_face_quad(
    center: np.ndarray, cube_size: float, axis: int, sign: int,
) -> np.ndarray:
    """Create 4 vertices for a quad face on a cube."""
    half = cube_size / 2.0
    # Two tangent axes
    ax1 = (axis + 1) % 3
    ax2 = (axis + 2) % 3

    corners = np.empty((4, 3), dtype=np.float64)
    for i, (s1, s2) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        corners[i] = center.copy()
        corners[i, axis] += sign * half
        corners[i, ax1] += s1 * half
        corners[i, ax2] += s2 * half
    return corners


def _validate_params(level: int, size: float) -> None:
    """Validate Menger sponge parameters."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    if level > _MAX_LEVEL:
        raise ValueError(f"level must be <= {_MAX_LEVEL}, got {level}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")


@register
class MengerSpongeGenerator(GeneratorBase):
    """Menger sponge fractal — recursive cube subdivision."""

    name = "menger_sponge"
    category = "fractals"
    aliases = ("menger",)
    description = "Menger sponge fractal via recursive cube subdivision"
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Menger sponge."""
        return {
            "level": _DEFAULT_LEVEL,
            "size": _DEFAULT_SIZE,
        }

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for parameters."""
        return {
            "level": {"min": 0, "max": _MAX_LEVEL, "step": 1},
            "size": {"min": 0.1, "max": 5.0, "step": 0.1},
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Menger sponge mesh.

        Fully deterministic — seed is stored for metadata only.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        level = int(merged["level"])
        size = float(merged["size"])

        _validate_params(level, size)

        mesh = _build_menger_mesh(level, size)

        half = size / 2.0
        bbox = BoundingBox(
            min_corner=(-half, -half, -half),
            max_corner=(half, half, half),
        )

        logger.info(
            "Generated menger_sponge: level=%d, size=%.3f, "
            "vertices=%d, faces=%d",
            level, size, len(mesh.vertices), len(mesh.faces),
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
