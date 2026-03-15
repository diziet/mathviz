"""3D Penrose tiling generator.

Generates a 2D Penrose tiling (P3 rhombus type) via Robinson triangle
subdivision with inflate-and-subdivide, then extrudes each tile as a
prism to create a relief surface. Thick and thin rhombuses receive
different extrusion heights.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

PHI = (1.0 + np.sqrt(5.0)) / 2.0

_THICK = 0  # Half of thick rhombus (golden triangle, 36-72-72)
_THIN = 1  # Half of thin rhombus (golden gnomon, 108-36-36)

_DEFAULT_GENERATIONS = 5
_DEFAULT_TILE_HEIGHT_RATIO = 0.3
_DEFAULT_EXTENT = 5.0
_MIN_GENERATIONS = 1
_MAX_GENERATIONS = 8
_MIN_EXTENT = 0.1


def _validate_params(
    generations: int, tile_height_ratio: float, extent: float,
) -> None:
    """Validate Penrose tiling parameters."""
    if generations < _MIN_GENERATIONS:
        raise ValueError(
            f"generations must be >= {_MIN_GENERATIONS}, got {generations}"
        )
    if generations > _MAX_GENERATIONS:
        raise ValueError(
            f"generations must be <= {_MAX_GENERATIONS}, got {generations}"
        )
    if tile_height_ratio < 0:
        raise ValueError(
            f"tile_height_ratio must be >= 0, got {tile_height_ratio}"
        )
    if extent < _MIN_EXTENT:
        raise ValueError(f"extent must be >= {_MIN_EXTENT}, got {extent}")


def _generate_initial_sun(
    rotation: float,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Create initial sun pattern: 10 golden triangles in a decagon."""
    triangles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(10):
        angle_b = rotation + 2.0 * np.pi * i / 10
        angle_c = rotation + 2.0 * np.pi * (i + 1) / 10
        b_pt = np.array([np.cos(angle_b), np.sin(angle_b)])
        c_pt = np.array([np.cos(angle_c), np.sin(angle_c)])
        if i % 2 == 0:
            triangles.append((_THICK, np.array([0.0, 0.0]), b_pt, c_pt))
        else:
            triangles.append((_THICK, np.array([0.0, 0.0]), c_pt, b_pt))
    return triangles


def _subdivide_once(
    triangles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Apply one level of Robinson triangle subdivision."""
    result: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for tri_type, a_pt, b_pt, c_pt in triangles:
        if tri_type == _THICK:
            p_pt = a_pt + (b_pt - a_pt) / PHI
            result.append((_THICK, c_pt, p_pt, b_pt))
            result.append((_THIN, p_pt, c_pt, a_pt))
        else:
            q_pt = b_pt + (a_pt - b_pt) / PHI
            r_pt = b_pt + (c_pt - b_pt) / PHI
            result.append((_THIN, r_pt, c_pt, a_pt))
            result.append((_THIN, q_pt, r_pt, b_pt))
            result.append((_THICK, r_pt, q_pt, a_pt))
    return result


def _clip_to_extent(
    triangles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    extent: float,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Keep only triangles whose centroid falls within the extent box."""
    clipped: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for tri_type, a_pt, b_pt, c_pt in triangles:
        centroid = (a_pt + b_pt + c_pt) / 3.0
        if abs(centroid[0]) <= extent and abs(centroid[1]) <= extent:
            clipped.append((tri_type, a_pt, b_pt, c_pt))
    return clipped


def _generate_penrose_tiling(
    generations: int, extent: float, rotation: float,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate 2D Penrose tiling via inflate-and-subdivide."""
    triangles = _generate_initial_sun(rotation)
    for _ in range(generations):
        triangles = [
            (t, a * PHI, b * PHI, c * PHI)
            for t, a, b, c in triangles
        ]
        triangles = _subdivide_once(triangles)
    return _clip_to_extent(triangles, extent)


def _extrude_tiles(
    triangles: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    tile_height_ratio: float,
) -> Mesh:
    """Extrude 2D triangles into 3D prisms with type-dependent height."""
    num_tris = len(triangles)
    vertices = np.zeros((num_tris * 6, 3), dtype=np.float64)
    faces = np.zeros((num_tris * 8, 3), dtype=np.int64)

    thick_height = 1.0
    thin_height = tile_height_ratio

    for i, (tri_type, a_pt, b_pt, c_pt) in enumerate(triangles):
        h = thick_height if tri_type == _THICK else thin_height
        b_idx = i * 6

        vertices[b_idx + 0] = [a_pt[0], a_pt[1], 0.0]
        vertices[b_idx + 1] = [b_pt[0], b_pt[1], 0.0]
        vertices[b_idx + 2] = [c_pt[0], c_pt[1], 0.0]
        vertices[b_idx + 3] = [a_pt[0], a_pt[1], h]
        vertices[b_idx + 4] = [b_pt[0], b_pt[1], h]
        vertices[b_idx + 5] = [c_pt[0], c_pt[1], h]

        f_idx = i * 8
        b = b_idx
        faces[f_idx + 0] = [b, b + 2, b + 1]
        faces[f_idx + 1] = [b + 3, b + 4, b + 5]
        faces[f_idx + 2] = [b, b + 1, b + 4]
        faces[f_idx + 3] = [b, b + 4, b + 3]
        faces[f_idx + 4] = [b + 1, b + 2, b + 5]
        faces[f_idx + 5] = [b + 1, b + 5, b + 4]
        faces[f_idx + 6] = [b + 2, b, b + 3]
        faces[f_idx + 7] = [b + 2, b + 3, b + 5]

    return Mesh(vertices=vertices, faces=faces)


@register
class PenroseTiling3D(GeneratorBase):
    """3D Penrose tiling — aperiodic rhombus tiling extruded as relief."""

    name = "penrose_3d"
    category = "procedural"
    aliases = ("penrose_tiling", "penrose")
    description = "Aperiodic Penrose P3 rhombus tiling extruded as 3D relief"
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Penrose 3D tiling."""
        return {
            "generations": _DEFAULT_GENERATIONS,
            "tile_height_ratio": _DEFAULT_TILE_HEIGHT_RATIO,
            "extent": _DEFAULT_EXTENT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 3D Penrose tiling relief mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        generations = int(merged["generations"])
        tile_height_ratio = float(merged["tile_height_ratio"])
        extent = float(merged["extent"])

        _validate_params(generations, tile_height_ratio, extent)

        rng = default_rng(seed)
        rotation = float(rng.uniform(0, 2.0 * np.pi / 10))

        triangles = _generate_penrose_tiling(generations, extent, rotation)
        mesh = _extrude_tiles(triangles, tile_height_ratio)
        bbox = BoundingBox.from_points(mesh.vertices)

        num_thick = sum(1 for t, _, _, _ in triangles if t == _THICK)
        num_thin = len(triangles) - num_thick

        logger.info(
            "Generated penrose_3d: generations=%d, tiles=%d "
            "(thick=%d, thin=%d), extent=%.1f",
            generations, len(triangles), num_thick, num_thin, extent,
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
