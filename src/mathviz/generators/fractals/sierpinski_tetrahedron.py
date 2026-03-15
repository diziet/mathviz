"""Sierpinski tetrahedron (tetrix) fractal generator.

Recursively subdivides a regular tetrahedron by replacing it with 4
half-scale tetrahedra at the corners, producing the 3D analogue of the
Sierpinski triangle. Default representation: SURFACE_SHELL.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_LEVEL = 5
_MAX_LEVEL = 8
_DEFAULT_SIZE = 1.0
_MIN_SIZE = 1e-6

# Regular tetrahedron vertices (unit edge length, centered at origin)
_UNIT_EDGE = 1.0
_SQRT2 = np.sqrt(2.0)
_BASE_VERTICES = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
], dtype=np.float64) / (2.0 * _SQRT2)


def _initial_tetrahedron(size: float) -> np.ndarray:
    """Return 4 vertices of a regular tetrahedron with given edge length."""
    return _BASE_VERTICES * size


def _subdivide(tetrahedra: np.ndarray) -> np.ndarray:
    """Subdivide each tetrahedron into 4 half-scale corner copies.

    Args:
        tetrahedra: Array of shape (N, 4, 3) — N tetrahedra, 4 vertices each.

    Returns:
        Array of shape (4*N, 4, 3).
    """
    n = len(tetrahedra)
    a, b, c, d = (
        tetrahedra[:, 0],
        tetrahedra[:, 1],
        tetrahedra[:, 2],
        tetrahedra[:, 3],
    )
    m_ab = (a + b) / 2.0
    m_ac = (a + c) / 2.0
    m_ad = (a + d) / 2.0
    m_bc = (b + c) / 2.0
    m_bd = (b + d) / 2.0
    m_cd = (c + d) / 2.0

    result = np.empty((4 * n, 4, 3), dtype=np.float64)
    result[0::4] = np.stack([a, m_ab, m_ac, m_ad], axis=1)
    result[1::4] = np.stack([b, m_ab, m_bc, m_bd], axis=1)
    result[2::4] = np.stack([c, m_ac, m_bc, m_cd], axis=1)
    result[3::4] = np.stack([d, m_ad, m_bd, m_cd], axis=1)
    return result


def _build_tetrahedra(level: int, size: float) -> np.ndarray:
    """Build all leaf tetrahedra for a Sierpinski tetrix at given level.

    Returns:
        Array of shape (4^level, 4, 3).
    """
    verts = _initial_tetrahedron(size)
    tetrahedra = verts.reshape(1, 4, 3)
    for _ in range(level):
        tetrahedra = _subdivide(tetrahedra)
    return tetrahedra


def _tetrahedra_to_mesh(tetrahedra: np.ndarray) -> Mesh:
    """Convert an array of tetrahedra into a triangle mesh.

    Each tetrahedron contributes 4 triangular faces with unshared vertices
    (12 vertices, 4 faces per tetrahedron).
    """
    n = len(tetrahedra)
    # 4 faces per tet, 3 vertices per face
    vertices = np.empty((n * 12, 3), dtype=np.float64)
    faces = np.empty((n * 4, 3), dtype=np.int64)

    a = tetrahedra[:, 0]
    b = tetrahedra[:, 1]
    c = tetrahedra[:, 2]
    d = tetrahedra[:, 3]

    # Face 0: (a, c, b) — outward from d
    vertices[0::12] = a
    vertices[1::12] = c
    vertices[2::12] = b
    # Face 1: (a, b, d) — outward from c
    vertices[3::12] = a
    vertices[4::12] = b
    vertices[5::12] = d
    # Face 2: (a, d, c) — outward from b
    vertices[6::12] = a
    vertices[7::12] = d
    vertices[8::12] = c
    # Face 3: (b, c, d) — outward from a
    vertices[9::12] = b
    vertices[10::12] = c
    vertices[11::12] = d

    base_indices = np.arange(n, dtype=np.int64) * 12
    for fi in range(4):
        offset = fi * 3
        faces[fi::4, 0] = base_indices + offset
        faces[fi::4, 1] = base_indices + offset + 1
        faces[fi::4, 2] = base_indices + offset + 2

    return Mesh(vertices=vertices, faces=faces)


def _validate_params(level: int, size: float) -> None:
    """Validate Sierpinski tetrahedron parameters."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    if level > _MAX_LEVEL:
        raise ValueError(f"level must be <= {_MAX_LEVEL}, got {level}")
    if size < _MIN_SIZE:
        raise ValueError(f"size must be >= {_MIN_SIZE}, got {size}")


@register
class SierpinskiTetrahedronGenerator(GeneratorBase):
    """Sierpinski tetrahedron — recursive tetrahedral subdivision."""

    name = "sierpinski_tetrahedron"
    category = "fractals"
    aliases = ("tetrix", "sierpinski_tetrix")
    description = (
        "Sierpinski tetrahedron fractal via recursive corner subdivision"
    )
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Sierpinski tetrahedron."""
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
        """Generate a Sierpinski tetrahedron mesh.

        Fully deterministic — seed is stored for metadata only.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        level = int(merged["level"])
        size = float(merged["size"])

        _validate_params(level, size)

        tetrahedra = _build_tetrahedra(level, size)
        mesh = _tetrahedra_to_mesh(tetrahedra)

        # Bounding box from initial tetrahedron vertices
        init_verts = _initial_tetrahedron(size)
        min_corner = tuple(float(v) for v in init_verts.min(axis=0))
        max_corner = tuple(float(v) for v in init_verts.max(axis=0))
        bbox = BoundingBox(min_corner=min_corner, max_corner=max_corner)

        logger.info(
            "Generated sierpinski_tetrahedron: level=%d, size=%.3f, "
            "tetrahedra=%d, vertices=%d, faces=%d",
            level, size, len(tetrahedra), len(mesh.vertices), len(mesh.faces),
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
