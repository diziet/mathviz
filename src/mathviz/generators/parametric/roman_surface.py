"""Roman (Steiner) surface parametric generator.

The Roman surface is a self-intersecting non-orientable surface with
tetrahedral symmetry. It is a projection of the real projective plane
into three-dimensional space.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_open_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_SCALE = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_roman_surface(
    u: np.ndarray, v: np.ndarray, scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Roman surface immersion f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid, both in [0, pi].
    """
    a_sq = scale * scale
    half_a_sq = a_sq / 2.0

    sin_2u = np.sin(2.0 * u)
    sin_u = np.sin(u)
    cos_u = np.cos(u)
    cos_v_sq = np.cos(v) ** 2
    sin_2v = np.sin(2.0 * v)

    x = half_a_sq * sin_2u * cos_v_sq
    y = half_a_sq * sin_u * sin_2v
    z = half_a_sq * cos_u * sin_2v
    return x, y, z


def _validate_params(scale: float, grid_resolution: int) -> None:
    """Validate Roman surface parameters."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _compute_bounding_box(scale: float) -> BoundingBox:
    """Compute axis-aligned bounding box for the Roman surface."""
    extent = (scale * scale) / 2.0 * 1.05
    return BoundingBox(
        min_corner=(-extent, -extent, -extent),
        max_corner=(extent, extent, extent),
    )


def _generate_roman_mesh(scale: float, grid_resolution: int) -> Mesh:
    """Build triangle mesh for the Roman surface."""
    n = grid_resolution
    u_vals = np.linspace(0, np.pi, n, endpoint=True)
    v_vals = np.linspace(0, np.pi, n, endpoint=True)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_roman_surface(uu, vv, scale)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_open_grid_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class RomanSurfaceGenerator(GeneratorBase):
    """Parametric Roman (Steiner) surface with tetrahedral symmetry."""

    name = "roman_surface"
    category = "parametric"
    aliases = ("steiner_surface",)
    description = "Self-intersecting non-orientable surface with tetrahedral symmetry"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Roman surface."""
        return {"scale": _DEFAULT_SCALE}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Roman surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        scale = float(merged["scale"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(scale, grid_resolution)

        mesh = _generate_roman_mesh(scale, grid_resolution)
        bbox = _compute_bounding_box(scale)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated roman_surface: scale=%.3f, grid=%d, vertices=%d, faces=%d",
            scale, grid_resolution, len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the Roman surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
