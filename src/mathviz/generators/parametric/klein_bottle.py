"""Klein bottle parametric surface generator.

Uses the figure-8 immersion in R³, which self-intersects but produces
visually striking geometry. The surface wraps periodically in both u and v.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    DEFAULT_SEPARATION_EPSILON,
    build_klein_wrapped_faces,
    separate_coincident_vertices,
    validate_separation_epsilon,
)

logger = logging.getLogger(__name__)

_DEFAULT_SCALE = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_klein_bottle(
    u: np.ndarray, v: np.ndarray, scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the figure-8 Klein bottle immersion f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid, both in [0, 2*pi).
    """
    half_u = u / 2.0
    cos_hu = np.cos(half_u)
    sin_hu = np.sin(half_u)
    sin_v = np.sin(v)
    sin_2v = np.sin(2.0 * v)

    r = scale * (2.0 + cos_hu * sin_v - sin_hu * sin_2v)
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = scale * (sin_hu * sin_v + cos_hu * sin_2v)
    return x, y, z


def _validate_params(scale: float, grid_resolution: int) -> None:
    """Validate Klein bottle parameters."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _compute_bounding_box(
    scale: float, separation_epsilon: float,
) -> BoundingBox:
    """Compute axis-aligned bounding box for the Klein bottle."""
    xy_extent = scale * 3.5 + separation_epsilon
    z_extent = scale * 1.5 + separation_epsilon
    return BoundingBox(
        min_corner=(-xy_extent, -xy_extent, -z_extent),
        max_corner=(xy_extent, xy_extent, z_extent),
    )


def _generate_klein_mesh(
    scale: float, grid_resolution: int, separation_epsilon: float,
) -> Mesh:
    """Build triangle mesh for the Klein bottle."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_klein_bottle(uu, vv, scale)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_klein_wrapped_faces(n, n)
    vertices = separate_coincident_vertices(vertices, faces, separation_epsilon)
    return Mesh(vertices=vertices, faces=faces)


@register
class KleinBottleGenerator(GeneratorBase):
    """Parametric Klein bottle surface (figure-8 immersion)."""

    name = "klein_bottle"
    category = "parametric"
    aliases = ()
    description = "Klein bottle immersion with self-intersection"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Klein bottle."""
        return {
            "scale": _DEFAULT_SCALE,
            "separation_epsilon": DEFAULT_SEPARATION_EPSILON,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Klein bottle mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        scale = float(merged["scale"])
        separation_epsilon = float(merged["separation_epsilon"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(scale, grid_resolution)
        validate_separation_epsilon(separation_epsilon)

        mesh = _generate_klein_mesh(scale, grid_resolution, separation_epsilon)
        bbox = _compute_bounding_box(scale, separation_epsilon)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated klein_bottle: scale=%.3f, grid=%d, vertices=%d, faces=%d",
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
        """Return the recommended representation for the Klein bottle."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
