"""Boy surface parametric generator.

Boy's surface is an immersion of the real projective plane in R³. It has
self-intersections (a triple point) but no boundary. This implementation
uses the Apéry parameterization.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_open_grid_faces,
    separate_coincident_vertices,
)

logger = logging.getLogger(__name__)

_DEFAULT_SCALE = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_DEFAULT_SEPARATION_EPSILON = 0.005
_MIN_GRID_RESOLUTION = 4
_SQRT2 = np.sqrt(2.0)


def _evaluate_boy_surface(
    u: np.ndarray, v: np.ndarray, scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Boy's surface using Apéry's parameterization.

    u in [0, pi], v in [0, pi]. The denominator can approach zero near
    certain parameter values, so we clamp it for numerical stability.
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)
    cos_2u = np.cos(2.0 * u)
    sin_2u = np.sin(2.0 * u)
    sin_2v = np.sin(2.0 * v)
    cos2_v = cos_v * cos_v
    sin_3u = np.sin(3.0 * u)

    denom = 2.0 - _SQRT2 * sin_3u * sin_2v
    # Clamp denominator preserving sign to avoid division by zero
    sign = np.where(denom >= 0, 1.0, -1.0)
    denom = np.where(np.abs(denom) < 1e-10, sign * 1e-10, denom)

    x = scale * (_SQRT2 * cos2_v * cos_2u + cos_u * sin_2v) / denom
    y = scale * (_SQRT2 * cos2_v * sin_2u - sin_u * sin_2v) / denom
    z = scale * 3.0 * cos2_v / denom
    return x, y, z


def _validate_params(scale: float, grid_resolution: int) -> None:
    """Validate Boy surface parameters."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_boy_mesh(
    scale: float, grid_resolution: int, separation_epsilon: float,
) -> Mesh:
    """Build triangle mesh for Boy's surface."""
    n = grid_resolution
    u_vals = np.linspace(0, np.pi, n)
    v_vals = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_boy_surface(uu, vv, scale)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_open_grid_faces(n, n)
    vertices = separate_coincident_vertices(vertices, faces, separation_epsilon)
    return Mesh(vertices=vertices, faces=faces)


def _compute_bounding_box(
    scale: float, separation_epsilon: float,
) -> BoundingBox:
    """Compute conservative bounding box for Boy's surface."""
    extent = scale * 4.0
    pad = separation_epsilon
    return BoundingBox(
        min_corner=(-extent - pad, -extent - pad, -pad),
        max_corner=(extent + pad, extent + pad, extent * 2.0 + pad),
    )


@register
class BoySurfaceGenerator(GeneratorBase):
    """Boy surface immersion of the real projective plane."""

    name = "boy_surface"
    category = "parametric"
    aliases = ()
    description = "Boy surface (RP² immersion) with triple self-intersection"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Boy's surface."""
        return {
            "scale": _DEFAULT_SCALE,
            "separation_epsilon": _DEFAULT_SEPARATION_EPSILON,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Boy surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        scale = float(merged["scale"])
        separation_epsilon = float(merged["separation_epsilon"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(scale, grid_resolution)

        mesh = _generate_boy_mesh(scale, grid_resolution, separation_epsilon)
        bbox = _compute_bounding_box(scale, separation_epsilon)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated boy_surface: scale=%.3f, grid=%d, vertices=%d, faces=%d",
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
        """Return the recommended representation for Boy's surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
