"""Enneper surface parametric generator.

The Enneper surface is a classical minimal surface with self-intersections
at large parameter ranges. The mesh is open in both u and v (not watertight).
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_open_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_RANGE = 2.0
_DEFAULT_ORDER = 1
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4


def _evaluate_enneper(
    u: np.ndarray, v: np.ndarray, order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the generalized Enneper surface f(u, v) -> (x, y, z).

    For order n, the generalized Enneper minimal surface is:
      x = u - u^(2n+1)/(2n+1) + u * v^(2n)
      y = v - v^(2n+1)/(2n+1) + v * u^(2n)
      z = u^(2n) - v^(2n)
    For n=1 this reduces to the classical Enneper surface.
    """
    n = order
    exp = 2 * n + 1
    cross_exp = 2 * n
    x = u - u**exp / exp + u * v**cross_exp
    y = v - v**exp / exp + v * u**cross_exp
    z = u**cross_exp - v**cross_exp
    return x, y, z


def _validate_params(
    param_range: float, order: int, grid_resolution: int,
) -> None:
    """Validate Enneper surface parameters."""
    if param_range <= 0:
        raise ValueError(f"range must be positive, got {param_range}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _compute_bounding_box(param_range: float, order: int) -> BoundingBox:
    """Compute bounding box from mesh vertices extent estimate."""
    r = param_range
    n = order
    exp = 2 * n + 1
    cross_exp = 2 * n
    xy_extent = r + r**exp / exp + r * r**cross_exp
    z_extent = r**cross_exp
    return BoundingBox(
        min_corner=(-xy_extent, -xy_extent, -z_extent),
        max_corner=(xy_extent, xy_extent, z_extent),
    )


def _generate_enneper_mesh(
    param_range: float, order: int, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for the Enneper surface."""
    n = grid_resolution
    u_vals = np.linspace(-param_range, param_range, n)
    v_vals = np.linspace(-param_range, param_range, n)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_enneper(uu, vv, order)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_open_grid_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class EnneperSurfaceGenerator(GeneratorBase):
    """Classical Enneper minimal surface generator."""

    name = "enneper_surface"
    category = "parametric"
    aliases = ()
    description = "Enneper minimal surface with configurable range and order"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Enneper surface."""
        return {
            "range": _DEFAULT_RANGE,
            "order": _DEFAULT_ORDER,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an Enneper surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        param_range = float(merged["range"])
        order = int(merged["order"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(param_range, order, grid_resolution)

        mesh = _generate_enneper_mesh(param_range, order, grid_resolution)
        bbox = _compute_bounding_box(param_range, order)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated enneper_surface: range=%.3f, order=%d, "
            "grid=%d, vertices=%d, faces=%d",
            param_range, order, grid_resolution,
            len(mesh.vertices), len(mesh.faces),
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
        """Return the recommended representation for the Enneper surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
