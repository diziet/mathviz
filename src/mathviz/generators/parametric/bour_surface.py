"""Bour's minimal surface parametric generator.

Bour's minimal surface interpolates between a helicoid and a catenoid.
It is parameterized by an order n that controls the surface's shape,
with n=2 giving the classic helicoid-catenoid interpolation.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_mixed_grid_faces,
    compute_padded_bounding_box,
)

logger = logging.getLogger(__name__)

_DEFAULT_N = 2
_DEFAULT_R_MAX = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 8
_MIN_N = 1
_R_EPSILON = 1e-6


def _evaluate_bour_surface(
    r: np.ndarray, theta: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Bour's surface immersion f(r, theta) -> (x, y, z)."""
    r_n = np.power(r, n)
    n_half = n / 2.0
    x = r * np.cos(theta) - r_n * np.cos(n * theta) / (2.0 * n)
    y = -r * np.sin(theta) - r_n * np.sin(n * theta) / (2.0 * n)
    z = 2.0 * np.power(r, n_half) * np.cos(n_half * theta) / n
    return x, y, z


def _validate_params(
    n: int, r_max: float, grid_resolution: int,
) -> None:
    """Validate Bour surface parameters."""
    if n < _MIN_N:
        raise ValueError(f"n must be >= {_MIN_N}, got {n}")
    if r_max <= 0:
        raise ValueError(f"r_max must be positive, got {r_max}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_bour_mesh(
    n: int, r_max: float, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for Bour's surface."""
    res = grid_resolution
    r_vals = np.linspace(_R_EPSILON, r_max, res, endpoint=True)
    theta_vals = np.linspace(0, 2.0 * np.pi, res, endpoint=False)
    rr, tt = np.meshgrid(r_vals, theta_vals, indexing="ij")

    x, y, z = _evaluate_bour_surface(rr, tt, n)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    # Open in r (radial), wrapped in theta (angular)
    faces = build_mixed_grid_faces(res, res, wrap_u=False, wrap_v=True)
    return Mesh(vertices=vertices, faces=faces)


@register
class BourSurfaceGenerator(GeneratorBase):
    """Parametric Bour's minimal surface — helicoid-catenoid interpolation."""

    name = "bour_surface"
    category = "parametric"
    aliases = ("bour",)
    description = "Minimal surface interpolating between helicoid and catenoid"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Bour's surface."""
        return {
            "n": _DEFAULT_N,
            "r_max": _DEFAULT_R_MAX,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Bour's surface mesh.

        Surface is analytically deterministic; seed is stored for
        metadata provenance only (no RNG used).
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        n = int(merged["n"])
        r_max = float(merged["r_max"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(n, r_max, grid_resolution)

        mesh = _generate_bour_mesh(n, r_max, grid_resolution)
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated bour_surface: n=%d, r_max=%.3f, grid=%d, "
            "vertices=%d, faces=%d",
            n, r_max, grid_resolution,
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

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for parameters."""
        return {
            "n": {"min": 1, "max": 10, "step": 1},
            "r_max": {"min": 0.1, "max": 3.0, "step": 0.1},
        }

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for Bour's surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
