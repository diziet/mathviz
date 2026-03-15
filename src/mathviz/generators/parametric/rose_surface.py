"""Rose surface parametric generator.

A rhodonea (rose) curve extended into 3D by modulating radius on a sphere:
r(θ, φ) = cos(k₁·θ) · cos(k₂·φ), producing flower-like petal patterns.
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

_DEFAULT_K1 = 3
_DEFAULT_K2 = 2
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4
_MIN_K = 1


def _evaluate_rose_surface(
    theta: np.ndarray,
    phi: np.ndarray,
    k1: int,
    k2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the rose surface f(θ, φ) -> (x, y, z).

    Maps spherical coordinates with a rose-modulated radius:
    r = cos(k1·θ) · cos(k2·φ), then converts to Cartesian.
    """
    r = np.cos(k1 * theta) * np.cos(k2 * phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def _build_rose_surface_mesh(k1: int, k2: int, grid_resolution: int) -> Mesh:
    """Build triangle mesh for the rose surface."""
    n = grid_resolution
    theta_vals = np.linspace(0, np.pi, n)
    phi_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    tt, pp = np.meshgrid(theta_vals, phi_vals, indexing="ij")

    x, y, z = _evaluate_rose_surface(tt, pp, k1, k2)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)

    # Open in θ (poles), wrapped in φ (azimuthal)
    faces = build_mixed_grid_faces(n, n, wrap_u=False, wrap_v=True)

    return Mesh(vertices=vertices, faces=faces)


def _validate_params(k1: int, k2: int, grid_resolution: int) -> None:
    """Validate rose surface parameters."""
    if k1 < _MIN_K:
        raise ValueError(f"k1 must be >= {_MIN_K}, got {k1}")
    if k2 < _MIN_K:
        raise ValueError(f"k2 must be >= {_MIN_K}, got {k2}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


@register
class RoseSurfaceGenerator(GeneratorBase):
    """Parametric rose surface generator."""

    name = "rose_surface"
    category = "parametric"
    aliases = ()
    description = (
        "Rhodonea (rose) curve revolved into 3D, producing flower-like "
        "petal patterns via r = cos(k1·θ)·cos(k2·φ) on a sphere"
    )
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for rose surface parameters."""
        return {
            "k1": {"min": 1, "max": 12, "step": 1},
            "k2": {"min": 1, "max": 12, "step": 1},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the rose surface."""
        return {
            "k1": _DEFAULT_K1,
            "k2": _DEFAULT_K2,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a rose surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        k1 = int(merged["k1"])
        k2 = int(merged["k2"])
        defaults = self._resolution_defaults
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", defaults["grid_resolution"])
        )

        _validate_params(k1, k2, grid_resolution)

        mesh = _build_rose_surface_mesh(k1, k2, grid_resolution)
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated rose_surface: k1=%d, k2=%d, grid=%d, "
            "vertices=%d, faces=%d",
            k1, k2, grid_resolution,
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
        """Return the recommended representation for the rose surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
