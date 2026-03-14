"""Schwarz D implicit surface generator.

The Schwarz D (Diamond) surface is a triply periodic minimal surface defined by
the implicit equation:
    sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
    + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = 0

Like the gyroid and Schwarz P, it tiles space; the ``periods`` parameter
controls how many unit cells are included.

Note: field evaluation is O(N³) where N is voxel_resolution.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_CELL_SIZE = 1.0
_DEFAULT_PERIODS = 2
_DEFAULT_VOXEL_RESOLUTION = 128
_MIN_VOXEL_RESOLUTION = 4
_MIN_PERIODS = 1


def _evaluate_schwarz_d_field(
    voxel_resolution: int,
    bounds: SpatialBounds,
) -> np.ndarray:
    """Evaluate the Schwarz D scalar field on an N³ voxel grid.

    Field: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
           + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z).
    """
    n = voxel_resolution
    x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
    y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
    z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    sx, cx = np.sin(xx), np.cos(xx)
    sy, cy = np.sin(yy), np.cos(yy)
    sz, cz = np.sin(zz), np.cos(zz)

    return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


def _compute_bounds(cell_size: float, periods: int) -> SpatialBounds:
    """Compute spatial bounds for the given number of periods."""
    extent = cell_size * periods * 2 * np.pi
    half = extent / 2.0
    return SpatialBounds(
        min_corner=(-half, -half, -half),
        max_corner=(half, half, half),
    )


def _compute_bounding_box(bounds: SpatialBounds) -> BoundingBox:
    """Convert spatial bounds to a BoundingBox."""
    return BoundingBox(
        min_corner=bounds.min_corner,
        max_corner=bounds.max_corner,
    )


def _validate_params(
    cell_size: float, periods: int, voxel_resolution: int,
) -> None:
    """Validate Schwarz D parameters, raising ValueError for invalid inputs."""
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    if periods < _MIN_PERIODS:
        raise ValueError(
            f"periods must be >= {_MIN_PERIODS}, got {periods}"
        )
    if voxel_resolution < _MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {_MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )


@register
class SchwarzDGenerator(GeneratorBase):
    """Implicit Schwarz D surface generator.

    Evaluates the Schwarz D (Diamond) scalar field on an N³ voxel grid and
    extracts the isosurface via marching cubes. The Diamond TPMS has the
    implicit form: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
    + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = 0.

    Seed has no effect on output; the surface is fully deterministic
    for given cell_size, periods, and voxel_resolution.
    """

    name = "schwarz_d"
    category = "implicit"
    aliases = ()
    description = "Triply periodic Schwarz D minimal surface via marching cubes"
    resolution_params = {
        "voxel_resolution": "Number of voxels per axis (N³ cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Schwarz D generator."""
        return {
            "cell_size": _DEFAULT_CELL_SIZE,
            "periods": _DEFAULT_PERIODS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Schwarz D mesh via marching cubes on the implicit field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        cell_size = float(merged["cell_size"])
        periods = int(merged["periods"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", _DEFAULT_VOXEL_RESOLUTION)
        )

        _validate_params(cell_size, periods, voxel_resolution)

        merged["voxel_resolution"] = voxel_resolution

        bounds = _compute_bounds(cell_size, periods)
        field = _evaluate_schwarz_d_field(voxel_resolution, bounds)
        mesh = extract_mesh(field, bounds, isolevel=0.0)
        bbox = _compute_bounding_box(bounds)

        logger.info(
            "Generated Schwarz D: cell_size=%.3f, periods=%d, "
            "voxel_resolution=%d, vertices=%d, faces=%d",
            cell_size, periods, voxel_resolution,
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
        """Return the recommended representation for Schwarz D."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
