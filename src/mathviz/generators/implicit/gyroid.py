"""Gyroid implicit surface generator.

The gyroid is a triply periodic minimal surface defined by the implicit equation
sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0. It tiles space infinitely;
the ``periods`` parameter controls how many unit cells are included.

Note: field evaluation is O(N³) where N is voxel_resolution. High resolutions
(e.g. 256+) will be slow and memory-intensive.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_PERIODS = 2
_DEFAULT_VOXEL_RESOLUTION = 128
_MIN_VOXEL_RESOLUTION = 4
_MIN_PERIODS = 1


def _evaluate_gyroid_field(
    voxel_resolution: int,
    bounds: SpatialBounds,
) -> np.ndarray:
    """Evaluate the gyroid scalar field on an N³ voxel grid.

    Field: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x).
    O(N³) in both time and memory.
    """
    n = voxel_resolution
    x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
    y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
    z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    field = (
        np.sin(xx) * np.cos(yy)
        + np.sin(yy) * np.cos(zz)
        + np.sin(zz) * np.cos(xx)
    )
    return field


def _compute_bounds(periods: int) -> SpatialBounds:
    """Compute spatial bounds for the given number of periods."""
    extent = periods * 2 * np.pi
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


def _validate_params(periods: int, voxel_resolution: int) -> None:
    """Validate gyroid parameters, raising ValueError for invalid inputs."""
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
class GyroidGenerator(GeneratorBase):
    """Implicit gyroid surface generator.

    Evaluates the gyroid scalar field sin(x)cos(y) + sin(y)cos(z) +
    sin(z)cos(x) on an N³ voxel grid and extracts the isosurface via
    marching cubes. Cost is O(N³) in voxel_resolution.

    Seed has no effect on output; the gyroid is fully deterministic
    for given periods and voxel_resolution.
    """

    name = "gyroid"
    category = "implicit"
    aliases = ()
    description = "Triply periodic gyroid minimal surface via marching cubes"
    resolution_params = {
        "voxel_resolution": "Number of voxels per axis (N³ cost)",
    }
    _resolution_defaults = {"voxel_resolution": _DEFAULT_VOXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the gyroid generator."""
        return {
            "periods": _DEFAULT_PERIODS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a gyroid mesh via marching cubes on the implicit field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        # Silently ignore cell_size if passed (removed parameter)
        merged.pop("cell_size", None)

        periods = int(merged["periods"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", _DEFAULT_VOXEL_RESOLUTION)
        )

        _validate_params(periods, voxel_resolution)

        # Record voxel_resolution so output is self-describing
        merged["voxel_resolution"] = voxel_resolution

        bounds = _compute_bounds(periods)
        field = _evaluate_gyroid_field(voxel_resolution, bounds)
        mesh = extract_mesh(field, bounds, isolevel=0.0)
        bbox = _compute_bounding_box(bounds)

        logger.info(
            "Generated gyroid: periods=%d, voxel_resolution=%d, "
            "vertices=%d, faces=%d",
            periods, voxel_resolution,
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
        """Return the recommended representation for the gyroid."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
