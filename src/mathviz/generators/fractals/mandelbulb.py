"""Mandelbulb 3D fractal generator.

Evaluates the Mandelbulb escape-time field on a voxel grid and extracts the
boundary surface via marching cubes. The inner iteration kernel is
numba-JIT-compiled for acceptable performance at voxel_resolution >= 128.
Default representation: SPARSE_SHELL.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._escape_kernel import mandelbulb_escape_field
from mathviz.shared.marching_cubes import SpatialBounds, extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_POWER = 8.0
_DEFAULT_MAX_ITERATIONS = 10
_DEFAULT_VOXEL_RESOLUTION = 128
_DEFAULT_EXTENT = 1.5
_MIN_VOXEL_RESOLUTION = 4
_MIN_MAX_ITERATIONS = 1


def _validate_params(
    power: float,
    max_iterations: int,
    voxel_resolution: int,
    extent: float,
) -> None:
    """Validate Mandelbulb parameters."""
    if power < 2.0:
        raise ValueError(f"power must be >= 2.0, got {power}")
    if max_iterations < _MIN_MAX_ITERATIONS:
        raise ValueError(
            f"max_iterations must be >= {_MIN_MAX_ITERATIONS}, "
            f"got {max_iterations}"
        )
    if voxel_resolution < _MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {_MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )
    if extent <= 0:
        raise ValueError(f"extent must be positive, got {extent}")


def _build_escape_field(
    voxel_resolution: int,
    extent: float,
    max_iterations: int,
    power: float,
) -> tuple[np.ndarray, SpatialBounds]:
    """Build the 3D escape-time field for the Mandelbulb."""
    bounds = SpatialBounds(
        min_corner=(-extent, -extent, -extent),
        max_corner=(extent, extent, extent),
    )
    xs = np.linspace(-extent, extent, voxel_resolution)
    ys = np.linspace(-extent, extent, voxel_resolution)
    zs = np.linspace(-extent, extent, voxel_resolution)

    field = mandelbulb_escape_field(xs, ys, zs, max_iterations, power)
    return field, bounds


@register
class MandelbulbGenerator(GeneratorBase):
    """Mandelbulb 3D fractal via escape-time iteration and marching cubes."""

    name = "mandelbulb"
    category = "fractals"
    aliases = ()
    description = "Mandelbulb 3D fractal with numba-JIT escape-time kernel"
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Mandelbulb."""
        return {
            "power": _DEFAULT_POWER,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "extent": _DEFAULT_EXTENT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Mandelbulb mesh via marching cubes."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        power = float(merged["power"])
        max_iterations = int(merged["max_iterations"])
        extent = float(merged["extent"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", _DEFAULT_VOXEL_RESOLUTION)
        )

        _validate_params(power, max_iterations, voxel_resolution, extent)
        merged["voxel_resolution"] = voxel_resolution

        field, bounds = _build_escape_field(
            voxel_resolution, extent, max_iterations, power,
        )

        # Isosurface at the boundary: points that just escaped (iteration > 0)
        # Use a small threshold to extract the surface between inside/outside
        iso = 0.5
        mesh = extract_mesh(field, bounds, isolevel=iso)

        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated mandelbulb: power=%.1f, max_iter=%d, "
            "voxel_res=%d, vertices=%d, faces=%d",
            power, max_iterations, voxel_resolution,
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
        """Return SPARSE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
