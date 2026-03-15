"""Fractal cross-section (slice) generator.

Evaluates the Mandelbulb 3D escape-time field on a 2D plane, producing a
scalar field suitable for HEIGHTMAP_RELIEF representation. The plane is
defined by a constant coordinate along the chosen axis.

Seed is stored for metadata/provenance only — the computation is fully
deterministic for given parameters.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._common import (
    DEFAULT_EXTENT,
    DEFAULT_PIXEL_RESOLUTION,
    DEFAULT_POWER,
    MIN_PIXEL_RESOLUTION,
    validate_fractal_params,
)
from mathviz.generators.fractals._escape_kernel import mandelbulb_escape_field

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 20
_DEFAULT_SLICE_AXIS = "z"
_DEFAULT_SLICE_POSITION = 0.0
_VALID_AXES = frozenset({"x", "y", "z"})


def _validate_slice_params(
    slice_axis: str,
    slice_position: float,
    extent: float,
) -> None:
    """Validate fractal-slice-specific parameters."""
    if slice_axis not in _VALID_AXES:
        raise ValueError(
            f"slice_axis must be one of {sorted(_VALID_AXES)}, "
            f"got {slice_axis!r}"
        )
    if abs(slice_position) > extent:
        raise ValueError(
            f"slice_position {slice_position} is outside extent "
            f"[-{extent}, {extent}]"
        )


def _evaluate_slice(
    pixel_resolution: int,
    extent: float,
    max_iterations: int,
    power: float,
    slice_axis: str,
    slice_position: float,
) -> np.ndarray:
    """Evaluate the Mandelbulb field on a 2D slice plane."""
    plane_coords = np.linspace(-extent, extent, pixel_resolution)
    fixed_coord = np.array([slice_position])

    if slice_axis == "z":
        field_3d = mandelbulb_escape_field(
            plane_coords, plane_coords, fixed_coord,
            max_iterations, power,
        )
        return field_3d[:, :, 0]
    elif slice_axis == "y":
        field_3d = mandelbulb_escape_field(
            plane_coords, fixed_coord, plane_coords,
            max_iterations, power,
        )
        return field_3d[:, 0, :]
    else:  # x
        field_3d = mandelbulb_escape_field(
            fixed_coord, plane_coords, plane_coords,
            max_iterations, power,
        )
        return field_3d[0, :, :]


@register
class FractalSliceGenerator(GeneratorBase):
    """Fractal cross-section: 2D slice through a 3D Mandelbulb field."""

    name = "fractal_slice"
    category = "fractals"
    aliases = ("fractal_cross_section",)
    description = "2D cross-section through a Mandelbulb for heightmap relief"
    resolution_params = {
        "pixel_resolution": "Grid points per axis (N² cost)",
    }
    _resolution_defaults = {"pixel_resolution": DEFAULT_PIXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the fractal slice."""
        return {
            "power": DEFAULT_POWER,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "extent": DEFAULT_EXTENT,
            "slice_axis": _DEFAULT_SLICE_AXIS,
            "slice_position": _DEFAULT_SLICE_POSITION,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 2D fractal cross-section as a scalar field.

        Seed is stored for metadata only — output is fully deterministic.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        power = float(merged["power"])
        max_iterations = int(merged["max_iterations"])
        extent = float(merged["extent"])
        slice_axis = str(merged["slice_axis"])
        slice_position = float(merged["slice_position"])
        pixel_resolution = int(
            resolution_kwargs.get("pixel_resolution", DEFAULT_PIXEL_RESOLUTION)
        )

        validate_fractal_params(
            power, max_iterations, pixel_resolution, extent,
            resolution_name="pixel_resolution",
            min_resolution=MIN_PIXEL_RESOLUTION,
        )
        _validate_slice_params(slice_axis, slice_position, extent)
        merged["pixel_resolution"] = pixel_resolution

        field = _evaluate_slice(
            pixel_resolution, extent, max_iterations,
            power, slice_axis, slice_position,
        )

        z_min = float(np.min(field))
        z_max = float(np.max(field))
        bbox = BoundingBox(
            min_corner=(0.0, 0.0, z_min),
            max_corner=(1.0, 1.0, z_max),
        )

        logger.info(
            "Generated fractal_slice: power=%.1f, axis=%s, pos=%.3f, "
            "pixel_res=%d, field_range=[%.2f, %.2f]",
            power, slice_axis, slice_position, pixel_resolution,
            z_min, z_max,
        )

        return MathObject(
            scalar_field=field,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return HEIGHTMAP_RELIEF as default representation."""
        return RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
