"""Fractal cross-section (slice) generator.

Evaluates the Mandelbulb 3D escape-time field on a 2D plane, producing a
scalar field suitable for HEIGHTMAP_RELIEF representation. The plane is
defined by a constant coordinate along the chosen axis.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._escape_kernel import mandelbulb_escape_field

logger = logging.getLogger(__name__)

_DEFAULT_POWER = 8.0
_DEFAULT_MAX_ITERATIONS = 20
_DEFAULT_PIXEL_RESOLUTION = 256
_DEFAULT_EXTENT = 1.5
_DEFAULT_SLICE_AXIS = "z"
_DEFAULT_SLICE_POSITION = 0.0
_MIN_PIXEL_RESOLUTION = 4
_MIN_MAX_ITERATIONS = 1
_VALID_AXES = frozenset({"x", "y", "z"})


def _validate_params(
    power: float,
    max_iterations: int,
    pixel_resolution: int,
    extent: float,
    slice_axis: str,
    slice_position: float,
) -> None:
    """Validate fractal slice parameters."""
    if power < 2.0:
        raise ValueError(f"power must be >= 2.0, got {power}")
    if max_iterations < _MIN_MAX_ITERATIONS:
        raise ValueError(
            f"max_iterations must be >= {_MIN_MAX_ITERATIONS}, "
            f"got {max_iterations}"
        )
    if pixel_resolution < _MIN_PIXEL_RESOLUTION:
        raise ValueError(
            f"pixel_resolution must be >= {_MIN_PIXEL_RESOLUTION}, "
            f"got {pixel_resolution}"
        )
    if extent <= 0:
        raise ValueError(f"extent must be positive, got {extent}")
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

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the fractal slice."""
        return {
            "power": _DEFAULT_POWER,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "extent": _DEFAULT_EXTENT,
            "slice_axis": _DEFAULT_SLICE_AXIS,
            "slice_position": _DEFAULT_SLICE_POSITION,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 2D fractal cross-section as a scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        power = float(merged["power"])
        max_iterations = int(merged["max_iterations"])
        extent = float(merged["extent"])
        slice_axis = str(merged["slice_axis"])
        slice_position = float(merged["slice_position"])
        pixel_resolution = int(
            resolution_kwargs.get("pixel_resolution", _DEFAULT_PIXEL_RESOLUTION)
        )

        _validate_params(
            power, max_iterations, pixel_resolution,
            extent, slice_axis, slice_position,
        )
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
