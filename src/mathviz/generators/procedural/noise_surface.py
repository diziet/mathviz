"""Noise surface generator.

Evaluates seed-controlled simplex noise as a heightmap and outputs a scalar
field for HEIGHTMAP_RELIEF representation. The noise frequency, amplitude,
and height scale are configurable.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.procedural._utils import compute_heightmap_bounding_box
from mathviz.generators.procedural.simplex import evaluate_2d_grid

logger = logging.getLogger(__name__)

_DEFAULT_FREQUENCY = 4.0
_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_PIXEL_RESOLUTION = 256
_MIN_PIXEL_RESOLUTION = 4
_MIN_FREQUENCY = 0.01


def _validate_params(
    frequency: float,
    height_scale: float,
    pixel_resolution: int,
) -> None:
    """Validate noise surface parameters."""
    if frequency < _MIN_FREQUENCY:
        raise ValueError(f"frequency must be >= {_MIN_FREQUENCY}, got {frequency}")
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
    if pixel_resolution < _MIN_PIXEL_RESOLUTION:
        raise ValueError(
            f"pixel_resolution must be >= {_MIN_PIXEL_RESOLUTION}, "
            f"got {pixel_resolution}"
        )


def _compute_noise_field(
    frequency: float,
    height_scale: float,
    pixel_resolution: int,
    seed: int,
) -> np.ndarray:
    """Generate a simplex noise heightmap field."""
    extent = frequency
    field = evaluate_2d_grid(
        resolution=pixel_resolution,
        x_range=(0.0, extent),
        y_range=(0.0, extent),
        seed=seed,
    )
    return field * height_scale


@register
class NoiseSurfaceGenerator(GeneratorBase):
    """Simplex noise heightmap surface.

    Evaluates seed-controlled simplex noise on a 2D grid and outputs the
    result as a scalar field for HEIGHTMAP_RELIEF representation.
    """

    name = "noise_surface"
    category = "procedural"
    aliases = ("simplex_surface",)
    description = "Seed-controlled simplex noise heightmap surface"
    resolution_params = {
        "pixel_resolution": "Grid points per axis (N² cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for noise surface."""
        return {
            "frequency": _DEFAULT_FREQUENCY,
            "height_scale": _DEFAULT_HEIGHT_SCALE,
        }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"pixel_resolution": _DEFAULT_PIXEL_RESOLUTION}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a simplex noise scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        frequency = float(merged["frequency"])
        height_scale = float(merged["height_scale"])
        pixel_resolution = int(
            resolution_kwargs.get("pixel_resolution", _DEFAULT_PIXEL_RESOLUTION)
        )

        _validate_params(frequency, height_scale, pixel_resolution)

        merged["pixel_resolution"] = pixel_resolution

        field = _compute_noise_field(frequency, height_scale, pixel_resolution, seed)
        bbox = compute_heightmap_bounding_box(field)

        logger.info(
            "Generated noise_surface: freq=%.2f, height_scale=%.2f, "
            "pixel_res=%d, field_range=[%.4f, %.4f]",
            frequency, height_scale, pixel_resolution,
            bbox.min_corner[2], bbox.max_corner[2],
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
