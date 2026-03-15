"""Burning Ship fractal heightmap generator.

Computes the Burning Ship escape-time iteration count on a 2D grid and stores
it as a scalar field. The iteration rule takes absolute values of the real and
imaginary parts of z before squaring: z_next = (|Re(z)| + i·|Im(z)|)² + c.
This produces an asymmetric, aggressive-looking fractal distinct from the
Mandelbrot set. Uses HEIGHTMAP_RELIEF representation.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._heightmap_common import (
    build_complex_grid,
    compute_heightmap_bbox,
    escape_time_loop,
    validate_heightmap_params,
)

logger = logging.getLogger(__name__)

_DEFAULT_CENTER_X = -0.4
_DEFAULT_CENTER_Y = -0.6
_DEFAULT_ZOOM = 3.0
_DEFAULT_MAX_ITERATIONS = 256
_DEFAULT_HEIGHT_SCALE = 0.3
_DEFAULT_PIXEL_RESOLUTION = 512

# Burning Ship spans roughly [-2.5, 1.5] x [-2, 1] — zoom=1 covers ~4 units
_BASE_EXTENT = 2.0


def _burning_ship_step(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Single Burning Ship iteration: (|Re(z)| + i·|Im(z)|)² + c."""
    z_abs = np.abs(z.real) + 1j * np.abs(z.imag)
    return z_abs * z_abs + c


@register
class BurningShipGenerator(GeneratorBase):
    """Burning Ship fractal as a heightmap relief surface.

    Escape-time iteration count on a pixel_resolution² grid becomes the
    z-height of a relief surface via HEIGHTMAP_RELIEF representation.
    Seed has no effect — the fractal is fully deterministic.
    """

    name = "burning_ship"
    category = "fractals"
    aliases = ()
    description = "Burning Ship escape-time heightmap for 3D relief engraving"
    resolution_params = {
        "pixel_resolution": "Grid points per axis (N² cost)",
    }
    _resolution_defaults = {"pixel_resolution": _DEFAULT_PIXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Burning Ship heightmap."""
        return {
            "center_x": _DEFAULT_CENTER_X,
            "center_y": _DEFAULT_CENTER_Y,
            "zoom": _DEFAULT_ZOOM,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "height_scale": _DEFAULT_HEIGHT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Burning Ship escape-time scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        center_x = float(merged["center_x"])
        center_y = float(merged["center_y"])
        zoom = float(merged["zoom"])
        max_iterations = int(merged["max_iterations"])
        height_scale = float(merged["height_scale"])
        pixel_resolution = int(
            resolution_kwargs.get(
                "pixel_resolution",
                self._resolution_defaults["pixel_resolution"],
            )
        )

        validate_heightmap_params(
            zoom, max_iterations, pixel_resolution, height_scale,
            center_x, center_y,
        )

        merged["pixel_resolution"] = pixel_resolution

        c = build_complex_grid(
            center_x, center_y, zoom, _BASE_EXTENT, pixel_resolution,
        )
        iteration_count, _z, _escaped = escape_time_loop(
            c, max_iterations, _burning_ship_step,
        )

        field = iteration_count * height_scale
        bbox = compute_heightmap_bbox(field)

        logger.info(
            "Generated burning_ship: center=(%.3f, %.3f), zoom=%.3f, "
            "max_iter=%d, pixel_res=%d, field_range=[%.2f, %.2f]",
            center_x, center_y, zoom, max_iterations, pixel_resolution,
            float(np.min(field)), float(np.max(field)),
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
