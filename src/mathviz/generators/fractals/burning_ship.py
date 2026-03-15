"""Burning Ship fractal heightmap generator.

Computes the Burning Ship escape-time iteration count on a 2D grid and stores
it as a scalar field. The iteration rule is z → (|Re(z)| + i|Im(z)|)² + c,
producing an asymmetric, aggressive-looking fractal distinct from the
Mandelbrot set. Uses HEIGHTMAP_RELIEF representation.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CENTER_X = -0.4
_DEFAULT_CENTER_Y = -0.6
_DEFAULT_ZOOM = 3.0
_DEFAULT_MAX_ITERATIONS = 256
_DEFAULT_HEIGHT_SCALE = 0.3
_DEFAULT_PIXEL_RESOLUTION = 512
_MIN_PIXEL_RESOLUTION = 4
_MIN_MAX_ITERATIONS = 1

# Burning Ship spans roughly [-2.5, 1.5] x [-2, 1] — zoom=1 covers ~4 units
_BASE_EXTENT = 2.0


def _validate_params(
    zoom: float,
    max_iterations: int,
    pixel_resolution: int,
    height_scale: float,
) -> None:
    """Validate Burning Ship parameters, raising ValueError for invalid inputs."""
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
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


def _compute_burning_ship_field(
    center_x: float,
    center_y: float,
    zoom: float,
    max_iterations: int,
    pixel_resolution: int,
) -> np.ndarray:
    """Compute the Burning Ship escape-time field on a 2D grid."""
    extent = _BASE_EXTENT / zoom
    real_min = center_x - extent
    real_max = center_x + extent
    imag_min = center_y - extent
    imag_max = center_y + extent

    real_axis = np.linspace(real_min, real_max, pixel_resolution)
    imag_axis = np.linspace(imag_min, imag_max, pixel_resolution)
    real_grid, imag_grid = np.meshgrid(real_axis, imag_axis)
    c = real_grid + 1j * imag_grid

    return _escape_time(c, max_iterations)


def _escape_time(
    c: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    """Vectorized escape-time iteration for the Burning Ship fractal.

    Iteration rule: z → (|Re(z)| + i·|Im(z)|)² + c
    """
    z = np.zeros_like(c)
    iteration_count = np.zeros(c.shape, dtype=np.float64)
    escaped = np.zeros(c.shape, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(max_iterations):
            mask = ~escaped
            # Burning Ship: take absolute values before squaring
            z_abs = np.abs(z[mask].real) + 1j * np.abs(z[mask].imag)
            active_z = z_abs * z_abs + c[mask]
            z[mask] = active_z
            mag_exceeded = active_z.real ** 2 + active_z.imag ** 2 > 4.0
            newly_escaped_idx = np.where(mask)[0][mag_exceeded]
            iteration_count.ravel()[newly_escaped_idx] = float(i + 1)
            escaped.ravel()[newly_escaped_idx] = True

    return iteration_count


def _compute_bounding_box(field: np.ndarray) -> BoundingBox:
    """Compute bounding box for the heightmap mesh that will be generated."""
    z_min = float(np.min(field))
    z_max = float(np.max(field))
    return BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(1.0, 1.0, z_max),
    )


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
            resolution_kwargs.get("pixel_resolution", _DEFAULT_PIXEL_RESOLUTION)
        )

        _validate_params(zoom, max_iterations, pixel_resolution, height_scale)

        merged["pixel_resolution"] = pixel_resolution

        field = _compute_burning_ship_field(
            center_x, center_y, zoom,
            max_iterations, pixel_resolution,
        )

        field = field * height_scale

        bbox = _compute_bounding_box(field)

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
