"""Mandelbrot set heightmap generator.

Computes the Mandelbrot escape-time iteration count on a 2D grid and stores
it as a scalar field. The HEIGHTMAP_RELIEF representation extrudes this into
a 3D relief surface. Vectorized NumPy — no numba needed for pixel_resolution
up to ~512.
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

_DEFAULT_CENTER_REAL = -0.5
_DEFAULT_CENTER_IMAG = 0.0
_DEFAULT_ZOOM = 1.0
_DEFAULT_MAX_ITERATIONS = 256
_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_SMOOTHING = True
_DEFAULT_PIXEL_RESOLUTION = 512

# Mandelbrot set spans roughly [-2, 0.5] x [-1.25, 1.25] at zoom=1
_BASE_EXTENT = 2.5


def _mandelbrot_step(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Single Mandelbrot iteration: z² + c."""
    return z * z + c


def _apply_smoothing(
    iteration_count: np.ndarray,
    z: np.ndarray,
    escaped: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    """Apply smooth iteration count to escaped points."""
    smooth = iteration_count.copy()
    esc = escaped & (iteration_count > 0)
    abs_z = np.abs(z[esc])
    # Continuous (smooth) iteration count: subtract fractional escape
    log2_abs = np.log2(np.maximum(abs_z, 1e-300))
    smooth[esc] = iteration_count[esc] - np.log2(np.maximum(log2_abs, 1e-300))
    # Clamp to non-negative: smoothing can undershoot for points escaping
    # on iteration 1 with large |z| (e.g. at low zoom levels).
    return np.maximum(smooth, 0.0)


@register
class MandelbrotHeightmapGenerator(GeneratorBase):
    """Mandelbrot set as a heightmap relief surface.

    Escape-time iteration count on a pixel_resolution² grid becomes the
    z-height of a relief surface via HEIGHTMAP_RELIEF representation.
    Seed has no effect — the Mandelbrot set is fully deterministic.
    """

    name = "mandelbrot_heightmap"
    category = "fractals"
    aliases = ()
    description = "Mandelbrot escape-time heightmap for 3D relief engraving"
    resolution_params = {
        "pixel_resolution": "Grid points per axis (N² cost)",
    }
    _resolution_defaults = {"pixel_resolution": _DEFAULT_PIXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Mandelbrot heightmap."""
        return {
            "center_real": _DEFAULT_CENTER_REAL,
            "center_imag": _DEFAULT_CENTER_IMAG,
            "zoom": _DEFAULT_ZOOM,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "height_scale": _DEFAULT_HEIGHT_SCALE,
            "smoothing": _DEFAULT_SMOOTHING,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Mandelbrot escape-time scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        center_real = float(merged["center_real"])
        center_imag = float(merged["center_imag"])
        zoom = float(merged["zoom"])
        max_iterations = int(merged["max_iterations"])
        height_scale = float(merged["height_scale"])
        is_smooth = bool(merged["smoothing"])
        pixel_resolution = int(
            resolution_kwargs.get(
                "pixel_resolution",
                self._resolution_defaults["pixel_resolution"],
            )
        )

        validate_heightmap_params(
            zoom, max_iterations, pixel_resolution, height_scale,
            center_real, center_imag,
        )

        merged["pixel_resolution"] = pixel_resolution

        c = build_complex_grid(
            center_real, center_imag, zoom, _BASE_EXTENT, pixel_resolution,
        )
        iteration_count, z_final, escaped = escape_time_loop(
            c, max_iterations, _mandelbrot_step,
        )

        if is_smooth:
            iteration_count = _apply_smoothing(
                iteration_count, z_final, escaped, max_iterations,
            )

        field = iteration_count * height_scale
        bbox = compute_heightmap_bbox(field)

        logger.info(
            "Generated mandelbrot_heightmap: center=(%.3f, %.3f), zoom=%.3f, "
            "max_iter=%d, pixel_res=%d, field_range=[%.2f, %.2f]",
            center_real, center_imag, zoom, max_iterations, pixel_resolution,
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
