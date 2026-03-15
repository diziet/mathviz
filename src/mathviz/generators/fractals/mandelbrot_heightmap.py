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
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CENTER_REAL = -0.5
_DEFAULT_CENTER_IMAG = 0.0
_DEFAULT_ZOOM = 1.0
_DEFAULT_MAX_ITERATIONS = 256
_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_SMOOTHING = True
_DEFAULT_PIXEL_RESOLUTION = 512
_MIN_PIXEL_RESOLUTION = 4
_MIN_MAX_ITERATIONS = 1

# Mandelbrot set spans roughly [-2, 0.5] x [-1.25, 1.25] at zoom=1
_BASE_EXTENT = 2.5


def _validate_params(
    zoom: float,
    max_iterations: int,
    pixel_resolution: int,
    height_scale: float,
) -> None:
    """Validate Mandelbrot parameters, raising ValueError for invalid inputs."""
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


def _compute_mandelbrot_field(
    center_real: float,
    center_imag: float,
    zoom: float,
    max_iterations: int,
    pixel_resolution: int,
    is_smooth: bool,
) -> np.ndarray:
    """Compute the Mandelbrot escape-time field on a 2D grid."""
    extent = _BASE_EXTENT / zoom
    real_min = center_real - extent
    real_max = center_real + extent
    imag_min = center_imag - extent
    imag_max = center_imag + extent

    real_axis = np.linspace(real_min, real_max, pixel_resolution)
    imag_axis = np.linspace(imag_min, imag_max, pixel_resolution)
    real_grid, imag_grid = np.meshgrid(real_axis, imag_axis)
    c = real_grid + 1j * imag_grid

    return _escape_time(c, max_iterations, is_smooth)


def _escape_time(
    c: np.ndarray,
    max_iterations: int,
    is_smooth: bool,
) -> np.ndarray:
    """Vectorized escape-time iteration for the Mandelbrot set."""
    z = np.zeros_like(c)
    iteration_count = np.zeros(c.shape, dtype=np.float64)
    escaped = np.zeros(c.shape, dtype=bool)

    # Overflow to inf is expected for escaped points; suppress warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(max_iterations):
            mask = ~escaped
            active_z = z[mask] * z[mask] + c[mask]
            z[mask] = active_z
            mag_exceeded = active_z.real ** 2 + active_z.imag ** 2 > 4.0
            newly_escaped_idx = np.where(mask)[0][mag_exceeded]
            iteration_count.ravel()[newly_escaped_idx] = float(i + 1)
            escaped.ravel()[newly_escaped_idx] = True

    if is_smooth:
        iteration_count = _apply_smoothing(
            iteration_count, z, escaped, max_iterations,
        )

    return iteration_count


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
            resolution_kwargs.get("pixel_resolution", _DEFAULT_PIXEL_RESOLUTION)
        )

        _validate_params(zoom, max_iterations, pixel_resolution, height_scale)

        merged["pixel_resolution"] = pixel_resolution

        field = _compute_mandelbrot_field(
            center_real, center_imag, zoom,
            max_iterations, pixel_resolution, is_smooth,
        )

        field = field * height_scale

        bbox = _compute_bounding_box(field)

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


def _compute_bounding_box(field: np.ndarray) -> BoundingBox:
    """Compute bounding box for the heightmap mesh that will be generated."""
    z_min = float(np.min(field))
    z_max = float(np.max(field))
    return BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(1.0, 1.0, z_max),
    )
