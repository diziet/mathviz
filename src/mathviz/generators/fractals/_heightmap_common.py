"""Shared helpers for 2D heightmap fractal generators.

Provides validation, complex-plane grid construction, escape-time iteration,
and bounding-box computation used by both Mandelbrot and Burning Ship
heightmap generators.
"""

import math
from collections.abc import Callable

import numpy as np

from mathviz.core.math_object import BoundingBox

MIN_PIXEL_RESOLUTION = 4
MIN_MAX_ITERATIONS = 1


def validate_heightmap_params(
    zoom: float,
    max_iterations: int,
    pixel_resolution: int,
    height_scale: float,
    center_x: float,
    center_y: float,
) -> None:
    """Validate common heightmap fractal parameters."""
    if not math.isfinite(center_x):
        raise ValueError(f"center_x must be finite, got {center_x}")
    if not math.isfinite(center_y):
        raise ValueError(f"center_y must be finite, got {center_y}")
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
    if max_iterations < MIN_MAX_ITERATIONS:
        raise ValueError(
            f"max_iterations must be >= {MIN_MAX_ITERATIONS}, "
            f"got {max_iterations}"
        )
    if pixel_resolution < MIN_PIXEL_RESOLUTION:
        raise ValueError(
            f"pixel_resolution must be >= {MIN_PIXEL_RESOLUTION}, "
            f"got {pixel_resolution}"
        )


def build_complex_grid(
    center_x: float,
    center_y: float,
    zoom: float,
    base_extent: float,
    pixel_resolution: int,
) -> np.ndarray:
    """Build a 2D complex-plane grid centered at (center_x, center_y)."""
    extent = base_extent / zoom
    real_axis = np.linspace(center_x - extent, center_x + extent, pixel_resolution)
    imag_axis = np.linspace(center_y - extent, center_y + extent, pixel_resolution)
    real_grid, imag_grid = np.meshgrid(real_axis, imag_axis)
    return real_grid + 1j * imag_grid


# Type alias for the per-pixel iteration step function.
# Signature: step_fn(z_active, c_active) -> z_next
StepFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def escape_time_loop(
    c: np.ndarray,
    max_iterations: int,
    step_fn: StepFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run vectorized escape-time iteration with a pluggable step function.

    Returns (iteration_count, z_final, escaped) — all shaped like ``c``.
    The caller can use z_final and escaped for optional smoothing.
    """
    z_flat = np.zeros(c.size, dtype=c.dtype)
    c_flat = c.ravel()
    iteration_count = np.zeros(c.size, dtype=np.float64)
    active = np.arange(c.size)

    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(max_iterations):
            z_flat[active] = step_fn(z_flat[active], c_flat[active])
            mag = z_flat[active].real ** 2 + z_flat[active].imag ** 2
            escaped_mask = mag > 4.0
            iteration_count[active[escaped_mask]] = float(i + 1)
            active = active[~escaped_mask]
            if active.size == 0:
                break

    shape = c.shape
    escaped = iteration_count > 0
    return (
        iteration_count.reshape(shape),
        z_flat.reshape(shape),
        escaped.reshape(shape),
    )


def compute_heightmap_bbox(field: np.ndarray) -> BoundingBox:
    """Compute bounding box for a heightmap scalar field."""
    z_min = float(np.min(field))
    z_max = float(np.max(field))
    return BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(1.0, 1.0, z_max),
    )
