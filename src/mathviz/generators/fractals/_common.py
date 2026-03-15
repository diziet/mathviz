"""Shared constants, validation, and helpers for 3D fractal generators."""

import numpy as np

from mathviz.shared.marching_cubes import SpatialBounds

# Shared defaults for Mandelbulb, Julia 3D, and fractal slice
DEFAULT_POWER = 8.0
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_VOXEL_RESOLUTION = 128
DEFAULT_PIXEL_RESOLUTION = 256
DEFAULT_EXTENT = 1.5
MIN_VOXEL_RESOLUTION = 4
MIN_PIXEL_RESOLUTION = 4
MIN_MAX_ITERATIONS = 1
MIN_POWER = 2.0

# Isolevel for escape-time boundary extraction via marching cubes.
# The kernel returns 0.0 for points inside the set and integer >= 1
# for escaped points. An isolevel of 0.5 extracts the boundary between
# inside (0) and first-escape (1) regions.
ESCAPE_ISOLEVEL = 0.5

# Maximum allowed absolute value for Julia c parameters to prevent
# floating-point overflow in the iteration kernel.
MAX_C_MAGNITUDE = 10.0


def validate_fractal_params(
    power: float,
    max_iterations: int,
    resolution: int,
    extent: float,
    *,
    resolution_name: str = "voxel_resolution",
    min_resolution: int = MIN_VOXEL_RESOLUTION,
) -> None:
    """Validate common fractal generator parameters."""
    if power < MIN_POWER:
        raise ValueError(f"power must be >= {MIN_POWER}, got {power}")
    if max_iterations < MIN_MAX_ITERATIONS:
        raise ValueError(
            f"max_iterations must be >= {MIN_MAX_ITERATIONS}, "
            f"got {max_iterations}"
        )
    if resolution < min_resolution:
        raise ValueError(
            f"{resolution_name} must be >= {min_resolution}, "
            f"got {resolution}"
        )
    if extent <= 0:
        raise ValueError(f"extent must be positive, got {extent}")


def validate_c_params(c_re: float, c_im: float, c_z: float) -> None:
    """Validate Julia 3D c parameters are within safe bounds."""
    for name, val in [("c_re", c_re), ("c_im", c_im), ("c_z", c_z)]:
        if abs(val) > MAX_C_MAGNITUDE:
            raise ValueError(
                f"{name} magnitude {abs(val):.1f} exceeds maximum "
                f"{MAX_C_MAGNITUDE}; use a value in [-{MAX_C_MAGNITUDE}, "
                f"{MAX_C_MAGNITUDE}]"
            )


def validate_quaternion_c_params(
    c_real: float, c_i: float, c_j: float, c_k: float,
) -> None:
    """Validate quaternion Julia c parameters are within safe bounds."""
    for name, val in [
        ("c_real", c_real), ("c_i", c_i), ("c_j", c_j), ("c_k", c_k),
    ]:
        if abs(val) > MAX_C_MAGNITUDE:
            raise ValueError(
                f"{name} magnitude {abs(val):.1f} exceeds maximum "
                f"{MAX_C_MAGNITUDE}; use a value in [-{MAX_C_MAGNITUDE}, "
                f"{MAX_C_MAGNITUDE}]"
            )


def validate_escape_radius(escape_radius: float) -> None:
    """Validate escape radius is positive."""
    if escape_radius <= 0:
        raise ValueError(f"escape_radius must be positive, got {escape_radius}")


def build_voxel_grid(
    voxel_resolution: int,
    extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SpatialBounds]:
    """Build coordinate arrays and bounds for a symmetric 3D voxel grid."""
    bounds = SpatialBounds(
        min_corner=(-extent, -extent, -extent),
        max_corner=(extent, extent, extent),
    )
    xs = np.linspace(-extent, extent, voxel_resolution)
    ys = np.linspace(-extent, extent, voxel_resolution)
    zs = np.linspace(-extent, extent, voxel_resolution)
    return xs, ys, zs, bounds
