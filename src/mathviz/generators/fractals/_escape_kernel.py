"""Numba-JIT-compiled escape-time kernels for 3D fractals.

These kernels evaluate whether each voxel in a 3D grid escapes under
the Mandelbulb or Julia 3D iteration. The inner loops are compiled with
``@numba.njit`` for acceptable performance at voxel_resolution >= 128.

The kernel returns 0.0 for points inside the fractal set (did not escape
within max_iterations) and a positive float (the iteration count at which
escape occurred) for points outside the set.
"""

import logging

import numba
import numpy as np

logger = logging.getLogger(__name__)

# Runtime check: True when numba JIT compilation is enabled.
# Reflects NUMBA_DISABLE_JIT env var and numba configuration.
NUMBA_JIT_ACTIVE: bool = not numba.config.DISABLE_JIT

_BAILOUT_SQ = 256.0 * 256.0


@numba.njit(cache=True)
def _iterate_voxel(
    x0: float,
    y0: float,
    z0: float,
    cx: float,
    cy: float,
    cz: float,
    max_iterations: int,
    power: float,
) -> float:
    """Evaluate a single voxel for escape-time iteration.

    Shared by Mandelbulb (where c = starting point) and Julia 3D
    (where c is a fixed parameter). Returns iteration count at escape,
    or 0.0 if the point did not escape.
    """
    x, y, z = x0, y0, z0
    for i in range(max_iterations):
        r_sq = x * x + y * y + z * z
        if r_sq > _BAILOUT_SQ:
            return float(i)
        r = r_sq ** 0.5
        theta = np.arctan2((x * x + y * y) ** 0.5, z)
        phi = np.arctan2(y, x)
        r_n = r ** power
        theta_n = theta * power
        phi_n = phi * power
        sin_theta_n = np.sin(theta_n)
        cos_theta_n = np.cos(theta_n)
        x = r_n * sin_theta_n * np.cos(phi_n) + cx
        y = r_n * sin_theta_n * np.sin(phi_n) + cy
        z = r_n * cos_theta_n + cz
    return 0.0


@numba.njit(parallel=True, cache=True)
def mandelbulb_escape_field(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_iterations: int,
    power: float,
) -> np.ndarray:
    """Evaluate Mandelbulb escape-time on a 3D voxel grid.

    For the Mandelbulb, each voxel's position is used as both the starting
    point and the c parameter (c = position). Returns a 3D float64 array
    where 0.0 means inside the set and positive values are escape iteration
    counts.
    """
    nx = xs.shape[0]
    ny = ys.shape[0]
    nz = zs.shape[0]
    field = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix in numba.prange(nx):
        cx = xs[ix]
        for iy in range(ny):
            cy = ys[iy]
            for iz in range(nz):
                cz = zs[iz]
                field[ix, iy, iz] = _iterate_voxel(
                    cx, cy, cz, cx, cy, cz, max_iterations, power,
                )
    return field


@numba.njit(cache=True)
def _iterate_quaternion(
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    c0: float,
    c1: float,
    c2: float,
    c3: float,
    max_iterations: int,
    escape_radius_sq: float,
) -> float:
    """Evaluate a single quaternion Julia iteration.

    Iterates q → q² + c in quaternion space. Returns iteration count at
    escape, or 0.0 if the point did not escape.
    """
    a, b, c, d = q0, q1, q2, q3
    for i in range(max_iterations):
        norm_sq = a * a + b * b + c * c + d * d
        if norm_sq > escape_radius_sq:
            return float(i)
        # q² = (a²-b²-c²-d², 2ab, 2ac, 2ad)
        new_a = a * a - b * b - c * c - d * d + c0
        new_b = 2.0 * a * b + c1
        new_c = 2.0 * a * c + c2
        new_d = 2.0 * a * d + c3
        a, b, c, d = new_a, new_b, new_c, new_d
    return 0.0


@numba.njit(parallel=True, cache=True)
def quaternion_julia_escape_field(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    slice_w: float,
    max_iterations: int,
    escape_radius: float,
    c_real: float,
    c_i: float,
    c_j: float,
    c_k: float,
) -> np.ndarray:
    """Evaluate quaternion Julia escape-time on a 3D voxel grid.

    Each voxel (x, y, z) maps to quaternion (x, y, z, slice_w). The 4th
    component is fixed, giving a 3D slice of the 4D fractal.
    """
    nx = xs.shape[0]
    ny = ys.shape[0]
    nz = zs.shape[0]
    escape_radius_sq = escape_radius * escape_radius
    field = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix in numba.prange(nx):
        q0 = xs[ix]
        for iy in range(ny):
            q1 = ys[iy]
            for iz in range(nz):
                q2 = zs[iz]
                field[ix, iy, iz] = _iterate_quaternion(
                    q0, q1, q2, slice_w,
                    c_real, c_i, c_j, c_k,
                    max_iterations, escape_radius_sq,
                )
    return field


@numba.njit(parallel=True, cache=True)
def julia3d_escape_field(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_iterations: int,
    power: float,
    c_re: float,
    c_im: float,
    c_z: float,
) -> np.ndarray:
    """Evaluate Julia 3D escape-time on a 3D voxel grid.

    Same iteration as Mandelbulb but with a fixed c parameter instead of
    using the voxel position. Returns a 3D float64 array where 0.0 means
    inside the set and positive values are escape iteration counts.
    """
    nx = xs.shape[0]
    ny = ys.shape[0]
    nz = zs.shape[0]
    field = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix in numba.prange(nx):
        x0 = xs[ix]
        for iy in range(ny):
            y0 = ys[iy]
            for iz in range(nz):
                z0 = zs[iz]
                field[ix, iy, iz] = _iterate_voxel(
                    x0, y0, z0, c_re, c_im, c_z, max_iterations, power,
                )
    return field
