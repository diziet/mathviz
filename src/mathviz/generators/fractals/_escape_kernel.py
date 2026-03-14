"""Numba-JIT-compiled escape-time kernels for 3D fractals.

These kernels evaluate whether each voxel in a 3D grid escapes under
the Mandelbulb or Julia 3D iteration. The inner loops are compiled with
``@numba.njit`` for acceptable performance at voxel_resolution >= 128.
"""

import logging

import numba
import numpy as np

logger = logging.getLogger(__name__)

# Flag indicating numba JIT is active (testable)
NUMBA_JIT_ACTIVE = True

_BAILOUT = 256.0
_BAILOUT_SQ = _BAILOUT * _BAILOUT


@numba.njit(cache=True)
def mandelbulb_escape_field(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    max_iterations: int,
    power: float,
) -> np.ndarray:
    """Evaluate Mandelbulb escape-time on a 3D voxel grid.

    Returns a 3D float64 array of iteration counts (0 = inside set).
    """
    nx = xs.shape[0]
    ny = ys.shape[0]
    nz = zs.shape[0]
    field = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix in range(nx):
        cx = xs[ix]
        for iy in range(ny):
            cy = ys[iy]
            for iz in range(nz):
                cz = zs[iz]
                field[ix, iy, iz] = _mandelbulb_voxel(
                    cx, cy, cz, max_iterations, power,
                )
    return field


@numba.njit(cache=True)
def _mandelbulb_voxel(
    cx: float,
    cy: float,
    cz: float,
    max_iterations: int,
    power: float,
) -> float:
    """Evaluate a single voxel for the Mandelbulb."""
    x, y, z = cx, cy, cz
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
        x = r_n * np.sin(theta_n) * np.cos(phi_n) + cx
        y = r_n * np.sin(theta_n) * np.sin(phi_n) + cy
        z = r_n * np.cos(theta_n) + cz
    return 0.0


@numba.njit(cache=True)
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

    Same iteration as Mandelbulb but with fixed c parameter.
    Returns a 3D float64 array of iteration counts (0 = inside set).
    """
    nx = xs.shape[0]
    ny = ys.shape[0]
    nz = zs.shape[0]
    field = np.zeros((nx, ny, nz), dtype=np.float64)

    for ix in range(nx):
        x0 = xs[ix]
        for iy in range(ny):
            y0 = ys[iy]
            for iz in range(nz):
                z0 = zs[iz]
                field[ix, iy, iz] = _julia3d_voxel(
                    x0, y0, z0, max_iterations, power,
                    c_re, c_im, c_z,
                )
    return field


@numba.njit(cache=True)
def _julia3d_voxel(
    x0: float,
    y0: float,
    z0: float,
    max_iterations: int,
    power: float,
    c_re: float,
    c_im: float,
    c_z: float,
) -> float:
    """Evaluate a single voxel for the Julia 3D set."""
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
        x = r_n * np.sin(theta_n) * np.cos(phi_n) + c_re
        y = r_n * np.sin(theta_n) * np.sin(phi_n) + c_im
        z = r_n * np.cos(theta_n) + c_z
    return 0.0
