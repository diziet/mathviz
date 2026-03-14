"""2D simplex noise implementation.

Seed-controlled simplex noise for procedural generation. Uses a permutation
table derived from the seed for reproducibility. Based on the simplex noise
algorithm by Ken Perlin.
"""

import numpy as np
from numpy.random import default_rng

# Skew/unskew constants for 2D simplex
_F2 = 0.5 * (np.sqrt(3.0) - 1.0)
_G2 = (3.0 - np.sqrt(3.0)) / 6.0

# Gradient vectors for 2D
_GRADIENTS_2D = np.array(
    [
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [1, 0], [-1, 0], [0, 1], [0, -1],
    ],
    dtype=np.float64,
)


def _build_perm_table(seed: int) -> np.ndarray:
    """Build a 512-element permutation table from a seed."""
    rng = default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    rng.shuffle(perm)
    return np.concatenate([perm, perm])


def _noise2d_single(x: float, y: float, perm: np.ndarray) -> float:
    """Evaluate 2D simplex noise at a single point."""
    s = (x + y) * _F2
    i = int(np.floor(x + s))
    j = int(np.floor(y + s))

    t = (i + j) * _G2
    x0 = x - (i - t)
    y0 = y - (j - t)

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1

    x1 = x0 - i1 + _G2
    y1 = y0 - j1 + _G2
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2

    ii = i & 255
    jj = j & 255

    n = 0.0
    for corner_idx, (dx, dy) in enumerate([(x0, y0), (x1, y1), (x2, y2)]):
        t_val = 0.5 - dx * dx - dy * dy
        if t_val > 0:
            if corner_idx == 0:
                gi = perm[ii + perm[jj]] % 8
            elif corner_idx == 1:
                gi = perm[ii + i1 + perm[jj + j1]] % 8
            else:
                gi = perm[ii + 1 + perm[jj + 1]] % 8
            grad = _GRADIENTS_2D[gi]
            t_val *= t_val
            n += t_val * t_val * (grad[0] * dx + grad[1] * dy)

    return 70.0 * n


def evaluate_2d_grid(
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    seed: int,
) -> np.ndarray:
    """Evaluate 2D simplex noise on a grid.

    Returns a (resolution, resolution) float64 array with values in ~[-1, 1].
    """
    perm = _build_perm_table(seed)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)

    field = np.empty((resolution, resolution), dtype=np.float64)
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            field[row_idx, col_idx] = _noise2d_single(x, y, perm)

    return field


def evaluate_2d_grid_octaves(
    resolution: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    seed: int,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Evaluate multi-octave 2D simplex noise on a grid.

    Sums multiple octaves with increasing frequency and decreasing amplitude
    for fractal-like detail. Returns (resolution, resolution) float64 array.
    """
    field = np.zeros((resolution, resolution), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0

    for octave_idx in range(octaves):
        octave_seed = seed + octave_idx * 1000
        scale = lacunarity ** octave_idx
        scaled_x = (x_range[0] * scale, x_range[1] * scale)
        scaled_y = (y_range[0] * scale, y_range[1] * scale)

        layer = evaluate_2d_grid(resolution, scaled_x, scaled_y, octave_seed)
        field += amplitude * layer
        total_amplitude += amplitude
        amplitude *= persistence

    return field / total_amplitude
