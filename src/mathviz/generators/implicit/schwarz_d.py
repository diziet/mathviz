"""Schwarz D implicit surface generator.

The Schwarz D (Diamond) surface is a triply periodic minimal surface defined by
the implicit equation:
    sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
    + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = 0

Like the gyroid and Schwarz P, it tiles space; the ``periods`` parameter
controls how many unit cells are included.

Note: field evaluation is O(N³) where N is voxel_resolution.
"""

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.implicit._tpms_base import TPMSGeneratorBase
from mathviz.shared.marching_cubes import SpatialBounds


@register
class SchwarzDGenerator(TPMSGeneratorBase):
    """Implicit Schwarz D surface generator.

    Evaluates the Schwarz D (Diamond) scalar field on an N³ voxel grid and
    extracts the isosurface via marching cubes. The Diamond TPMS has the
    implicit form: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
    + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = 0.

    Seed has no effect on output; the surface is fully deterministic
    for given cell_size, periods, and voxel_resolution.
    """

    name = "schwarz_d"
    aliases = ()
    description = "Triply periodic Schwarz D minimal surface via marching cubes"

    def evaluate_field(
        self, voxel_resolution: int, bounds: SpatialBounds,
    ) -> np.ndarray:
        """Evaluate the Schwarz D field on an N³ voxel grid."""
        n = voxel_resolution
        x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
        y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
        z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        sx, cx = np.sin(xx), np.cos(xx)
        sy, cy = np.sin(yy), np.cos(yy)
        sz, cz = np.sin(zz), np.cos(zz)

        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
