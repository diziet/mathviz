"""Schwarz P implicit surface generator.

The Schwarz P (Primitive) surface is a triply periodic minimal surface defined
by the implicit equation cos(x) + cos(y) + cos(z) = 0. Like the gyroid, it
tiles space; the ``periods`` parameter controls how many unit cells are included.

Note: field evaluation is O(N³) where N is voxel_resolution. High resolutions
(e.g. 256+) will be slow and memory-intensive.
"""

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.implicit._tpms_base import TPMSGeneratorBase
from mathviz.shared.marching_cubes import SpatialBounds


@register
class SchwarzPGenerator(TPMSGeneratorBase):
    """Implicit Schwarz P surface generator.

    Evaluates the Schwarz P scalar field cos(x) + cos(y) + cos(z) on an N³
    voxel grid and extracts the isosurface via marching cubes.

    Seed has no effect on output; the surface is fully deterministic
    for given cell_size, periods, and voxel_resolution.
    """

    name = "schwarz_p"
    aliases = ()
    description = "Triply periodic Schwarz P minimal surface via marching cubes"

    def evaluate_field(
        self, voxel_resolution: int, bounds: SpatialBounds,
    ) -> np.ndarray:
        """Evaluate cos(x) + cos(y) + cos(z) on an N³ voxel grid."""
        n = voxel_resolution
        x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
        y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
        z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        return np.cos(xx) + np.cos(yy) + np.cos(zz)
