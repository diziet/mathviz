"""Gyroid implicit surface generator.

The gyroid is a triply periodic minimal surface defined by the implicit equation
sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0. It tiles space infinitely;
the ``periods`` parameter controls how many unit cells are included.

Note: field evaluation is O(N³) where N is voxel_resolution. High resolutions
(e.g. 256+) will be slow and memory-intensive.
"""

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.implicit._tpms_base import TPMSGeneratorBase
from mathviz.shared.marching_cubes import SpatialBounds


@register
class GyroidGenerator(TPMSGeneratorBase):
    """Implicit gyroid surface generator.

    Evaluates the gyroid scalar field sin(x)cos(y) + sin(y)cos(z) +
    sin(z)cos(x) on an N³ voxel grid and extracts the isosurface via
    marching cubes. Cost is O(N³) in voxel_resolution.

    Seed has no effect on output; the gyroid is fully deterministic
    for given periods and voxel_resolution.
    """

    name = "gyroid"
    aliases = ()
    description = "Triply periodic gyroid minimal surface via marching cubes"

    def evaluate_field(
        self, voxel_resolution: int, bounds: SpatialBounds,
    ) -> np.ndarray:
        """Evaluate sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) on an N³ grid."""
        n = voxel_resolution
        x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
        y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
        z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        return (
            np.sin(xx) * np.cos(yy)
            + np.sin(yy) * np.cos(zz)
            + np.sin(zz) * np.cos(xx)
        )
