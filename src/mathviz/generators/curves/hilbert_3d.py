"""3D Hilbert curve generator.

A space-filling curve that visits every cell in a cubic grid exactly once,
using the Skilling transpose algorithm for index-to-coordinate mapping.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_ORDER = 4
_MAX_ORDER = 6
_DEFAULT_SIZE = 1.0
_DEFAULT_TUBE_RADIUS = 0.02
_NDIM = 3


def _d2xyz(d: int, order: int) -> tuple[int, int, int]:
    """Convert a 1D Hilbert index to 3D coordinates.

    Implements the Skilling transpose algorithm (2004) which converts
    a Hilbert distance to N-dimensional coordinates via bit interleaving
    and Gray-code transformations.
    """
    coords = [0, 0, 0]

    # Step 1: Distribute bits of d into coordinates (transpose)
    for s in range(order):
        for dim in range(_NDIM):
            bit_pos = _NDIM * s + (_NDIM - 1 - dim)
            if d & (1 << bit_pos):
                coords[dim] |= 1 << s

    # Step 2: TransposeToAxes — Gray decode and undo rotations
    n = 1 << order

    # Gray decode
    t = coords[_NDIM - 1] >> 1
    for i in range(_NDIM - 1, 0, -1):
        coords[i] ^= coords[i - 1]
    coords[0] ^= t

    # Undo rotations
    q = 2
    while q < n:
        p = q - 1
        for i in range(_NDIM - 1, -1, -1):
            if coords[i] & q:
                coords[0] ^= p
            else:
                t2 = (coords[0] ^ coords[i]) & p
                coords[0] ^= t2
                coords[i] ^= t2
        q <<= 1

    return (coords[0], coords[1], coords[2])


def _generate_hilbert_points(order: int) -> np.ndarray:
    """Generate all 3D Hilbert curve points for the given order."""
    num_points = 8 ** order
    points = np.empty((num_points, 3), dtype=np.float64)

    for d in range(num_points):
        x, y, z = _d2xyz(d, order)
        points[d] = (x, y, z)

    return points


def _validate_params(order: int, size: float) -> None:
    """Validate Hilbert curve parameters."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order > _MAX_ORDER:
        raise ValueError(f"order must be <= {_MAX_ORDER}, got {order}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")


@register
class Hilbert3DGenerator(GeneratorBase):
    """3D Hilbert space-filling curve generator."""

    name = "hilbert_3d"
    category = "curves"
    aliases = ("hilbert_curve_3d",)
    description = "3D Hilbert space-filling curve visiting every grid cell"
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "order": _DEFAULT_ORDER,
            "size": _DEFAULT_SIZE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 3D Hilbert curve.

        Note: this generator is deterministic by construction — the output
        depends only on ``order`` and ``size``. The seed is stored as
        metadata but does not influence the geometry.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        order = int(merged["order"])
        size = float(merged["size"])
        _validate_params(order, size)

        points = _generate_hilbert_points(order)

        # Normalize to [0, size] range
        grid_max = (1 << order) - 1
        if grid_max > 0:
            points = points / grid_max * size

        curve = Curve(points=points.astype(np.float64), closed=False)
        bbox = BoundingBox.from_points(points)

        num_points = len(points)
        logger.info(
            "Generated 3D Hilbert curve: order=%d, points=%d, size=%.2f",
            order, num_points, size,
        )

        return MathObject(
            curves=[curve],
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for 3D Hilbert curves."""
        return RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=_DEFAULT_TUBE_RADIUS,
        )
