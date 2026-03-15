"""Clifford attractor generator (iterated map).

The Clifford attractor is a 2D iterated map producing fractal point clouds.
Equations:
    x_{n+1} = sin(a * y_n) + c * cos(a * x_n)
    y_{n+1} = sin(b * x_n) + d * cos(b * y_n)

Extended to 3D by using a scaled iteration count as the z-coordinate.
Default parameters: a=-1.4, b=1.6, c=1.0, d=0.7.

This generator uses direct iteration (not ODE integration) and produces
a point cloud with SPARSE_SHELL representation.
"""

import logging
import math
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.attractors._base import compute_bounding_box

logger = logging.getLogger(__name__)

DEFAULT_NUM_POINTS = 500_000
MIN_NUM_POINTS = 100


def _iterate_clifford(
    a: float, b: float, c: float, d: float,
    x0: float, y0: float, num_points: int,
) -> np.ndarray:
    """Iterate the Clifford map and return an (N, 3) point array."""
    points = np.empty((num_points, 3), dtype=np.float64)
    x, y = x0, y0

    for i in range(num_points):
        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = float(i) / num_points
        x_new = math.sin(a * y) + c * math.cos(a * x)
        y_new = math.sin(b * x) + d * math.cos(b * y)
        x, y = x_new, y_new

    return points


@register
class CliffordGenerator(GeneratorBase):
    """Clifford attractor iterated map generator."""

    name = "clifford"
    aliases = ("clifford_attractor",)
    description = "Clifford 2D iterated-map attractor (point cloud)"
    category = "attractors"

    resolution_params = {"num_points": "Number of iteration points to generate"}
    _resolution_defaults = {"num_points": DEFAULT_NUM_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Clifford attractor."""
        return {
            "a": -1.4,
            "b": 1.6,
            "c": 1.0,
            "d": 0.7,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Clifford attractor point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_points = int(
            resolution_kwargs.get("num_points", DEFAULT_NUM_POINTS)
        )
        if num_points < MIN_NUM_POINTS:
            raise ValueError(
                f"num_points must be >= {MIN_NUM_POINTS}, got {num_points}"
            )

        a = float(merged["a"])
        b = float(merged["b"])
        c = float(merged["c"])
        d = float(merged["d"])

        rng = default_rng(seed)
        x0 = rng.normal(scale=0.1)
        y0 = rng.normal(scale=0.1)

        points = _iterate_clifford(a, b, c, d, x0, y0, num_points)

        merged["num_points"] = num_points
        cloud = PointCloud(points=points)
        bbox = compute_bounding_box(points)

        logger.info(
            "Generated Clifford attractor: %d points", num_points,
        )

        return MathObject(
            point_cloud=cloud,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SPARSE_SHELL as the default representation."""
        return RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
