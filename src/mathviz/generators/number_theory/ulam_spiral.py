"""Ulam spiral generator.

Integers are placed on a rectangular spiral grid. Prime numbers are
elevated in z to create a 3D point cloud with per-point intensity weights.
Default representation: WEIGHTED_CLOUD.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.number_theory._primes import (
    is_prime_array,
    validate_point_cloud_params,
)

logger = logging.getLogger(__name__)

_DEFAULT_NUM_POINTS = 1000

# Direction vectors for the rectangular spiral: right, up, left, down
_DIRECTIONS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)


def _compute_spiral_positions(num_points: int) -> np.ndarray:
    """Compute 2D grid positions for integers 0..num_points-1 on a rectangular spiral."""
    positions = np.zeros((num_points, 2), dtype=np.int64)
    x, y = 0, 0
    direction = 0
    steps_in_leg = 1
    steps_taken = 0
    leg_count = 0

    for i in range(1, num_points):
        dx, dy = _DIRECTIONS[direction]
        x += dx
        y += dy
        positions[i] = [x, y]
        steps_taken += 1

        if steps_taken == steps_in_leg:
            steps_taken = 0
            direction = (direction + 1) % 4
            leg_count += 1
            if leg_count % 2 == 0:
                steps_in_leg += 1

    return positions


def _build_ulam_cloud(
    num_points: int, prime_height: float, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3D points and intensities for the Ulam spiral."""
    positions = _compute_spiral_positions(num_points)
    primality = is_prime_array(num_points)

    points = np.zeros((num_points, 3), dtype=np.float64)
    points[:, 0] = positions[:, 0].astype(np.float64) * spacing
    points[:, 1] = positions[:, 1].astype(np.float64) * spacing
    points[:, 2] = np.where(primality, prime_height, 0.0)

    intensities = np.where(primality, 1.0, 0.2).astype(np.float64)
    return points, intensities


@register
class UlamSpiralGenerator(GeneratorBase):
    """Ulam spiral: integers on a rectangular grid with primes elevated."""

    name = "ulam_spiral"
    category = "number_theory"
    aliases = ("ulam",)
    description = "Ulam spiral with primes elevated in z"
    resolution_params = {
        "num_points": "Number of integers to place on the spiral",
    }
    _resolution_defaults = {"num_points": _DEFAULT_NUM_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Ulam spiral."""
        return {
            "prime_height": 1.0,
            "spacing": 0.1,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an Ulam spiral point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_points = int(resolution_kwargs.get("num_points", _DEFAULT_NUM_POINTS))
        prime_height = float(merged["prime_height"])
        spacing = float(merged["spacing"])
        validate_point_cloud_params(num_points, prime_height)

        points, intensities = _build_ulam_cloud(num_points, prime_height, spacing)
        cloud = PointCloud(points=points, intensities=intensities)
        bbox = BoundingBox.from_points(points)

        logger.info("Generated Ulam spiral with %d points", num_points)

        return MathObject(
            point_cloud=cloud,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters={**merged, "num_points": num_points},
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return WEIGHTED_CLOUD as the default representation."""
        return RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD)
