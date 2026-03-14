"""Sacks spiral generator.

Integers are placed on an Archimedean spiral where each integer n is at
angle sqrt(n) * 2π and radius sqrt(n). Prime numbers are marked with
elevated z-values and higher intensity weights.
Default representation: WEIGHTED_CLOUD.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.number_theory._primes import is_prime_array

logger = logging.getLogger(__name__)


def _validate_params(num_points: int, prime_height: float) -> None:
    """Validate Sacks spiral parameters."""
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    if prime_height < 0:
        raise ValueError(f"prime_height must be >= 0, got {prime_height}")


def _build_sacks_cloud(
    num_points: int, prime_height: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3D points and intensities for the Sacks spiral."""
    n = np.arange(num_points, dtype=np.float64)
    theta = np.sqrt(n) * 2.0 * np.pi
    radius = np.sqrt(n) * scale

    primality = is_prime_array(num_points)

    points = np.zeros((num_points, 3), dtype=np.float64)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = np.where(primality, prime_height, 0.0)

    intensities = np.where(primality, 1.0, 0.2).astype(np.float64)
    return points, intensities


@register
class SacksSpiralGenerator(GeneratorBase):
    """Sacks spiral: Archimedean spiral with primes marked."""

    name = "sacks_spiral"
    category = "number_theory"
    aliases = ("sacks",)
    description = "Sacks spiral — primes on an Archimedean spiral"
    resolution_params = {
        "num_points": "Number of integers to place on the spiral",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Sacks spiral."""
        return {
            "prime_height": 1.0,
            "scale": 0.1,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Sacks spiral point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_points = int(resolution_kwargs.get("num_points", 1000))
        prime_height = float(merged["prime_height"])
        scale = float(merged["scale"])
        _validate_params(num_points, prime_height)

        points, intensities = _build_sacks_cloud(num_points, prime_height, scale)
        cloud = PointCloud(points=points, intensities=intensities)
        bbox = BoundingBox.from_points(points)

        logger.info("Generated Sacks spiral with %d points", num_points)

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
