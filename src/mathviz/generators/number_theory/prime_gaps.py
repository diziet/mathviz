"""Prime gaps generator.

Maps consecutive prime gaps to a 3D ribbon. Each gap is placed along the
x-axis, the gap size determines the y-coordinate, and z varies smoothly
to create a ribbon shape. Default representation: WEIGHTED_CLOUD.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.number_theory._primes import first_n_primes

logger = logging.getLogger(__name__)

_DEFAULT_NUM_PRIMES = 500


def _validate_params(num_primes: int) -> None:
    """Validate prime gaps parameters."""
    if num_primes < 2:
        raise ValueError(f"num_primes must be >= 2, got {num_primes}")


def _build_prime_gaps_cloud(
    num_primes: int, x_spacing: float, y_scale: float, ribbon_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D ribbon points, intensities, and gap sizes from prime gaps."""
    primes = first_n_primes(num_primes)
    gaps = np.diff(primes).astype(np.float64)
    num_gaps = len(gaps)

    points = np.zeros((num_gaps, 3), dtype=np.float64)
    points[:, 0] = np.arange(num_gaps, dtype=np.float64) * x_spacing
    points[:, 1] = gaps * y_scale
    # Z varies sinusoidally for a ribbon shape
    t = np.linspace(0.0, 2.0 * np.pi, num_gaps)
    points[:, 2] = np.sin(t) * ribbon_width

    # Intensity proportional to gap size (normalized)
    max_gap = gaps.max() if gaps.max() > 0 else 1.0
    intensities = (gaps / max_gap).astype(np.float64)

    return points, intensities, gaps


@register
class PrimeGapsGenerator(GeneratorBase):
    """Prime gaps: consecutive gaps mapped to a 3D ribbon."""

    name = "prime_gaps"
    category = "number_theory"
    aliases = ()
    description = "Prime gaps visualized as a 3D ribbon"
    resolution_params = {
        "num_primes": "Number of primes to compute gaps between",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for prime gaps."""
        return {
            "x_spacing": 0.1,
            "y_scale": 0.1,
            "ribbon_width": 0.5,
        }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"num_primes": _DEFAULT_NUM_PRIMES}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a prime gaps ribbon point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_primes = int(resolution_kwargs.get("num_primes", _DEFAULT_NUM_PRIMES))
        x_spacing = float(merged["x_spacing"])
        y_scale = float(merged["y_scale"])
        ribbon_width = float(merged["ribbon_width"])
        _validate_params(num_primes)

        points, intensities, _gaps = _build_prime_gaps_cloud(
            num_primes, x_spacing, y_scale, ribbon_width
        )
        cloud = PointCloud(points=points, intensities=intensities)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated prime gaps ribbon with %d gaps from %d primes",
            len(points), num_primes,
        )

        return MathObject(
            point_cloud=cloud,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters={**merged, "num_primes": num_primes},
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return WEIGHTED_CLOUD as the default representation."""
        return RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD)
