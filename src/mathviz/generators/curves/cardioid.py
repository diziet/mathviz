"""Cardioid curve generator.

A cardioid is a heart-shaped curve defined in polar coordinates as
r = a(1 + cos(theta)). Extended to 3D with a configurable height component.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TUBE_RADIUS = 0.04
_MIN_CURVE_POINTS = 16

_DEFAULT_RADIUS = 1.0
_DEFAULT_HEIGHT = 0.3


def _compute_cardioid_points(
    radius: float, height: float, num_points: int
) -> np.ndarray:
    """Compute points on a 3D cardioid curve."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    r = radius * (1.0 + np.cos(t))
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = height * np.sin(t)

    return np.column_stack([x, y, z]).astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from curve points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    radius: float, curve_points: int
) -> None:
    """Validate cardioid parameters."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class CardioidGenerator(GeneratorBase):
    """3D cardioid curve generator."""

    name = "cardioid"
    category = "curves"
    aliases = ()
    description = "Heart-shaped cardioid curve extended to 3D"
    resolution_params = {
        "curve_points": "Number of sample points along the curve",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "radius": _DEFAULT_RADIUS,
            "height": _DEFAULT_HEIGHT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a cardioid curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        radius = float(merged["radius"])
        height = float(merged["height"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(radius, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_cardioid_points(radius, height, curve_points)

        # Cardioid is a closed curve over [0, 2*pi)
        curve = Curve(points=points, closed=True)
        bbox = _compute_bounding_box(points)

        logger.info(
            "Generated cardioid: radius=%.2f, height=%.2f, points=%d",
            radius, height, curve_points,
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
        """Return the recommended representation for cardioid curves."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
