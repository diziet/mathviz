"""Fibonacci spiral generator.

A Fibonacci (golden) spiral approximation where the radius grows
according to the golden ratio. Extended to 3D with a height component.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TUBE_RADIUS = 0.03
_MIN_CURVE_POINTS = 16

_DEFAULT_TURNS = 4.0
_DEFAULT_HEIGHT = 0.5
_DEFAULT_SCALE = 1.0
_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0


def _compute_fibonacci_spiral_points(
    turns: float, height: float, scale: float, num_points: int
) -> np.ndarray:
    """Compute points on a 3D Fibonacci (golden) spiral."""
    # Growth rate derived from golden ratio: one full turn scales by phi
    growth_rate = np.log(_GOLDEN_RATIO) / (2.0 * np.pi)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, num_points)

    r = np.exp(growth_rate * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z_range = height * turns if turns > 0 else 0.0
    z = np.linspace(0.0, z_range, num_points)

    return np.column_stack([x, y, z]).astype(np.float64) * scale


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from curve points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    turns: float, scale: float, curve_points: int
) -> None:
    """Validate Fibonacci spiral parameters."""
    if turns <= 0:
        raise ValueError(f"turns must be positive, got {turns}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class FibonacciSpiralGenerator(GeneratorBase):
    """3D Fibonacci (golden) spiral generator."""

    name = "fibonacci_spiral"
    category = "curves"
    aliases = ("golden_spiral",)
    description = "Golden-ratio spiral with exponential radius growth"
    resolution_params = {
        "curve_points": "Number of sample points along the spiral",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "turns": _DEFAULT_TURNS,
            "height": _DEFAULT_HEIGHT,
            "scale": _DEFAULT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Fibonacci spiral curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        turns = float(merged["turns"])
        height = float(merged["height"])
        scale = float(merged["scale"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(turns, scale, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_fibonacci_spiral_points(
            turns, height, scale, curve_points,
        )

        # Spirals are open curves
        curve = Curve(points=points, closed=False)
        bbox = _compute_bounding_box(points)

        logger.info(
            "Generated Fibonacci spiral: turns=%.1f, points=%d",
            turns, curve_points,
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
        """Return the recommended representation for Fibonacci spirals."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
