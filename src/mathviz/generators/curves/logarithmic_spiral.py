"""Logarithmic spiral generator.

A logarithmic (equiangular) spiral in 3D where the radius grows
exponentially with the angle. The extent scales with the number of turns.
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

_DEFAULT_GROWTH_RATE = 0.15
_DEFAULT_TURNS = 3.0
_DEFAULT_HEIGHT = 1.0
_DEFAULT_SCALE = 1.0


def _compute_logarithmic_spiral_points(
    growth_rate: float,
    turns: float,
    height: float,
    scale: float,
    num_points: int,
) -> np.ndarray:
    """Compute points on a 3D logarithmic spiral."""
    theta = np.linspace(0.0, 2.0 * np.pi * turns, num_points)

    r = np.exp(growth_rate * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = height * theta / (2.0 * np.pi * turns) if turns > 0 else np.zeros_like(theta)

    return np.column_stack([x, y, z]).astype(np.float64) * scale


def _validate_params(
    growth_rate: float, turns: float, height: float, scale: float,
    curve_points: int,
) -> None:
    """Validate logarithmic spiral parameters."""
    if growth_rate <= 0:
        raise ValueError(f"growth_rate must be positive, got {growth_rate}")
    if turns <= 0:
        raise ValueError(f"turns must be positive, got {turns}")
    if not np.isfinite(height):
        raise ValueError(f"height must be finite, got {height}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class LogarithmicSpiralGenerator(GeneratorBase):
    """3D logarithmic spiral generator."""

    name = "logarithmic_spiral"
    category = "curves"
    aliases = ()
    description = "Logarithmic spiral with exponential radius growth"
    resolution_params = {
        "curve_points": "Number of sample points along the spiral",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "growth_rate": _DEFAULT_GROWTH_RATE,
            "turns": _DEFAULT_TURNS,
            "height": _DEFAULT_HEIGHT,
            "scale": _DEFAULT_SCALE,
        }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"curve_points": _DEFAULT_CURVE_POINTS}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a logarithmic spiral curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        growth_rate = float(merged["growth_rate"])
        turns = float(merged["turns"])
        height = float(merged["height"])
        scale = float(merged["scale"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(growth_rate, turns, height, scale, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_logarithmic_spiral_points(
            growth_rate, turns, height, scale, curve_points,
        )

        # Spirals are open curves
        curve = Curve(points=points, closed=False)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated logarithmic spiral: growth=%.3f, turns=%.1f, points=%d",
            growth_rate, turns, curve_points,
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
        """Return the recommended representation for logarithmic spirals."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
