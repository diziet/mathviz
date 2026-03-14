"""Figure-eight knot (4_1) generator.

The figure-eight knot is the simplest non-torus knot. It has crossing
number 4 and is amphichiral (equivalent to its mirror image).
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TUBE_RADIUS = 0.1
_DEFAULT_SCALE = 1.0
_MIN_CURVE_POINTS = 16


def _compute_figure_eight_points(
    scale: float, num_points: int
) -> np.ndarray:
    """Compute points on a figure-eight knot curve."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # Standard parametrization of the figure-eight knot
    x = (2.0 + np.cos(2.0 * t)) * np.cos(3.0 * t)
    y = (2.0 + np.cos(2.0 * t)) * np.sin(3.0 * t)
    z = np.sin(4.0 * t)

    points = np.column_stack([x, y, z]).astype(np.float64)
    return points * scale



def _validate_params(scale: float, curve_points: int) -> None:
    """Validate figure-eight knot parameters."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class FigureEightKnotGenerator(GeneratorBase):
    """Figure-eight knot (4_1) generator."""

    name = "figure_eight_knot"
    category = "knots"
    aliases = ()
    description = "Figure-eight knot with crossing number 4"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {"scale": _DEFAULT_SCALE}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a figure-eight knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        scale = float(merged["scale"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(scale, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_figure_eight_points(scale, curve_points)
        curve = Curve(points=points, closed=True)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated figure-eight knot: scale=%.2f, points=%d",
            scale, curve_points,
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
        """Return the recommended representation for the figure-eight knot."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
