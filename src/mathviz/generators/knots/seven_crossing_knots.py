"""Seven-crossing knot generator.

Generates knots with crossing number 7 (7_1 through 7_7) using
harmonic parametric representations. A knot_index parameter selects
which of the seven knot types to produce.
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
_DEFAULT_KNOT_INDEX = 1
_DEFAULT_SCALE = 1.0
_MIN_CURVE_POINTS = 16
_MIN_KNOT_INDEX = 1
_MAX_KNOT_INDEX = 7

# Harmonic coefficients for each 7-crossing knot.
# Each entry: list of (amplitude, frequency, phase) for x, y, z components.
# These produce topologically distinct knots with 7 crossings.
_KNOT_COEFFICIENTS: dict[int, dict[str, list[tuple[float, int, float]]]] = {
    1: {  # 7_1: torus knot (2, 7)
        "x": [(2.0, 1, 0.0), (1.0, 6, 0.0)],
        "y": [(2.0, 1, 1.5708), (1.0, 6, 1.5708)],
        "z": [(1.0, 7, 0.0)],
    },
    2: {  # 7_2
        "x": [(2.0, 1, 0.0), (0.8, 2, 0.5), (0.3, 4, 1.0)],
        "y": [(2.0, 1, 1.5708), (0.8, 2, 2.0708), (0.3, 4, 2.5708)],
        "z": [(0.8, 3, 0.3), (0.5, 5, 0.7)],
    },
    3: {  # 7_3
        "x": [(2.0, 1, 0.0), (0.7, 3, 0.3), (0.4, 5, 0.8)],
        "y": [(2.0, 1, 1.5708), (0.7, 3, 1.8708), (0.4, 5, 2.3708)],
        "z": [(0.9, 2, 0.5), (0.4, 4, 1.2)],
    },
    4: {  # 7_4
        "x": [(2.0, 1, 0.0), (0.6, 2, 0.8), (0.5, 4, 0.2)],
        "y": [(2.0, 1, 1.5708), (0.6, 2, 2.3708), (0.5, 4, 1.7708)],
        "z": [(0.7, 3, 0.6), (0.6, 5, 1.1)],
    },
    5: {  # 7_5
        "x": [(2.0, 1, 0.0), (0.9, 3, 0.4), (0.3, 6, 0.9)],
        "y": [(2.0, 1, 1.5708), (0.9, 3, 1.9708), (0.3, 6, 2.4708)],
        "z": [(0.8, 2, 0.2), (0.5, 5, 1.5)],
    },
    6: {  # 7_6
        "x": [(2.0, 1, 0.0), (0.5, 2, 1.0), (0.6, 3, 0.3)],
        "y": [(2.0, 1, 1.5708), (0.5, 2, 2.5708), (0.6, 3, 1.8708)],
        "z": [(0.9, 4, 0.4), (0.4, 6, 0.8)],
    },
    7: {  # 7_7
        "x": [(2.0, 1, 0.0), (0.7, 2, 0.6), (0.4, 3, 1.2)],
        "y": [(2.0, 1, 1.5708), (0.7, 2, 2.1708), (0.4, 3, 2.7708)],
        "z": [(0.6, 4, 0.5), (0.5, 5, 0.9), (0.3, 7, 1.3)],
    },
}


def _evaluate_harmonic(
    t: np.ndarray, coeffs: list[tuple[float, int, float]]
) -> np.ndarray:
    """Evaluate a sum of harmonic terms: sum(amp * cos(freq * t + phase))."""
    result = np.zeros_like(t)
    for amplitude, frequency, phase in coeffs:
        result += amplitude * np.cos(frequency * t + phase)
    return result


def _compute_seven_crossing_points(
    knot_index: int, scale: float, num_points: int
) -> np.ndarray:
    """Compute points for a specific seven-crossing knot."""
    coeffs = _KNOT_COEFFICIENTS[knot_index]
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    x = _evaluate_harmonic(t, coeffs["x"])
    y = _evaluate_harmonic(t, coeffs["y"])
    z = _evaluate_harmonic(t, coeffs["z"])

    return np.column_stack([x, y, z]).astype(np.float64) * scale


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from curve points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    knot_index: int, scale: float, curve_points: int
) -> None:
    """Validate seven-crossing knot parameters."""
    if knot_index < _MIN_KNOT_INDEX or knot_index > _MAX_KNOT_INDEX:
        raise ValueError(
            f"knot_index must be between {_MIN_KNOT_INDEX} and "
            f"{_MAX_KNOT_INDEX}, got {knot_index}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class SevenCrossingKnotsGenerator(GeneratorBase):
    """Seven-crossing knots (7_1 through 7_7) generator."""

    name = "seven_crossing_knots"
    category = "knots"
    aliases = ()
    description = "Knots with crossing number 7, selectable by knot_index"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "knot_index": _DEFAULT_KNOT_INDEX,
            "scale": _DEFAULT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a seven-crossing knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        knot_index = int(merged["knot_index"])
        scale = float(merged["scale"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(knot_index, scale, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_seven_crossing_points(
            knot_index, scale, curve_points,
        )

        curve = Curve(points=points, closed=True)
        bbox = _compute_bounding_box(points)

        logger.info(
            "Generated 7_%d knot: scale=%.2f, points=%d",
            knot_index, scale, curve_points,
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
        """Return the recommended representation for seven-crossing knots."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
