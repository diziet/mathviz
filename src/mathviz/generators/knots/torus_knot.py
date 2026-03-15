"""Torus knot generator with alias support for trefoil and cinquefoil.

A torus knot winds p times around a torus in one direction and q times in
the other. The trefoil is (2, 3) and the cinquefoil is (2, 5).
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_P = 2
_DEFAULT_Q = 3
_DEFAULT_MAJOR_RADIUS = 1.0
_DEFAULT_MINOR_RADIUS = 0.4
_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TUBE_RADIUS = 0.1
_MIN_CURVE_POINTS = 16

# Alias definitions: alias_name -> default param overrides
_ALIASES: dict[str, dict[str, int]] = {
    "trefoil": {"p": 2, "q": 3},
    "cinquefoil": {"p": 2, "q": 5},
}


def _compute_torus_knot_points(
    p: int,
    q: int,
    major_radius: float,
    minor_radius: float,
    num_points: int,
) -> np.ndarray:
    """Compute points on a (p, q) torus knot curve."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    r = major_radius + minor_radius * np.cos(q * t)
    x = r * np.cos(p * t)
    y = r * np.sin(p * t)
    z = minor_radius * np.sin(q * t)

    return np.column_stack([x, y, z]).astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from curve points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    p: int, q: int, major_radius: float, minor_radius: float, curve_points: int
) -> None:
    """Validate torus knot parameters."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    # p == q is intentionally allowed: it produces a valid closed curve
    # (an unknot wound around the torus), which is still useful for visualization.
    if major_radius <= 0:
        raise ValueError(f"R (major_radius) must be positive, got {major_radius}")
    if minor_radius <= 0:
        raise ValueError(f"r (minor_radius) must be positive, got {minor_radius}")
    if minor_radius >= major_radius:
        raise ValueError(
            f"r must be less than R for a valid torus knot, "
            f"got R={major_radius}, r={minor_radius}"
        )
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class TorusKnotGenerator(GeneratorBase):
    """Torus knot generator with trefoil and cinquefoil aliases."""

    name = "torus_knot"
    category = "knots"
    aliases = ("trefoil", "cinquefoil")
    description = "Torus knot curve with configurable (p, q) winding numbers"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters, adjusted for alias if applicable."""
        defaults: dict[str, Any] = {
            "p": _DEFAULT_P,
            "q": _DEFAULT_Q,
            "R": _DEFAULT_MAJOR_RADIUS,
            "r": _DEFAULT_MINOR_RADIUS,
        }
        if self.resolved_name in _ALIASES:
            defaults.update(_ALIASES[self.resolved_name])
        return defaults

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"curve_points": _DEFAULT_CURVE_POINTS}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a torus knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        p = int(merged["p"])
        q = int(merged["q"])
        major_radius = float(merged["R"])
        minor_radius = float(merged["r"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(p, q, major_radius, minor_radius, curve_points)

        merged["curve_points"] = curve_points

        points = _compute_torus_knot_points(
            p, q, major_radius, minor_radius, curve_points
        )

        curve = Curve(points=points, closed=True)
        bbox = _compute_bounding_box(points)

        logger.info(
            "Generated torus knot: p=%d, q=%d, R=%.2f, r=%.2f, points=%d",
            p, q, major_radius, minor_radius, curve_points,
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
        """Return the recommended representation for torus knots."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
