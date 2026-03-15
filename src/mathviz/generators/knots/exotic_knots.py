"""Exotic single-component knot generators: pretzel knot and cinquefoil knot.

The pretzel knot is a (p, q) pretzel knot parameterization. The cinquefoil
knot is the (2, 5) torus knot rendered as a standalone generator.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots._knot_utils import (
    DEFAULT_TUBE_RADIUS,
    extract_curve_points,
    validate_curve_points,
)

logger = logging.getLogger(__name__)


def _compute_pretzel_knot_points(
    p: int, q: int, num_points: int,
) -> np.ndarray:
    """Compute points on a (p, q) pretzel knot curve.

    Uses a parametric embedding that creates p left-hand twists and
    q right-hand twists, forming a closed curve in 3-space.
    """
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # Pretzel knot: a curve with p+q lobes, alternating twist directions
    total = p + q
    r = 2.0 + np.cos(total * t)
    x = r * np.cos(t) + 0.5 * np.cos(p * t)
    y = r * np.sin(t) + 0.5 * np.sin(p * t)
    z = np.sin(q * t) + 0.3 * np.sin(total * t)

    return np.column_stack([x, y, z]).astype(np.float64)


def _compute_cinquefoil_knot_points(num_points: int) -> np.ndarray:
    """Compute points on a (2, 5) torus knot — a five-lobed star knot."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    major_r = 1.0
    minor_r = 0.4
    r = major_r + minor_r * np.cos(5.0 * t)
    x = r * np.cos(2.0 * t)
    y = r * np.sin(2.0 * t)
    z = minor_r * np.sin(5.0 * t)

    return np.column_stack([x, y, z]).astype(np.float64)


@register
class PretzelKnotGenerator(GeneratorBase):
    """Pretzel knot with configurable twist counts."""

    name = "pretzel_knot"
    category = "knots"
    aliases = ()
    description = "Pretzel knot with p left-hand and q right-hand twists"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }
    _resolution_defaults = {"curve_points": 1024}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the pretzel knot."""
        return {"p": 2, "q": 3}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a pretzel knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        merged, curve_points = extract_curve_points(
            merged, resolution_kwargs, 1024,
        )

        p = int(merged["p"])
        q = int(merged["q"])

        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        validate_curve_points(curve_points)

        merged["curve_points"] = curve_points

        points = _compute_pretzel_knot_points(p, q, curve_points)
        curve = Curve(points=points, closed=True)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated pretzel knot: p=%d, q=%d, points=%d",
            p, q, curve_points,
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
        """Return the recommended representation for the pretzel knot."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=DEFAULT_TUBE_RADIUS,
        )


@register
class CinquefoilKnotGenerator(GeneratorBase):
    """Cinquefoil knot — the (2,5) torus knot as a five-lobed star."""

    name = "cinquefoil_knot"
    category = "knots"
    aliases = ()
    description = "Cinquefoil (2,5) torus knot — a five-lobed star knot"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }
    _resolution_defaults = {"curve_points": 1024}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the cinquefoil knot."""
        return {}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a cinquefoil knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        merged, curve_points = extract_curve_points(
            merged, resolution_kwargs, 1024,
        )
        validate_curve_points(curve_points)

        merged["curve_points"] = curve_points

        points = _compute_cinquefoil_knot_points(curve_points)
        curve = Curve(points=points, closed=True)
        bbox = BoundingBox.from_points(points)

        logger.info("Generated cinquefoil knot: points=%d", curve_points)

        return MathObject(
            curves=[curve],
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the cinquefoil knot."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=DEFAULT_TUBE_RADIUS,
        )
