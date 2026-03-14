"""Kepler orbit generator.

Generates an elliptical orbit from classical orbital elements (semi-major axis,
eccentricity). The orbit is a parametric curve in the orbital plane.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.physics import MIN_CURVE_POINTS

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TUBE_RADIUS = 0.04

_DEFAULT_SEMI_MAJOR_AXIS = 1.0
_DEFAULT_ECCENTRICITY = 0.5
_DEFAULT_INCLINATION = 0.0


def _validate_params(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    curve_points: int,
) -> None:
    """Validate Kepler orbit parameters."""
    if semi_major_axis <= 0:
        raise ValueError(
            f"semi_major_axis must be positive, got {semi_major_axis}"
        )
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError(
            f"eccentricity must be in [0, 1), got {eccentricity}"
        )
    if not np.isfinite(inclination):
        raise ValueError(f"inclination must be finite, got {inclination}")
    if curve_points < MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {MIN_CURVE_POINTS}, got {curve_points}"
        )


def _compute_orbit_points(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    num_points: int,
) -> np.ndarray:
    """Compute points on an elliptical Kepler orbit."""
    # True anomaly from 0 to 2*pi
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # Radius from the focus (polar equation of an ellipse)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    r = semi_latus_rectum / (1.0 + eccentricity * np.cos(theta))

    # Orbital plane coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Apply inclination rotation around x-axis
    cos_i = np.cos(inclination)
    sin_i = np.sin(inclination)
    z = y * sin_i
    y = y * cos_i

    return np.column_stack([x, y, z]).astype(np.float64)


@register
class KeplerOrbitGenerator(GeneratorBase):
    """Elliptical Kepler orbit generator."""

    name = "kepler_orbit"
    category = "physics"
    aliases = ()
    description = "Elliptical orbit from classical orbital elements"
    resolution_params = {
        "curve_points": "Number of sample points along the orbit",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "semi_major_axis": _DEFAULT_SEMI_MAJOR_AXIS,
            "eccentricity": _DEFAULT_ECCENTRICITY,
            "inclination": _DEFAULT_INCLINATION,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Kepler orbit curve.

        Note: seed is accepted for interface conformance but does not affect
        output — the orbit is fully determined by the explicit parameters.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        semi_major_axis = float(merged["semi_major_axis"])
        eccentricity = float(merged["eccentricity"])
        inclination = float(merged["inclination"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(semi_major_axis, eccentricity, inclination, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_orbit_points(
            semi_major_axis, eccentricity, inclination, curve_points
        )

        curve = Curve(points=points, closed=True)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated kepler_orbit: a=%.2f, e=%.4f, i=%.2f, points=%d",
            semi_major_axis, eccentricity, inclination, curve_points,
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
        """Return the recommended representation for Kepler orbits."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
