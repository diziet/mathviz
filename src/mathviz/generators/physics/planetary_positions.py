"""Planetary positions generator.

Renders the solar system at a given epoch with orbital trails (Curves) and
body positions. Uses simplified Keplerian orbital elements for the eight
major planets.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.physics import MIN_CURVE_POINTS

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 512
_DEFAULT_TUBE_RADIUS = 0.02

# J2000.0 epoch in Julian Date
_J2000_EPOCH = 2451545.0
_DEFAULT_EPOCH_JD = _J2000_EPOCH

# Degrees to radians
_DEG2RAD = np.pi / 180.0


@dataclass
class _OrbitalElements:
    """Simplified Keplerian orbital elements for a planet."""

    name: str
    semi_major_axis_au: float
    eccentricity: float
    inclination_deg: float
    long_ascending_node_deg: float
    arg_perihelion_deg: float
    mean_longitude_deg_j2000: float
    orbital_period_days: float


# Simplified J2000 mean orbital elements for the 8 planets
_PLANETS = [
    _OrbitalElements("Mercury", 0.387, 0.2056, 7.00, 48.33, 29.12, 174.79, 87.97),
    _OrbitalElements("Venus", 0.723, 0.0068, 3.39, 76.68, 54.88, 50.42, 224.70),
    _OrbitalElements("Earth", 1.000, 0.0167, 0.00, 0.0, 102.94, 100.46, 365.26),
    _OrbitalElements("Mars", 1.524, 0.0934, 1.85, 49.56, 286.50, 355.45, 686.97),
    _OrbitalElements("Jupiter", 5.203, 0.0484, 1.31, 100.46, 275.07, 34.40, 4332.59),
    _OrbitalElements("Saturn", 9.537, 0.0539, 2.49, 113.72, 336.01, 49.94, 10759.22),
    _OrbitalElements("Uranus", 19.19, 0.0473, 0.77, 74.00, 96.54, 313.23, 30688.5),
    _OrbitalElements("Neptune", 30.07, 0.0086, 1.77, 131.72, 273.19, 304.88, 60182.0),
]


_KEPLER_MAX_ITERATIONS = 50
_KEPLER_TOLERANCE = 1e-12


def _solve_kepler_equation(mean_anomaly: float, eccentricity: float) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    eccentric_anomaly = mean_anomaly
    for _ in range(_KEPLER_MAX_ITERATIONS):
        delta = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= delta / derivative
        if abs(delta) < _KEPLER_TOLERANCE:
            return float(eccentric_anomaly)

    logger.warning(
        "Kepler equation did not converge after %d iterations "
        "(M=%.6f, e=%.6f, residual=%.2e)",
        _KEPLER_MAX_ITERATIONS, mean_anomaly, eccentricity, abs(delta),
    )
    return float(eccentric_anomaly)


def _compute_planet_orbit(
    elements: _OrbitalElements, num_points: int
) -> np.ndarray:
    """Compute full orbital path for a planet as 3D points."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    semi_latus = elements.semi_major_axis_au * (1.0 - elements.eccentricity**2)
    r = semi_latus / (1.0 + elements.eccentricity * np.cos(theta))

    # Orbital plane
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)

    # Rotation angles
    omega = elements.arg_perihelion_deg * _DEG2RAD
    big_omega = elements.long_ascending_node_deg * _DEG2RAD
    inc = elements.inclination_deg * _DEG2RAD

    return _rotate_orbital_to_ecliptic(x_orb, y_orb, omega, big_omega, inc)


def _rotate_orbital_to_ecliptic(
    x_orb: np.ndarray,
    y_orb: np.ndarray,
    omega: float,
    big_omega: float,
    inc: float,
) -> np.ndarray:
    """Rotate orbital plane coordinates to ecliptic frame."""
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    cos_o = np.cos(big_omega)
    sin_o = np.sin(big_omega)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)

    x = (
        (cos_w * cos_o - sin_w * sin_o * cos_i) * x_orb
        + (-sin_w * cos_o - cos_w * sin_o * cos_i) * y_orb
    )
    y = (
        (cos_w * sin_o + sin_w * cos_o * cos_i) * x_orb
        + (-sin_w * sin_o + cos_w * cos_o * cos_i) * y_orb
    )
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    return np.column_stack([x, y, z]).astype(np.float64)


def _compute_planet_position(
    elements: _OrbitalElements, epoch_jd: float
) -> np.ndarray:
    """Compute planet position at a given Julian Date."""
    days_since_j2000 = epoch_jd - _J2000_EPOCH
    mean_anomaly_deg = (
        elements.mean_longitude_deg_j2000
        - elements.arg_perihelion_deg
        + 360.0 * days_since_j2000 / elements.orbital_period_days
    )
    mean_anomaly = (mean_anomaly_deg % 360.0) * _DEG2RAD

    eccentric_anomaly = _solve_kepler_equation(
        mean_anomaly, elements.eccentricity
    )

    # True anomaly
    cos_e = np.cos(eccentric_anomaly)
    sin_e = np.sin(eccentric_anomaly)
    ecc = elements.eccentricity
    true_anomaly = np.arctan2(
        np.sqrt(1.0 - ecc**2) * sin_e, cos_e - ecc
    )

    r = elements.semi_major_axis_au * (1.0 - ecc * cos_e)

    x_orb = np.array([r * np.cos(true_anomaly)])
    y_orb = np.array([r * np.sin(true_anomaly)])

    omega = elements.arg_perihelion_deg * _DEG2RAD
    big_omega = elements.long_ascending_node_deg * _DEG2RAD
    inc = elements.inclination_deg * _DEG2RAD

    return _rotate_orbital_to_ecliptic(x_orb, y_orb, omega, big_omega, inc)


def _validate_params(epoch_jd: float, curve_points: int) -> None:
    """Validate planetary positions parameters."""
    if not np.isfinite(epoch_jd):
        raise ValueError(f"epoch_jd must be finite, got {epoch_jd}")
    if curve_points < MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class PlanetaryPositionsGenerator(GeneratorBase):
    """Solar system planetary orbits and positions generator."""

    name = "planetary_positions"
    category = "physics"
    aliases = ("solar_system",)
    description = "Solar system orbits and planet positions at a given epoch"
    resolution_params = {
        "curve_points": "Number of sample points per orbital curve",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "epoch_jd": _DEFAULT_EPOCH_JD,
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
        """Generate planetary orbits and positions.

        Note: seed is accepted for interface conformance but does not affect
        output — positions are fully determined by the epoch parameter.
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

        epoch_jd = float(merged["epoch_jd"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(epoch_jd, curve_points)
        merged["curve_points"] = curve_points

        curves: list[Curve] = []
        position_points: list[np.ndarray] = []

        for planet in _PLANETS:
            orbit_points = _compute_planet_orbit(planet, curve_points)
            curves.append(Curve(points=orbit_points, closed=True))

        # Collect planet positions at epoch as a PointCloud (not Curves,
        # since single-point curves crash tube thickening)
        for planet in _PLANETS:
            position = _compute_planet_position(planet, epoch_jd)
            position_points.append(position)

        position_cloud = PointCloud(
            points=np.vstack(position_points).astype(np.float64)
        )

        all_points = np.vstack(
            [c.points for c in curves] + [position_cloud.points]
        )
        bbox = BoundingBox.from_points(all_points)

        logger.info(
            "Generated planetary_positions: planets=%d, epoch_jd=%.1f, "
            "points_per_orbit=%d",
            len(_PLANETS), epoch_jd, curve_points,
        )

        return MathObject(
            curves=curves,
            point_cloud=position_cloud,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for planetary orbits."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
