"""Magnetic field line generator.

Generates 3D magnetic field line visualizations for dipole and quadrupole
configurations. Field lines are computed by numerically integrating the
magnetic field vector from seed points distributed on a ring.
"""

import logging
import math
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.physics import MIN_CURVE_POINTS

logger = logging.getLogger(__name__)

_DEFAULT_FIELD_TYPE = "dipole"
_DEFAULT_NUM_LINES = 24
_DEFAULT_LINE_POINTS = 500
_DEFAULT_SPREAD = 0.3

_VALID_FIELD_TYPES = ("dipole", "quadrupole")
_MIN_NUM_LINES = 1
_MAX_NUM_LINES = 200
_MIN_LINE_POINTS = MIN_CURVE_POINTS
_RK4_STEP_SIZE = 0.02
_MAX_RADIUS = 20.0


def _validate_params(
    field_type: str,
    num_lines: int,
    line_points: int,
    spread: float,
) -> None:
    """Validate magnetic field parameters."""
    if field_type not in _VALID_FIELD_TYPES:
        raise ValueError(
            f"field_type must be one of {_VALID_FIELD_TYPES}, got '{field_type}'"
        )
    if num_lines < _MIN_NUM_LINES:
        raise ValueError(
            f"num_lines must be >= {_MIN_NUM_LINES}, got {num_lines}"
        )
    if num_lines > _MAX_NUM_LINES:
        raise ValueError(
            f"num_lines must be <= {_MAX_NUM_LINES}, got {num_lines}"
        )
    if line_points < _MIN_LINE_POINTS:
        raise ValueError(
            f"line_points must be >= {_MIN_LINE_POINTS}, got {line_points}"
        )
    if spread <= 0:
        raise ValueError(f"spread must be positive, got {spread}")


def _dipole_field(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Compute magnetic dipole field at a position."""
    r_sq = x * x + y * y + z * z
    if r_sq < 1e-20:
        return (0.0, 0.0, 0.0)
    r = math.sqrt(r_sq)
    inv_r = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r
    # Dipole moment along z-axis: m = (0, 0, 1)
    # B = (3(m·r_hat)r_hat - m) / r^3
    m_dot_rhat = z * inv_r  # only z-component of m is nonzero
    coeff = 3.0 * m_dot_rhat * inv_r * inv_r3
    return (coeff * x, coeff * y, coeff * z - inv_r3)


def _quadrupole_field(
    x: float, y: float, z: float,
) -> tuple[float, float, float]:
    """Compute magnetic quadrupole field as two offset dipoles."""
    # Two dipoles with opposite moments, offset along z by ±0.5
    ux, uy, uz = _dipole_field(x, y, z - 0.5)
    lx, ly, lz = _dipole_field(x, y, z + 0.5)
    return (ux - lx, uy - ly, uz - lz)


def _field_direction(
    x: float, y: float, z: float, field_type: str,
) -> tuple[float, float, float]:
    """Return normalized field direction at a position."""
    if field_type == "dipole":
        fx, fy, fz = _dipole_field(x, y, z)
    else:
        fx, fy, fz = _quadrupole_field(x, y, z)
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    inv_mag = 1.0 / mag
    return (fx * inv_mag, fy * inv_mag, fz * inv_mag)


def _rk4_step(
    x: float, y: float, z: float, field_type: str, step: float,
) -> tuple[float, float, float]:
    """Perform one RK4 integration step along the field direction."""
    k1x, k1y, k1z = _field_direction(x, y, z, field_type)
    hs = 0.5 * step
    k2x, k2y, k2z = _field_direction(
        x + hs * k1x, y + hs * k1y, z + hs * k1z, field_type,
    )
    k3x, k3y, k3z = _field_direction(
        x + hs * k2x, y + hs * k2y, z + hs * k2z, field_type,
    )
    k4x, k4y, k4z = _field_direction(
        x + step * k3x, y + step * k3y, z + step * k3z, field_type,
    )
    s6 = step / 6.0
    return (
        x + s6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + s6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + s6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


def _integrate_half_line(
    x: float, y: float, z: float,
    field_type: str,
    num_steps: int,
    step: float,
) -> np.ndarray:
    """Integrate field line in one direction from a starting point."""
    points = np.empty((num_steps, 3), dtype=np.float64)
    max_r_sq = _MAX_RADIUS * _MAX_RADIUS
    for i in range(num_steps):
        x, y, z = _rk4_step(x, y, z, field_type, step)
        if x * x + y * y + z * z > max_r_sq:
            return points[:i]
        points[i] = (x, y, z)
    return points


def _integrate_field_line(
    seed_point: np.ndarray,
    field_type: str,
    num_steps: int,
) -> np.ndarray:
    """Integrate a field line from a seed point in both directions."""
    step = _RK4_STEP_SIZE
    half_steps = num_steps // 2
    sx, sy, sz = float(seed_point[0]), float(seed_point[1]), float(seed_point[2])

    forward = _integrate_half_line(sx, sy, sz, field_type, half_steps, step)
    backward = _integrate_half_line(sx, sy, sz, field_type, half_steps, -step)

    # Combine: reversed backward + seed + forward
    parts = []
    if len(backward) > 0:
        parts.append(backward[::-1])
    parts.append(seed_point.reshape(1, 3))
    if len(forward) > 0:
        parts.append(forward)

    return np.concatenate(parts, axis=0)


def _generate_seed_points(
    num_lines: int, spread: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate seed points on a ring around the source."""
    offset = rng.uniform(0.0, 2.0 * np.pi)
    angles = np.linspace(0.0, 2.0 * np.pi, num_lines, endpoint=False) + offset
    x = spread * np.cos(angles)
    y = spread * np.sin(angles)
    z = np.zeros(num_lines, dtype=np.float64)
    return np.column_stack([x, y, z])


@register
class MagneticFieldGenerator(GeneratorBase):
    """Magnetic field line generator for dipole and quadrupole configurations."""

    name = "magnetic_field"
    category = "physics"
    aliases = ("mag_field",)
    description = "3D magnetic field lines for dipole and quadrupole configurations"
    resolution_params = {
        "line_points": "Integration steps per field line",
    }
    _resolution_defaults = {"line_points": _DEFAULT_LINE_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "field_type": _DEFAULT_FIELD_TYPE,
            "num_lines": _DEFAULT_NUM_LINES,
            "spread": _DEFAULT_SPREAD,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate magnetic field lines as curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "line_points" in merged:
            logger.warning(
                "line_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("line_points")

        field_type = str(merged["field_type"])
        num_lines = int(merged["num_lines"])
        spread = float(merged["spread"])
        line_points = int(
            resolution_kwargs.get("line_points", _DEFAULT_LINE_POINTS)
        )

        _validate_params(field_type, num_lines, line_points, spread)
        merged["line_points"] = line_points

        rng = default_rng(seed)
        seed_points = _generate_seed_points(num_lines, spread, rng)

        curves = []
        all_points_list: list[np.ndarray] = []
        for sp in seed_points:
            points = _integrate_field_line(sp, field_type, line_points)
            if len(points) >= 2:
                curves.append(Curve(points=points, closed=False))
                all_points_list.append(points)

        if not curves:
            raise ValueError(
                f"All {num_lines} field lines were too short to produce "
                f"valid curves (field_type={field_type!r}, spread={spread})"
            )

        all_points = np.concatenate(all_points_list, axis=0)
        bbox = BoundingBox.from_points(all_points)

        logger.info(
            "Generated magnetic_field: type=%s, lines=%d, steps=%d",
            field_type, len(curves), line_points,
        )

        return MathObject(
            curves=curves,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for magnetic field lines."""
        return RepresentationConfig(type=RepresentationType.TUBE)
