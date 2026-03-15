"""Magnetic field line generator.

Generates 3D magnetic field line visualizations for dipole and quadrupole
configurations. Field lines are computed by numerically integrating the
magnetic field vector from seed points distributed on a ring.
"""

import logging
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
_DEFAULT_TUBE_RADIUS = 0.02

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


def _dipole_field(position: np.ndarray) -> np.ndarray:
    """Compute magnetic dipole field at a position."""
    # Dipole moment along z-axis: m = (0, 0, 1)
    r = np.linalg.norm(position)
    if r < 1e-10:
        return np.zeros(3, dtype=np.float64)
    r_hat = position / r
    m = np.array([0.0, 0.0, 1.0])
    m_dot_r = np.dot(m, r_hat)
    field = (3.0 * m_dot_r * r_hat - m) / (r**3)
    return field


def _quadrupole_field(position: np.ndarray) -> np.ndarray:
    """Compute magnetic quadrupole field as two offset dipoles."""
    offset = np.array([0.0, 0.0, 0.5])
    # Two dipoles with opposite moments, offset along z
    field_upper = _dipole_field(position - offset)
    field_lower = -_dipole_field(position + offset)
    return field_upper + field_lower


def _compute_field(
    position: np.ndarray, field_type: str
) -> np.ndarray:
    """Compute magnetic field at a given position."""
    if field_type == "dipole":
        return _dipole_field(position)
    return _quadrupole_field(position)


def _rk4_step(
    position: np.ndarray, field_type: str, step: float
) -> np.ndarray:
    """Perform one RK4 integration step along the field direction."""
    k1 = _field_direction(position, field_type)
    k2 = _field_direction(position + 0.5 * step * k1, field_type)
    k3 = _field_direction(position + 0.5 * step * k2, field_type)
    k4 = _field_direction(position + step * k3, field_type)
    return position + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _field_direction(
    position: np.ndarray, field_type: str
) -> np.ndarray:
    """Return normalized field direction at a position."""
    field = _compute_field(position, field_type)
    magnitude = np.linalg.norm(field)
    if magnitude < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return field / magnitude


def _integrate_field_line(
    seed_point: np.ndarray,
    field_type: str,
    num_steps: int,
) -> np.ndarray:
    """Integrate a field line from a seed point in both directions."""
    step = _RK4_STEP_SIZE
    half_steps = num_steps // 2

    # Forward integration
    forward = np.empty((half_steps, 3), dtype=np.float64)
    pos = seed_point.copy()
    for i in range(half_steps):
        pos = _rk4_step(pos, field_type, step)
        if np.linalg.norm(pos) > _MAX_RADIUS:
            forward = forward[:i]
            break
        forward[i] = pos

    # Backward integration
    backward = np.empty((half_steps, 3), dtype=np.float64)
    pos = seed_point.copy()
    for i in range(half_steps):
        pos = _rk4_step(pos, field_type, -step)
        if np.linalg.norm(pos) > _MAX_RADIUS:
            backward = backward[:i]
            break
        backward[i] = pos

    # Combine: reversed backward + seed + forward
    parts = []
    if len(backward) > 0:
        parts.append(backward[::-1])
    parts.append(seed_point.reshape(1, 3))
    if len(forward) > 0:
        parts.append(forward)

    return np.concatenate(parts, axis=0)


def _generate_seed_points(
    num_lines: int, spread: float, rng: np.random.Generator
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
        all_points_list = []
        for sp in seed_points:
            points = _integrate_field_line(sp, field_type, line_points)
            if len(points) >= 2:
                curves.append(Curve(points=points, closed=False))
                all_points_list.append(points)

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
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
