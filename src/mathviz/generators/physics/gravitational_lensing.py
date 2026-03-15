"""Gravitational lensing grid generator.

Generates a warped coordinate grid showing spacetime curvature around a
point mass. Grid lines are deflected using the Schwarzschild deflection
formula and extruded to 3D using the deflection magnitude as z-displacement.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_MASS = 1.0
_DEFAULT_GRID_LINES = 20
_DEFAULT_GRID_EXTENT = 5.0
_DEFAULT_GRID_POINTS = 200

_MIN_GRID_LINES = 2
_MAX_GRID_LINES = 100
_MIN_GRID_POINTS = 10
_MAX_GRID_POINTS = 10_000
_MIN_GRID_EXTENT = 0.1
_SOFTENING = 0.3


def _validate_params(
    mass: float,
    grid_lines: int,
    grid_extent: float,
    grid_points: int,
) -> None:
    """Validate gravitational lensing parameters."""
    if mass < 0:
        raise ValueError(f"mass must be >= 0, got {mass}")
    if grid_lines < _MIN_GRID_LINES:
        raise ValueError(
            f"grid_lines must be >= {_MIN_GRID_LINES}, got {grid_lines}"
        )
    if grid_lines > _MAX_GRID_LINES:
        raise ValueError(
            f"grid_lines must be <= {_MAX_GRID_LINES}, got {grid_lines}"
        )
    if grid_extent < _MIN_GRID_EXTENT:
        raise ValueError(
            f"grid_extent must be >= {_MIN_GRID_EXTENT}, got {grid_extent}"
        )
    if grid_points < _MIN_GRID_POINTS:
        raise ValueError(
            f"grid_points must be >= {_MIN_GRID_POINTS}, got {grid_points}"
        )
    if grid_points > _MAX_GRID_POINTS:
        raise ValueError(
            f"grid_points must be <= {_MAX_GRID_POINTS}, got {grid_points}"
        )


def _generate_grid_line(
    positions: np.ndarray,
    fixed_val: float,
    mass: float,
    is_horizontal: bool,
) -> np.ndarray:
    """Generate a single deflected grid line (vectorized).

    For horizontal lines, positions vary along x with fixed y.
    For vertical lines, positions vary along y with fixed x.
    Deflection uses softened radius consistently for both magnitude
    and direction to prevent overshoot near the origin.
    """
    if is_horizontal:
        x = positions
        y = np.full_like(positions, fixed_val)
    else:
        x = np.full_like(positions, fixed_val)
        y = positions

    r_sq = x * x + y * y
    # Softened radius used for both magnitude and direction
    r = np.sqrt(r_sq + _SOFTENING * _SOFTENING)
    deflection = mass / r
    inv_r = 1.0 / r
    ux = x * inv_r
    uy = y * inv_r

    dx = x - deflection * ux
    dy = y - deflection * uy
    dz = deflection
    return np.column_stack((dx, dy, dz))


def _build_grid_curves(
    mass: float,
    grid_lines: int,
    grid_extent: float,
    grid_points: int,
) -> list[Curve]:
    """Build all deflected grid line curves."""
    positions = np.linspace(-grid_extent, grid_extent, grid_points)
    line_positions = np.linspace(-grid_extent, grid_extent, grid_lines)

    curves: list[Curve] = []
    # Horizontal lines (constant y, varying x)
    for y_val in line_positions:
        pts = _generate_grid_line(positions, y_val, mass, is_horizontal=True)
        curves.append(Curve(points=pts, closed=False))

    # Vertical lines (constant x, varying y)
    for x_val in line_positions:
        pts = _generate_grid_line(positions, x_val, mass, is_horizontal=False)
        curves.append(Curve(points=pts, closed=False))

    return curves


@register
class GravitationalLensingGenerator(GeneratorBase):
    """Gravitational lensing grid generator showing spacetime curvature."""

    name = "gravitational_lensing"
    category = "physics"
    aliases = ("grav_lens", "spacetime_grid")
    description = (
        "Warped coordinate grid showing spacetime curvature "
        "around a point mass via Schwarzschild deflection"
    )
    resolution_params = {
        "grid_points": "Sample points per grid line",
    }
    _resolution_defaults = {"grid_points": _DEFAULT_GRID_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "mass": _DEFAULT_MASS,
            "grid_lines": _DEFAULT_GRID_LINES,
            "grid_extent": _DEFAULT_GRID_EXTENT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a gravitational lensing grid as curves.

        Note: seed is accepted per GeneratorBase contract but this
        generator is fully deterministic — seed does not affect output.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "grid_points" in merged:
            logger.warning(
                "grid_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("grid_points")

        mass = float(merged["mass"])
        grid_lines = int(merged["grid_lines"])
        grid_extent = float(merged["grid_extent"])
        grid_points = int(
            resolution_kwargs.get("grid_points", _DEFAULT_GRID_POINTS)
        )

        _validate_params(mass, grid_lines, grid_extent, grid_points)
        merged["grid_points"] = grid_points

        curves = _build_grid_curves(mass, grid_lines, grid_extent, grid_points)

        all_points = np.concatenate(
            [c.points for c in curves], axis=0,
        )
        bbox = BoundingBox.from_points(all_points)

        logger.info(
            "Generated gravitational_lensing: mass=%.2f, "
            "grid_lines=%d, grid_points=%d",
            mass, grid_lines, grid_points,
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
        """Return the recommended representation for lensing grid."""
        return RepresentationConfig(type=RepresentationType.TUBE)
