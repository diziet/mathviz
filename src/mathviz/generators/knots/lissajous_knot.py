"""Lissajous knot generator.

A Lissajous knot is a knot defined by parametric equations using three
frequency/phase pairs. When the frequencies are pairwise coprime and
phases are chosen to avoid self-intersections, the result is a closed knot.
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
_MIN_CURVE_POINTS = 16

_DEFAULT_NX = 2
_DEFAULT_NY = 3
_DEFAULT_NZ = 5
_DEFAULT_PHASE_X = 0.0
_DEFAULT_PHASE_Y = 0.7
_DEFAULT_PHASE_Z = 0.2
_DEFAULT_SCALE = 1.0


def _compute_lissajous_knot_points(
    nx: int,
    ny: int,
    nz: int,
    phase_x: float,
    phase_y: float,
    phase_z: float,
    scale: float,
    num_points: int,
) -> np.ndarray:
    """Compute points on a Lissajous knot curve."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    x = np.cos(nx * t + phase_x)
    y = np.cos(ny * t + phase_y)
    z = np.cos(nz * t + phase_z)

    return np.column_stack([x, y, z]).astype(np.float64) * scale



def _validate_params(
    nx: int, ny: int, nz: int, scale: float, curve_points: int
) -> None:
    """Validate Lissajous knot parameters."""
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise ValueError(f"ny must be >= 1, got {ny}")
    if nz < 1:
        raise ValueError(f"nz must be >= 1, got {nz}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class LissajousKnotGenerator(GeneratorBase):
    """Lissajous knot from three frequency/phase pairs."""

    name = "lissajous_knot"
    category = "knots"
    aliases = ()
    description = "Lissajous knot with configurable frequencies and phases"
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
    }
    _resolution_defaults = {"curve_points": _DEFAULT_CURVE_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "nx": _DEFAULT_NX,
            "ny": _DEFAULT_NY,
            "nz": _DEFAULT_NZ,
            "phase_x": _DEFAULT_PHASE_X,
            "phase_y": _DEFAULT_PHASE_Y,
            "phase_z": _DEFAULT_PHASE_Z,
            "scale": _DEFAULT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Lissajous knot curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        nx = int(merged["nx"])
        ny = int(merged["ny"])
        nz = int(merged["nz"])
        phase_x = float(merged["phase_x"])
        phase_y = float(merged["phase_y"])
        phase_z = float(merged["phase_z"])
        scale = float(merged["scale"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(nx, ny, nz, scale, curve_points)
        merged["curve_points"] = curve_points

        points = _compute_lissajous_knot_points(
            nx, ny, nz, phase_x, phase_y, phase_z, scale, curve_points,
        )

        # Lissajous curves with integer frequencies are always closed
        # over 2*pi period
        curve = Curve(points=points, closed=True)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated Lissajous knot: nx=%d, ny=%d, nz=%d, points=%d",
            nx, ny, nz, curve_points,
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
        """Return the recommended representation for Lissajous knots."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
