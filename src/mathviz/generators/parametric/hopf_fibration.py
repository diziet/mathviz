"""Hopf fibration generator.

Maps points on S² to great circles on S³, then projects to R³ via
stereographic projection. Fibers are organized into nested tori of
linked rings — one of the most visually striking objects in topology.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_NUM_FIBERS = 32
_DEFAULT_NUM_CIRCLES = 5
_DEFAULT_FIBER_POINTS = 256
_DEFAULT_PROJECTION_POINT = (0.0, 0.0, 0.0, 2.0)

_MIN_NUM_FIBERS = 1
_MIN_NUM_CIRCLES = 1
_MIN_FIBER_POINTS = 4


_PROJECTION_CLAMP = 1e6


def _validate_params(
    num_fibers: int,
    num_circles: int,
    fiber_points: int,
    projection_point: tuple[float, ...],
) -> None:
    """Validate Hopf fibration parameters."""
    if num_fibers < _MIN_NUM_FIBERS:
        raise ValueError(
            f"num_fibers must be >= {_MIN_NUM_FIBERS}, got {num_fibers}"
        )
    if num_circles < _MIN_NUM_CIRCLES:
        raise ValueError(
            f"num_circles must be >= {_MIN_NUM_CIRCLES}, got {num_circles}"
        )
    if fiber_points < _MIN_FIBER_POINTS:
        raise ValueError(
            f"fiber_points must be >= {_MIN_FIBER_POINTS}, got {fiber_points}"
        )
    if len(projection_point) != 4:
        raise ValueError(
            f"projection_point must have exactly 4 elements, "
            f"got {len(projection_point)}"
        )


def _base_circle_points(
    num_circles: int, num_fibers: int,
) -> list[tuple[float, float, float]]:
    """Generate base points on S² organized along latitude circles.

    Each circle is at a different polar angle (latitude), with fibers
    distributed uniformly around the azimuth.
    """
    points = []
    for circle_idx in range(num_circles):
        theta = np.pi * (circle_idx + 1) / (num_circles + 1)
        for fiber_idx in range(num_fibers):
            phi = 2.0 * np.pi * fiber_idx / num_fibers
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append((float(x), float(y), float(z)))
    return points


def _hopf_fiber(
    base_x: float,
    base_y: float,
    base_z: float,
    fiber_points: int,
    projection_point: tuple[float, float, float, float],
) -> np.ndarray:
    """Compute one Hopf fiber as a closed curve in R³.

    Given a point (base_x, base_y, base_z) on S², compute the
    corresponding great circle on S³ and project via stereographic
    projection to R³.
    """
    # Convert S² point to spherical half-angles for the Hopf map
    theta = np.arccos(np.clip(base_z, -1.0, 1.0))
    phi = np.arctan2(base_y, base_x)

    half_theta = theta / 2.0
    cos_ht = np.cos(half_theta)
    sin_ht = np.sin(half_theta)

    t = np.linspace(0.0, 2.0 * np.pi, fiber_points, endpoint=False)

    # Points on S³: (w, x, y, z) parameterized by t
    # The Hopf map fiber over (base_x, base_y, base_z) is:
    w = cos_ht * np.cos(t)
    x = cos_ht * np.sin(t)
    y = sin_ht * np.cos(t + phi)
    z = sin_ht * np.sin(t + phi)

    # Stereographic projection from S³ to R³
    pw, px, py, pz = projection_point
    denom = pz - z
    # Avoid division by zero near the projection pole (preserve sign)
    safe_denom = np.where(
        np.abs(denom) < 1e-12, np.copysign(1e-12, denom), denom,
    )

    proj_x = (w - pw) / safe_denom
    proj_y = (x - px) / safe_denom
    proj_z = (y - py) / safe_denom

    result = np.column_stack([proj_x, proj_y, proj_z]).astype(np.float64)
    # Clamp extreme points near the projection pole
    np.clip(result, -_PROJECTION_CLAMP, _PROJECTION_CLAMP, out=result)
    return result


@register
class HopfFibrationGenerator(GeneratorBase):
    """Hopf fibration — circles in S³ projected to R³ as nested tori."""

    name = "hopf_fibration"
    category = "parametric"
    aliases = ("hopf",)
    description = (
        "Hopf fibration: S² base circles mapped to linked fiber "
        "tori in R³ via stereographic projection"
    )
    resolution_params = {
        "fiber_points": "Number of sample points per fiber curve",
    }
    _resolution_defaults = {"fiber_points": _DEFAULT_FIBER_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "num_fibers": _DEFAULT_NUM_FIBERS,
            "num_circles": _DEFAULT_NUM_CIRCLES,
            "projection_point": list(_DEFAULT_PROJECTION_POINT),
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate Hopf fibration as closed curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "fiber_points" in merged:
            logger.warning(
                "fiber_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("fiber_points")

        num_fibers = int(merged["num_fibers"])
        num_circles = int(merged["num_circles"])
        projection_point = tuple(float(v) for v in merged["projection_point"])
        fiber_points = int(
            resolution_kwargs.get(
                "fiber_points", self._resolution_defaults["fiber_points"],
            )
        )

        _validate_params(num_fibers, num_circles, fiber_points, projection_point)
        merged["fiber_points"] = fiber_points

        base_points = _base_circle_points(num_circles, num_fibers)

        curves: list[Curve] = []
        for bx, by, bz in base_points:
            pts = _hopf_fiber(bx, by, bz, fiber_points, projection_point)
            curves.append(Curve(points=pts, closed=True))

        all_points = np.concatenate([c.points for c in curves], axis=0)
        bbox = BoundingBox.from_points(all_points)

        logger.info(
            "Generated hopf_fibration: num_fibers=%d, num_circles=%d, "
            "total_curves=%d, fiber_points=%d",
            num_fibers, num_circles, len(curves), fiber_points,
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
        """Return the recommended representation for Hopf fibration."""
        return RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.02,
        )
