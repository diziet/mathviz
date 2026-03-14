"""Chen attractor dynamical system generator.

Integrates the Chen system of ODEs to produce a 3D trajectory curve.
The Chen system is defined by:
    dx/dt = a*(y - x)
    dy/dt = (c - a)*x - x*z + c*y
    dz/dt = x*y - b*z

With default parameters a=35, b=3, c=28, the attractor produces a
double-scroll chaotic trajectory.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_A = 35.0
_DEFAULT_B = 3.0
_DEFAULT_C = 28.0
_DEFAULT_TRANSIENT_STEPS = 1000
_DEFAULT_INTEGRATION_STEPS = 100_000
_DEFAULT_INITIAL_CONDITION = (-10.0, 0.0, 37.0)
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-3
_T_SPAN_END = 50.0


def _chen_rhs(
    _t: float,
    state: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> list[float]:
    """Compute the right-hand side of the Chen system."""
    x, y, z = state
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return [dx, dy, dz]


def _integrate_chen(
    a: float,
    b: float,
    c: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the Chen system and return trajectory points after transient."""
    t_eval = np.linspace(0.0, _T_SPAN_END, integration_steps)

    result = solve_ivp(
        fun=lambda t, state: _chen_rhs(t, state, a, b, c),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Chen integration failed: {result.message}")

    trajectory = result.y.T[transient_steps:]
    return trajectory.astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    a: float,
    b: float,
    c: float,
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate Chen parameters, raising ValueError for invalid inputs."""
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if integration_steps < _MIN_INTEGRATION_STEPS:
        raise ValueError(
            f"integration_steps must be >= {_MIN_INTEGRATION_STEPS}, "
            f"got {integration_steps}"
        )
    if transient_steps < 0:
        raise ValueError(
            f"transient_steps must be >= 0, got {transient_steps}"
        )
    trajectory_points = integration_steps - transient_steps
    if trajectory_points < _MIN_TRAJECTORY_POINTS:
        raise ValueError(
            f"integration_steps - transient_steps must be >= "
            f"{_MIN_TRAJECTORY_POINTS}, got {trajectory_points}"
        )


@register
class ChenGenerator(GeneratorBase):
    """Chen attractor dynamical system generator."""

    name = "chen"
    category = "attractors"
    aliases = ("chen_attractor",)
    description = "Chen double-scroll strange attractor trajectory"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Chen generator."""
        return {
            "a": _DEFAULT_A,
            "b": _DEFAULT_B,
            "c": _DEFAULT_C,
            "transient_steps": _DEFAULT_TRANSIENT_STEPS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Chen attractor trajectory curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        a = float(merged["a"])
        b = float(merged["b"])
        c = float(merged["c"])
        transient_steps = int(merged["transient_steps"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(a, b, c, integration_steps, transient_steps)
        merged["integration_steps"] = integration_steps

        rng = default_rng(seed)
        perturbation = rng.normal(scale=_PERTURBATION_SCALE, size=3)
        initial_condition = np.array(_DEFAULT_INITIAL_CONDITION) + perturbation

        trajectory = _integrate_chen(
            a, b, c, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated Chen attractor: a=%.1f, b=%.1f, c=%.1f, "
            "points=%d (discarded %d transient)",
            a, b, c, len(trajectory), transient_steps,
        )

        return MathObject(
            curves=[curve],
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the Chen attractor."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
