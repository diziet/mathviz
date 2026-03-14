"""Rössler attractor dynamical system generator.

Integrates the Rössler system of ODEs to produce a 3D trajectory curve.
The Rössler system is defined by:
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

With default parameters a=0.2, b=0.2, c=5.7, the attractor exhibits a
characteristic folded-band shape that is wider than tall.
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

_DEFAULT_A = 0.2
_DEFAULT_B = 0.2
_DEFAULT_C = 5.7
_DEFAULT_TRANSIENT_STEPS = 1000
_DEFAULT_INTEGRATION_STEPS = 100_000
_DEFAULT_INITIAL_CONDITION = (1.0, 1.0, 0.0)
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-3
_T_SPAN_END = 300.0


def _rossler_rhs(
    _t: float,
    state: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> list[float]:
    """Compute the right-hand side of the Rössler system."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]


def _integrate_rossler(
    a: float,
    b: float,
    c: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the Rössler system and return trajectory points after transient."""
    t_eval = np.linspace(0.0, _T_SPAN_END, integration_steps)

    result = solve_ivp(
        fun=lambda t, state: _rossler_rhs(t, state, a, b, c),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Rössler integration failed: {result.message}")

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
    """Validate Rössler parameters, raising ValueError for invalid inputs."""
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
class RosslerGenerator(GeneratorBase):
    """Rössler attractor dynamical system generator."""

    name = "rossler"
    category = "attractors"
    aliases = ("rossler_attractor",)
    description = "Rössler strange attractor trajectory"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Rössler generator."""
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
        """Generate a Rössler attractor trajectory curve."""
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

        trajectory = _integrate_rossler(
            a, b, c, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated Rössler attractor: a=%.2f, b=%.2f, c=%.2f, "
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
        """Return the recommended representation for the Rössler attractor."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
