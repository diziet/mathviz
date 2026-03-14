"""Halvorsen attractor dynamical system generator.

Integrates the Halvorsen system of ODEs to produce a 3D trajectory curve.
The Halvorsen system is defined by:
    dx/dt = -a*x - 4*y - 4*z - y^2
    dy/dt = -a*y - 4*z - 4*x - z^2
    dz/dt = -a*z - 4*x - 4*y - x^2

With default parameter a=1.89, the attractor produces a three-winged
chaotic trajectory with approximate three-fold symmetry.
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

_DEFAULT_A = 1.89
_DEFAULT_TRANSIENT_STEPS = 1000
_DEFAULT_INTEGRATION_STEPS = 100_000
_DEFAULT_INITIAL_CONDITION = (-1.48, -1.51, 2.04)
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-3
_T_SPAN_END = 100.0


def _halvorsen_rhs(
    _t: float,
    state: np.ndarray,
    a: float,
) -> list[float]:
    """Compute the right-hand side of the Halvorsen system."""
    x, y, z = state
    dx = -a * x - 4.0 * y - 4.0 * z - y * y
    dy = -a * y - 4.0 * z - 4.0 * x - z * z
    dz = -a * z - 4.0 * x - 4.0 * y - x * x
    return [dx, dy, dz]


def _integrate_halvorsen(
    a: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the Halvorsen system and return trajectory after transient."""
    t_eval = np.linspace(0.0, _T_SPAN_END, integration_steps)

    result = solve_ivp(
        fun=lambda t, state: _halvorsen_rhs(t, state, a),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Halvorsen integration failed: {result.message}")

    trajectory = result.y.T[transient_steps:]
    return trajectory.astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    a: float,
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate Halvorsen parameters, raising ValueError for invalid inputs."""
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
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
class HalvorsenGenerator(GeneratorBase):
    """Halvorsen attractor dynamical system generator."""

    name = "halvorsen"
    category = "attractors"
    aliases = ("halvorsen_attractor",)
    description = "Halvorsen three-winged strange attractor trajectory"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Halvorsen generator."""
        return {
            "a": _DEFAULT_A,
            "transient_steps": _DEFAULT_TRANSIENT_STEPS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Halvorsen attractor trajectory curve."""
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
        transient_steps = int(merged["transient_steps"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(a, integration_steps, transient_steps)
        merged["integration_steps"] = integration_steps

        rng = default_rng(seed)
        perturbation = rng.normal(scale=_PERTURBATION_SCALE, size=3)
        initial_condition = np.array(_DEFAULT_INITIAL_CONDITION) + perturbation

        trajectory = _integrate_halvorsen(
            a, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated Halvorsen attractor: a=%.2f, "
            "points=%d (discarded %d transient)",
            a, len(trajectory), transient_steps,
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
        """Return the recommended representation for the Halvorsen attractor."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
