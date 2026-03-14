"""Thomas attractor dynamical system generator.

Integrates the Thomas cyclically symmetric system of ODEs to produce a
3D trajectory curve. The Thomas system is defined by:
    dx/dt = sin(y) - b*x
    dy/dt = sin(z) - b*y
    dz/dt = sin(x) - b*z

With default parameter b≈0.208186, the attractor produces a bounded chaotic
trajectory with three-fold rotational symmetry. The parameter b controls
dissipation — values near 0.208186 yield the classic strange attractor.
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

_DEFAULT_B = 0.208186
_DEFAULT_TRANSIENT_STEPS = 1000
_DEFAULT_INTEGRATION_STEPS = 100_000
_DEFAULT_INITIAL_CONDITION = (1.1, 1.1, -0.01)
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-3
_T_SPAN_END = 500.0


def _thomas_rhs(
    _t: float,
    state: np.ndarray,
    b: float,
) -> list[float]:
    """Compute the right-hand side of the Thomas system."""
    x, y, z = state
    dx = np.sin(y) - b * x
    dy = np.sin(z) - b * y
    dz = np.sin(x) - b * z
    return [dx, dy, dz]


def _integrate_thomas(
    b: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the Thomas system and return trajectory points after transient."""
    t_eval = np.linspace(0.0, _T_SPAN_END, integration_steps)

    result = solve_ivp(
        fun=lambda t, state: _thomas_rhs(t, state, b),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Thomas integration failed: {result.message}")

    trajectory = result.y.T[transient_steps:]
    return trajectory.astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    b: float,
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate Thomas parameters, raising ValueError for invalid inputs."""
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
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
class ThomasGenerator(GeneratorBase):
    """Thomas cyclically symmetric attractor generator."""

    name = "thomas"
    category = "attractors"
    aliases = ("thomas_attractor",)
    description = "Thomas cyclically symmetric strange attractor trajectory"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Thomas generator."""
        return {
            "b": _DEFAULT_B,
            "transient_steps": _DEFAULT_TRANSIENT_STEPS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Thomas attractor trajectory curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        b = float(merged["b"])
        transient_steps = int(merged["transient_steps"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(b, integration_steps, transient_steps)
        merged["integration_steps"] = integration_steps

        rng = default_rng(seed)
        perturbation = rng.normal(scale=_PERTURBATION_SCALE, size=3)
        initial_condition = np.array(_DEFAULT_INITIAL_CONDITION) + perturbation

        trajectory = _integrate_thomas(
            b, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated Thomas attractor: b=%.6f, "
            "points=%d (discarded %d transient)",
            b, len(trajectory), transient_steps,
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
        """Return the recommended representation for the Thomas attractor."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
