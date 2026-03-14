"""Double pendulum chaotic dynamical system generator.

Integrates the double pendulum equations of motion in 4D phase space
(θ₁, θ₂, ω₁, ω₂) and projects to 3D for visualization.

The equations of motion for a double pendulum with equal masses (m) and
equal lengths (l), under gravity (g):

    dθ₁/dt = ω₁
    dθ₂/dt = ω₂
    dω₁/dt = (-g(2m)sin(θ₁) - m*g*sin(θ₁-2θ₂) - 2sin(θ₁-θ₂)*m*(ω₂²l + ω₁²l*cos(θ₁-θ₂)))
              / (l*(2m - m*cos(2θ₁-2θ₂)))
    dω₂/dt = (2sin(θ₁-θ₂)*(ω₁²l(2m) + g(2m)cos(θ₁) + ω₂²l*m*cos(θ₁-θ₂)))
              / (l*(2m - m*cos(2θ₁-2θ₂)))

**3D projection choice:** The 4D phase space (θ₁, θ₂, ω₁, ω₂) is projected
to 3D as (θ₁, θ₂, ω₁). This preserves both angle coordinates (the primary
observable quantities) and one angular velocity, giving a visually rich
attractor structure. The omitted ω₂ is correlated with ω₁ via conservation
constraints, so minimal information is lost.
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

_DEFAULT_MASS = 1.0
_DEFAULT_LENGTH = 1.0
_DEFAULT_GRAVITY = 9.81
_DEFAULT_TRANSIENT_STEPS = 500
_DEFAULT_INTEGRATION_STEPS = 100_000
# Initial angles in radians; high energy for chaotic motion
_DEFAULT_THETA1 = 2.5
_DEFAULT_THETA2 = 2.0
_DEFAULT_OMEGA1 = 0.0
_DEFAULT_OMEGA2 = 0.0
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-2
_T_SPAN_END = 100.0


def _double_pendulum_rhs(
    _t: float,
    state: np.ndarray,
    mass: float,
    length: float,
    gravity: float,
) -> list[float]:
    """Compute the right-hand side of the double pendulum system."""
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denom = length * (2.0 * mass - mass * np.cos(2.0 * delta))

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (
        -gravity * (2.0 * mass) * np.sin(theta1)
        - mass * gravity * np.sin(theta1 - 2.0 * theta2)
        - 2.0 * sin_delta * mass
        * (omega2 * omega2 * length + omega1 * omega1 * length * cos_delta)
    ) / denom

    domega2 = (
        2.0 * sin_delta * (
            omega1 * omega1 * length * (2.0 * mass)
            + gravity * (2.0 * mass) * np.cos(theta1)
            + omega2 * omega2 * length * mass * cos_delta
        )
    ) / denom

    return [dtheta1, dtheta2, domega1, domega2]


def _integrate_double_pendulum(
    mass: float,
    length: float,
    gravity: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the double pendulum and return 3D projected trajectory."""
    t_eval = np.linspace(0.0, _T_SPAN_END, integration_steps)

    result = solve_ivp(
        fun=lambda t, s: _double_pendulum_rhs(t, s, mass, length, gravity),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(
            f"Double pendulum integration failed: {result.message}"
        )

    # result.y is (4, N) — project to 3D: (θ₁, θ₂, ω₁)
    full_trajectory = result.y.T[transient_steps:]
    projected = full_trajectory[:, :3].astype(np.float64)
    return projected


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    mass: float,
    length: float,
    gravity: float,
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate double pendulum parameters."""
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if gravity <= 0:
        raise ValueError(f"gravity must be positive, got {gravity}")
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
class DoublePendulumGenerator(GeneratorBase):
    """Double pendulum chaotic dynamical system generator."""

    name = "double_pendulum"
    category = "attractors"
    aliases = ("double_pendulum_attractor",)
    description = "Double pendulum chaotic trajectory (4D phase space → 3D)"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the double pendulum generator."""
        return {
            "mass": _DEFAULT_MASS,
            "length": _DEFAULT_LENGTH,
            "gravity": _DEFAULT_GRAVITY,
            "theta1": _DEFAULT_THETA1,
            "theta2": _DEFAULT_THETA2,
            "omega1": _DEFAULT_OMEGA1,
            "omega2": _DEFAULT_OMEGA2,
            "transient_steps": _DEFAULT_TRANSIENT_STEPS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a double pendulum trajectory projected to 3D."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        mass = float(merged["mass"])
        length = float(merged["length"])
        gravity = float(merged["gravity"])
        theta1 = float(merged["theta1"])
        theta2 = float(merged["theta2"])
        omega1 = float(merged["omega1"])
        omega2 = float(merged["omega2"])
        transient_steps = int(merged["transient_steps"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(
            mass, length, gravity, integration_steps, transient_steps,
        )
        merged["integration_steps"] = integration_steps

        # Perturb initial angles with seed for variation
        rng = default_rng(seed)
        perturbation = rng.normal(scale=_PERTURBATION_SCALE, size=4)
        initial_condition = np.array(
            [theta1, theta2, omega1, omega2]
        ) + perturbation

        trajectory = _integrate_double_pendulum(
            mass, length, gravity, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated double pendulum: m=%.2f, l=%.2f, g=%.2f, "
            "points=%d (discarded %d transient), "
            "projection=(θ₁, θ₂, ω₁)",
            mass, length, gravity,
            len(trajectory), transient_steps,
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
        """Return the recommended representation for the double pendulum."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
