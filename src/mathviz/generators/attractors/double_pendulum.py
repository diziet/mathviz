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

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.attractors._base import AttractorGeneratorBase


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@register
class DoublePendulumGenerator(AttractorGeneratorBase):
    """Double pendulum chaotic dynamical system generator."""

    name = "double_pendulum"
    aliases = ("double_pendulum_attractor",)
    description = "Double pendulum chaotic trajectory (4D phase space → 3D)"

    _t_span_end = 100.0
    _default_initial_condition = (2.5, 2.0, 0.0, 0.0)
    _perturbation_scale = 1e-2
    _output_dims = 3  # Project 4D → 3D: (θ₁, θ₂, ω₁)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the double pendulum generator."""
        return {
            "mass": 1.0,
            "length": 1.0,
            "gravity": 9.81,
            "theta1": 2.5,
            "theta2": 2.0,
            "omega1": 0.0,
            "omega2": 0.0,
            "transient_steps": 500,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate double pendulum ODE parameters."""
        _validate_positive("mass", float(params["mass"]))
        _validate_positive("length", float(params["length"]))
        _validate_positive("gravity", float(params["gravity"]))

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the double pendulum system."""
        theta1, theta2, omega1, omega2 = state
        mass = float(params["mass"])
        length = float(params["length"])
        gravity = float(params["gravity"])

        delta = theta1 - theta2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        denom = length * (2.0 * mass - mass * np.cos(2.0 * delta))

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

        return [omega1, omega2, domega1, domega2]
