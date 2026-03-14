"""Chen attractor dynamical system generator.

Integrates the Chen system of ODEs to produce a 3D trajectory curve.
The Chen system is defined by:
    dx/dt = a*(y - x)
    dy/dt = (c - a)*x - x*z + c*y
    dz/dt = x*y - b*z

With default parameters a=35, b=3, c=28, the attractor produces a
double-scroll chaotic trajectory.
"""

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@register
class ChenGenerator(AttractorGeneratorBase):
    """Chen attractor dynamical system generator."""

    name = "chen"
    aliases = ("chen_attractor",)
    description = "Chen double-scroll strange attractor trajectory"

    _t_span_end = 50.0
    _default_initial_condition = (-10.0, 0.0, 37.0)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Chen generator."""
        return {
            "a": 35.0,
            "b": 3.0,
            "c": 28.0,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Chen ODE parameters."""
        _validate_positive("a", float(params["a"]))
        _validate_positive("b", float(params["b"]))
        _validate_positive("c", float(params["c"]))

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Chen system."""
        x, y, z = state
        a, b, c = float(params["a"]), float(params["b"]), float(params["c"])
        return [
            a * (y - x),
            (c - a) * x - x * z + c * y,
            x * y - b * z,
        ]
