"""Rössler attractor dynamical system generator.

Integrates the Rössler system of ODEs to produce a 3D trajectory curve.
The Rössler system is defined by:
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

With default parameters a=0.2, b=0.2, c=5.7, the attractor exhibits a
characteristic folded-band shape that is wider than tall.
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
class RosslerGenerator(AttractorGeneratorBase):
    """Rössler attractor dynamical system generator."""

    name = "rossler"
    aliases = ("rossler_attractor",)
    description = "Rössler strange attractor trajectory"

    _t_span_end = 300.0
    _default_initial_condition = (1.0, 1.0, 0.0)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Rössler generator."""
        return {
            "a": 0.2,
            "b": 0.2,
            "c": 5.7,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Rössler ODE parameters."""
        _validate_positive("a", float(params["a"]))
        _validate_positive("b", float(params["b"]))
        _validate_positive("c", float(params["c"]))

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Rössler system."""
        x, y, z = state
        a, b, c = float(params["a"]), float(params["b"]), float(params["c"])
        return [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ]
