"""Aizawa attractor dynamical system generator.

Integrates the Aizawa system of ODEs to produce a 3D trajectory curve.
The Aizawa system is defined by:
    dx/dt = (z - b)*x - d*y
    dy/dt = d*x + (z - b)*y
    dz/dt = c + a*z - z^3/3 - (x^2 + y^2)*(1 + e*z) + f*z*x^3

With default parameters a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1,
the attractor produces a torus-like chaotic trajectory.
"""

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


@register
class AizawaGenerator(AttractorGeneratorBase):
    """Aizawa attractor dynamical system generator."""

    name = "aizawa"
    aliases = ("aizawa_attractor",)
    description = "Aizawa torus-like strange attractor trajectory"

    _t_span_end = 200.0
    _default_initial_condition = (0.1, 0.0, 0.0)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Aizawa generator."""
        return {
            "a": 0.95,
            "b": 0.7,
            "c": 0.6,
            "d": 3.5,
            "e": 0.25,
            "f": 0.1,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Aizawa ODE parameters."""
        a = float(params["a"])
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        d = float(params["d"])
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Aizawa system."""
        x, y, z = state
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        d = float(params["d"])
        e = float(params["e"])
        f = float(params["f"])
        r_sq = x * x + y * y
        return [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - z * z * z / 3.0 - r_sq * (1.0 + e * z) + f * z * x * x * x,
        ]
