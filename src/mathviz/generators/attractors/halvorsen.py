"""Halvorsen attractor dynamical system generator.

Integrates the Halvorsen system of ODEs to produce a 3D trajectory curve.
The Halvorsen system is defined by:
    dx/dt = -a*x - 4*y - 4*z - y^2
    dy/dt = -a*y - 4*z - 4*x - z^2
    dz/dt = -a*z - 4*x - 4*y - x^2

With default parameter a=1.89, the attractor produces a three-winged
chaotic trajectory with approximate three-fold symmetry.
"""

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


@register
class HalvorsenGenerator(AttractorGeneratorBase):
    """Halvorsen attractor dynamical system generator."""

    name = "halvorsen"
    aliases = ("halvorsen_attractor",)
    description = "Halvorsen three-winged strange attractor trajectory"

    _t_span_end = 100.0
    _default_initial_condition = (-1.48, -1.51, 2.04)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Halvorsen generator."""
        return {
            "a": 1.89,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Halvorsen ODE parameters."""
        a = float(params["a"])
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Halvorsen system."""
        x, y, z = state
        a = float(params["a"])
        return [
            -a * x - 4.0 * y - 4.0 * z - y * y,
            -a * y - 4.0 * z - 4.0 * x - z * z,
            -a * z - 4.0 * x - 4.0 * y - x * x,
        ]
