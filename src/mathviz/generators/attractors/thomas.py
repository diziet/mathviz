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

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


@register
class ThomasGenerator(AttractorGeneratorBase):
    """Thomas cyclically symmetric attractor generator."""

    name = "thomas"
    aliases = ("thomas_attractor",)
    description = "Thomas cyclically symmetric strange attractor trajectory"

    _t_span_end = 500.0
    _default_initial_condition = (1.1, 1.1, -0.01)

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Thomas generator."""
        return {
            "b": 0.208186,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Thomas ODE parameters."""
        b = float(params["b"])
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Thomas system."""
        x, y, z = state
        b = float(params["b"])
        return [
            np.sin(y) - b * x,
            np.sin(z) - b * y,
            np.sin(x) - b * z,
        ]
