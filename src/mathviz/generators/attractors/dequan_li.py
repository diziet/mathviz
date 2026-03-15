"""Dequan Li attractor dynamical system generator.

Integrates the Dequan Li system of ODEs to produce a 3D trajectory curve.
The system is defined by:
    dx/dt = a*(y - x) + d*x*z
    dy/dt = k*x + f*y - x*z
    dz/dt = c*z + x*y - e*x^2

With default parameters a=40, c=1.833, d=0.16, e=0.65, f=20, k=55,
the attractor produces a complex multi-scroll chaotic trajectory.
"""

from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


@register
class DequanLiGenerator(AttractorGeneratorBase):
    """Dequan Li chaotic attractor generator."""

    name = "dequan_li"
    aliases = ("dequan_li_attractor",)
    description = "Dequan Li multi-scroll chaotic attractor trajectory"

    _t_span_end = 50.0
    _default_initial_condition = (0.349, 0.0, -0.16)
    _perturbation_scale = 1e-3

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Dequan Li system."""
        return {
            "a": 40.0,
            "c": 1.833,
            "d": 0.16,
            "e": 0.65,
            "f": 20.0,
            "k": 55.0,
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate Dequan Li ODE parameters."""
        a = float(params["a"])
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        k = float(params["k"])
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Compute the right-hand side of the Dequan Li system."""
        x, y, z = state
        a = float(params["a"])
        c = float(params["c"])
        d = float(params["d"])
        e = float(params["e"])
        f = float(params["f"])
        k = float(params["k"])
        return [
            a * (y - x) + d * x * z,
            k * x + f * y - x * z,
            c * z + x * y - e * x * x,
        ]

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as the default representation for Dequan Li."""
        return RepresentationConfig(type=RepresentationType.TUBE)
