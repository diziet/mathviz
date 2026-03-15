"""Sprott minimal chaotic systems generator.

Implements several of J.C. Sprott's minimal chaotic flows, selectable via
a ``system`` parameter. Each is a 3D ODE with very few terms, producing
visually distinct strange attractors.

Supported systems:
    sprott_a:  dx/dt = y,        dy/dt = -x + y*z,  dz/dt = 1 - y^2
    sprott_b:  dx/dt = y*z,      dy/dt = x - y,     dz/dt = 1 - x*y
    sprott_g:  dx/dt = 0.4*x + z, dy/dt = x*z - y,  dz/dt = -x + y
    sprott_n:  dx/dt = -2*y,     dy/dt = x + z^2,   dz/dt = 1 + y - 2*z
    sprott_s:  dx/dt = -x - 4*y, dy/dt = x + z^2,   dz/dt = 1 + x
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from mathviz.core.generator import register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.attractors._base import (
    DEFAULT_TRANSIENT_STEPS,
    AttractorGeneratorBase,
)


def _rhs_sprott_a(state: np.ndarray) -> list[float]:
    """Sprott Case A: simplest quadratic flow with chaos."""
    x, y, z = state
    return [y, -x + y * z, 1.0 - y * y]


def _rhs_sprott_b(state: np.ndarray) -> list[float]:
    """Sprott Case B: minimal chaotic jerk system."""
    x, y, z = state
    return [y * z, x - y, 1.0 - x * y]


def _rhs_sprott_g(state: np.ndarray) -> list[float]:
    """Sprott Case G: chaotic flow with one quadratic nonlinearity."""
    x, y, z = state
    return [0.4 * x + z, x * z - y, -x + y]


def _rhs_sprott_n(state: np.ndarray) -> list[float]:
    """Sprott Case N: chaotic flow with quadratic z term."""
    x, y, z = state
    return [-2.0 * y, x + z * z, 1.0 + y - 2.0 * z]


def _rhs_sprott_s(state: np.ndarray) -> list[float]:
    """Sprott Case S: chaotic flow with quadratic z term."""
    x, y, z = state
    return [-x - 4.0 * y, x + z * z, 1.0 + x]


_SPROTT_SYSTEMS: dict[str, tuple[Callable, tuple[float, float, float]]] = {
    "sprott_a": (_rhs_sprott_a, (0.0, 5.0, 0.0)),
    "sprott_b": (_rhs_sprott_b, (0.1, 0.1, 0.1)),
    "sprott_g": (_rhs_sprott_g, (0.1, 0.1, 0.1)),
    "sprott_n": (_rhs_sprott_n, (0.1, 0.1, 0.1)),
    "sprott_s": (_rhs_sprott_s, (0.1, 0.1, 0.1)),
}

SPROTT_VARIANTS = frozenset(_SPROTT_SYSTEMS.keys())


@register
class SprottGenerator(AttractorGeneratorBase):
    """Sprott minimal chaotic systems generator."""

    name = "sprott"
    aliases = ("sprott_attractor",)
    description = "Sprott minimal chaotic flows (multiple variants)"

    _t_span_end = 200.0
    _perturbation_scale = 1e-3

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Sprott generator."""
        return {
            "system": "sprott_a",
            "transient_steps": DEFAULT_TRANSIENT_STEPS,
        }

    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate that the selected Sprott system is recognized."""
        system = str(params.get("system", "sprott_a"))
        if system not in SPROTT_VARIANTS:
            raise ValueError(
                f"Unknown Sprott system {system!r}, "
                f"must be one of {sorted(SPROTT_VARIANTS)}"
            )

    def _get_initial_condition(self, params: dict[str, Any]) -> np.ndarray:
        """Return system-specific initial condition."""
        system = str(params.get("system", "sprott_a"))
        _, ic = _SPROTT_SYSTEMS[system]
        return np.array(ic)

    def _rhs(
        self, _t: float, state: np.ndarray, params: dict[str, Any],
    ) -> list[float]:
        """Dispatch to the selected Sprott system RHS."""
        system = str(params["system"])
        rhs_fn, _ = _SPROTT_SYSTEMS[system]
        return rhs_fn(state)

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as the default representation for Sprott systems."""
        return RepresentationConfig(type=RepresentationType.TUBE)
