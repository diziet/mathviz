"""Lorenz attractor dynamical system generator.

Integrates the Lorenz system of ODEs to produce a 3D trajectory curve.
The representation strategy converts the curve to a point cloud or tube mesh.
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

_DEFAULT_SIGMA = 10.0
_DEFAULT_RHO = 28.0
_DEFAULT_BETA = 8.0 / 3.0
_DEFAULT_TRANSIENT_STEPS = 1000
_DEFAULT_INTEGRATION_STEPS = 100_000
_DEFAULT_INITIAL_CONDITION = (1.0, 1.0, 1.0)
_MIN_INTEGRATION_STEPS = 100
_MIN_TRAJECTORY_POINTS = 10
_PERTURBATION_SCALE = 1e-3
_T_SPAN_END = 100.0


def _lorenz_rhs(
    _t: float,
    state: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
) -> list[float]:
    """Compute the right-hand side of the Lorenz system."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def _integrate_lorenz(
    sigma: float,
    rho: float,
    beta: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
) -> np.ndarray:
    """Integrate the Lorenz system and return trajectory points after transient."""
    total_steps = integration_steps
    t_eval = np.linspace(0.0, _T_SPAN_END, total_steps)

    result = solve_ivp(
        fun=lambda t, state: _lorenz_rhs(t, state, sigma, rho, beta),
        t_span=(0.0, _T_SPAN_END),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Lorenz integration failed: {result.message}")

    # result.y is (3, N), transpose to (N, 3) and discard transient
    trajectory = result.y.T[transient_steps:]
    return trajectory.astype(np.float64)


def _compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    sigma: float,
    rho: float,
    beta: float,
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate Lorenz parameters, raising ValueError for invalid inputs."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
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
class LorenzGenerator(GeneratorBase):
    """Lorenz attractor dynamical system generator."""

    name = "lorenz"
    category = "attractors"
    aliases = ("lorenz_attractor",)
    description = "Lorenz strange attractor trajectory"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }
    _resolution_defaults = {"integration_steps": _DEFAULT_INTEGRATION_STEPS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Lorenz generator."""
        return {
            "sigma": _DEFAULT_SIGMA,
            "rho": _DEFAULT_RHO,
            "beta": _DEFAULT_BETA,
            "transient_steps": _DEFAULT_TRANSIENT_STEPS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Lorenz attractor trajectory curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        # integration_steps is a resolution kwarg, not a regular param
        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        sigma = float(merged["sigma"])
        rho = float(merged["rho"])
        beta = float(merged["beta"])
        transient_steps = int(merged["transient_steps"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(sigma, rho, beta, integration_steps, transient_steps)

        # Record integration_steps so output is self-describing
        merged["integration_steps"] = integration_steps

        # Perturb initial condition with seed for variation
        rng = default_rng(seed)
        perturbation = rng.normal(scale=_PERTURBATION_SCALE, size=3)
        initial_condition = np.array(_DEFAULT_INITIAL_CONDITION) + perturbation

        trajectory = _integrate_lorenz(
            sigma, rho, beta, initial_condition,
            integration_steps, transient_steps,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = _compute_bounding_box(trajectory)

        logger.info(
            "Generated Lorenz attractor: σ=%.1f, ρ=%.1f, β=%.4f, "
            "points=%d (discarded %d transient)",
            sigma, rho, beta,
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
        """Return the recommended representation for the Lorenz attractor."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05,
        )
