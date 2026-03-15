"""Shared base class and utilities for attractor generators.

Provides AttractorGeneratorBase with common ODE integration scaffolding,
parameter validation, and bounding box computation. Concrete attractors
define only their RHS function, default parameters, and initial condition.
"""

import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp

from mathviz.core.generator import GeneratorBase
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

DEFAULT_TRANSIENT_STEPS = 1000
DEFAULT_INTEGRATION_STEPS = 100_000
MIN_INTEGRATION_STEPS = 100
MIN_TRAJECTORY_POINTS = 10
PERTURBATION_SCALE = 1e-3


def validate_integration_params(
    integration_steps: int,
    transient_steps: int,
) -> None:
    """Validate integration and transient step counts."""
    if integration_steps < MIN_INTEGRATION_STEPS:
        raise ValueError(
            f"integration_steps must be >= {MIN_INTEGRATION_STEPS}, "
            f"got {integration_steps}"
        )
    if transient_steps < 0:
        raise ValueError(
            f"transient_steps must be >= 0, got {transient_steps}"
        )
    trajectory_points = integration_steps - transient_steps
    if trajectory_points < MIN_TRAJECTORY_POINTS:
        raise ValueError(
            f"integration_steps - transient_steps must be >= "
            f"{MIN_TRAJECTORY_POINTS}, got {trajectory_points}"
        )


def compute_bounding_box(points: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from trajectory points."""
    min_corner = tuple(float(v) for v in points.min(axis=0))
    max_corner = tuple(float(v) for v in points.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def integrate_ode(
    rhs_fn: Callable[[float, np.ndarray], list[float]],
    t_span_end: float,
    initial_condition: np.ndarray,
    integration_steps: int,
    transient_steps: int,
    system_name: str,
    output_dims: int = 3,
) -> np.ndarray:
    """Integrate an ODE system and return trajectory after transient.

    Args:
        rhs_fn: Right-hand side function f(t, state) -> derivatives.
        t_span_end: End time for integration.
        initial_condition: Initial state vector.
        integration_steps: Total number of time steps.
        transient_steps: Steps to discard from start.
        system_name: Name for error messages.
        output_dims: Number of dimensions to keep (for projection).
    """
    t_eval = np.linspace(0.0, t_span_end, integration_steps)

    result = solve_ivp(
        fun=rhs_fn,
        t_span=(0.0, t_span_end),
        y0=initial_condition,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"{system_name} integration failed: {result.message}")

    trajectory = result.y.T[transient_steps:]
    if output_dims < trajectory.shape[1]:
        trajectory = trajectory[:, :output_dims]
    return trajectory.astype(np.float64)


class AttractorGeneratorBase(GeneratorBase):
    """Base class for ODE-based attractor generators."""

    category = "attractors"
    resolution_params = {
        "integration_steps": "Total number of integration time steps",
    }
    _resolution_defaults = {"integration_steps": DEFAULT_INTEGRATION_STEPS}

    # Subclasses must set these
    _t_span_end: float = 100.0
    _default_initial_condition: tuple[float, ...] = (0.0, 0.0, 0.0)
    _perturbation_scale: float = PERTURBATION_SCALE
    _output_dims: int = 3

    @abstractmethod
    def _rhs(self, _t: float, state: np.ndarray, params: dict[str, Any]) -> list[float]:
        """Compute the right-hand side of the ODE system."""
        ...

    @abstractmethod
    def _validate_ode_params(self, params: dict[str, Any]) -> None:
        """Validate ODE-specific parameters (called before integration)."""
        ...

    def _get_initial_condition(self, params: dict[str, Any]) -> np.ndarray:
        """Return the initial condition vector for integration.

        Override in subclasses that need parameter-dependent initial
        conditions (e.g. Sprott systems with per-variant ICs).
        """
        return np.array(self._default_initial_condition)

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an attractor trajectory curve."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        transient_steps = int(merged.pop("transient_steps"))
        integration_steps = int(
            resolution_kwargs.get("integration_steps", DEFAULT_INTEGRATION_STEPS)
        )

        self._validate_ode_params(merged)
        validate_integration_params(integration_steps, transient_steps)

        merged["transient_steps"] = transient_steps
        merged["integration_steps"] = integration_steps

        rng = default_rng(seed)
        ic = self._get_initial_condition(merged)
        perturbation = rng.normal(scale=self._perturbation_scale, size=len(ic))
        initial_condition = ic + perturbation

        trajectory = integrate_ode(
            rhs_fn=lambda t, s: self._rhs(t, s, merged),
            t_span_end=self._t_span_end,
            initial_condition=initial_condition,
            integration_steps=integration_steps,
            transient_steps=transient_steps,
            system_name=self.name,
            output_dims=self._output_dims,
        )

        curve = Curve(points=trajectory, closed=False)
        bbox = compute_bounding_box(trajectory)

        logger.info(
            "Generated %s attractor: points=%d (discarded %d transient)",
            self.name, len(trajectory), transient_steps,
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
        """Return the recommended representation for attractor generators."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
