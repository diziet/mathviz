"""N-body gravitational simulation generator.

Integrates a gravitational N-body system using scipy and outputs trajectories
as curves. Initial conditions are derived from a seed for reproducibility.
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

_DEFAULT_NUM_BODIES = 3
_DEFAULT_TIME_SPAN = 10.0
_DEFAULT_INTEGRATION_STEPS = 5000
_MIN_INTEGRATION_STEPS = 100
_MIN_NUM_BODIES = 2
_MAX_NUM_BODIES = 20
_GRAVITATIONAL_CONSTANT = 1.0
_SOFTENING = 1e-4


def _validate_params(
    num_bodies: int,
    time_span: float,
    integration_steps: int,
) -> None:
    """Validate N-body parameters."""
    if num_bodies < _MIN_NUM_BODIES:
        raise ValueError(
            f"num_bodies must be >= {_MIN_NUM_BODIES}, got {num_bodies}"
        )
    if num_bodies > _MAX_NUM_BODIES:
        raise ValueError(
            f"num_bodies must be <= {_MAX_NUM_BODIES}, got {num_bodies}"
        )
    if time_span <= 0:
        raise ValueError(f"time_span must be positive, got {time_span}")
    if integration_steps < _MIN_INTEGRATION_STEPS:
        raise ValueError(
            f"integration_steps must be >= {_MIN_INTEGRATION_STEPS}, "
            f"got {integration_steps}"
        )


def _generate_initial_conditions(
    num_bodies: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate reproducible initial positions, velocities, and masses."""
    positions = rng.uniform(-1.0, 1.0, size=(num_bodies, 3))
    velocities = rng.uniform(-0.5, 0.5, size=(num_bodies, 3))
    masses = rng.uniform(0.5, 2.0, size=num_bodies)
    return positions, velocities, masses


def _nbody_rhs(
    _t: float, state: np.ndarray, num_bodies: int, masses: np.ndarray
) -> np.ndarray:
    """Compute derivatives for the N-body gravitational system."""
    positions = state[: num_bodies * 3].reshape(num_bodies, 3)
    velocities = state[num_bodies * 3 :].reshape(num_bodies, 3)

    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            dist_sq = np.dot(diff, diff) + _SOFTENING**2
            dist = np.sqrt(dist_sq)
            accelerations[i] += (
                _GRAVITATIONAL_CONSTANT * masses[j] * diff / (dist_sq * dist)
            )

    derivs = np.empty_like(state)
    derivs[: num_bodies * 3] = velocities.ravel()
    derivs[num_bodies * 3 :] = accelerations.ravel()
    return derivs


def _integrate_nbody(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    time_span: float,
    integration_steps: int,
) -> np.ndarray:
    """Integrate the N-body system and return trajectory array."""
    num_bodies = len(masses)
    initial_state = np.concatenate([positions.ravel(), velocities.ravel()])
    t_eval = np.linspace(0.0, time_span, integration_steps)

    result = solve_ivp(
        fun=lambda t, s: _nbody_rhs(t, s, num_bodies, masses),
        t_span=(0.0, time_span),
        y0=initial_state,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    if not result.success:
        raise RuntimeError(f"N-body integration failed: {result.message}")

    # Shape: (integration_steps, num_bodies, 3)
    trajectory = result.y[: num_bodies * 3].T.reshape(
        integration_steps, num_bodies, 3
    )
    return trajectory.astype(np.float64)


@register
class NBodyGenerator(GeneratorBase):
    """N-body gravitational simulation generator."""

    name = "nbody"
    category = "physics"
    aliases = ("n_body",)
    description = "Gravitational N-body simulation with seed-based initial conditions"
    resolution_params = {
        "integration_steps": "Number of integration time steps",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "num_bodies": _DEFAULT_NUM_BODIES,
            "time_span": _DEFAULT_TIME_SPAN,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate N-body trajectories as curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "integration_steps" in merged:
            logger.warning(
                "integration_steps should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("integration_steps")

        num_bodies = int(merged["num_bodies"])
        time_span = float(merged["time_span"])
        integration_steps = int(
            resolution_kwargs.get("integration_steps", _DEFAULT_INTEGRATION_STEPS)
        )

        _validate_params(num_bodies, time_span, integration_steps)
        merged["integration_steps"] = integration_steps

        rng = default_rng(seed)
        positions, velocities, masses = _generate_initial_conditions(
            num_bodies, rng
        )

        trajectory = _integrate_nbody(
            positions, velocities, masses, time_span, integration_steps
        )

        # One curve per body
        curves = [
            Curve(points=trajectory[:, i, :], closed=False)
            for i in range(num_bodies)
        ]

        all_points = trajectory.reshape(-1, 3)
        bbox = BoundingBox.from_points(all_points)

        logger.info(
            "Generated nbody: bodies=%d, steps=%d, t=%.2f",
            num_bodies, integration_steps, time_span,
        )

        return MathObject(
            curves=curves,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for N-body trajectories."""
        return RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
