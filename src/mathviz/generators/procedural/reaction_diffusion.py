"""Reaction-diffusion (Gray-Scott) generator.

Runs the Gray-Scott reaction-diffusion model on a 2D grid and outputs the
concentration field as a scalar field for HEIGHTMAP_RELIEF representation.
Initial perturbation is seed-controlled for reproducibility.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.procedural._utils import compute_heightmap_bounding_box

logger = logging.getLogger(__name__)

_DEFAULT_FEED_RATE = 0.035
_DEFAULT_KILL_RATE = 0.065
_DEFAULT_DIFFUSION_U = 0.16
_DEFAULT_DIFFUSION_V = 0.08
_DEFAULT_TIMESTEPS = 5000
_DEFAULT_DT = 1.0
_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_GRID_SIZE = 128
_MIN_GRID_SIZE = 16
_MIN_TIMESTEPS = 100
_MAX_TIMESTEPS = 50000


def _validate_params(
    feed_rate: float,
    kill_rate: float,
    diffusion_u: float,
    diffusion_v: float,
    timesteps: int,
    grid_size: int,
    dt: float,
    height_scale: float,
) -> None:
    """Validate reaction-diffusion parameters."""
    if feed_rate <= 0:
        raise ValueError(f"feed_rate must be positive, got {feed_rate}")
    if kill_rate <= 0:
        raise ValueError(f"kill_rate must be positive, got {kill_rate}")
    if diffusion_u <= 0:
        raise ValueError(f"diffusion_u must be positive, got {diffusion_u}")
    if diffusion_v <= 0:
        raise ValueError(f"diffusion_v must be positive, got {diffusion_v}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
    if timesteps < _MIN_TIMESTEPS:
        raise ValueError(
            f"timesteps must be >= {_MIN_TIMESTEPS}, got {timesteps}"
        )
    if timesteps > _MAX_TIMESTEPS:
        raise ValueError(
            f"timesteps must be <= {_MAX_TIMESTEPS}, got {timesteps}"
        )
    if grid_size < _MIN_GRID_SIZE:
        raise ValueError(
            f"grid_size must be >= {_MIN_GRID_SIZE}, got {grid_size}"
        )


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Compute discrete Laplacian with periodic boundary conditions."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _init_concentrations(
    grid_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize U and V concentration fields with seed-controlled perturbation."""
    rng = default_rng(seed)

    u = np.ones((grid_size, grid_size), dtype=np.float64)
    v = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Add seed-controlled square perturbations
    num_seeds = rng.integers(3, 8)
    patch_size = max(4, grid_size // 8)

    for _ in range(num_seeds):
        cx = rng.integers(patch_size, grid_size - patch_size)
        cy = rng.integers(patch_size, grid_size - patch_size)
        half = patch_size // 2
        u[cx - half : cx + half, cy - half : cy + half] = 0.50
        v[cx - half : cx + half, cy - half : cy + half] = 0.25
        # Add small noise to break symmetry
        noise = rng.uniform(-0.01, 0.01, (2 * half, 2 * half))
        u[cx - half : cx + half, cy - half : cy + half] += noise
        v[cx - half : cx + half, cy - half : cy + half] += noise * 0.5

    return u, v


def _run_gray_scott(
    u: np.ndarray,
    v: np.ndarray,
    feed_rate: float,
    kill_rate: float,
    diffusion_u: float,
    diffusion_v: float,
    timesteps: int,
    dt: float,
) -> np.ndarray:
    """Run Gray-Scott simulation and return the V concentration field."""
    for _ in range(timesteps):
        lap_u = _laplacian(u)
        lap_v = _laplacian(v)
        uvv = u * v * v

        u += dt * (diffusion_u * lap_u - uvv + feed_rate * (1.0 - u))
        v += dt * (diffusion_v * lap_v + uvv - (feed_rate + kill_rate) * v)

        np.clip(u, 0.0, 1.0, out=u)
        np.clip(v, 0.0, 1.0, out=v)

    return v


@register
class ReactionDiffusionGenerator(GeneratorBase):
    """Gray-Scott reaction-diffusion pattern generator.

    Runs the Gray-Scott model on a 2D grid with seed-controlled initial
    conditions. Outputs the V concentration field as a scalar field for
    HEIGHTMAP_RELIEF representation.
    """

    name = "reaction_diffusion"
    category = "procedural"
    aliases = ("gray_scott",)
    description = "Gray-Scott reaction-diffusion pattern as heightmap"
    resolution_params = {
        "grid_size": "Grid points per axis for simulation",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for reaction-diffusion."""
        return {
            "feed_rate": _DEFAULT_FEED_RATE,
            "kill_rate": _DEFAULT_KILL_RATE,
            "diffusion_u": _DEFAULT_DIFFUSION_U,
            "diffusion_v": _DEFAULT_DIFFUSION_V,
            "timesteps": _DEFAULT_TIMESTEPS,
            "dt": _DEFAULT_DT,
            "height_scale": _DEFAULT_HEIGHT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Gray-Scott reaction-diffusion scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        feed_rate = float(merged["feed_rate"])
        kill_rate = float(merged["kill_rate"])
        diffusion_u = float(merged["diffusion_u"])
        diffusion_v = float(merged["diffusion_v"])
        timesteps = int(merged["timesteps"])
        dt = float(merged["dt"])
        height_scale = float(merged["height_scale"])
        grid_size = int(
            resolution_kwargs.get("grid_size", _DEFAULT_GRID_SIZE)
        )

        _validate_params(
            feed_rate, kill_rate, diffusion_u, diffusion_v,
            timesteps, grid_size, dt, height_scale,
        )

        merged["grid_size"] = grid_size

        u, v = _init_concentrations(grid_size, seed)
        field = _run_gray_scott(
            u, v, feed_rate, kill_rate,
            diffusion_u, diffusion_v, timesteps, dt,
        )
        field = field * height_scale
        bbox = compute_heightmap_bounding_box(field)

        logger.info(
            "Generated reaction_diffusion: feed=%.4f, kill=%.4f, "
            "steps=%d, grid=%d, field_range=[%.4f, %.4f]",
            feed_rate, kill_rate, timesteps, grid_size,
            bbox.min_corner[2], bbox.max_corner[2],
        )

        return MathObject(
            scalar_field=field,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return HEIGHTMAP_RELIEF as default representation."""
        return RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
