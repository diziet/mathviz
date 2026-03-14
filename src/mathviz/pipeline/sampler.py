"""Sampler: mesh-to-point-cloud conversion with multiple algorithms."""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import trimesh
from pydantic import BaseModel, Field, model_validator

from mathviz.core.math_object import MathObject, Mesh, PointCloud

logger = logging.getLogger(__name__)

DEFAULT_SURFACE_DENSITY = 1.0  # points per mm^2
DEFAULT_VOLUME_DENSITY = 1.0  # points per mm^3


class SamplingMethod(str, Enum):
    """Available sampling algorithms."""

    UNIFORM_SURFACE = "uniform_surface"
    RANDOM_SURFACE = "random_surface"
    VOLUME_FILL = "volume_fill"


class SamplerConfig(BaseModel):
    """Configuration for the sampler stage.

    Provide either density or num_points, not both. If neither is given,
    DEFAULT_SURFACE_DENSITY or DEFAULT_VOLUME_DENSITY is used.
    """

    method: SamplingMethod = SamplingMethod.UNIFORM_SURFACE
    density: Optional[float] = Field(
        default=None, gt=0, description="Points/mm^2 (surface) or points/mm^3 (volume)"
    )
    num_points: Optional[int] = Field(default=None, gt=0, description="Explicit point count")
    seed: int = Field(default=42, description="RNG seed for reproducibility")
    resample: bool = Field(
        default=False, description="Force resampling even if point_cloud exists"
    )

    @model_validator(mode="after")
    def _check_density_num_points_exclusive(self) -> "SamplerConfig":
        """Ensure density and num_points are not both set."""
        if self.density is not None and self.num_points is not None:
            raise ValueError("density and num_points are mutually exclusive")
        return self


def sample(obj: MathObject, config: SamplerConfig) -> MathObject:
    """Convert mesh to point cloud on a MathObject.

    Mutates obj.point_cloud in place and returns obj.
    Skips sampling if point_cloud already exists and resample is False.
    """
    if obj.point_cloud is not None and not config.resample:
        logger.info("Skipping sampler: point_cloud already exists (resample=False)")
        return obj

    if obj.mesh is None:
        raise ValueError("Sampler requires a mesh on the MathObject")

    tm = _to_trimesh(obj.mesh)
    rng = np.random.default_rng(config.seed)

    if config.method == SamplingMethod.UNIFORM_SURFACE:
        points = _sample_uniform_surface(tm, config, rng)
    elif config.method == SamplingMethod.RANDOM_SURFACE:
        points = _sample_random_surface(tm, config, rng)
    elif config.method == SamplingMethod.VOLUME_FILL:
        points = _sample_volume_fill(tm, config, rng)
    else:
        raise ValueError(f"Unknown sampling method: {config.method}")

    obj.point_cloud = PointCloud(points=points.astype(np.float64))

    logger.info(
        "Sampled %d points using %s (method=%s)",
        len(points),
        "density" if config.density is not None else "num_points",
        config.method.value,
    )
    return obj


def _to_trimesh(mesh: Mesh) -> trimesh.Trimesh:
    """Convert a MathViz Mesh to a trimesh.Trimesh."""
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )


def _resolve_point_count(measure: float, config: SamplerConfig, default_density: float) -> int:
    """Resolve target point count from config and a geometric measure."""
    if config.num_points is not None:
        return config.num_points
    density = config.density if config.density is not None else default_density
    return max(1, int(round(density * measure)))


def _sample_uniform_surface(
    tm: trimesh.Trimesh,
    config: SamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximately even surface sampling via Poisson-disk rejection."""
    count = _resolve_point_count(tm.area, config, DEFAULT_SURFACE_DENSITY)
    points, _ = trimesh.sample.sample_surface_even(
        tm, count, seed=rng.integers(0, 2**31)
    )
    return np.asarray(points)


def _sample_random_surface(
    tm: trimesh.Trimesh,
    config: SamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random surface sampling (barycentric, face-area-weighted)."""
    count = _resolve_point_count(tm.area, config, DEFAULT_SURFACE_DENSITY)
    points, _ = trimesh.sample.sample_surface(
        tm, count, seed=rng.integers(0, 2**31)
    )
    return np.asarray(points)


def _sample_volume_fill(
    tm: trimesh.Trimesh,
    config: SamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Jittered grid volume fill inside a watertight mesh."""
    mesh_volume = abs(float(tm.volume))
    if mesh_volume < 1e-15:
        raise ValueError("Mesh has zero volume; is it watertight?")

    target_count = _resolve_point_count(mesh_volume, config, DEFAULT_VOLUME_DENSITY)
    bounds = tm.bounds  # (2, 3): [min, max]
    bbox_volume = float(np.prod(bounds[1] - bounds[0]))

    if bbox_volume < 1e-15:
        raise ValueError("Mesh bounding box has zero volume")

    fill_ratio = mesh_volume / bbox_volume
    adjusted_count = int(round(target_count / max(fill_ratio, 0.01)))
    spacing = _compute_grid_spacing(bbox_volume, adjusted_count)
    grid_points = _build_jittered_grid(bounds[0], bounds[1], spacing, rng)

    if len(grid_points) == 0:
        raise ValueError("Volume fill produced no candidate points")

    inside_mask = tm.contains(grid_points)
    interior_points = grid_points[inside_mask]

    if len(interior_points) == 0:
        raise ValueError("No points found inside mesh; is the mesh watertight?")

    logger.debug(
        "Volume fill: %d grid candidates, %d inside mesh (%.1f%%)",
        len(grid_points),
        len(interior_points),
        100.0 * len(interior_points) / len(grid_points),
    )
    return interior_points


def _compute_grid_spacing(volume: float, target_count: int) -> float:
    """Compute grid spacing to produce approximately target_count points."""
    return (volume / target_count) ** (1.0 / 3.0)


def _build_jittered_grid(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    spacing: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a jittered 3D grid within the bounding box."""
    axes = [
        np.arange(bbox_min[i], bbox_max[i], spacing)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    jitter = rng.uniform(-spacing * 0.4, spacing * 0.4, size=grid.shape)
    return grid + jitter
